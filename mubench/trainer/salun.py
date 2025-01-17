import os
import copy
import time
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from typing import *

from .base import UnlearningTrainer

from transformers.utils import logging

logger = logging.get_logger(__name__)


class SalUnTrainer(UnlearningTrainer):
    def method_specific_setup(self):
        self.salient_mask = None
        # self.salient_mask = self.compute_salient_mask()

    # def compute_loss_cl(self, model, inputs, return_outputs=False):
    #     # inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
    #     if return_outputs:
    #         loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
    #     else:
    #         loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

    #     loss = calculate_superloss(loss, inputs).mean()

    #     return (loss, outputs) if return_outputs else loss

    # def compute_loss_non_cl(self, model, inputs, return_outputs=False):
    #     # inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
    #     if return_outputs:
    #         loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
    #     else:
    #         loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

    #     return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        # Salient Mask
        for name, param in model.named_parameters():
            if name in ['hubert.masked_spec_embed', 'wav2vec2.masked_spec_embed']:
                continue
            if param.grad is not None:
                param.grad *= self.salient_mask[name].to(param.grad.device)

        return loss.detach() / self.args.gradient_accumulation_steps

    def unlearn(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs):

        self.unlearn_start_time = time.time()

        self.train_dataset = self.raw_datasets['df_train']
        self.salient_mask = self.compute_salient_mask()
        self.train_dataset = self.raw_datasets['train']
        out = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        self.unlearn_end_time = time.time()
        self.unlearn_time = self.unlearn_end_time - self.unlearn_start_time

        return out

    def compute_salient_mask(self):
        if os.path.exists(os.path.join(self.args.output_dir, 'sal_mask_with_0.5.pt')):
            salient_mask = torch.load(os.path.join(self.args.output_dir, 'sal_mask_with_0.5.pt'))
            
        else:
            model = self.model
            model.eval()
            # self._move_model_to_device(self.model, self.args.device)
            train_loader = self.get_train_dataloader()

            gradient = {}
            length = 0
            for inputs in tqdm(train_loader, desc='Salient Map'):
                model.zero_grad()
                inputs = self._prepare_inputs(inputs)
                loss = -1 * super().compute_loss(model, inputs).mean()
                self.accelerator.backward(loss)

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if name in gradient:
                                gradient[name] += param.grad.data.to('cpu')
                            else:
                                gradient[name] = 0

            with torch.no_grad():
                for name in gradient:
                    gradient[name] = torch.abs_(gradient[name])

            threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            threshold_list = [0.5]

            for i in threshold_list:
                sorted_dict_positions = {}
                salient_mask = {}

                # Concatenate all tensors into a single tensor
                all_elements = - torch.cat([tensor.flatten() for tensor in gradient.values()])

                # Calculate the threshold index for the top 10% elements
                threshold_index = int(len(all_elements) * i)

                # Calculate positions of all elements
                positions = torch.argsort(all_elements)
                ranks = torch.argsort(positions)

                start_index = 0
                for key, tensor in gradient.items():
                    num_elements = tensor.numel()
                    # tensor_positions = positions[start_index: start_index + num_elements]
                    tensor_ranks = ranks[start_index : start_index + num_elements]

                    sorted_positions = tensor_ranks.reshape(tensor.shape)
                    sorted_dict_positions[key] = sorted_positions

                    # Set the corresponding elements to 1
                    threshold_tensor = torch.zeros_like(tensor_ranks)
                    threshold_tensor[tensor_ranks < threshold_index] = 1
                    threshold_tensor = threshold_tensor.reshape(tensor.shape)
                    salient_mask[key] = threshold_tensor
                    start_index += num_elements

                torch.save(salient_mask, os.path.join(self.args.output_dir, f"sal_mask_with_{i}.pt"))

        return salient_mask
