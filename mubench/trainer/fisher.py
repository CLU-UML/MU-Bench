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

from mubench.evaluation import Evaluator, TextGenEvaluator
from .base import superloss
from .base import UnlearningTrainer

from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_norm(model):
    total = 0
    for n, p in model.named_parameters():
        total += torch.norm(p)

    return total


class FisherTrainer(Trainer):
    def method_specific_setup(self):
        pass

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        # inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        loss = calculate_superloss(loss, inputs).mean()

        return (loss, outputs) if return_outputs else loss

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        # inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        return (loss, outputs) if return_outputs else loss

    def unlearn(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs):

        self.unlearn_start_time = time.time()

        fisher_info_matrix = self.compute_fisher()
        self.scrub(fisher_info_matrix)
        out = None

        self.unlearn_end_time = time.time()
        self.unlearn_time = self.unlearn_end_time - self.unlearn_start_time

        return out

    def compute_fisher(self):
        if os.path.exists(os.path.join(self.args.output_dir, 'fim.pt')):
            fisher_info_matrix = torch.load(os.path.join(self.args.output_dir, 'fim.pt'))
            
        else:
            model = self.model
            self._move_model_to_device(self.model, self.args.device)
            train_loader = self.get_train_dataloader()
            fisher_info_matrix = {}

            length = 0
            for inputs in tqdm(train_loader, desc='Fisher'):
                inputs = self._prepare_inputs(inputs)
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                model.zero_grad()
                loss.backward(retain_graph=True)

                logits = outputs.logits.detach()
                prob = F.softmax(logits, 1)
                log_prob = F.log_softmax(logits, 1)
                # gradient = torch.autograd.grad(y, x, retain_graph=True, grad_outputs=torch.ones_like(y))[0]

                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        continue

                    # Some parameters does not have gradients due to with torch.no_grad()
                    if p.grad is None:
                        continue

                    if n not in fisher_info_matrix:
                        fisher_info_matrix[n] = torch.zeros_like(p)
                    else:
                        # fisher_of_p = []
                        for _prob in prob:
                            for y in range(prob.shape[1]):
                                fisher_info_matrix[n] += _prob[y] * p.grad.detach() * p.grad.detach()
                        # fisher_info_matrix[n] += torch.stack(fisher_of_p).sum(0)

                length += logits.shape[0]

            for n, p in fisher_info_matrix.items():
                fisher_info_matrix[n] /= length

            torch.save(fisher_info_matrix, os.path.join(self.args.output_dir, 'fim.pt'))

        return fisher_info_matrix

    def scrub(self, fisher_info_matrix):
        print(f'Model parameter norm before scrubbing:', get_norm(self.model))
        def get_mean_var(n, p, is_base_dist=False, alpha=3e-6):
            '''Source: https://github.com/AdityaGolatkar/SelectiveForgetting/blob/master/Forgetting.ipynb'''
            var = copy.deepcopy(1. / (fisher_info_matrix[n] + 1e-8))
            var = var.clamp(max=1e3)
            if p.size(0) == self.num_labels:
                var = var.clamp(max=1e2)
            var = alpha * var
            
            if p.ndim > 1:
                var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
            if not is_base_dist:
                mu = copy.deepcopy(p.data.clone())
            else:
                mu = copy.deepcopy(p.data.clone())
            if p.size(0) == self.num_labels:
                # Last layer
                var *= 10
            elif p.ndim == 1:
                # BatchNorm
                var *= 10
            return mu, var

        for n, p in self.model.named_parameters():
            if n not in fisher_info_matrix:
                print(f'Parameter {n} not found in fisher information matrix')
                continue
            if p.requires_grad:
                mu, var = get_mean_var(n, p, False, alpha=1e-8)
                try:
                    assert (mu == p.data).all()
                except:
                    breakpoint()
                p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()
        print(f'Model parameter norm after scrubbing:', get_norm(self.model))
