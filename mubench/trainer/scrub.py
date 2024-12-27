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

from .base import calculate_superloss, UnlearningTrainer

from transformers.utils import logging

logger = logging.get_logger(__name__)


class SCRUBTrainer(UnlearningTrainer):
    def method_specific_setup(self):
        self.ori_model = copy.deepcopy(self.model)
        for n, p in self.ori_model.named_parameters():
            p.requires_grad = False
        self.ori_model.eval()
        self.do_max_step = True

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        if model.training:
            if self.do_max_step:
                # Do max epoch on Df. Use Df as the training set. We only need prediction, not label
                if 'labels' in inputs:
                    inputs.pop('labels')
                if 'label_ids' in inputs:
                    inputs.pop('label_ids')
                outputs = model(**inputs, return_dict=True)
                with torch.no_grad():
                    ori_outputs = self.ori_model(**inputs, return_dict=True)

                kl_loss = nn.KLDivLoss(reduction='none')
                loss = kl_loss(outputs.logits, ori_outputs.logits)
                loss = calculate_superloss(loss, inputs).mean()
                loss = -1 * loss

            else:
                # We need label for task loss
                outputs = model(**inputs, return_dict=True)
                with torch.no_grad():
                    ori_outputs = self.ori_model(**inputs, return_dict=True)

                kl_loss = nn.KLDivLoss(reduction='none')
                loss = outputs.loss + kl_loss(outputs.logits, ori_outputs.logits).mean(axis=1)
                loss = calculate_superloss(loss, inputs).mean()

        else:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss.mean()

        return (loss, outputs) if return_outputs else loss

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        if model.training:
            if self.do_max_step:
                # Do max epoch on Df. Use Df as the training set. We only need prediction, not label
                if 'labels' in inputs:
                    inputs.pop('labels')
                if 'label_ids' in inputs:
                    inputs.pop('label_ids')
                outputs = model(**inputs, return_dict=True)
                with torch.no_grad():
                    ori_outputs = self.ori_model(**inputs, return_dict=True)

                kl_loss = nn.KLDivLoss(reduction="batchmean")
                loss = kl_loss(outputs.logits, ori_outputs.logits)
                loss = -1 * loss

            else:
                # Do min epoch on Dr. We need label for task loss
                outputs = model(**inputs, return_dict=True)
                with torch.no_grad():
                    ori_outputs = self.ori_model(**inputs, return_dict=True)

                kl_loss = nn.KLDivLoss(reduction="batchmean")
                loss = outputs.loss + kl_loss(outputs.logits, ori_outputs.logits)

        else:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def unlearn(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs):

        self.unlearn_start_time = time.time()
        self.args.num_train_epochs = self.args.num_train_epochs / 2

        self.train_dataset = self.raw_datasets['df_train']
        logger.info(f'******** Doing max step for {self.args.num_train_epochs} epochs ********')
        print(f'******** Doing max step for {self.args.num_train_epochs} epochs ********')
        super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        self.do_max_step = False
        self.train_dataset = self.raw_datasets['train']
        logger.info(f'******** Doing min step for {self.args.num_train_epochs} epochs ********')
        print(f'******** Doing min step for {self.args.num_train_epochs} epochs ********')
        out = super().train(None, trial, ignore_keys_for_eval, **kwargs)

        self.unlearn_end_time = time.time()
        self.unlearn_time = self.unlearn_end_time - self.unlearn_start_time

        return out
