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
from transformers import Trainer, Seq2SeqTrainer

from .base import calculate_superloss, UnlearningTrainer

from transformers.utils import logging

logger = logging.get_logger(__name__)


class DPOTrainer(UnlearningTrainer):
        
    def method_specific_setup(self):
        self.oracle_model = copy.deepcopy(self.model)
        self.oracle_model.eval()

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        if model.training:
            # Unlearned model
            input_df_mask = inputs['is_df'] == 1
            inputs.pop('is_df')
            dr_inputs = {k: v[~input_df_mask] for k, v in inputs.items()}
            # dr_inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
            if 'labels' in dr_inputs:
                dr_inputs.pop('labels')
            if 'label_ids' in dr_inputs:
                dr_inputs.pop('label_ids')
            
            dr_outputs = model(**dr_inputs, return_dict=True)
            with torch.no_grad():
                good_outputs = self.good_teacher(**dr_inputs, return_dict=True)

            df_inputs = {k: v[input_df_mask] for k, v in inputs.items()}
            # df_inputs = {k[len('df_'):]: v for k, v in inputs.items() if k.startswith('df_')}
            if 'labels' in df_inputs:
                df_inputs.pop('labels')
            if 'label_ids' in df_inputs:
                df_inputs.pop('label_ids')

            df_outputs = model(**df_inputs, return_dict=True)
            with torch.no_grad():
                bad_outputs = self.bad_teacher(**df_inputs, return_dict=True)

            kl_loss = nn.KLDivLoss(reduction='none' if self.use_cl else "batchmean")

            dr_loss = kl_loss(dr_outputs.logits, good_outputs.logits)
            df_loss = kl_loss(df_outputs.logits, bad_outputs.logits)

            dr_loss = calculate_superloss(dr_loss, dr_inputs).mean()
            df_loss = calculate_superloss(df_loss, df_inputs).mean()

            loss = dr_loss + df_loss
            outputs = df_outputs

        else:
            if 'is_df' in inputs:
                inputs.pop('is_df')
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        """wrapper for compute_loss training with non-SuperLoss."""

        if model.training:
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, labels)


            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle

            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

            outputs = forget_outputs

        else:
            if 'is_df' in inputs:
                inputs.pop('is_df')
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
