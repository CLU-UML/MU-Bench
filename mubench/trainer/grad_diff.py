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
from transformers import Trainer

from .base import UnlearningTrainer

from transformers.utils import logging

logger = logging.get_logger(__name__)


class GradDiffTrainer(UnlearningTrainer):
    def method_specific_setup(self):
        pass

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        if model.training:
            pass

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
            # Loss on Df
            df_inputs = {k[len('df_'):]: v for k, v in inputs.items() if k.startswith('df_')}
            df_outputs = model(**df_inputs, return_dict=True)

            # Loss on Dr
            dr_inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
            dr_outputs = model(**dr_inputs, return_dict=True)

            loss = -df_outputs.loss + dr_outputs.loss
            outputs = dr_outputs

        else:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
