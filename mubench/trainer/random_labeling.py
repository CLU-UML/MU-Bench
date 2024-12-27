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


class RandomLabelingTrainer(UnlearningTrainer):
    def method_specific_setup(self):
        pass

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
