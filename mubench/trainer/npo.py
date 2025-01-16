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

from .base import UnlearningTrainer

from transformers.utils import logging

logger = logging.get_logger(__name__)


class NPOTrainer(UnlearningTrainer):
    def method_specific_setup(self):
        self.beta = 0.1
        self.orig_model = copy.deepcopy(self.model)
        self.orig_model.eval()

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        current_forget_loss = outputs.loss

        with torch.no_grad():
            ref_forget_loss = self.orig_model(**inputs).loss
        
        neg_log_ratios = current_forget_loss - ref_forget_loss
        loss = -2 / self.beta * F.logsigmoid(self.beta * neg_log_ratios).mean()

        return (loss, outputs) if return_outputs else loss
