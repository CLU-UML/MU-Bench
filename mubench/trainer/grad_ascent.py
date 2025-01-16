import os
import copy
import time
import math
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader
from typing import *

from .base import UnlearningTrainer

from transformers.utils import logging

logger = logging.get_logger(__name__)



class GradAscentTrainer(UnlearningTrainer):
    def method_specific_setup(self):
        pass

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = self.calculate_superloss(loss, inputs).mean()
        loss = -1 * loss

        return (loss, outputs) if return_outputs else loss

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = -1 * loss

        return (loss, outputs) if return_outputs else loss
