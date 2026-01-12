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



class DeleteTrainer(UnlearningTrainer):
    """
      - Uses KL(teacher || student) via PyTorch's F.kl_div(log_p_student, p_teacher)
      - Supports temperature scaling (standard distillation scaling T^2)
      - For CL version, applies calculate_superloss on per-sample KL values
    """

    def method_specific_setup(self):
        # Create the frozen teacher model once.
        # Assumes `self.model` exists (typical in Trainer-like setups).
        if not hasattr(self, "model") or self.model is None:
            logger.warning("DeleteTrainer.method_specific_setup: self.model is None.")
            self.original_model = None
            return

        self.original_model = copy.deepcopy(self.model)
        self.original_model.eval()

        for p in self.original_model.parameters():
            p.requires_grad_(False)

        # Optional temperature; fall back to 1.0 if not provided.
        self.dd_temperature = float(getattr(self.args, "dd_temperature", 1.0))
        # Optional: large negative for masking; -inf can be used but -1e9 is safer on some kernels.
        self.dd_mask_value = float(getattr(self.args, "dd_mask_value", -1e9))

        logger.info(
            f"DeleteTrainer setup: original_model created | "
            f"T={self.dd_temperature} | mask_value={self.dd_mask_value}"
        )

    def _get_logits_from_outputs(self, outputs):
        # HF models usually return dict with "logits" or a tuple whose first element is logits
        if isinstance(outputs, dict):
            return outputs.get("logits", None)
        # tuple/list outputs
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        return None

    def _compute_decoupled_soft_targets(self, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        teacher_logits: (B, K)
        labels: (B,) int class indices (or (B,1))
        returns soft_targets: (B, K) probs, with p(y)=0 and others renormalized
        """
        if labels.dim() > 1:
            labels = labels.view(-1)

        B, K = teacher_logits.shape
        # Build mask (B, K) with 0 everywhere, mask_value at true label
        mask = torch.zeros((B, K), device=teacher_logits.device, dtype=teacher_logits.dtype)
        mask.scatter_(1, labels.view(B, 1), self.dd_mask_value)

        masked_logits = teacher_logits + mask
        T = self.dd_temperature
        soft_targets = F.softmax(masked_logits / T, dim=-1)
        return soft_targets

    def _per_sample_kl_teacher_student(
        self,
        student_logits: torch.Tensor,
        soft_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns per-sample KL(teacher || student): sum_k p_t(k) * (log p_t(k) - log p_s(k))
        student_logits: (B, K)
        soft_targets:  (B, K)
        """
        T = self.dd_temperature
        log_p_student = F.log_softmax(student_logits / T, dim=-1)

        # Avoid log(0) warnings; soft_targets can have exact zeros on masked class.
        # Clamp for log only; keep probabilities effectively zero.
        eps = 1e-12
        log_p_teacher = torch.log(soft_targets.clamp_min(eps))

        per_sample_kl = (soft_targets * (log_p_teacher - log_p_student)).sum(dim=-1)
        # Standard distillation scaling
        per_sample_kl = per_sample_kl * (T * T)
        return per_sample_kl

    def compute_loss_cl(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss wrapper for decoupled distillation unlearning (CL variant).
        Applies calculate_superloss on per-sample KL and then averages.
        """
        if getattr(self, "original_model", None) is None:
            # Fallback: behave like standard forward loss if frozen model missing
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = self.calculate_superloss(loss, inputs).mean()
            return (loss, outputs) if return_outputs else loss

        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("DeleteTrainer requires `labels` in inputs for masking.")

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.original_model(**inputs)
            teacher_logits = teacher_outputs.logits
            if teacher_logits is None:
                raise ValueError("Could not extract logits from original_model outputs.")

            soft_targets = self._compute_decoupled_soft_targets(teacher_logits, labels)

        # Student forward
        outputs = model(**inputs)
        student_logits = outputs.logits
        if student_logits is None:
            raise ValueError("Could not extract logits from model outputs.")

        per_sample_kl = self._per_sample_kl_teacher_student(student_logits, soft_targets)

        # Apply superloss on per-sample values
        loss = self.calculate_superloss(per_sample_kl, inputs).mean()

        return (loss, outputs) if return_outputs else loss

    def compute_loss_non_cl(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss wrapper for decoupled distillation unlearning (non-CL variant).
        Uses mean per-sample KL directly.
        """
        if getattr(self, "original_model", None) is None:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("DeleteTrainer requires `labels` in inputs for masking.")

        with torch.no_grad():
            teacher_outputs = self.original_model(**inputs)
            teacher_logits = teacher_outputs.logits
            if teacher_logits is None:
                raise ValueError("Could not extract logits from original_model outputs.")

            soft_targets = self._compute_decoupled_soft_targets(teacher_logits, labels)

        outputs = model(**inputs)
        student_logits = outputs.logits
        if student_logits is None:
            raise ValueError("Could not extract logits from model outputs.")

        per_sample_kl = self._per_sample_kl_teacher_student(student_logits, soft_targets)
        loss = per_sample_kl.mean()

        return (loss, outputs) if return_outputs else loss
