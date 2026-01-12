import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from .base import UnlearningTrainer

from transformers.utils import logging

logger = logging.get_logger(__name__)


def l1_regularization(model: nn.Module) -> torch.Tensor:
    params_vec = []
    for p in model.parameters():
        if p is not None:
            params_vec.append(p.view(-1))
    if len(params_vec) == 0:
        # should basically never happen, but keep it safe
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


class L1SparseTrainer(UnlearningTrainer):
    def method_specific_setup(self):
        pass

    def _get_epoch_int(self) -> int:
        # HF Trainer has self.state.epoch as float; prefer that if available
        ep = getattr(getattr(self, "state", None), "epoch", None)
        if ep is None:
            return 0
        try:
            return int(ep)
        except Exception:
            return 0

    def _current_alpha(self) -> float:
        alpha = float(getattr(self.args, "alpha", 0.0))
        unlearn_epochs = int(getattr(self.args, "unlearn_epochs", 0))
        no_l1_epochs = int(getattr(self.args, "no_l1_epochs", 0))

        denom = unlearn_epochs - no_l1_epochs
        epoch = self._get_epoch_int()

        if denom > 0:
            if epoch < denom:
                return alpha * (1.0 - epoch / float(denom))
            return 0.0
        elif denom == 0:
            # mirrors the original code branch
            return alpha
        else:
            # weird config; be safe
            return 0.0

    def compute_loss_cl(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = self.calculate_superloss(loss, inputs).mean()

        cur_alpha = self._current_alpha()
        if cur_alpha != 0.0:
            loss = loss + (cur_alpha * l1_regularization(model))

        return (loss, outputs) if return_outputs else loss

    def compute_loss_non_cl(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        cur_alpha = self._current_alpha()
        if cur_alpha != 0.0:
            loss = loss + (cur_alpha * l1_regularization(model))

        return (loss, outputs) if return_outputs else loss
