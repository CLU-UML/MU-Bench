'''Source: https://github.com/timgaripov/dnn-mode-connectivity/blob/master/curves.py'''

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from scipy.special import binom


class LinearInterpolation(nn.Module):
    def __init__(self, num_bends):
        super().__init__()
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def forward(self, t):
        t = t.to(self.range.device)
        return torch.tensor([1-t, t], device=t.device)


class Bezier(nn.Module):
    def __init__(self, num_bends):
        super().__init__()
        self.register_buffer(
            'binom',
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32))
        )
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        t = t.to(self.range.device)
        return self.binom * torch.pow(t, self.range) * torch.pow((1.0 - t), self.rev_range)


class PolyChain(nn.Module):
    def __init__(self, num_bends):
        super().__init__()
        self.num_bends = num_bends
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def forward(self, t):
        t = t.to(self.range.device)
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))


class CurveModel(nn.Module):
    def __init__(self, base_model_fn, curve_type, init_start, init_end, midpoint_ckpt, num_bends):
        """
        Initialize the Curve Model.
        Args:
            base_model_fn: A function that returns an instance of the base model (e.g., ResNet, MLP).
            num_bends: Number of bends (points) in the curve.
        """
        super().__init__()
        self.num_bends = num_bends
        self.curve = curve_mapping[curve_type](num_bends)

        # Initialize non-endpoints with random weights (trainable)
        models = []
        for _ in range(num_bends-2):
            m = base_model_fn.from_pretrained(midpoint_ckpt)
            # m = m.apply(m._init_weights)
            m.eval()
            models.append(m)
        self.models = nn.ModuleList(models)

        # Load endpoints (frozen)
        start_model = base_model_fn.from_pretrained(init_start)
        end_model = base_model_fn.from_pretrained(init_end)
        for n, p in start_model.named_parameters():
            if p.requires_grad:
                self.register_buffer(f"start_{n.replace('.', '__')}", p.detach())
        for n, p in end_model.named_parameters():
            if p.requires_grad:
                self.register_buffer(f"end_{n.replace('.', '__')}", p.detach())

        # interpolated_params = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
        # for n in interpolated_params.keys():
        #     interpolated_params[n] = 0.5 * getattr(self, f"start_{n.replace('.', '__')}") + 0.5 * getattr(self, f"end_{n.replace('.', '__')}")
        # self.models[0].load_state_dict(interpolated_params, strict=False)
        del start_model, end_model

        # Interpolated model
        self.final_model = base_model_fn.from_pretrained(init_start)

    def interpolate_weights(self, t):
        """Interpolate the model parameters based on the weights."""
        weights = self.curve(t)

        # Placeholder for interpolated weights
        interpolated_params = {n: torch.zeros_like(p) for n, p in self.models[0].named_parameters() if p.requires_grad}

        # Get current weights of end points
        state_dicts = [m.state_dict() for m in self.models]

        for n in interpolated_params.keys():
            try:
                for w, d in zip(weights[1:-1], state_dicts):
                    interpolated_params[n] += w * d[n]
                interpolated_params[n] += weights[0] * getattr(self, f"start_{n.replace('.', '__')}")
                interpolated_params[n] += weights[-1] * getattr(self, f"end_{n.replace('.', '__')}")
            except Exception as e:
                print(e)
                breakpoint()

        return interpolated_params

    def forward(self, **kwargs):
        if self.training:
            if 't' in kwargs:
                t = kwargs.pop('t')
            else:
                t = torch.rand(1)
                # t = torch.empty(1).normal_(mean=0.5, std=0.1)
                # t = torch.clamp(t, min=0, max=1)
            interpolated_params = self.interpolate_weights(t)
            self.final_model.load_state_dict(interpolated_params, strict=False)

        # print('xsssssssssssss', [p for p in self.models[0].parameters()][0].sum())
        # interpolated_params = self.interpolate_weights(t)
        # self.final_model.load_state_dict(interpolated_params)

        return self.final_model(**kwargs)

curve_mapping = {
    'linear': LinearInterpolation,
    'bezier': Bezier,
    'polychain': PolyChain,
}
