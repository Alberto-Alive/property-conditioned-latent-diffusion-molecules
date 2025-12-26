from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from .schedules import DiffusionSchedule


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t).reshape(b, *((1,) * (len(x_shape) - 1)))
    return out


@dataclass
class DDPM:
    schedule: DiffusionSchedule
    prediction_type: str = "v"  # "eps" or "v"

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = extract(self.schedule.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mab = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def v_target(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = extract(self.schedule.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mab = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * eps - sqrt_1mab * x0

    def eps_from_v(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = extract(self.schedule.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mab = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_ab * x_t + sqrt_1mab * v

    def x0_from_v(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = extract(self.schedule.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mab = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_ab * x_t - sqrt_1mab * v

    def x0_from_eps(self, x_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = extract(self.schedule.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1mab = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_1mab * eps) / torch.clamp(sqrt_ab, min=1e-8)
