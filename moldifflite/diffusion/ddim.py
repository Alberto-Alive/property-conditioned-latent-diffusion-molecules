from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import torch
from .ddpm import extract, DDPM


@dataclass
class DDIMSampler:
    ddpm: DDPM
    eta: float = 0.0

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        shape: torch.Size,
        cond: Optional[torch.Tensor],
        steps: int,
        device: torch.device,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        T = self.ddpm.schedule.timesteps
        steps = min(steps, T)
        ts = torch.linspace(T - 1, 0, steps, device=device).long()

        x = torch.randn(shape, device=device)

        for i in range(steps):
            t = ts[i].expand(shape[0])

            if cfg_scale > 0.0 and cond is not None:
                pred_u = model_fn(x, t, None)
                pred_c = model_fn(x, t, cond)
                pred = pred_u + cfg_scale * (pred_c - pred_u)
            else:
                pred = model_fn(x, t, cond)

            if self.ddpm.prediction_type == "v":
                eps = self.ddpm.eps_from_v(x, pred, t)
                x0 = self.ddpm.x0_from_v(x, pred, t)
            elif self.ddpm.prediction_type == "eps":
                eps = pred
                x0 = self.ddpm.x0_from_eps(x, eps, t)
            else:
                raise ValueError(f"Unknown prediction_type: {self.ddpm.prediction_type}")

            ab = extract(self.ddpm.schedule.alphas_cumprod, t, x.shape)
            ab_prev = extract(self.ddpm.schedule.alphas_cumprod_prev, t, x.shape)

            sigma = (
                self.eta
                * torch.sqrt((1 - ab_prev) / torch.clamp(1 - ab, min=1e-8))
                * torch.sqrt(torch.clamp(1 - ab / torch.clamp(ab_prev, min=1e-8), min=0.0))
            )
            dir_xt = torch.sqrt(torch.clamp(1 - ab_prev - sigma**2, min=0.0)) * eps
            noise = sigma * torch.randn_like(x) if self.eta > 0 else 0.0
            x = torch.sqrt(torch.clamp(ab_prev, min=0.0)) * x0 + dir_xt + noise

        return x
