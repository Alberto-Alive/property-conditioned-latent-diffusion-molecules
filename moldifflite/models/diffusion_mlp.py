from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from .time_embed import FourierTimeEmbedding


@dataclass
class DiffusionMLPConfig:
    latent_dim: int = 256
    cond_dim: int = 5
    hidden_dim: int = 1024
    depth: int = 4
    dropout: float = 0.0
    time_embed_dim: int = 256


class DiffusionMLP(nn.Module):
    def __init__(self, cfg: DiffusionMLPConfig):
        super().__init__()
        self.cfg = cfg
        self.time_emb = FourierTimeEmbedding(cfg.time_embed_dim, fourier_dim=64)
        in_dim = cfg.latent_dim + cfg.time_embed_dim + cfg.cond_dim

        layers = []
        d = in_dim
        for _ in range(cfg.depth):
            layers += [nn.Linear(d, cfg.hidden_dim), nn.SiLU(), nn.Dropout(cfg.dropout)]
            d = cfg.hidden_dim
        layers += [nn.Linear(d, cfg.latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        te = self.time_emb(t)
        if cond is None:
            cond = torch.zeros((x_t.shape[0], self.cfg.cond_dim), device=x_t.device, dtype=x_t.dtype)
        return self.net(torch.cat([x_t, te, cond], dim=-1))
