from __future__ import annotations
import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int, fourier_dim: int = 64):
        super().__init__()
        self.B = nn.Parameter(torch.randn(fourier_dim) * 10.0)
        self.proj = nn.Sequential(
            nn.Linear(2 * fourier_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        angles = t[:, None] * self.B[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(emb)
