from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from .positional import SinusoidalPositionalEncoding


@dataclass
class AEConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 256
    latent_dim: int = 256


class SelfiesTransformerAE(nn.Module):
    def __init__(self, cfg: AEConfig, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.to_latent = nn.Sequential(nn.Linear(cfg.d_model, cfg.latent_dim), nn.Tanh())

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_layers)
        self.latent_to_mem = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    def encode(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.pos(self.tok_emb(x))
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is None:
            pooled = h.mean(dim=1)
        else:
            keep = (~src_key_padding_mask).float()
            denom = keep.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (h * keep.unsqueeze(-1)).sum(dim=1) / denom
        return self.to_latent(pooled)

    def decode(
        self,
        z: torch.Tensor,
        y_in: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mem = self.latent_to_mem(z).unsqueeze(1)
        y = self.pos(self.tok_emb(y_in))
        h = self.decoder(
            tgt=y,
            memory=mem,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )
        return self.out(h)

    def forward(
        self,
        x: torch.Tensor,
        y_in: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ):
        z = self.encode(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.decode(z, y_in, tgt_key_padding_mask=tgt_key_padding_mask, causal_mask=causal_mask)
        return logits, z


def causal_attention_mask(T: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)
