from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from moldifflite.utils import pad_sequences


class SelfiesDataset(Dataset):
    def __init__(self, data: Dict[str, Any]):
        self.seqs = data["seqs"]
        self.props = data["props"]
        self.smiles = data.get("smiles", None)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        return {"seq": self.seqs[idx], "props": self.props[idx], "smiles": self.smiles[idx] if self.smiles else None}


def collate_ae(batch: List[Dict[str, Any]], pad_id: int):
    seqs = [b["seq"] for b in batch]
    x, _mask = pad_sequences(seqs, pad_value=pad_id)
    y_in = x[:, :-1].contiguous()
    y_out = x[:, 1:].contiguous()
    src_pad = (x == pad_id)
    tgt_pad = (y_in == pad_id)
    props = torch.stack([b["props"] for b in batch], dim=0)
    return {"x": x, "y_in": y_in, "y_out": y_out, "src_pad": src_pad, "tgt_pad": tgt_pad, "props": props}


class LatentDataset(Dataset):
    def __init__(self, latents: torch.Tensor, cond: torch.Tensor, smiles: Optional[List[str]] = None):
        self.latents = latents
        self.cond = cond
        self.smiles = smiles

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int):
        return {"z": self.latents[idx], "cond": self.cond[idx], "smiles": self.smiles[idx] if self.smiles else None}
