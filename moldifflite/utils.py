from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad: int
    bos: int
    eos: int
    unk: int

    @classmethod
    def from_json(cls, path: str) -> "Vocab":
        obj = load_json(path)
        stoi = {k: int(v) for k, v in obj["stoi"].items()}
        itos = list(obj["itos"])
        return cls(
            stoi=stoi,
            itos=itos,
            pad=obj["special_tokens"]["pad"],
            bos=obj["special_tokens"]["bos"],
            eos=obj["special_tokens"]["eos"],
            unk=obj["special_tokens"]["unk"],
        )

    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "special_tokens": {"pad": self.pad, "bos": self.bos, "eos": self.eos, "unk": self.unk},
        }

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]


def pad_sequences(seqs: List[torch.Tensor], pad_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        padded: [B, T]
        attn_mask: [B, T] with True for real tokens, False for padding
    """
    lengths = [int(s.numel()) for s in seqs]
    max_len = max(lengths)
    batch = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        l = int(s.numel())
        batch[i, :l] = s
        mask[i, :l] = True
    return batch, mask