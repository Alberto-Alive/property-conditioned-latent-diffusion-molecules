from __future__ import annotations
import argparse
import os
import random
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm

from moldifflite.chemistry.selfies_utils import smiles_to_selfies, split_selfies
from moldifflite.chemistry.rdkit_props import DEFAULT_PROP_NAMES, canonicalize_smiles, compute_properties
from moldifflite.utils import save_json


SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

def build_vocab(token_lists: List [List[str]], min_freq: int = 1) -> Dict:
    from collections import Counter
    c = Counter()
    for toks in token_lists:
        c.update(toks)
    
    itos = list(SPECIAL)
    for tok, f in c.items():
        if f >= min_freq and tok not in SPECIAL:
            itos.append(tok)
    stoi = {t: i for i, t in enumerate(itos)}
    return {
        "itos" : itos,
        "stoi" : stoi,
        "special_tokens" : {"pad" : stoi["<pad>"], "bos":stoi["<bos>"], "eos": stoi["<eos>"], "unk": stoi["<unk>"]}
    }

def encode_tokens(tokens: List[str], stoi: Dict[str, int]) -> List[int]:
    unk = stoi["<unk>"]
    return [stoi.get(t, nk)]