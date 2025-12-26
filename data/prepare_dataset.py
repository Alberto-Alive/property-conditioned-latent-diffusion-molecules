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
    return [stoi.get(t, unk) for t in tokens]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_mols", type=int, default=300000)
    ap.add_argument("--prop_names", type=str, default=",".join(DEFAULT_PROP_NAMES))
    
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    prop_names = [p.strip() for p in args.prop_names.split(",") if p.strip()]
    