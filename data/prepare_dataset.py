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
    smiles_in = []
    with open(args.smiles_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                smiles_in.append(s)
    random.shuffle(smiles_in)
    smiles_in = smiles_in[: args.max_mols]

    token_lists: List[List[str]] = []
    canon_smiles: List[str] = []
    props_list: List[List[float]] = []

    for s in tqdm(smiles_in, desc="Filtering + SELFIES + props"):
        c = canonicalize_smiles(s)
        if c is None:
            continue
        pr = compute_properties(c, prop_names=prop_names)
        if pr is None:
            continue
        sf = smiles_to_selfies(c)
        if sf is None:
            continue
        toks = split_selfies(sf)
        if len(toks) < args.min_len or len(toks) > args.max_len:
            continue

        token_lists.append(toks)
        canon_smiles.append(c)
        props_list.append([float(pr[p]) for p in prop_names])

    if len(token_lists) == 0:
        raise RuntimeError("No molecules passed filtering. Check RDKit and input SMILES.")

    vocab = build_vocab(token_lists, min_freq=1)
    save_json(os.path.join(args.out_dir, "vocab.json"), vocab)

    stoi = vocab["stoi"]
    bos, eos = stoi["<bos>"], stoi["<eos>"]

    seqs = []
    for toks in token_lists:
        ids = [bos] + encode_tokens(toks, stoi) + [eos]
        seqs.append(torch.tensor(ids, dtype=torch.long))

    props = torch.tensor(np.array(props_list, dtype="float32"))
    dataset = {"seqs": seqs, "props": props, "prop_names": prop_names, "smiles": canon_smiles}
    out_pt = os.path.join(args.out_dir, "dataset.pt")
    torch.save(dataset, out_pt)

    mean = props.mean(dim=0).tolist()
    std = props.std(dim=0).tolist()
    save_json(os.path.join(args.out_dir, "prop_normalizer.json"), {"prop_names": prop_names, "mean": mean, "std": std})

    # convenience for novelty checks
    with open(os.path.join(args.out_dir, "train_smiles.txt"), "w", encoding="utf-8") as f:
        for s in canon_smiles:
            f.write(s + "\n")

    print(f"Saved {len(seqs)} molecules to {out_pt}")
    print(f"Vocab size: {len(vocab['itos'])}")


if __name__ == "__main__":
    main()
