from __future__ import annotations
import argparse
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from moldifflite.chemistry.metrics import compute_basic_metrics
from moldifflite.chemistry.rdkit_props import PropNormalizer, canonicalize_smiles, compute_properties
from moldifflite.chemistry.selfies_utils import join_selfies, selfies_to_smiles
from moldifflite.diffusion.ddpm import DDPM
from moldifflite.diffusion.ddim import DDIMSampler
from moldifflite.diffusion.schedules import DiffusionSchedule
from moldifflite.models.ae_transformer import AEConfig, SelfiesTransformerAE
from moldifflite.models.diffusion_mlp import DiffusionMLP, DiffusionMLPConfig
from moldifflite.utils import Vocab, save_json, set_seed


def parse_cond(cond_str: str, prop_names: List[str]) -> Dict[str, float]:
    out = {}
    parts = [p.strip() for p in cond_str.split(",") if p.strip()]
    for p in parts:
        k, v = p.split("=")
        out[k.strip().lower()] = float(v)
    for name in prop_names:
        out.setdefault(name, float("nan"))
    return out


def decode_latents_to_smiles(ae: SelfiesTransformerAE, vocab: Vocab, z: torch.Tensor, max_len: int, device: str):
    ae.eval()
    bos, eos, pad = vocab.bos, vocab.eos, vocab.pad
    z = z.to(device)

    B = z.shape[0]
    y = torch.full((B, 1), bos, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = ae.decode(z, y_in=y, tgt_key_padding_mask=(y == pad), causal_mask=None)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            y = torch.cat([y, next_id], dim=1)
            if torch.all(next_id.squeeze(1) == eos):
                break

    smiles_out: List[Optional[str]] = []
    y_cpu = y.detach().cpu().tolist()
    for ids in y_cpu:
        toks = []
        for tid in ids[1:]:
            if tid == eos:
                break
            if tid == pad:
                continue
            toks.append(vocab.itos[tid] if 0 <= tid < len(vocab.itos) else "<unk>")
        selfies = join_selfies(toks)
        smi = selfies_to_smiles(selfies)
        if smi is None:
            smiles_out.append(None)
        else:
            smiles_out.append(canonicalize_smiles(smi))
    return smiles_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_json", type=str, required=True)
    ap.add_argument("--ae_ckpt", type=str, required=True)
    ap.add_argument("--diff_ckpt", type=str, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--cond", type=str, default="")
    ap.add_argument("--max_decode_len", type=int, default=140)
    ap.add_argument("--out_dir", type=str, default="runs/samples")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    vocab = Vocab.from_json(args.vocab_json)

    ae_ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    ae_cfg = AEConfig(**ae_ckpt["cfg"])
    ae = SelfiesTransformerAE(ae_cfg, pad_id=vocab.pad).to(args.device)
    ae.load_state_dict(ae_ckpt["model"], strict=True)
    ae.eval()

    dckpt = torch.load(args.diff_ckpt, map_location="cpu")
    dcfg = DiffusionMLPConfig(**dckpt["cfg"])
    model = DiffusionMLP(dcfg).to(args.device)
    model.load_state_dict(dckpt["model"], strict=True)
    model.eval()

    ddpm_cfg = dckpt["ddpm"]
    schedule = DiffusionSchedule.create(ddpm_cfg["timesteps"], schedule=ddpm_cfg["schedule"])
    ddpm = DDPM(schedule=schedule, prediction_type=ddpm_cfg["prediction_type"])
    sampler = DDIMSampler(ddpm=ddpm, eta=args.eta)

    normalizer = PropNormalizer.from_json(dckpt["prop_normalizer"])
    prop_names = normalizer.prop_names

    if args.cond.strip():
        cond_raw = parse_cond(args.cond, prop_names)
        vec = []
        for i, p in enumerate(prop_names):
            v = cond_raw[p]
            if np.isnan(v):
                v = float(normalizer.mean[i])
            vec.append(v)
        cond_raw_vec = np.array(vec, dtype="float32")
    else:
        cond_raw_vec = np.array(normalizer.mean, dtype="float32")

    cond_norm = (cond_raw_vec - np.array(normalizer.mean, dtype="float32")) / np.maximum(
        np.array(normalizer.std, dtype="float32"), 1e-6
    )
    cond = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).repeat(args.n, 1).to(args.device)

    def model_fn(x_t, t, cond_tensor):
        return model(x_t, t, cond_tensor)

    z = sampler.sample(
        model_fn=model_fn,
        shape=torch.Size([args.n, dcfg.latent_dim]),
        cond=cond,
        steps=args.steps,
        device=torch.device(args.device),
        cfg_scale=args.cfg_scale,
    )

    smiles = decode_latents_to_smiles(ae, vocab, z, max_len=args.max_decode_len, device=args.device)

    rows = []
    for s in smiles:
        row = {"smiles": s, "valid": False}
        if s is not None:
            pr = compute_properties(s, prop_names=prop_names)
            if pr is not None:
                row["valid"] = True
                for p in prop_names:
                    row[p] = float(pr[p])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "samples.csv")
    df.to_csv(csv_path, index=False)

    train_smiles_path = os.path.join(os.path.dirname(args.vocab_json), "train_smiles.txt")
    train_set = None
    if os.path.exists(train_smiles_path):
        with open(train_smiles_path, "r", encoding="utf-8") as f:
            train_set = set([line.strip() for line in f if line.strip()])

    metrics = compute_basic_metrics([s for s in smiles if s is not None], train_smiles_set=train_set)
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    save_json(metrics_path, metrics)

    print(f"Wrote {csv_path}")
    print(f"Wrote {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()
