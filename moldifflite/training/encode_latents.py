from __future__ import annotations
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from moldifflite.models.ae_transformer import AEConfig, SelfiesTransformerAE
from moldifflite.training.datasets import SelfiesDataset, collate_ae
from moldifflite.utils import Vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pt", type=str, required=True)
    ap.add_argument("--vocab_json", type=str, required=True)
    ap.add_argument("--ae_ckpt", type=str, required=True)
    ap.add_argument("--out_pt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = torch.load(args.data_pt, map_location="cpu")
    vocab = Vocab.from_json(args.vocab_json)

    ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    cfg = AEConfig(**ckpt["cfg"])
    model = SelfiesTransformerAE(cfg, pad_id=vocab.pad).to(args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = SelfiesDataset(data)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: collate_ae(b, pad_id=vocab.pad),
    )

    latents = []
    cond_raw = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Encoding latents"):
            x = batch["x"].to(args.device)
            src_pad = batch["src_pad"].to(args.device)
            z = model.encode(x, src_key_padding_mask=src_pad).detach().cpu()
            latents.append(z)
            cond_raw.append(batch["props"].cpu())

    latents = torch.cat(latents, dim=0)
    cond_raw = torch.cat(cond_raw, dim=0)

    out = {"latents": latents, "cond_raw": cond_raw, "prop_names": data.get("prop_names", None), "smiles": data.get("smiles", None)}
    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    torch.save(out, args.out_pt)
    print(f"Saved latents to {args.out_pt}  latents={tuple(latents.shape)}  cond={tuple(cond_raw.shape)}")


if __name__ == "__main__":
    main()
