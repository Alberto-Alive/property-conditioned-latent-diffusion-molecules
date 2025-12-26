from __future__ import annotations
import argparse
import os
from dataclasses import asdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from moldifflite.models.ae_transformer import AEConfig, SelfiesTransformerAE, causal_attention_mask
from moldifflite.training.datasets import SelfiesDataset, collate_ae
from moldifflite.utils import Vocab, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pt", type=str, required=True)
    ap.add_argument("--vocab_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data = torch.load(args.data_pt, map_location="cpu")
    vocab = Vocab.from_json(args.vocab_json)

    ds = SelfiesDataset(data)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: collate_ae(b, pad_id=vocab.pad),
    )

    cfg = AEConfig(
        vocab_size=len(vocab.itos),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        max_len=args.max_len,
    )

    model = SelfiesTransformerAE(cfg, pad_id=vocab.pad).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=vocab.pad)

    save_json(os.path.join(args.out_dir, "ae_config.json"), asdict(cfg))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(dl, desc=f"AE epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch["x"].to(args.device)
            y_in = batch["y_in"].to(args.device)
            y_out = batch["y_out"].to(args.device)
            src_pad = batch["src_pad"].to(args.device)
            tgt_pad = batch["tgt_pad"].to(args.device)

            T = y_in.shape[1]
            cmask = causal_attention_mask(T, device=torch.device(args.device))

            logits, _z = model(x, y_in, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad, causal_mask=cmask)
            loss = crit(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=total_loss / max(n, 1))

        torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, os.path.join(args.out_dir, "ae.pt"))

    print(f"Saved AE to {os.path.join(args.out_dir, 'ae.pt')}")


if __name__ == "__main__":
    main()
