from __future__ import annotations
import argparse
import os
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from moldifflite.chemistry.rdkit_props import PropNormalizer
from moldifflite.diffusion.ddpm import DDPM
from moldifflite.diffusion.losses import mse_loss
from moldifflite.diffusion.schedules import DiffusionSchedule
from moldifflite.models.diffusion_mlp import DiffusionMLP, DiffusionMLPConfig
from moldifflite.training.datasets import LatentDataset
from moldifflite.utils import load_json, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents_pt", type=str, required=True)
    ap.add_argument("--prop_normalizer_json", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--schedule", type=str, default="cosine")
    ap.add_argument("--prediction_type", type=str, default="v", choices=["v", "eps"])
    ap.add_argument("--cfg_drop_prob", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    pack = torch.load(args.latents_pt, map_location="cpu")
    z = pack["latents"].float()
    cond_raw = pack["cond_raw"].float()

    if args.prop_normalizer_json is None:
        args.prop_normalizer_json = os.path.join(os.path.dirname(args.latents_pt), "prop_normalizer.json")
    norm_obj = load_json(args.prop_normalizer_json)
    normalizer = PropNormalizer.from_json(norm_obj)

    mean = torch.tensor(normalizer.mean, dtype=torch.float32).view(1, -1)
    std = torch.tensor(normalizer.std, dtype=torch.float32).view(1, -1).clamp(min=1e-6)
    cond = (cond_raw - mean) / std

    ds = LatentDataset(z, cond, smiles=pack.get("smiles", None))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    schedule = DiffusionSchedule.create(args.timesteps, schedule=args.schedule)
    ddpm = DDPM(schedule=schedule, prediction_type=args.prediction_type)

    cfg = DiffusionMLPConfig(
        latent_dim=z.shape[1],
        cond_dim=cond.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        time_embed_dim=z.shape[1],
    )
    model = DiffusionMLP(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_json(
        os.path.join(args.out_dir, "diffusion_config.json"),
        {
            "ddpm": {"timesteps": args.timesteps, "schedule": args.schedule, "prediction_type": args.prediction_type},
            "model": asdict(cfg),
            "prop_normalizer_json": args.prop_normalizer_json,
        },
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        pbar = tqdm(dl, desc=f"Diff epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x0 = batch["z"].to(args.device)
            c = batch["cond"].to(args.device)

            if args.cfg_drop_prob > 0:
                drop = (torch.rand((x0.shape[0], 1), device=args.device) < args.cfg_drop_prob).float()
                c = c * (1.0 - drop)

            t = torch.randint(0, ddpm.schedule.timesteps, (x0.shape[0],), device=args.device, dtype=torch.long)
            eps = torch.randn_like(x0)
            x_t = ddpm.q_sample(x0, t, noise=eps)

            pred = model(x_t, t, c)
            target = ddpm.v_target(x0, eps, t) if ddpm.prediction_type == "v" else eps

            loss = mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * x0.size(0)
            n += x0.size(0)
            pbar.set_postfix(loss=total / max(n, 1))

        torch.save(
            {"model": model.state_dict(), "cfg": asdict(cfg),
             "ddpm": {"timesteps": args.timesteps, "schedule": args.schedule, "prediction_type": args.prediction_type},
             "prop_normalizer": norm_obj},
            os.path.join(args.out_dir, "diffusion.pt"),
        )

    print(f"Saved diffusion model to {os.path.join(args.out_dir, 'diffusion.pt')}")


if __name__ == "__main__":
    main()
