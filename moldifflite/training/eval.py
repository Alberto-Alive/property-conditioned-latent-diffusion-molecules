from __future__ import annotations
import argparse
import os
import pandas as pd
from moldifflite.chemistry.metrics import compute_basic_metrics
from moldifflite.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_csv", type=str, required=True)
    ap.add_argument("--train_smiles_txt", type=str, default=None)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.samples_csv)
    gen = [s if isinstance(s, str) and len(s) > 0 else None for s in df["smiles"].tolist()]

    train_set = None
    if args.train_smiles_txt and os.path.exists(args.train_smiles_txt):
        with open(args.train_smiles_txt, "r", encoding="utf-8") as f:
            train_set = set([line.strip() for line in f if line.strip()])

    metrics = compute_basic_metrics([s for s in gen if s is not None], train_smiles_set=train_set)
    out_json = args.out_json or os.path.join(os.path.dirname(args.samples_csv), "metrics.json")
    save_json(out_json, metrics)
    print(metrics)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
