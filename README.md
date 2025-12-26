# property-conditioned-latent-diffusion-molecules
From-scratch PyTorch latent diffusion for de novo molecule generation (SELFIES AE + DDPM/DDIM + CFG) with RDKit evaluation.


Implemented:

- a selfies sequence autoencoder (Transformer) to map molecules to a continuous latent vector.

- a latent-space diffusion model (DDPM training + DDIM sampling) conditioned on desired properties.

- classifier-free guidance (CFG) at sampling time.

- RDKit-based evaluation: validity / uniqueness / novelty / constraint satisfaction.

> Core inspirations: DDPM (Ho et al., 2020), DDIM (Song et al., 2020), Latent Diffusion (Rombach et al., 2022), Classifier-Free Guidance (Ho & Salimans, 2022), cosine schedule (Nichol & Dhariwal, 2021), SELFIES (Krenn et al.).



## 1) Setup

### Recommended (conda)
```bash
conda create -n moldifflite python=3.10 -y
conda activate moldifflite
pip install -r requirements.txt
```

**RDKit** is easiest via conda-forge:
```bash
conda install -c conda-forge rdkit -y
```

## 2) Data

Provide a file with **one SMILES per line**, e.g. `data/raw/smiles.txt`.

Then run:
```bash
python data/prepare_dataset.py \
  --smiles_path data/raw/smiles.txt \
  --out_dir data/processed \
  --max_len 120 \
  --max_mols 300000
```

Outputs:
- `data/processed/dataset.pt` (tokenized SELFIES + properties + smiles list)
- `data/processed/vocab.json`

## 3) Train the autoencoder

```bash
python moldifflite/training/train_ae.py \
  --data_pt data/processed/dataset.pt \
  --vocab_json data/processed/vocab.json \
  --out_dir runs/ae \
  --epochs 10
```

## 4) Encode latents (optional but faster for diffusion)

```bash
python moldifflite/training/encode_latents.py \
  --data_pt data/processed/dataset.pt \
  --vocab_json data/processed/vocab.json \
  --ae_ckpt runs/ae/ae.pt \
  --out_pt data/processed/latents.pt
```

## 5) Train diffusion in latent space

```bash
python moldifflite/training/train_diffusion.py \
  --latents_pt data/processed/latents.pt \
  --out_dir runs/diffusion \
  --epochs 50
```

## 6) Sample molecules

Example: target QED=0.7, logP=2.0, MW=350, TPSA=60, HBD=1.

```bash
python moldifflite/training/sample.py \
  --vocab_json data/processed/vocab.json \
  --ae_ckpt runs/ae/ae.pt \
  --diff_ckpt runs/diffusion/diffusion.pt \
  --n 200 \
  --cfg_scale 3.0 \
  --cond "qed=0.7,logp=2.0,mw=350,tpsa=60,hbd=1"
```

Outputs:
- `runs/samples/samples.csv` with SMILES + computed properties + validity
- `runs/samples/metrics.json`

## Notes

- Conditioning vector uses these RDKit properties by default:
  `qed, logp, mw, tpsa, hbd`.
  You can add more in `moldifflite/chemistry/rdkit_props.py`.

- This is a compact reference implementation meant for learning + showcasing, not a production system.

## License
MIT
