# property-conditioned-latent-diffusion-molecules
From-scratch PyTorch latent diffusion for de novo molecule generation (SELFIES AE + DDPM/DDIM + CFG) with RDKit evaluation.


Implemented:

- a selfies sequence autoencoder (Transformer) to map molecules to a continuous latent vector.

- a latent-space diffusion model (DDPM training + DDIM sampling) conditioned on desired properties.

- classifier-free guidance (CFG) at sampling time.

- RDKit-based evaluation: validity / uniqueness / novelty / constraint satisfaction.

> Core inspirations: DDPM (Ho et al., 2020), DDIM (Song et al., 2020), Latent Diffusion (Rombach et al., 2022), Classifier-Free Guidance (Ho & Salimans, 2022), cosine schedule (Nichol & Dhariwal, 2021), SELFIES (Krenn et al.).