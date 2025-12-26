# property-conditioned-latent-diffusion-molecules
From-scratch PyTorch latent diffusion for de novo molecule generation (SELFIES AE + DDPM/DDIM + CFG) with RDKit evaluation.


Implemented:

- a selfies sequence autoencoder (Transformer) to map molecules to a continuous latent vector.

- a latent-space diffusion model (DDPM training + DDIM sampling) conditioned on desired properties.

- classifier-free guidance (CFG) at sampling time.

- R