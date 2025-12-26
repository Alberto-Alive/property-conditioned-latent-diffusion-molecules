## The how to

1. **Selfies autoencoder**
- tokenise selfies into bracket tokens
- train a transformer seq2seq model with teacher forcing
- latent vector is mean-pooled encoder states -> projection
2. **Latent diffusion**
- train DDPM in latent space with cosine schedule
- predict `v` (or `eps`) with a multi layer perceptron denoiser conditioned on normalised properties
- CFG training: randomly drop conditioning with probability `p_drop`
3. **Sampling**
- use DDIM for fast sampling with `steps` and << `T`
- apply classifier free guidance:
`pred = pred_uncond + s*(pred_cond - pred_uncond)`
4. **Decoding & evaluation**
- decode latent via AE decoder -> selfies -> smiles
- evaluate with RDKit: validity, uniqueness, novelty.