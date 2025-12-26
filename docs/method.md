## The how to

1. **Selfies autoencoder**
- tokenise selfies into bracket tokens
- train a transformer seq2seq model with teacher forcing
- latent vector is mean-pooled encoder states -> projection
2. **Latent diffusion**
- train DDPM in latent space with cosine schedule
- predict `v` (or `eps`) with an MLP denoiser conditioned on normalised properties
3. **Sampling**
4. **Decoding & evaluation**