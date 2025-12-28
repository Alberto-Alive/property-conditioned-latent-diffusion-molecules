Ok, let's outline the major parts of the model:

1. Autoencoder (compression model) - this one I understand
2. Latent diffusion model (the generator)

! so far the idea is that the autoencoder works as a compression model that outputs a latent the diffusion model can work on - less pixels = less computation so the paper manually separated the compression algorithms from the generation algorithm.


3. Conditioning model
- this would be a separate encoder for text to image

4. The diffusion process plumbing
this is fixed end to end - is just a way to tell the model how to add noise and remove it 