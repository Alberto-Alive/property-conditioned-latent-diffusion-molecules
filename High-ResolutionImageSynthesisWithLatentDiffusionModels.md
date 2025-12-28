Ok, let's outline the major parts of the model:

1. Autoencoder (compression model) - this one I understand
2. Latent diffusion model (the generator)

! so far the idea is that the autoencoder works as a compression model that outputs a latent the diffusion model can work on - less pixels = less computation so the paper manually separated the compression algorithms from the generation algorithm.


3. Conditioning model
- this would be a separate encoder for text to image

4. The diffusion process plumbing
this is fixed end to end - is just a way to tell the model how to add noise and remove it 


P.S. Comparring this to how humans draw images it feels like humans store relative information like relative positions as this information is what can adapt memorised information. Simply put being able to understand relativism allows adaptation of static information across different context.

As an example, drawing a chair on the floor is easy but asking the model to draw a chair on the ceiling will make the model struggle... whereas a human knows the relative position of the chair in every aspect (relative within the context and relative to the human body)... plus planning.