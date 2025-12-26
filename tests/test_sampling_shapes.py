import torch
from moldifflite.diffsuion.schedules import DiffusionSchedule
from moldifflite.diffusion.ddpm import DDPM
from moldifflite.diffusion.ddim import DDIMSampler
from moldifflite.models.diffusion_mlp import DiffusionMLP, DiffusionMLPConfig

@torch.no_grad()
def test_ddim_shapes():
    schedule = DiffusionSchedule.create(100, "cosine")
    ddpm = DDPM(schedule=schedule, prediction_type="v")
    cfg = DiffusionMLPConfig(latent_dim=32, cond_dim=5, hidden_dim=64, depth=2, time_embed_dim=32)
    model = DiffusionMLP(cfg)
    sampler = DDIMSampler(ddpm=ddpm, eta=0.0)
    
    def fn(x, t, c):
        return model(x, t, c)
    
    z = sampler