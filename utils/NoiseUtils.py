import torch


class LatentNoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(0.0001, 0.012, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, z0, t):
        alpha_bar = self.alpha_bars.to(z0.device)[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(z0)
        zt = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1 - alpha_bar) * noise
        return zt, noise

