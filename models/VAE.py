import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTVAEEncoder(nn.Module):
    def __init__(self, latent_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 28 → 14
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 14 → 7
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, latent_ch * 2, 3, 1, 1)  # μ, logvar
        )

    def forward(self, x):
        h = self.net(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class MNISTVAEDecoder(nn.Module):
    def __init__(self, latent_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 64, 4, 2, 1),  # 7 → 14
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),         # 14 → 28
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class MNISTVAE(nn.Module):
    def __init__(self, latent_ch=4):
        super().__init__()
        self.encoder = MNISTVAEEncoder(latent_ch)
        self.decoder = MNISTVAEDecoder(latent_ch)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar