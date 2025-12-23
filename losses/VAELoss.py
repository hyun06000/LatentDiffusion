import torch
import torch.nn.functional as F

def vae_loss(x, x_hat, mu, logvar):
    recon = F.mse_loss(x_hat, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + 0.0001 * kl
