import math

import torch
from torch import nn

from models.CrossAttention import AttentionBlock, spatial_to_tokens, tokens_to_spatial



def timestep_embedding(timesteps, dim):
    """
    timesteps: (B,)
    return: (B, dim)
    """
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.time_mlp = nn.Linear(time_dim, out_ch)

        self.act = nn.SiLU()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, context_dim=None):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_dim)
        self.attn = (
            AttentionBlock(out_ch, context_dim)
            if context_dim is not None else None
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t, context=None):
        h = self.conv(x, t)
        if self.attn is not None:
            h = self.attn(h, context)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, context_dim=None):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch, time_dim, context_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.act = nn.SiLU()

    def forward(self, x, t, context):
        h = self.block(x, t, context)
        x = self.down(h)
        x = self.act(x)
        return x, h


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, context_dim=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.block = ResBlock(in_ch + out_ch, out_ch, time_dim, context_dim)
        self.act = nn.SiLU()
    def forward(self, x, skip, t, context):
        x = self.up(x)
        x = self.act(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t, context)



class DiffusionUNet(nn.Module):
    def __init__(self, time_dim=128, context_dim=None):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.down1 = DownBlock(4, 32, time_dim, context_dim)
        self.down2 = DownBlock(32, 64, time_dim, context_dim)
        self.down3 = DownBlock(64, 128, time_dim, context_dim)

        self.bot = ResBlock(128, 128, time_dim, context_dim)

        self.up3 = UpBlock(128, 64, time_dim, context_dim)
        self.up2 = UpBlock(64, 32, time_dim, context_dim)
        self.up1 = UpBlock(32, 4, time_dim, context_dim)

        self.out = nn.Conv2d(4, 4, 1)

    def forward(self, x, t, context):
        t = timestep_embedding(t, self.time_dim)
        t = self.time_mlp(t)

        x, skip1 = self.down1(x, t, context)
        x, skip2 = self.down2(x, t, context)
        x, skip3 = self.down3(x, t, context)

        x = self.bot(x, t, context)

        x = self.up3(x, skip3, t, context)
        x = self.up2(x, skip2, t, context)
        x = self.up1(x, skip1, t, context)

        return self.out(x)
