import torch
from torch import nn



class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        """
        x: (B, N, dim)        image tokens
        context: (B, 1, C)   digit embedding
        """
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        return self.to_out(out)


def spatial_to_tokens(x):
    B, C, H, W = x.shape
    return x.view(B, C, H * W).permute(0, 2, 1)


def tokens_to_spatial(x, H, W):
    B, N, C = x.shape
    return x.permute(0, 2, 1).view(B, C, H, W)


class AttentionBlock(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, context_dim)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_tokens = spatial_to_tokens(x)
        x_tokens = self.norm(x_tokens)
        x_tokens = self.attn(x_tokens, context)
        return tokens_to_spatial(x_tokens, H, W) + x


