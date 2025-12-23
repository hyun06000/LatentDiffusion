from torch import nn

class DigitTextEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.embedding = nn.Embedding(11, embed_dim)

    def forward(self, labels):
        x = self.embedding(labels).unsqueeze(1)  # (B, 1, D)
        x = self.norm(x)
        return x