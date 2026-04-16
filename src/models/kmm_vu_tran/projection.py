import torch.nn as nn
import torch.nn.functional as F

class Projection(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, in_dim)
        x = self.linear(x)
        x = self.norm(x)
        x = F.gelu(x)
        return x  # (B, T, hidden_dim)