import torch
import torch.nn as nn

class SharedAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pattern, value):
        # pattern: (B, T, H)
        # value:   (B, T, H)

        Q = self.q_proj(pattern)   # (B, T, H)
        K = self.k_proj(pattern)   # (B, T, H)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (pattern.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # IMPORTANT: V = value (NOT pattern)
        out = torch.matmul(attn_weights, value)  # (B, T, H)

        return out