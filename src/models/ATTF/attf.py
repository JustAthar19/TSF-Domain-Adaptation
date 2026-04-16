from src.models.ATTF.value_embedding import ValueEmbedding
from src.models.ATTF.pattern_embedding import PatternEmbedding
from src.models.ATTF.attention import SharedAttention
from src.models.ATTF.decoder import Decoder

import torch.nn as nn


class AttF(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_steps=7):
        super().__init__()

        self.value_embed = ValueEmbedding(input_dim, hidden_dim)
        self.pattern_embed = PatternEmbedding(input_dim, hidden_dim)
        self.attention = SharedAttention(hidden_dim)
        self.decoder = Decoder(hidden_dim, output_steps)
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        v = self.value_embed(x)
        p = self.pattern_embed(x)
        attn_out = self.attention(p, v)

        y_hat = self.decoder(attn_out)         # (B, 7)
        x_recon = self.reconstruction_head(attn_out)
        
        return y_hat, x_recon