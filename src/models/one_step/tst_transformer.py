import math
from typing import List

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (Khanal et al., AAAI AI4TS 2024)."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


class TSTransformer(nn.Module):
    """
    Time-Series Transformer for one-step fine-tuning domain adaptation.
    Input projection -> sinusoidal PE -> encoder stack -> linear decoder.
    """

    def __init__(
        self,
        n_features: int,
        input_len: int,
        horizon: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_len = input_len
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)
        self.decoder = nn.Linear(d_model * input_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.decoder(x)

    def layer_groups(self) -> List[List[nn.Parameter]]:
        """Parameter groups ordered top (output) -> bottom (input) for gradual unfreezing."""
        groups = [list(self.decoder.parameters())]
        for enc_layer in reversed(list(self.encoder.layers)):
            groups.append(list(enc_layer.parameters()))
        groups.append(
            list(self.pos_enc.parameters()) + list(self.input_proj.parameters())
        )
        return groups

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_top_n_groups(self, n: int):
        for i, grp in enumerate(self.layer_groups()):
            for p in grp:
                p.requires_grad = i < n
