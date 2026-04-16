import torch.nn as nn
import torch

class VanillaTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=32, nhead=4, num_layers=2, dropout=0.1, horizon=7):
        """
        Model Parameter:
          - input_dim = 9. Number of feature
          - d_model = 32. Internal Feature Size of transformer. 9 Features -> projected to 32 features
          - nhead = 4. Number of Attention Heads
          - num_layers = 2.  Stack 2 Transformer Layers. Each Layer has (self-attention + Feedforward Network)
          - Horizon = 7. Predict 7 days ahead
        """

        super().__init__()
        ## input Projection. Transform from 9 features to 32 Features
        self.input_proj = nn.Linear(input_dim, d_model)
        # positional encoding
        self.positional_encoding = nn.Parameter(torch.rand(1, 500, d_model))
        # transformer decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (B,T,F)
        x = self.input_proj(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)
