from src.models.kmm_vu_tran.projection import Projection

import torch.nn as nn
import torch

class TemperatureTransformer(nn.Module):
    def __init__(
        self,
        input_dim_primary=1,     # temperature
        input_dim_cov=8,         # covariates
        hidden_dim=64,
        n_heads=4,
        num_layers=2,
        forecast_horizon=7
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # 🔹 Projection layers
        self.primary_proj = Projection(input_dim_primary, hidden_dim)
        self.cov_proj = Projection(input_dim_cov, hidden_dim)

        # 🔹 Aggregation projection
        self.agg_proj = Projection(hidden_dim * 2, hidden_dim)

        # 🔹 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 🔹 Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # 🔹 Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)


    def forward(self, z_past, x_cov_past, x_cov_future):
        """
        z_past: (B, T, 1)
        x_cov_past: (B, T, d)
        x_cov_future: (B, tau, d)
        """

        B, T, _ = z_past.shape
        tau = x_cov_future.shape[1]

        # 🔹 1. Projection
        p = self.primary_proj(z_past)            # (B, T, H)
        q_past = self.cov_proj(x_cov_past)       # (B, T, H)
        q_future = self.cov_proj(x_cov_future)   # (B, tau, H)

        # 🔹 2. Concatenate + aggregate
        h = torch.cat([p, q_past], dim=-1)       # (B, T, 2H)
        h = self.agg_proj(h)                     # (B, T, H)

        # 🔹 3. Encoder
        memory = self.encoder(h)                 # (B, T, H)

        # 🔹 4. Decoder (forecast)
        tgt = q_future                           # (B, tau, H)
        out_dec = self.decoder(tgt, memory)      # (B, tau, H)

        forecast = self.output_layer(out_dec)    # (B, tau, 1)

        # 🔹 5. Reconstruction (important!)
        recon_dec = self.decoder(q_past, memory)  # (B, T, H)
        recon = self.output_layer(recon_dec)      # (B, T, 1)

        return forecast, recon