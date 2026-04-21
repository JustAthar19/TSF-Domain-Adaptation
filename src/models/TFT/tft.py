import torch.nn as nn
from src.models.TFT.tft_backbone import TFT_Backbone

# class ClimateTFT(nn.Module):
#     def __init__(
#         self,
#         config: dict,
#         horizon=7,
#         d_model=32,
#         nhead=4,
#         num_layers=2,
#         dropout=0.1,
#     ):
#         super().__init__()

#         self.idx_temp = [config['feature_cols'].index(c) for c in config['local_temporal_cols']]
#         self.idx_geo = [config['feature_cols'].index(c) for c in config['geo_cols']]
#         self.idx_clim = [config['feature_cols'].index(c) for c in config['climate_cols']]

#         # --- Variable Selection ---f
#         self.vsn = TFTVariableSelectionNetwork(len(self.idx_temp), d_model, 64)

#         # --- Transformer (same as before but cleaner) ---
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # --- Static encoders (GRN instead of Linear) ---
#         self.geo_grn = TFTGRN(3, 32, 16)
#         self.clim_grn = TFTGRN(2, 32, 16)

#         # --- Temporal GRN ---
#         self.temporal_grn = TFTGRN(d_model, 64, d_model)

#         # --- Fusion GRN ---
#         self.fusion_grn = TFTGRN(d_model + 16 + 16, 64, 64)

#         # --- Output ---
#         self.head = nn.Linear(64, horizon)

#     def forward(self, x):
#         # Split inputs
#         xt = x[:, :, self.idx_temp]
#         xg = x[:, 0, self.idx_geo]
#         xc = x[:, 0, self.idx_clim]

#         # --- Variable selection ---
#         xt = self.vsn(xt)  # (B, T, d_model)

#         # --- Transformer ---
#         h = self.transformer(xt)

#         # Pooling (same as before)
#         h = h.mean(dim=1)

#         # --- Temporal refinement ---
#         h = self.temporal_grn(h)

#         # --- Static features ---
#         g = self.geo_grn(xg)
#         c = self.clim_grn(xc)

#         # --- Fusion ---
#         rep = torch.cat([h, g, c], dim=1)
#         rep = self.fusion_grn(rep)

#         return self.head(rep)


class TFT_Target_Model(nn.Module):
    def __init__(self, input_dim, num_vars, hidden_dim=64, horizon=7):
        super().__init__()

        self.horizon = horizon

        self.backbone = TFT_Backbone(
            input_dim=input_dim,
            num_vars=num_vars,
            hidden_dim=hidden_dim
        )

        # Multi-horizon forecast head
        self.forecast_head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # features: [B, T, H]
        features, var_w, attn_w = self.backbone(x)

        # 🔥 IMPORTANT DESIGN CHOICE
        # Option 1: Mean pooling (more stable)
        feat = features.mean(dim=1)

        # Option 2 (alternative): last timestep
        # feat = features[:, -1, :]

        forecast = self.forecast_head(feat)  # [B, horizon]

        return forecast, var_w, attn_w