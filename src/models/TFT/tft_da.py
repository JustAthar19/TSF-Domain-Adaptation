import torch
import torch.nn as nn

from src.models.TFT.tft_backbone import TFT_Backbone
from src.models.TFT.domain_classifier import TFTDomainClassifier
from src.models.TFT.gradient_reversal_layer import TFTGRL

class TFT_DA_Model(nn.Module):
    def __init__(
        self,
        input_dim,
        num_vars,
        hidden_dim=64,
        horizon=7
    ):
        super().__init__()

        self.horizon = horizon

        self.backbone = TFT_Backbone(
            input_dim=input_dim,
            num_vars=num_vars,
            hidden_dim=hidden_dim
        )

        # Forecast head (multi-horizon)
        self.forecast_head = nn.Linear(hidden_dim, horizon)

        # Domain classifier
        self.domain_classifier = TFTDomainClassifier(hidden_dim)

    def forward(self, x, lambda_grl=1.0):
        features, var_w, attn_w = self.backbone(x)

        # Pooling (important!)
        feat = features.mean(dim=1)  # [B, H]

        # Forecast
        forecast = self.forecast_head(feat)  # [B, horizon]

        # Domain prediction with GRL
        feat_grl = TFTGRL.apply(feat, lambda_grl)
        domain_pred = self.domain_classifier(feat_grl)

        return forecast, domain_pred, var_w, attn_w