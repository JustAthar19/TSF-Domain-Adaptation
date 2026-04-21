import torch.nn as nn

from src.models.TFT.variable_selection_network import TFTVariableSelectionNetwork
from src.models.TFT.gated_residual_network import TFTGRN

class TFT_Backbone(nn.Module):
    def __init__(
        self,
        input_dim,
        num_vars,
        hidden_dim=64,
        lstm_layers=1,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()

        self.vsn = TFTVariableSelectionNetwork(input_dim, num_vars, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.post_grn = TFTGRN(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, T, N, D]

        x, var_weights = self.vsn(x)

        lstm_out, _ = self.lstm(x)

        attn_out, attn_weights = self.attn(lstm_out, lstm_out, lstm_out)

        features = self.post_grn(attn_out)

        return features, var_weights, attn_weights