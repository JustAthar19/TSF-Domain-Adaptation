import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.TFT.gated_residual_network import TFTGRN

class TFTVariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, num_vars, hidden_dim):
        super().__init__()
        self.num_vars = num_vars

        self.var_grns = nn.ModuleList([
            TFTGRN(input_dim, hidden_dim, hidden_dim)
            for _ in range(num_vars)
        ])

        self.weight_grn = TFTGRN(input_dim * num_vars, hidden_dim, num_vars)

    def forward(self, x):
        # x: [B, T, N, D]
        # B, T, N, D = x.shape
        B, T, N, D = x.shape

        var_outputs = []
        for i in range(N):
            var_outputs.append(self.var_grns[i](x[:, :, i, :]))

        var_outputs = torch.stack(var_outputs, dim=2)  # [B, T, N, H]

        flat = x.reshape(B, T, -1)
        weights = F.softmax(self.weight_grn(flat), dim=-1).unsqueeze(-1)

        out = (weights * var_outputs).sum(dim=2)

        return out, weights
    