import torch
import torch.nn as nn

class PrivateDecoder(nn.Module):
    """
    Position-wise MLP: context o_t → scalar prediction z_hat_t  (Eq.6 / §4.1)

    Args:
        d_model    : input dimension
        output_dim : prediction dimension (1 for univariate)
        n_mlp_layers: depth
    """

    def __init__(self, d_model: int = 64, output_dim: int = 1, n_mlp_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(d_model, d_model), nn.ReLU()]
        for _ in range(n_mlp_layers - 1):
            layers += [nn.Linear(d_model, d_model), nn.ReLU()]
        layers.append(nn.Linear(d_model, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """o : [B, T_or_τ, d_model] → [B, T_or_τ, output_dim]"""
        return self.mlp(o)
