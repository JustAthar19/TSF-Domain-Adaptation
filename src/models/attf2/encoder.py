import torch
import torch.nn as nn
import torch.nn.functional as F


class PrivateEncoder(nn.Module):
    """
    Produces:
      P  [B, T, d_model]  – multi-scale pattern embedding (concat of M convs)
      V  [B, T, d_model]  – value embedding (position-wise MLP on raw input)

    Args:
        input_dim   : number of input features per time step (1 for univariate)
        d_model     : output dimension for both P and V
        kernel_sizes: tuple of conv kernel sizes, one conv branch per entry (M branches)
        n_mlp_layers: depth of the value MLP
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        kernel_sizes: tuple = (3, 5),
        n_mlp_layers: int = 2,
    ):
        super().__init__()
        self.M = len(kernel_sizes)
        self.d_model = d_model

        # --- Value MLP  (position-wise, shared across time) ---
        value_layers = [nn.Linear(input_dim, d_model), nn.ReLU()]
        for _ in range(n_mlp_layers - 1):
            value_layers += [nn.Linear(d_model, d_model), nn.ReLU()]
        self.value_mlp = nn.Sequential(*value_layers)

        # --- Pattern convolutions  (M independent temporal convs) ---
        # Each conv maps input_dim → (d_model // M) channels so that
        # concatenation across M branches gives exactly d_model channels.
        branch_dim = d_model // self.M
        self.pattern_convs = nn.ModuleList()
        for ks in kernel_sizes:
            pad = ks // 2           # "same" padding to preserve length
            self.pattern_convs.append(
                nn.Sequential(
                    # Conv1d expects (B, C_in, T)
                    nn.Conv1d(input_dim, branch_dim, kernel_size=ks, padding=pad),
                    nn.ReLU(),
                )
            )
        # If d_model isn't evenly divisible by M, a projection fixes the dim
        conv_out_dim = branch_dim * self.M
        self.pattern_proj = (
            nn.Linear(conv_out_dim, d_model)
            if conv_out_dim != d_model
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        """
        x : [B, T, input_dim]
        returns P [B, T, d_model], V [B, T, d_model]
        """
        # Value embedding
        V = self.value_mlp(x)                           # [B, T, d_model]

        # Pattern embedding: run each conv branch on (B, input_dim, T)
        x_t = x.permute(0, 2, 1)                        # [B, input_dim, T]
        branches = [conv(x_t).permute(0, 2, 1)          # [B, T, branch_dim]
                    for conv in self.pattern_convs]
        P = torch.cat(branches, dim=-1)                  # [B, T, conv_out_dim]
        P = self.pattern_proj(P)                         # [B, T, d_model]

        return P, V