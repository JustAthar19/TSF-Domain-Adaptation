
import torch
import torch.nn as nn

from encoder import PrivateEncoder
from attention import AttentionModule
from decoder import PrivateDecoder 


class AttF(nn.Module):
    """
    Attention-based Forecaster (AttF).

    Single-domain baseline used in the DAF paper.  It is identical to the
    target-domain branch of DAF trained in isolation (no source data, no
    domain discriminator).

    Forward pass:
      1. Encode history X → P, V
      2. Interpolation attention → reconstruct X̂
      3. Extrapolation attention → forecast Ŷ  (autoregressive)

    Args:
        input_dim    : number of features per time step (1 = univariate)
        output_dim   : prediction dimensionality (usually 1)
        d_model      : latent dimension throughout the model
        kernel_sizes : tuple of conv kernel sizes for the pattern encoder
        n_enc_layers : MLP depth in encoder value branch
        n_attn_layers: MLP depth in Q/K projection
        n_dec_layers : MLP depth in decoder
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 64,
        kernel_sizes: tuple = (3, 5),
        n_enc_layers: int = 2,
        n_attn_layers: int = 2,
        n_dec_layers: int = 2,
    ):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        # Use the first (or only) kernel size to set s̄ in the attention module
        primary_ks = kernel_sizes[0] if isinstance(kernel_sizes, (list, tuple)) else kernel_sizes

        self.encoder = PrivateEncoder(
            input_dim=input_dim,
            d_model=d_model,
            kernel_sizes=kernel_sizes,
            n_mlp_layers=n_enc_layers,
        )
        self.attention = AttentionModule(
            d_model=d_model,
            n_mlp_layers=n_attn_layers,
            kernel_size=primary_ks,
        )
        self.decoder = PrivateDecoder(
            d_model=d_model,
            output_dim=output_dim,
            n_mlp_layers=n_dec_layers,
        )

    def forward(self, x: torch.Tensor, tau: int = 1):
        """
        x   : [B, T, input_dim]  historical observations
        tau : forecast horizon (number of future steps)

        Returns:
            x_hat : [B, T, output_dim]   reconstructed history
            y_hat : [B, tau, output_dim] multi-horizon forecast
            Q     : [B, T, d_model]      queries  (for domain adaptation use)
            K     : [B, T, d_model]      keys     (for domain adaptation use)
        """
        # Step 1 – encode
        P, V = self.encoder(x)                              # [B, T, d_model] each

        # Step 2 & 3 – attention (interpolation + extrapolation)
        O_recon, O_fore, Q, K = self.attention(
            P, V,
            future_steps=tau,
            kernel_size=self.kernel_sizes[0],
        )

        # Step 4 – decode
        x_hat = self.decoder(O_recon)                       # [B, T, output_dim]
        y_hat = self.decoder(O_fore) if tau > 0 \
                else torch.zeros(*x.shape[:2], self.decoder.mlp[-1].out_features,
                                 device=x.device)           # [B, tau, output_dim]

        return x_hat, y_hat, Q, K
