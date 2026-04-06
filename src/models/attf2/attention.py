import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    """
    Shared attention module (Section 4.1).

    Computes Q, K from pattern embeddings P via a position-wise MLP,
    then produces context vectors O via scaled-dot-product attention
    on value embeddings V.

    Two operational modes:
      - interpolation (t ≤ T) : reconstruct input  – each position attends to
                                 all *other* positions (leave-one-out mask).
      - extrapolation (t > T) : forecast future     – each future query attends
                                 to a causal window of historical keys/values.

    Args:
        d_model    : query/key/value dimension
        n_mlp_layers: depth of the Q/K projection MLP
        kernel_size: convolution kernel size s used in the encoder
                     (needed to derive the neighbourhood offset s̄ = ⌈(s-1)/2⌉)
    """

    def __init__(self, d_model: int = 64, n_mlp_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        # s̄ = ⌈(s-1)/2⌉  used in extrapolation (eq. after Eq.6 in paper)
        self.s_bar = math.ceil((kernel_size - 1) / 2)

        # Position-wise MLP  P → (Q, K),  output dim = 2 * d_model
        qk_layers = [nn.Linear(d_model, d_model), nn.ReLU()]
        for _ in range(n_mlp_layers - 1):
            qk_layers += [nn.Linear(d_model, d_model), nn.ReLU()]
        qk_layers.append(nn.Linear(d_model, 2 * d_model))
        self.qk_mlp = nn.Sequential(*qk_layers)

        # Output projection MLP  (Eq.6)
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def _scaled_dot(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Scaled dot-product: q [B, Nq, d], k [B, Nk, d] → scores [B, Nq, Nk]"""
        return torch.bmm(q, k.transpose(1, 2)) / self.scale

    # ------------------------------------------------------------------ #
    #  Interpolation: attend to all positions except self (leave-one-out)
    # ------------------------------------------------------------------ #
    def _interpolate(self, Q, K, V):
        """
        Q, K, V : [B, T, d_model]
        Returns  : [B, T, d_model]  – reconstructed context
        """
        B, T, _ = Q.shape
        scores = self._scaled_dot(Q, K)          # [B, T, T]

        # Mask out diagonal so position t never attends to itself
        mask = torch.eye(T, device=Q.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)       # [B, T, T]
        context = torch.bmm(weights, V)           # [B, T, d_model]
        return context

    # ------------------------------------------------------------------ #
    #  Extrapolation: future query attends to a causal window
    # ------------------------------------------------------------------ #
    def _extrapolate_one(self, q_t: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          s: int, T: int):
        """
        Generate one future step using the extrapolation rule (Section 4.1).

        q_t  : [B, d_model]  query for the future step  (= q_{T - s̄})
        K    : [B, T, d_model]
        V    : [B, T, d_model]

        Neighbourhood of keys: N(T+1) = {s, …, T - s̄ - 1}   (0-indexed here)
        Paired value index   : µ(t') = t' + s̄ + 1
        """
        s_bar = self.s_bar
        # key indices: [s-1 … T - s_bar - 2]  (0-indexed; paper uses 1-indexed)
        key_start = max(s - 1, 0)
        key_end   = max(T - s_bar - 1, key_start + 1)   # at least 1 key
        k_slice   = K[:, key_start:key_end, :]           # [B, Nk, d]

        # Corresponding value indices: t' + s_bar + 1
        val_indices = [min(i + s_bar + 1, T - 1)
                       for i in range(key_start, key_end)]
        v_slice = V[:, val_indices, :]                   # [B, Nk, d]

        scores  = torch.bmm(q_t.unsqueeze(1), k_slice.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)              # [B, 1, Nk]
        context = torch.bmm(weights, v_slice).squeeze(1) # [B, d_model]
        return context

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, P: torch.Tensor, V: torch.Tensor,
                future_steps: int = 0, kernel_size: int = None):
        """
        P            : [B, T, d_model]  pattern embeddings from encoder
        V            : [B, T, d_model]  value embeddings from encoder
        future_steps : τ  number of autoregressive steps to forecast
        kernel_size  : override the module-level kernel_size for s̄ computation

        Returns:
            O_recon  [B, T, d_model]  – interpolation context (reconstruction)
            O_fore   [B, τ, d_model]  – extrapolation context (forecast)
                                         empty tensor when future_steps == 0
        """
        B, T, _ = P.shape
        s = kernel_size or (2 * self.s_bar + 1)   # recover s from s̄

        # Project P → Q, K
        qk = self.qk_mlp(P)                       # [B, T, 2*d_model]
        Q, K = qk.chunk(2, dim=-1)                # each [B, T, d_model]

        # --- Interpolation (reconstruction) ---
        O_recon_raw = self._interpolate(Q, K, V)
        O_recon = self.out_mlp(O_recon_raw)        # [B, T, d_model]

        # --- Extrapolation (forecast) ---
        if future_steps == 0:
            O_fore = torch.zeros(B, 0, self.d_model, device=P.device)
            return O_recon, O_fore, Q, K

        # The extrapolation query for step T+1 is q_{T - s̄}  (paper eq.)
        q_extra = Q[:, T - self.s_bar - 1, :]     # [B, d_model]

        fore_contexts = []
        # We build the context for each future step in an autoregressive
        # fashion; since AttF doesn't update K/V with future values in the
        # attention (it feeds predictions back through the encoder), we
        # keep K and V fixed from the historical window for the attention.
        for _ in range(future_steps):
            ctx = self._extrapolate_one(q_extra, K, V, s, T)
            ctx = self.out_mlp(ctx)                # [B, d_model]
            fore_contexts.append(ctx.unsqueeze(1)) # [B, 1, d_model]

        O_fore = torch.cat(fore_contexts, dim=1)   # [B, τ, d_model]
        return O_recon, O_fore, Q, K
