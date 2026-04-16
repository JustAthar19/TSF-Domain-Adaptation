import torch
import torch.nn as nn

class DomainClassifier(nn.Module):
    """MLP: Linear(in→32) ReLU Linear(32→1) Sigmoid."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # (B,)



