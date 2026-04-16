import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_steps):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_steps)
        )

    def forward(self, x):
        # x: (B, T, H)
        x = x[:, -1, :]  # last timestep
        return self.fc(x)  # (B, 7)