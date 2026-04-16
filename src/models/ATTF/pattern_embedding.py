from torch import nn

class PatternEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        p = self.conv(x)       # (B, H, T)
        return p.transpose(1, 2)  # (B, T, H)