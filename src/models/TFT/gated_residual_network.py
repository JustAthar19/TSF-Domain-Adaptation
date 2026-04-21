import torch.nn as nn

class TFTGRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        output_dim = output_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Linear(output_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = x

        x = self.elu(self.fc1(x))
        x = self.dropout(self.fc2(x))

        gate = self.sigmoid(self.gate(x))
        x = gate * x

        if self.skip is not None:
            residual = self.skip(residual)

        return self.layer_norm(x + residual)