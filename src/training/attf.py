import torch.nn as nn
import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

from src.evaluation.attf import attf_eval_model_mae


def train_attf_earlystop_target_mae(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tgt_val: np.ndarray,
    y_tgt_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    lambda_recon: float = 1e-4,  # 🔥 important
):
    """
    ATTF Training:
    - Forecast loss (MSE)
    - Reconstruction loss (MSE)
    - Early stop on target MAE
    """
    
    model = model.to(device).float()

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_mae, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0

        for xb, yb in train_loader:
            opt.zero_grad()
            y_pred, x_recon = model(xb)
    
            loss_forecast = nn.functional.mse_loss(y_pred, yb)
            loss_recon = ((x_recon - xb) ** 2).mean(dim=(1,2)).mean()
            loss = loss_forecast + lambda_recon * loss_recon

            loss.backward()
            opt.step()

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= max(1, X_train.shape[0])

        val_mae = attf_eval_model_mae(model, X_tgt_val, y_tgt_val, batch_size, device)
        print(f"Loss Foreacst: {loss_forecast.item()} | Loss Recon: {loss_recon.item()}")
        print(
            f"  Epoch {ep+1}/{epochs}  "
            f"Train Loss: {tr_loss:.6f}  "
            f"Target Val MAE: {val_mae:.4f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.to(device), best_mae