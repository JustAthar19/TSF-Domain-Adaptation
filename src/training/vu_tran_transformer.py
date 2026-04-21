import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from src.evaluation.kmm_vu_tran import eval_kmm_vu_tran_mae, eval_model_kmm_vu_tran_metrics



def train_transformer_non_da(
    model: nn.Module,
    X_source: pd.DataFrame, 
    y_source: pd.DataFrame,          # Source domain (Java)
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    config:dict,     # Target validation (Papua)
    epochs: int=100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device="cuda"
):
    """
    Train Transformer with KMM-based domain adaptation.

    X_source: (N, seq_len, features)
    y_source: (N, horizon)
    beta:     (N,) KMM weights
    """

    model.to(device)

    # Convert to tensors
    X_source = torch.from_numpy(X_source).float()
    y_source = torch.from_numpy(y_source).float()
    # beta = torch.from_numpy(beta).float()

    # If provided, convert validation arrays once (we will run the same
    # TemperatureTransformer slicing logic during validation).
    if X_val is not None and y_val is not None:
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).float()

    train_ds = TensorDataset(X_source, y_source)
    train_loader = DataLoader(train_ds, batch_size=256,  num_workers=0)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    best_val_mae, best_state = float('inf'), None
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            z_past = xb[:, :config['input_len'], :1]
            x_cov_past = xb[:, :config['input_len'], 1:]
            x_cov_future = xb[:, config['input_len']:, 1:]

            forecast, _ = model(z_past, x_cov_past, x_cov_future)
            preds = forecast.squeeze(-1)

        
            loss = criterion(preds, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_mae = eval_kmm_vu_tran_mae(model, X_val, y_val, config,batch_size)

        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val MAE: {val_mae:.4f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state) 
    return model.to(device), best_val_mae



def train_transformer_da(
    model: nn.Module,
    X_source: pd.DataFrame, 
    y_source: pd.DataFrame,          # Source domain (Java)
    beta: np.ndarray,                        # KMM weights (same length as X_source)
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    config: dict,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device="cuda"
):
    """
    Train Transformer with KMM-based domain adaptation.

    X_source: (N, seq_len, features)
    y_source: (N, horizon)
    beta:     (N,) KMM weights
    """

    model.to(device)

    # Convert to tensors
    X_source = torch.from_numpy(X_source).float()
    y_source = torch.from_numpy(y_source).float()
    beta = torch.from_numpy(beta).float()

    # If provided, convert validation arrays once (we will run the same
    # TemperatureTransformer slicing logic during validation).
    if X_val is not None and y_val is not None:
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).float()

    # Dataset includes weights
    train_ds = TensorDataset(X_source, y_source, beta)
    train_loader = DataLoader(train_ds, batch_size=256,  num_workers=0)
    # train_loader = DataLoader(train_ds, batch_size=256,  **_dataloader_kwargs(shuffle=True, drop_last=False))

# 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Important: we need per-sample loss → no reduction
    criterion = nn.MSELoss(reduction="none")
    best_val_mae, best_state = float('inf'), None
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()

            # TemperatureTransformer expects (z_past, x_cov_past, x_cov_future)
            # Here, xb contains stacked past+future features:
            #   xb[:, :INPUT_LEN, :1]   -> z_past
            #   xb[:, :INPUT_LEN, 1:]   -> x_cov_past
            #   xb[:, INPUT_LEN:, 1:]   -> x_cov_future
            z_past = xb[:, :config['input_len'], :1]
            x_cov_past = xb[:, :config['input_len'], 1:]
            x_cov_future = xb[:, config['input_len']:, 1:]

            forecast, _recon = model(z_past, x_cov_past, x_cov_future)
            preds = forecast.squeeze(-1)  # (B, H)

            # Step 1: per-sample loss
            loss_per_sample = criterion(preds, yb)   # (B, H)

            # Step 2: reduce over horizon → (B,)
            loss_per_sample = loss_per_sample.mean(dim=1)

            # Step 3: apply KMM weights
            weighted_loss = (loss_per_sample * wb).mean()

            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()

        avg_loss = total_loss / len(train_loader)
        
        val_mae = eval_kmm_vu_tran_mae(model, X_val, y_val, config, batch_size)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss (Target): {val_mae:.4f}")    
        if best_val_mae < val_mae:
            best_val_mae = val_mae
            best_state = {k : v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device), best_val_mae