import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import os

from src.models.Vanilla.vanilla import VanillaTransformer
from src.evaluation.kmm_vu_tran import eval_kmm_vu_tran_mae

def train_vanilla_transformer_weighted(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    config: dict,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
):
    """
    Train Transformer with weighted MSE: mean(weights * (pred - target)^2).
    Returns (model, best_val_loss).
    """
    
    w = torch.from_numpy(np.asarray(weights, dtype=np.float32)).float()
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), w[: len(X_train)].clone())
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = VanillaTransformer(
        input_dim=X_train.shape[2], d_model=32, nhead=4, num_layers=2,
        dropout=0.1, horizon=config['horizon'],
    ).to(config['device'])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, wait = float("inf"), None, 0
    
    USE_AMP = (config["device"] =='cuda' and os.environ.get("TSF_USE_AMP", "1") != 0)
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb, wb in train_loader:
            xb = xb.to(config["device"], non_blocking=(config["device"] == "cuda"))
            yb = yb.to(config["device"], non_blocking=(config["device"] == "cuda"))
            wb = wb.to(config["device"], non_blocking=(config["device"] == "cuda")).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=config["device"], dtype=torch.float16, enabled=USE_AMP):
                pred = model(xb)
                sq = (pred - yb) ** 2
                loss = (wb.unsqueeze(1) * sq).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        if len(X_val) > 0:
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(config["device"], non_blocking=(config["device"] == "cuda"))
                    yb = yb.to(config["device"], non_blocking=(config["device"] == "cuda"))
                    with torch.autocast(device_type=config["device"], dtype=torch.float16, enabled=USE_AMP):
                        pred = model(xb)
                        val_loss += nn.functional.mse_loss(pred, yb).item() * xb.size(0)
            val_loss /= len(X_val)
        else:
            val_loss = train_loss

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  KMM Epoch {ep+1}/{epochs}  Train loss: {train_loss:.6f}  Val loss: {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        #     wait = 0
        # else:
        #     wait += 1
        #     if wait >= patience:
        #         break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(config["device"]), float(best_val)



def train_transformer_weighted_kmm(
    model,
    X_source,
    y_source,          # Source domain (Java)
    beta,                        # KMM weights (same length as X_source)
    config: dict,
    X_val=None, 
    y_val=None,      # Target validation (Papua)
    epochs=100,
    batch_size=256,
    lr=1e-3,
):
    """
    Train Transformer with KMM-based domain adaptation.

    X_source: (N, seq_len, features)
    y_source: (N, horizon)
    beta:     (N,) KMM weights
    """

    model.to(config['device'])

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
            xb = xb.to(config['device'])
            yb = yb.to(config['device'])
            wb = wb.to(config['device'])

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

        val_abs_sum = 0.0
        val_count = 0
        model.eval()
        with torch.no_grad():
            for i in range(0, X_val.shape[0], 256):
                xb_b = X_val[i : i + 256].to(config['device'])
                yb_b = y_val[i : i + 256].to(config['device'])

                z_past = xb_b[:, :config["input_len"], :1]
                x_cov_past = xb_b[:, :config["input_len"], 1:]
                x_cov_future = xb_b[:, config["input_len"]:, 1:]

                forecast, _recon = model(z_past, x_cov_past, x_cov_future)
                pred = forecast.squeeze(-1)  # (B, H)

                val_abs_sum += torch.sum(torch.abs(pred - yb_b)).item()
                val_count += pred.numel()

        val_mae = val_abs_sum / max(1, val_count)
       
        # val_mae = eval_kmm_vu_tran_mae(model, X_val, y_val, batch_size)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss (Target): {val_mae:.4f}")    
        if best_val_mae < val_mae:
            best_val_mae = val_mae
            best_state = {k : v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(config['device']), best_val_mae