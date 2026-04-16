import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from src.evaluation.supervised import eval_model_mae


def train_vanilla_earlystop_target_mae(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tgt_val: np.ndarray,
    y_tgt_val: np.ndarray,
    config: dict,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    
):
    """Train with MSE; early stop on target validation MAE."""
    
    if X_train.shape[0] == 0:
        return model.to(config['device']), float("nan")

    model = model.to(config['device']).float()
    train_ds = TensorDataset(torch.from_numpy(X_train).to(config['device']), torch.from_numpy(y_train).to(config['device']))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_mae, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        tr_mse = 0.0
        for xb, yb in train_loader:
            opt.zero_grad() # resetting the gradient from the previous batch. But why?? -> so on the next training iteration the gradient is get set to 0
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward() # back propagation
            opt.step() # update weights
            tr_mse += loss.item() * xb.size(0) # *** track training loss (accumulates loss accross batches)
        tr_mse /= max(1, X_train.shape[0]) # *** mean training loss per sample

        val_mae = eval_model_mae(model, X_tgt_val, y_tgt_val, config, batch_size=256)
        print(f"  Epoch {ep+1}/{epochs}  Train MSE: {tr_mse:.6f}  Target Val MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        # else:
        #     wait += 1
        #     if wait >= patience:
        #         print(f"  Early stopping at epoch {ep+1}")
        #         break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(config['device']), best_mae
