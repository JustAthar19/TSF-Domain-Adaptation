import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from src.evaluation.tft import eval_tft_mae_da, eval_tft_mae_non_da

from src.models.TFT.gradient_reversal_layer import dann_lambda_schedule

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def train_tft_target(
    model: nn.Module,
    X_tgt: np.ndarray,
    y_tgt: np.ndarray,
    X_tgt_val: np.ndarray,
    y_tgt_val: np.ndarray,
    config: dict,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    
):
    from torch.utils.data import TensorDataset, DataLoader

    if X_tgt.shape[0] == 0:
        return None, float("nan")

    # ✅ Single dataset (target only)
    tgt_ds = TensorDataset(
        torch.from_numpy(X_tgt).to(config['device']),
        torch.from_numpy(y_tgt).to(config['device'])
    )

    tgt_loader = DataLoader(
        tgt_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    model = model.to(config['device']).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_mae, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        model.train()

        tr_loss = 0.0
        n_steps = len(tgt_loader)

        for xb, yb in tgt_loader:
            opt.zero_grad()

            # 🔥 Forward (no GRL, no domain output)
            yhat, _, _ = model(xb)

            loss = F.mse_loss(yhat, yb)

            loss.backward()
            opt.step()

            tr_loss += loss.item()

        tr_loss /= max(1, n_steps)

        # ✅ Validation on target
        val_mae = eval_tft_mae_non_da(model, X_tgt_val, y_tgt_val, config, batch_size=256)

        print(f"Epoch {ep+1}/{epochs} | Train Loss: {tr_loss:.6f} | Val MAE: {val_mae:.4f}")

        # ✅ Early stopping (you had it commented before)
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_mae

# def train_tft(
#     model: nn.Module,
#     X_train: pd.DataFrame,
#     y_train: pd.DataFrame,
#     X_val: pd.DataFrame,
#     y_val: pd.DataFrame,
#     config: dict,
#     epochs=100,
#     batch_size=256,
#     lr=1e-3,
#     patience=5,
#     device='cuda'
# ):
#     model = model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     best_val = float("inf")

#     for epoch in range(epochs):
#         model.train()
#         perm = np.random.permutation(len(X_train))

#         for i in range(0, len(X_train), batch_size):
#             idx = perm[i:i+batch_size]

#             xb = torch.from_numpy(X_train[idx]).float().to(device)
#             yb = torch.from_numpy(y_train[idx]).float().to(device)

#             pred = model(xb)

#             loss = criterion(pred, yb)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         val_mae = eval_model_mae(model, X_val, y_val, config, batch_size=256)

#         print(f"Epoch {epoch+1} | Val MAE: {val_mae:.4f}")

#         if val_mae < best_val:
#             best_val = val_mae
#             best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#         # else:
#         #     patience_counter += 1

#         # if patience_counter >= patience:
#         #     break

#     model.load_state_dict(best_state)
#     return model.to(device), best_val


def train_tft_da(
    model: nn.Module,
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_tgt: np.ndarray,
    y_tgt: np.ndarray,
    X_tgt_val: np.ndarray,
    y_tgt_val: np.ndarray,
    config: dict,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    device='cuda',
    use_target_task_loss: bool = False,
):
    from torch.utils.data import TensorDataset, DataLoader

    if X_src.shape[0] == 0 or X_tgt.shape[0] == 0:
        return None, float("nan")

    src_bs = max(1, batch_size // 2)
    tgt_bs = max(1, batch_size // 2)

    src_ds = TensorDataset(torch.from_numpy(X_src).to(config['device']), torch.from_numpy(y_src).to(config['device']))
    tgt_ds = TensorDataset(torch.from_numpy(X_tgt).to(config['device']), torch.from_numpy(y_tgt).to(config['device']))

    src_loader = DataLoader(src_ds, batch_size=src_bs, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=tgt_bs, shuffle=True, drop_last=True)

    model = model.to(device).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps = epochs * min(len(src_loader), len(tgt_loader))
    global_step = 0

    best_mae, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        model.train()

        tr_task, tr_dom = 0.0, 0.0
        n_steps = min(len(src_loader), len(tgt_loader))

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        for _ in range(n_steps):
            xs, ys = next(src_iter)
            xt, yt = next(tgt_iter)

            xb = torch.cat([xs, xt], dim=0)

            dom_y = torch.cat([
                torch.zeros(xs.size(0)),
                torch.ones(xt.size(0))
            ]).long().to(device)

            p = global_step / max(1, total_steps)
            lambd = dann_lambda_schedule(p)
            global_step += 1

            opt.zero_grad()

            yhat, dom_pred, _, _ = model(xb, lambda_grl=lambd)

            # Task loss
            task_loss = F.mse_loss(yhat[:xs.size(0)], ys)

            if use_target_task_loss:
                task_loss += F.mse_loss(yhat[xs.size(0):], yt)

            # Domain loss
            dom_loss = F.cross_entropy(dom_pred, dom_y)

            loss = task_loss + lambd * dom_loss
            loss.backward()
            opt.step()

            tr_task += task_loss.item()
            tr_dom += dom_loss.item()


        val_mae = eval_tft_mae_da(model, X_tgt_val, y_tgt_val, config, batch_size=256)

        tr_task /= max(1, n_steps)
        tr_dom /= max(1, n_steps)

        print(f"Epoch {ep+1}/{epochs} | Task: {tr_task:.6f} | Domain: {tr_dom:.6f} | Val MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        #     wait = 0
        # else:
        #     wait += 1
        #     if wait >= patience:
        #         print(f"Early stopping at epoch {ep+1}")
        #         break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_mae