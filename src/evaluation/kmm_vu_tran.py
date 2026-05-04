import torch
import numpy as np

def eval_kmm_vu_tran_mae(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, input_len: int, batch_size: int, device: str) -> float:
    if X.shape[0] == 0:
        return float("nan")
    val_abs_sum = 0.0
    val_count = 0
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb_b = X[i : i + batch_size].to(device)
            yb_b = y[i : i + batch_size].to(device)
            z_past = xb_b[:, :input_len, :1]
            x_cov_past = xb_b[:, :input_len, 1:]
            x_cov_future = xb_b[:, input_len:, 1:]

            forecast, _ = model(z_past, x_cov_past, x_cov_future)
            pred = forecast.squeeze(-1)  # (B, H)

            val_abs_sum += torch.sum(torch.abs(pred - yb_b)).item()
            val_count += pred.numel()
    val_mae = val_abs_sum / max(1, val_count)
    return val_mae



def eval_model_kmm_vu_tran_metrics(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, input_len: int, device: str):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 256):
            xb = torch.from_numpy(X[i : i + 256]).float()
            xb = xb.to(device, non_blocking=(device == "cuda"))

            z_past = xb[:, :input_len, :1]
            x_cov_past = xb[:, :input_len, 1:]
            x_cov_future = xb[:, input_len:, 1:]

            forecast, _recon = model(z_past, x_cov_past, x_cov_future)
            preds.append(forecast.squeeze(-1).detach().cpu().numpy().astype(np.float32))

    pred = np.concatenate(preds, axis=0)
    y_true = y.astype(np.float32, copy=False)
    err = pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))

    kmm_metrics = {"mae": mae, "mse": mse, "rmse": rmse}
    return kmm_metrics