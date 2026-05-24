import numpy as np
import torch
import torch.nn as nn


def osft_eval_model_metrics(
    model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> dict:
    if X.shape[0] == 0:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            preds.append(model(xb).detach().cpu().numpy().astype(np.float32))

    pred = np.concatenate(preds, axis=0).astype(np.float32)
    y = y.astype(np.float32, copy=False)
    err = pred - y
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}
