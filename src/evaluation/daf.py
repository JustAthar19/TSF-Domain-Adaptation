import torch 
import numpy as np


def daf_eval_model_metrics(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: str):
    """
    Compute MAE/MSE/RMSE on window targets (y shape: [N, H]).
    Metrics are computed over all elements (N*H).
    Returns dict: {"mae": float, "mse": float, "rmse": float}.
    """
    if X.shape[0] == 0:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    model.eval() # switch model into evaluation mode
    preds = []
    with torch.no_grad(): # disable gradients
        for i in range(0, X.shape[0], batch_size):
            # Extract batch 
            xb = torch.from_numpy(X[i : i + batch_size])
            xb = xb.to(device, non_blocking=(device == "cuda"))
            pred, _, _ = model(xb)            
            preds.append(pred.detach().cpu().numpy().astype(np.float32))
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    y = y.astype(np.float32, copy=False)
    err = (pred - y).astype(np.float32)
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}



def daf_eval_model_mae(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> float:
    if X.shape[0] == 0:
        return float("nan")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size])
            xb = xb.to(device, non_blocking=(device == "cuda"))
            pred, _, _ = model(xb)
            # pred = pred[:, -1, :]
            # pred = pred[:, :y.shape[1], 0]
            preds.append(pred.detach().cpu().numpy().astype(np.float32))
    pred = np.concatenate(preds, axis=0)
    return float(np.mean(np.abs(pred - y)))

