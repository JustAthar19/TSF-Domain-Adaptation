import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def eval_tft_mae_da(model: nn.Module, X: pd.DataFrame, y: pd.DataFrame, config: dict, batch_size=256):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(config['device'])
            yhat, _, _, _ = model(xb, lambda_grl=0.0)
            preds.append(yhat.cpu())

    preds = torch.cat(preds, dim=0)
    return torch.mean(torch.abs(preds - torch.from_numpy(y))).item()

def tft_eval_model_metrics_da(model: nn.Module, X: pd.DataFrame, y: pd.DataFrame, config: dict, batch_size=256):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(config['device'])
            yhat, _, _, _ = model(xb, lambda_grl=0.0)
            preds.append(yhat.cpu())

    preds = torch.cat(preds, dim=0)
    y = torch.from_numpy(y)

    mae = torch.mean(torch.abs(preds - y)).item()
    mse = torch.mean((preds - y) ** 2).item()
    rmse = mse ** 0.5

    return {"mae": mae, "mse": mse, "rmse": rmse}


def eval_tft_mae_non_da(model: nn.Module, X: pd.DataFrame, y: pd.DataFrame, config: dict, batch_size=256):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(config['device'])
            yhat, _, _ = model(xb)
            preds.append(yhat.cpu())

    preds = torch.cat(preds, dim=0)
    return torch.mean(torch.abs(preds - torch.from_numpy(y))).item()

def tft_eval_model_metrics_non_da(model: torch.nn.Module,X: np.ndarray, y: np.ndarray, config: dict, batch_size: int = 256
):
    """
    Evaluate model on test data and return MAE, MSE, RMSE
    """

    if X.shape[0] == 0:
        return {"mae": np.nan, "mse": np.nan, "rmse": np.nan}

    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float()
            xb = xb.to(config['device'], non_blocking=(config['device'] == "cuda"))

            out, _ ,_ = model(xb) 
            preds.append(out.cpu().numpy())

    pred = np.concatenate(preds, axis=0).astype(np.float32)

    mae = np.mean(np.abs(pred - y))
    mse = np.mean((pred - y) ** 2)
    rmse = np.sqrt(mse)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse)
    }