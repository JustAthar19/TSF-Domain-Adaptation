import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def run_station_arima(lid, train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict,horizon: int = 7  ):
    """
    Fit an ARIMA model per station and compute rolling forecasts
    over the test horizon. This module is intentionally torch-free
    so it can be used safely from multiprocessing workers.
    """

    train_ser = (
        train_df[train_df["location_id"] == lid]
        .sort_values("time")[config["target_col"]]
    )
    test_grp = (
        test_df[test_df["location_id"] == lid]
        .sort_values("time")
    )

    if len(train_ser) < 30 or len(test_grp) < horizon:
        return None

    train_ser = train_ser.values.astype(np.float64)
    test_vals = test_grp[config["target_col"]].values.astype(np.float32)
    # test_vals = test_grp["max_temperature"].values.astype(np.float32)

    order = selet_arima_order(train_ser)

    preds = []

    for i in range(0, len(test_vals) - horizon + 1):
        history = (
            np.concatenate([train_ser, test_vals[:i]])
            if i > 0
            else train_ser
        )

        try:
            model = ARIMA(history, order=order)
            fitted = model.fit(method_kwargs={"maxiter": 200})
            forecast = fitted.forecast(steps=horizon)
            preds.append(forecast)
        except Exception:
            break

    if not preds:
        return None

    preds = np.array(preds)

    y_true = np.array(
        [test_vals[i : i + horizon] for i in range(len(preds))]
    )

    mae = np.mean(np.abs(y_true - preds))
    rmse = np.sqrt(np.mean((y_true - preds) ** 2))

    return mae, rmse


def arima_baseline_rolling(test_df, train_df, config, horizon: int = 7):
    """
    Compute ARIMA rolling baseline across all stations in the test
    DataFrame using joblib-based parallelization. Only this module
    is imported inside worker processes, avoiding any torch / CUDA
    initialization overhead.
    """
    test_df = test_df.sort_values(["location_id", "time"])
    locations = test_df["location_id"].unique()

    results = Parallel(n_jobs=-1)(
        delayed(run_station_arima)(lid, train_df, test_df, config, horizon)
        for lid in locations
    )

    all_mae, all_rmse = [], []

    for r in results:
        if r is None:
            continue
        mae, rmse = r
        all_mae.append(mae)
        all_rmse.append(rmse)

    if not all_mae:
        return float("nan"), float("nan")

    mae, rmse = float(np.mean(all_mae)), float(np.mean(all_rmse))
    mse = float(rmse ** 2)

    return {"mae": mae, "mse": mse, "rmse": rmse}


def selet_arima_order(series):
    """
    Simple (p, d, q) search for ARIMA order based on AIC.
    """
    series = np.array(series, dtype=np.float64)

    # determine differencing
    d = 0
    try:
        adf = adfuller(series)
        if adf[1] > 0.05:
            d = 1
    except Exception:
        pass

    best_aic = np.inf
    best_order = (1, d, 0)

    for p in range(2):
        for q in range(2):
            try:
                model = ARIMA(series, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except Exception:
                continue
    return best_order

