import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


import os

# -----------------------------------------------------------------------------
# RUNTIME / PERFORMANCE DEFAULTS (auto-tuned for CPU+GPU when available)
# -----------------------------------------------------------------------------
_CPU_CORES = os.cpu_count() or 4
# Threadripper 3960X = 24 cores (48 threads). For NumPy/torch this is a good default.
DEFAULT_CPU_THREADS = min(24, _CPU_CORES)

# Only set if user hasn't already configured these in their environment.
os.environ.setdefault("OMP_NUM_THREADS", str(DEFAULT_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(DEFAULT_CPU_THREADS))

import math
import random
from datetime import datetime, timezone

import torch

# from training.arima_baseline import arima_baseline_rolling

torch.set_default_dtype(torch.float32)
torch.set_num_threads(DEFAULT_CPU_THREADS)
torch.set_num_interop_threads(max(1, min(4, DEFAULT_CPU_THREADS // 2)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    # Ampere GPU (3090 Ti): TF32 is a big speedup for matmul/conv with minimal impact for this task.
    print(f"Training With: {DEVICE}")
    _use_tf32 = (os.environ.get("TSF_USE_TF32", "1") != "0")
    if _use_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True

USE_AMP = (DEVICE.type == "cuda" and os.environ.get("TSF_USE_AMP", "1") != "0")
USE_TORCH_COMPILE = (os.environ.get("TSF_TORCH_COMPILE", "0") == "1")


def _maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    if not USE_TORCH_COMPILE:
        return model
    try:
        return torch.compile(model)  # PyTorch 2.x
    except Exception:
        return model


def _dataloader_kwargs(shuffle: bool, drop_last: bool = False):
    """
    Sensible DataLoader defaults for this workstation.
    - Use CPU workers to overlap preprocessing/host->device transfers.
    - Use pin_memory for faster H2D copies when using CUDA.
    """
    num_workers = 0
    if DEVICE.type == "cuda":
        # Windows multiprocessing can be heavier; keep this conservative but non-zero.
        num_workers = min(8, max(2, DEFAULT_CPU_THREADS // 3))

    kwargs = dict(
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=bool(drop_last),
    )
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs

TARGET_COL = "max_temperature"
FEATURE_COLS = [
    "max_temperature",
    "latitude",
    "longitude",
    "elevation",
    "sin_doy",
    "cos_doy",
    "solar_declination",
    "dmi east",
    "nino anom 3.4",
]
INPUT_LEN = 14
HORIZON = 7
STRIDE = 2


LOCAL_TEMPORAL_COLS = ["max_temperature", "sin_doy", "cos_doy", "solar_declination"]
GEO_COLS = ["latitude", "longitude", "elevation"]
CLIMATE_COLS = ["nino anom 3.4", "dmi east"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_split(data_path: str):
    """Load merged_dataset.csv and split by time. Return (train, val, test) DataFrames."""
    df = pd.read_csv(data_path, dtype={"location_id": str})
    # Normalize column names (e.g. if CSV uses "temperature_2m_max (°C)" or "DMI EAST")
    col_map = {}
    for c in list(df.columns):
        c_lower = c.strip().lower()
        if "temperature" in c_lower and "max" in c_lower and c != "max_temperature":
            col_map[c] = "max_temperature"
        if "dmi" in c_lower and "east" in c_lower and c != "dmi east":
            col_map[c] = "dmi east"
        if "nino" in c_lower and "3.4" in c_lower and c != "nino anom 3.4":
            col_map[c] = "nino anom 3.4"
    df = df.rename(columns=col_map)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["location_id", "time"]).reset_index(drop=True)

    train = df[(df["time"] >= "2005-01-01") & (df["time"] <= "2018-12-31")]
    val   = df[(df["time"] >= "2019-01-01") & (df["time"] <= "2021-12-31")]
    test  = df[(df["time"] >= "2022-01-01") & (df["time"] <= "2025-05-01")]

    return train, val, test

# build windows for on one specific location_id
def build_windows_one_location(times, values, input_len, horizon, stride):
    """Build (X, y) windows for one location. values: (T, n_features)."""
    T = len(times)
    if T < input_len + horizon: # if the station does not have enough values to form one full window -> we skip, to prevent indexing error
        return None, None
    X_list, y_list = [], []
    for start in range(0, T - input_len - horizon + 1, stride):
        end_in = start + input_len
        end_out = end_in + horizon
        X_list.append(values[start:end_in])
        y_list.append(values[end_in:end_out, 0])  # target is first column (max_temperature)
    if not X_list:
        return None, None
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_all_windows(train, val, test):
    """Build X, y for train/val/test. No cross location_id, no cross split boundary."""
    def run_split(split_df):
        X_list, y_list = [], []
        for lid, grp in split_df.groupby("location_id"):
            grp = grp.sort_values("time")
            mat = grp[FEATURE_COLS].values.astype(np.float32)
            times = grp["time"].values
            X, y = build_windows_one_location(times, mat, INPUT_LEN, HORIZON, STRIDE)
            if X is not None:
                X_list.append(X)
                y_list.append(y)
        if not X_list:
            return np.zeros((0, INPUT_LEN, len(FEATURE_COLS)), dtype=np.float32), np.zeros((0, HORIZON), dtype=np.float32)
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    X_train, y_train = run_split(train)
    X_val,   y_val   = run_split(val)
    X_test,  y_test  = run_split(test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_windows(split_df: pd.DataFrame):
    """Build (X, y) windows for a single split DataFrame (across all location_id)."""
    X_list, y_list = [], []
    for _, grp in split_df.groupby("location_id"):
        grp = grp.sort_values("time")
        mat = grp[FEATURE_COLS].values.astype(np.float32)
        times = grp["time"].values
        X, y = build_windows_one_location(times, mat, INPUT_LEN, HORIZON, STRIDE)
        if X is not None:
            X_list.append(X)
            y_list.append(y)
    if not X_list:
        return (
            np.zeros((0, INPUT_LEN, len(FEATURE_COLS)), dtype=np.float32),
            np.zeros((0, HORIZON), dtype=np.float32),
        )
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def region_mask(df: pd.DataFrame, region_name: str):
    return df["region"].astype(str).str.strip().str.lower() == region_name.strip().lower()


def compute_station_stats(papua_train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-station statistics from Papua *training* rows:
      - mean temperature (TARGET_COL)
      - std temperature (TARGET_COL)
      - elevation (mean of 'elevation' column)
    Returns a DataFrame with columns: location_id, mean_temp, std_temp, elevation, n_rows.
    """
    if papua_train is None or len(papua_train) == 0:
        return pd.DataFrame(columns=["location_id", "mean_temp", "std_temp", "elevation", "n_rows"])

    df = papua_train.copy()
    df["location_id"] = df["location_id"].astype(str)
    g = df.groupby("location_id", dropna=False)

    stats = g.agg(
        mean_temp=(TARGET_COL, "mean"),
        std_temp=(TARGET_COL, "std"),
        elevation=("elevation", "mean"),
        n_rows=(TARGET_COL, "size"),
    ).reset_index()

    stats["mean_temp"] = stats["mean_temp"].astype(np.float32)
    stats["std_temp"] = stats["std_temp"].fillna(0.0).astype(np.float32)
    stats["elevation"] = stats["elevation"].astype(np.float32)
    stats["n_rows"] = stats["n_rows"].astype(int)
    return stats


def select_target_stations_papua_by_elevation(papua_train: pd.DataFrame):
    """
    Select station IDs from Papua training set by elevation:
      - Single station: median elevation station
      - Three stations: lowest, median, highest elevation stations
    Returns (single_station_id: str, three_station_ids: list[str], stats_sorted: pd.DataFrame).
    """
    stats = compute_station_stats(papua_train)
    if len(stats) == 0:
        return None, [], stats

    stats_sorted = stats.sort_values(["elevation", "location_id"], ascending=[True, True]).reset_index(drop=True)
    mid = len(stats_sorted) // 2
    single_id = str(stats_sorted.loc[mid, "location_id"])

    low_id = str(stats_sorted.loc[0, "location_id"])
    high_id = str(stats_sorted.loc[len(stats_sorted) - 1, "location_id"])
    three_ids = [low_id, single_id, high_id]

    # If very small number of stations, deduplicate while keeping order
    seen = set()
    three_ids = [x for x in three_ids if not (x in seen or seen.add(x))]
    return single_id, three_ids, stats_sorted


def filter_df_by_station_ids(df: pd.DataFrame, station_ids) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df.iloc[0:0].copy() if df is not None else df
    ids = set([str(x) for x in station_ids])
    out = df.copy()
    out["location_id"] = out["location_id"].astype(str)
    return out[out["location_id"].isin(ids)]


def eval_model_metrics(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int = 256):
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
            xb = xb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            preds.append(model(xb).detach().cpu().numpy().astype(np.float32))
    pred = np.concatenate(preds, axis=0).astype(np.float32)
    y = y.astype(np.float32, copy=False)
    err = (pred - y).astype(np.float32)
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    return {"mae": mae, "mse": mse, "rmse": rmse}


def run_target_station_experiment(
    experiment_name: str,
    train_java_df: pd.DataFrame,
    train_papua_df: pd.DataFrame,
    val_papua_df: pd.DataFrame,
    test_papua_df: pd.DataFrame,
    target_station_ids,
    arima_baseline: dict | None = None,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
):
    """
    Train multiple models for a given target-station subset and evaluate on FULL Papua test set:
      - Vanilla Transformer (Papua-only, supervised)
      - DANN (Java vs selected Papua)
      - KMM (Java reweighted to selected Papua)
      - Climate-Aware Transformer (Papua-only, supervised)
      - Climate-Aware + DANN (Java vs selected Papua)
    Optionally includes a precomputed ARIMA baseline dict.
    """
    print("\n" + "=" * 60)
    print(f"{experiment_name}")
    print("=" * 60)
    print(f"Selected Papua target stations (train only): {', '.join([str(x) for x in target_station_ids])}")

    train_papua_sel = filter_df_by_station_ids(train_papua_df, target_station_ids)

    X_src, y_src = build_windows(train_java_df)
    X_tgt, y_tgt = build_windows(train_papua_sel)
    X_tgt_val, y_tgt_val = build_windows(val_papua_df)  # FULL Papua val
    X_tgt_test, y_tgt_test = build_windows(test_papua_df)  # FULL Papua test

    print(f"Java train windows:         {X_src.shape[0]}")
    print(f"Papua train windows (sel):  {X_tgt.shape[0]}")
    print(f"Papua val windows (FULL):   {X_tgt_val.shape[0]}")
    print(f"Papua test windows (FULL):  {X_tgt_test.shape[0]}")

    if X_tgt.shape[0] == 0 or X_tgt_val.shape[0] == 0 or X_tgt_test.shape[0] == 0:
        print("Not enough windows for this experiment; skipping.")
        out = {
            "arima": (arima_baseline or {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}),
            "vanilla": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
            "dann": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
            "kmm": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
            "climate_aware": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
            "climate_aware_dann": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
        }
        return out

    # -----------------------------
    # A) Vanilla: Papua-only (no Java)
    # -----------------------------
    print("\n" + "-" * 60)
    print("A) Vanilla Transformer (Papua only)")
    print("-" * 60)
    vanilla = VanillaTransformer(
        input_dim=X_tgt.shape[2],
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        horizon=HORIZON,
    )
    vanilla, best_val_mae = train_supervised_earlystop_target_mae(
        vanilla,
        X_tgt,
        y_tgt,
        X_tgt_val,
        y_tgt_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )
    vanilla_metrics = eval_model_metrics(vanilla, X_tgt_test, y_tgt_test, batch_size=256)
    print(f"Best Papua Val MAE (early stop): {best_val_mae:.4f}")
    print(
        f"Papua Test  MAE: {vanilla_metrics['mae']:.4f}  "
        f"RMSE: {vanilla_metrics['rmse']:.4f}  MSE: {vanilla_metrics['mse']:.6f}"
    )

    # -----------------------------
    # B) DANN: adversarial DA
    # -----------------------------
    dann_metrics = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    if X_src.shape[0] > 0:
        print("\n" + "-" * 60)
        print("B) DANN (Java vs selected Papua; balanced batches)")
        print("-" * 60)
        feat = FeatureExtractor(input_dim=X_src.shape[2], d_model=32, nhead=4, num_layers=3, dropout=0.1)
        task_head = nn.Linear(32, HORIZON)
        dom = DomainClassifier(in_dim=32)
        dann_model, best_val_mae_dann = train_domain_adversarial_dann(
            feat,
            task_head,
            dom,
            X_src,
            y_src,
            X_tgt,
            y_tgt,
            X_tgt_val,
            y_tgt_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            use_target_task_loss=False,
        )
        dann_metrics = eval_model_metrics(dann_model, X_tgt_test, y_tgt_test, batch_size=256)
        print(f"Best Papua Val MAE (early stop): {best_val_mae_dann:.4f}")
        print(
            f"Papua Test  MAE: {dann_metrics['mae']:.4f}  "
            f"RMSE: {dann_metrics['rmse']:.4f}  MSE: {dann_metrics['mse']:.6f}"
        )
    else:
        print("\n" + "-" * 60)
        print("B) DANN skipped (no Java/source windows)")
        print("-" * 60)

    # -----------------------------
    # C) KMM: reweight Java -> selected Papua
    # -----------------------------
    kmm_metrics = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    if X_src.shape[0] > 0 and X_tgt.shape[0] > 0:
        print("\n" + "-" * 60)
        print("C) KMM (Java reweighted to selected Papua)")
        print("-" * 60)
        try:
            X_src_flat = X_src.reshape(X_src.shape[0], -1)
            X_tgt_flat = X_tgt.reshape(X_tgt.shape[0], -1)
            subsample_src = min(2000, X_src_flat.shape[0])
            subsample_tgt = min(800, X_tgt_flat.shape[0])
            rs_s = np.random.RandomState(42)
            rs_t = np.random.RandomState(43)
            idx_s = rs_s.choice(X_src_flat.shape[0], subsample_src, replace=False)
            idx_t = rs_t.choice(X_tgt_flat.shape[0], subsample_tgt, replace=False)
            beta = kmm_weights(X_src_flat[idx_s], X_tgt_flat[idx_t], B=2.0, eps=0.1)
            kmm_model, best_val_loss = train_transformer_weighted(
                X_src[idx_s],
                y_src[idx_s],
                X_tgt_val,
                y_tgt_val,
                beta,
                epochs=max(5, min(epochs, 25)),
                batch_size=batch_size,
                lr=lr,
                patience=patience,
            )
            kmm_metrics = eval_model_metrics(kmm_model, X_tgt_test, y_tgt_test, batch_size=256)
            print(f"Best Val loss (early stop): {best_val_loss:.6f}")
            print(
                f"Papua Test  MAE: {kmm_metrics['mae']:.4f}  "
                f"RMSE: {kmm_metrics['rmse']:.4f}  MSE: {kmm_metrics['mse']:.6f}"
            )
        except Exception as e:
            print(f"KMM skipped: {e}")
    else:
        print("\n" + "-" * 60)
        print("C) KMM skipped (no Java/source or no target windows)")
        print("-" * 60)

    # -----------------------------
    # D) Climate-Aware: Papua-only supervised
    # -----------------------------
    print("\n" + "-" * 60)
    print("D) Climate-Aware Transformer (Papua only)")
    print("-" * 60)
    climate_model = ClimateAwareTransformer(horizon=HORIZON, d_model=32, nhead=4, num_layers=2, dropout=0.1)
    climate_model, best_val_mae_clim = train_supervised_earlystop_target_mae(
        climate_model,
        X_tgt,
        y_tgt,
        X_tgt_val,
        y_tgt_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )
    climate_metrics = eval_model_metrics(climate_model, X_tgt_test, y_tgt_test, batch_size=256)
    print(f"Best Papua Val MAE (early stop): {best_val_mae_clim:.4f}")
    print(
        f"Papua Test  MAE: {climate_metrics['mae']:.4f}  "
        f"RMSE: {climate_metrics['rmse']:.4f}  MSE: {climate_metrics['mse']:.6f}"
    )

    # -----------------------------
    # E) Climate-Aware + DANN
    # -----------------------------
    ca_dann_metrics = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    if X_src.shape[0] > 0:
        print("\n" + "-" * 60)
        print("E) Climate-Aware + DANN (Java vs selected Papua)")
        print("-" * 60)
        ca_feat = ClimateAwareRep(d_model=32, nhead=4, num_layers=2, dropout=0.1)
        ca_task = nn.Linear(64, HORIZON)
        ca_dom = DomainClassifier(in_dim=64)
        ca_dann_model, best_val_mae_cadann = train_domain_adversarial_dann(
            ca_feat,
            ca_task,
            ca_dom,
            X_src,
            y_src,
            X_tgt,
            y_tgt,
            X_tgt_val,
            y_tgt_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            use_target_task_loss=False,
        )
        ca_dann_metrics = eval_model_metrics(ca_dann_model, X_tgt_test, y_tgt_test, batch_size=256)
        print(f"Best Papua Val MAE (early stop): {best_val_mae_cadann:.4f}")
        print(
            f"Papua Test  MAE: {ca_dann_metrics['mae']:.4f}  "
            f"RMSE: {ca_dann_metrics['rmse']:.4f}  MSE: {ca_dann_metrics['mse']:.6f}"
        )
    else:
        print("\n" + "-" * 60)
        print("E) Climate-Aware + DANN skipped (no Java/source windows)")
        print("-" * 60)

    out = {
        "arima": (arima_baseline or {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}),
        "vanilla": vanilla_metrics,
        "dann": dann_metrics,
        "kmm": kmm_metrics,
        "climate_aware": climate_metrics,
        "climate_aware_dann": ca_dann_metrics,
    }

    print("\n" + "-" * 60)
    print("Result (Papua test):")
    for k in ["arima", "vanilla", "dann", "kmm", "climate_aware", "climate_aware_dann"]:
        m = out[k]
        print(f"  {k:>18s}  MAE={m['mae']:.4f}  MSE={m['mse']:.6f}  RMSE={m['rmse']:.4f}")
    return out


def eval_model_mae(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
    if X.shape[0] == 0:
        return float("nan")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size])
            xb = xb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            preds.append(model(xb).detach().cpu().numpy().astype(np.float32))
    pred = np.concatenate(preds, axis=0)
    return float(np.mean(np.abs(pred - y)))


def eval_model_mae_masked(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, mask: np.ndarray, batch_size: int = 256) -> float:
    if X.shape[0] == 0:
        return float("nan")
    if mask is None or mask.size == 0:
        return float("nan")
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    return eval_model_mae(model, X[mask], y[mask], batch_size=batch_size)


def enso_extreme_mask(X: np.ndarray, thresh_abs_nino: float = 1.0) -> np.ndarray:
    """Mask windows where mean(|Nino 3.4|) over input window >= thresh."""
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    nino_idx = FEATURE_COLS.index("nino anom 3.4")
    m = np.mean(np.abs(X[:, :, nino_idx]), axis=1)
    return (m >= float(thresh_abs_nino))


# from training.arima_baseline import selet_arima_order


# def selet_arima_order(series):
#     """
#     Backwards-compat wrapper kept for any legacy callers inside this
#     module. The implementation now lives in `arima_baseline`.
#     """
#     from training.arima_baseline import selet_arima_order as _selet_arima_order

#     return _selet_arima_order(series)

# ARIMA BASELINE (per-station on train, rolling 7-day forecast on test)
# def arima_forecast_one_series(series, horizon=7, max_end=None):
#     """Fit ARIMA with (p,d,q) in [0,2], pick lowest AIC; then forecast horizon steps."""
#     try:
#         from statsmodels.tsa.arima.model import ARIMA
#         from statsmodels.tsa.stattools import adfuller
#     except ImportError:
#         return None

#     if max_end is not None and len(series) > max_end:
#         series = series[:max_end]
#     if len(series) < 30:
#         return None
#     series = np.asarray(series, dtype=np.float64)
#     # Simple differencing for d
#     d = 0
#     try:
#         adf = adfuller(series, maxlag=1, autolag=None)
#         if adf[1] > 0.05:
#             d = 1
#     except Exception:
#         d = 0

#     best_aic, best_model = np.inf, None
#     for p in range(3):
#         for q in range(3):
#             try:
#                 m = ARIMA(series, order=(p, d, q))
#                 fitted = m.fit()
#                 if fitted.aic < best_aic:
#                     best_aic = fitted.aic
#                     best_model = fitted
#             except Exception:
#                 continue
#     if best_model is None:
#         return None
#     try:
#         f = best_model.forecast(steps=horizon)
#         return np.asarray(f, dtype=np.float32)
#     except Exception:
#         return None
        

# def arima_baseline_rolling(test_df, train_df, horizon=7):
#     from statsmodels.tsa.arima.model import ARIMA
#     test_df = test_df.sort_values(["location_id", "time"])
#     locations = test_df["location_id"].unique()
#     all_mae, all_rmse = [], []
#     for lid in locations:
#         train_ser = train_df[train_df["location_id"] == lid].sort_values("time")[TARGET_COL]
#         test_grp = test_df[test_df["location_id"] == lid].sort_values("time")
#         if len(train_ser) < 30 or len(test_grp) < horizon:
#             continue
#         train_ser = train_ser.values.astype(np.float64)
#         test_vals = test_grp[TARGET_COL].values.astype(np.float32)
#         order = selet_arima_order(train_ser)
#         preds = []
#         for i in range(0, len(test_vals) - horizon + 1):
#             history = np.concatenate([train_ser, test_vals[:i]]) if i > 0 else train_ser
#             try:
#                 model = ARIMA(history, order=order)
#                 fitted = model.fit(method_kwargs={"maxiter": 200})
#                 forecast = fitted.forecast(steps=horizon)
#                 preds.append(forecast)
#             except Exception:
#                 break
#         if not preds:
#             continue
#         preds = np.array(preds, dtype=np.float32)
#         y_true = np.array([test_vals[i:i+horizon] for i in range(len(preds))])
#         y_pred = preds
#         mae = np.mean(np.abs(y_true - y_pred))
#         rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
#         all_mae.append(mae)
#         all_rmse.append(rmse)
#     if not all_mae:
#         return float("nan"), float("nan")
#     return np.mean(all_mae), np.mean(all_rmse)


# def arima_baseline_rolling(test_df, train_df, horizon=7):
#     """Per station: fit on train, rolling 7-day forecast on test. Return MAE, RMSE (averaged)."""
#     from statsmodels.tsa.arima.model import ARIMA

#     test_df = test_df.sort_values(["location_id", "time"])
#     locations = test_df["location_id"].unique()
#     all_mae, all_rmse = [], []

#     for lid in locations:
#         train_ser = train_df[train_df["location_id"] == lid].sort_values("time")[TARGET_COL]
#         test_grp = test_df[test_df["location_id"] == lid].sort_values("time")
#         if len(train_ser) < 30 or len(test_grp) < horizon:
#             continue
#         train_ser = train_ser.values.astype(np.float64)
#         test_vals = test_grp[TARGET_COL].values.astype(np.float32)
#         preds = []
#         for i in range(0, len(test_vals) - horizon + 1):
#             hist = np.concatenate([train_ser, test_vals[:i]]) if i > 0 else train_ser
#             f = arima_forecast_one_series(hist, horizon=horizon)
#             if f is None:
#                 break
#             preds.append(f)
#         if not preds:
#             continue
#         preds = np.array(preds, dtype=np.float32)
#         n_steps = min(len(preds), (len(test_vals) - horizon + 1))
#         y_true = np.array([test_vals[i : i + horizon] for i in range(n_steps)], dtype=np.float32)
#         y_pred = preds[:n_steps]
#         mae = np.mean(np.abs(y_true - y_pred))
#         rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
#         all_mae.append(mae)
#         all_rmse.append(rmse)

#     if not all_mae:
#         return float("nan"), float("nan")
#     return np.mean(all_mae), np.mean(all_rmse)


## vanilla transformer
import torch.nn as nn
class FeatureExtractor(nn.Module):
    """Transformer encoder + mean pooling. Output: pooled representation (d_model)."""

    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        return x.mean(dim=1)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class DomainClassifier(nn.Module):
    """MLP: Linear(in→32) ReLU Linear(32→1) Sigmoid."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # (B,)


def dann_lambda_schedule(p: float) -> float:
    # λ = 2 / (1 + exp(-10 p)) - 1
    return float(2.0 / (1.0 + math.exp(-10.0 * float(p))) - 1.0)


# train the model 

def train_supervised_earlystop_target_mae(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tgt_val: np.ndarray,
    y_tgt_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
):
    """Train with MSE; early stop on target validation MAE."""
    from torch.utils.data import TensorDataset, DataLoader

    if X_train.shape[0] == 0:
        return model.to(DEVICE), float("nan")

    model = model.to(DEVICE).float()
    train_ds = TensorDataset(torch.from_numpy(X_train).to(DEVICE), torch.from_numpy(y_train).to(DEVICE))
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

        val_mae = eval_model_mae(model, X_tgt_val, y_tgt_val, batch_size=256)
        print(f"  Epoch {ep+1}/{epochs}  Train MSE: {tr_mse:.6f}  Target Val MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(DEVICE), best_mae


def train_domain_adversarial_dann(
    feat_extractor: nn.Module,
    task_head: nn.Module,
    domain_clf: nn.Module,
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_tgt: np.ndarray,
    y_tgt: np.ndarray,
    X_tgt_val: np.ndarray,
    y_tgt_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    use_target_task_loss: bool = False,
):
    """
    DANN:
      loss = forecast_MSE + λ * domain_BCE
      λ schedule: 2/(1+exp(-10p)) - 1, p=step/total_steps
      balanced batches: batch_size/2 source + batch_size/2 target
      early stop: target validation MAE
    """
    from torch.utils.data import TensorDataset, DataLoader

    if X_src.shape[0] == 0 or X_tgt.shape[0] == 0:
        return None, float("nan")

    src_bs = max(1, batch_size // 2)
    tgt_bs = max(1, batch_size // 2)

    src_ds = TensorDataset(torch.from_numpy(X_src).to(DEVICE), torch.from_numpy(y_src).to(DEVICE))
    tgt_ds = TensorDataset(torch.from_numpy(X_tgt).to(DEVICE), torch.from_numpy(y_tgt).to(DEVICE))
    src_loader = DataLoader(src_ds, batch_size=src_bs, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=tgt_bs, shuffle=True, num_workers=0, drop_last=True)

    feat_extractor = feat_extractor.to(DEVICE).float()
    task_head = task_head.to(DEVICE).float()
    domain_clf = domain_clf.to(DEVICE).float()

    params = list(feat_extractor.parameters()) + list(task_head.parameters()) + list(domain_clf.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    bce = nn.BCELoss()

    total_steps = epochs * min(len(src_loader), len(tgt_loader))
    global_step = 0

    best_mae, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        feat_extractor.train()
        task_head.train()
        domain_clf.train()

        tr_task, tr_dom = 0.0, 0.0
        n_steps = min(len(src_loader), len(tgt_loader))
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        for _ in range(n_steps):
            xs, ys = next(src_iter)
            xt, yt = next(tgt_iter)
            xb = torch.cat([xs, xt], dim=0)
            dom_y = torch.cat(
                [torch.zeros(xs.size(0), device=DEVICE), torch.ones(xt.size(0), device=DEVICE)],
                dim=0,
            ).float()

            p = global_step / max(1, total_steps)
            lambd = dann_lambda_schedule(p)
            global_step += 1

            opt.zero_grad()
            rep = feat_extractor(xb)
            yhat = task_head(rep)

            task_loss = nn.functional.mse_loss(yhat[: xs.size(0)], ys)
            if use_target_task_loss:
                task_loss = task_loss + nn.functional.mse_loss(yhat[xs.size(0) :], yt)

            dom_pred = domain_clf(grl(rep, lambd))
            dom_loss = bce(dom_pred, dom_y)

            loss = task_loss + (lambd * dom_loss)
            loss.backward()
            opt.step()

            tr_task += float(task_loss.item())
            tr_dom += float(dom_loss.item())

        eval_model = nn.Sequential(feat_extractor, task_head)
        val_mae = eval_model_mae(eval_model, X_tgt_val, y_tgt_val, batch_size=256)
        tr_task /= max(1, n_steps)
        tr_dom /= max(1, n_steps)
        print(f"  Epoch {ep+1}/{epochs}  Task MSE: {tr_task:.6f}  Domain BCE: {tr_dom:.6f}  Target Val MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {
                "feat": {k: v.cpu().clone() for k, v in feat_extractor.state_dict().items()},
                "task": {k: v.cpu().clone() for k, v in task_head.state_dict().items()},
                "dom": {k: v.cpu().clone() for k, v in domain_clf.state_dict().items()},
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {ep+1}")
                break

    if best_state is not None:
        feat_extractor.load_state_dict(best_state["feat"])
        task_head.load_state_dict(best_state["task"])
        domain_clf.load_state_dict(best_state["dom"])

    return nn.Sequential(feat_extractor, task_head).to(DEVICE), best_mae


class ClimateAwareTransformer(nn.Module):
    """
    Temporal local features -> Transformer encoder -> 32-dim pooled
    Geo features (lat, lon, elev) -> Linear(3->16)
    Climate regime (Nino, DMI) -> Linear(2->16)
    Concat -> 64 -> Linear(64->7)
    """

    def __init__(self, horizon: int = 7, d_model: int = 32, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.idx_temp = [FEATURE_COLS.index(c) for c in LOCAL_TEMPORAL_COLS]
        self.idx_geo = [FEATURE_COLS.index(c) for c in GEO_COLS]
        self.idx_clim = [FEATURE_COLS.index(c) for c in CLIMATE_COLS]

        self.temporal = FeatureExtractor(input_dim=len(self.idx_temp), d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.geo_proj = nn.Linear(3, 16)
        self.clim_proj = nn.Linear(2, 16)
        self.head = nn.Linear(d_model + 16 + 16, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x[:, :, self.idx_temp]
        rep_t = self.temporal(xt)
        rep_g = self.geo_proj(x[:, 0, self.idx_geo])
        rep_c = self.clim_proj(x[:, 0, self.idx_clim])
        rep = torch.cat([rep_t, rep_g, rep_c], dim=1)
        return self.head(rep)


class ClimateAwareRep(nn.Module):
    """Same as ClimateAwareTransformer but returns 64-dim representation (for optional DANN)."""

    def __init__(self, d_model: int = 32, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.idx_temp = [FEATURE_COLS.index(c) for c in LOCAL_TEMPORAL_COLS]
        self.idx_geo = [FEATURE_COLS.index(c) for c in GEO_COLS]
        self.idx_clim = [FEATURE_COLS.index(c) for c in CLIMATE_COLS]

        self.temporal = FeatureExtractor(input_dim=len(self.idx_temp), d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.geo_proj = nn.Linear(3, 16)
        self.clim_proj = nn.Linear(2, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x[:, :, self.idx_temp]
        rep_t = self.temporal(xt)
        rep_g = self.geo_proj(x[:, 0, self.idx_geo])
        rep_c = self.clim_proj(x[:, 0, self.idx_clim])
        return torch.cat([rep_t, rep_g, rep_c], dim=1)  # (B, 64)

class VanillaTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=32, nhead=4, num_layers=2, dropout=0.1, horizon=7):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


def train_eval_transformer(X_train, y_train, X_val, y_val, X_test, y_test,
                           epochs=20, batch_size=64, lr=1e-3, patience=5):
    """Train Transformer on CPU, early stopping on val loss. Return model, test MAE/RMSE."""
    from torch.utils.data import TensorDataset, DataLoader

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(DEVICE),
        torch.from_numpy(y_train).to(DEVICE),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).to(DEVICE),
        torch.from_numpy(y_val).to(DEVICE),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VanillaTransformer(
        input_dim=X_train.shape[2], d_model=32, nhead=4, num_layers=2,
        dropout=0.1, horizon=HORIZON,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        if len(X_val) > 0:
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb)
                    val_loss += nn.functional.mse_loss(pred, yb).item() * xb.size(0)
            val_loss /= len(X_val)
        else:
            val_loss = train_loss

        print(f"  Epoch {ep+1}/{epochs}  Train loss: {train_loss:.6f}  Val loss: {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).to(DEVICE)
        pred = model(Xt).cpu().numpy()
    mae = np.mean(np.abs(pred - y_test))
    rmse = np.sqrt(np.mean((pred - y_test) ** 2))
    return model, mae, rmse


# KMM domain adaptiation (Java-papua)
def rbf_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / (X.shape[1] * X.var() + 1e-8)
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    return sk_rbf(X, Y, gamma=gamma)


def kmm_weights(X_src, X_tgt, B=2.0, eps=0.1, gamma=None):
    """
    Kernel Mean Matching: minimize || (1/n_s) sum_i beta_i phi(x_i) - (1/n_t) sum_j phi(z_j) ||^2
    s.t. 0 <= beta_i <= B, | (1/n_s) sum beta_i - 1 | <= eps.
    """
    n_s, n_t = X_src.shape[0], X_tgt.shape[0]
    K_ss = rbf_kernel(X_src, X_src, gamma=gamma)
    K_st = rbf_kernel(X_src, X_tgt, gamma=gamma)
    kappa = (1.0 / n_t) * K_st.sum(axis=1)
    # min  (1/2) beta' Q beta - kappa' beta   with Q = (1/n_s^2) K_ss
    Q = (1.0 / (n_s * n_s)) * K_ss
    try:
        from cvxopt import matrix, solvers
        solvers.options["show_progress"] = False
        P = matrix(2 * Q)
        q = matrix(-kappa.astype(np.float64))
        G = np.vstack([-np.eye(n_s), np.eye(n_s)])
        h = np.hstack([np.zeros(n_s), B * np.ones(n_s)])
        A = np.ones((1, n_s))
        b = np.array([1.0])
        # | (1/n_s) sum beta - 1 | <= eps  =>  two inequalities
        A2 = np.vstack([A / n_s, -A / n_s])
        b2 = np.array([1 + eps, -(1 - eps)])
        G2 = np.vstack([G, A2])
        h2 = np.hstack([h, b2])
        G2 = matrix(G2.astype(np.float64))
        h2 = matrix(h2.astype(np.float64))
        A = matrix(A.astype(np.float64))
        b = matrix(b.astype(np.float64))
        sol = solvers.qp(P, q, G2, h2, A, b)
        if sol["status"] == "optimal":
            beta = np.array(sol["x"]).ravel().astype(np.float32)
            return np.clip(beta, 0, B)
    except Exception:
        pass
    # Fallback: scipy minimize
    from scipy.optimize import minimize
    def obj(b):
        return 0.5 * b @ Q @ b - kappa @ b

    # | (1/n_s) sum beta - 1 | <= eps  =>  (1-eps) <= sum/n_s <= (1+eps)
    cons = [
        {"type": "ineq", "fun": lambda b: (b.sum() / n_s) - (1 - eps)},
        {"type": "ineq", "fun": lambda b: (1 + eps) - (b.sum() / n_s)},
    ]
    res = minimize(obj, np.ones(n_s), method="SLSQP", bounds=[(0, B)] * n_s, constraints=cons)
    if res.success:
        return np.clip(res.x.astype(np.float32), 0, B)
    return np.ones(n_s, dtype=np.float32)


def train_transformer_weighted(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
):
    """
    Train Transformer with weighted MSE: mean(weights * (pred - target)^2).
    Returns (model, best_val_loss).
    """
    from torch.utils.data import TensorDataset, DataLoader

    if X_train.shape[0] == 0:
        dummy = VanillaTransformer(input_dim=1, d_model=32, nhead=4, num_layers=2, dropout=0.1, horizon=HORIZON)
        return _maybe_compile(dummy).to(DEVICE), float("nan")

    w = torch.from_numpy(np.asarray(weights, dtype=np.float32)).float()
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), w[: len(X_train)].clone())
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, **_dataloader_kwargs(shuffle=True, drop_last=False))
    val_loader = DataLoader(val_ds, batch_size=batch_size, **_dataloader_kwargs(shuffle=False, drop_last=False))

    model = VanillaTransformer(
        input_dim=X_train.shape[2], d_model=32, nhead=4, num_layers=2,
        dropout=0.1, horizon=HORIZON,
    )
    model = _maybe_compile(model).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, wait = float("inf"), None, 0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb, wb in train_loader:
            xb = xb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            yb = yb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            wb = wb.to(DEVICE, non_blocking=(DEVICE.type == "cuda")).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
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
                    xb = xb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
                    yb = yb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
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
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(DEVICE), float(best_val)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    data_path = "data/merged_dataset.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "data", "merged_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Not found: {data_path}")

    set_seed(42)

    # -----------------------------
    # Results logging (CSV)
    # -----------------------------
    run_ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    results_rows = []

    def log_row(**kwargs):
        row = {
            "run_timestamp": run_ts,
            "target_region": "papua",
            "source_region": "java",
            "TARGET_COL": TARGET_COL,
            "INPUT_LEN": INPUT_LEN,
            "HORIZON": HORIZON,
            "STRIDE": STRIDE,
            "n_features": len(FEATURE_COLS),
        }
        row.update(kwargs)
        results_rows.append(row)

    print("=" * 60)
    print("PART 1 — DATA PREPROCESSING (time-separated splits)")
    print("=" * 60)
    train, val, test = load_and_split(data_path)
    print(f"Train: {train['time'].min()} to {train['time'].max()}  rows: {len(train)}")
    print(f"Val:   {val['time'].min()} to {val['time'].max()}  rows: {len(val)}")
    print(f"Test:  {test['time'].min()} to {test['time'].max()}  rows: {len(test)}")

    # Domain dataframes (source=Java; target=Papua)
    train_java = train[region_mask(train, "java")]
    train_papua = train[region_mask(train, "papua")]
    val_papua = val[region_mask(val, "papua")]
    test_papua = test[region_mask(test, "papua")]

    print("\n" + "=" * 60)
    print("PART 2 — STATION SELECTION LOGIC (Papua target)")
    print("=" * 60)
    single_id, three_ids, stats_sorted = select_target_stations_papua_by_elevation(train_papua)
    if single_id is None:
        print("No Papua stations found in training split; aborting.")
        return

    print(f"Papua training stations: {len(stats_sorted)}")
    print("Selected station IDs (by elevation):")
    print(f"  Single-station (median elevation): {single_id}")
    if len(three_ids) == 3:
        print(f"  Three-station (low/median/high):   {three_ids[0]}, {three_ids[1]}, {three_ids[2]}")
    else:
        print(f"  Three-station (deduped):           {', '.join(three_ids)}")

    # -----------------------------
    # ARIMA baseline (Papua-only) — used in single/three station experiments
    # -----------------------------
    arima_baseline = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    try:
        arima_mae, arima_rmse = arima_baseline_rolling(test_papua, train_papua, horizon=HORIZON)
        arima_mse = float(arima_rmse ** 2) if not np.isnan(arima_rmse) else float("nan")
        arima_baseline = {"mae": float(arima_mae), "mse": float(arima_mse), "rmse": float(arima_rmse)}
        print("\nARIMA baseline (Papua only, FULL Papua test):")
        print(f"  MAE: {arima_baseline['mae']:.4f}  RMSE: {arima_baseline['rmse']:.4f}  MSE: {arima_baseline['mse']:.6f}")
    except Exception as e:
        print(f"\nARIMA baseline skipped: {e}")

    print("\n" + "=" * 60)
    print("PART 3 — SINGLE-STATION TARGET EXPERIMENT")
    print("=" * 60)
    res_single = run_target_station_experiment(
        "Single-Station Experiment (Target=Papua; Train target=1 station; Val/Test=FULL Papua)",
        train_java_df=train_java,
        train_papua_df=train_papua,
        val_papua_df=val_papua,
        test_papua_df=test_papua,
        target_station_ids=[single_id],
        arima_baseline=arima_baseline,
        epochs=25,
        batch_size=64,
        lr=1e-3,
        patience=5,
    )
    log_row(
        experiment="single_station",
        target_station_ids="|".join([str(single_id)]),
        method="arima",
        metric_mae=res_single["arima"]["mae"],
        metric_mse=res_single["arima"]["mse"],
        metric_rmse=res_single["arima"]["rmse"],
        note="Papua-only; FULL Papua test",
    )
    log_row(
        experiment="single_station",
        target_station_ids="|".join([str(single_id)]),
        method="vanilla",
        metric_mae=res_single["vanilla"]["mae"],
        metric_mse=res_single["vanilla"]["mse"],
        metric_rmse=res_single["vanilla"]["rmse"],
    )
    log_row(
        experiment="single_station",
        target_station_ids="|".join([str(single_id)]),
        method="dann",
        metric_mae=res_single["dann"]["mae"],
        metric_mse=res_single["dann"]["mse"],
        metric_rmse=res_single["dann"]["rmse"],
    )
    log_row(
        experiment="single_station",
        target_station_ids="|".join([str(single_id)]),
        method="kmm",
        metric_mae=res_single["kmm"]["mae"],
        metric_mse=res_single["kmm"]["mse"],
        metric_rmse=res_single["kmm"]["rmse"],
    )
    log_row(
        experiment="single_station",
        target_station_ids="|".join([str(single_id)]),
        method="climate_aware",
        metric_mae=res_single["climate_aware"]["mae"],
        metric_mse=res_single["climate_aware"]["mse"],
        metric_rmse=res_single["climate_aware"]["rmse"],
    )
    log_row(
        experiment="single_station",
        target_station_ids="|".join([str(single_id)]),
        method="climate_aware_dann",
        metric_mae=res_single["climate_aware_dann"]["mae"],
        metric_mse=res_single["climate_aware_dann"]["mse"],
        metric_rmse=res_single["climate_aware_dann"]["rmse"],
    )

    print("\n" + "=" * 60)
    print("PART 4 — THREE-STATION TARGET EXPERIMENT")
    print("=" * 60)
    res_three = run_target_station_experiment(
        "Three-Station Experiment (Target=Papua; Train target=3 stations; Val/Test=FULL Papua)",
        train_java_df=train_java,
        train_papua_df=train_papua,
        val_papua_df=val_papua,
        test_papua_df=test_papua,
        target_station_ids=three_ids,
        arima_baseline=arima_baseline,
        epochs=25,
        batch_size=64,
        lr=1e-3,
        patience=5,
    )
    log_row(
        experiment="three_station",
        target_station_ids="|".join([str(x) for x in three_ids]),
        method="arima",
        metric_mae=res_three["arima"]["mae"],
        metric_mse=res_three["arima"]["mse"],
        metric_rmse=res_three["arima"]["rmse"],
        note="Papua-only; FULL Papua test",
    )
    log_row(
        experiment="three_station",
        target_station_ids="|".join([str(x) for x in three_ids]),
        method="vanilla",
        metric_mae=res_three["vanilla"]["mae"],
        metric_mse=res_three["vanilla"]["mse"],
        metric_rmse=res_three["vanilla"]["rmse"],
    )
    log_row(
        experiment="three_station",
        target_station_ids="|".join([str(x) for x in three_ids]),
        method="dann",
        metric_mae=res_three["dann"]["mae"],
        metric_mse=res_three["dann"]["mse"],
        metric_rmse=res_three["dann"]["rmse"],
    )
    log_row(
        experiment="three_station",
        target_station_ids="|".join([str(x) for x in three_ids]),
        method="kmm",
        metric_mae=res_three["kmm"]["mae"],
        metric_mse=res_three["kmm"]["mse"],
        metric_rmse=res_three["kmm"]["rmse"],
    )
    log_row(
        experiment="three_station",
        target_station_ids="|".join([str(x) for x in three_ids]),
        method="climate_aware",
        metric_mae=res_three["climate_aware"]["mae"],
        metric_mse=res_three["climate_aware"]["mse"],
        metric_rmse=res_three["climate_aware"]["rmse"],
    )
    log_row(
        experiment="three_station",
        target_station_ids="|".join([str(x) for x in three_ids]),
        method="climate_aware_dann",
        metric_mae=res_three["climate_aware_dann"]["mae"],
        metric_mse=res_three["climate_aware_dann"]["mse"],
        metric_rmse=res_three["climate_aware_dann"]["rmse"],
    )

    print("\n" + "=" * 60)
    print("PART 5 — LOW-RESOURCE TARGET EXPERIMENT (Papua-only + Java→Papua DA methods)")
    print("=" * 60)
    # Only this low-resource block is repeated for multiple input lengths and multiple random subsamples.
    # Methods:
    #   - Papua-only: Vanilla, Climate-Aware
    #   - Java→Papua: DANN, KMM, Climate-Aware + DANN
    input_lens = [7, 14, 21]
    low_resource_fracs = [0.05, 0.15, 0.25]
    n_repeats = 30

    # Wilcoxon signed-rank for paired comparisons (same random subsample across methods)
    try:
        from scipy.stats import wilcoxon as wilcoxon_signed_rank
        _has_wilcoxon = True
    except Exception:
        wilcoxon_signed_rank = None
        _has_wilcoxon = False

    global INPUT_LEN
    original_input_len = INPUT_LEN

    for in_len in input_lens:
        INPUT_LEN = int(in_len)
        print("\n" + "-" * 60)
        print(f"Input length setting: INPUT_LEN={INPUT_LEN}")
        print("-" * 60)

        X_src_lr, y_src_lr = build_windows(train_java)
        X_tgt_full, y_tgt_full = build_windows(train_papua)
        X_tgt_val_lr, y_tgt_val_lr = build_windows(val_papua)
        X_tgt_test_lr, y_tgt_test_lr = build_windows(test_papua)

        if X_src_lr.shape[0] == 0 or X_tgt_full.shape[0] == 0 or X_tgt_val_lr.shape[0] == 0 or X_tgt_test_lr.shape[0] == 0:
            print("Not enough windows for low-resource experiment at this input length; skipping.")
            continue

        rs_base = np.random.RandomState(123 + int(in_len))
        frac_results = {}

        for frac in low_resource_fracs:
            n = max(1, int(round(frac * X_tgt_full.shape[0])))
            vanilla_mae_runs, vanilla_mse_runs = [], []
            dann_mae_runs, dann_mse_runs = [], []
            kmm_mae_runs, kmm_mse_runs = [], []
            ca_mae_runs, ca_mse_runs = [], []
            cadann_mae_runs, cadann_mse_runs = [], []

            for rep_idx in range(n_repeats):
                seed = int(rs_base.randint(0, 2**31 - 1))
                rs = np.random.RandomState(seed)
                idx = rs.choice(X_tgt_full.shape[0], n, replace=False)
                X_tgt_small = X_tgt_full[idx]
                y_tgt_small = y_tgt_full[idx]

                # (a) Vanilla Transformer (Papua-only, supervised)
                vanilla_lr = VanillaTransformer(
                    input_dim=X_tgt_small.shape[2],
                    d_model=32,
                    nhead=4,
                    num_layers=2,
                    dropout=0.1,
                    horizon=HORIZON,
                )
                vanilla_lr, _ = train_supervised_earlystop_target_mae(
                    vanilla_lr,
                    X_tgt_small,
                    y_tgt_small,
                    X_tgt_val_lr,
                    y_tgt_val_lr,
                    epochs=25,
                    batch_size=64,
                    lr=1e-3,
                    patience=5,
                )
                vanilla_metrics = eval_model_metrics(vanilla_lr, X_tgt_test_lr, y_tgt_test_lr, batch_size=256)
                log_row(
                    experiment="low_resource",
                    phase="repeat",
                    method="vanilla",
                    input_len=INPUT_LEN,
                    frac_target=frac,
                    n_target=n,
                    rep_idx=rep_idx,
                    seed=seed,
                    metric_mae=float(vanilla_metrics["mae"]),
                    metric_mse=float(vanilla_metrics["mse"]),
                    metric_rmse=float(vanilla_metrics["rmse"]),
                )

                # (b) DANN (Java + small Papua; domain adaptation)
                feat_lr = FeatureExtractor(input_dim=X_src_lr.shape[2], d_model=32, nhead=4, num_layers=3, dropout=0.1)
                task_lr = nn.Linear(32, HORIZON)
                dom_lr = DomainClassifier(in_dim=32)
                dann_lr_model, _ = train_domain_adversarial_dann(
                    feat_lr,
                    task_lr,
                    dom_lr,
                    X_src_lr,
                    y_src_lr,
                    X_tgt_small,
                    y_tgt_small,
                    X_tgt_val_lr,
                    y_tgt_val_lr,
                    epochs=25,
                    batch_size=64,
                    lr=1e-3,
                    patience=5,
                    use_target_task_loss=True,
                )
                dann_metrics = eval_model_metrics(dann_lr_model, X_tgt_test_lr, y_tgt_test_lr, batch_size=256)
                log_row(
                    experiment="low_resource",
                    phase="repeat",
                    method="dann",
                    input_len=INPUT_LEN,
                    frac_target=frac,
                    n_target=n,
                    rep_idx=rep_idx,
                    seed=seed,
                    metric_mae=float(dann_metrics["mae"]),
                    metric_mse=float(dann_metrics["mse"]),
                    metric_rmse=float(dann_metrics["rmse"]),
                    use_target_task_loss=True,
                )

                # (c) KMM (Java reweighted to small Papua)
                kmm_metrics = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
                try:
                    X_src_flat = X_src_lr.reshape(X_src_lr.shape[0], -1)
                    X_tgt_flat = X_tgt_small.reshape(X_tgt_small.shape[0], -1)
                    subsample_src = min(2000, X_src_flat.shape[0])
                    subsample_tgt = min(800, X_tgt_flat.shape[0])
                    idx_s = np.random.RandomState(seed ^ 0xA5A5).choice(X_src_flat.shape[0], subsample_src, replace=False)
                    idx_t = np.random.RandomState(seed ^ 0x5A5A).choice(X_tgt_flat.shape[0], subsample_tgt, replace=False)
                    beta = kmm_weights(X_src_flat[idx_s], X_tgt_flat[idx_t], B=2.0, eps=0.1)
                    kmm_model, _ = train_transformer_weighted(
                        X_src_lr[idx_s],
                        y_src_lr[idx_s],
                        X_tgt_val_lr,
                        y_tgt_val_lr,
                        beta,
                        epochs=25,
                        batch_size=64,
                        lr=1e-3,
                        patience=5,
                    )
                    kmm_metrics = eval_model_metrics(kmm_model, X_tgt_test_lr, y_tgt_test_lr, batch_size=256)
                except Exception:
                    pass
                log_row(
                    experiment="low_resource",
                    phase="repeat",
                    method="kmm",
                    input_len=INPUT_LEN,
                    frac_target=frac,
                    n_target=n,
                    rep_idx=rep_idx,
                    seed=seed,
                    metric_mae=float(kmm_metrics["mae"]),
                    metric_mse=float(kmm_metrics["mse"]),
                    metric_rmse=float(kmm_metrics["rmse"]),
                )

                # (d) Climate-Aware (Papua-only, supervised)
                ca_lr = ClimateAwareTransformer(horizon=HORIZON, d_model=32, nhead=4, num_layers=2, dropout=0.1)
                ca_lr, _ = train_supervised_earlystop_target_mae(
                    ca_lr,
                    X_tgt_small,
                    y_tgt_small,
                    X_tgt_val_lr,
                    y_tgt_val_lr,
                    epochs=25,
                    batch_size=64,
                    lr=1e-3,
                    patience=5,
                )
                ca_metrics = eval_model_metrics(ca_lr, X_tgt_test_lr, y_tgt_test_lr, batch_size=256)
                log_row(
                    experiment="low_resource",
                    phase="repeat",
                    method="climate_aware",
                    input_len=INPUT_LEN,
                    frac_target=frac,
                    n_target=n,
                    rep_idx=rep_idx,
                    seed=seed,
                    metric_mae=float(ca_metrics["mae"]),
                    metric_mse=float(ca_metrics["mse"]),
                    metric_rmse=float(ca_metrics["rmse"]),
                )

                # (e) Climate-Aware + DANN (Java + small Papua)
                ca_feat_lr = ClimateAwareRep(d_model=32, nhead=4, num_layers=2, dropout=0.1)
                ca_task_lr = nn.Linear(64, HORIZON)
                ca_dom_lr = DomainClassifier(in_dim=64)
                cadann_lr_model, _ = train_domain_adversarial_dann(
                    ca_feat_lr,
                    ca_task_lr,
                    ca_dom_lr,
                    X_src_lr,
                    y_src_lr,
                    X_tgt_small,
                    y_tgt_small,
                    X_tgt_val_lr,
                    y_tgt_val_lr,
                    epochs=25,
                    batch_size=64,
                    lr=1e-3,
                    patience=5,
                    use_target_task_loss=True,
                )
                cadann_metrics = eval_model_metrics(cadann_lr_model, X_tgt_test_lr, y_tgt_test_lr, batch_size=256)
                log_row(
                    experiment="low_resource",
                    phase="repeat",
                    method="climate_aware_dann",
                    input_len=INPUT_LEN,
                    frac_target=frac,
                    n_target=n,
                    rep_idx=rep_idx,
                    seed=seed,
                    metric_mae=float(cadann_metrics["mae"]),
                    metric_mse=float(cadann_metrics["mse"]),
                    metric_rmse=float(cadann_metrics["rmse"]),
                    use_target_task_loss=True,
                )

                vanilla_mae_runs.append(float(vanilla_metrics["mae"]))
                vanilla_mse_runs.append(float(vanilla_metrics["mse"]))
                dann_mae_runs.append(float(dann_metrics["mae"]))
                dann_mse_runs.append(float(dann_metrics["mse"]))
                kmm_mae_runs.append(float(kmm_metrics["mae"]))
                kmm_mse_runs.append(float(kmm_metrics["mse"]))
                ca_mae_runs.append(float(ca_metrics["mae"]))
                ca_mse_runs.append(float(ca_metrics["mse"]))
                cadann_mae_runs.append(float(cadann_metrics["mae"]))
                cadann_mse_runs.append(float(cadann_metrics["mse"]))

            vanilla_mae_runs = np.asarray(vanilla_mae_runs, dtype=np.float64)
            vanilla_mse_runs = np.asarray(vanilla_mse_runs, dtype=np.float64)
            dann_mae_runs = np.asarray(dann_mae_runs, dtype=np.float64)
            dann_mse_runs = np.asarray(dann_mse_runs, dtype=np.float64)
            kmm_mae_runs = np.asarray(kmm_mae_runs, dtype=np.float64)
            kmm_mse_runs = np.asarray(kmm_mse_runs, dtype=np.float64)
            ca_mae_runs = np.asarray(ca_mae_runs, dtype=np.float64)
            ca_mse_runs = np.asarray(ca_mse_runs, dtype=np.float64)
            cadann_mae_runs = np.asarray(cadann_mae_runs, dtype=np.float64)
            cadann_mse_runs = np.asarray(cadann_mse_runs, dtype=np.float64)

            v_mae_mean, v_mae_std = float(np.mean(vanilla_mae_runs)), float(np.std(vanilla_mae_runs, ddof=1))
            d_mae_mean, d_mae_std = float(np.mean(dann_mae_runs)), float(np.std(dann_mae_runs, ddof=1))
            k_mae_mean, k_mae_std = float(np.mean(kmm_mae_runs)), float(np.std(kmm_mae_runs, ddof=1))
            c_mae_mean, c_mae_std = float(np.mean(ca_mae_runs)), float(np.std(ca_mae_runs, ddof=1))
            cd_mae_mean, cd_mae_std = float(np.mean(cadann_mae_runs)), float(np.std(cadann_mae_runs, ddof=1))
            v_mse_mean, v_mse_std = float(np.mean(vanilla_mse_runs)), float(np.std(vanilla_mse_runs, ddof=1))
            d_mse_mean, d_mse_std = float(np.mean(dann_mse_runs)), float(np.std(dann_mse_runs, ddof=1))
            k_mse_mean, k_mse_std = float(np.mean(kmm_mse_runs)), float(np.std(kmm_mse_runs, ddof=1))
            c_mse_mean, c_mse_std = float(np.mean(ca_mse_runs)), float(np.std(ca_mse_runs, ddof=1))
            cd_mse_mean, cd_mse_std = float(np.mean(cadann_mse_runs)), float(np.std(cadann_mse_runs, ddof=1))
            delta_mae_mean = float(np.mean(vanilla_mae_runs - dann_mae_runs))
            delta_mse_mean = float(np.mean(vanilla_mse_runs - dann_mse_runs))

            p_wilc_mae = float("nan")
            p_wilc_mse = float("nan")
            if _has_wilcoxon:
                try:
                    _, p_wilc_mae = wilcoxon_signed_rank(vanilla_mae_runs, dann_mae_runs, alternative="greater", zero_method="wilcox")
                    p_wilc_mae = float(p_wilc_mae)
                except Exception:
                    p_wilc_mae = float("nan")
                try:
                    _, p_wilc_mse = wilcoxon_signed_rank(vanilla_mse_runs, dann_mse_runs, alternative="greater", zero_method="wilcox")
                    p_wilc_mse = float(p_wilc_mse)
                except Exception:
                    p_wilc_mse = float("nan")

            frac_results[frac] = {
                "n": n,
                "vanilla_mae_mean": v_mae_mean,
                "vanilla_mae_std": v_mae_std,
                "dann_mae_mean": d_mae_mean,
                "dann_mae_std": d_mae_std,
                "kmm_mae_mean": k_mae_mean,
                "kmm_mae_std": k_mae_std,
                "climate_aware_mae_mean": c_mae_mean,
                "climate_aware_mae_std": c_mae_std,
                "climate_aware_dann_mae_mean": cd_mae_mean,
                "climate_aware_dann_mae_std": cd_mae_std,
                "vanilla_mse_mean": v_mse_mean,
                "vanilla_mse_std": v_mse_std,
                "dann_mse_mean": d_mse_mean,
                "dann_mse_std": d_mse_std,
                "kmm_mse_mean": k_mse_mean,
                "kmm_mse_std": k_mse_std,
                "climate_aware_mse_mean": c_mse_mean,
                "climate_aware_mse_std": c_mse_std,
                "climate_aware_dann_mse_mean": cd_mse_mean,
                "climate_aware_dann_mse_std": cd_mse_std,
                "delta_mae_mean": delta_mae_mean,
                "delta_mse_mean": delta_mse_mean,
                "p_wilcoxon_mae": p_wilc_mae,
                "p_wilcoxon_mse": p_wilc_mse,
            }
            log_row(
                experiment="low_resource",
                phase="summary",
                method="paired_summary",
                input_len=INPUT_LEN,
                frac_target=frac,
                n_target=n,
                n_repeats=n_repeats,
                vanilla_mae_mean=v_mae_mean,
                vanilla_mae_std=v_mae_std,
                vanilla_mse_mean=v_mse_mean,
                vanilla_mse_std=v_mse_std,
                dann_mae_mean=d_mae_mean,
                dann_mae_std=d_mae_std,
                dann_mse_mean=d_mse_mean,
                dann_mse_std=d_mse_std,
                kmm_mae_mean=k_mae_mean,
                kmm_mae_std=k_mae_std,
                kmm_mse_mean=k_mse_mean,
                kmm_mse_std=k_mse_std,
                climate_aware_mae_mean=c_mae_mean,
                climate_aware_mae_std=c_mae_std,
                climate_aware_mse_mean=c_mse_mean,
                climate_aware_mse_std=c_mse_std,
                climate_aware_dann_mae_mean=cd_mae_mean,
                climate_aware_dann_mae_std=cd_mae_std,
                climate_aware_dann_mse_mean=cd_mse_mean,
                climate_aware_dann_mse_std=cd_mse_std,
                delta_mae_mean=delta_mae_mean,
                delta_mse_mean=delta_mse_mean,
                wilcoxon_p_mae=p_wilc_mae,
                wilcoxon_p_mse=p_wilc_mse,
            )

            print(
                f"  Target {int(frac*100):>2d}% (n={n:>5d})  "
                f"Vanilla MAE={v_mae_mean:.4f}±{v_mae_std:.4f}  "
                f"DANN MAE={d_mae_mean:.4f}±{d_mae_std:.4f}  "
                f"KMM MAE={k_mae_mean:.4f}±{k_mae_std:.4f}  "
                f"CA MAE={c_mae_mean:.4f}±{c_mae_std:.4f}  "
                f"CA+DANN MAE={cd_mae_mean:.4f}±{cd_mae_std:.4f}  "
                f"Δ(V-DANN)={delta_mae_mean:+.4f}"
                + (
                    f"  Wilcoxon p(MAE)={p_wilc_mae:.3g} p(MSE)={p_wilc_mse:.3g}"
                    if _has_wilcoxon
                    else "  Wilcoxon p=NA (scipy not installed)"
                )
            )

        if frac_results:
            print("\nLow-resource summary (Papua test metrics):")
            print("  INPUT_LEN  %Target   nTarget   Vanilla MAE±std        DANN MAE±std         KMM MAE±std          CA MAE±std          CA+DANN MAE±std        p(V>DANN MAE)")
            for frac in low_resource_fracs:
                if frac not in frac_results:
                    continue
                r = frac_results[frac]
                pv_mae = r["p_wilcoxon_mae"]
                pv_mse = r["p_wilcoxon_mse"]
                pv_mae_txt = f"{pv_mae:.3g}" if (not np.isnan(pv_mae)) else ("NA" if not _has_wilcoxon else "nan")
                print(
                    f"  {INPUT_LEN:>8d}  {int(frac*100):>3d}%   {r['n']:>7d}   "
                    f"{r['vanilla_mae_mean']:>7.4f}±{r['vanilla_mae_std']:<7.4f}   "
                    f"{r['dann_mae_mean']:>7.4f}±{r['dann_mae_std']:<7.4f}   "
                    f"{r['kmm_mae_mean']:>7.4f}±{r['kmm_mae_std']:<7.4f}   "
                    f"{r['climate_aware_mae_mean']:>7.4f}±{r['climate_aware_mae_std']:<7.4f}   "
                    f"{r['climate_aware_dann_mae_mean']:>7.4f}±{r['climate_aware_dann_mae_std']:<7.4f}   "
                    f"{pv_mae_txt:>10s}"
                )

            try:
                import matplotlib.pyplot as plt
                xs = [int(fr * 100) for fr in low_resource_fracs if fr in frac_results]
                mv = [frac_results[fr]["vanilla_mae_mean"] for fr in low_resource_fracs if fr in frac_results]
                md = [frac_results[fr]["dann_mae_mean"] for fr in low_resource_fracs if fr in frac_results]
                mk = [frac_results[fr]["kmm_mae_mean"] for fr in low_resource_fracs if fr in frac_results]
                mc = [frac_results[fr]["climate_aware_mae_mean"] for fr in low_resource_fracs if fr in frac_results]
                mcd = [frac_results[fr]["climate_aware_dann_mae_mean"] for fr in low_resource_fracs if fr in frac_results]
                plt.figure(figsize=(8, 4))
                plt.plot(xs, mv, marker="o", label="Vanilla (Papua)")
                plt.plot(xs, md, marker="o", label="DANN (Java→Papua)")
                plt.plot(xs, mk, marker="o", label="KMM (Java→Papua)")
                plt.plot(xs, mc, marker="o", label="Climate-Aware (Papua)")
                plt.plot(xs, mcd, marker="o", label="Climate-Aware + DANN")
                plt.xlabel("% target (Papua) train windows")
                plt.ylabel("Papua Test MAE (mean over repeats)")
                plt.title(f"Low-resource target: MAE vs % target data (INPUT_LEN={INPUT_LEN}, repeats={n_repeats})")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                out_path = f"single_station_low_resource_mae_len{INPUT_LEN}.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"\nSaved plot: {out_path}")
            except Exception as e:
                print(f"\nPlot skipped (matplotlib not available): {e}")

    INPUT_LEN = original_input_len

    print("\n" + "=" * 60)
    print("PART 6 — FINAL SUMMARY TABLE (Papua test MAE)")
    print("=" * 60)
    print("Experiment        |   ARIMA |  Vanilla |    DANN |     KMM |      CA |  CA+DANN")
    print("-" * 75)
    def _fmt(x):
        return f"{x:7.3f}" if (x is not None and not np.isnan(x)) else "   nan"
    print(
        "Single Station    | "
        f"{_fmt(res_single['arima']['mae'])} | {_fmt(res_single['vanilla']['mae'])} | {_fmt(res_single['dann']['mae'])} | "
        f"{_fmt(res_single['kmm']['mae'])} | {_fmt(res_single['climate_aware']['mae'])} | {_fmt(res_single['climate_aware_dann']['mae'])}"
    )
    print(
        "Three Stations    | "
        f"{_fmt(res_three['arima']['mae'])} | {_fmt(res_three['vanilla']['mae'])} | {_fmt(res_three['dann']['mae'])} | "
        f"{_fmt(res_three['kmm']['mae'])} | {_fmt(res_three['climate_aware']['mae'])} | {_fmt(res_three['climate_aware_dann']['mae'])}"
    )
    print("-" * 75)
    # Also report MSE summary (same experiments)
    print("\n" + "-" * 60)
    print("Experiment        | ARIMA MSE       | Vanilla MSE     | DANN MSE       | KMM MSE        | CA MSE         | CA+DANN MSE")
    print("-" * 60)
    print(
        "Single Station    | "
        f"{res_single['arima']['mse']:13.6f} | {res_single['vanilla']['mse']:13.6f} | {res_single['dann']['mse']:13.6f} | "
        f"{res_single['kmm']['mse']:13.6f} | {res_single['climate_aware']['mse']:13.6f} | {res_single['climate_aware_dann']['mse']:13.6f}"
    )
    print(
        "Three Stations    | "
        f"{res_three['arima']['mse']:13.6f} | {res_three['vanilla']['mse']:13.6f} | {res_three['dann']['mse']:13.6f} | "
        f"{res_three['kmm']['mse']:13.6f} | {res_three['climate_aware']['mse']:13.6f} | {res_three['climate_aware_dann']['mse']:13.6f}"
    )
    print("-" * 60)
    print("-------------------------------------------")

    # -----------------------------
    # Save results to CSV
    # -----------------------------
    try:
        out_dir = os.path.dirname(__file__)
        safe_ts = run_ts.replace(":", "").replace("-", "").replace("+", "").replace("T", "_")
        out_csv = os.path.join(out_dir, f"single_station_results_{safe_ts}.csv")
        df_out = pd.DataFrame(results_rows)
        # Put some common columns first if they exist
        first_cols = [
            "run_timestamp",
            "experiment",
            "phase",
            "method",
            "INPUT_LEN",
            "input_len",
            "frac_target",
            "n_target",
            "rep_idx",
            "seed",
            "metric_mae",
            "metric_mse",
            "metric_rmse",
        ]
        cols = [c for c in first_cols if c in df_out.columns] + [c for c in df_out.columns if c not in first_cols]
        df_out = df_out[cols]
        df_out.to_csv(out_csv, index=False)
        print(f"\nSaved results CSV: {out_csv}")
    except Exception as e:
        print(f"\nFailed to save results CSV: {e}")


if __name__ == "__main__":
    import torch
    main()
