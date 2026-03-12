import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# EFFICIENCY: CPU-only settings
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import math
import random

import torch
torch.set_num_threads(4)
torch.set_default_dtype(torch.float32)
DEVICE = torch.device("cpu")

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


def eval_model_mae(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
    if X.shape[0] == 0:
        return float("nan")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(DEVICE)
            preds.append(model(xb).cpu().numpy().astype(np.float32))
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



# ARIMA BASELINE (per-station on train, rolling 7-day forecast on test)
def arima_forecast_one_series(series, horizon=7, max_end=None):
    """Fit ARIMA with (p,d,q) in [0,2], pick lowest AIC; then forecast horizon steps."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return None

    if max_end is not None and len(series) > max_end:
        series = series[:max_end]
    if len(series) < 30:
        return None
    series = np.asarray(series, dtype=np.float64)
    # Simple differencing for d
    d = 0
    try:
        adf = adfuller(series, maxlag=1, autolag=None)
        if adf[1] > 0.05:
            d = 1
    except Exception:
        d = 0

    best_aic, best_model = np.inf, None
    for p in range(3):
        for q in range(3):
            try:
                m = ARIMA(series, order=(p, d, q))
                fitted = m.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
            except Exception:
                continue
    if best_model is None:
        return None
    try:
        f = best_model.forecast(steps=horizon)
        return np.asarray(f, dtype=np.float32)
    except Exception:
        return None


def arima_baseline_rolling(test_df, train_df, horizon=7):
    """Per station: fit on train, rolling 7-day forecast on test. Return MAE, RMSE (averaged)."""
    from statsmodels.tsa.arima.model import ARIMA

    test_df = test_df.sort_values(["location_id", "time"])
    locations = test_df["location_id"].unique()
    all_mae, all_rmse = [], []

    for lid in locations:
        train_ser = train_df[train_df["location_id"] == lid].sort_values("time")[TARGET_COL]
        test_grp = test_df[test_df["location_id"] == lid].sort_values("time")
        if len(train_ser) < 30 or len(test_grp) < horizon:
            continue
        train_ser = train_ser.values.astype(np.float64)
        test_vals = test_grp[TARGET_COL].values.astype(np.float32)
        preds = []
        for i in range(0, len(test_vals) - horizon + 1):
            hist = np.concatenate([train_ser, test_vals[:i]]) if i > 0 else train_ser
            f = arima_forecast_one_series(hist, horizon=horizon)
            if f is None:
                break
            preds.append(f)
        if not preds:
            continue
        preds = np.array(preds, dtype=np.float32)
        n_steps = min(len(preds), (len(test_vals) - horizon + 1))
        y_true = np.array([test_vals[i : i + horizon] for i in range(n_steps)], dtype=np.float32)
        y_pred = preds[:n_steps]
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        all_mae.append(mae)
        all_rmse.append(rmse)

    if not all_mae:
        return float("nan"), float("nan")
    return np.mean(all_mae), np.mean(all_rmse)


## vanilla transformer -> Train Using All of the data without the tokyo-osaka
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
            opt.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            tr_mse += loss.item() * xb.size(0)
        tr_mse /= max(1, X_train.shape[0])

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


def train_transformer_weighted(X_train, y_train, X_val, y_val, X_test, y_test, weights,
                               epochs=20, batch_size=64, lr=1e-3, patience=5):
    """Train Transformer with weighted MSE: mean(weights * (pred - target)^2)."""
    from torch.utils.data import TensorDataset, DataLoader

    w = torch.from_numpy(weights).to(DEVICE).float()
    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(DEVICE),
        torch.from_numpy(y_train).to(DEVICE),
        w[: len(X_train)].clone(),
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
        for xb, yb, wb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            sq = (pred - yb) ** 2
            loss = (wb.unsqueeze(1) * sq).mean()
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
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).to(DEVICE)
        pred = model(Xt).cpu().numpy()
    mae = np.mean(np.abs(pred - y_test))
    return mae


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

    print("=" * 60)
    print("PART 1 — DATA PREPROCESSING")
    print("=" * 60)
    train, val, test = load_and_split(data_path)
    print(f"Train: {train['time'].min()} to {train['time'].max()}  rows: {len(train)}")
    print(f"Val:   {val['time'].min()} to {val['time'].max()}  rows: {len(val)}")
    print(f"Test:  {test['time'].min()} to {test['time'].max()}  rows: {len(test)}")

    print("\n" + "=" * 60)
    print("PART 2 — SLIDING WINDOW")
    print("=" * 60)
    X_train, y_train, X_val, y_val, X_test, y_test = build_all_windows(train, val, test)
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}  y_test:  {y_test.shape}")

    # Target-only feature windows: keep only the target column as input feature
    tgt_feat_idx = FEATURE_COLS.index(TARGET_COL)
    X_train_target_only = X_train[:, :, [tgt_feat_idx]]
    X_val_target_only = X_val[:, :, [tgt_feat_idx]]
    X_test_target_only = X_test[:, :, [tgt_feat_idx]]

    # Papua-only splits and windows (for region-specific baselines)
    train_papua_all = train[region_mask(train, "papua")]
    val_papua_all = val[region_mask(val, "papua")]
    test_papua_all = test[region_mask(test, "papua")]
    X_train_papua, y_train_papua, X_val_papua, y_val_papua, X_test_papua, y_test_papua = build_all_windows(
        train_papua_all, val_papua_all, test_papua_all
    )
    print(f"Papua X_train: {X_train_papua.shape}  y_train: {y_train_papua.shape}")
    print(f"Papua X_val:   {X_val_papua.shape}   y_val:   {y_val_papua.shape}")
    print(f"Papua X_test:  {X_test_papua.shape}  y_test:  {y_test_papua.shape}")

    # Target-only windows restricted to Papua region
    X_train_papua_target_only = X_train_papua[:, :, [tgt_feat_idx]]
    X_val_papua_target_only = X_val_papua[:, :, [tgt_feat_idx]]
    X_test_papua_target_only = X_test_papua[:, :, [tgt_feat_idx]]

    # print("\n" + "=" * 60)
    # print("PART 3 — ARIMA BASELINE (Papua only)")
    # print("=" * 60)
    # train_papua_arima = train_papua_all
    # test_papua_arima = test_papua_all
    # arima_papua_mae, arima_papua_rmse = arima_baseline_rolling(test_papua_arima, train_papua_arima, horizon=HORIZON)
    # print(f"ARIMA Papua MAE:  {arima_papua_mae:.4f}")
    # print(f"ARIMA Papua RMSE: {arima_papua_rmse:.4f}")
    print("=" * 60)
    print("Java -> Papua")

    print("\n" + "=" * 60)
    print("PART 4 — VANILLA TRANSFORMER (NO DA)")
    print("=" * 60)
    model_vanilla, vanilla_mae, vanilla_rmse = train_eval_transformer(
        X_train_papua,
        y_train_papua,
        X_val_papua,
        y_val_papua,
        X_test_papua,
        y_test_papua,
        epochs=20, batch_size=64, lr=1e-3, patience=5,
    )
    print(f"Vanilla Transformer (Papua) Test MAE:  {vanilla_mae:.4f}")
    print(f"Vanilla Transformer (Papua) Test RMSE: {vanilla_rmse:.4f}")

    # Vanilla Transformer trained with only target variable as input feature
    model_vanilla_target_only, vanilla_mae_target_only, vanilla_rmse_target_only = train_eval_transformer(
        X_train_papua_target_only,
        y_train_papua,
        X_val_papua_target_only,
        y_val_papua,
        X_test_papua_target_only,
        y_test_papua,
        epochs=20,
        batch_size=64,
        lr=1e-3,
        patience=5,
    )
    print(f"Vanilla Transformer (Papua, target-only) Test MAE:  {vanilla_mae_target_only:.4f}")
    print(f"Vanilla Transformer (Papua, target-only) Test RMSE: {vanilla_rmse_target_only:.4f}")

    print("\n" + "=" * 60)
    print("PART 5 — KMM DOMAIN ADAPTATION (Java → Papua)")
    print("=" * 60)
    train_java = train[train["region"].str.strip().str.lower() == "java"]
    test_papua = test[test["region"].str.strip().str.lower() == "papua"]
    X_java, y_java, _, _, X_papua_test, y_papua_test = build_all_windows(
        train_java, train_java.iloc[0:0], test_papua
    )
    if X_java.shape[0] == 0 or X_papua_test.shape[0] == 0:
        print("Not enough Java train or Papua test windows; skipping KMM.")
        kmm_mae = float("nan")
        kmm_mae_target_only = float("nan")
    else:
        X_src_flat = X_java.reshape(X_java.shape[0], -1)
        X_tgt_flat = X_papua_test.reshape(X_papua_test.shape[0], -1)
        subsample_src = min(3000, X_src_flat.shape[0])
        subsample_tgt = min(1500, X_tgt_flat.shape[0])
        idx_s = np.random.RandomState(42).choice(X_src_flat.shape[0], subsample_src, replace=False)
        idx_t = np.random.RandomState(43).choice(X_tgt_flat.shape[0], subsample_tgt, replace=False)
        beta = kmm_weights(X_src_flat[idx_s], X_tgt_flat[idx_t], B=2.0, eps=0.1)
        # Train on subsampled Java with KMM weights; use rest of Java as val for early stopping
        mask = np.ones(X_java.shape[0], dtype=bool)
        mask[idx_s] = False
        X_val_java = X_java[mask]
        y_val_java = y_java[mask]
        if X_val_java.shape[0] < 100:
            X_val_java = np.concatenate([X_val_java, X_java[idx_s[:200]]], axis=0)
            y_val_java = np.concatenate([y_val_java, y_java[idx_s[:200]]], axis=0)
        kmm_mae = train_transformer_weighted(
            X_java[idx_s], y_java[idx_s], X_val_java, y_val_java,
            X_papua_test, y_papua_test, beta,
            epochs=20, batch_size=64, lr=1e-3, patience=5,
        )
        # Target-only KMM scenario: only use the target variable as input feature
        X_java_target_only = X_java[:, :, [tgt_feat_idx]]
        X_papua_test_target_only = X_papua_test[:, :, [tgt_feat_idx]]
        X_java_t_flat = X_java_target_only.reshape(X_java_target_only.shape[0], -1)
        X_papua_t_flat = X_papua_test_target_only.reshape(X_papua_test_target_only.shape[0], -1)
        subsample_src_t = min(3000, X_java_t_flat.shape[0])
        subsample_tgt_t = min(1500, X_papua_t_flat.shape[0])
        idx_s_t = np.random.RandomState(44).choice(X_java_t_flat.shape[0], subsample_src_t, replace=False)
        idx_t_t = np.random.RandomState(45).choice(X_papua_t_flat.shape[0], subsample_tgt_t, replace=False)
        beta_target_only = kmm_weights(X_java_t_flat[idx_s_t], X_papua_t_flat[idx_t_t], B=2.0, eps=0.1)
        mask_t = np.ones(X_java_target_only.shape[0], dtype=bool)
        mask_t[idx_s_t] = False
        X_val_java_t = X_java_target_only[mask_t]
        y_val_java_t = y_java[mask_t]
        if X_val_java_t.shape[0] < 100:
            X_val_java_t = np.concatenate([X_val_java_t, X_java_target_only[idx_s_t[:200]]], axis=0)
            y_val_java_t = np.concatenate([y_val_java_t, y_java[idx_s_t[:200]]], axis=0)
        kmm_mae_target_only = train_transformer_weighted(
            X_java_target_only[idx_s_t],
            y_java[idx_s_t],
            X_val_java_t,
            y_val_java_t,
            X_papua_test_target_only,
            y_papua_test,
            beta_target_only,
            epochs=20,
            batch_size=64,
            lr=1e-3,
            patience=5,
        )
    vanilla_papua_mae = float("nan")
    vanilla_papua_mae_target_only = float("nan")
    if X_papua_test.shape[0] > 0:
        with torch.no_grad():
            vanilla_papua_mae = np.mean(np.abs(
                model_vanilla(torch.from_numpy(X_papua_test).to(DEVICE)).cpu().numpy() - y_papua_test
            ))
            # Evaluate vanilla Transformer trained on target-only features
            vanilla_papua_mae_target_only = np.mean(
                np.abs(
                    model_vanilla_target_only(
                        torch.from_numpy(X_papua_test_target_only).to(DEVICE)
                    ).cpu().numpy()
                    - y_papua_test
                )
            )
    print(f"Vanilla Transformer Papua MAE: {vanilla_papua_mae:.4f}")
    print(f"KMM Transformer Papua MAE:     {kmm_mae:.4f}")
    print(f"Vanilla Transformer (target-only) Papua MAE: {vanilla_papua_mae_target_only:.4f}")
    print(f"KMM Transformer (target-only) Papua MAE:     {kmm_mae_target_only:.4f}")
    if not np.isnan(vanilla_papua_mae) and not np.isnan(kmm_mae) and vanilla_papua_mae > 0:
        imp = (1 - kmm_mae / vanilla_papua_mae) * 100
        print(f"Improvement %: {imp:.2f}%")

    print("\n" + "=" * 60)
    print("PART 6 — ADVERSARIAL DOMAIN ADAPTATION (DANN) Java → Papua")
    print("=" * 60)
    train_papua = train[region_mask(train, "papua")]
    val_papua = val[region_mask(val, "papua")]
    X_src, y_src = build_windows(train_java)
    X_tgt, y_tgt = build_windows(train_papua)
    X_tgt_val, y_tgt_val = build_windows(val_papua)
    X_tgt_test, y_tgt_test = build_windows(test_papua)

    # Target-only windows for DANN experiments
    if X_src.shape[0] > 0:
        X_src_target_only = X_src[:, :, [tgt_feat_idx]]
    else:
        X_src_target_only = X_src
    if X_tgt.shape[0] > 0:
        X_tgt_target_only = X_tgt[:, :, [tgt_feat_idx]]
    else:
        X_tgt_target_only = X_tgt
    if X_tgt_val.shape[0] > 0:
        X_tgt_val_target_only = X_tgt_val[:, :, [tgt_feat_idx]]
    else:
        X_tgt_val_target_only = X_tgt_val
    if X_tgt_test.shape[0] > 0:
        X_tgt_test_target_only = X_tgt_test[:, :, [tgt_feat_idx]]
    else:
        X_tgt_test_target_only = X_tgt_test

    if X_src.shape[0] == 0 or X_tgt.shape[0] == 0 or X_tgt_val.shape[0] == 0 or X_tgt_test.shape[0] == 0:
        print("Not enough Java/Papua windows for DANN; skipping.")
        dann_papua_mae = float("nan")
        dann_papua_mae_target_only = float("nan")
    else:
        feat = FeatureExtractor(input_dim=X_src.shape[2], d_model=32, nhead=4, num_layers=2, dropout=0.1)
        task_head = nn.Linear(32, HORIZON)
        dom = DomainClassifier(in_dim=32)
        dann_model, _ = train_domain_adversarial_dann(
            feat,
            task_head,
            dom,
            X_src,
            y_src,
            X_tgt,
            y_tgt,
            X_tgt_val,
            y_tgt_val,
            epochs=25,
            batch_size=64,
            lr=1e-3,
            patience=5,
            use_target_task_loss=False,
        )
        dann_papua_mae = eval_model_mae(dann_model, X_tgt_test, y_tgt_test, batch_size=256)
        # DANN scenario with only the target variable as input feature
        feat_t = FeatureExtractor(input_dim=X_src_target_only.shape[2], d_model=32, nhead=4, num_layers=2, dropout=0.1)
        task_head_t = nn.Linear(32, HORIZON)
        dom_t = DomainClassifier(in_dim=32)
        dann_model_target_only, _ = train_domain_adversarial_dann(
            feat_t,
            task_head_t,
            dom_t,
            X_src_target_only,
            y_src,
            X_tgt_target_only,
            y_tgt,
            X_tgt_val_target_only,
            y_tgt_val,
            epochs=25,
            batch_size=64,
            lr=1e-3,
            patience=5,
            use_target_task_loss=False,
        )
        dann_papua_mae_target_only = eval_model_mae(
            dann_model_target_only, X_tgt_test_target_only, y_tgt_test, batch_size=256
        )
    print(f"DANN Transformer Papua Test MAE: {dann_papua_mae:.4f}")
    print(f"DANN Transformer (target-only) Papua Test MAE: {dann_papua_mae_target_only:.4f}")

    print("\n" + "=" * 60)
    """REMOVE"""
    print("PART 7 — CLIMATE-AWARE DOMAIN ADAPTATION")
    print("=" * 60)
    if X_src.shape[0] == 0 or X_tgt_val.shape[0] == 0 or X_tgt_test.shape[0] == 0:
        climate_papua_mae = float("nan")
        climate_dann_papua_mae = float("nan")
        target_only_papua_mae = float("nan")
        print("Not enough windows for climate-aware experiments; skipping.")
    else:
        climate_model = ClimateAwareTransformer(horizon=HORIZON, d_model=32, nhead=4, num_layers=2, dropout=0.1)
        climate_model, _ = train_supervised_earlystop_target_mae(
            climate_model,
            X_src,
            y_src,
            X_tgt_val,
            y_tgt_val,
            epochs=25,
            batch_size=64,
            lr=1e-3,
            patience=5,
        )
        climate_papua_mae = eval_model_mae(climate_model, X_tgt_test, y_tgt_test, batch_size=256)

        # Target-only supervised Transformer using same training/validation splits
        target_only_model = VanillaTransformer(
            input_dim=X_src_target_only.shape[2],
            d_model=32,
            nhead=4,
            num_layers=2,
            dropout=0.1,
            horizon=HORIZON,
        )
        target_only_model, _ = train_supervised_earlystop_target_mae(
            target_only_model,
            X_src_target_only,
            y_src,
            X_tgt_val_target_only,
            y_tgt_val,
            epochs=25,
            batch_size=64,
            lr=1e-3,
            patience=5,
        )
        target_only_papua_mae = eval_model_mae(
            target_only_model, X_tgt_test_target_only, y_tgt_test, batch_size=256
        )

        ca_feat = ClimateAwareRep(d_model=32, nhead=4, num_layers=2, dropout=0.1)
        ca_task = nn.Linear(64, HORIZON)
        ca_dom = DomainClassifier(in_dim=64)
        ca_dann_model, _ = train_domain_adversarial_dann(
            ca_feat,
            ca_task,
            ca_dom,
            X_src,
            y_src,
            X_tgt,
            y_tgt,
            X_tgt_val,
            y_tgt_val,
            epochs=25,
            batch_size=64,
            lr=1e-3,
            patience=5,
            use_target_task_loss=False,
        )
        climate_dann_papua_mae = eval_model_mae(ca_dann_model, X_tgt_test, y_tgt_test, batch_size=256)

    print(f"Climate-aware Transformer Papua Test MAE:  {climate_papua_mae:.4f}")
    print(f"Climate-aware + DANN Papua Test MAE:       {climate_dann_papua_mae:.4f}")
    print(f"Target-only Transformer Papua Test MAE:    {target_only_papua_mae:.4f}")
    if X_tgt_test.shape[0] > 0 and not np.isnan(climate_papua_mae):
        mask_enso = enso_extreme_mask(X_tgt_test, thresh_abs_nino=1.0)
        if mask_enso.sum() > 0:
            vanilla_model = VanillaTransformer(
                input_dim=X_train.shape[2], d_model=32, nhead=4, num_layers=2,
                dropout=0.1, horizon=HORIZON,
            ).to(DEVICE)
            mae_v_ext = eval_model_mae_masked(vanilla_model, X_tgt_test, y_tgt_test, mask_enso, batch_size=256)
            mae_c_ext = eval_model_mae_masked(climate_model, X_tgt_test, y_tgt_test, mask_enso, batch_size=256)
            print(f"Extreme ENSO windows (|Nino|>=1.0) count={int(mask_enso.sum())}:")
            print(f"  Vanilla MAE:        {mae_v_ext:.4f}")
            print(f"  Climate-aware MAE:  {mae_c_ext:.4f}")

    print("\n" + "=" * 60)
    print("PART 8 — LOW-RESOURCE TARGET EXPERIMENT")
    print("=" * 60)
    low_resource_fracs = [0.05, 0.15, 0.25]
    low_resource_rows = []
    
    if X_src.shape[0] == 0 or X_tgt.shape[0] == 0 or X_tgt_val.shape[0] == 0 or X_tgt_test.shape[0] == 0:
        print("Not enough windows for low-resource experiment; skipping.")
    else:
        rs = np.random.RandomState(123)
        for frac in low_resource_fracs:
            n = max(1, int(round(frac * X_tgt.shape[0])))
            idx = rs.choice(X_tgt.shape[0], n, replace=False)
            X_tgt_small = X_tgt[idx]
            y_tgt_small = y_tgt[idx]

            # (a) Vanilla Transformer (Papua-only, supervised)
            X_mix = X_tgt_small
            y_mix = y_tgt_small
            vanilla_lr = VanillaTransformer(
                input_dim=X_mix.shape[2],
                d_model=32,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                horizon=HORIZON,
            )
            vanilla_lr, _ = train_supervised_earlystop_target_mae(
                vanilla_lr, X_mix, y_mix, X_tgt_val, y_tgt_val, epochs=25, batch_size=64, lr=1e-3, patience=5
            )
            mae_v = eval_model_mae(vanilla_lr, X_tgt_test, y_tgt_test, batch_size=256)

            # (a-tgt) Vanilla Transformer using only target variable as input feature
            X_tgt_small_target_only = X_tgt_target_only[idx]
            X_mix_target_only = X_tgt_small_target_only
            vanilla_lr_target_only = VanillaTransformer(
                input_dim=X_mix_target_only.shape[2],
                d_model=32,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                horizon=HORIZON,
            )
            vanilla_lr_target_only, _ = train_supervised_earlystop_target_mae(
                vanilla_lr_target_only,
                X_mix_target_only,
                y_mix,
                X_tgt_val_target_only,
                y_tgt_val,
                epochs=25,
                batch_size=64,
                lr=1e-3,
                patience=5,
            )
            mae_v_target_only = eval_model_mae(
                vanilla_lr_target_only, X_tgt_test_target_only, y_tgt_test, batch_size=256
            )

            # (b) DANN (Java + small Papua; include labeled target subset in task loss)
            feat_lr = FeatureExtractor(input_dim=X_src.shape[2], d_model=32, nhead=4, num_layers=2, dropout=0.1)
            task_lr = nn.Linear(32, HORIZON)
            dom_lr = DomainClassifier(in_dim=32)
            dann_lr_model, _ = train_domain_adversarial_dann(
                feat_lr,
                task_lr,
                dom_lr,
                X_src,
                y_src,
                X_tgt_small,
                y_tgt_small,
                X_tgt_val,
                y_tgt_val,
                epochs=25,
                batch_size=64,
                lr=1e-3,
                patience=5,
                use_target_task_loss=True,
            )
            mae_d = eval_model_mae(dann_lr_model, X_tgt_test, y_tgt_test, batch_size=256)

            # (c) Climate-aware DA (CA-DANN; include labeled target subset in task loss)
            ca_feat_lr = ClimateAwareRep(d_model=32, nhead=4, num_layers=2, dropout=0.1)
            ca_task_lr = nn.Linear(64, HORIZON)
            ca_dom_lr = DomainClassifier(in_dim=64)
            ca_dann_lr_model, _ = train_domain_adversarial_dann(
                ca_feat_lr,
                ca_task_lr,
                ca_dom_lr,
                X_src,
                y_src,
                X_tgt_small,
                y_tgt_small,
                X_tgt_val,
                y_tgt_val,
                epochs=25,
                batch_size=64,
                lr=1e-3,
                patience=5,
                use_target_task_loss=True,
            )
            mae_c = eval_model_mae(ca_dann_lr_model, X_tgt_test, y_tgt_test, batch_size=256)

            low_resource_rows.append((frac, n, mae_v, mae_d, mae_c))
            print(
                f"  Target {int(frac*100):>2d}% (n={n:>5d})  "
                # f"ARIMA={arima_papua_mae:.4f}  "
                f"Vanilla={mae_v:.4f}  DANN={mae_d:.4f}  ClimateAware-DA={mae_c:.4f}  "
                f"Vanilla-target-only={mae_v_target_only:.4f}"
            )

        if low_resource_rows:
            print("\nLow-resource results (Papua test MAE):")
            print("  %Target   nTarget   Vanilla     DANN   Δ(DANN)  %Imp(DANN)   ClimateAware-DA   Δ(CA-DA)  %Imp(CA-DA)")
            for frac, n, mv, md, mc in low_resource_rows:
                dd = mv - md
                dc = mv - mc
                pd = (1 - md / mv) * 100 if mv > 0 else float("nan")
                pc = (1 - mc / mv) * 100 if mv > 0 else float("nan")
                print(
                    f"  {int(frac*100):>3d}%   {n:>7d}   {mv:>8.4f}  {md:>8.4f}  {dd:>7.4f}   {pd:>8.2f}%"
                    f"     {mc:>14.4f}  {dc:>8.4f}   {pc:>8.2f}%"
                )

            try:
                import matplotlib.pyplot as plt

                xs = [int(fr * 100) for fr, *_ in low_resource_rows]
                mv = [r[2] for r in low_resource_rows]
                md = [r[3] for r in low_resource_rows]
                mc = [r[4] for r in low_resource_rows]
                plt.figure(figsize=(7, 4))
                plt.plot(xs, mv, marker="o", label="Vanilla")
                plt.plot(xs, md, marker="o", label="DANN")
                plt.plot(xs, mc, marker="o", label="ClimateAware-DA")
                plt.xlabel("% target (Papua) train windows")
                plt.ylabel("Papua Test MAE")
                plt.title("Low-resource target: MAE vs % target data")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                out_path = "low_resource_mae.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"\nSaved plot: {out_path}")
            except Exception as e:
                print(f"\nPlot skipped (matplotlib not available): {e}")

    print("\n" + "=" * 60)
    print("PART 9 — FINAL OUTPUT")
    print("=" * 60)
    # print(f"ARIMA Papua MAE:                  {arima_papua_mae:.4f}")
    print(f"Vanilla Transformer overall MAE: {vanilla_mae:.4f}")
    print(f"Vanilla Transformer Papua MAE:   {vanilla_papua_mae:.4f}")
    print(f"KMM Transformer Papua MAE:       {kmm_mae:.4f}")
    print(f"DANN Transformer Papua MAE:      {dann_papua_mae:.4f}")
    print(f"Climate-aware Papua MAE:         {climate_papua_mae:.4f}")
    print(f"Climate-aware + DANN Papua MAE:  {climate_dann_papua_mae:.4f}")
    print(f"Vanilla Transformer (target-only) overall MAE: {vanilla_mae_target_only:.4f}")
    print(f"Vanilla Transformer (target-only) Papua MAE:   {vanilla_papua_mae_target_only:.4f}")
    print(f"KMM Transformer (target-only) Papua MAE:       {kmm_mae_target_only:.4f}")
    print(f"DANN Transformer (target-only) Papua MAE:      {dann_papua_mae_target_only:.4f}")
    print(f"Target-only Papua MAE (supervised):            {target_only_papua_mae:.4f}")
    if not np.isnan(vanilla_papua_mae) and not np.isnan(kmm_mae) and vanilla_papua_mae > 0:
        rel_imp = (1 - kmm_mae / vanilla_papua_mae) * 100
        print(f"Relative improvement from KMM (on Papua): {rel_imp:.2f}%")


if __name__ == "__main__":
    main()
