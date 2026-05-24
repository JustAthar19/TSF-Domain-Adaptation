import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.data.scaling import fit_transform_source, transform_target
from src.data.windowing import build_windows
from src.models.one_step.mmd import compute_mmd, mmd_to_mix_pct, mix_source_into_target_df
from src.training.one_step import finetune_on_target, pretrain_source_model
from src.evaluation.one_step import osft_eval_model_metrics


def _chrono_split_windows(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.85,
):
    n = len(X)
    i = max(1, int(n * train_ratio))
    if i >= n:
        i = max(1, n - 1)
    return X[:i], y[:i], X[i:], y[i:]


def run_one_step_finetuning(
    source_train_df: pd.DataFrame,
    target_train_df: pd.DataFrame,
    target_val_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    input_len: int,
    horizon: int,
    stride: int,
    device: str,
    batch_size: int = 32,
    epochs_pretrain: int = 50,
    epochs_finetune: int = 35,
    lr_pretrain: float = 1e-3,
    lr_finetune: float = 5e-4,
    patience: int = 10,
    mmd_threshold: float = 0.11,
    mmd_subsample: int = 3000,
    d_model: int = 64,
    n_heads: int = 4,
    n_encoder_layers: int = 3,
    d_ff: int = 256,
    dropout: float = 0.1,
    mix_seed: int = 42,
    region_name: str = "target",
) -> Tuple[Dict, Dict]:
    """
    Full one-step fine-tuning pipeline:
      1. Scale with source-fitted StandardScaler
      2. Pre-train TSTransformer on source
      3. Compute MMD, mix source rows into target train
      4. Fine-tune with gradual unfreezing on mixed target data
      5. Evaluate on target test set
    """
    src_scaled, scaler = fit_transform_source(source_train_df, feature_cols)
    tgt_train_scaled = transform_target(target_train_df, feature_cols, scaler)
    tgt_val_scaled = transform_target(target_val_df, feature_cols, scaler)
    tgt_test_scaled = transform_target(target_test_df, feature_cols, scaler)

    X_src, y_src = build_windows(src_scaled, input_len, horizon, stride, feature_cols)
    X_src_tr, y_src_tr, X_src_val, y_src_val = _chrono_split_windows(X_src, y_src)

    print(f"  Source windows: train={X_src_tr.shape[0]}  val={X_src_val.shape[0]}")
    source_model = pretrain_source_model(
        X_src_tr,
        y_src_tr,
        X_src_val,
        y_src_val,
        n_features=len(feature_cols),
        input_len=input_len,
        horizon=horizon,
        epochs=epochs_pretrain,
        batch_size=max(batch_size, 64),
        lr=lr_pretrain,
        device=device,
        patience=patience,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        d_ff=d_ff,
        dropout=dropout,
    )

    mmd_val = compute_mmd(
        src_scaled[target_col].values,
        tgt_train_scaled[target_col].values,
        subsample=mmd_subsample,
        seed=mix_seed,
    )
    mix_pct = mmd_to_mix_pct(mmd_val, threshold=mmd_threshold)
    print(f"  MMD(source, {region_name}) = {mmd_val:.4f}  ->  mix_pct = {mix_pct * 100:.0f}%")

    mixed_train_df = mix_source_into_target_df(
        src_scaled, tgt_train_scaled, mix_pct, seed=mix_seed
    )
    X_mixed, y_mixed = build_windows(
        mixed_train_df, input_len, horizon, stride, feature_cols
    )
    X_val, y_val = build_windows(
        tgt_val_scaled, input_len, horizon, stride, feature_cols
    )
    X_test, y_test = build_windows(
        tgt_test_scaled, input_len, horizon, stride, feature_cols
    )

    ft_model = finetune_on_target(
        source_model,
        X_mixed,
        y_mixed,
        X_val,
        y_val,
        epochs=epochs_finetune,
        batch_size=batch_size,
        lr=lr_finetune,
        device=device,
        patience=patience,
        region_name=region_name,
    )

    metrics = osft_eval_model_metrics(ft_model, X_test, y_test, batch_size, device)
    meta = {"mmd": mmd_val, "mix_pct": mix_pct}
    return metrics, meta
