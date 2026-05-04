import pandas as pd
import numpy as np

import torch.nn as nn

from datetime import datetime

from src.data.stations import filter_df_by_station_ids
from src.data.windowing import stack_for_temperature, build_windows, build_windows_temp_transformer

from src.models.Vanilla.vanilla import VanillaTransformer
from src.models.DANN.domain_classifier import DomainClassifier
from src.models.DANN.feat_extractor import FeatureExtractor
from src.models.ATTF.attf import AttF
from src.models.DAF.daf import DAF
from src.models.DAF.domain_discriminator import DomainDiscriminator
from src.models.kmm_vu_tran.transformer import TemperatureTransformer
from src.models.kmm_vu_tran.kmm import kmm_weights
from src.models.TFT.tft import TFT_Target_Model
from src.models.TFT.tft_da import TFT_DA_Model


from src.training.arima_baseline import arima_baseline_rolling
from src.training.vu_tran_transformer import train_transformer_da, train_transformer_non_da
from src.training.attf import train_attf_earlystop_target_mae
from src.training.daf import train_daf_earlystop_target_mae
from src.training.transformer_weighted import train_transformer_weighted_kmm, train_vanilla_transformer_weighted
from src.training.tft import train_tft_non_da, train_tft_da

from src.evaluation.supervised import eval_model_metrics
from src.evaluation.attf import attf_eval_model_metrics
from src.evaluation.daf import daf_eval_model_metrics
from src.evaluation.kmm_vu_tran import eval_model_kmm_vu_tran_metrics
from src.evaluation.tft import  tft_eval_model_metrics_non_da, tft_eval_model_metrics_da

from src.utils.io import log_row
from src.utils.time import convert_seconds


def run_target_station_experiment(
    run_ts: datetime,
    result_rows: list,
    experiment_name: str,
    source_train_df: pd.DataFrame,
    target_train_df: pd.DataFrame,
    target_val_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    target_station_ids: list,
    input_len: int,
    horizon: int,
    stride: int,
    feature_cols: list,
    target_col: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str
):
    """
    Train multiple models for a given target-station subset and evaluate on FULL Papua test set:
      - Arima (Baseline | Non Domain Adaptation)
      - Vu-Tran Non Domain Adaptation
      - KMM - Vu Tran (Baseline | Non Domain Adaptation)
      - ATTF
      - DAF
      - TFT ( (Our Method) Baseline | Non Domain Adaptation)
      - TFT-DANN (Our Method | Domain Adaptation)
    Optionally includes a precomputed ARIMA baseline dict.
    """
    

    print("\n" + "=" * 60)
    print(f"{experiment_name}")
    print("=" * 60)
    print(f"Selected Papua target stations (train only): {', '.join([str(x) for x in target_station_ids])}")
    
    target_train_sel = filter_df_by_station_ids(target_train_df, target_station_ids)
    target_test_sel = filter_df_by_station_ids(target_test_df, target_station_ids)

    # Load Data
    X_src, y_src = build_windows(split_df=source_train_df, input_len=input_len, horizon=horizon, stride=stride, feature_cols=feature_cols)
    X_tgt, y_tgt = build_windows(split_df=target_train_df,input_len=input_len, horizon=horizon, stride=stride, feature_cols=feature_cols)
    X_tgt_val, y_tgt_val = build_windows(target_val_df, input_len=input_len, horizon=horizon, stride=stride, feature_cols=feature_cols)  # FULL Papua val
    X_tgt_test, y_tgt_test = build_windows(target_test_df, input_len=input_len, horizon=horizon, stride=stride, feature_cols=feature_cols)  # FULL Papua test
    
    # Load Data for KMM
    X_src_flat = X_src.reshape(X_src.shape[0], -1)
    X_tgt_flat = X_tgt.reshape(X_tgt.shape[0], -1)
    subsample_src = min(2000, X_src_flat.shape[0])
    subsample_tgt = min(800, X_tgt_flat.shape[0])
    rs_s = np.random.RandomState(42)
    rs_t = np.random.RandomState(43)
    idx_s = rs_s.choice(X_src_flat.shape[0], subsample_src, replace=False)
    idx_t = rs_t.choice(X_tgt_flat.shape[0], subsample_tgt, replace=False)

    print(f"X_tgt.shape = {X_tgt.shape}")

    
    z_src_kmm, x_cov_past_src_kmm, x_cov_future_src_kmm, y_src_kmm = build_windows_temp_transformer(split_df=source_train_df, feature_cols=feature_cols, input_len=input_len, horizon=horizon, stride=stride)
    z_tgt_kmm, x_cov_past_tgt_kmm, x_cov_future_tgt_kmm, y_tgt_kmm = build_windows_temp_transformer(split_df=target_train_sel, feature_cols=feature_cols, input_len=input_len, horizon=horizon, stride=stride)
    z_tgt_val_kmm, x_cov_past_tgt_val_kmm, x_cov_future_tgt_val_kmm, y_tgt_val_kmm = build_windows_temp_transformer(split_df=target_val_df,feature_cols=feature_cols, input_len=input_len, horizon=horizon, stride=stride)
    z_tgt_test_kmm, x_cov_past_tgt_test_kmm, x_cov_future_tgt_test_kmm, y_tgt_test_kmm = build_windows_temp_transformer(split_df=target_test_df, feature_cols=feature_cols, input_len=input_len, horizon=horizon, stride=stride)

    X_src_stack = stack_for_temperature(z_src_kmm, x_cov_past_src_kmm, x_cov_future_src_kmm, horizon)
    X_tgt_stack = stack_for_temperature(z_tgt_kmm, x_cov_past_tgt_kmm, x_cov_future_tgt_kmm, horizon)
    X_tgt_val_stack = stack_for_temperature(z_tgt_val_kmm, x_cov_past_tgt_val_kmm, x_cov_future_tgt_val_kmm, horizon)
    X_tgt_test_stack = stack_for_temperature(z_tgt_test_kmm, x_cov_past_tgt_test_kmm, x_cov_future_tgt_test_kmm, horizon)

    print(f"Java train windows:         {X_src.shape[0]}")
    print(f"Papua train windows (sel):  {X_tgt.shape[0]}")
    print(f"Papua val windows (FULL):   {X_tgt_val.shape[0]}")
    print(f"Papua test windows (FULL):  {X_tgt_test.shape[0]}")


    # -----------------------------
    # A) ARIMA: Non-DA
    # -----------------------------
    
    print("\n" + "-" * 60)
    print("A) ARIMA [Non-DA]")
    print("-" * 60)
    arima_start_time = datetime.now()
    arima_metrics = arima_baseline_rolling(test_df=target_test_sel, train_df=target_train_sel, target_col=target_col, horizon=horizon)
    arima_end_time = convert_seconds((datetime.now() - arima_start_time).total_seconds())
    print(f"Training ARIMA took: {arima_end_time}")
    print(
        f"\n================= ARIMA =================" 
        f"\nMAE: {arima_metrics['mae']:.6f} MSE: {arima_metrics['mse']:.6f} RMSE: {arima_metrics['rmse']:.4f} "
    )

    # -----------------------------
    # B) Vu-Tran Transformer Non-DA
    # -----------------------------
    
    print("\n" + "-" * 60)
    print("B) Vu Tran Transformer [Non-Domain Adaptation]")
    print("-" * 60)
    vu_tran_non_kmm_model = TemperatureTransformer(
        input_dim_primary=1,     
        input_dim_cov=8,         
        hidden_dim=64,
        n_heads=4,
        num_layers=2,
        forecast_horizon=horizon)
    vu_tran_non_kmm_model, _ = train_transformer_non_da(
        model=vu_tran_non_kmm_model,
        X_target=X_tgt_stack,
        y_target=y_tgt_kmm,
        X_val=X_tgt_val_stack,
        y_val=y_tgt_val_kmm,
        input_len=input_len,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )
    vu_tran_non_kmm_metrics = eval_model_kmm_vu_tran_metrics(vu_tran_non_kmm_model, X_tgt_test_stack, y_tgt_test_kmm, input_len, device)
    print(
        f"\n================= Vu Tran Transformer =================" 
        f"\nMAE: {vu_tran_non_kmm_metrics['mae']:.6f}  MSE: {vu_tran_non_kmm_metrics['mse']:.6f}  RMSE: {vu_tran_non_kmm_metrics['rmse']:.4f}  "
    )

    # -----------------------------
    # C) Vu-Tran Transformer DA
    # -----------------------------
    print("\n" + "-" * 60)
    print("B) Vu Tran Transformer [Domain Adaptation]")
    print("-" * 60)
    print("Finding Beta Value")
    
    beta_start_time = datetime.now()
    beta = np.load('notebook/beta_weights.npy')
    print(f"Starts Finding Beta At: {beta_start_time.strftime('%H:%M:%S')}")
    # beta = kmm_weights(X_src_flat[idx_s], X_tgt_flat[idx_t], B=2.0, eps=0.1)
    beta_end_time = datetime.now()
    print(f"Finishing Finding Beta At: {beta_end_time.strftime('%H:%M:%S')}")
    beta_total_time = convert_seconds((beta_end_time - beta_start_time).total_seconds())
    print(f"finding beta value took: {beta_total_time}")

    vu_tran_kmm_model = TemperatureTransformer(
        input_dim_primary=1,     # temperature
        input_dim_cov=8,         # covariates
        hidden_dim=64,
        n_heads=4,
        num_layers=2,
        forecast_horizon=horizon)
    vu_tran_kmm_model, _ = train_transformer_da(
        model=vu_tran_kmm_model,
        X_source=X_src_stack[idx_s],
        y_source=y_src_kmm[idx_s],
        beta=beta,
        X_val=X_tgt_val_stack,
        y_val=y_tgt_val_kmm,
        input_len=input_len,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )
    vu_tran_kmm_metrics = eval_model_kmm_vu_tran_metrics(vu_tran_kmm_model, X_tgt_test_stack, y_tgt_test_kmm, input_len, device)
    print(
        f"\n================= Vu Tran KMM ================="
        f"\nMAE: {vu_tran_kmm_metrics['mae']:.6f}  MSE: {vu_tran_kmm_metrics['mse']:.6f}  RMSE: {vu_tran_kmm_metrics['rmse']:.4f}"
    )
    
    # -----------------------------
    # D) ATTF: Attention Sharing Non-DA
    # -----------------------------
    
    print("\n" + "-" * 60)
    print("D) ATTF [Non-Domain Adaptation]")
    print("-" * 60)
    attf = AttF(
        input_dim=X_tgt.shape[2],
        hidden_dim=64,
        output_steps=7,
    )
    attf, _ = train_attf_earlystop_target_mae(
        attf,
        X_tgt,
        y_tgt,
        X_tgt_val,
        y_tgt_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )
    attf_metrics = attf_eval_model_metrics(attf, X_tgt_test, y_tgt_test, batch_size, device)
    print(
        f"\n================= ATTF ================="
        f"\nMAE: {attf_metrics['mae']:.6f}  MSE: {attf_metrics['mse']:.6f}  RMSE: {attf_metrics['rmse']:.4f}"
    )

    # -----------------------------
    # E) DAF: Attention Sharing Non-DA
    # -----------------------------
    
    print("\n" + "-" * 60)
    print("E) DAF [Domain Adaptation]")
    print("-" * 60)

    print(X_tgt.shape)
    daf = DAF(
        input_dim=X_tgt.shape[2],
        hidden_dim=64,
        output_steps=horizon,
    )
    discriminator = DomainDiscriminator(hidden_dim=64)
    daf, _ = train_daf_earlystop_target_mae(
        model=daf,
        discriminator=discriminator,
        X_src=X_src,
        y_src=y_src,
        X_tgt=X_tgt,
        X_tgt_val=X_tgt_val,
        y_tgt_val=y_tgt_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device  
    )
    daf_metrics = daf_eval_model_metrics(daf, X_tgt_test, y_tgt_test, batch_size, device)
    print(
        f"\n================= DAF ================="
        f"\nMAE: {daf_metrics['mae']:.6f}  MSE: {daf_metrics['mse']:.6f}  RMSE: {daf_metrics['rmse']:.4f}"
    )

    print("\n" + "-" * 60)
    print("F) TFT [Non Domain Adaptation]")
    print("-" * 60)
    
    X_src = X_src.reshape(X_src.shape[0], X_src.shape[1], 9, 1)
    X_tgt = X_tgt.reshape(X_tgt.shape[0], X_tgt.shape[1], 9, 1)
    X_tgt_val = X_tgt_val.reshape(X_tgt_val.shape[0], X_tgt_val.shape[1], 9, 1)
    X_tgt_test = X_tgt_test.reshape(X_tgt_test.shape[0], X_tgt_test.shape[1], 9, 1)

   
    tft_model = TFT_Target_Model(
        input_dim=1,
        num_vars=9,
        hidden_dim=64,
        horizon=horizon
    )

    tft_model, best_val_mae = train_tft_non_da(
        tft_model,
        X_tgt,
        y_tgt,
        X_tgt_val,
        y_tgt_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    )

    tft_metrics = tft_eval_model_metrics_non_da(tft_model, X_tgt_test, y_tgt_test, batch_size, device)
    print(
        f"\n================= TFT Non DA ================="
        f"\nMAE: {tft_metrics['mae']:.6f}  MSE: {tft_metrics['mse']:.6f}  RMSE: {tft_metrics['rmse']:.4f}"
    )

    print("\n" + "-" * 60)
    print("G) TFT [Domain Adaptation]")
    print("-" * 60)

   
    tft_da_model = TFT_DA_Model(
        input_dim=X_src.shape[3],
        num_vars=X_src.shape[2],
        hidden_dim=64,
        horizon=horizon
    )

    tft_da_model, _ = train_tft_da(
        tft_da_model,
        X_src,
        y_src,
        X_tgt,
        y_tgt,
        X_tgt_val,
        y_tgt_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device
    ) 

    tft_da_metrics = tft_eval_model_metrics_da(tft_da_model, X_tgt_test, y_tgt_test, batch_size, device)
    print(
        f"\n================= TFT DA ================="
        f"\nMAE: {tft_da_metrics['mae']:.6f}  MSE: {tft_da_metrics['mse']:.6f}  RMSE: {tft_da_metrics['rmse']:.4f}"
    )

    out = {
        "arima" : arima_metrics,
        "vu tran transformer": vu_tran_non_kmm_metrics,
        "vu tran kmm": vu_tran_kmm_metrics,
        "attf": attf_metrics,
        "daf": daf_metrics,
        'tft non da': tft_metrics,
        'tft da': tft_da_metrics
    }

    print("\n" + "-" * 60)
    print("Results:")
    for k in [key for key, _ in out.items()]:
        m = out[k]
        print(f"  {k:>18s}  MAE={m['mae']:.4f}  MSE={m['mse']:.6f}  RMSE={m['rmse']:.4f}")
        result_rows = log_row(
            experiment_name,
            result_rows,
            run_ts,
            input_len,
            horizon,
            stride,
            feature_cols,
            target_station_ids="|".join([str(target_station_ids)]),
            method=k,
            metric_mae=m['mae'],
            metric_mse=m['mse'],
            metric_rmse=m['rmse']
        )

    return out, result_rows