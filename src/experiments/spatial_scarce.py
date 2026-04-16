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


from src.training.arima_baseline import arima_baseline_rolling
from src.training.supervised import train_vanilla_earlystop_target_mae
from src.training.dann import train_domain_adversarial_dann
from src.training.attf import train_attf_earlystop_target_mae
from src.training.daf import train_daf_earlystop_target_mae
from src.training.transformer_weighted import train_transformer_weighted_kmm, train_vanilla_transformer_weighted

from src.evaluation.supervised import eval_model_metrics
from src.evaluation.attf import attf_eval_model_metrics
from src.evaluation.daf import daf_eval_model_metrics
from src.evaluation.kmm_vu_tran import eval_model_kmm_vu_tran_metrics

from src.utils.io import log_row
from src.utils.time import convert_seconds


def run_target_station_experiment(
    run_ts: datetime,
    result_rows: list,
    experiment_name: str,
    train_java_df: pd.DataFrame,
    train_papua_df: pd.DataFrame,
    val_papua_df: pd.DataFrame,
    test_papua_df: pd.DataFrame,
    config: dict,
    target_station_ids: list,
    epochs: int = 100,
    batch_size: int = 256,
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
    test_papua_sel = filter_df_by_station_ids(test_papua_df, target_station_ids)

    # Load Data

    X_src, y_src = build_windows(train_java_df, config)
    X_tgt, y_tgt = build_windows(train_papua_sel, config)
    X_tgt_val, y_tgt_val = build_windows(val_papua_df, config)  # FULL Papua val
    X_tgt_test, y_tgt_test = build_windows(test_papua_df, config)  # FULL Papua test
    
    # Load Data for KMM
    X_src_flat = X_src.reshape(X_src.shape[0], -1)
    X_tgt_flat = X_tgt.reshape(X_tgt.shape[0], -1)
    subsample_src = min(2000, X_src_flat.shape[0])
    subsample_tgt = min(800, X_tgt_flat.shape[0])
    rs_s = np.random.RandomState(42)
    rs_t = np.random.RandomState(43)
    idx_s = rs_s.choice(X_src_flat.shape[0], subsample_src, replace=False)
    idx_t = rs_t.choice(X_tgt_flat.shape[0], subsample_tgt, replace=False)

    
    z_src_kmm, x_cov_past_src_kmm, x_cov_future_src_kmm, y_src_kmm = build_windows_temp_transformer(train_java_df, config)
    z_tgt_val_kmm, x_cov_past_tgt_val_kmm, x_cov_future_tgt_val_kmm, y_tgt_val_kmm = build_windows_temp_transformer(val_papua_df, config)
    z_tgt_test_kmm, x_cov_past_tgt_test_kmm, x_cov_future_tgt_test_kmm, y_tgt_test_kmm = build_windows_temp_transformer(test_papua_df, config)

    X_src_stack = stack_for_temperature(z_src_kmm, x_cov_past_src_kmm, x_cov_future_src_kmm, config)
    X_tgt_val_stack = stack_for_temperature(z_tgt_val_kmm, x_cov_past_tgt_val_kmm, x_cov_future_tgt_val_kmm, config)
    X_tgt_test_stack = stack_for_temperature(z_tgt_test_kmm, x_cov_past_tgt_test_kmm, x_cov_future_tgt_test_kmm, config)
    
    print(f"Java train windows:         {X_src.shape[0]}")
    print(f"Papua train windows (sel):  {X_tgt.shape[0]}")
    print(f"Papua val windows (FULL):   {X_tgt_val.shape[0]}")
    print(f"Papua test windows (FULL):  {X_tgt_test.shape[0]}")

    if X_tgt.shape[0] == 0 or X_tgt_val.shape[0] == 0 or X_tgt_test.shape[0] == 0:
        print("Not enough windows for this experiment; skipping.")
        out = {
            "arima": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
            "vanilla": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
            "dann": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
            "kmm": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
            "climate_aware": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
            "climate_aware_dann": {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")},
        }
        return out

    # -----------------------------
    # A) ARIMA: Non-DA
    # -----------------------------
    
    print("\n" + "-" * 60)
    print("A) ARIMA (Target Station Only)")
    print("-" * 60)
    arima_start_time = datetime.now()
    arima_metrics = arima_baseline_rolling(test_papua_sel, train_papua_sel, config)
    arima_end_time = convert_seconds((datetime.now() - arima_start_time).total_seconds())
    print(f"Training ARIMA took: {arima_end_time}")
    # arima_metrics = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    print(
        f"Papua Test  MAE: {arima_metrics['mae']:.4f}  "
        f"RMSE: {arima_metrics['rmse']:.4f}  MSE: {arima_metrics['mse']:.6f}"
    )

    # -----------------------------
    # B) Vanilla: Non-DA
    # -----------------------------
    
    print("\n" + "-" * 60)
    print("A) Vanilla Transformer (Target Station Only)")
    print("-" * 60)
    vanilla = VanillaTransformer(
        input_dim=X_tgt.shape[2],
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        horizon=7,
    )
    vanilla, best_val_mae = train_vanilla_earlystop_target_mae(
        vanilla,
        X_tgt,
        y_tgt,
        X_tgt_val,
        y_tgt_val,
        config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )
    vanilla_metrics = eval_model_metrics(vanilla, X_tgt_test, y_tgt_test, config, batch_size=256)
    print(f"Best Papua Val MAE (early stop): {best_val_mae:.4f}")
    print(
        f"Papua Test  MAE: {vanilla_metrics['mae']:.4f}  "
        f"RMSE: {vanilla_metrics['rmse']:.4f}  MSE: {vanilla_metrics['mse']:.6f}"
    )

    # -----------------------------
    # C) DANN: adversarial DA
    # -----------------------------
    dann_metrics = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    
    print("\n" + "-" * 60)
    print("B) DANN (Java vs selected Papua; balanced batches)")
    print("-" * 60)
    feat = FeatureExtractor(input_dim=X_src.shape[2], d_model=32, nhead=4, num_layers=3, dropout=0.1)
    task_head = nn.Linear(32, config['horizon'])
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
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            use_target_task_loss=False,
    )
    dann_metrics = eval_model_metrics(dann_model, X_tgt_test, y_tgt_test, config, batch_size=256)
    print(f"Best Papua Val MAE (early stop): {best_val_mae_dann:.4f}")
    print(
        f"Papua Test  MAE: {dann_metrics['mae']:.4f}  "
        f"RMSE: {dann_metrics['rmse']:.4f}  MSE: {dann_metrics['mse']:.6f}"
    )
    
    # -----------------------------
    # C) ATTF: Attention Sharing Non-DA
    # -----------------------------
    attf_metrics = {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    
    print("\n" + "-" * 60)
    print("D) ATTF (Java vs selected Papua; balanced batches)")
    print("-" * 60)
    attf = AttF(
        input_dim=X_tgt.shape[2],
        hidden_dim=64,
        output_steps=7,
    )
    attf, best_val_mae = train_attf_earlystop_target_mae(
        attf,
        X_tgt,
        y_tgt,
        X_tgt_val,
        y_tgt_val,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )
    attf_metrics = attf_eval_model_metrics(attf, X_tgt_test, y_tgt_test, config, batch_size=256)


    print(f"Best Papua Val MAE (early stop): {best_val_mae:.4f}")
    print(
        f"Papua Test  MAE: {attf_metrics['mae']:.4f}  "
        f"RMSE: {attf_metrics['rmse']:.4f}  MSE: {attf_metrics['mse']:.6f}"
    )


    
    
    print("\n" + "-" * 60)
    print("E) DAF (Java vs selected Papua; balanced batches)")
    print("-" * 60)

    print(X_tgt.shape)
    daf = DAF(
        input_dim=X_tgt.shape[2],
        hidden_dim=64,
        output_steps=7,
    )
    discriminator = DomainDiscriminator(hidden_dim=64)
    daf, best_val_mae = train_daf_earlystop_target_mae(
        model=daf,
        discriminator=discriminator,
        X_src=X_src,
        y_src=y_src,
        X_tgt=X_tgt,
        X_tgt_val=X_tgt_val,
        y_tgt_val=y_tgt_val,
        config=config,
        epochs=100,
        batch_size=256,
        lr=1e-3,
        lambda_recon=0.5,
        lambda_domain=0.1   # IMPORTANT hyperparameter
    )
    daf_metrics = daf_eval_model_metrics(daf, X_tgt_test, y_tgt_test, config, batch_size=256)


    print(f"Best Papua Val MAE (early stop): {best_val_mae:.4f}")
    print(
        f"Papua Test  MAE: {daf_metrics['mae']:.4f}  "
        f"RMSE: {daf_metrics['rmse']:.4f}  MSE: {daf_metrics['mse']:.6f}"
    )

        
    print("\n" + "-" * 60)
    print("Vanilla KMM (Java vs selected Papua; balanced batches)")
    print("-" * 60)

    print("Finding Beta Value")
    beta_start_time = datetime.now()
    # beta = np.load('notebook/beta_weights.npy')
    beta = kmm_weights(X_src_flat[idx_s], X_tgt_flat[idx_t], B=2.0, eps=0.1)
    beta_end_time = convert_seconds((datetime.now() - beta_start_time).total_seconds())
    print(f"Finding Beta Value Took: {beta_end_time}")
    
    kmm_vanilla_model, best_val_loss = train_vanilla_transformer_weighted(
         X_src[idx_s],
         y_src[idx_s],
         X_tgt_val,
         y_tgt_val,
         beta,
         config,
         epochs=100,
         batch_size=batch_size,
         lr=lr,
         patience=patience,
    )
    kmm_vanilla_metrics = eval_model_metrics(kmm_vanilla_model, X_tgt_test, y_tgt_test, config,batch_size=256)
    print(f"Best Val loss (early stop): {best_val_loss:.6f}")
    print(
        f"Papua Test  MAE: {kmm_vanilla_metrics['mae']:.4f}  "
        f"RMSE: {kmm_vanilla_metrics['rmse']:.4f}  MSE: {kmm_vanilla_metrics['mse']:.6f}"
    )

    print("\n" + "-" * 60)
    print("Vu Tran KMM (Java vs selected Papua; balanced batches)")
    print("-" * 60)


    kmm_vu_tran_model = TemperatureTransformer(
         input_dim_primary=1,     # temperature
         input_dim_cov=8,         # covariates
         hidden_dim=64,
         n_heads=4,
         num_layers=2,
         forecast_horizon=7)
    kmm_vu_tran_model, best_val_loss = train_transformer_weighted_kmm(
                kmm_vu_tran_model,
                X_src_stack[idx_s],
                y_src_kmm[idx_s],
                beta,
                config,
                X_tgt_val_stack,
                y_tgt_val_kmm,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
    kmm_vu_tran_metrics = eval_model_kmm_vu_tran_metrics(kmm_vu_tran_model, X_tgt_test_stack, y_tgt_test_kmm, config,batch_size)

    print(f"Best Val loss (early stop): {best_val_loss:.6f}")
    print(
        f"Papua Test  MAE: {kmm_vu_tran_metrics['mae']:.4f}  "
        f"RMSE: {kmm_vu_tran_metrics['rmse']:.4f}  MSE: {kmm_vu_tran_metrics['mse']:.6f}"
    )


    out = {
        "arima" : arima_metrics,
        "vanilla": vanilla_metrics,
        "dann": dann_metrics,
        "attf": attf_metrics,
        "daf": daf_metrics,
        'kmm_vanilla': kmm_vanilla_metrics,
        'kmm_vu_tran': kmm_vu_tran_metrics
    }


    print("\n" + "-" * 60)
    print("Result (Papua test):")
    for k in [key for key, _ in out.items()]:
        m = out[k]
        print(f"  {k:>18s}  MAE={m['mae']:.4f}  MSE={m['mse']:.6f}  RMSE={m['rmse']:.4f}")
        result_rows = log_row(
            experiment_name,
            result_rows,
            run_ts=run_ts,
            config=config,
            target_station_ids="|".join([str(target_station_ids)]),
            method=k,
            metric_mae=m['mae'],
            metric_mse=m['mse'],
            metric_rmse=m['rmse']
        )

    return out, result_rows