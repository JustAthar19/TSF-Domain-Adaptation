from datetime import datetime
from scipy.stats import wilcoxon as wilcoxon_signed_rank

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from src.data.windowing import stack_for_temperature, build_windows, build_windows_temp_transformer

from src.models.Vanilla.vanilla import VanillaTransformer
from src.models.DANN.domain_classifier import DomainClassifier
from src.models.DANN.feat_extractor import FeatureExtractor
from src.models.ATTF.attf import AttF
from src.models.DAF.daf import DAF
from src.models.DAF.domain_discriminator import DomainDiscriminator
from src.models.kmm_vu_tran.transformer import TemperatureTransformer
from src.models.kmm_vu_tran.kmm import kmm_weights

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




_has_wilcoxon = True
low_resources_fracs  = [0.05, 0.15, 0.25]

def run_low_quantity_target_experiment(
    run_ts: datetime,
    result_rows: list,
    experiment_name: str,
    train_java_df: pd.DataFrame,
    train_papua_df: pd.DataFrame,
    val_papua_df: pd.DataFrame,
    test_papua_df: pd.DataFrame,
    config: dict,
    input_len: list,
    low_resrouces_fracs: list,
    n_repeats: int = 30,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
):
    
    for in_len in input_len:
        in_len = int(in_len)    
        X_src_lr, y_src_lr = build_windows(train_java_df)
        X_tgt_full, y_tgt_full = build_windows(train_papua_df)
        X_tgt_val_lr, y_tgt_val_lr = build_windows(val_papua_df)
        X_tgt_test_lr, y_tgt_test_lr = build_windows(test_papua_df)

        z_src_kmm, x_cov_past_src_kmm, x_cov_future_src_kmm, y_src_kmm = build_windows_temp_transformer(train_java_df, config)
        z_tgt_val_kmm, x_cov_past_tgt_val_kmm, x_cov_future_tgt_val_kmm, y_tgt_val_kmm = build_windows_temp_transformer(val_papua_df, config)
        z_tgt_test_kmm, x_cov_past_tgt_test_kmm, x_cov_future_tgt_test_kmm, y_tgt_test_kmm = build_windows_temp_transformer(test_papua_df, config)

        X_src_stack = stack_for_temperature(z_src_kmm, x_cov_past_src_kmm, x_cov_future_src_kmm, config)
        X_tgt_val_stack = stack_for_temperature(z_tgt_val_kmm, x_cov_past_tgt_val_kmm, x_cov_future_tgt_val_kmm, config)
        X_tgt_test_stack = stack_for_temperature(z_tgt_test_kmm, x_cov_past_tgt_test_kmm, x_cov_future_tgt_test_kmm, config)
    
        if X_src_lr.shape[0] == 0 or X_tgt_full.shape[0] == 0 or X_tgt_val_lr.shape[0] == 0 or X_tgt_test_lr.shape[0] == 0:
            print("Not enough windows for low-resource experiment at this input length; skipping.")
        
            rs_base = np.random.RandomState(123 + int(in_len))
            frac_results = {}
        
        for frac in low_resources_fracs:
            n = max(1, int(round(frac * X_tgt_full.shape[0])))
            vanilla_mae_runs, vanilla_mse_runs, vanilla_rmse_runs = [], [], []
            dann_mae_runs, dann_mse_runs, dann_rmse_runs = [], [], []
            attf_mae_runs, attf_mse_runs, attf_rmse_runs = [], [], []
            daf_mae_runs, daf_mse_runs, daf_rmse_runs = [], [], []
            kmm_vanilla_mae_runs, kmm_vanilla_mse_runs, kmm_vanilla_rmse_runs = [], [], []
            kmm_vu_tran_mae_runs, kmm_vu_tran_mse_runs, kmm_vu_tran_rmse_runs = [], [], []
            
            for rep_idx in range(n_repeats):
                seed = int(rs_base.randint(0, 2**31 -1))
                rs = np.random.RandomState(seed)
                idx = rs.choice(X_tgt_full.shape[0], n, replace=False)
                X_tgt_small = X_tgt_full[idx]
                y_tgt_small = y_tgt_full[idx]

                # Vanilla Transformer Non-DA 
                print("A.) Vanilla Transformer (Low-Resouce) | (Domain Adaptation)")
                print("-" * 60)
                vanilla_lr, _ = train_vanilla_earlystop_target_mae(
                    vanilla_lr,
                    X_tgt_small,
                    y_tgt_small,
                    X_tgt_val_lr,
                    y_tgt_val_lr,
                    config,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    patience=patience,
                )
                vanilla_metrics = eval_model_metrics(vanilla_lr, X_tgt_test_lr, y_tgt_test_lr, config, batch_size=256)
                
                vanilla_mae_runs.append(vanilla_metrics['mae'])
                vanilla_mse_runs.append(vanilla_metrics['mse'])
                vanilla_rmse_runs.append(vanilla_metrics['rmse'])

                # DANN Domain Adaptation

                feat_lr = FeatureExtractor(input_dim=X_src_lr.shape[2], d_model=32, nhead=4, num_layers=3, dropout=0.1)
                task_lr = nn.Linear(32, config['horizon'])
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
                    config,
                    epochs=100,
                    batch_size=256,
                    lr=1e-3,
                    patience=5,
                    use_target_task_loss=True,
                )
                dann_metrics = eval_model_metrics(dann_lr_model, X_tgt_test_lr, y_tgt_test_lr, batch_size=256)
                dann_mae_runs.append(dann_metrics['mae'])
                dann_mse_runs.append(dann_metrics['mse'])
                dann_rmse_runs.append(dann_metrics['rmse'])
                # ATTF: Attention Sharing Non-DA
                attf_lr = AttF(
                    input_dim=X_src_lr.shape[2],
                    hidden_dim=64,
                    output_steps=7
                )
                attf_lr, _ = train_attf_earlystop_target_mae(
                    X_tgt_small,
                    y_tgt_small,
                    X_tgt_val_lr,
                    y_tgt_val_lr,
                    config=config,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    patience=patience
                )
                attf_metrics = attf_eval_model_metrics(attf_lr, X_tgt_test_lr, y_tgt_test_lr, config, batch_size=256)
                attf_mae_runs.append(attf_metrics['mae'])
                attf_mse_runs.append(attf_metrics['mse'])
                attf_rmse_runs.append(attf_metrics['rmse'])

                daf_lr = DAF(
                    input_dim=X_src_lr.shape[2],
                    hidden_dim=32, 
                    output_steps=7
                )
                discriminator_lr = DomainDiscriminator(hidden_dim=64)
                
                # DAF Attention Sharing Domain Adaptation
                daf_lr, _ = train_daf_earlystop_target_mae(
                    model=daf_lr,
                    discriminator=discriminator_lr,
                    X_src=X_src_lr,
                    y_src=y_src_lr,
                    X_tgt=X_tgt_small,
                    X_tgt_val=X_tgt_val_lr,
                    y_tgt_val=y_tgt_val_lr,
                    config=config,
                    epochs=100,
                    batch_size=256,
                    lr=1e-3,
                    lambda_recon=0.5,
                    lambda_domain=0.1,
                )    
                daf_metrics = daf_eval_model_metrics(daf_lr, X_tgt_test_lr, y_tgt_test_lr, config, batch_size=256)
                daf_mae_runs.append(daf_metrics['mae'])
                daf_mse_runs.append(daf_metrics['mse'])
                daf_rmse_runs.append(daf_metrics['rmse'])

                # Vanilla Transformer KMM
                print("Finding beta Value")
                X_src_flat = X_src_lr.reshape(X_src_lr.shape[0], -1)
                X_tgt_flat = X_tgt_small.reshape(X_tgt_small.shape[0], -1)
                subsample_src = min(2000, X_src_flat.shape[0])
                subsample_tgt = min(800, X_tgt_flat.shape[0])
                idx_s = np.random.RandomState(seed ^ 0xA5A5).choice(X_src_flat.shape[0], subsample_src, replace=False)
                idx_t = np.random.RandomState(seed ^ 0x5A5A).choice(X_tgt_flat.shape[0], subsample_tgt, replace=False)
                beta_start_time = datetime.now()
                beta = kmm_weights(X_src_flat[idx_s], X_tgt_flat[idx_t], B=2.0, eps=0.1)
                beta_end_time = convert_seconds((datetime.now()-beta_start_time).total_seconds())
                
                print(f"Finding Beta Value Took: {beta_end_time}")
                vanilla_kmm_lr, _ = train_vanilla_transformer_weighted(
                    X_src_lr[idx_s],
                    y_src_lr[idx_s],
                    X_tgt_val_lr,
                    y_tgt_val_lr,
                    beta,
                    config,
                    epochs=100,
                    batch_size=256,
                    lr=1e-3,
                    patience=5
                )
                vanilla_kmm_metrics = eval_model_metrics(vanilla_kmm_lr, X_tgt_test_lr, y_tgt_test_lr, batch_size=256)
                kmm_vanilla_mae_runs.appned(vanilla_kmm_metrics['mae'])
                kmm_vanilla_mse_runs.appned(vanilla_kmm_metrics['mse'])
                kmm_vanilla_rmse_runs.appned(vanilla_kmm_metrics['rmse'])

                # Vu Tran KMM
                kmm_vu_tran_lr = TemperatureTransformer(
                    input_dim_primary=1,
                    input_dim_cov=8,
                    hidden_dim=64,
                    n_heads=4,
                    num_layers=2,
                    forecast_horizon=7
                )

                kmm_vu_tran_model, _  = train_transformer_weighted_kmm(
                    kmm_vu_tran_lr,
                    X_src_stack[idx_s],
                    y_src_kmm[idx_s],
                    beta,
                    config,
                    X_tgt_val_stack,
                    y_tgt_val_kmm,
                    epochs=epochs,
                    batch_size=256,
                    lr=lr
                )
                kmm_vu_tran_metrics = eval_model_kmm_vu_tran_metrics(kmm_vu_tran_model, X_tgt_test_stack, y_tgt_test_kmm, config, batch_size=256)
                kmm_vu_tran_mae_runs.append(kmm_vu_tran_metrics['mae'])
                kmm_vu_tran_mse_runs.append(kmm_vu_tran_metrics['mse'])
                kmm_vu_tran_rmse_runs.append(kmm_vu_tran_metrics['rmse'])

                # Convert to Numpy Array
                vanilla_mae_runs, vanilla_mse_runs, vanilla_rmse_runs = np.asarray(vanilla_mae_runs, dtype=np.float64), np.asarray(vanilla_mse_runs, dtype=np.float64), np.asarray(vanilla_mse_runs, dtype=np.float64)
                dann_mae_runs, dann_mse_runs, dann_rmse_runs = np.asarray(dann_mae_runs, dtype=np.float64), np.asarray(dann_mse_runs, dtype=np.float64), np.asarray(dann_rmse_runs, dtype=np.float64)
                attf_mae_runs, attf_mse_runs, attf_rmse_runs = np.asarray(attf_mae_runs, dtype=np.float64), np.asarray(attf_mse_runs, dtype=np.float64), np.asarray(attf_rmse_runs, dtype=np.float64)
                daf_mae_runs, daf_mse_runs, daf_rmse_runs = np.asarray(daf_mae_runs, dtype=np.float64), np.asarray(daf_mse_runs, dtype=np.float64), np.asarray(daf_rmse_runs, dtype=np.float64)
                kmm_vanilla_mae_runs, kmm_vanilla_mse_runs, kmm_vanilla_rmse_runs = np.asarray(kmm_vanilla_mae_runs, dtype=np.float64), np.asarray(kmm_vanilla_mse_runs, dtype=np.float64), np.asarray(kmm_vanilla_rmse_runs, dtype=np.float64)
                kmm_vu_tran_mae_runs, kmm_vu_tran_mse_runs, kmm_vu_tran_rmse_runs = np.asarray(kmm_vu_tran_mae_runs, dtype=np.float64), np.asarray(kmm_vu_tran_mse_runs, dtype=np.float64), np.asarray(kmm_vu_tran_rmse_runs, dtype=np.float64)
                
                v_mae_mean, v_mae_std = float(np.mean(vanilla_mae_runs)), float(np.std(vanilla_mae_runs, ddof=1))
                v_mse_mean, v_mse_std = float(np.mean(vanilla_mse_runs)), float(np.std(vanilla_mse_runs, ddof=1))
                v_rmse_mean, v_rmse_std = float(np.mean(vanilla_rmse_runs)), float(np.std(vanilla_rmse_runs, ddof=1))

                d_mae_mean, v_mae_std = float(np.mean(dann_mae_runs)), float(np.std(dann_mae_runs, ddof=1))
                d_mse_mean, v_mse_std = float(np.mean(dann_mse_runs)), float(np.std(dann_mse_runs, ddof=1))
                d_mse_mean, v_mse_std = float(np.mean(dann_rmse_runs)), float(np.std(dann_rmse_runs, ddof=1))
                
                at_mae_mean, v_mae_std = float(np.mean(attf_mae_runs)), float(np.std(attf_mae_runs, ddof=1))
                at_mse_mean, v_mse_std = float(np.mean(attf_mse_runs)), float(np.std(attf_mse_runs, ddof=1))
                at_rmse_mean, v_rmse_std = float(np.mean(attf_rmse_runs)), float(np.std(attf_rmse_runs, ddof=1))

                d_mae_mean, d_mae_std = float(np.mean(daf_mae_runs)), float(np.std(daf_mae_runs, ddof=1))
                d_mse_mean, d_mse_std = float(np.mean(daf_mse_runs)), float(np.std(daf_mse_runs, ddof=1))
                d_rmse_mean, d_rmse_std = float(np.mean(daf_rmse_runs)), float(np.std(daf_rmse_runs, ddof=1))

                kv_mae_mean, kv_mae_std = float(np.mean(kmm_vanilla_mae_runs)), float(np.std(kmm_vanilla_mae_runs, ddof=1))    
                kv_mse_mean, kv_mse_std = float(np.mean(kmm_vanilla_mse_runs)), float(np.std(kmm_vanilla_mse_runs, ddof=1))    
                kv_rmse_mean, kv_rmse_std = float(np.mean(kmm_vanilla_rmse_runs)), float(np.std(kmm_vanilla_rmse_runs, ddof=1))    

                kvt_mae_mean, kvt_mae_std = float(np.mean(kmm_vu_tran_mae_runs)), float(np.std(kmm_vu_tran_mae_runs, ddof=1))
                kvt_mse_mean, kvt_mae_std = float(np.mean(kmm_vu_tran_mse_runs)), float(np.std(kmm_vu_tran_mse_runs, ddof=1))
                kvt_mae_mean, kvt_mae_std = float(np.mean(kmm_vu_tran_rmse_runs)), float(np.std(kmm_vu_tran_rmse_runs, ddof=1))

                delta_mae_mean = float("nan")
                delta_wilc_mse = float("nan")
                if _has_wilcoxon:
                    try:
                        _, p_wilc_mae = wilcoxon_signed_rank(van)
                
                
