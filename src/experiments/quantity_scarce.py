from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from src.data.windowing import stack_for_temperature, build_windows, build_windows_temp_transformer

from src.models.TFT.tft import TFT_Target_Model
from src.models.TFT.tft_da import TFT_DA_Model
from src.models.ATTF.attf import AttF
from src.models.DAF.daf import DAF
from src.models.DAF.domain_discriminator import DomainDiscriminator
from src.models.kmm_vu_tran.transformer import TemperatureTransformer
from src.models.kmm_vu_tran.kmm import kmm_weights

from src.training.vu_tran_transformer import train_transformer_non_da, train_transformer_da
from src.training.attf import train_attf_earlystop_target_mae
from src.training.daf import train_daf_earlystop_target_mae
from src.training.tft import train_tft_non_da, train_tft_da

from src.evaluation.kmm_vu_tran import eval_model_kmm_vu_tran_metrics
from src.evaluation.attf import attf_eval_model_metrics
from src.evaluation.daf import daf_eval_model_metrics
from src.evaluation.tft import tft_eval_model_metrics_non_da, tft_eval_model_metrics_da

from src.utils.io import log_row
from src.utils.time import convert_seconds

from src.utils.wilcoxon import wilcoxon_signed_rank

_has_wilcoxon = True
low_resources_fracs  = [0.05, 0.15, 0.25]

def fmt(mean, std, width=7, precision=4):
    return f"{mean:>{width}.{precision}f}±{std:<{width}.{precision}f}"

def fmt_p(p):
    if p is None:
        return "   -   "
    return f"{p:.3e}" if p < 1e-3 else f"{p:.4f}"

def run_low_quantity_tgt_experiment(
        run_ts: datetime,
        result_rows: list,
        experiment_name: str,
        source_train_df: pd.DataFrame,
        target_train_df: pd.DataFrame,
        target_val_df: pd.DataFrame,
        target_test_df: pd.DataFrame,
        n_repeats: int,
        low_resource_fracs: list,
        input_len: int,
        horizon: int,
        stride: int,
        feature_cols: list,
        target_col: str,
        epochs: int,
        batch_size: int, 
        lr: float,
        device: str
):
    X_src_lr, y_src_lr = build_windows(source_train_df, input_len, horizon, stride, feature_cols)
    X_tgt_full, y_tgt_full = build_windows(target_train_df, input_len, horizon, stride, feature_cols)
    X_tgt_val_lr, y_tgt_val_lr = build_windows(target_val_df, input_len, horizon, stride, feature_cols)
    X_tgt_test_lr, y_tgt_test_lr = build_windows(target_test_df, input_len, horizon, stride, feature_cols)

    z_src_kmm, x_cov_past_src_kmm, x_cov_future_src_kmm, y_src_kmm = build_windows_temp_transformer(source_train_df, feature_cols, input_len, horizon, stride)
    z_tgt_kmm, x_cov_past_tgt_kmm, x_cov_future_tgt_kmm, y_tgt_kmm  = build_windows_temp_transformer(target_train_df, feature_cols, input_len, horizon, stride)
    z_tgt_val_kmm, x_cov_past_tgt_val_kmm, x_cov_future_tgt_val_kmm, y_tgt_val_kmm = build_windows_temp_transformer(target_val_df, feature_cols, input_len, horizon, stride)
    z_tgt_test_kmm, x_cov_past_tgt_test_kmm, x_cov_future_tgt_test_kmm, y_tgt_test_kmm = build_windows_temp_transformer(target_test_df, feature_cols, input_len, horizon, stride)

    X_src_stack = stack_for_temperature(z_src_kmm, x_cov_past_src_kmm, x_cov_future_src_kmm, horizon)
    X_tgt_stack = stack_for_temperature(z_tgt_kmm, x_cov_past_tgt_kmm, x_cov_future_tgt_kmm, horizon)
    X_tgt_val_stack = stack_for_temperature(z_tgt_val_kmm, x_cov_past_tgt_val_kmm, x_cov_future_tgt_val_kmm, horizon)
    X_tgt_test_stack = stack_for_temperature(z_tgt_test_kmm, x_cov_past_tgt_test_kmm, x_cov_future_tgt_test_kmm, horizon)

    rs_base = np.random.RandomState(123)
    frac_results = {}
    for frac in low_resource_fracs:
        print(f"Training for {frac} target variable fractions")
        n = max(1, int(round(frac * X_tgt_full.shape[0])))
        vu_tran_transformer_mae_runs, vu_tran_transformer_mse_runs, vu_tran_transformer_rmse_runs = [], [], []
        vu_tran_kmm_mae_runs, vu_tran_kmm_mse_runs, vu_tran_kmm_rmse_runs = [], [], []
        attf_mae_runs, attf_mse_runs, attf_rmse_runs = [], [], []
        daf_mae_runs, daf_mse_runs, daf_rmse_runs = [], [], []
        tft_non_da_mae_runs, tft_non_da_mse_runs, tft_non_da_rmse_runs = [], [], []
        tft_da_mae_runs, tft_da_mse_runs, tft_da_rmse_runs = [], [], []

        for rep_idx in range(n_repeats):
            seed = int(rs_base.randint(0, 2**31-1))
            rs = np.random.RandomState(seed)
            idx = rs.choice(X_tgt_full.shape[0], n, replace=False)
            X_tgt_small = X_tgt_full[idx]
            y_tgt_small = y_tgt_full[idx]

            # -------------------------------
            # Vu-Tran Transformer (Non-DA)
            # -------------------------------
            vu_tran_non_kmm_lr_model = TemperatureTransformer(
                input_dim_primary=1,     # temperature
                input_dim_cov=8,         # covariates
                hidden_dim=64,
                n_heads=4,
                num_layers=2,
                forecast_horizon=horizon
            )
            X_src_flat = X_src_lr.reshape(X_src_lr.shape[0], -1)
            X_tgt_flat = X_tgt_small.reshape(X_tgt_small.shape[0], -1)
            
            subsample_src = min(2000, X_src_flat.shape[0])
            subsample_tgt = min(800, X_tgt_flat.shape[0])
            # idx_s = np.random.RandomState(seed ^ 0xA5A5).choice(X_src_flat.shape[0], subsample_src, replace=False)
            # idx_t = np.random.RandomState(seed ^ 0x5A5A).choice(X_tgt_flat.shape[0], subsample_tgt, replace=False)
            # sub-train sample for KMM and DA Training
            idx_s_train = np.random.RandomState(seed ^ 0xA5A5).choice(X_src_flat.shape[0], subsample_src, replace=False)
            idx_t_train = np.random.RandomState(seed ^ 0x5A5A).choice(X_tgt_flat.shape[0], subsample_tgt, replace=False)

            # Sample validation sub-sample
            subsample_t_val = min(800, X_tgt_val_stack.shape[0])
            idx_t_val = np.random.RandomState(seed ^ 0x5A5A).choice(X_tgt_val_stack.shape[0], subsample_t_val, replace=False)

            print("\n" + "-" * 60)
            print("Vu Tran Transformer [Non Domain Adaptation]")
            print("-" * 60)
            vu_tran_non_kmm_lr_model, _ = train_transformer_non_da(
                model=vu_tran_non_kmm_lr_model,
                X_target=X_tgt_stack[idx_t_train],
                y_target=y_tgt_kmm[idx_t_train],
                X_val=X_tgt_val_stack[idx_t_val],
                y_val= y_tgt_val_kmm[idx_t_val],
                input_len=input_len,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device
            )
            vu_tran_non_kmm_lr_metrics = eval_model_kmm_vu_tran_metrics(vu_tran_non_kmm_lr_model, X_tgt_test_stack, y_tgt_test_kmm, input_len, device)
                
            vu_tran_transformer_mae_runs.append(vu_tran_non_kmm_lr_metrics['mae'])
            vu_tran_transformer_mse_runs.append(vu_tran_non_kmm_lr_metrics['mse'])
            vu_tran_transformer_rmse_runs.append(vu_tran_non_kmm_lr_metrics['rmse'])

            # -------------------------------
            # Vu-Tran Transformer (DA)
            # -------------------------------        
            
            print("\n" + "-" * 60)
            print("Vu Tran KMM [Domain Adaptation]")
            print("-" * 60)
            print(f"finding beta value for iterations no: {rep_idx + 1}")
            beta_start_time = datetime.now()
            print(f"Starts Finding Beta At: {beta_start_time.strftime('%H:%M:%S')}")

            beta = kmm_weights(X_src_flat[idx_s_train], X_tgt_flat[idx_t_train], B=2.0, eps=0.1)
            # np.save("experiments/beta.npy")
            beta_end_time = datetime.now()
            print(f"Finished Finding Beta At: {beta_end_time.strftime('%H:%M:%S')}")
            beta_total_time = convert_seconds((beta_end_time - beta_start_time).total_seconds())
            print(f"finding beta value took: {beta_total_time}")
            
            vu_tran_kmm_lr_model = TemperatureTransformer(
                input_dim_primary=1,
                input_dim_cov=8,
                hidden_dim=64,
                n_heads=4,
                num_layers=2,
                forecast_horizon=horizon
            )      
            vu_tran_kmm_lr_model, _ = train_transformer_da(
                model=vu_tran_kmm_lr_model,
                X_source=X_src_stack[idx_s_train],
                y_source=y_src_kmm[idx_s_train],
                beta=beta,
                X_val=X_tgt_val_stack[idx_t_val],
                y_val=y_tgt_val_kmm[idx_t_val],
                input_len=input_len,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device
            )

            vu_tran_kmm_lr_metrics = eval_model_kmm_vu_tran_metrics(vu_tran_kmm_lr_model, X_tgt_test_stack, y_tgt_test_kmm, input_len, device)
            vu_tran_kmm_mae_runs.append(vu_tran_kmm_lr_metrics['mae'])
            vu_tran_kmm_mse_runs.append(vu_tran_kmm_lr_metrics['mse'])
            vu_tran_kmm_rmse_runs.append(vu_tran_kmm_lr_metrics['rmse'])



            # -------------------------------
            # ATTF (Non Domain Adaptation)
            # -------------------------------        
            print("\n" + "-" * 60)
            print("ATTF [Non Domain Adaptation]")
            print("-" * 60)
            attf_lr_model = AttF(
                input_dim=X_tgt_small.shape[2],
                hidden_dim=64,
                output_steps=horizon
            )

            attf_lr_model, _ = train_attf_earlystop_target_mae(
                model=attf_lr_model,
                X_train=X_tgt_small,
                y_train=y_tgt_small,
                X_tgt_val=X_tgt_val_lr,
                y_tgt_val=y_tgt_val_lr,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device
            )

            attf_lr_metrics = attf_eval_model_metrics(attf_lr_model, X_tgt_test_lr, y_tgt_test_lr, batch_size, device)
            attf_mae_runs.append(attf_lr_metrics['mae'])
            attf_mse_runs.append(attf_lr_metrics['mse'])
            attf_rmse_runs.append(attf_lr_metrics['rmse'])

            # -------------------------------
            # DAF (Domain Adaptation)
            # -------------------------------        
            print("\n" + "-" * 60)
            print("DAF [Domain Adaptation]")
            print("-" * 60)
            daf_lr_model = DAF(
                input_dim=X_tgt_small.shape[2],
                hidden_dim=64, 
                output_steps=horizon
            )
            discriminator_lr = DomainDiscriminator(hidden_dim=64)
            daf_lr_model, _  = train_daf_earlystop_target_mae(
                model=daf_lr_model,
                discriminator=discriminator_lr,
                X_src=X_src_lr,
                y_src=y_src_lr,
                X_tgt=X_tgt_small,
                X_tgt_val=X_tgt_val_lr,
                y_tgt_val=y_tgt_val_lr,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
            )
            daf_lr_metrics = daf_eval_model_metrics(daf_lr_model, X_tgt_test_lr, y_tgt_test_lr, batch_size, device)
            daf_mae_runs.append(daf_lr_metrics['mae'])
            daf_mse_runs.append(daf_lr_metrics['mse'])
            daf_rmse_runs.append(daf_lr_metrics['rmse'])

            # -------------------------------
            # TFT (Non Domain Adaptation)
            # -------------------------------
            print("\n" + "-" * 60)
            print("TFT [Non Domain Adaptation]")
            print("-" * 60)
            X_src_lr_tft = X_src_lr.reshape(X_src_lr.shape[0], X_src_lr.shape[1], 9, 1)
            X_tgt_small_tft = X_tgt_small.reshape(X_tgt_small.shape[0], X_tgt_small.shape[1], 9, 1)
            X_tgt_val_lr_tft = X_tgt_val_lr.reshape(X_tgt_val_lr.shape[0], X_tgt_val_lr.shape[1], 9, 1)
            X_tgt_test_lr_tft = X_tgt_test_lr.reshape(X_tgt_test_lr.shape[0], X_tgt_test_lr.shape[1], 9, 1)

            tft_model_lr = TFT_Target_Model(
                input_dim=X_src_lr_tft.shape[3],
                num_vars=X_src_lr_tft.shape[2],
                hidden_dim=64,
                horizon=horizon
            )    

            tft_model_lr, _ = train_tft_non_da(
                tft_model_lr,
                X_tgt_small_tft,
                y_tgt_small,
                X_tgt_val_lr_tft,
                y_tgt_val_lr,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device
            )
            tft_lr_metrics = tft_eval_model_metrics_non_da(tft_model_lr, X_tgt_test_lr_tft, y_tgt_test_lr, batch_size, device)
            tft_non_da_mae_runs.append(tft_lr_metrics['mae'])
            tft_non_da_mse_runs.append(tft_lr_metrics['mse'])
            tft_non_da_rmse_runs.append(tft_lr_metrics['rmse'])

            print("\n" + "-" * 60)
            print("TFT [Domain Adaptation]")
            print("-" * 60)

            tft_da_model_lr = TFT_DA_Model(
                input_dim=X_src_lr_tft.shape[3],
                num_vars=X_src_lr_tft.shape[2],
                hidden_dim=64,
                horizon=horizon
            )
            tft_da_model_lr, _ = train_tft_da(
                model=tft_da_model_lr,
                X_src=X_src_lr_tft,
                y_src=y_src_lr,
                X_tgt=X_tgt_small_tft,
                y_tgt=y_tgt_small,
                X_tgt_val=X_tgt_val_lr_tft,
                y_tgt_val=y_tgt_val_lr,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device
            )

            tft_da_lr_metrics = tft_eval_model_metrics_da(tft_da_model_lr, X_tgt_test_lr_tft, y_tgt_test_lr, batch_size, device)
            tft_da_mae_runs.append(tft_da_lr_metrics['mae'])
            tft_da_mse_runs.append(tft_da_lr_metrics['mse'])
            tft_da_rmse_runs.append(tft_da_lr_metrics['rmse'])
    

            rep_out = {
                "vu tran transformer": vu_tran_non_kmm_lr_metrics,
                "vu tran kmm": vu_tran_kmm_lr_metrics,
                "attf": attf_lr_metrics,
                "daf": daf_lr_metrics,
                'tft non da': tft_lr_metrics,
                'tft da': tft_da_lr_metrics
            }

            print(f"Results for iterations no: {rep_idx + 1} & fractions: {frac}")
            for k, v in rep_out.items():
                print(f"{k:>18s}  MAE={v['mae']:.6f}  MSE={v['mse']:.6f}  RMSE={v['rmse']:.4f} ")
                result_rows = log_row(
                    experiment_name=experiment_name,
                    result_rows=result_rows,
                    run_ts=run_ts,
                    input_len=input_len,
                    horizon=horizon, 
                    stride=stride,
                    feature_cols=feature_cols,
                    phase="repeat",
                    frac_target=frac,
                    n_target=n,
                    rep_idx = rep_idx,
                    seed=seed,
                    method=k,
                    metric_mae=v['mae'],
                    metric_mse=v['mse'],
                    metric_rmse=v['rmse']
                )                
        
        # Vu Tran Transformers        
        vu_tran_transformer_mae_runs = np.array(vu_tran_transformer_mae_runs, dtype=np.float64)
        vu_tran_transformer_mse_runs = np.array(vu_tran_transformer_mse_runs, dtype=np.float64)
        vu_tran_transformer_rmse_runs = np.array(vu_tran_transformer_rmse_runs, dtype=np.float64)
        vtt_mae_mean, vtt_mae_std = float(np.mean(vu_tran_transformer_mae_runs)), float(np.std(vu_tran_transformer_mae_runs, ddof=1))
        vtt_mse_mean, vtt_mse_std = float(np.mean(vu_tran_transformer_mse_runs)), float(np.std(vu_tran_transformer_mse_runs, ddof=1))
        vtt_rmse_mean, vtt_rmse_std = float(np.mean(vu_tran_transformer_rmse_runs)), float(np.std(vu_tran_transformer_rmse_runs, ddof=1))
        # # Vu Tran KMM
        vu_tran_kmm_mae_runs = np.array(vu_tran_kmm_mae_runs, dtype=np.float64)
        vu_tran_kmm_mse_runs = np.array(vu_tran_kmm_mse_runs, dtype=np.float64)
        vu_tran_kmm_rmse_runs = np.array(vu_tran_kmm_rmse_runs, dtype=np.float64)
        vtkm_mae_mean, vtkm_mae_std = float(np.mean(vu_tran_kmm_mae_runs)), float(np.std(vu_tran_kmm_mae_runs, ddof=1))
        vtkm_mse_mean, vtkm_mse_std = float(np.mean(vu_tran_kmm_mse_runs)), float(np.std(vu_tran_kmm_mse_runs, ddof=1))
        vtkm_rmse_mean, vtkm_rmse_std = float(np.mean(vu_tran_kmm_mse_runs)), float(np.std(vu_tran_kmm_mse_runs, ddof=1))
        # attf
        attf_mae_runs = np.array(attf_mae_runs, dtype=np.float64)
        attf_mse_runs = np.array(attf_mse_runs, dtype=np.float64)
        attf_rmse_runs = np.array(attf_rmse_runs, dtype=np.float64)
        attf_mae_mean, attf_mae_std = float(np.mean(attf_mae_runs)), float(np.std(attf_mae_runs, ddof=1))
        attf_mse_mean, attf_mse_std = float(np.mean(attf_mse_runs)), float(np.std(attf_mse_runs, ddof=1))
        attf_rmse_mean, attf_rmse_std = float(np.mean(attf_rmse_runs)), float(np.std(attf_rmse_runs, ddof=1))
        # daf
        daf_mae_runs = np.array(daf_mae_runs, dtype=np.float64)
        daf_mse_runs = np.array(daf_mse_runs, dtype=np.float64)
        daf_rmse_runs = np.array(daf_rmse_runs, dtype=np.float64)
        daf_mae_mean, daf_mae_std = float(np.mean(daf_mae_runs)), float(np.std(daf_mae_runs, ddof=1))
        daf_mse_mean, daf_mse_std = float(np.mean(daf_mse_runs)), float(np.std(daf_mse_runs, ddof=1))
        daf_rmse_mean, daf_rmse_std = float(np.mean(daf_rmse_runs)), float(np.std(daf_rmse_runs, ddof=1))
        # tft non domain adaptation
        tft_non_da_mae_runs = np.array(tft_non_da_mae_runs, dtype=np.float64)
        tft_non_da_mse_runs = np.array(tft_non_da_mse_runs, dtype=np.float64)
        tft_non_da_rmse_runs = np.array(tft_non_da_rmse_runs, dtype=np.float64)
        tft_non_da_mae_mean, tft_non_da_mae_std = float(np.mean(tft_non_da_mae_runs)), float(np.std(tft_non_da_mae_runs, ddof=1))
        tft_non_da_mse_mean, tft_non_da_mse_std = float(np.mean(tft_non_da_mse_runs)), float(np.std(tft_non_da_mse_runs, ddof=1))
        tft_non_da_rmse_mean, tft_non_da_rmse_std = float(np.mean(tft_non_da_rmse_runs)), float(np.std(tft_non_da_rmse_runs, ddof=1))
        # tft domain adaptation
        tft_da_mae_runs = np.array(tft_da_mae_runs, dtype=np.float64)
        tft_da_mse_runs = np.array(tft_da_mse_runs, dtype=np.float64)
        tft_da_rmse_runs = np.array(tft_da_rmse_runs, dtype=np.float64)
        tft_da_mae_mean, tft_da_mae_std = float(np.mean(tft_da_mae_runs)), float(np.std(tft_da_mae_runs, ddof=1))
        tft_da_mse_mean, tft_da_mse_std = float(np.mean(tft_da_mse_runs)), float(np.std(tft_da_mse_runs, ddof=1))
        tft_da_rmse_mean, tft_da_rmse_std = float(np.mean(tft_da_rmse_runs)), float(np.std(tft_da_rmse_runs, ddof=1))
         
        
        ### Non DA & DA Wilcoxon
        p_vtt_vtkm_wilc_mae, p_vtt_vtkm_wilc_mse, p_vtt_vtkm_wilc_rmse = wilcoxon_signed_rank(
            method1_mae=vu_tran_transformer_mae_runs, 
            method2_mae=vu_tran_kmm_mae_runs, 
            method1_mse=vu_tran_transformer_mse_runs,
            method2_mse=vu_tran_kmm_mse_runs,
            method1_rmse=vu_tran_transformer_rmse_runs,
            method2_rmse=vu_tran_kmm_rmse_runs
        )

        p_attf_daf_wilc_mae, p_attf_daf_wilc_mse, p_attf_daf_wilc_rmse = wilcoxon_signed_rank(
            method1_mae=attf_mae_runs, 
            method2_mae=daf_mae_runs, 
            method1_mse=attf_mse_runs,
            method2_mse=daf_mse_runs,
            method1_rmse=attf_rmse_runs,
            method2_rmse=daf_rmse_runs
        
        )
        p_tft_wilc_mae, p_tft_wilc_mse, p_tft_wilc_rmse = wilcoxon_signed_rank(
            method1_mae=tft_non_da_mae_runs, 
            method2_mae=tft_da_mae_runs, 
            method1_mse=tft_non_da_mse_runs,
            method2_mse=tft_da_mse_runs,
            method1_rmse=tft_da_rmse_runs,
            method2_rmse=tft_da_rmse_runs
        )

        ### TFT DA & KMM || TFT DA & DAF
        p_tft_kmm_mae, p_tft_kmm_mse, p_tft_kmm_rmse = wilcoxon_signed_rank(
            method1_mae=tft_da_mae_runs,
            method2_mae=vu_tran_kmm_mae_runs,
            method1_mse=tft_da_mse_runs,
            method2_mse=vu_tran_kmm_mse_runs,
            method1_rmse=tft_da_rmse_runs,
            method2_rmse=vu_tran_kmm_rmse_runs
        )
        p_tft_daf_mae, p_tft_daf_mse, p_tft_daf_rmse = wilcoxon_signed_rank(
            method1_mae=tft_da_mae_runs,
            method2_mae=daf_mae_runs,
            method1_mse=tft_da_mse_runs,
            method2_mse=daf_mse_runs,
            method1_rmse=tft_da_rmse_runs,
            method2_rmse=daf_rmse_runs
        )

        frac_results[frac] = {
            "n" : n,
            "vu tran transformer mae mean" : vtt_mae_mean,
            "vu tran transformer mae std" : vtt_mae_std,
            "vu tran kmm mae mean": vtkm_mae_mean,
            "vu tran kmm mae std": vtkm_mae_std,
            "vu tran transformer mse mean" : vtt_mse_mean,
            "vu tran transformer mse std" : vtt_mse_std,
            "vu tran kmm mse mean": vtkm_mse_mean,
            "vu tran kmm mse std": vtkm_mse_std,
            "vu tran transformer rmse mean" : vtt_rmse_mean,
            "vu tran transformer rmse std" : vtt_rmse_std,
            "vu tran kmm rmse mean": vtkm_rmse_mean,
            "vu tran kmm rmse std": vtkm_rmse_std,
            "attf mae mean": attf_mae_mean,
            "attf mae std" : attf_mae_std,
            "attf mse mean" : attf_mse_mean,
            "attf mse std" : attf_mae_std,
            "attf rmse mean" : attf_rmse_mean,
            "attf rmse std" : attf_rmse_std,
            "daf mae mean" : daf_mae_mean,
            "daf mae std" : daf_mae_std,
            "daf mse mean" : daf_mse_mean,
            "daf mse std" : daf_mse_std,
            "daf rmse mean" : daf_rmse_mean,
            "daf rmse std" : daf_rmse_std,
            "tft non da mae mean" : tft_non_da_mae_mean,
            "tft non da mae std" : tft_non_da_mae_std,
            "tft non da mse mean" : tft_non_da_mse_mean,
            "tft non da mse std" : tft_non_da_mse_std,
            "tft non da rmse mean" : tft_non_da_rmse_mean,
            "tft non da rmse std" : tft_non_da_rmse_std,
            "tft da mae mean" : tft_da_mae_mean,
            "tft da mae std" : tft_da_mae_std,
            "tft da mse mean" : tft_da_mse_mean,
            "tft da mse std" : tft_da_mse_std,
            "tft da rmse mean": tft_da_rmse_mean,
            "tft da rmse std" : tft_da_rmse_std,
            "vtt vtkm p wilcoxon mae" : float(p_vtt_vtkm_wilc_mae),
            "vtt vtkm p wilcoxon mse" : float(p_vtt_vtkm_wilc_mse),
            "vtt vtkm p wilcoxon rmse" : float(p_vtt_vtkm_wilc_rmse),
            "attf daf p wilcoxon mae" : float(p_attf_daf_wilc_mae),
            "attf daf p wilcoxon mse" : float(p_attf_daf_wilc_mse),
            "attf daf p wilcoxon rmse" : float(p_attf_daf_wilc_rmse),
            "tft p wilcoxon mae" : float(p_tft_wilc_mae),
            "tft p wilcoxon mse" : float(p_tft_wilc_mse),
            "tft p wilcoxon rmse" : float(p_tft_wilc_rmse),
            "tft kmm wilxocon mae" : float(p_tft_kmm_mae),
            "tft kmm wilxocon mse" : float(p_tft_kmm_mse),
            "tft kmm wilxocon rmse" : float(p_tft_kmm_rmse),
            "tft daf wilcoxon mae" : float(p_tft_daf_mae),
            "tft daf wilcoxon mse" : float(p_tft_daf_mse),
            "tft daf wilxocon rmse" : float(p_tft_daf_rmse)
        }
    
        result_rows = log_row(
            experiment_name=experiment_name,
            result_rows=result_rows,
            run_ts=run_ts,
            input_len=input_len,
            horizon=horizon, 
            stride=stride,
            feature_cols=feature_cols,
            phase="summary",
            frac_target=frac,
            n_target=n,
            rep_idx = rep_idx,
            seed=seed,
            method=k,
            vtt_mae_mean=vtt_mae_mean,
            vtt_mae_std=vtt_mae_std,
            vtt_mse_mean=vtt_mae_mean,
            vtt_mse_std=vtt_mse_std,
            vtt_rmse_mean=vtt_rmse_mean,
            vtt_rmse_std=vtt_rmse_std,
            vtkm_mae_mean=vtkm_mae_mean,
            vtkm_mae_std=vtkm_mae_std,
            vtkm_mse_mean = vtkm_mse_mean,
            vtkm_mse_std=vtkm_mse_std,
            vtkm_rmse_mean=vtkm_rmse_mean,
            vtkm_rmse_std=vtkm_rmse_std,
            attf_mae_mean = attf_mae_mean,
            attf_mae_std = attf_mae_std,
            attf_mse_mean = attf_mse_mean,
            attf_mse_std = attf_mse_std,
            attf_rmse_mean = attf_rmse_mean,
            attf_rmse_std = attf_rmse_std,
            daf_mae_mean = daf_mae_mean,
            daf_mae_std = daf_mae_std,
            daf_mse_mean = daf_mse_mean,
            daf_mse_std = daf_mse_std,
            daf_rmse_mean = daf_rmse_mean,
            daf_rmse_std = daf_rmse_std,
            tft_non_da_mae_mean=tft_non_da_mae_mean,
            tft_non_da_mae_std =tft_non_da_mae_std,
            tft_non_da_mse_mean=tft_non_da_mse_mean,
            tft_non_da_mse_std=tft_non_da_mse_std,
            tft_non_da_rmse_mean=tft_non_da_rmse_mean,
            tft_non_da_rmse_std=tft_non_da_rmse_std,
            tft_da_mae_mean=tft_da_mae_mean,
            tft_da_mae_std=tft_da_mae_std,
            tft_da_mse_mean=tft_da_mse_mean,
            tft_da_mse_std=tft_da_mae_std,
            tft_da_rmse_mean=tft_da_rmse_mean,
            tft_da_rmse_std=tft_da_rmse_std,
            wilcoxon_p_vtt_vtkm_mae=float(p_vtt_vtkm_wilc_mae),
            wilcoxon_p_vtt_vtkm_mse=float(p_vtt_vtkm_wilc_mse),
            wilcoxon_p_vtt_vtkm_rmse=float(p_vtt_vtkm_wilc_rmse),
            wilcoxon_p_attf_daf_mae=float(p_attf_daf_wilc_mae),
            wilcoxon_p_attf_daf_mse=float(p_attf_daf_wilc_mse),
            wilcoxon_p_attf_daf_rmse=float(p_attf_daf_wilc_rmse),
            wilcoxon_p_tft_mae=float(p_tft_wilc_mae),
            wilcoxon_p_tft_mse=float(p_tft_wilc_mse),
            wilcoxon_p_tft_rmse=float(p_tft_wilc_rmse),
            wilcoxon_p_tft_kmm_mae=float(p_tft_kmm_mae),
            wilcoxon_p_tft_kmm_mse=float(p_tft_kmm_mse),
            wilcoxon_p_tft_kmm_rmse=float(p_tft_kmm_rmse),
            wilcoxon_p_tft_daf_mae=float(p_tft_daf_mae),
            wilcoxon_p_tft_daf_mse=float(p_tft_daf_mse),
            wilcoxon_p_tft_daf_rmse=float(p_tft_daf_rmse),
        )
    if frac_results:
        methods = [
            "vu tran transformer",
            "vu tran kmm",
            "attf",
            "daf",
            "tft non da",
            "tft da"
        ]
        metrics = ["mae", "mse", "rmse"]

        comparisons = [
            ("TFT vs TFT-DA", "tft p wilcoxon"),
            ("KMM vs TFT-DA", "tft kmm wilxocon"),
            ("DAF vs TFT-DA", "tft daf wilcoxon"),
        ]
        header = (
            f"{'IN':>4}  {'%T':>4}  {'n':>5}  "
        )
        for name in methods:
            for m in metrics:
                header += f"{name+' '+m.upper():>18}  "
        
        for _, comp_key in comparisons:
            for m in metrics:
                header += f"{comp_key+' '+m.upper():>18}  "

        print(header)
        for frac in frac_results:
            r = frac_results[frac]
            row = (
            f"{input_len:>6d}  {int(frac*100):>6d}  {r['n']:>6d}  "
            )
        
        for name in methods:
            for m in metrics:
                mean = r[f"{name} {m} mean"]
                std = r[f"{name} {m} std"]
                row += f"{fmt(mean, std):>18} "
        
        for _, comp_key in comparisons:
            for m in metrics:
                p_val = f"{comp_key} {m}"
                row += f"{fmt_p(p_val):>18} "    
    
    return result_rows

