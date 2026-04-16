from datetime import datetime
from scipy.stats import wilcoxon as wilcoxon_signed_rank

import pandas as pd
import numpy as np

from src.data.windowing import build_windows, build_windows_temp_transformer

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




_has_wilcoxon = True
low_resources_fracs  = [0.05, 0.15, 0.25]

def run_target_station_experiment(
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
            kmm_vanilla_mae_runs, kmm_vanilla_runs, kmm_vanilla_runs = [], [], []
            kmm_vu_tran_runs, kmm_vu_tran_runs, kmm_vu_tran_rmse_runs = [], [], []
            
            for rep_idx in range(n_repeats):
                seed = int(rs_base.randint(0, 2**31 -1))
                rs = np.random.RandomState(seed)
                idx = rs.choice(X_tgt_full.shape[0], n, replace=False)
                X_tgt_small = X_tgt_full[idx]
                y_tgt_small = y_tgt_full[idx]

                # Vanilla Transformer 
                