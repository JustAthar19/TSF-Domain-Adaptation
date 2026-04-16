import numpy as np
from torch.fx import experimental
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import single_station as tsa 


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
torch.set_default_dtype(torch.float32)
torch.set_num_threads(DEFAULT_CPU_THREADS)
# torch.set_num_interop_threads(max(1, min(4, DEFAULT_CPU_THREADS // 2)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from src.training.arima_baseline import arima_baseline_rolling 


if DEVICE.type == "cuda":
    # Ampere GPU (3090 Ti): TF32 is a big speedup for matmul/conv with minimal impact for this task.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

def arima_one_three_stations_exp(
    experiment_name: str,
    train_papua_df: pd.DataFrame,
    test_papua_df: pd.DataFrame,
    target_station_ids,
):
    print("\n" + "=" * 60)
    print(f"{experiment_name}")
    print("=" * 60)
    print(f"Selected Papua target stations (train only): {', '.join([str(x) for x in target_station_ids])}")
    
    train_papua_sel = tsa.filter_df_by_station_ids(train_papua_df, target_station_ids)
    test_papua_sel = tsa.filter_df_by_station_ids(test_papua_df, target_station_ids)

    # arima_papua_mae, arima_papua_rmse = tsa.arima_baseline_rolling(test_papua_sel,train_papua_sel)
    arima_papua_mae, arima_papua_rmse = arima_baseline_rolling(test_papua_sel,train_papua_sel)
    arima_papua_mse = float(arima_papua_rmse ** 2)
    return arima_papua_mae, arima_papua_rmse, arima_papua_mse

  


def main():
    data_path = "data/merged_dataset.csv"
    
    tsa.set_seed(42)

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
            "TARGET_COL": tsa.TARGET_COL,
            "INPUT_LEN": tsa.INPUT_LEN,
            "HORIZON": tsa.HORIZON,
            "STRIDE": tsa.STRIDE,
            "n_features": len(tsa.FEATURE_COLS),
        }
        row.update(kwargs)
        results_rows.append(row)

    print("=" * 60)
    print("PART 1 — DATA PREPROCESSING (time-separated splits)")
    print("=" * 60)
    train, val, test = tsa.load_and_split(data_path)
    print(f"Train: {train['time'].min()} to {train['time'].max()}  rows: {len(train)}")
    print(f"Val:   {val['time'].min()} to {val['time'].max()}  rows: {len(val)}")
    print(f"Test:  {test['time'].min()} to {test['time'].max()}  rows: {len(test)}")

    # Domain dataframes (source=Java; target=Papua)
    train_papua = train[tsa.region_mask(train, "papua")]
    test_papua = test[tsa.region_mask(test, "papua")]

    print("\n" + "=" * 60)
    print("PART 2 — STATION SELECTION LOGIC (Papua target)")
    print("=" * 60)
    single_id, three_ids, stats_sorted = tsa.select_target_stations_papua_by_elevation(train_papua)
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

    print("\n" + "=" * 60)
    print("Arima Single Station Experiment")
    single_arima_papua_mae, single_arima_papua_rmse, single_arima_papua_mse = arima_one_three_stations_exp(
        experiment_name="Single-Station", 
        train_papua_df = train_papua,
        test_papua_df = test_papua, 
        target_station_ids=[single_id])
    log_row(
        experiment="Single_Station",
        target_station_ids="|".join([str(single_id)]),
        method="ARIMA",
        metric_mae=single_arima_papua_mae,
        matric_mse=single_arima_papua_mse,
        metric_rmse=single_arima_papua_rmse
    )
    print("Single STATION RESULT")
    print("="*60)
    print(f"mae: {single_arima_papua_mae}\n mse:{single_arima_papua_mse}\n rmse{single_arima_papua_rmse}")
    print("="*60)
    print("Arima Three Station Experiment")
    three_arima_papua_mae, three_arima_papua_rmse, three_arima_papua_mse = arima_one_three_stations_exp(
        experiment_name="Three-Station",
        train_papua_df=train_papua, 
        test_papua_df=test_papua, 
        target_station_ids=three_ids)
    log_row(
        experiment="Three-Station",
        target_station_ids="|".join([str(x) for x in three_ids]),
        method="ARIMA",
        metric_mae=three_arima_papua_mae,
        matric_mse=three_arima_papua_mse,
        metric_rmse=three_arima_papua_rmse
    )

    print("Single STATION RESULT")
    print("="*60)
    print(f"mae: {three_arima_papua_mae}\n mse:{three_arima_papua_mse}\n rmse{three_arima_papua_rmse}")
    print("="*60)
    print("Arima Three Station Experiment")
    print("Save to CSV")
    df_out = pd.DataFrame(results_rows)
    df_out.to_csv("arima_single_three_station_resuls.csv", index=False)

    print("=" * 60)


if __name__ == "__main__":
    main()


