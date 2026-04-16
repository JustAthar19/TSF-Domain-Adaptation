import pandas as pd
import os
from datetime import datetime

def log_row(experiment_name, result_rows,run_ts, config, **kwargs):
    row = {
            "experiment": experiment_name,
            "run_timestamp": run_ts,
            "INPUT_LEN": config["input_len"],
            "HORIZON": config["horizon"],
            "STRIDE": config["stride"],
            "n_features": len(config["feature_cols"]),
    }
    row.update(kwargs)
    result_rows.append(row)
    return result_rows


def save_to_csv(run_ts: datetime, results_rows: list):
    df_out = pd.DataFrame(results_rows)
    run_ts = run_ts.replace(" ", "").replace(":","-")
    out_dir = "results"
    out_path = os.path.join(out_dir, f"results_{run_ts}.csv")

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
    df_out.to_csv(out_path, index=False)

    