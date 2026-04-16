import pandas as pd
import numpy as np


def compute_station_stats(papua_train: pd.DataFrame, config: dict):
    if papua_train is None or len(papua_train) == 0:
        return pd.DataFrame(columns=["location_id", "mean_temp", "std_temp", "elevation", "n_rows"])
    df = papua_train.copy()
    df['location_id'] = df['location_id'].astype(str)
    g = df.groupby("location_id", dropna=False)
    
    stats = g.agg(
        mean_temp=(config["target_col"], "mean"),
        std_temp=(config["target_col"], "std"),
        elevation=("elevation", "mean"),
        n_rows=(config["target_col"], "size"),
    ).reset_index()
    
    stats["mean_temp"] = stats["mean_temp"].astype(np.float32)
    stats["std_temp"] = stats["std_temp"].fillna(0.0).astype(np.float32)
    stats["elevation"] = stats["elevation"].astype(np.float32)
    stats["n_rows"] = stats["n_rows"].astype(int)
    return stats

def select_target_stations_papua_by_elevation(papua_train: pd.DataFrame, config: dict):
    """
    Select station IDs from Papua training set by elevation:
      - Single station: median elevation station
      - Three stations: lowest, median, highest elevation stations
    Returns (single_station_id: str, three_station_ids: list[str], stats_sorted: pd.DataFrame).
    """
    stats = compute_station_stats(papua_train, config)
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

