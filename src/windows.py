import numpy as np
from .config import *

def build_windows_one_location(values):
    T = len(values)
    X, y = [], []
    for start in range(0, T - INPUT_LEN - HORIZON + 1, STRIDE):
        end_in = start + INPUT_LEN
        end_out = end_in + HORIZON
        X.append(values[start:end_in])
        y.append(values[end_in:end_out,0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_all_windows(split_df):
    X_all, y_all = [], []
    for lid, grp, in split_df.groupby("location_id"):
        mat = grp['FEATURE_COLS'].values.astype(np.float32)
        X, y = build_windows_one_location(mat)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
    if not X_all:
        return None, None
    return np.concatenate(X_all), np.concatenate(y_all)      

