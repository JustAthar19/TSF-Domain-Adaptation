
import numpy as np
import pandas as pd


def build_windows_one_location(times, values, input_len, horizon, stride):
    """Build (X, y) windows for one location. values: (T, n_features)."""
    T = len(times)
    # if the station does not have enough values to form a full window # we skip
    if T < input_len + horizon:
        return None, None
    X_list, y_list = [], []
    for start in range(0, T - input_len - horizon + 1, stride):
        end_in = start + input_len
        end_out = end_in + horizon
        X_list.append(values[start:end_in])
        y_list.append(values[end_in:end_out, 0])
    if not X_list:
        return None, None
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_windows_temp_transformer_one_location(times, values, input_len, horizon, stride):
    """Build windows for `TemperatureTransformer`.

    Returns:
      - z_past: (input_len, 1) temperature history
      - x_cov_past: (input_len, d_cov) past covariates (features 1:)
      - x_cov_future: (horizon, d_cov) future covariates (features 1:)
      - y: (horizon,) future temperature target (feature 0)

    `values` must be shaped (T, n_features) where feature 0 is temperature.
    """
    T = len(times)
    if T < input_len + horizon:
        return None, None, None, None

    z_past_list, x_cov_past_list, x_cov_future_list, y_list = [], [], [], []

    for start in range(0, T - input_len - horizon + 1, stride):
        end_in = start + input_len
        end_out = end_in + horizon

        past = values[start:end_in]       # (input_len, n_features)
        future = values[end_in:end_out]  # (horizon, n_features)

        z_past_list.append(past[:, :1])
        x_cov_past_list.append(past[:, 1:])
        x_cov_future_list.append(future[:, 1:])
        y_list.append(future[:, 0])

    if not z_past_list:
        return None, None, None, None

    return (
        np.array(z_past_list, dtype=np.float32),
        np.array(x_cov_past_list, dtype=np.float32),
        np.array(x_cov_future_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )


def stack_for_temperature(z_past, x_cov_past, x_cov_future, horizon):
    zeros_temp_future = np.zeros((z_past.shape[0], horizon, 1), dtype=np.float32)
    past = np.concatenate([z_past, x_cov_past], axis=-1)
    future = np.concatenate([zeros_temp_future, x_cov_future], axis=-1)
    return np.concatenate([past, future], axis=1)


def build_windows(split_df: pd.DataFrame, input_len: int, horizon: int, stride: int, feature_cols: list):
    """Build(X, y) for a single split dataframe (across all location_id)"""
    X_list, y_list = [], []
    for _, grp in split_df.groupby("location_id"):
        grp = grp.sort_values("time")
        mat = grp[feature_cols].values.astype(np.float32)
        times = grp['time'].values
        X, y = build_windows_one_location(times, mat, input_len, horizon, stride)
        if X is not None:
            X_list.append(X)
            y_list.append(y)
    if not X_list:
        return (
            np.zeros((0, input_len, len(feature_cols)), dtype=np.float32),
            np.zeros((0, horizon), dtype=np.float32)
        )
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def build_windows_temp_transformer(split_df: pd.DataFrame, feature_cols: list, input_len: int, horizon: 7, stride: int):
    """Build windows for `TemperatureTransformer` across all `location_id`.

    Returns:
      - z_past: (N, INPUT_LEN, 1)
      - x_cov_past: (N, INPUT_LEN, d_cov) where d_cov=len(FEATURE_COLS)-1
      - x_cov_future: (N, HORIZON, d_cov)
      - y: (N, HORIZON) future temperature target
    """
    z_list, x_cov_past_list, x_cov_future_list, y_list = [], [], [], []

    d_cov = len(feature_cols) - 1

    for _, grp in split_df.groupby("location_id"):
        grp = grp.sort_values("time")
        mat = grp[feature_cols].values.astype(np.float32)
        times = grp['time'].values

        z_past, x_cov_past, x_cov_future, y = build_windows_temp_transformer_one_location(
            times, mat, input_len, horizon, stride
        )

        if z_past is not None:
            z_list.append(z_past)
            x_cov_past_list.append(x_cov_past)
            x_cov_future_list.append(x_cov_future)
            y_list.append(y)

    if not z_list:
        return (
            np.zeros((0, input_len, 1), dtype=np.float32),
            np.zeros((0, input_len, d_cov), dtype=np.float32),
            np.zeros((0, horizon, d_cov), dtype=np.float32),
            np.zeros((0, horizon), dtype=np.float32),
        )

    return (
        np.concatenate(z_list, axis=0),
        np.concatenate(x_cov_past_list, axis=0),
        np.concatenate(x_cov_future_list, axis=0),
        np.concatenate(y_list, axis=0),
    )


