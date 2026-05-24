import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler


def fit_transform_source( src_df: pd.DataFrame, feature_cols: list) -> Tuple[pd.DataFrame, StandardScaler]:
  """Fit StandardScaler on source domain and return scaled copy."""
  scaler = StandardScaler()
  out = src_df.copy()
  out[feature_cols] = scaler.fit_transform(src_df[feature_cols].values.astype("float32"))
  return out, scaler


def transform_target( tgt_df: pd.DataFrame, feature_cols: list, scaler: StandardScaler,
) -> pd.DataFrame:
  """Transform target domain with a source-fitted scaler."""
  out = tgt_df.copy()
  out[feature_cols] = scaler.transform(tgt_df[feature_cols].values.astype("float32"))
  return out
