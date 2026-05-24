import numpy as np
import pandas as pd
import torch


def _rbf_kernel(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
  aa = (a * a).sum(1, keepdim=True)
  bb = (b * b).sum(1, keepdim=True)
  dist2 = aa + bb.T - 2.0 * (a @ b.T)
  return torch.exp(-dist2 / (2.0 * sigma ** 2))


def compute_mmd( src_values: np.ndarray, tgt_values: np.ndarray, subsample: int = 3000, sigma: float = 1.0, seed: int = 42) -> float:
  """Estimate sqrt(MMD²) on the target variable (1-D arrays)."""
  x = np.asarray(src_values, dtype=np.float32).ravel()
  y = np.asarray(tgt_values, dtype=np.float32).ravel()
  rng = np.random.default_rng(seed)
  if len(x) > subsample:
    x = x[rng.choice(len(x), subsample, replace=False)]
  if len(y) > subsample:
    y = y[rng.choice(len(y), subsample, replace=False)]

  xt = torch.tensor(x[:, None], dtype=torch.float32)
  yt = torch.tensor(y[:, None], dtype=torch.float32)
  n, m = len(xt), len(yt)

  kxx = _rbf_kernel(xt, xt, sigma)
  kyy = _rbf_kernel(yt, yt, sigma)
  kxy = _rbf_kernel(xt, yt, sigma)
  mmd2 = kxx.sum() / (n * n) + kyy.sum() / (m * m) - 2.0 * kxy.sum() / (n * m)
  return float(mmd2.clamp(min=0.0).sqrt())


def mmd_to_mix_pct(mmd_val: float, threshold: float = 0.11) -> float:
  """Paper ablation heuristic: high MMD -> 5% source mix, low MMD -> 20%."""
  return 0.05 if mmd_val > threshold else 0.20


def mix_source_into_target_df(src_df: pd.DataFrame, tgt_df: pd.DataFrame, mix_pct: float, seed: int = 42) -> pd.DataFrame:
  """Sample source rows and prepend to target training rows."""
  rng = np.random.default_rng(seed)
  n_add = max(1, round(len(src_df) * mix_pct))
  idx = rng.choice(len(src_df), n_add, replace=False)
  mixed = pd.concat([src_df.iloc[idx], tgt_df], ignore_index=True)
  print(
    f"    Mixing: +{n_add:,} source rows ({mix_pct * 100:.0f}% of "
    f"{len(src_df):,}) into {len(tgt_df):,} target rows "
    f"-> {len(mixed):,} total"
  )
  return mixed
