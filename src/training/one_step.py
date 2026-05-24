import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.one_step.tst_transformer import TSTransformer


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    train: bool,
    device: str,
) -> float:
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
    return total / max(len(loader.dataset), 1)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _build_gu_schedule(n_groups: int, n_epochs: int) -> List[Tuple[int, int]]:
    step = max(1, n_epochs // n_groups)
    return [(i * step + 1, i + 1) for i in range(n_groups)]


def pretrain_source_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    input_len: int,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    patience: int = 10,
    d_model: int = 64,
    n_heads: int = 4,
    n_encoder_layers: int = 3,
    d_ff: int = 256,
    dropout: float = 0.1,
) -> TSTransformer:
    model = TSTransformer(
        n_features=n_features,
        input_len=input_len,
        horizon=horizon,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True, device=device)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False, device=device)

    best_val, best_state, no_improve = float("inf"), None, 0

    print(f"  Pre-training source  epochs={epochs}  lr={lr}  batch={batch_size}")
    for ep in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, criterion, optimizer, train=True, device=device)
        val = _run_epoch(model, val_loader, criterion, None, train=False, device=device)

        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if ep % 5 == 0 or ep == 1:
            print(f"    ep {ep:>3}/{epochs}  train={tr:.4f}  val={val:.4f}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {ep}  (best val={best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model


def finetune_on_target(
    source_model: TSTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    patience: int = 10,
    region_name: str = "target",
) -> TSTransformer:
    model = copy.deepcopy(source_model).to(device)
    criterion = nn.L1Loss()
    n_groups = len(model.layer_groups())
    schedule = _build_gu_schedule(n_groups, epochs)

    model.freeze_all()
    model.unfreeze_top_n_groups(1)
    cur_unfrozen = 1
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True, device=device)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False, device=device)

    best_val, best_state, no_improve = float("inf"), None, 0

    print(f"  Fine-tuning {region_name}  groups={n_groups}  schedule={schedule}")
    for ep in range(1, epochs + 1):
        target_unfrozen = cur_unfrozen
        for start_ep, n_unf in schedule:
            if ep >= start_ep and n_unf > target_unfrozen:
                target_unfrozen = n_unf

        if target_unfrozen > cur_unfrozen:
            cur_unfrozen = target_unfrozen
            model.unfreeze_top_n_groups(cur_unfrozen)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr
            )
            print(f"    [GU] ep {ep:>2}: unfroze top-{cur_unfrozen}/{n_groups} groups")

        tr = _run_epoch(model, train_loader, criterion, optimizer, train=True, device=device)
        val = _run_epoch(model, val_loader, criterion, None, train=False, device=device)

        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if ep % 5 == 0:
            print(f"    ep {ep:>3}/{epochs}  train={tr:.4f}  val={val:.4f}")

        # if no_improve >= patience:
        #     print(f"    Early stopping at epoch {ep}  (best val={best_val:.4f})")
        #     break

    model.load_state_dict(best_state)
    return model
