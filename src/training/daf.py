import torch.nn as nn
import torch
import numpy as np

from src.models.DAF.grad_reverse import grad_reverse
from src.evaluation.daf import daf_eval_model_mae

from torch.utils.data import DataLoader, TensorDataset

def train_daf_earlystop_target_mae(
    model: nn.Module,
    discriminator: nn.Module,
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_tgt: np.ndarray,
    X_tgt_val: np.ndarray,
    y_tgt_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    lambda_recon: float = 0.5,
    lambda_domain: float = 0.1,
):
    """
    DAF Training:
    - Source supervised (forecast)
    - Target unsupervised (reconstruction)
    - Domain adversarial alignment (pattern space)
    - Early stop on target MAE
    """
    
    model = model.to(device).float()
    discriminator = discriminator.to(device).float()

    src_ds = TensorDataset(
        torch.from_numpy(X_src).to(device),
        torch.from_numpy(y_src).to(device)
    )
    tgt_ds = TensorDataset(
        torch.from_numpy(X_tgt).to(device)
    )

    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt_model = torch.optim.Adam(model.parameters(), lr=lr)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr)

    bce = nn.BCEWithLogitsLoss()

    best_mae, best_state = float("inf"), None

    for ep in range(epochs):
        model.train()
        discriminator.train()

        tgt_iter = iter(tgt_loader)

        total_loss = 0.0

        for xb_src, yb_src in src_loader:
            try:
                xb_tgt = next(tgt_iter)[0]
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xb_tgt = next(tgt_iter)[0]

            y_pred_src, x_recon_src, p_src = model(xb_src)
            _, x_recon_tgt, p_tgt = model(xb_tgt)

            loss_forecast = nn.functional.mse_loss(y_pred_src, yb_src)

            loss_recon_src = nn.functional.mse_loss(x_recon_src, xb_src)
            loss_recon_tgt = nn.functional.mse_loss(x_recon_tgt, xb_tgt)

            loss_recon = (loss_recon_src + loss_recon_tgt) / 2

            # Pool temporal dimension
            p_src_mean = torch.mean(p_src, dim=1)
            p_tgt_mean = torch.mean(p_tgt, dim=1)

            # Labels
            domain_src = torch.zeros(p_src_mean.size(0), 1).to(device)
            domain_tgt = torch.ones(p_tgt_mean.size(0), 1).to(device)

            d_src = discriminator(p_src_mean.detach())
            d_tgt = discriminator(p_tgt_mean.detach())

            loss_disc = bce(d_src, domain_src) + bce(d_tgt, domain_tgt)

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            p_src_rev = grad_reverse(p_src_mean)
            p_tgt_rev = grad_reverse(p_tgt_mean)

            d_src = discriminator(p_src_rev)
            d_tgt = discriminator(p_tgt_rev)

            loss_domain = bce(d_src, domain_tgt) + bce(d_tgt, domain_src)

            loss = (
                loss_forecast
                + lambda_recon * loss_recon
                + lambda_domain * loss_domain
            )

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

            total_loss += loss.item()

        val_mae = daf_eval_model_mae(model, X_tgt_val, y_tgt_val, batch_size, device)

        print(
            f"Epoch {ep+1}/{epochs} | "
            f"Loss: {total_loss:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.to(device), best_mae