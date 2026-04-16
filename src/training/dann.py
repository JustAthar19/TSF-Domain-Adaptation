import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


from src.evaluation.supervised import eval_model_mae
from src.models.DANN.gradient_adversarial_layer import grl

def dann_lambda_schedule(p: float) -> float:
    import math
    # λ = 2 / (1 + exp(-10 p)) - 1
    return float(2.0 / (1.0 + math.exp(-10.0 * float(p))) - 1.0)


def train_domain_adversarial_dann(
    feat_extractor: nn.Module,
    task_head: nn.Module,
    domain_clf: nn.Module,
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_tgt: np.ndarray,
    y_tgt: np.ndarray,
    X_tgt_val: np.ndarray,
    y_tgt_val: np.ndarray,
    config: dict,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    use_target_task_loss: bool = False,
):
    """
    DANN:
      loss = forecast_MSE + λ * domain_BCE
      λ schedule: 2/(1+exp(-10p)) - 1, p=step/total_steps
      balanced batches: batch_size/2 source + batch_size/2 target
      early stop: target validation MAE
    """
    if X_src.shape[0] == 0 or X_tgt.shape[0] == 0:
        return None, float("nan")

    src_bs = max(1, batch_size // 2)
    tgt_bs = max(1, batch_size // 2)

    src_ds = TensorDataset(torch.from_numpy(X_src).to(config["device"]), torch.from_numpy(y_src).to(config["device"]))
    tgt_ds = TensorDataset(torch.from_numpy(X_tgt).to(config["device"]), torch.from_numpy(y_tgt).to(config["device"]))
    src_loader = DataLoader(src_ds, batch_size=src_bs, shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=tgt_bs, shuffle=True, num_workers=0, drop_last=True)

    feat_extractor = feat_extractor.to(config["device"]).float()
    task_head = task_head.to(config["device"]).float()
    domain_clf = domain_clf.to(config["device"]).float()

    params = list(feat_extractor.parameters()) + list(task_head.parameters()) + list(domain_clf.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    bce = nn.BCELoss()

    total_steps = epochs * min(len(src_loader), len(tgt_loader))
    global_step = 0

    best_mae, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        feat_extractor.train()
        task_head.train()
        domain_clf.train()

        tr_task, tr_dom = 0.0, 0.0
        n_steps = min(len(src_loader), len(tgt_loader))
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        for _ in range(n_steps):
            xs, ys = next(src_iter)
            xt, yt = next(tgt_iter)
            xb = torch.cat([xs, xt], dim=0)
            dom_y = torch.cat(
                [torch.zeros(xs.size(0), device=config["device"]), torch.ones(xt.size(0), device=config["device"])],
                dim=0,
            ).float()

            p = global_step / max(1, total_steps)
            lambd = dann_lambda_schedule(p)
            global_step += 1

            opt.zero_grad()
            rep = feat_extractor(xb)
            yhat = task_head(rep)

            task_loss = nn.functional.mse_loss(yhat[: xs.size(0)], ys)
            if use_target_task_loss:
                task_loss = task_loss + nn.functional.mse_loss(yhat[xs.size(0) :], yt)

            dom_pred = domain_clf(grl(rep, lambd))
            dom_loss = bce(dom_pred, dom_y)

            loss = task_loss + (lambd * dom_loss)
            loss.backward()
            opt.step()

            tr_task += float(task_loss.item())
            tr_dom += float(dom_loss.item())

        eval_model = nn.Sequential(feat_extractor, task_head)
        val_mae = eval_model_mae(eval_model, X_tgt_val, y_tgt_val, config, batch_size=256)
        tr_task /= max(1, n_steps)
        tr_dom /= max(1, n_steps)
        print(f"  Epoch {ep+1}/{epochs}  Task MSE: {tr_task:.6f}  Domain BCE: {tr_dom:.6f}  Target Val MAE: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {
                "feat": {k: v.cpu().clone() for k, v in feat_extractor.state_dict().items()},
                "task": {k: v.cpu().clone() for k, v in task_head.state_dict().items()},
                "dom": {k: v.cpu().clone() for k, v in domain_clf.state_dict().items()},
            }
        #     wait = 0
        # else:
        #     wait += 1
        #     if wait >= patience:
        #         print(f"  Early stopping at epoch {ep+1}")
        #         break

    if best_state is not None:
        feat_extractor.load_state_dict(best_state["feat"])
        task_head.load_state_dict(best_state["task"])
        domain_clf.load_state_dict(best_state["dom"])

    return nn.Sequential(feat_extractor, task_head).to(config["device"]), best_mae

