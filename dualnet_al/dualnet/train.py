"""Training, inference and evaluation utilities."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def augment(x: torch.Tensor, flip: bool) -> torch.Tensor:
    """Random 2-pixel translation (+ horizontal flip for natural images)."""
    n, _, h, w = x.shape
    pad = 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    dx, dy = np.random.randint(0, 2 * pad + 1, 2)
    x = xp[:, :, dy:dy + h, dx:dx + w]
    if flip and np.random.rand() < 0.5:
        x = torch.flip(x, dims=[3])
    return x


def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device, steps: int = 300,
                bs: int = 64, lr: float = 2e-3, wd: float = 5e-4, flip: bool = False,
                class_weight: torch.Tensor | None = None) -> float:
    """Train from scratch for a fixed number of SGD steps (budget-independent compute)."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    Xd, yd = X.to(device), y.to(device)
    cw = class_weight.to(device) if class_weight is not None else None
    n = len(Xd)
    loss_v = 0.0
    for s in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), device=device)
        xb = augment(Xd[idx], flip)
        loss = F.cross_entropy(model(xb), yd[idx], weight=cw, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        loss_v = float(loss.detach()) if s == steps - 1 else loss_v
    model.eval()
    return loss_v


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device, bs: int = 1000):
    """Returns (probs [N,C] numpy, embeddings [N,D] numpy)."""
    model.to(device).eval()
    P, E = [], []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        e = model.embed(xb)
        P.append(F.softmax(model.fc(e), dim=1).float().cpu())
        E.append(e.float().cpu())
    return torch.cat(P).numpy(), torch.cat(E).numpy()


@torch.no_grad()
def mc_dropout_probs(model: nn.Module, X: torch.Tensor, device, T: int = 10, bs: int = 1000):
    """MC-dropout samples [T,N,C] (only Dropout layers stochastic; norm layers in eval)."""
    model.to(device).eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        e = model.embed(xb)
        out.append(torch.stack([F.softmax(model.fc(model.drop(e)), 1) for _ in range(T)]).cpu())
    model.eval()
    return torch.cat(out, dim=1).numpy()


def metrics(probs: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    pred = probs.argmax(1)
    return {
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0)),
        "per_class_recall": [float(v) for v in
                             recall_score(y, pred, average=None, labels=list(range(n_classes)), zero_division=0)],
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-8, 1)).mean()),
    }


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    acc = probs.argmax(1) == y
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
