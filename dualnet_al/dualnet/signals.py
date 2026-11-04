"""The five acquisition signals of DualNet-AL.

Every signal is computed for every unlabelled pool sample and mapped to [0, 1]
by rank-percentile normalisation (robust to outliers and to the very different
raw scales of entropy, divergence and distances).
"""
from __future__ import annotations

import numpy as np
import torch

SIGNAL_NAMES = ["uncertainty", "disagreement", "representativeness", "diversity", "class_balance"]


def rank_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if len(v) <= 1 or np.allclose(v, v[0]):
        return np.zeros_like(v)
    r = v.argsort(kind="stable").argsort(kind="stable").astype(np.float64)
    return r / (len(v) - 1)


def uncertainty(pA: np.ndarray, pB: np.ndarray) -> np.ndarray:
    """Normalised predictive entropy of the averaged (ensemble) distribution."""
    p = 0.5 * (pA + pB)
    H = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    return H / np.log(p.shape[1])


def margin_uncertainty(pA: np.ndarray, pB: np.ndarray) -> np.ndarray:
    """1 - (top1 - top2) of the averaged distribution: high when the two best classes are close."""
    ps = np.sort(0.5 * (pA + pB), 1)
    return 1.0 - (ps[:, -1] - ps[:, -2])


def disagreement(pA: np.ndarray, pB: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence between the two networks (in [0,1]) + hard argmax mismatch."""
    m = 0.5 * (pA + pB)

    def kl(p, q):
        return (p * (np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(q, 1e-12, 1)))).sum(1)

    js = 0.5 * kl(pA, m) + 0.5 * kl(pB, m)
    js = js / np.log(2)
    mismatch = (pA.argmax(1) != pB.argmax(1)).astype(np.float64)
    return 0.7 * js + 0.3 * mismatch


def _l2n(e: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(e, dtype=np.float32))
    return torch.nn.functional.normalize(t, dim=1)


def representativeness(emb_pool: np.ndarray, k: int = 10, chunk: int = 2048) -> np.ndarray:
    """Local density: mean cosine similarity to the k nearest pool neighbours (outliers score low)."""
    E = _l2n(emb_pool)
    k = min(k, len(E) - 1)
    out = torch.empty(len(E))
    for i in range(0, len(E), chunk):
        s = E[i:i + chunk] @ E.T
        top = s.topk(k + 1, dim=1).values[:, 1:]  # drop self-similarity
        out[i:i + chunk] = top.mean(1)
    return out.numpy().astype(np.float64)


def diversity(emb_pool: np.ndarray, emb_lab: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Core-set style novelty: distance to the closest already-labelled sample."""
    P, L = _l2n(emb_pool), _l2n(emb_lab)
    out = torch.empty(len(P))
    for i in range(0, len(P), chunk):
        out[i:i + chunk] = torch.cdist(P[i:i + chunk], L,
                                       compute_mode="donot_use_mm_for_euclid_dist").min(1).values
    return out.numpy().astype(np.float64)


def class_balance(pred: np.ndarray, y_lab: np.ndarray, n_classes: int) -> np.ndarray:
    """Higher for samples predicted to belong to classes that are rare in the labelled set."""
    counts = np.bincount(y_lab, minlength=n_classes).astype(np.float64)
    freq = (counts + 1) / (counts.sum() + n_classes)
    rarity = (freq.max() - freq) / (freq.max() - freq.min() + 1e-12)
    return rarity[pred]


def compute_all(pA, pB, emb_pool, emb_lab, y_lab, n_classes, unc: str = "entropy") -> tuple[np.ndarray, np.ndarray]:
    """Returns (S_norm [N,5] in [0,1], S_raw [N,5]). unc = 'entropy' (v1) or 'margin'."""
    pred = (0.5 * (pA + pB)).argmax(1)
    u = margin_uncertainty(pA, pB) if unc == "margin" else uncertainty(pA, pB)
    raw = np.stack([
        u,
        disagreement(pA, pB),
        representativeness(emb_pool),
        diversity(emb_pool, emb_lab),
        class_balance(pred, y_lab, n_classes),
    ], axis=1)
    norm = np.stack([rank_norm(raw[:, k]) for k in range(raw.shape[1])], axis=1)
    return norm, raw
