"""Batch construction.

v1: rank -> de-duplicate -> class-balance  (select_batch)
v2: budget-aware score + one pick per k-means cluster  (select_v2)
"""
from __future__ import annotations

import math

import numpy as np
import torch


def select_batch(score: np.ndarray, emb: np.ndarray, pred: np.ndarray, B: int, n_classes: int,
                 tau: float = 0.95, cand_mult: int = 4, quota_mult: float = 1.5) -> tuple[np.ndarray, dict]:
    """Pick B indices.

    1. rank all pool samples by fused score, keep the top cand_mult*B candidates;
    2. greedily accept in score order, rejecting a candidate whose cosine similarity to an
       already-accepted sample exceeds tau (near-duplicate removal);
    3. cap each predicted class at ceil(quota_mult * B / C) so one class cannot dominate.
    If the constraints leave the batch short, they are relaxed in that order (quota, then tau).
    """
    B = min(B, len(score))
    order = np.argsort(-score, kind="stable")
    cand = order[: max(B * cand_mult, B)]
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    quota = math.ceil(quota_mult * B / n_classes)
    chosen: list[int] = []
    per_class = np.zeros(n_classes, int)
    stats = {"rejected_dup": 0, "rejected_quota": 0, "relaxed": 0}

    def try_fill(pool, use_quota, use_dup):
        for i in pool:
            if len(chosen) >= B:
                return
            if i in chosen_set:
                continue
            if use_quota and per_class[pred[i]] >= quota:
                stats["rejected_quota"] += 1
                continue
            if use_dup and chosen:
                if float((E[chosen] @ E[i]).max()) > tau:
                    stats["rejected_dup"] += 1
                    continue
            chosen.append(int(i))
            chosen_set.add(int(i))
            per_class[pred[i]] += 1

    chosen_set: set[int] = set()
    try_fill(cand, True, True)
    if len(chosen) < B:
        stats["relaxed"] += 1
        try_fill(order, True, True)
    if len(chosen) < B:
        stats["relaxed"] += 1
        try_fill(order, False, True)
    if len(chosen) < B:
        stats["relaxed"] += 1
        try_fill(order, False, False)
    stats["per_class"] = per_class.tolist()
    return np.array(chosen, dtype=np.int64), stats


def kmeans(emb: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> np.ndarray:
    """Cosine k-means (k-means++ init) on L2-normalised embeddings; returns a cluster id per row.

    Runs on the GPU when one is available; centroid updates are vectorised (index_add) instead of a
    Python loop over clusters. The k-means++ sampling uses a seeded CPU generator, so it is reproducible.
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    X = torch.nn.functional.normalize(torch.from_numpy(np.asarray(emb, dtype=np.float32)), dim=1).to(dev)
    n, dim = X.shape
    k = min(k, n)
    g = torch.Generator().manual_seed(seed)
    idx = [int(torch.randint(0, n, (1,), generator=g))]
    d2 = ((X - X[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        p = (d2 / d2.sum()) if float(d2.sum()) > 0 else torch.full((n,), 1.0 / n, device=dev)
        j = int(torch.multinomial(p.cpu(), 1, generator=g))
        idx.append(j)
        d2 = torch.minimum(d2, ((X - X[j]) ** 2).sum(1))
    C = X[idx].clone()
    lab = torch.zeros(n, dtype=torch.long, device=dev)
    for _ in range(iters):
        lab = (X @ C.T).argmax(1)
        sums = torch.zeros(k, dim, device=dev).index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=k)
        nonempty = cnt > 0
        C[nonempty] = torch.nn.functional.normalize(sums[nonempty], dim=1)
    return lab.cpu().numpy()


def select_v2(S: np.ndarray, w: np.ndarray, emb: np.ndarray, B: int, rnd: int, cfg: dict,
              seed: int = 0) -> tuple[np.ndarray, dict]:
    """DualNet-AL v2 batch: budget-aware fused score, one pick per k-means cluster.

    score = (1 - lam) * representativeness + lam * (S @ w), where lam ramps from cfg['lam0'] at
    round 0 to 1 after cfg['ramp'] rounds: with very few labels the model's own uncertainty is
    unreliable, so typical (dense-region) samples are preferred first; informativeness takes over
    as the model improves. The pool is clustered into B groups and the best-scoring sample of each
    group is taken, so the batch covers the whole data distribution with no redundancy.
    """
    lam0, ramp = float(cfg.get("lam0", 0.0)), max(int(cfg.get("ramp", 4)), 1)
    lam = lam0 + (1 - lam0) * min(1.0, rnd / ramp)
    score = (1 - lam) * S[:, 2] + lam * (S @ w)
    lab = kmeans(emb, B, seed=seed)
    chosen = []
    for c in range(lab.max() + 1):
        m = np.where(lab == c)[0]
        if len(m):
            chosen.append(int(m[np.argmax(score[m])]))
    if len(chosen) < B:  # empty clusters: fill by global score
        taken = set(chosen)
        for i in np.argsort(-score):
            if len(chosen) >= B:
                break
            if int(i) not in taken:
                chosen.append(int(i))
                taken.add(int(i))
    return np.array(chosen[:B], dtype=np.int64), {"lam": lam, "n_clusters": int(lab.max() + 1)}
