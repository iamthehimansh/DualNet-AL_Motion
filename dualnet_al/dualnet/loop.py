"""Active-learning loop and all acquisition strategies (DualNet-AL v1, v2 + baselines).

Every strategy trains the SAME two networks (A and B) on the same labelled set with the same
compute, and the reported accuracy is always that of the A+B ensemble. Strategies differ only
in how they pick the next batch, so differences in accuracy are attributable to acquisition.

Data discipline: the networks train only on the labelled subset of the pool; the meta-learner and
any configuration choice use only the validation set (also drawn from the training split); the test
set is touched only to REPORT accuracy and never influences selection, training or tuning.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .log import log
from .meta import MetaWeighter
from .models import build_pair
from .select import select_batch, select_v2
from .signals import SIGNAL_NAMES, compute_all, disagreement
from .train import ece, entropy, mc_dropout_probs, metrics, predict, train_model

STRATEGIES = ["random", "entropy", "margin", "bald", "coreset", "qbc", "dualnet_fixed", "dualnet_meta",
              "dualnet_v2"]
LABELS = {
    "random": "Random", "entropy": "Entropy", "margin": "Margin", "bald": "BALD (MC-Dropout)",
    "coreset": "Core-Set (k-Center)", "qbc": "QBC (A/B disagreement)",
    "dualnet_fixed": "DualNet-AL v1 (fixed weights)", "dualnet_meta": "DualNet-AL v1 (meta-learned)",
    "dualnet_v2": "DualNet-AL v2 (budget-aware, clustered)",
}
V2_DEFAULT = {"lam0": 0.0, "ramp": 4, "unc": "margin", "eta": 1.5, "eta_c": 0.8, "w_min": 0.05}


def kcenter_greedy(emb_pool: np.ndarray, emb_lab: np.ndarray, B: int) -> np.ndarray:
    P = torch.nn.functional.normalize(torch.from_numpy(emb_pool.astype(np.float32)), dim=1)
    L = torch.nn.functional.normalize(torch.from_numpy(emb_lab.astype(np.float32)), dim=1)
    d = torch.full((len(P),), float("inf"))
    for i in range(0, len(L), 2048):
        d = torch.minimum(d, torch.cdist(P, L[i:i + 2048]).min(1).values)
    chosen = []
    for _ in range(B):
        j = int(d.argmax())
        chosen.append(j)
        d = torch.minimum(d, torch.cdist(P, P[j:j + 1]).squeeze(1))
    return np.array(chosen, dtype=np.int64)


def run_al(data, strategy: str, seed: int = 0, init: int = 100, batch: int = 100, rounds: int = 8,
           steps: int = 300, device=None, on_round=None, verbose=True, tau: float = 0.95,
           v2cfg: dict | None = None, eval_test: bool = True) -> dict:
    """Run one active-learning experiment; returns a JSON-serialisable history.

    eval_test=False skips the test set entirely (used during tuning, so tuning cannot see it).
    """
    assert strategy in STRATEGIES, strategy
    cfg = dict(V2_DEFAULT, **(v2cfg or {}))
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    yp = data.y_pool.numpy()
    yt = data.y_test.numpy()
    yv = data.y_val.numpy()
    C = data.n_classes
    lab = rng.choice(len(yp), init, replace=False)
    is_lab = np.zeros(len(yp), bool)
    is_lab[lab] = True
    meta = None
    if strategy == "dualnet_meta":
        meta = MetaWeighter()
    elif strategy == "dualnet_v2":
        meta = MetaWeighter(eta=cfg["eta"], eta_c=cfg["eta_c"], w_min=cfg["w_min"])
    fixed_w = np.ones(len(SIGNAL_NAMES)) / len(SIGNAL_NAMES)
    hist = {"strategy": strategy, "dataset": data.name, "seed": seed, "init": init, "batch": batch,
            "rounds": rounds, "steps": steps, "rounds_log": [],
            "v2cfg": cfg if strategy == "dualnet_v2" else None}
    prev_val = None
    S_prev = sel_prev = pred_prev = pool_idx_prev = None
    t_start = time.time()
    for r in range(rounds + 1):
        t0 = time.time()
        idx_lab = np.where(is_lab)[0]
        Xl, yl = data.X_pool[idx_lab], data.y_pool[idx_lab]
        A, Bn = build_pair(data.in_ch, C, seed * 100 + r)
        la = train_model(A, Xl, yl, device, steps=steps, flip=data.flip_aug)
        lb = train_model(Bn, Xl, yl, device, steps=steps, flip=data.flip_aug)
        t_train = time.time() - t0
        pA_v, _ = predict(A, data.X_val, device)
        pB_v, _ = predict(Bn, data.X_val, device)
        pv = 0.5 * (pA_v + pB_v)
        val_acc = float((pv.argmax(1) == yv).mean())
        val_f1 = metrics(pv, yv, C)["macro_f1"]
        rec = {"round": r, "n_labeled": int(is_lab.sum()), "val_acc": val_acc, "val_f1": val_f1,
               "loss_A": la, "loss_B": lb, "label_hist": np.bincount(yl.numpy(), minlength=C).tolist(),
               "t_train": t_train}
        if eval_test:
            pA_t, _ = predict(A, data.X_test, device)
            pB_t, _ = predict(Bn, data.X_test, device)
            p_ens = 0.5 * (pA_t + pB_t)
            m = metrics(p_ens, yt, C)
            rec.update({"acc": m["acc"], "macro_f1": m["macro_f1"], "per_class_recall": m["per_class_recall"],
                        "nll": m["nll"], "ece": ece(p_ens, yt), "acc_A": metrics(pA_t, yt, C)["acc"],
                        "acc_B": metrics(pB_t, yt, C)["acc"]})
        # meta update for the batch chosen in the previous round (its effect is now measurable)
        if meta is not None and sel_prev is not None:
            err = (pred_prev[sel_prev] != yp[pool_idx_prev[sel_prev]]).astype(float)
            rec["meta_update"] = meta.update(S_prev, sel_prev, err, val_acc - prev_val)
        prev_val = val_acc
        shown = rec.get("acc", val_acc)
        if r == rounds:
            rec["t_select"] = 0.0
            hist["rounds_log"].append(rec)
            if verbose:
                log(f"  [{strategy}|{data.name}|s{seed}] round {r}/{rounds} n={rec['n_labeled']} "
                    f"{'acc' if eval_test else 'val'}={shown:.4f} train {t_train:.1f}s")
            if on_round:
                on_round(rec, None)
            break
        # ---- acquisition ----
        t1 = time.time()
        pool_idx = np.where(~is_lab)[0]
        Xu = data.X_pool[pool_idx]
        pA, eA = predict(A, Xu, device)
        pB, eB = predict(Bn, Xu, device)
        emb = np.concatenate([eA, eB], 1)
        sel_info = {}
        if strategy == "random":
            sel = rng.choice(len(pool_idx), batch, replace=False)
        elif strategy == "entropy":
            sel = np.argsort(-entropy(0.5 * (pA + pB)))[:batch]
        elif strategy == "margin":
            ps = np.sort(0.5 * (pA + pB), 1)
            sel = np.argsort(ps[:, -1] - ps[:, -2])[:batch]
        elif strategy == "bald":
            mc = mc_dropout_probs(A, Xu, device, T=10)
            mean = mc.mean(0)
            mi = entropy(mean) - np.mean([entropy(mc[t]) for t in range(mc.shape[0])], 0)
            sel = np.argsort(-mi)[:batch]
        elif strategy == "coreset":
            _, eAl = predict(A, Xl, device)
            _, eBl = predict(Bn, Xl, device)
            sel = kcenter_greedy(emb, np.concatenate([eAl, eBl], 1), batch)
        elif strategy == "qbc":
            sel = np.argsort(-disagreement(pA, pB))[:batch]
        else:  # dualnet_fixed / dualnet_meta / dualnet_v2
            _, eAl = predict(A, Xl, device)
            _, eBl = predict(Bn, Xl, device)
            unc = cfg["unc"] if strategy == "dualnet_v2" else "entropy"
            S, _raw = compute_all(pA, pB, emb, np.concatenate([eAl, eBl], 1), yl.numpy(), C, unc=unc)
            w = meta.weights() if meta is not None else fixed_w
            pred = (0.5 * (pA + pB)).argmax(1)
            if strategy == "dualnet_v2":
                sel, st = select_v2(S, w, emb, batch, r, cfg, seed=seed * 100 + r)
            else:
                sel, st = select_batch(S @ w, emb, pred, batch, C, tau=tau)
            sel_info = {"weights": w.tolist(), "select_stats": st,
                        "signal_mean_batch": S[sel].mean(0).tolist(), "signal_mean_pool": S.mean(0).tolist()}
            S_prev, pred_prev = S, pred
        rec["t_select"] = time.time() - t1
        rec.update(sel_info)
        sel = np.asarray(sel)
        if meta is not None:
            sel_prev, pool_idx_prev = sel, pool_idx
        chosen = pool_idx[sel]
        rec["chosen_err_rate"] = float(((0.5 * (pA + pB)).argmax(1)[sel] != yp[chosen]).mean())
        rec["chosen_class_hist"] = np.bincount(yp[chosen], minlength=C).tolist()
        rec["chosen_ids"] = chosen[:16].tolist()
        is_lab[chosen] = True
        hist["rounds_log"].append(rec)
        if verbose:
            extra = f" w={np.round(sel_info['weights'], 2).tolist()}" if "weights" in sel_info else ""
            log(f"  [{strategy}|{data.name}|s{seed}] round {r}/{rounds} n={rec['n_labeled']} "
                f"{'acc' if eval_test else 'val'}={shown:.4f} train {t_train:.1f}s select {rec['t_select']:.1f}s"
                f" err@batch={rec['chosen_err_rate']:.2f}{extra}")
        if on_round:
            on_round(rec, chosen)
    hist["wall_time"] = time.time() - t_start
    if meta is not None:
        hist["meta_history"] = meta.history
    return hist


def aulc(hist: dict, key: str = "acc") -> float:
    """Area under the learning curve (metric vs #labels), normalised to [0,1]."""
    xs = np.array([r["n_labeled"] for r in hist["rounds_log"]], float)
    ys = np.array([r[key] for r in hist["rounds_log"]], float)
    return float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if len(xs) > 1 else float(ys[0])
