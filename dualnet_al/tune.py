"""Choose the DualNet-AL v2 configuration using ONLY validation data.

Protocol (no test-set leakage):
  * tuning seeds 0, 1, 2 (each seed = a different pool / validation split + initial labelled set);
  * run_al(..., eval_test=False): the test set is never evaluated during tuning;
  * score = area under the VALIDATION-accuracy curve (500 validation images drawn from the official
    training split), averaged over datasets and seeds;
  * ONE configuration is chosen for all datasets and frozen in results/tuning/chosen_config.json;
  * the final comparison then runs on NEW seeds 3, 4, 5 that tuning never saw (run_experiments.py).

Usage: python tune.py --shard 0/3   (three shards share the GPU)   then   python tune.py --choose
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from dualnet.data import load
from dualnet.log import log
from dualnet.loop import aulc, run_al
from dualnet.train import get_device

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "tuning")

GRID = {
    "c1_ramp4_margin": {"lam0": 0.0, "ramp": 4, "unc": "margin"},
    "c2_ramp8_margin": {"lam0": 0.0, "ramp": 8, "unc": "margin"},
    "c3_half_ramp4_margin": {"lam0": 0.5, "ramp": 4, "unc": "margin"},
    "c4_ramp4_entropy": {"lam0": 0.0, "ramp": 4, "unc": "entropy"},
    "c5_noprior_margin": {"lam0": 1.0, "ramp": 1, "unc": "margin"},
    "c6_ramp2_margin": {"lam0": 0.0, "ramp": 2, "unc": "margin"},
}
REFS = ["random", "dualnet_meta"]  # reference points, also scored on validation only
DATASETS = ["fmnist", "svhn", "cifar10"]
SEEDS = [0, 1, 2]


def path(ds, name, seed):
    return os.path.join(OUT, f"{ds}__{name}__s{seed}.json")


def run(shard: str, datasets=DATASETS):
    k, n = (int(x) for x in shard.split("/"))
    dev = get_device()
    jobs = [(ds, s, name) for ds in datasets for s in SEEDS for name in list(GRID) + REFS]
    jobs = [j for i, j in enumerate(jobs) if i % n == k]
    todo = [j for j in jobs if not os.path.exists(path(j[0], j[2], j[1]))]
    log(f"tuning shard {k}/{n}: {len(jobs)} runs, {len(todo)} to run (validation only, test set unused)")
    cache, t_all = {}, time.time()
    for i, (ds, seed, name) in enumerate(todo, 1):
        if (ds, seed) not in cache:
            cache = {(ds, seed): load(ds, val_size=500, seed=seed)}
        t0 = time.time()
        strat = name if name in REFS else "dualnet_v2"
        h = run_al(cache[(ds, seed)], strat, seed=seed, device=dev, verbose=False, eval_test=False,
                   v2cfg=GRID.get(name))
        h["config_name"] = name
        json.dump(h, open(path(ds, name, seed), "w"))
        el = time.time() - t_all
        log(f"[{i}/{len(todo)}] {ds}/{name}/s{seed}: val-AULC {aulc(h, 'val_acc'):.4f} "
            f"final val {h['rounds_log'][-1]['val_acc']:.3f} ({time.time() - t0:.0f}s, "
            f"ETA {el / i * (len(todo) - i) / 60:.1f} min)")


def choose():
    rows = {}
    for name in list(GRID) + REFS:
        per = {}
        for ds in DATASETS:
            v = [aulc(json.load(open(path(ds, name, s))), "val_acc") for s in SEEDS if os.path.exists(path(ds, name, s))]
            per[ds] = float(np.mean(v)) if v else None
        rows[name] = per
    complete = {k: v for k, v in rows.items() if all(x is not None for x in v.values())}
    rand = complete.get("random")
    log("validation AULC (mean over tuning seeds 0-2):")
    for name, per in complete.items():
        gain = "" if not rand else "  vs random " + " ".join(f"{(per[d] - rand[d]) * 100:+.2f}" for d in DATASETS)
        log(f"  {name:22s} " + " ".join(f"{d}={per[d] * 100:.2f}" for d in DATASETS) +
            f"  mean={np.mean(list(per.values())) * 100:.2f}{gain}")
    cands = {k: v for k, v in complete.items() if k in GRID}
    # rank by mean improvement over random across datasets (each dataset counted equally)
    best = max(cands, key=lambda k: np.mean([cands[k][d] - (rand[d] if rand else 0) for d in DATASETS]))
    out = {"chosen": best, "config": GRID[best], "protocol": "validation accuracy only, tuning seeds 0-2, "
           "one config for all datasets; test set never evaluated during tuning",
           "val_aulc": complete, "chosen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(out, open(os.path.join(OUT, "chosen_config.json"), "w"), indent=1)
    log(f"CHOSEN: {best} {GRID[best]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    choose() if a.choose else run(a.shard, a.datasets)
