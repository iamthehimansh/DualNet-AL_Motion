"""Run a benchmark grid (resumable, shardable) and aggregate it.

Usage:
  python run_experiments.py --out results_heldout --seeds 3 4 5 --v2cfg results/tuning/chosen_config.json --shard 0/3
  python run_experiments.py --out results_heldout --summarize-only

Every run is one JSON file  <out>/raw/<dataset>__<strategy>__s<seed>.json , so a grid can be split across
several processes (--shard i/n) sharing one GPU and resumed after an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from dualnet import __version__
from dualnet.data import load
from dualnet.log import log
from dualnet.loop import LABELS, STRATEGIES, aulc, run_al
from dualnet.models import NetA, NetB, count_params
from dualnet.train import get_device

ROOT = os.path.dirname(os.path.abspath(__file__))


def raw_path(out, ds, strat, seed):
    return os.path.join(out, "raw", f"{ds}__{strat}__s{seed}.json")


def summarize(out: str) -> dict:
    rawdir = os.path.join(out, "raw")
    runs = [json.load(open(os.path.join(rawdir, f))) for f in sorted(os.listdir(rawdir)) if f.endswith(".json")]
    res = {"version": __version__, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "datasets": {}}
    for ds in sorted({r["dataset"] for r in runs}):
        dsr = [r for r in runs if r["dataset"] == ds]
        present = {r["strategy"] for r in dsr}
        # compare strategies only on seeds that EVERY strategy has finished (paired, equal-n)
        common = set.intersection(*[{r["seed"] for r in dsr if r["strategy"] == st} for st in present])
        if not common:
            continue
        dsr = [r for r in dsr if r["seed"] in common]
        rand = sorted([r for r in dsr if r["strategy"] == "random"], key=lambda r: r["seed"])
        rand_final = float(np.mean([r["rounds_log"][-1]["acc"] for r in rand])) if rand else None
        table = {}
        for st in STRATEGIES:
            rs = sorted([r for r in dsr if r["strategy"] == st], key=lambda r: r["seed"])
            if not rs:
                continue
            n_lab = [x["n_labeled"] for x in rs[0]["rounds_log"]]
            curves = np.array([[x["acc"] for x in r["rounds_log"]] for r in rs])
            f1s = np.array([[x["macro_f1"] for x in r["rounds_log"]] for r in rs])
            mean_curve = curves.mean(0)
            need = None
            if rand_final is not None:
                for i, a in enumerate(mean_curve):
                    if a >= rand_final:
                        if i == 0:
                            need = n_lab[0]
                        else:
                            a0 = mean_curve[i - 1]
                            t = (rand_final - a0) / max(a - a0, 1e-9)
                            need = n_lab[i - 1] + t * (n_lab[i] - n_lab[i - 1])
                        break
            au = [aulc(r) for r in rs]
            # paired AULC difference vs Random on the same seeds (same data split + same initial labels)
            paired = None
            if rand and st != "random":
                d = [aulc(a) - aulc(b) for a, b in zip(rs, rand)]
                paired = {"mean": float(np.mean(d)), "per_seed": [float(x) for x in d],
                          "wins": int(sum(x > 0 for x in d)), "n": len(d)}
            table[st] = {
                "label": LABELS[st], "n_seeds": len(rs), "seeds": [r["seed"] for r in rs], "n_labeled": n_lab,
                "acc_mean": mean_curve.tolist(), "acc_std": curves.std(0).tolist(),
                "f1_mean": f1s.mean(0).tolist(), "f1_std": f1s.std(0).tolist(),
                "final_acc": float(curves[:, -1].mean()), "final_acc_std": float(curves[:, -1].std()),
                "final_f1": float(f1s[:, -1].mean()), "final_f1_std": float(f1s[:, -1].std()),
                "aulc": float(np.mean(au)), "aulc_std": float(np.std(au)),
                "aulc_f1": float(np.mean([aulc(r, "macro_f1") for r in rs])),
                "paired_vs_random": paired,
                "labels_to_random_final": None if need is None else float(need),
                "label_saving_pct": None if need is None else
                float(100 * (1 - (need - n_lab[0]) / (n_lab[-1] - n_lab[0]))),
                "min_class_recall": float(np.mean([min(r["rounds_log"][-1]["per_class_recall"]) for r in rs])),
                "final_ece": float(np.mean([r["rounds_log"][-1]["ece"] for r in rs])),
                "batch_error_rate": float(np.mean([np.mean([x.get("chosen_err_rate", 0) for x in r["rounds_log"][:-1]])
                                                   for r in rs])),
                "t_select_s": float(np.mean([np.mean([x.get("t_select", 0) for x in r["rounds_log"][:-1]]) for r in rs])),
                "t_train_s": float(np.mean([np.mean([x["t_train"] for x in r["rounds_log"]]) for r in rs])),
                "wall_time_s": float(np.mean([r["wall_time"] for r in rs])),
                "per_seed_final": curves[:, -1].tolist(),
            }
            if "meta_history" in rs[0]:
                table[st]["meta_weights_mean"] = np.mean([r["meta_history"] for r in rs], 0).tolist()
            if rs[0].get("v2cfg"):
                table[st]["v2cfg"] = rs[0]["v2cfg"]
        res["datasets"][ds] = {"random_final": rand_final, "table": table, "seeds": sorted(common)}
    res["env"] = env_info()
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def env_info():
    dev = get_device()
    return {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "device": str(dev),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "params_NetA_gray": count_params(NetA(1, 10)), "params_NetB_gray": count_params(NetB(1, 10)),
            "params_NetA_rgb": count_params(NetA(3, 10)), "params_NetB_rgb": count_params(NetB(3, 10))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results_heldout"))
    ap.add_argument("--datasets", nargs="+", default=["fmnist", "svhn", "cifar10"])
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--v2cfg", default=None, help="JSON file with the chosen v2 configuration")
    ap.add_argument("--init", type=int, default=100)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--pool", type=int, default=10000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    if a.summarize_only:
        summarize(a.out)
        log("summary written")
        return
    v2cfg = json.load(open(a.v2cfg))["config"] if a.v2cfg else None
    k, n = (int(x) for x in a.shard.split("/"))
    dev = get_device()
    jobs = [(s, d, st) for d in a.datasets for s in a.seeds for st in a.strategies]
    jobs = [j for i, j in enumerate(jobs) if i % n == k]
    todo = [j for j in jobs if not os.path.exists(raw_path(a.out, j[1], j[2], j[0]))]
    log(f"device={dev} shard {k}/{n}: {len(jobs)} runs, {len(jobs) - len(todo)} done, {len(todo)} to run; "
        f"v2cfg={v2cfg}")
    cache = {}
    t_all = time.time()
    for i, (seed, ds, st) in enumerate(todo, 1):
        if (ds, seed) not in cache:
            cache = {(ds, seed): load(ds, pool_size=a.pool, val_size=a.val, seed=seed)}
        el = time.time() - t_all
        eta = el / (i - 1) * (len(todo) - i + 1) if i > 1 else float("nan")
        log(f"[{i}/{len(todo)}] START {ds}/{st}/seed {seed} (elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min)")
        t0 = time.time()
        h = run_al(cache[(ds, seed)], st, seed=seed, init=a.init, batch=a.batch, rounds=a.rounds, steps=a.steps,
                   device=dev, v2cfg=v2cfg, verbose=False)
        tmp = raw_path(a.out, ds, st, seed) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(h, f)
        os.replace(tmp, raw_path(a.out, ds, st, seed))
        log(f"[{i}/{len(todo)}] DONE  {ds}/{st}/seed {seed}: final acc {h['rounds_log'][-1]['acc']:.4f} "
            f"AULC {aulc(h):.4f} in {time.time() - t0:.0f}s")
    log("shard finished")


if __name__ == "__main__":
    main()
