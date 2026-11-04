"""Figures for the slides / video / app, from <out>/summary.json (default results_heldout).

Usage: python make_figures.py [--out results_heldout]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from dualnet.log import log  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DS_TITLE = {"fmnist": "Fashion-MNIST", "svhn": "SVHN", "cifar10": "CIFAR-10"}
ORDER = ["random", "entropy", "margin", "bald", "coreset", "qbc", "dualnet_fixed", "dualnet_meta", "dualnet_v2"]
SHORT = {"random": "Random", "entropy": "Entropy", "margin": "Margin", "bald": "BALD", "coreset": "Core-Set",
         "qbc": "QBC", "dualnet_fixed": "DualNet v1 fixed", "dualnet_meta": "DualNet v1 meta",
         "dualnet_v2": "DualNet-AL v2"}
COLORS = {"random": "#9e9e9e", "entropy": "#4c72b0", "margin": "#55a868", "bald": "#8172b2", "coreset": "#ccb974",
          "qbc": "#64b5cd", "dualnet_fixed": "#f6b26b", "dualnet_meta": "#e69138", "dualnet_v2": "#d62728"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160})


def curves(s, out):
    for ds, v in s["datasets"].items():
        t = v["table"]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for st in ORDER:
            if st not in t:
                continue
            r = t[st]
            x = r["n_labeled"]
            m, sd = np.array(r["acc_mean"]) * 100, np.array(r["acc_std"]) * 100
            main = st in ("dualnet_v2", "random")
            ax.plot(x, m, color=COLORS[st], lw=3.0 if main else 1.3, alpha=1 if main else 0.75,
                    marker="o", ms=4 if main else 2, label=SHORT[st], zorder=5 if st == "dualnet_v2" else 2)
            if main:
                ax.fill_between(x, m - sd, m + sd, color=COLORS[st], alpha=0.12)
        ax.set_title(f"{DS_TITLE.get(ds, ds)} — test accuracy vs labels (held-out seeds {', '.join(map(str, v['seeds']))})",
                     fontsize=11)
        ax.set_xlabel("# labelled images")
        ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        plt.tight_layout()
        p = os.path.join(out, f"curve_{ds}.png")
        plt.savefig(p)
        plt.close()
        log(f"wrote {p}")


def gain_vs_random(s, out):
    dss = list(s["datasets"])
    strats = [x for x in ORDER if x != "random"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.85 / len(strats)
    for i, st in enumerate(strats):
        vals, errs = [], []
        for ds in dss:
            p = s["datasets"][ds]["table"].get(st, {}).get("paired_vs_random")
            if p:
                vals.append(p["mean"] * 100)
                errs.append(np.std(p["per_seed"]) * 100)
            else:
                vals.append(np.nan)
                errs.append(0)
        ax.bar(np.arange(len(dss)) + i * w - 0.425 + w / 2, vals, w, yerr=errs, color=COLORS[st], label=SHORT[st],
               capsize=2, edgecolor="black" if st == "dualnet_v2" else "none", linewidth=1.2)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(range(len(dss)))
    ax.set_xticklabels([DS_TITLE.get(d, d) for d in dss])
    ax.set_ylabel("AULC gain over Random (pp)")
    ax.set_title("Label efficiency vs Random on the same held-out seeds (positive = better)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out, "gain_vs_random.png")
    plt.savefig(p)
    plt.close()
    log(f"wrote {p}")


def _box(ax, x, y, w, h, text, fc, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, a, b, c="#333"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=c, lw=1.3))


def architecture(out):
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    _box(ax, 0.1, 3.9, 1.8, 1.0, "Labelled set L\n(starts at 100)", "#e3f2fd")
    _box(ax, 0.1, 0.6, 1.8, 1.0, "Unlabelled pool U\n(10 000 images)", "#e3f2fd")
    _box(ax, 2.5, 4.1, 2.1, 0.8, "Net A\n3x3 deep + BatchNorm", "#fff3e0", 9)
    _box(ax, 2.5, 3.0, 2.1, 0.8, "Net B\n5x5 wide + GroupNorm", "#fff3e0", 9)
    _box(ax, 5.2, 0.35, 2.6, 2.75, "5 acquisition signals\n\n1 Uncertainty (margin)\n2 Disagreement (JS A||B)\n"
         "3 Representativeness\n4 Diversity\n5 Class balance", "#e8f5e9", 9)
    _box(ax, 5.2, 3.6, 2.6, 1.3, "Meta-learner\nadapts signal weights\nfrom validation gain", "#fce4ec", 9)
    _box(ax, 8.4, 1.25, 1.6, 1.0, "Budget-aware\nscore", "#ede7f6", 9)
    _box(ax, 10.3, 1.25, 1.6, 1.0, "1 pick per\nk-means cluster", "#ede7f6", 9)
    _box(ax, 10.3, 3.6, 1.6, 1.0, "Oracle /\nannotator", "#ffebee", 9)
    _box(ax, 8.4, 3.6, 1.6, 1.0, "Validation set\n(500, train split)", "#e0f7fa", 9)
    _arrow(ax, (1.9, 4.4), (2.5, 4.5))
    _arrow(ax, (1.9, 4.2), (2.5, 3.4))
    _arrow(ax, (4.6, 3.4), (5.2, 2.6))
    _arrow(ax, (4.6, 4.5), (5.2, 2.9))
    _arrow(ax, (1.9, 1.1), (5.2, 1.4))
    _arrow(ax, (7.8, 1.75), (8.4, 1.75))
    _arrow(ax, (6.5, 3.6), (6.5, 3.1))
    _arrow(ax, (10.0, 1.75), (10.3, 1.75))
    _arrow(ax, (11.1, 2.25), (11.1, 3.6))
    _arrow(ax, (8.4, 4.1), (7.8, 4.2))
    ax.plot([11.1, 11.1, 1.0], [4.6, 5.25, 5.25], color="#c62828", lw=1.4)
    _arrow(ax, (1.0, 5.25), (1.0, 4.92), "#c62828")
    ax.text(6.0, 5.33, "new labels added to L (next round)", ha="center", fontsize=9, color="#c62828")
    ax.text(6.0, 6.0, "DualNet-AL v2 — one acquisition round", ha="center", fontsize=13, weight="bold")
    ax.text(6.0, 5.72, "test set is never used for training, selection or tuning", ha="center", fontsize=9,
            style="italic")
    p = os.path.join(out, "architecture.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    log(f"wrote {p}")


def protocol(out):
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    _box(ax, 0.2, 1.0, 3.4, 1.4, "Official TRAIN split\n→ pool (10 000) + validation (500)", "#e3f2fd")
    _box(ax, 4.3, 1.0, 3.4, 1.4, "Tuning: seeds 0-2\nscore = VALIDATION accuracy\n1 config chosen, frozen", "#fff3e0")
    _box(ax, 8.4, 1.0, 3.4, 1.4, "Final: NEW seeds 3-5\nreport TEST accuracy\n(official TEST split)", "#e8f5e9")
    _arrow(ax, (3.6, 1.7), (4.3, 1.7))
    _arrow(ax, (7.7, 1.7), (8.4, 1.7))
    ax.text(6, 2.75, "Evaluation protocol — no test-set leakage", ha="center", fontsize=13, weight="bold")
    ax.text(6, 0.45, "same networks, same training steps and same starting labels for every method; "
            "only the selection rule differs", ha="center", fontsize=9, style="italic")
    p = os.path.join(out, "protocol.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    log(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results_heldout"))
    a = ap.parse_args()
    plots = os.path.join(a.out, "plots")
    os.makedirs(plots, exist_ok=True)
    log("[1/4] architecture + protocol diagrams")
    architecture(plots)
    protocol(plots)
    sp = os.path.join(a.out, "summary.json")
    if not os.path.exists(sp):
        log("no summary.json yet - only diagrams written")
        return
    s = json.load(open(sp))
    log("[2/4] learning curves")
    curves(s, plots)
    log("[3/4] gain vs random")
    gain_vs_random(s, plots)
    log("[4/4] all figures done")


if __name__ == "__main__":
    main()
