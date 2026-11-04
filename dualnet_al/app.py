"""DualNet-AL live prototype demo.

Run:  streamlit run app.py     (or the one-command launchers run_demo.bat / run_demo.sh)
Live tab: runs the real active-learning loop round by round (accuracy of Net A, Net B and the ensemble,
the meta-learned signal weights, the images selected each round, class balance).
Results tab: the held-out benchmark (new seeds never used for tuning), read from results_heldout/summary.json.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dualnet.data import load
from dualnet.loop import LABELS, STRATEGIES, run_al
from dualnet.signals import SIGNAL_NAMES
from dualnet.train import get_device

ROOT = os.path.dirname(os.path.abspath(__file__))
CHOSEN = os.path.join(ROOT, "results", "tuning", "chosen_config.json")
st.set_page_config(page_title="DualNet-AL prototype", layout="wide")
st.title("DualNet-AL — Bi-Network Meta-Learning Active Learning")
st.caption("IILM University · Capstone BT-20 · Himansh Raj & Om Narayan Pandey · Guide: Dr. Jaswinder Singh")

tab_live, tab_res, tab_about = st.tabs(["Live active-learning run", "Benchmark results (held-out)", "How it works"])


@st.cache_resource(show_spinner="Loading dataset (first time downloads it)…")
def get_data(name, pool, seed):
    return load(name, pool_size=pool, val_size=500, seed=seed)


def v2_config():
    return json.load(open(CHOSEN))["config"] if os.path.exists(CHOSEN) else None


def show_images(X, ids, labels, names, ncol=8):
    n = min(len(ids), ncol)
    fig, axes = plt.subplots(1, n, figsize=(1.4 * n, 1.6))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, ids[:n]):
        img = X[i].numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        ax.imshow(img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0), cmap="gray")
        ax.set_title(names[labels[i]], fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    return fig


with tab_live:
    c1, c2, c3, c4, c5 = st.columns(5)
    ds = c1.selectbox("Dataset", ["fmnist", "svhn", "cifar10", "synthetic"], index=0)
    strat = c2.selectbox("Strategy", STRATEGIES, index=STRATEGIES.index("dualnet_v2"),
                         format_func=lambda s: LABELS[s])
    rounds = c3.slider("Rounds", 1, 10, 5)
    batch = c4.select_slider("Batch size (labels / round)", [25, 50, 100, 200], 100)
    steps = c5.select_slider("Training steps / round", [50, 100, 200, 300], 200)
    seed = st.number_input("Seed", 0, 99, 3)
    compare = st.checkbox("Also run Random on the same seed for a side-by-side comparison", value=True)
    pool = 5000 if ds != "synthetic" else 600
    if st.button("▶ Run active learning", type="primary"):
        data = get_data(ds, pool, int(seed))
        dev = get_device()
        init = 100 if ds != "synthetic" else 20
        st.info(f"Device: **{dev}** · pool {len(data.y_pool)} unlabelled · test {len(data.y_test)} · "
                f"validation {len(data.y_val)} · initial labelled {init}")
        prog = st.progress(0.0)
        chart = st.empty()
        wchart = st.empty()
        imgs = st.container()
        rows, rand_rows = [], []

        def draw():
            import altair as alt

            main = f"{LABELS[strat].split(' (')[0]} (A+B ensemble)"
            recs = []
            for r in rows:
                recs += [{"labels": r["n_labeled"], "acc": r["acc"] * 100, "series": main},
                         {"labels": r["n_labeled"], "acc": r["acc_A"] * 100, "series": "Net A alone"},
                         {"labels": r["n_labeled"], "acc": r["acc_B"] * 100, "series": "Net B alone"}]
            for r in rand_rows:
                recs.append({"labels": r["n_labeled"], "acc": r["acc"] * 100, "series": "Random (same seed)"})
            names = [main, "Random (same seed)", "Net A alone", "Net B alone"]
            colors = ["#d62728", "#7f7f7f", "#9ecae1", "#c7b8e0"]
            c = (alt.Chart(pd.DataFrame(recs))
                 .mark_line(point=True)
                 .encode(x=alt.X("labels:Q", title="# labelled images"),
                         y=alt.Y("acc:Q", title="test accuracy (%)", scale=alt.Scale(zero=False)),
                         color=alt.Color("series:N", scale=alt.Scale(domain=names, range=colors),
                                         legend=alt.Legend(title=None, orient="top")),
                         strokeWidth=alt.condition(alt.FieldOneOfPredicate("series", [main, "Random (same seed)"]),
                                                   alt.value(4), alt.value(1.5)))
                 .properties(height=380))
            chart.altair_chart(c, use_container_width=True)

        def on_round(rec, chosen):
            rows.append(rec)
            prog.progress(len(rows) / (rounds + 1), text=f"Round {rec['round']}: test acc {rec['acc']:.3f}")
            draw()
            if "weights" in rec:
                wdf = pd.DataFrame([r["weights"] for r in rows if "weights" in r], columns=SIGNAL_NAMES)
                wdf.index.name = "round"
                wchart.bar_chart(wdf)
            if chosen is not None:
                with imgs:
                    st.markdown(f"**Round {rec['round']} — first 8 selected samples** "
                                f"(ensemble was wrong on {rec['chosen_err_rate'] * 100:.0f}% of the batch)")
                    st.pyplot(show_images(data.X_pool, chosen[:8], data.y_pool.numpy(), data.class_names))

        h = run_al(data, strat, seed=int(seed), init=init, batch=batch, rounds=rounds, steps=steps,
                   device=dev, on_round=on_round, verbose=False, v2cfg=v2_config())
        if compare and strat != "random":
            with st.spinner("Running Random on the same seed and initial labels…"):
                hr = run_al(data, "random", seed=int(seed), init=init, batch=batch, rounds=rounds, steps=steps,
                            device=dev, verbose=False)
            rand_rows = hr["rounds_log"]
            draw()
        last = h["rounds_log"][-1]
        msg = (f"Finished in {h['wall_time']:.1f}s · final test accuracy **{last['acc'] * 100:.2f}%** · "
               f"macro-F1 **{last['macro_f1'] * 100:.2f}%** with {last['n_labeled']} labels")
        if rand_rows:
            msg += f" · Random on the same seed: **{rand_rows[-1]['acc'] * 100:.2f}%**"
        st.success(msg)
        st.caption("A single run is noisy; the Results tab shows the mean over held-out seeds.")
        st.subheader("Labelled-set class distribution")
        st.bar_chart(pd.Series(last["label_hist"], index=data.class_names))
        st.subheader("Per-class recall")
        st.bar_chart(pd.Series(last["per_class_recall"], index=data.class_names))

with tab_res:
    p = os.path.join(ROOT, "results_heldout", "summary.json")
    if not os.path.exists(p):
        st.warning("results_heldout/summary.json not found — run the held-out benchmark first.")
    else:
        s = json.load(open(p))
        st.markdown("Held-out protocol: configuration chosen on **validation** data with seeds 0–2; "
                    "numbers below are on the **test set** with new seeds "
                    f"{', '.join(map(str, next(iter(s['datasets'].values()))['seeds']))}.")
        for ds_name, v in s["datasets"].items():
            st.subheader(ds_name)
            tbl = pd.DataFrame([{
                "Strategy": r["label"], "Seeds": r["n_seeds"],
                "AULC %": f"{r['aulc'] * 100:.2f} ± {r['aulc_std'] * 100:.2f}",
                "vs Random (paired, pp)": "—" if not r.get("paired_vs_random") else
                f"{r['paired_vs_random']['mean'] * 100:+.2f} ({r['paired_vs_random']['wins']}/{r['paired_vs_random']['n']} seeds)",
                "Final acc %": f"{r['final_acc'] * 100:.2f} ± {r['final_acc_std'] * 100:.2f}",
                "Macro-F1 %": f"{r['final_f1'] * 100:.2f}",
                "Worst-class recall %": f"{r['min_class_recall'] * 100:.1f}",
            } for r in v["table"].values()])
            st.dataframe(tbl, hide_index=True, use_container_width=True)
            fig_path = os.path.join(ROOT, "results_heldout", "plots", f"curve_{ds_name}.png")
            if os.path.exists(fig_path):
                st.image(fig_path)

with tab_about:
    st.markdown("""
**Cycle:** train Net A + Net B → score every unlabelled sample with 5 signals → fuse with meta-learned
weights → budget-aware prior → one pick per k-means cluster → annotate → repeat.

| Signal | What it measures |
|---|---|
| Uncertainty | margin between the two most likely classes of the A+B averaged prediction |
| Disagreement | Jensen-Shannon divergence between A and B + argmax mismatch |
| Representativeness | mean cosine similarity to 10 nearest pool neighbours (avoids outliers) |
| Diversity | distance to the closest labelled sample (core-set novelty) |
| Class balance | rarity of the predicted class in the labelled set |

**Budget-aware prior (v2):** with very few labels the model's own uncertainty is unreliable, so early rounds
favour typical (dense-region) samples and the learned informativeness score takes over as labels accumulate.

**Meta-learner:** weights = floor-mixed softmax(θ). After each round θ is updated by (i) validation-accuracy
gain × how strongly the batch used each signal, and (ii) correlation of each signal with the errors revealed
once the batch is labelled. The test set is never used for any of this.
""")
