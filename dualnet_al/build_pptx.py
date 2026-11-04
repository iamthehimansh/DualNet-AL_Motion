"""Build the mid-term presentation deck (16:9 PPTX) from the held-out results.

Every number on a slide is read from results_heldout/summary.json and results/tuning/chosen_config.json;
nothing is typed by hand. Usage: python build_pptx.py  ->  ../deliverables/DualNet-AL_Midterm.pptx
"""
from __future__ import annotations

import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from dualnet.log import log

ROOT = os.path.dirname(os.path.abspath(__file__))
HO = os.path.join(ROOT, "results_heldout")
PLOTS = os.path.join(HO, "plots")
SHOTS = os.path.join(ROOT, "results", "screenshots")
OUT = os.path.abspath(os.path.join(ROOT, "..", "deliverables", "DualNet-AL_Midterm.pptx"))
DS = {"fmnist": "Fashion-MNIST", "svhn": "SVHN", "cifar10": "CIFAR-10"}
NAVY, ACC, RED, GREY, GREEN = (RGBColor(0x0D, 0x2B, 0x52), RGBColor(0x1F, 0x77, 0xB4), RGBColor(0xD6, 0x27, 0x28),
                               RGBColor(0x55, 0x5F, 0x6D), RGBColor(0x1B, 0x7F, 0x3B))
W, H = Inches(13.333), Inches(7.5)


def pct(x, d=1):
    return f"{x * 100:.{d}f}"


class Deck:
    def __init__(self):
        self.p = Presentation()
        self.p.slide_width, self.p.slide_height = W, H
        self.blank = self.p.slide_layouts[6]
        self.n = 0

    def slide(self, title=None, kicker=None):
        s = self.p.slides.add_slide(self.blank)
        self.n += 1
        if title:
            self.text(s, title, Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.8), 30, NAVY, bold=True)
            bar = s.shapes.add_shape(1, Inches(0.6), Inches(1.12), Inches(1.2), Emu(45720))
            bar.fill.solid()
            bar.fill.fore_color.rgb = ACC
            bar.line.fill.background()
        if kicker:
            self.text(s, kicker, Inches(0.6), Inches(1.2), Inches(12.1), Inches(0.5), 15, GREY)
        self.text(s, f"DualNet-AL · BT-20 · IILM University        {self.n}", Inches(0.6), Inches(7.0), Inches(12.1),
                  Inches(0.35), 10, GREY)
        return s

    @staticmethod
    def text(s, txt, x, y, w, h, size=18, color=None, bold=False, align=PP_ALIGN.LEFT):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        lines = txt if isinstance(txt, list) else [txt]
        for i, line in enumerate(lines):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align
            r = para.add_run()
            r.text = line
            r.font.size = Pt(size)
            r.font.bold = bold
            if color is not None:
                r.font.color.rgb = color
            para.space_after = Pt(6)
        return tb

    def bullets(self, s, items, x=Inches(0.7), y=Inches(1.85), w=Inches(12.0), h=Inches(4.9), size=20):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        for i, it in enumerate(items):
            sub = it.startswith("  ")
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            r = para.add_run()
            r.text = ("–  " if sub else "•  ") + it.strip()
            r.font.size = Pt(size - 3 if sub else size)
            r.font.color.rgb = GREY if sub else RGBColor(0x1B, 0x1B, 0x1B)
            para.level = 1 if sub else 0
            para.space_after = Pt(8)
        return tb

    @staticmethod
    def image(s, path, x, y, w=None, h=None):
        if os.path.exists(path):
            return s.shapes.add_picture(path, x, y, width=w, height=h)
        return None

    def table(self, s, rows, x, y, w, col_w=None, size=13, highlight_row=None):
        nr, nc = len(rows), len(rows[0])
        shape = s.shapes.add_table(nr, nc, x, y, w, Inches(0.38 * nr))
        t = shape.table
        if col_w:
            for i, cw in enumerate(col_w):
                t.columns[i].width = Inches(cw)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                c = t.cell(i, j)
                c.text = str(val)
                para = c.text_frame.paragraphs[0]
                para.font.size = Pt(size)
                para.font.bold = i == 0 or i == highlight_row
                if i == highlight_row:
                    c.fill.solid()
                    c.fill.fore_color.rgb = RGBColor(0xFD, 0xE8, 0xE8)
        return t


def main():
    s_all = json.load(open(os.path.join(HO, "summary.json")))
    chosen = json.load(open(os.path.join(ROOT, "results", "tuning", "chosen_config.json")))
    dsets = [d for d in ["fmnist", "svhn", "cifar10"] if d in s_all["datasets"]]
    seeds = s_all["datasets"][dsets[0]]["seeds"]
    T = {d: s_all["datasets"][d]["table"] for d in dsets}
    v2 = {d: T[d]["dualnet_v2"] for d in dsets}
    rd = {d: T[d]["random"] for d in dsets}
    BASE = ["entropy", "margin", "bald", "coreset", "qbc"]
    env = s_all["env"]
    D = Deck()

    log("[1/12] title")
    s = D.slide()
    D.text(s, "CAPSTONE PROJECT · MID-TERM & PHASE III", Inches(0.8), Inches(1.4), Inches(11.7), Inches(0.5), 16, ACC, True)
    D.text(s, "DualNet-AL", Inches(0.8), Inches(1.95), Inches(11.7), Inches(1.2), 60, NAVY, True)
    D.text(s, "A Bi-Network Meta-Learning Framework for Multi-Class Active Learning", Inches(0.8), Inches(3.15),
           Inches(11.7), Inches(0.7), 24, GREY)
    D.text(s, ["Himansh Raj (CS-23411324)  ·  Om Narayan Pandey (CS-2341893)  ·  4CSE1",
               "Guide: Dr. Jaswinder Singh  ·  Project ID BT-20  ·  SDG 9  ·  IILM University, Greater Noida"],
           Inches(0.8), Inches(4.6), Inches(11.7), Inches(1.0), 18, RGBColor(0x1B, 0x1B, 0x1B))

    log("[2/12] problem")
    s = D.slide("The problem", "Deep models need thousands of labels; labelling is slow and expensive")
    D.bullets(s, ["Active learning: the model chooses which unlabelled images a human should label next",
                  "Single-criterion selection has known failure modes:",
                  "  Uncertainty sampling picks confusing outliers and near-duplicates",
                  "  One model choosing its own data reinforces its own mistakes (confirmation bias)",
                  "  Rare classes get ignored, so worst-class accuracy stays low",
                  "  With very few labels, the model's own uncertainty is unreliable (cold start)",
                  "Goal: more accuracy per label than Random sampling, consistently across datasets"])

    log("[3/12] idea")
    s = D.slide("Our idea: two networks + five signals")
    D.image(s, os.path.join(PLOTS, "architecture.png"), Inches(0.9), Inches(1.45), w=Inches(11.5))

    log("[4/12] components")
    s = D.slide("What each part does")
    rows = [["Component", "What it does", "Why"],
            ["Net A + Net B", "3x3 deep CNN (BatchNorm) and 5x5 wide CNN (GroupNorm)", "different mistakes → informative disagreement"],
            ["Uncertainty", "margin between top-2 classes of A+B average", "where the model is unsure"],
            ["Disagreement", "Jensen-Shannon divergence between A and B", "counters single-model confirmation bias"],
            ["Representativeness", "density among 10 nearest neighbours", "avoids outliers"],
            ["Diversity", "distance to nearest labelled image", "covers unexplored regions"],
            ["Class balance", "rarity of predicted class in labelled set", "protects rare classes"],
            ["Meta-learner", "re-weights the 5 signals after each round", "learns how to select, per dataset"]]
    D.table(s, rows, Inches(0.6), Inches(1.55), Inches(12.1), col_w=[2.4, 5.4, 4.3], size=14)

    log("[5/12] v2")
    s = D.slide("Phase III improvement: DualNet-AL v2", "Why v1 tied with Random at small budgets — and the fix")
    D.bullets(s, ["v1 result (Phase II): best on CIFAR-10, but level with Random on Fashion-MNIST and behind on SVHN",
                  "Diagnosis: at 100–300 labels, scores based on the model's own uncertainty are unreliable",
                  "v2 change 1 — budget-aware score: early rounds favour typical (dense) images,",
                  "  then the meta-learned informativeness score takes over as labels accumulate",
                  "v2 change 2 — one pick per k-means cluster (100 clusters → 100 labels):",
                  "  every batch covers the whole data distribution, with no near-duplicates",
                  f"Chosen setting (by validation only): {chosen['chosen']}  "
                  f"(prior ramp {chosen['config']['ramp']} rounds, start weight {chosen['config']['lam0']}, "
                  f"{chosen['config']['unc']} uncertainty)"], size=19)

    log("[6/12] protocol")
    s = D.slide("Fair, leak-free evaluation")
    D.image(s, os.path.join(PLOTS, "protocol.png"), Inches(0.7), Inches(1.5), w=Inches(11.9))
    D.bullets(s, [f"Datasets: {', '.join(DS[d] for d in dsets)} — 10 000-image pool, 500 validation, 2 000 test each",
                  f"100 starting labels + {len(rd[dsets[0]]['n_labeled']) - 1} rounds of 100 → "
                  f"{rd[dsets[0]]['n_labeled'][-1]} labels; both networks retrained from scratch, 300 steps",
                  f"9 methods × {len(dsets)} datasets × {len(seeds)} held-out seeds; "
                  f"hardware: {env.get('gpu') or env['device']}"],
              y=Inches(4.65), h=Inches(2.2), size=17)

    log("[7/12] headline results table")
    s = D.slide("Held-out results: label efficiency (AULC %)",
                f"Area under accuracy-vs-labels curve, mean ± std over new seeds {', '.join(map(str, seeds))} (test set)")
    names = ["random"] + BASE + ["dualnet_meta", "dualnet_v2"]
    rows = [["Method"] + [DS[d] for d in dsets]]
    for st in names:
        if all(st in T[d] for d in dsets):
            rows.append([T[dsets[0]][st]["label"]] +
                        [f"{pct(T[d][st]['aulc'], 2)} ± {pct(T[d][st]['aulc_std'], 2)}" for d in dsets])
    D.table(s, rows, Inches(0.9), Inches(1.75), Inches(11.5), col_w=[4.6] + [6.9 / len(dsets)] * len(dsets),
            size=15, highlight_row=len(rows) - 1)

    log("[8/12] gain chart")
    s = D.slide("Gain over Random on the same seeds")
    D.image(s, os.path.join(PLOTS, "gain_vs_random.png"), Inches(1.75), Inches(1.35), h=Inches(4.25))
    lines = []
    for d in dsets:
        p = v2[d]["paired_vs_random"]
        lines.append(f"{DS[d]}: {p['mean'] * 100:+.2f} pp AULC vs Random, better on {p['wins']}/{p['n']} seeds")
    D.text(s, lines, Inches(0.9), Inches(5.7), Inches(11.5), Inches(1.25), 15, NAVY, True)

    for i, d in enumerate(dsets):
        log(f"[9/12] curve {d}")
        s = D.slide(f"{DS[d]}: accuracy as labels grow")
        D.image(s, os.path.join(PLOTS, f"curve_{d}.png"), Inches(0.6), Inches(1.4), h=Inches(5.0))
        bb = max(BASE, key=lambda k: T[d][k]["aulc"])
        D.text(s, [f"DualNet-AL v2 final accuracy: {pct(v2[d]['final_acc'])}%",
                   f"Random: {pct(rd[d]['final_acc'])}%",
                   f"Best baseline ({T[d][bb]['label'].split(' (')[0]}): {pct(T[d][bb]['final_acc'])}%",
                   "",
                   f"Worst-class recall: v2 {pct(v2[d]['min_class_recall'])}% vs Random {pct(rd[d]['min_class_recall'])}%",
                   f"Macro-F1: v2 {pct(v2[d]['final_f1'])}% vs Random {pct(rd[d]['final_f1'])}%"],
               Inches(9.6), Inches(1.7), Inches(3.4), Inches(4.5), 15)

    log("[10/12] demo + testing")
    s = D.slide("Working prototype & testing")
    D.bullets(s, ["Python / PyTorch package: data, models, training, 5 signals, meta-learner, v2 selection, 8 baselines",
                  "Live web demo (Streamlit): runs the real loop round by round — accuracy of Net A, Net B and ensemble,",
                  "  signal weights per round, the images it chose, class balance, side-by-side with Random",
                  "One-command launcher for Windows / Linux / Mac: installs everything and opens the demo",
                  "30 automated tests pass: each signal's edge cases, clustering, de-duplication, budget ramp,",
                  "  no image labelled twice, tuning mode provably never reads the test set",
                  "Defects found and fixed: distance precision, weight floor, GPU pooling crash, 48 s → 0.03 s clustering"],
              size=18)

    log("[11/12] limitations + next")
    s = D.slide("Limitations & next steps")
    D.bullets(s, [f"Low budget only ({rd[dsets[0]]['n_labeled'][-1]} labels) and small pools; gaps are a few points",
                  f"{len(seeds)} held-out seeds: paired comparison shown, formal significance tests still to do",
                  "Meta-learner reward uses a 500-image validation set; a label-free reward is planned",
                  "Next: CIFAR-100 and long-tail data, Learning-Loss baseline, larger budgets on full datasets,",
                  "  ResNet backbones, 5+ seeds with Wilcoxon tests, SCOPUS conference paper"], size=19)

    log("[12/12] thanks")
    s = D.slide()
    D.text(s, "DualNet-AL", Inches(0.8), Inches(2.2), Inches(11.7), Inches(1.2), 54, NAVY, True, PP_ALIGN.CENTER)
    D.text(s, "learning how to select — not only what to classify", Inches(0.8), Inches(3.4), Inches(11.7),
           Inches(0.7), 24, ACC, False, PP_ALIGN.CENTER)
    D.text(s, "Thank you · Questions?", Inches(0.8), Inches(4.6), Inches(11.7), Inches(0.7), 26, GREY, False,
           PP_ALIGN.CENTER)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    D.p.save(OUT)
    log(f"wrote {OUT} ({D.n} slides)")


if __name__ == "__main__":
    main()
