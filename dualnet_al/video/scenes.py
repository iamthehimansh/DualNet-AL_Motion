"""Manim motion-graphics scenes for the DualNet-AL mid-term / Phase III video.

Each scene pads itself to the narration length in durations.json (written by make_video.py).
All result numbers come from results_heldout/summary.json (held-out seeds, test set only).
"""
import json
import os

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, UL, Arrow, Circle, Create, Dot, FadeIn, FadeOut, GrowArrow,
                   GrowFromEdge, ImageMobject, Indicate, LaggedStart, Line, Rectangle, RoundedRectangle, Scene,
                   Text, ValueTracker, VGroup, Write, always_redraw, config, rate_functions)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DUR = json.load(open(os.path.join(HERE, "durations.json"))) if os.path.exists(os.path.join(HERE, "durations.json")) else {}
SUM = json.load(open(os.path.join(ROOT, "results_heldout", "summary.json")))
CHOSEN = json.load(open(os.path.join(ROOT, "results", "tuning", "chosen_config.json")))
SHOTS = os.path.join(ROOT, "results", "screenshots")

BG, ACC, ACC2, OK, WARN, FG, DIM = "#0b1220", "#38bdf8", "#f472b6", "#4ade80", "#fbbf24", "#e5e7eb", "#94a3b8"
SIG_COL = ["#f87171", "#60a5fa", "#4ade80", "#c084fc", "#fb923c"]
SIG_NAMES = ["Uncertainty", "Disagreement", "Representative", "Diversity", "Class balance"]
DS_TITLE = {"fmnist": "Fashion-MNIST", "svhn": "SVHN", "cifar10": "CIFAR-10"}
DSETS = [d for d in ["fmnist", "svhn", "cifar10"] if d in SUM["datasets"]]
config.background_color = BG
FONT = "DejaVu Sans"


def T(s, size=28, color=FG, weight="NORMAL"):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def box(text, w=3.2, h=1.0, color=ACC, size=22):
    r = RoundedRectangle(corner_radius=0.18, width=w, height=h, stroke_color=color, stroke_width=2.5,
                         fill_color=color, fill_opacity=0.12)
    return VGroup(r, T(text, size, FG).move_to(r))


class Base(Scene):
    key = ""

    def header(self, n, title):
        tag = T(f"{n:02d}", 20, ACC, "BOLD")
        t = T(title, 22, DIM)
        g = VGroup(tag, t).arrange(RIGHT, buff=0.25).to_corner(UL, buff=0.45)
        brand = T("DualNet-AL · BT-20", 18, DIM).to_corner(UP + RIGHT, buff=0.45)
        line = Line(LEFT * 7, RIGHT * 7, color="#1e293b").next_to(g, DOWN, buff=0.2).set_x(0)
        self.play(FadeIn(g, shift=RIGHT * 0.3), FadeIn(brand), Create(line), run_time=0.6)

    def finish(self):
        left = DUR.get(self.key, 0) - self.renderer.time
        if left > 0.05:
            self.wait(left)


class S01Title(Base):
    key = "S01Title"

    def construct(self):
        top = T("IILM University · School of CSE · Capstone Project BT-20", 22, DIM).to_edge(UP, buff=0.8)
        title = T("DualNet-AL", 96, FG, "BOLD")
        sub = T("A Bi-Network Meta-Learning Framework for Multi-Class Active Learning", 25, ACC)
        ph = T("Mid-Term & Phase III Progress", 30, ACC2)
        VGroup(title, sub, ph).arrange(DOWN, buff=0.4)
        team = T("Himansh Raj  ·  Om Narayan Pandey      Guide: Dr. Jaswinder Singh", 22, DIM).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(top))
        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.play(FadeIn(ph, scale=1.1), FadeIn(team))
        self.finish()


class S02Problem(Base):
    key = "S02Problem"

    def construct(self):
        self.header(1, "The problem")
        grid = VGroup(*[Rectangle(width=0.34, height=0.34, stroke_width=1, stroke_color="#334155",
                                  fill_color="#1e293b", fill_opacity=1) for _ in range(120)])
        grid.arrange_in_grid(10, 12, buff=0.07).move_to([-4.0, -0.3, 0])
        self.play(LaggedStart(*[FadeIn(g) for g in grid], lag_ratio=0.01), run_time=1.5)
        cost = T("Labelling every image is expensive", 30, FG)
        al = T("Active learning: label only\nthe most useful few", 26, ACC)
        bad = VGroup(T("× one model reinforces its own mistakes", 22, WARN),
                     T("× top-scored images are near-duplicates", 22, WARN),
                     T("× with few labels, uncertainty is unreliable", 22, WARN)).arrange(DOWN, aligned_edge=LEFT,
                                                                                         buff=0.25)
        col = VGroup(cost, al, bad).arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        if col.width > 7.0:
            col.width = 7.0
        col.move_to([2.9, -0.2, 0])
        self.play(FadeIn(cost, shift=LEFT * 0.3))
        picks = np.random.default_rng(1).choice(120, 12, replace=False)
        self.play(*[grid[i].animate.set_fill(ACC, 0.9) for i in picks], run_time=1)
        self.play(FadeIn(al))
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.2) for b in bad], lag_ratio=0.5), run_time=2)
        self.finish()


class S03TwoNets(Base):
    key = "S03TwoNets"

    def construct(self):
        self.header(2, "Two networks, two views")
        img = RoundedRectangle(corner_radius=0.1, width=1.4, height=1.4, fill_color="#334155", fill_opacity=1,
                               stroke_color=FG).shift(LEFT * 5)
        q = T("?", 60, FG).move_to(img)
        a = box("Net A\n3×3 deep · BatchNorm", 3.6, 1.3, ACC).shift(UP * 1.4 + LEFT * 0.6)
        b = box("Net B\n5×5 wide · GroupNorm", 3.6, 1.3, ACC2).shift(DOWN * 1.4 + LEFT * 0.6)
        self.play(FadeIn(img), Write(q))
        self.play(GrowArrow(Arrow(img.get_right(), a.get_left(), color=DIM)),
                  GrowArrow(Arrow(img.get_right(), b.get_left(), color=DIM)), FadeIn(a), FadeIn(b))
        pa = T("“Shirt” 0.81", 26, ACC).next_to(a, RIGHT, buff=0.6)
        pb = T("“Coat” 0.64", 26, ACC2).next_to(b, RIGHT, buff=0.6)
        self.play(FadeIn(pa, shift=RIGHT * 0.2), FadeIn(pb, shift=RIGHT * 0.2))
        dis = T("They disagree →\nworth labelling", 30, WARN, "BOLD").shift(RIGHT * 4.6)
        self.play(Write(dis), Indicate(pa), Indicate(pb))
        self.play(FadeIn(T("illustration: two networks that make different mistakes", 20, DIM).to_edge(DOWN, buff=0.5)))
        self.finish()


class S04Signals(Base):
    key = "S04Signals"

    def construct(self):
        self.header(3, "Five acquisition signals")
        desc = ["margin of the A+B prediction", "Jensen-Shannon divergence A‖B", "density of nearest neighbours",
                "distance to labelled set", "rarity of predicted class"]
        rows = VGroup()
        for n, d, c in zip(SIG_NAMES, desc, SIG_COL):
            rows.add(VGroup(Circle(0.16, color=c, fill_opacity=1), T(n, 28, c, "BOLD"), T(d, 22, DIM))
                     .arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.42).shift(LEFT * 1.8)
        for r in rows:
            r[2].next_to(r[1], RIGHT, buff=0.4)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in rows], lag_ratio=0.35), run_time=3)
        self.play(FadeIn(T("a meta-learner re-weights them after every round", 24, ACC).to_edge(DOWN, buff=0.6)))
        self.finish()


class S05V2(Base):
    key = "S05V2"

    def construct(self):
        self.header(4, "Phase III: DualNet-AL v2")
        ramp = int(CHOSEN["config"]["ramp"])
        lam0 = float(CHOSEN["config"]["lam0"])
        rounds = 8
        lam = [lam0 + (1 - lam0) * min(1.0, r / ramp) for r in range(rounds)]
        base_y, xs = -2.3, np.linspace(-5.6, -0.4, rounds)
        bars = VGroup()
        for x, l in zip(xs, lam):
            h1, h2 = (1 - l) * 3.6, l * 3.6
            g = VGroup(Rectangle(width=0.5, height=max(h1, 0.02), fill_color=SIG_COL[2], fill_opacity=0.95,
                                 stroke_width=0).move_to([x, base_y + h1 / 2, 0]),
                       Rectangle(width=0.5, height=max(h2, 0.02), fill_color=ACC2, fill_opacity=0.95,
                                 stroke_width=0).move_to([x, base_y + h1 + h2 / 2, 0]))
            bars.add(g)
        labels = VGroup(*[T(str(i), 16, DIM).move_to([x, base_y - 0.3, 0]) for i, x in enumerate(xs)])
        cap = T("round", 18, DIM).next_to(labels, DOWN, buff=0.15)
        leg = VGroup(VGroup(Rectangle(width=0.3, height=0.3, fill_color=SIG_COL[2], fill_opacity=1, stroke_width=0),
                            T("typical (dense) images", 18, FG)).arrange(RIGHT, buff=0.15),
                     VGroup(Rectangle(width=0.3, height=0.3, fill_color=ACC2, fill_opacity=1, stroke_width=0),
                            T("learned informativeness", 18, FG)).arrange(RIGHT, buff=0.15)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([-3.0, 2.2, 0])
        self.play(FadeIn(leg), FadeIn(labels), FadeIn(cap))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.15), run_time=2.5)
        pts = np.random.default_rng(3).normal(size=(60, 2)) * [1.1, 0.9] + [3.6, -0.3]
        dots = VGroup(*[Dot([p[0], p[1], 0], radius=0.05, color=DIM) for p in pts])
        right = VGroup(T("1 pick per cluster", 24, ACC, "BOLD"), T("whole distribution covered,", 20, FG),
                       T("no near-duplicates", 20, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.6, 2.2, 0])
        self.play(FadeIn(dots), FadeIn(right))
        cent = np.array([[2.7, 0.3], [4.4, 0.4], [3.4, -1.2], [4.6, -0.9], [2.7, -0.8]])
        self.play(LaggedStart(*[FadeIn(Dot([c[0], c[1], 0], radius=0.13, color=ACC2), scale=2) for c in cent],
                              lag_ratio=0.2), run_time=1.5)
        self.finish()


class S06Protocol(Base):
    key = "S06Protocol"

    def construct(self):
        self.header(5, "Fair, leak-free evaluation")
        seeds = SUM["datasets"][DSETS[0]]["seeds"]
        a = box("Train split\npool + validation", 3.6, 1.6, ACC, 22).shift(LEFT * 4.4)
        b = box("Tune on\nVALIDATION\nseeds 0-2", 3.6, 1.6, WARN, 22)
        c = box(f"Report TEST\nnew seeds\n{', '.join(map(str, seeds))}", 3.6, 1.6, OK, 22).shift(RIGHT * 4.4)
        self.play(FadeIn(a))
        self.play(GrowArrow(Arrow(a.get_right(), b.get_left(), color=DIM)), FadeIn(b))
        self.play(GrowArrow(Arrow(b.get_right(), c.get_left(), color=DIM)), FadeIn(c))
        self.play(FadeIn(T("same networks, same compute, same starting labels for every method", 22, DIM)
                         .to_edge(DOWN, buff=0.9)))
        self.finish()


class S07Results(Base):
    key = "S07Results"

    def construct(self):
        self.header(6, "Held-out results")
        keys = ["random", "bald", "coreset", "dualnet_meta", "dualnet_v2"]
        cols = {"random": "#9ca3af", "bald": "#a78bfa", "coreset": "#facc15", "dualnet_meta": "#fb923c",
                "dualnet_v2": ACC2}
        names = {"random": "Random", "bald": "BALD", "coreset": "Core-Set", "dualnet_meta": "DualNet v1",
                 "dualnet_v2": "DualNet-AL v2"}
        allv = [SUM["datasets"][d]["table"][k]["aulc"] * 100 for d in DSETS for k in keys
                if k in SUM["datasets"][d]["table"]]
        groups = VGroup()
        gw = 12.5 / len(DSETS)
        for gi, d in enumerate(DSETS):
            t = SUM["datasets"][d]["table"]
            x0 = -6.25 + gw * gi + gw / 2
            g = VGroup()
            ks = [k for k in keys if k in t]
            # per-dataset axis so differences inside each dataset are visible; labels show real values
            vals = [t[k]["aulc"] * 100 for k in ks]
            lo, hi = min(vals) - 1.5, max(vals) + 0.5
            for j, k in enumerate(ks):
                v = t[k]["aulc"] * 100
                h = (v - lo) / (hi - lo) * 3.6
                x = x0 + (j - (len(ks) - 1) / 2) * 0.62
                bar = Rectangle(width=0.5, height=max(h, 0.03), fill_color=cols[k], fill_opacity=0.95,
                                stroke_width=2 if k == "dualnet_v2" else 0, stroke_color=FG).move_to([x, -2.5 + h / 2, 0])
                g.add(VGroup(bar, T(f"{v:.1f}", 14, FG).next_to(bar, UP, buff=0.06)))
            groups.add(VGroup(g, T(DS_TITLE[d], 22, FG).move_to([x0, -2.85, 0])))
        legend = VGroup(*[VGroup(Rectangle(width=0.25, height=0.25, fill_color=cols[k], fill_opacity=1, stroke_width=0),
                                 T(names[k], 16, FG)).arrange(RIGHT, buff=0.1) for k in keys]).arrange(RIGHT, buff=0.35)
        legend.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(T("AULC % — area under accuracy-vs-labels curve (axis starts above zero per dataset)", 17, DIM)
                         .move_to([0, 2.15, 0])), FadeIn(legend))
        for g in groups:
            self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in g[0]], lag_ratio=0.15), FadeIn(g[1]), run_time=1.1)
            self.play(LaggedStart(*[FadeIn(b[1]) for b in g[0]], lag_ratio=0.1), run_time=0.4)
        self.finish()


class S08Findings(Base):
    key = "S08Findings"

    def construct(self):
        self.header(7, "What the numbers say")
        lines = VGroup()
        for d in DSETS:
            t = SUM["datasets"][d]["table"]
            v2, rd = t["dualnet_v2"], t["random"]
            p = v2["paired_vs_random"]
            dr = p["mean"] * 100
            lines.add(VGroup(T(DS_TITLE[d], 26, FG, "BOLD"),
                             T(f"vs Random {dr:+.2f} pp AULC", 24, OK if dr > 0 else WARN),
                             T(f"better on {p['wins']}/{p['n']} seeds", 22, OK if p["wins"] * 2 > p["n"] else WARN),
                             T(f"final acc {v2['final_acc'] * 100:.1f}% vs {rd['final_acc'] * 100:.1f}%", 22, DIM))
                      .arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        lines.arrange(RIGHT, buff=0.8, aligned_edge=UP).shift(UP * 0.3)
        if lines.width > 13:
            lines.width = 13
        self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.3) for l in lines], lag_ratio=0.4), run_time=2.5)
        n = SUM["datasets"][DSETS[0]]["seeds"]
        self.play(FadeIn(T(f"mean of {len(n)} held-out seeds · test set · reported exactly as measured", 20, DIM)
                         .to_edge(DOWN, buff=0.6)))
        self.finish()


class S09Prototype(Base):
    key = "S09Prototype"

    def construct(self):
        self.header(8, "Working prototype")
        mods = VGroup(*[T(s, 22, OK) for s in [
            "✓ two networks + five signals", "✓ meta-learner + v2 selection", "✓ 8 comparison methods, same code",
            "✓ live web demo, round by round", "✓ one-command launcher", "✓ 30 automated tests pass"]]
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).to_edge(LEFT, buff=0.6).shift(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.2) for m in mods], lag_ratio=0.25), run_time=2.5)
        files = sorted(f for f in os.listdir(SHOTS) if f.endswith(".png")) if os.path.isdir(SHOTS) else []
        shown = None
        for f in files[:3]:
            im = ImageMobject(os.path.join(SHOTS, f))
            im.height = 5.0
            if im.width > 6.6:
                im.width = 6.6
            im.to_edge(RIGHT, buff=0.4).shift(DOWN * 0.3)
            self.play(FadeOut(shown), FadeIn(im)) if shown is not None else self.play(FadeIn(im, shift=LEFT * 0.4))
            shown = im
            self.wait(max((DUR.get(self.key, 12) - 4) / max(len(files[:3]), 1) - 1, 0.5))
        self.finish()


class S10Next(Base):
    key = "S10Next"

    def construct(self):
        self.header(9, "Status & next steps")
        done = VGroup(T("Completed", 28, OK, "BOLD"), *[T("• " + s, 21, FG) for s in [
            "Full pipeline + v2 method", "8 baselines, same framework", "Leak-free held-out benchmark",
            "Tests + live demo"]]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        todo = VGroup(T("Next", 28, WARN, "BOLD"), *[T("• " + s, 21, FG) for s in [
            "CIFAR-100 & long-tail data", "Larger budgets, more seeds", "Significance tests",
            "SCOPUS conference paper"]]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        VGroup(done, todo).arrange(RIGHT, buff=1.6, aligned_edge=UP).shift(DOWN * 0.2)
        self.play(FadeIn(done, shift=RIGHT * 0.3))
        self.play(FadeIn(todo, shift=LEFT * 0.3))
        self.finish()


class S11End(Base):
    key = "S11End"

    def construct(self):
        g = VGroup(T("DualNet-AL", 80, FG, "BOLD"), T("learning how to select — not only what to classify", 28, ACC),
                   T("Thank you", 36, ACC2)).arrange(DOWN, buff=0.5)
        self.play(FadeIn(g[0], scale=1.2))
        self.play(Write(g[1]))
        self.play(FadeIn(g[2]))
        self.finish()


SCENES = ["S01Title", "S02Problem", "S03TwoNets", "S04Signals", "S05V2", "S06Protocol", "S07Results",
          "S08Findings", "S09Prototype", "S10Next", "S11End"]
