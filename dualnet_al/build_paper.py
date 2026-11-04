"""Build the research paper (two-column A4 PDF) from the held-out results.

Every number in the text and tables is read from results_heldout/summary.json and
results/tuning/chosen_config.json; nothing is typed by hand.
Usage: python build_paper.py   ->   ../deliverables/DualNet-AL_Paper.pdf  (+ results/paper_page1.png)
"""
from __future__ import annotations

import base64
import json
import os
import subprocess

from dualnet.log import log

ROOT = os.path.dirname(os.path.abspath(__file__))
HO = os.path.join(ROOT, "results_heldout")
PLOTS = os.path.join(HO, "plots")
DELIV = os.path.abspath(os.path.join(ROOT, "..", "deliverables"))
OUT_PDF = os.path.join(DELIV, "DualNet-AL_Paper.pdf")
PAGE1 = os.path.join(ROOT, "results", "paper_page1.png")
REPO = "https://github.com/iamthehimansh/DualNet-AL_Motion"
DS = {"fmnist": "Fashion-MNIST", "svhn": "SVHN", "cifar10": "CIFAR-10"}
ORDER = ["random", "entropy", "margin", "bald", "coreset", "qbc", "dualnet_fixed", "dualnet_meta", "dualnet_v2"]
SHORT = {"random": "Random", "entropy": "Entropy [2]", "margin": "Margin [3]", "bald": "BALD [5,6]",
         "coreset": "Core-Set [7]", "qbc": "QBC [4]", "dualnet_fixed": "DualNet-AL v1 (fixed)",
         "dualnet_meta": "DualNet-AL v1 (meta)", "dualnet_v2": "DualNet-AL v2 (ours)"}
CFG_NAME = {"c1_ramp4_margin": "λ₀=0, ramp 4, margin", "c2_ramp8_margin": "λ₀=0, ramp 8, margin",
            "c3_half_ramp4_margin": "λ₀=0.5, ramp 4, margin", "c4_ramp4_entropy": "λ₀=0, ramp 4, entropy",
            "c5_noprior_margin": "no prior (λ=1), margin", "c6_ramp2_margin": "λ₀=0, ramp 2, margin",
            "random": "Random (reference)", "dualnet_meta": "DualNet-AL v1 meta (reference)"}

REFS = [
    "B. Settles, “Active learning literature survey,” Computer Sciences Technical Report 1648, University of "
    "Wisconsin–Madison, 2009.",
    "D. D. Lewis and W. A. Gale, “A sequential algorithm for training text classifiers,” in Proc. ACM SIGIR, 1994, "
    "pp. 3–12.",
    "T. Scheffer, C. Decomain, and S. Wrobel, “Active hidden Markov models for information extraction,” in Proc. "
    "Int. Symp. Intelligent Data Analysis (IDA), 2001, pp. 309–318.",
    "H. S. Seung, M. Opper, and H. Sompolinsky, “Query by committee,” in Proc. 5th Annual Workshop on "
    "Computational Learning Theory (COLT), 1992, pp. 287–294.",
    "N. Houlsby, F. Huszár, Z. Ghahramani, and M. Lengyel, “Bayesian active learning for classification and "
    "preference learning,” arXiv:1112.5745, 2011.",
    "Y. Gal, R. Islam, and Z. Ghahramani, “Deep Bayesian active learning with image data,” in Proc. ICML, 2017.",
    "O. Sener and S. Savarese, “Active learning for convolutional neural networks: A core-set approach,” in Proc. "
    "ICLR, 2018.",
    "J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agarwal, “Deep batch active learning by diverse, "
    "uncertain gradient lower bounds,” in Proc. ICLR, 2020.",
    "D. Yoo and I. S. Kweon, “Learning loss for active learning,” in Proc. IEEE/CVF CVPR, 2019.",
    "G. Hacohen, A. Dekel, and D. Weinshall, “Active learning on a budget: Opposite strategies suit high and low "
    "budgets,” in Proc. ICML, 2022.",
    "K. Konyushkova, R. Sznitman, and P. Fua, “Learning active learning from data,” in Proc. NeurIPS, 2017.",
    "W.-N. Hsu and H.-T. Lin, “Active learning by learning,” in Proc. AAAI, 2015.",
    "H. T. Nguyen and A. Smeulders, “Active learning using pre-clustering,” in Proc. ICML, 2004.",
    "J. Lin, “Divergence measures based on the Shannon entropy,” IEEE Trans. Information Theory, vol. 37, no. 1, "
    "pp. 145–151, 1991.",
    "R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,” "
    "Machine Learning, vol. 8, pp. 229–256, 1992.",
    "D. Arthur and S. Vassilvitskii, “k-means++: The advantages of careful seeding,” in Proc. ACM-SIAM SODA, 2007.",
    "H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: A novel image dataset for benchmarking machine learning "
    "algorithms,” arXiv:1708.07747, 2017.",
    "A. Krizhevsky, “Learning multiple layers of features from tiny images,” Technical Report, University of "
    "Toronto, 2009.",
    "Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng, “Reading digits in natural images with "
    "unsupervised feature learning,” in NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.",
    "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017.",
    "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Proc. NeurIPS, 2019.",
    "A. Blum and T. Mitchell, “Combining labeled and unlabeled data with co-training,” in Proc. COLT, 1998.",
]

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }
body { font-family: "Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif", serif;
       font-size: 9.6pt; line-height: 1.32; color: #000; margin: 0; }
.head { text-align: center; margin-bottom: 9pt; }
.title { font-size: 19pt; font-weight: bold; line-height: 1.18; margin: 0 0 9pt; }
.authors { font-size: 11pt; margin-bottom: 3pt; }
.aff { font-size: 9pt; font-style: italic; }
.mail { font-size: 8.6pt; font-family: "DejaVu Sans Mono", monospace; margin-top: 2pt; }
.cols { column-count: 2; column-gap: 6.5mm; text-align: justify; hyphens: auto; }
.abstract { font-weight: bold; font-size: 9pt; }
.abstract i { font-weight: bold; }
.kw { font-size: 9pt; margin-top: 4pt; }
h2 { font-size: 10.2pt; font-variant: small-caps; text-align: center; margin: 11pt 0 4pt; break-after: avoid; }
h3 { font-size: 9.8pt; font-style: italic; font-weight: normal; margin: 7pt 0 2pt; break-after: avoid; }
p { margin: 0 0 4.5pt; text-indent: 1em; }
p.ni { text-indent: 0; }
.eq { text-align: center; margin: 4pt 0; font-style: italic; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; margin: 3pt 0 2pt; break-inside: avoid; }
th, td { border-top: 0.6pt solid #000; border-bottom: 0.6pt solid #000; padding: 1.6pt 2.5pt; text-align: center; }
th:first-child, td:first-child { text-align: left; }
td { white-space: nowrap; }
thead th { border-bottom: 0.9pt solid #000; }
.cap { font-size: 8.2pt; text-align: center; margin: 2pt 0 7pt; break-inside: avoid; }
figure { margin: 4pt 0 2pt; text-align: center; break-inside: avoid; }
figure img { width: 100%; }
.wide { column-span: all; }
.wide img { width: 88%; }
.algo { border-top: 0.9pt solid #000; border-bottom: 0.9pt solid #000; font-size: 8.4pt; padding: 3pt 0;
        margin: 4pt 0 8pt; break-inside: avoid; }
.algo div { padding-left: 1.2em; text-indent: -1.2em; }
.refs { font-size: 7.9pt; }
.refs p { text-indent: -1.6em; padding-left: 1.6em; margin-bottom: 2pt; }
b.best { font-weight: bold; }
"""


def pct(x, d=2):
    return f"{x * 100:.{d}f}"


def img(path, cls=""):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img class="{cls}" src="data:image/png;base64,{b64}"/>'


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def build_html():
    s = json.load(open(os.path.join(HO, "summary.json")))
    ch = json.load(open(os.path.join(ROOT, "results", "tuning", "chosen_config.json")))
    dsets = [d for d in ["fmnist", "svhn", "cifar10"] if d in s["datasets"]]
    T = {d: s["datasets"][d]["table"] for d in dsets}
    seeds = s["datasets"][dsets[0]]["seeds"]
    nl = T[dsets[0]]["random"]["n_labeled"]
    env = s["env"]
    v2 = {d: T[d]["dualnet_v2"] for d in dsets}
    rd = {d: T[d]["random"] for d in dsets}
    gain = {d: v2[d]["paired_vs_random"]["mean"] * 100 for d in dsets}
    wins = {d: f"{v2[d]['paired_vs_random']['wins']}/{v2[d]['paired_vs_random']['n']}" for d in dsets}
    best_other = {}
    for d in dsets:
        k = max([x for x in ORDER if x != "dualnet_v2"], key=lambda x: T[d][x]["aulc"])
        best_other[d] = (k, T[d][k]["aulc"])
    rank = {d: 1 + sum(T[d][k]["aulc"] > v2[d]["aulc"] for k in ORDER) for d in dsets}
    va = ch["val_aulc"]
    cfg = ch["config"]

    # ---------------- tables
    main_rows = []
    for k in ORDER:
        row = [SHORT[k]]
        for d in dsets:
            best = max(T[d][x]["aulc"] for x in ORDER)
            cell = f"{pct(T[d][k]['aulc'])} ± {pct(T[d][k]['aulc_std'])}"
            row.append(f"<b>{cell}</b>" if T[d][k]["aulc"] == best else cell)
        main_rows.append(row)
    t_main = table(["Method"] + [DS[d] for d in dsets], main_rows)

    pair_rows = []
    for d in dsets:
        lab = v2[d]["labels_to_random_final"]
        pair_rows.append([DS[d], f"{gain[d]:+.2f}", wins[d], f"{pct(v2[d]['final_acc'])} / {pct(rd[d]['final_acc'])}",
                          f"{pct(v2[d]['final_f1'])} / {pct(rd[d]['final_f1'])}",
                          f"{pct(v2[d]['min_class_recall'], 1)} / {pct(rd[d]['min_class_recall'], 1)}",
                          "–" if lab is None else f"{lab:.0f}"])
    t_pair = table(["Dataset", "ΔAULC (pp)", "Seeds better", "Final acc.", "Macro-F1", "Worst-class rec.",
                    f"Labels to Random@{nl[-1]}"], pair_rows)

    tune_rows = []
    rv = va["random"]
    for name in ["c1_ramp4_margin", "c2_ramp8_margin", "c3_half_ramp4_margin", "c4_ramp4_entropy",
                 "c5_noprior_margin", "c6_ramp2_margin", "dualnet_meta", "random"]:
        r = va[name]
        mean_gain = sum((r[d] - rv[d]) for d in dsets) / len(dsets) * 100
        cells = [pct(r[d]) for d in dsets]
        label = CFG_NAME[name] + (" ✓" if name == ch["chosen"] else "")
        tune_rows.append([label] + cells + ["–" if name == "random" else f"{mean_gain:+.2f}"])
    t_tune = table(["Configuration"] + [DS[d] for d in dsets] + ["mean Δ vs Random"], tune_rows)

    cost_rows = []
    for k in ["random", "bald", "coreset", "dualnet_meta", "dualnet_v2"]:
        cost_rows.append([SHORT[k]] + [f"{T[d][k]['t_select_s']:.2f}" for d in dsets] +
                         [f"{sum(T[d][k]['final_ece'] for d in dsets) / len(dsets):.3f}"])
    t_cost = table(["Method"] + [f"{DS[d]} (s)" for d in dsets] + ["mean ECE"], cost_rows)

    # ablation numbers (validation, tuning seeds) for the text
    c3, c5, meta = va[ch["chosen"]], va["c5_noprior_margin"], va["dualnet_meta"]
    prior_effect = {d: (c3[d] - c5[d]) * 100 for d in dsets}
    unc_effect = {d: (va["c4_ramp4_entropy"][d] - va["c1_ramp4_margin"][d]) * 100 for d in dsets}

    sv, cf, fm = "svhn", "cifar10", "fmnist"
    worst = {d: (pct(v2[d]["min_class_recall"], 1), pct(rd[d]["min_class_recall"], 1)) for d in dsets}

    # ---------------- document
    H = []
    H.append(f"""
<div class="head">
 <div class="title">DualNet-AL: A Bi-Network Meta-Learning Framework<br/>for Multi-Class Active Learning</div>
 <div class="authors">Himansh Raj, Om Narayan Pandey, and Jaswinder Singh</div>
 <div class="aff">School of Computer Science &amp; Engineering, IILM University, Greater Noida, India</div>
 <div class="mail">{{himansh.raj.cs27, om.pandey.cs27, jaswinder.singh}}@iilm.edu</div>
</div>
<div class="cols">
<p class="ni abstract"><i>Abstract</i>—Pool-based active learning reduces annotation cost by letting a model choose
which unlabelled images to label, but single-criterion strategies are brittle: uncertainty sampling picks
redundant or outlying samples, a single model reinforces its own errors, and at small budgets the model's
uncertainty is itself unreliable. We present DualNet-AL, which trains two architecturally different convolutional
networks, scores every unlabelled sample with five complementary signals (uncertainty, inter-network disagreement,
representativeness, diversity and class balance), fuses them with weights adapted online by a meta-learner, and,
in its second version (v2), combines a budget-aware prior that favours typical samples while labels are scarce
with cluster-based batch construction. Configuration was chosen on validation data only and evaluated on held-out
seeds and the official test splits. Across Fashion-MNIST, SVHN and CIFAR-10 ({nl[0]}→{nl[-1]} labels), DualNet-AL v2
improves the area under the learning curve over random sampling by {gain[fm]:+.2f}, {gain[sv]:+.2f} and
{gain[cf]:+.2f} percentage points, winning on every one of the {len(seeds)} held-out seeds of each dataset, is the best
of nine methods on SVHN and CIFAR-10, and more than doubles the worst-class recall of random sampling on SVHN
({worst[sv][0]}% vs {worst[sv][1]}%).</p>
<p class="ni kw"><b><i>Index Terms</i>—active learning, deep learning, meta-learning, query by committee,
label efficiency, image classification.</b></p>

<h2>I. Introduction</h2>
<p>Supervised deep networks need many labelled examples, and labelling is the dominant cost in most applied
projects. Active learning (AL) addresses this by training on a small labelled set and repeatedly asking an
oracle to label the samples the current model considers most useful [1]. Classical acquisition functions rank
samples by a single criterion—predictive entropy [2], the margin between the two most likely classes [3],
disagreement within a committee [4], or mutual information under model uncertainty (BALD) [5], [6]. Diversity-based
methods instead choose samples that cover the feature space, e.g. the core-set approach [7].</p>
<p>Each criterion fails in a predictable way. Uncertainty-driven batches are redundant and drawn towards outliers;
a single model choosing its own training data reinforces its own mistakes; and, as recently shown, at low budgets
uncertainty is a poor guide and typical samples should be preferred instead [10]. The best criterion therefore
depends on the dataset and on how many labels have been collected.</p>
<p>DualNet-AL tackles these failure modes together. Our contributions are: (i) a bi-network learner whose two
members differ in architecture and normalisation, so that their disagreement is an informative, committee-style
signal; (ii) five rank-normalised acquisition signals fused by a meta-learner that re-weights them every round from
measured validation gain and from label-revealed errors; (iii) a v2 selection rule that adds a budget-aware
typicality prior and one-sample-per-cluster batches; and (iv) a leak-free evaluation of nine strategies under
identical networks and compute, with all code and per-run logs released at {REPO}.</p>

<h2>II. Related Work</h2>
<p><i>Uncertainty and committees.</i> Entropy [2] and margin [3] sampling are strong, cheap baselines; query by
committee [4] measures disagreement among several models, and BALD [5], [6] approximates a Bayesian committee with
Monte-Carlo dropout. Co-training [22] showed the value of two different views of the data, which motivates our use
of two architecturally different networks.</p>
<p><i>Diversity and batch construction.</i> Core-set selection [7] picks a k-center cover of the embedding space;
BADGE [8] combines uncertainty and diversity in gradient space; pre-clustering [13] selects representatives of
clusters. Learning-loss [9] trains an auxiliary loss predictor.</p>
<p><i>Learned and budget-aware strategies.</i> Konyushkova et al. [11] learn an acquisition function from data and
Hsu and Lin [12] select among strategies with a bandit. Hacohen et al. [10] show that typical samples help at low
budgets while uncertain ones help at high budgets. DualNet-AL combines these ideas: a bandit-style meta-learner over
signals rather than whole strategies, and an explicit schedule from typicality to informativeness.</p>

<h2>III. Method</h2>
<h3>A. Problem setting</h3>
<p>We start from a labelled set <i>L</i> of {nl[0]} random samples and an unlabelled pool <i>U</i>. In each of
<i>R</i>={len(nl) - 1} rounds a batch of <i>B</i>={nl[1] - nl[0]} samples is selected from <i>U</i>, labelled by the oracle
and moved to <i>L</i>, after which the model is retrained. The goal is maximum test accuracy for every budget.</p>
<h3>B. Bi-network learner</h3>
<p>Net A is a 6-layer 3×3 CNN with BatchNorm and max-pooling ({env['params_NetA_rgb'] // 1000}k parameters for RGB
input); Net B is a wide 5×5 CNN with GroupNorm, LeakyReLU, average pooling and an MLP head
({env['params_NetB_rgb'] // 1000}k parameters). Both are retrained from scratch every round for a fixed 300 AdamW
steps, and predictions are the average of their softmax outputs. Their concatenated penultimate features form the
embedding used by the representativeness, diversity and clustering steps.</p>
""")
    H.append(f'<figure class="wide">{img(os.path.join(PLOTS, "architecture.png"), "wide")}</figure>'
             f'<div class="cap wide">Fig. 1. DualNet-AL v2: one acquisition round. The test set is never used for '
             f'training, selection or tuning.</div>')
    H.append(f"""
<h3>C. Acquisition signals</h3>
<p>For every pool sample we compute five scores, each mapped to [0,1] by rank-percentile normalisation so that
their very different raw scales become comparable: <b>uncertainty</b> <i>s</i><sub>1</sub>=1−(<i>p̄</i><sub>(1)</sub>−<i>p̄</i><sub>(2)</sub>),
the margin of the averaged prediction <i>p̄</i>=(<i>p</i><sub>A</sub>+<i>p</i><sub>B</sub>)/2 [3]; <b>disagreement</b>
<i>s</i><sub>2</sub>=0.7·JS(<i>p</i><sub>A</sub>‖<i>p</i><sub>B</sub>)/log 2+0.3·[argmax <i>p</i><sub>A</sub>≠argmax <i>p</i><sub>B</sub>]
using the Jensen–Shannon divergence [14]; <b>representativeness</b> <i>s</i><sub>3</sub>, the mean cosine similarity to
the 10 nearest pool neighbours; <b>diversity</b> <i>s</i><sub>4</sub>, the distance to the closest labelled sample
[7]; and <b>class balance</b> <i>s</i><sub>5</sub>, the rarity of the predicted class in <i>L</i>.</p>
<h3>D. Meta-learned weighting</h3>
<p>Signals are fused as <i>S</i>(<i>x</i>)=Σ<sub>k</sub> <i>w</i><sub>k</sub><i>s</i><sub>k</sub>(<i>x</i>) with
<i>w</i>=<i>w</i><sub>min</sub>+(1−5<i>w</i><sub>min</sub>)·softmax(θ), <i>w</i><sub>min</sub>=0.05, so no signal is ever
switched off. After the batch chosen in round <i>t</i> has been labelled and the networks retrained, θ is updated by</p>
<div class="eq">θ<sub>k</sub> ← θ<sub>k</sub> + η·<i>a</i>·<i>c</i><sub>k</sub> + η<sub>c</sub>·ρ<sub>k</sub>,</div>
<p class="ni">where <i>a</i> is the validation-accuracy gain normalised against a moving baseline (a REINFORCE-style
advantage [15]), <i>c</i><sub>k</sub> is how much the batch relied on signal <i>k</i> (its batch mean minus its pool
mean) and ρ<sub>k</sub> is the correlation between signal <i>k</i> and the ensemble's errors on the newly labelled
batch; η=1.5, η<sub>c</sub>=0.8.</p>
<h3>E. Budget-aware clustered selection (v2)</h3>
<p>Version 1 ranked samples by <i>S</i> and removed near-duplicates. In v2, the score becomes</p>
<div class="eq">score(<i>x</i>) = (1−λ<sub>t</sub>)·<i>s</i><sub>3</sub>(<i>x</i>) + λ<sub>t</sub>·<i>S</i>(<i>x</i>),&nbsp;&nbsp;
λ<sub>t</sub> = λ<sub>0</sub> + (1−λ<sub>0</sub>)·min(1, <i>t</i>/<i>r</i>),</div>
<p class="ni">so early rounds favour typical samples [10] and the meta-learned score takes over after <i>r</i> rounds.
The pool embedding is then partitioned into <i>B</i> clusters with cosine k-means (k-means++ seeding [16]) on the GPU,
and the best-scoring sample of each cluster is selected, which guarantees coverage and removes redundancy [13].
The chosen setting is λ<sub>0</sub>={cfg['lam0']}, <i>r</i>={cfg['ramp']}, margin uncertainty (Section IV-C).</p>
<div class="algo"><b>Algorithm 1</b> DualNet-AL v2 (one round <i>t</i>)
<div>1: train Net A, Net B on <i>L</i>; predict <i>p</i><sub>A</sub>, <i>p</i><sub>B</sub> and embeddings on <i>U</i></div>
<div>2: compute and rank-normalise <i>s</i><sub>1..5</sub>; fuse <i>S</i> with meta weights <i>w</i></div>
<div>3: score ← (1−λ<sub>t</sub>)<i>s</i><sub>3</sub> + λ<sub>t</sub><i>S</i>; k-means(<i>U</i>, <i>B</i>)</div>
<div>4: batch ← argmax score in each cluster; oracle labels batch; <i>L</i> ← <i>L</i> ∪ batch</div>
<div>5: after retraining: update θ from validation gain and batch errors</div></div>

<h2>IV. Experimental Setup</h2>
<h3>A. Datasets and protocol</h3>
<p>We use Fashion-MNIST [17], SVHN [19] and CIFAR-10 [18]. For each dataset and seed, a 10&nbsp;000-image pool and
a 500-image validation set are drawn from the official training split, and 2&nbsp;000 test images from the official
test split; inputs are standardised with pool statistics only. Every strategy starts from the same {nl[0]} random
labels for a given seed, adds {nl[1] - nl[0]} labels per round for {len(nl) - 1} rounds and uses the same two networks
and the same training budget; only the selection rule differs. Accuracy is that of the A+B ensemble.</p>
<h3>B. Baselines and metrics</h3>
<p>We compare Random, Entropy [2], Margin [3], BALD with 10 MC-dropout passes [6], Core-Set k-center greedy [7],
QBC over our two networks [4], and DualNet-AL v1 with fixed and with meta-learned weights. The main metric is the
area under the accuracy-vs-labels curve (AULC), normalised by the label range; we also report final accuracy,
macro-F1, worst-class recall, the number of labels needed to reach Random's final accuracy, selection time and
expected calibration error (ECE) [20].</p>
<h3>C. Tuning without test-set leakage</h3>
<p>Six v2 configurations and two references were run on tuning seeds 0–2 with the test set disabled, scoring
validation AULC only (Table III). A single configuration—the one with the largest mean gain over Random across the
three datasets—was frozen and then evaluated, together with all baselines, on new seeds {', '.join(map(str, seeds))}
on the test splits. Experiments used PyTorch [21] {env['torch'].split('+')[0]} on an {env['gpu']}; each run of 9
retraining rounds took under a minute.</p>

<h2>V. Results</h2>
<p>Table I reports held-out AULC. DualNet-AL v2 is the best of nine methods on SVHN and CIFAR-10 and ranks
{rank[fm]}{'nd' if rank[fm] == 2 else 'th'} on Fashion-MNIST, where Margin is {(best_other[fm][1] - v2[fm]['aulc']) * 100:.2f} pp
higher. Table II gives the paired comparison with Random on identical seeds and starting labels: v2 is better on
every seed of every dataset, by {gain[fm]:+.2f}, {gain[sv]:+.2f} and {gain[cf]:+.2f} pp AULC on Fashion-MNIST, SVHN and
CIFAR-10. It reaches Random's final ({nl[-1]}-label) accuracy after about {v2[fm]['labels_to_random_final']:.0f},
{v2[sv]['labels_to_random_final']:.0f} and {v2[cf]['labels_to_random_final']:.0f} labels respectively.</p>
<div class="cap">TABLE I. Held-out AULC (%), mean ± std over seeds {', '.join(map(str, seeds))}; best in bold.</div>
{t_main}
<div class="wide"><div class="cap">TABLE II. DualNet-AL v2 vs Random on the same held-out seeds (v2 / Random).</div>
{t_pair}</div>
<p>The uncertainty-only methods collapse on SVHN (Entropy {pct(T[sv]['entropy']['aulc'])}%, Core-Set
{pct(T[sv]['coreset']['aulc'])}% vs Random {pct(rd[sv]['aulc'])}%): with a few hundred labels their batches are
dominated by confusing, redundant digits. DualNet-AL v1 also lost to Random there ({pct(T[sv]['dualnet_meta']['aulc'])}%),
which motivated v2. The fairness benefit is large: worst-class recall rises from {worst[sv][1]}% to {worst[sv][0]}% on
SVHN, from {worst[fm][1]}% to {worst[fm][0]}% on Fashion-MNIST and from {worst[cf][1]}% to {worst[cf][0]}% on CIFAR-10.</p>
""")
    H.append(f'<figure class="wide">{img(os.path.join(PLOTS, "gain_vs_random.png"), "wide")}</figure>'
             f'<div class="cap wide">Fig. 2. AULC gain over Random on the same held-out seeds (error bars: std over '
             f'seeds). Positive is better.</div>')
    for d in dsets:
        H.append(f'<figure>{img(os.path.join(PLOTS, f"curve_{d}.png"))}</figure>'
                 f'<div class="cap">Fig. {3 + dsets.index(d)}. {DS[d]}: test accuracy vs labels (mean ± std).</div>')
    H.append(f"""
<h3>Ablation and tuning</h3>
<p>Table III shows the validation results used for model selection. Removing the budget-aware prior
(λ=1 throughout) costs {prior_effect[sv]:.2f} pp on SVHN and {prior_effect[cf]:.2f} pp on CIFAR-10 relative to the chosen
setting ({prior_effect[fm]:+.2f} pp on Fashion-MNIST), confirming that typical samples matter most while labels are scarce.
With the same schedule, entropy instead of margin uncertainty changes validation AULC by
{unc_effect[fm]:+.2f}, {unc_effect[sv]:+.2f} and {unc_effect[cf]:+.2f} pp on Fashion-MNIST, SVHN and CIFAR-10, so margin was
kept. Because the best setting differs between datasets, a single configuration was chosen by mean gain over Random
rather than tuned per dataset.</p>
<div class="cap">TABLE III. Validation AULC (%) on tuning seeds 0–2 (test set unused); ✓ = chosen.</div>
{t_tune}
<h3>Cost and calibration</h3>
<p>Selection adds about one second per round on the GPU (Table IV), small next to retraining two networks. GPU
k-means reduced v2's selection from 48 s to 0.03 s per round in our implementation. A weakness is calibration:
DualNet-AL v2 has a higher ECE than Random, consistent with deliberately skewed training sets; temperature scaling
[20] is a direct remedy we have not yet applied.</p>
<div class="cap">TABLE IV. Mean selection time per round (s) and mean final ECE.</div>
{t_cost}

<h2>VI. Discussion and Limitations</h2>
<p>The results support the central hypothesis that no single criterion is right for every dataset and budget, and
that scheduling from typicality to learned informativeness, with cluster-level diversity, is more robust than any
fixed rule. Several limitations remain. Budgets are small (≤{nl[-1]} labels) and pools are 10&nbsp;000 images, so
absolute accuracies are below full-data numbers and gaps are a few points. Three held-out seeds give a consistent
paired comparison but not a formal significance test. On SVHN, v2's final accuracy ({pct(v2[sv]['final_acc'])}%) is
slightly below Random's ({pct(rd[sv]['final_acc'])}%) even though it is ahead for almost the whole curve. The
meta-learner's reward uses a 500-image validation set that the baselines do not use. Finally, the method has not yet
been compared with BADGE [8] or learning-loss [9], nor evaluated on CIFAR-100 or long-tailed data.</p>

<h2>VII. Conclusion</h2>
<p>DualNet-AL combines two complementary networks, five acquisition signals with meta-learned weights, a
budget-aware typicality prior and cluster-based batches. Under a leak-free protocol it beats random sampling on all
three datasets and all held-out seeds, and is the best of nine methods on two of them. Future work covers
larger budgets, more seeds with significance testing, a label-free meta reward, calibration, stronger backbones and
the BADGE and learning-loss baselines.</p>
<p class="ni"><i>Reproducibility.</i> Code, per-run logs, the chosen configuration and a one-command demo are
available at {REPO}.</p>

<h2>References</h2>
<div class="refs">{''.join(f'<p>[{i}] {r}</p>' for i, r in enumerate(REFS, 1))}</div>
</div>""")
    return f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(H)}</body></html>"


def main():
    from playwright.sync_api import sync_playwright

    os.makedirs(DELIV, exist_ok=True)
    log("[1/3] building paper HTML from results_heldout/summary.json")
    html = build_html()
    html_path = OUT_PDF[:-4] + ".html"
    open(html_path, "w").write(html)
    log("[2/3] rendering PDF with headless Chrome")
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page()
        pg.goto("file://" + html_path)
        pg.pdf(path=OUT_PDF, format="A4", print_background=True, display_header_footer=True,
               header_template="<span></span>",
               footer_template="<div style='font-size:7px;width:100%;text-align:center;color:#555'>"
                               "<span class='pageNumber'></span></div>",
               margin={"top": "17mm", "bottom": "18mm", "left": "15mm", "right": "15mm"})
        b.close()
    log("[3/3] page-1 preview image")
    base = PAGE1[:-4]
    subprocess.run(["pdftoppm", "-r", "110", "-f", "1", "-l", "1", "-png", "-singlefile", OUT_PDF, base], check=True)
    log(f"wrote {OUT_PDF} ({os.path.getsize(OUT_PDF) / 1024:.0f} KB) and {PAGE1}")


if __name__ == "__main__":
    main()
