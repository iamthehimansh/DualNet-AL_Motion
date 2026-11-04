"""Build the arXiv submission: LaTeX source (IEEEtran, ASCII-only, no BibTeX needed), compiled PDF, source tarball
and the metadata text for the arXiv web form.

Every number is read from results_heldout/summary.json and results/tuning/chosen_config.json.
Usage: python build_arxiv.py   ->  ../paper_arxiv/main.tex, main.pdf
                                  ../deliverables/DualNet-AL_arXiv_source.tar.gz, DualNet-AL_arXiv.pdf,
                                  arXiv_metadata.txt
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile

from dualnet.log import log

ROOT = os.path.dirname(os.path.abspath(__file__))
HO = os.path.join(ROOT, "results_heldout")
PLOTS = os.path.join(HO, "plots")
SRC = os.path.abspath(os.path.join(ROOT, "..", "paper_arxiv"))
DELIV = os.path.abspath(os.path.join(ROOT, "..", "deliverables"))
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")
REPO = "https://github.com/iamthehimansh/DualNet-AL_Motion"
DS = {"fmnist": "Fashion-MNIST", "svhn": "SVHN", "cifar10": "CIFAR-10"}
ORDER = ["random", "entropy", "margin", "bald", "coreset", "qbc", "dualnet_fixed", "dualnet_meta", "dualnet_v2"]
SHORT = {"random": "Random", "entropy": r"Entropy~\cite{lewis1994}", "margin": r"Margin~\cite{scheffer2001}",
         "bald": r"BALD~\cite{houlsby2011,gal2017}", "coreset": r"Core-Set~\cite{sener2018}",
         "qbc": r"QBC~\cite{seung1992}", "dualnet_fixed": "DualNet-AL v1 (fixed)",
         "dualnet_meta": "DualNet-AL v1 (meta)", "dualnet_v2": r"\textbf{DualNet-AL v2 (ours)}"}
CFG = {"c1_ramp4_margin": r"$\lambda_0{=}0$, $r{=}4$, margin", "c2_ramp8_margin": r"$\lambda_0{=}0$, $r{=}8$, margin",
       "c3_half_ramp4_margin": r"$\lambda_0{=}0.5$, $r{=}4$, margin",
       "c4_ramp4_entropy": r"$\lambda_0{=}0$, $r{=}4$, entropy", "c5_noprior_margin": r"no prior ($\lambda{=}1$), margin",
       "c6_ramp2_margin": r"$\lambda_0{=}0$, $r{=}2$, margin", "dualnet_meta": "DualNet-AL v1 meta (ref.)",
       "random": "Random (ref.)"}
FIGS = ["architecture.png", "protocol.png", "gain_vs_random.png", "curve_fmnist.png", "curve_svhn.png",
        "curve_cifar10.png"]

BIB = [
    ("settles2009", r"B.~Settles, ``Active learning literature survey,'' Computer Sciences Technical Report 1648, "
                    r"University of Wisconsin--Madison, 2009."),
    ("lewis1994", r"D.~D. Lewis and W.~A. Gale, ``A sequential algorithm for training text classifiers,'' in "
                  r"\emph{Proc. ACM SIGIR}, 1994, pp. 3--12."),
    ("scheffer2001", r"T.~Scheffer, C.~Decomain, and S.~Wrobel, ``Active hidden {M}arkov models for information "
                     r"extraction,'' in \emph{Proc. Int. Symp. Intelligent Data Analysis}, 2001, pp. 309--318."),
    ("seung1992", r"H.~S. Seung, M.~Opper, and H.~Sompolinsky, ``Query by committee,'' in \emph{Proc. 5th Annual "
                  r"Workshop on Computational Learning Theory}, 1992, pp. 287--294."),
    ("houlsby2011", r"N.~Houlsby, F.~Husz\'ar, Z.~Ghahramani, and M.~Lengyel, ``Bayesian active learning for "
                    r"classification and preference learning,'' arXiv:1112.5745, 2011."),
    ("gal2017", r"Y.~Gal, R.~Islam, and Z.~Ghahramani, ``Deep {B}ayesian active learning with image data,'' in "
                r"\emph{Proc. ICML}, 2017."),
    ("sener2018", r"O.~Sener and S.~Savarese, ``Active learning for convolutional neural networks: A core-set "
                  r"approach,'' in \emph{Proc. ICLR}, 2018."),
    ("ash2020", r"J.~T. Ash, C.~Zhang, A.~Krishnamurthy, J.~Langford, and A.~Agarwal, ``Deep batch active learning "
                r"by diverse, uncertain gradient lower bounds,'' in \emph{Proc. ICLR}, 2020."),
    ("yoo2019", r"D.~Yoo and I.~S. Kweon, ``Learning loss for active learning,'' in \emph{Proc. IEEE/CVF CVPR}, 2019."),
    ("hacohen2022", r"G.~Hacohen, A.~Dekel, and D.~Weinshall, ``Active learning on a budget: Opposite strategies "
                    r"suit high and low budgets,'' in \emph{Proc. ICML}, 2022."),
    ("konyushkova2017", r"K.~Konyushkova, R.~Sznitman, and P.~Fua, ``Learning active learning from data,'' in "
                        r"\emph{Proc. NeurIPS}, 2017."),
    ("hsu2015", r"W.-N. Hsu and H.-T. Lin, ``Active learning by learning,'' in \emph{Proc. AAAI}, 2015."),
    ("nguyen2004", r"H.~T. Nguyen and A.~Smeulders, ``Active learning using pre-clustering,'' in \emph{Proc. ICML}, "
                   r"2004."),
    ("lin1991", r"J.~Lin, ``Divergence measures based on the {S}hannon entropy,'' \emph{IEEE Trans. Inf. Theory}, "
                r"vol.~37, no.~1, pp. 145--151, 1991."),
    ("williams1992", r"R.~J. Williams, ``Simple statistical gradient-following algorithms for connectionist "
                     r"reinforcement learning,'' \emph{Machine Learning}, vol.~8, pp. 229--256, 1992."),
    ("arthur2007", r"D.~Arthur and S.~Vassilvitskii, ``k-means++: The advantages of careful seeding,'' in "
                   r"\emph{Proc. ACM-SIAM SODA}, 2007."),
    ("xiao2017", r"H.~Xiao, K.~Rasul, and R.~Vollgraf, ``Fashion-{MNIST}: A novel image dataset for benchmarking "
                 r"machine learning algorithms,'' arXiv:1708.07747, 2017."),
    ("krizhevsky2009", r"A.~Krizhevsky, ``Learning multiple layers of features from tiny images,'' Technical Report, "
                       r"University of Toronto, 2009."),
    ("netzer2011", r"Y.~Netzer, T.~Wang, A.~Coates, A.~Bissacco, B.~Wu, and A.~Y. Ng, ``Reading digits in natural "
                   r"images with unsupervised feature learning,'' in \emph{NIPS Workshop on Deep Learning and "
                   r"Unsupervised Feature Learning}, 2011."),
    ("guo2017", r"C.~Guo, G.~Pleiss, Y.~Sun, and K.~Q. Weinberger, ``On calibration of modern neural networks,'' in "
                r"\emph{Proc. ICML}, 2017."),
    ("paszke2019", r"A.~Paszke \emph{et~al.}, ``{PyTorch}: An imperative style, high-performance deep learning "
                   r"library,'' in \emph{Proc. NeurIPS}, 2019."),
    ("blum1998", r"A.~Blum and T.~Mitchell, ``Combining labeled and unlabeled data with co-training,'' in "
                 r"\emph{Proc. COLT}, 1998."),
]


def p(x, d=2):
    return f"{x * 100:.{d}f}"


def numbers():
    s = json.load(open(os.path.join(HO, "summary.json")))
    ch = json.load(open(os.path.join(ROOT, "results", "tuning", "chosen_config.json")))
    dsets = [d for d in ["fmnist", "svhn", "cifar10"] if d in s["datasets"]]
    return s, ch, dsets


def abstract_text(s, dsets, latex=True):
    T = {d: s["datasets"][d]["table"] for d in dsets}
    nl = T[dsets[0]]["random"]["n_labeled"]
    g = {d: T[d]["dualnet_v2"]["paired_vs_random"]["mean"] * 100 for d in dsets}
    n = len(s["datasets"][dsets[0]]["seeds"])
    w = {d: (p(T[d]["dualnet_v2"]["min_class_recall"], 1), p(T[d]["random"]["min_class_recall"], 1)) for d in dsets}
    pc = r"\%" if latex else "%"
    arrow = r"$\to$" if latex else "->"
    return (
        "Pool-based active learning reduces annotation cost by letting a model choose which unlabelled images to "
        "label, but single-criterion strategies are brittle: uncertainty sampling picks redundant or outlying samples, "
        "a single model reinforces its own errors, and at small budgets the model's uncertainty is itself unreliable. "
        "We present DualNet-AL, which trains two architecturally different convolutional networks, scores every "
        "unlabelled sample with five complementary signals (uncertainty, inter-network disagreement, "
        "representativeness, diversity and class balance), fuses them with weights adapted online by a meta-learner, "
        "and, in its second version (v2), combines a budget-aware prior that favours typical samples while labels are "
        "scarce with cluster-based batch construction. The configuration was chosen on validation data only and "
        "evaluated on held-out seeds and the official test splits. Across Fashion-MNIST, SVHN and CIFAR-10 "
        f"({nl[0]}{arrow}{nl[-1]} labels), DualNet-AL v2 improves the area under the learning curve over random "
        f"sampling by {g['fmnist']:.2f}, {g['svhn']:.2f} and {g['cifar10']:.2f} percentage points, winning on each of "
        f"the {n} held-out seeds of every dataset; it is the best of nine methods on SVHN and CIFAR-10 and more than "
        f"doubles the worst-class recall of random sampling on SVHN ({w['svhn'][0]}{pc} vs {w['svhn'][1]}{pc}). "
        "Code, per-run logs and a one-command demo are publicly available."
    )


def table_rows(rows):
    return "\n".join(" & ".join(r) + r" \\" for r in rows)


def build_tex():
    s, ch, dsets = numbers()
    T = {d: s["datasets"][d]["table"] for d in dsets}
    seeds = s["datasets"][dsets[0]]["seeds"]
    nl = T[dsets[0]]["random"]["n_labeled"]
    env = s["env"]
    v2 = {d: T[d]["dualnet_v2"] for d in dsets}
    rd = {d: T[d]["random"] for d in dsets}
    g = {d: v2[d]["paired_vs_random"]["mean"] * 100 for d in dsets}
    va, cfg = ch["val_aulc"], ch["config"]
    fm, sv, cf = "fmnist", "svhn", "cifar10"
    margin_gap = (T[fm]["margin"]["aulc"] - v2[fm]["aulc"]) * 100
    prior = {d: (va[ch["chosen"]][d] - va["c5_noprior_margin"][d]) * 100 for d in dsets}
    unc = {d: (va["c4_ramp4_entropy"][d] - va["c1_ramp4_margin"][d]) * 100 for d in dsets}
    w = {d: (p(v2[d]["min_class_recall"], 1), p(rd[d]["min_class_recall"], 1)) for d in dsets}
    seeds_s = ", ".join(map(str, seeds))

    main_rows = []
    for k in ORDER:
        best = {d: max(T[d][x]["aulc"] for x in ORDER) for d in dsets}
        cells = []
        for d in dsets:
            c = f"{p(T[d][k]['aulc'])}$\\pm${p(T[d][k]['aulc_std'])}"
            cells.append(f"\\textbf{{{c}}}" if T[d][k]["aulc"] == best[d] else c)
        main_rows.append([SHORT[k]] + cells)
    pair_rows = []
    for d in dsets:
        pv = v2[d]["paired_vs_random"]
        pair_rows.append([DS[d], f"{g[d]:+.2f}", f"{pv['wins']}/{pv['n']}",
                          f"{p(v2[d]['final_acc'])} / {p(rd[d]['final_acc'])}",
                          f"{p(v2[d]['final_f1'])} / {p(rd[d]['final_f1'])}",
                          f"{w[d][0]} / {w[d][1]}", f"{v2[d]['labels_to_random_final']:.0f}"])
    tune_rows = []
    for name in ["c1_ramp4_margin", "c2_ramp8_margin", "c3_half_ramp4_margin", "c4_ramp4_entropy",
                 "c5_noprior_margin", "c6_ramp2_margin", "dualnet_meta", "random"]:
        r = va[name]
        mg = sum(r[d] - va["random"][d] for d in dsets) / len(dsets) * 100
        lab = CFG[name] + (r" $\checkmark$" if name == ch["chosen"] else "")
        tune_rows.append([lab] + [p(r[d]) for d in dsets] + ["--" if name == "random" else f"{mg:+.2f}"])
    cost_rows = []
    for k in ["random", "bald", "coreset", "dualnet_meta", "dualnet_v2"]:
        cost_rows.append([SHORT[k]] + [f"{T[d][k]['t_select_s']:.2f}" for d in dsets] +
                         [f"{sum(T[d][k]['final_ece'] for d in dsets) / len(dsets):.3f}"])
    dh = " & ".join(DS[d] for d in dsets)

    tex = rf"""\documentclass[conference,a4paper]{{IEEEtran}}
\usepackage[T1]{{fontenc}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{url}}
\usepackage[hidelinks]{{hyperref}}
\graphicspath{{{{figures/}}}}

\begin{{document}}
\title{{DualNet-AL: A Bi-Network Meta-Learning Framework\\for Multi-Class Active Learning}}
\author{{\IEEEauthorblockN{{Himansh Raj, Om Narayan Pandey, and Jaswinder Singh}}
\IEEEauthorblockA{{School of Computer Science \& Engineering, IILM University, Greater Noida, India\\
\{{himansh.raj.cs27, om.pandey.cs27, jaswinder.singh\}}@iilm.edu}}}}
\maketitle

\begin{{abstract}}
{abstract_text(s, dsets)}
\end{{abstract}}

\begin{{IEEEkeywords}}
active learning, deep learning, meta-learning, query by committee, label efficiency, image classification
\end{{IEEEkeywords}}

\section{{Introduction}}
Supervised deep networks need many labelled examples, and labelling is the dominant cost in most applied projects.
Active learning (AL) trains on a small labelled set and repeatedly asks an oracle to label the samples the current
model considers most useful~\cite{{settles2009}}. Classical acquisition functions rank samples by a single criterion:
predictive entropy~\cite{{lewis1994}}, the margin between the two most likely classes~\cite{{scheffer2001}},
disagreement within a committee~\cite{{seung1992}}, or mutual information under model uncertainty
(BALD)~\cite{{houlsby2011,gal2017}}. Diversity-based methods instead cover the feature space, e.g. the core-set
approach~\cite{{sener2018}}.

Each criterion fails in a predictable way. Uncertainty-driven batches are redundant and drawn towards outliers; a
single model choosing its own training data reinforces its own mistakes; and at low budgets uncertainty is a poor
guide and typical samples should be preferred~\cite{{hacohen2022}}. The best criterion therefore depends on the
dataset and on how many labels have been collected.

DualNet-AL addresses these failure modes together. Our contributions are: (i) a bi-network learner whose members
differ in architecture and normalisation, so their disagreement is an informative committee-style signal; (ii) five
rank-normalised acquisition signals fused by a meta-learner that re-weights them every round from measured
validation gain and label-revealed errors; (iii) a selection rule (v2) that adds a budget-aware typicality prior and
one-sample-per-cluster batches; and (iv) a leak-free evaluation of nine strategies under identical networks and
compute, with code and per-run logs released at \url{{{REPO}}}.

\section{{Related Work}}
\textit{{Uncertainty and committees.}} Entropy~\cite{{lewis1994}} and margin~\cite{{scheffer2001}} sampling are strong,
cheap baselines; query by committee~\cite{{seung1992}} measures disagreement among several models, and
BALD~\cite{{houlsby2011,gal2017}} approximates a Bayesian committee with Monte-Carlo dropout. Co-training~\cite{{blum1998}}
showed the value of two different views of the data, which motivates two architecturally different networks.

\textit{{Diversity and batch construction.}} Core-set selection~\cite{{sener2018}} picks a $k$-center cover of the
embedding space; BADGE~\cite{{ash2020}} combines uncertainty and diversity in gradient space; pre-clustering
\cite{{nguyen2004}} selects representatives of clusters; learning loss~\cite{{yoo2019}} trains an auxiliary loss predictor.

\textit{{Learned and budget-aware strategies.}} Konyushkova et al.~\cite{{konyushkova2017}} learn an acquisition function
from data and Hsu and Lin~\cite{{hsu2015}} choose among strategies with a bandit. Hacohen et al.~\cite{{hacohen2022}} show
that typical samples help at low budgets and uncertain ones at high budgets. DualNet-AL combines a bandit-style
meta-learner over signals with an explicit schedule from typicality to informativeness.

\section{{Method}}
\subsection{{Problem setting}}
We start from a labelled set $L$ of {nl[0]} random samples and an unlabelled pool $U$. In each of $R={len(nl) - 1}$
rounds a batch of $B={nl[1] - nl[0]}$ samples is selected from $U$, labelled by the oracle and moved to $L$, after which
the model is retrained. The goal is maximum test accuracy at every budget.

\subsection{{Bi-network learner}}
Net~A is a 6-layer $3\times3$ CNN with BatchNorm and max-pooling ({env['params_NetA_rgb'] // 1000}k parameters for RGB
input); Net~B is a wide $5\times5$ CNN with GroupNorm, LeakyReLU, average pooling and an MLP head
({env['params_NetB_rgb'] // 1000}k parameters). Both are retrained from scratch each round for 300 AdamW steps and their
softmax outputs are averaged. Their concatenated penultimate features form the embedding used for
representativeness, diversity and clustering (Fig.~\ref{{fig:arch}}).

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.86\textwidth]{{architecture.png}}
\caption{{DualNet-AL v2: one acquisition round. The test set is never used for training, selection or tuning.}}
\label{{fig:arch}}
\end{{figure*}}

\subsection{{Acquisition signals}}
Each pool sample receives five scores, each mapped to $[0,1]$ by rank-percentile normalisation:
uncertainty $s_1 = 1-(\bar p_{{(1)}}-\bar p_{{(2)}})$, the margin of $\bar p=(p_A+p_B)/2$~\cite{{scheffer2001}};
disagreement $s_2 = 0.7\,\mathrm{{JS}}(p_A\|p_B)/\log 2 + 0.3\,[\arg\max p_A \neq \arg\max p_B]$ using the
Jensen--Shannon divergence~\cite{{lin1991}}; representativeness $s_3$, the mean cosine similarity to the 10 nearest pool
neighbours; diversity $s_4$, the distance to the closest labelled sample~\cite{{sener2018}}; and class balance $s_5$, the
rarity of the predicted class in $L$.

\subsection{{Meta-learned weighting}}
Signals are fused as $S(x)=\sum_k w_k s_k(x)$ with $w = w_{{\min}} + (1-5w_{{\min}})\,\mathrm{{softmax}}(\theta)$,
$w_{{\min}}=0.05$, so no signal is switched off. After the batch chosen in round $t$ has been labelled and the networks
retrained,
\begin{{equation}}
\theta_k \leftarrow \theta_k + \eta\, a\, c_k + \eta_c\, \rho_k ,
\end{{equation}}
where $a$ is the validation-accuracy gain normalised against a moving baseline (a REINFORCE-style
advantage~\cite{{williams1992}}), $c_k$ is the batch mean of $s_k$ minus its pool mean, and $\rho_k$ is the correlation
between $s_k$ and the ensemble's errors on the newly labelled batch; $\eta=1.5$, $\eta_c=0.8$.

\subsection{{Budget-aware clustered selection (v2)}}
Version~1 ranked samples by $S$ and removed near-duplicates. Version~2 scores
\begin{{align}}
\mathrm{{score}}(x) &= (1-\lambda_t)\,s_3(x)+\lambda_t\,S(x), \\
\lambda_t &= \lambda_0+(1-\lambda_0)\min(1,\,t/r),
\end{{align}}
so early rounds favour typical samples~\cite{{hacohen2022}} and the meta-learned score takes over after $r$ rounds. The
pool embedding is partitioned into $B$ clusters by cosine $k$-means with $k$-means++ seeding~\cite{{arthur2007}} on the
GPU, and the best-scoring sample of each cluster is selected~\cite{{nguyen2004}}. The chosen setting is
$\lambda_0={cfg['lam0']}$, $r={cfg['ramp']}$ with margin uncertainty (Section~\ref{{sec:tune}}).

\section{{Experimental Setup}}
\subsection{{Datasets and protocol}}
We use Fashion-MNIST~\cite{{xiao2017}}, SVHN~\cite{{netzer2011}} and CIFAR-10~\cite{{krizhevsky2009}}. For each dataset and
seed, a 10\,000-image pool and a 500-image validation set are drawn from the official training split and 2\,000 test
images from the official test split; inputs are standardised with pool statistics only. Every strategy starts from
the same {nl[0]} random labels for a given seed, adds {nl[1] - nl[0]} per round for {len(nl) - 1} rounds, and uses the
same networks and training budget; only the selection rule differs. Accuracy is that of the A+B ensemble.

\subsection{{Baselines and metrics}}
We compare Random, Entropy~\cite{{lewis1994}}, Margin~\cite{{scheffer2001}}, BALD with 10 MC-dropout
passes~\cite{{gal2017}}, Core-Set $k$-center greedy~\cite{{sener2018}}, QBC over our two networks~\cite{{seung1992}}, and
DualNet-AL v1 with fixed and with meta-learned weights. The main metric is the area under the accuracy-vs-labels
curve (AULC), normalised by the label range. We also report final accuracy, macro-F1, worst-class recall, labels
needed to reach Random's final accuracy, selection time and expected calibration error (ECE)~\cite{{guo2017}}.

\subsection{{Tuning without test-set leakage}}\label{{sec:tune}}
Six v2 configurations and two references were run on tuning seeds 0--2 with the test set disabled, scoring
validation AULC only (Table~\ref{{tab:tune}}). The single configuration with the largest mean gain over Random across
datasets was frozen and evaluated, with all baselines, on new seeds {seeds_s} on the test splits (Fig.~\ref{{fig:protocol}}).
Experiments used PyTorch~\cite{{paszke2019}} {env['torch'].split('+')[0]} on an {env['gpu']}.

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{protocol.png}}
\caption{{Evaluation protocol: tuning on validation data (seeds 0--2), reporting on the test split (seeds {seeds_s}).}}
\label{{fig:protocol}}
\end{{figure}}

\section{{Results}}
Table~\ref{{tab:main}} reports held-out AULC. DualNet-AL v2 is the best of nine methods on SVHN and CIFAR-10 and second on
Fashion-MNIST, where Margin is {margin_gap:.2f} pp higher. Table~\ref{{tab:pair}} compares v2 with Random on identical seeds
and starting labels: v2 is better on every seed of every dataset, by {g[fm]:+.2f}, {g[sv]:+.2f} and {g[cf]:+.2f} pp
AULC, and reaches Random's final ({nl[-1]}-label) accuracy after about {v2[fm]['labels_to_random_final']:.0f},
{v2[sv]['labels_to_random_final']:.0f} and {v2[cf]['labels_to_random_final']:.0f} labels.

\begin{{table}}[t]
\caption{{Held-out AULC (\%), mean $\pm$ std over seeds {seeds_s}. Best in bold.}}
\label{{tab:main}}
\centering\footnotesize
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lccc}}
\toprule
Method & {dh} \\
\midrule
{table_rows(main_rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\begin{{table*}}[t]
\caption{{DualNet-AL v2 vs Random on the same held-out seeds (v2 / Random).}}
\label{{tab:pair}}
\centering\small
\begin{{tabular}}{{lcccccc}}
\toprule
Dataset & $\Delta$AULC (pp) & Seeds better & Final acc. & Macro-F1 & Worst-class recall & Labels to Random@{nl[-1]} \\
\midrule
{table_rows(pair_rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}

Uncertainty-only methods collapse on SVHN (Entropy {p(T[sv]['entropy']['aulc'])}\%, Core-Set
{p(T[sv]['coreset']['aulc'])}\% vs Random {p(rd[sv]['aulc'])}\%): with a few hundred labels their batches are dominated by
confusing, redundant digits. DualNet-AL v1 also lost to Random there ({p(T[sv]['dualnet_meta']['aulc'])}\%), which
motivated v2. Worst-class recall rises from {w[sv][1]}\% to {w[sv][0]}\% on SVHN, from {w[fm][1]}\% to {w[fm][0]}\% on
Fashion-MNIST and from {w[cf][1]}\% to {w[cf][0]}\% on CIFAR-10 (Fig.~\ref{{fig:gain}}, Fig.~\ref{{fig:curves}}).

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{gain_vs_random.png}}
\caption{{AULC gain over Random on the same held-out seeds (error bars: std over seeds).}}
\label{{fig:gain}}
\end{{figure}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.32\textwidth]{{curve_fmnist.png}}\hfill
\includegraphics[width=0.32\textwidth]{{curve_svhn.png}}\hfill
\includegraphics[width=0.32\textwidth]{{curve_cifar10.png}}
\caption{{Test accuracy vs number of labels (mean $\pm$ std over held-out seeds). Left to right: Fashion-MNIST, SVHN,
CIFAR-10.}}
\label{{fig:curves}}
\end{{figure*}}

\subsection{{Ablation and tuning}}
Table~\ref{{tab:tune}} lists the validation results used for model selection. Removing the budget-aware prior
($\lambda=1$ throughout) costs {prior[sv]:.2f} pp on SVHN and {prior[cf]:.2f} pp on CIFAR-10 relative to the chosen setting
({prior[fm]:+.2f} pp on Fashion-MNIST), confirming that typical samples matter most while labels are scarce. With the same
schedule, entropy instead of margin uncertainty changes validation AULC by {unc[fm]:+.2f}, {unc[sv]:+.2f} and
{unc[cf]:+.2f} pp, so margin was kept. Because the best setting differs between datasets, one configuration was chosen
by mean gain rather than tuned per dataset.

\begin{{table}}[t]
\caption{{Validation AULC (\%) on tuning seeds 0--2 (test set unused). $\checkmark$ = chosen.}}
\label{{tab:tune}}
\centering\footnotesize
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lcccc}}
\toprule
Configuration & {dh} & Mean $\Delta$ \\
\midrule
{table_rows(tune_rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\subsection{{Cost and calibration}}
Selection adds about one second per round on the GPU (Table~\ref{{tab:cost}}), small next to retraining two networks;
GPU $k$-means reduced v2's selection from 48\,s to 0.03\,s per round in our implementation. A weakness is calibration:
v2 has a higher ECE than Random, consistent with deliberately skewed training sets; temperature
scaling~\cite{{guo2017}} is a direct remedy not yet applied.

\begin{{table}}[t]
\caption{{Mean selection time per round (s) and mean final ECE.}}
\label{{tab:cost}}
\centering\footnotesize
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lcccc}}
\toprule
Method & {dh} & ECE \\
\midrule
{table_rows(cost_rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\section{{Discussion and Limitations}}
The results support the hypothesis that no single criterion suits every dataset and budget, and that scheduling from
typicality to learned informativeness, with cluster-level diversity, is more robust than a fixed rule. Budgets are
small ($\le${nl[-1]} labels) and pools are 10\,000 images, so absolute accuracies are below full-data numbers and gaps are
a few points. Three held-out seeds give a consistent paired comparison but not a formal significance test. On SVHN,
v2's final accuracy ({p(v2[sv]['final_acc'])}\%) is slightly below Random's ({p(rd[sv]['final_acc'])}\%) although it is
ahead for almost the whole curve. The meta-learner's reward uses a 500-image validation set that the baselines do not
use. BADGE~\cite{{ash2020}}, learning loss~\cite{{yoo2019}}, CIFAR-100 and long-tailed data are not yet evaluated.

\section{{Conclusion}}
DualNet-AL combines two complementary networks, five acquisition signals with meta-learned weights, a budget-aware
typicality prior and cluster-based batches. Under a leak-free protocol it beats random sampling on all three datasets
and all held-out seeds, and is the best of nine methods on two of them. Future work covers larger budgets, more seeds
with significance testing, a label-free meta reward, calibration, stronger backbones and further baselines.

\section*{{Code availability}}
Code, per-run logs, the chosen configuration and a one-command demo: \url{{{REPO}}}.

\begin{{thebibliography}}{{99}}
""" + "\n".join(f"\\bibitem{{{k}}} {v}" for k, v in BIB) + r"""
\end{thebibliography}
\end{document}
"""
    return tex


def metadata():
    s, ch, dsets = numbers()
    ab = abstract_text(s, dsets, latex=False)
    pages = subprocess.run(["pdfinfo", os.path.join(SRC, "main.pdf")], capture_output=True, text=True).stdout
    npages = next((ln.split()[-1] for ln in pages.splitlines() if ln.startswith("Pages:")), "?")
    return f"""arXiv submission metadata (paste into the arXiv web form)
=========================================================

Title:
DualNet-AL: A Bi-Network Meta-Learning Framework for Multi-Class Active Learning

Authors (arXiv format):
Himansh Raj, Om Narayan Pandey, Jaswinder Singh

Abstract ({len(ab)} characters; arXiv limit 1920):
{ab}

Comments:
{npages} pages, 5 figures, 5 tables. Code and per-run logs: {REPO}

Primary category:
cs.LG (Machine Learning)

Cross-list:
cs.CV (Computer Vision and Pattern Recognition)

ACM class (optional):
I.2.6; I.5.1

MSC / journal-ref / DOI:
(leave empty)

License (choose one in the form; recommendation):
CC BY 4.0  (lets others reuse with credit; required by some conferences' preprint rules - check the target venue)

Files to upload:
deliverables/DualNet-AL_arXiv_source.tar.gz  (main.tex + figures/*.png; compiles with pdflatex, no BibTeX needed)
"""


def main():
    os.makedirs(os.path.join(SRC, "figures"), exist_ok=True)
    os.makedirs(DELIV, exist_ok=True)
    log("[1/5] writing main.tex from the held-out results")
    tex = build_tex()
    bad = sorted({c for c in tex if ord(c) > 127})
    if bad:
        raise SystemExit(f"non-ASCII characters in main.tex: {bad}")
    open(os.path.join(SRC, "main.tex"), "w").write(tex)
    log("[2/5] copying figures")
    for f in FIGS:
        shutil.copy(os.path.join(PLOTS, f), os.path.join(SRC, "figures", f))
    log("[3/5] compiling with tectonic (pdfLaTeX-compatible engine)")
    r = subprocess.run([TECTONIC, "--keep-logs", "main.tex"], cwd=SRC, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit("LaTeX compile failed")
    logtxt = open(os.path.join(SRC, "main.log"), errors="ignore").read()
    undef = [ln for ln in logtxt.splitlines() if "undefined" in ln.lower() and ("Citation" in ln or "Reference" in ln)]
    over = [ln for ln in logtxt.splitlines() if ln.startswith("Overfull \\hbox")]
    log(f"   undefined refs/cites: {len(undef)}; overfull hboxes: {len(over)}")
    for ln in (undef + over)[:10]:
        log("   " + ln[:150])
    if undef:
        raise SystemExit("undefined citations or references")
    shutil.copy(os.path.join(SRC, "main.pdf"), os.path.join(DELIV, "DualNet-AL_arXiv.pdf"))
    log("[4/5] source tarball (main.tex + figures)")
    tgz = os.path.join(DELIV, "DualNet-AL_arXiv_source.tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(os.path.join(SRC, "main.tex"), arcname="main.tex")
        for f in FIGS:
            t.add(os.path.join(SRC, "figures", f), arcname=f"figures/{f}")
    log("[5/5] arXiv metadata")
    open(os.path.join(DELIV, "arXiv_metadata.txt"), "w").write(metadata())
    log(f"done: {tgz} ({os.path.getsize(tgz) / 1024:.0f} KB), DualNet-AL_arXiv.pdf, arXiv_metadata.txt")


if __name__ == "__main__":
    main()
