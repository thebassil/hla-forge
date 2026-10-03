"""Pitch figures from saved results. Two plots, both of which make an argument.

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hlaforge.experiment import load_results  # noqa: E402

OUT = Path("artifacts/figures")
SPLIT_ORDER = ["random", "peptide", "peptide_cluster", "allele", "strict"]
SPLIT_LABEL = {
    "random": "random\n(89% peptide leak)",
    "peptide": "unseen\npeptide",
    "peptide_cluster": "unseen\npeptide cluster",
    "allele": "unseen\nallele",
    "strict": "both\nunseen",
}


#: The systems worth putting in front of a reader, in the order they should be read.
#: Everything else in artifacts/results is an ablation that supports one of these.
HEADLINE = [
    ("R6_affinity_plus_R1", "BLOSUM + affinity transfer (ours)", "#0b6e4f", 2.6, "-"),
    ("R1C_target_rank", "BLOSUM + XGBoost", "#1b4f9c", 2.2, "-"),
    ("netmhcstabpan_arch", "NetMHCstabpan architecture", "#555555", 2.0, "--"),
    ("prott5_mean_xgb", "ProtT5-XL (3B)", "#d08c18", 1.8, "-"),
    ("esm2_t33_mean_xgb_svd", "ESM2-650M frozen", "#c0392b", 1.8, "-"),
    ("R4_reference", "ProteinMPNN geometry (13 features)", "#7b4fa0", 1.6, ":"),
]


def split_ladder(df, metric: str = "spearman", out_dir: Path = OUT) -> None:
    """Six systems across the split ladder. The collapse from left to right is the argument.

    Only the headline systems are drawn. Plotting all 34 experiments produces an unreadable
    tangle -- the ablations live in `hla report` and `hla delta`, not in a figure.
    """
    # The random split is excluded on purpose: 89% of its test peptides are in training, and
    # the strongest configurations were never run on it because the number would mean nothing.
    # The leakage audit (`hla splits`) is where that story belongs.
    order = [s for s in SPLIT_ORDER if s in set(df["split"]) and s != "random"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for name, label, colour, width, style in HEADLINE:
        grp = df[df["name"] == name].set_index("split").reindex(order)
        series = grp[metric].dropna()
        if series.empty:
            continue
        xs = [order.index(s) for s in series.index]
        ax.plot(xs, series.to_numpy(), marker="o", markersize=5, label=label,
                color=colour, linewidth=width, linestyle=style, zorder=3)
        std_col = f"{metric}_std"
        if std_col in grp:
            lo = (series - grp.loc[series.index, std_col].fillna(0)).to_numpy()
            hi = (series + grp.loc[series.index, std_col].fillna(0)).to_numpy()
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.10, zorder=1)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SPLIT_LABEL.get(s, s) for s in order], fontsize=9)
    ax.set_ylabel({"spearman": r"Spearman $\rho$",
                   "spearman_per_allele_mean": r"within-allele Spearman $\rho$"}
                  .get(metric, metric) + "  (5-fold mean ± sd)")
    ax.set_xlabel("evaluation regime, easiest to hardest")
    ax.set_ylim(0, 0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"split_ladder_{metric}.png", dpi=200)
    print(f"-> {out_dir / f'split_ladder_{metric}.png'}")


def overall_vs_per_allele(df, out_dir: Path = OUT) -> None:
    """Overall Spearman flatters every model; within-allele ranking is the honest number."""
    names = {n for n, *_ in HEADLINE}
    sub = df[df["name"].isin(names)].dropna(subset=["spearman", "spearman_per_allele_mean"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    for split, grp in sub.groupby("split"):
        ax.scatter(grp["spearman"], grp["spearman_per_allele_mean"], label=split, s=42)
    lim = [0, max(sub["spearman"].max(), 0.9) * 1.05]
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(0, lim[1])
    ax.set_xlabel("overall Spearman")
    ax.set_ylabel("mean within-allele Spearman")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "overall_vs_per_allele.png", dpi=200)
    print(f"-> {out_dir / 'overall_vs_per_allele.png'}")




def anchor_positions(out_dir: Path = OUT) -> None:
    """Zero-shot ProteinMPNN ranks peptide positions roughly by how buried they are.

    Per-position Spearman between the inverse-folding log-probability of each peptide residue
    and the measured half-life, with no training of any kind. P2 leads and P9 is elevated --
    the canonical anchors in the B and F pockets -- along with P4, a secondary anchor in several
    alleles. The only negative bars are P6 and P7, which point out of the groove toward solvent.
    """
    import numpy as np
    from scipy.stats import spearmanr

    from hlaforge.data import load_raw
    from hlaforge.structure import FEATURE_NAMES, score_structure

    df = load_raw()
    feat = score_structure(df)
    y = df["y"].to_numpy()
    rhos = [spearmanr(y, feat[:, i]).statistic for i in range(9)]

    anchors = {1, 8}  # zero-indexed P2 and P9, the canonical B- and F-pocket anchors
    colours = ["#c0392b" if i in anchors else "#95a5a6" for i in range(9)]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(range(1, 10), rhos, color=colours, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f"P{i}" for i in range(1, 10)])
    ax.set_xlabel("peptide position")
    ax.set_ylabel(r"Spearman $\rho$ with measured half-life")
    for i in anchors:
        ax.annotate("anchor", (i + 1, rhos[i]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color="#c0392b")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "mpnn_anchor_positions.png", dpi=200)
    print(f"-> {out_dir / 'mpnn_anchor_positions.png'}")


def scaling_curve(df, out_dir: Path = OUT) -> None:
    """87x the parameters, +0.023. Plotted against the baseline it has to beat."""
    import numpy as np

    sizes = {"esm2_t6_mean_xgb_svd": 7.5, "esm2_t12_mean_xgb_svd": 33.5,
             "esm2_t30_mean_xgb_svd": 148.1, "esm2_t33_mean_xgb_svd": 651.0}
    sub = df[df["name"].isin(sizes) & (df["split"] == "peptide")].copy()
    if sub.empty:
        return
    sub["params"] = sub["name"].map(sizes)
    sub = sub.sort_values("params")

    ref = df[(df["name"] == "blosum_xgb") & (df["split"] == "peptide")]["spearman"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogx(sub["params"], sub["spearman"], marker="o", linewidth=2,
                color="#2980b9", label="ESM2 (frozen, mean-pooled)")
    if not ref.empty:
        ax.axhline(float(ref.iloc[0]), color="#c0392b", linestyle="--", linewidth=2,
                   label="BLOSUM62 + XGBoost")
    ax.set_xlabel("ESM2 parameters (millions, log scale)")
    ax.set_ylabel(r"Spearman $\rho$, unseen peptides")
    ax.set_ylim(0, 0.9)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "scaling_curve.png", dpi=200)
    print(f"-> {out_dir / 'scaling_curve.png'}")


if __name__ == "__main__":
    results = load_results()
    if results.empty:
        raise SystemExit("no results in artifacts/results/ yet")
    results = results.sort_values("file").drop_duplicates(["name", "split"], keep="last")
    split_ladder(results, "spearman")
    split_ladder(results, "spearman_per_allele_mean")
    overall_vs_per_allele(results)
    scaling_curve(results)
    try:
        anchor_positions()
    except FileNotFoundError as exc:
        print(f"skipping anchor figure: {exc}")
