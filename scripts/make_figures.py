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


def split_ladder(df, metric: str = "spearman") -> None:
    """One line per system across the split ladder. The collapse is the point."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    order = [s for s in SPLIT_ORDER if s in set(df["split"])]
    for name, grp in df.groupby("name"):
        grp = grp.set_index("split").reindex(order).dropna(subset=[metric])
        if grp.empty:
            continue
        xs = [order.index(s) for s in grp.index]
        ax.plot(xs, grp[metric], marker="o", label=name, linewidth=2)
        if f"{metric}_std" in grp:
            ax.fill_between(
                xs,
                grp[metric] - grp[f"{metric}_std"],
                grp[metric] + grp[f"{metric}_std"],
                alpha=0.12,
            )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SPLIT_LABEL.get(s, s) for s in order], fontsize=8)
    ax.set_ylabel(f"{metric} (5-fold mean ± sd)")
    ax.set_xlabel("evaluation regime, easiest to hardest")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"split_ladder_{metric}.png", dpi=200)
    print(f"-> {OUT / f'split_ladder_{metric}.png'}")


def overall_vs_per_allele(df) -> None:
    """Overall Spearman flatters every model; within-allele ranking is the honest number."""
    sub = df.dropna(subset=["spearman", "spearman_per_allele_mean"])
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
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "overall_vs_per_allele.png", dpi=200)
    print(f"-> {OUT / 'overall_vs_per_allele.png'}")


if __name__ == "__main__":
    results = load_results()
    if results.empty:
        raise SystemExit("no results in artifacts/results/ yet")
    results = results.sort_values("file").drop_duplicates(["name", "split"], keep="last")
    split_ladder(results, "spearman")
    split_ladder(results, "spearman_per_allele_mean")
    overall_vs_per_allele(results)
