"""How long would a full cross-system ablation actually take?

All per-fit costs below are MEASURED on this machine (Apple M3, 8 cores, 24 GB) on the full
28,166-row dataset, not guessed. Run this to re-derive the budget after you measure new costs.

    python scripts/ablation_budget.py
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- measured seconds per split
# One "split" = n_folds fits + scoring. Values are the mean over the four regimes already run.
COST_S_PER_SPLIT = {
    ("cheap", "ridge"): 8.7,     # 860 features, measured 9.6 / 9.5 / 8.6 / 7.2
    ("cheap", "xgboost"): 41.8,  # 1,180 features, measured 39.6 / 53.6 / 40.3 / 33.5
    ("plm", "ridge"): 2.7,       # 960 dense features, measured 2.3 / 2.4 / 4.6 / 1.4
    ("plm", "xgboost"): 193.0,   # 960 dense features, measured on the random split
}
# Dense continuous embeddings are the problem: XGBoost is ~4.6x slower on 960 dense columns
# than on 1,180 mostly-integer BLOSUM columns, because every column is a candidate split point.

# Estimated from the measured pair above for predictors not yet timed end to end.
COST_S_PER_SPLIT.update(
    {
        ("cheap", "rf"): 60.0,
        ("cheap", "hgb"): 20.0,
        ("cheap", "mlp"): 90.0,
        ("plm", "rf"): 240.0,
        ("plm", "hgb"): 35.0,
        ("plm", "mlp"): 120.0,
    }
)

N_SPLITS_IN_LADDER = 5


@dataclass
class Axis:
    name: str
    cheap_variants: int
    plm_variants: int


# ------------------------------------------------------------------------------- the grid
REPRESENTATIONS = {
    # cheap: onehot, blosum, physchem, composition, blosum+physchem, everything
    "cheap": 6,
    # plm: 4 ESM2 sizes x 4 poolings x 3 layers x 2 HLA fields (pseudoseq vs full groove)
    "plm": 4 * 4 * 3 * 2,
    # hybrid: 4 sizes x 6 cheap combinations, pooling fixed to mean
    "hybrid": 4 * 6,
    # likelihood: 4 sizes x 2 scoring modes (wt, masked)
    "likelihood": 4 * 2,
}
INTERACTIONS_CHEAP = 3   # concat, peptide_only, hla_only
INTERACTIONS_PLM = 6     # the above plus product, absdiff, all
PREDICTORS = ["ridge", "rf", "hgb", "xgboost", "mlp"]
TARGETS = 2              # log1p, raw
SEEDS = 3


def _experiment_seconds(family: str, predictor: str, n_splits: int = N_SPLITS_IN_LADDER) -> float:
    bucket = "cheap" if family in ("cheap",) else "plm"
    return COST_S_PER_SPLIT[(bucket, predictor)] * n_splits


def full_factorial() -> tuple[int, float]:
    n = 0
    secs = 0.0
    for family, n_reps in REPRESENTATIONS.items():
        n_int = INTERACTIONS_CHEAP if family == "cheap" else INTERACTIONS_PLM
        for predictor in PREDICTORS:
            count = n_reps * n_int * TARGETS * SEEDS
            n += count
            secs += count * _experiment_seconds(family, predictor)
    return n, secs


def ofat() -> tuple[int, float]:
    """One factor at a time from a fixed anchor: sum of axis sizes, not their product."""
    # Representation variants explored individually, anchor = xgboost / concat / log1p.
    rep_variants = 6 + (4 + 4 + 3 + 2) + 4 + 8  # cheap, plm sub-axes, hybrid, likelihood
    configs = rep_variants + INTERACTIONS_PLM + len(PREDICTORS) + TARGETS
    n = configs * SEEDS
    # Most of these sit in the PLM family under the anchor predictor.
    secs = n * _experiment_seconds("plm", "xgboost")
    return n, secs


def tiered(subsample_rows: int = 8000, full_rows: int = 28166) -> tuple[int, float, str]:
    """Screen cheaply, confirm expensively. This is the design we can actually afford."""
    scale = subsample_rows / full_rows  # tree fit time is roughly linear in rows here
    rep_variants = 6 + (4 + 4 + 3 + 2) + 4 + 8
    screen_configs = rep_variants + INTERACTIONS_PLM + len(PREDICTORS) + TARGETS

    # Tier 1: subsampled rows, 3 splits, 3 folds, SVD-256, hgb predictor, one seed.
    tier1_per = COST_S_PER_SPLIT[("plm", "hgb")] * scale * (3 / 5) * 3
    tier1 = screen_configs * tier1_per

    # Tier 2: top 8 configs, full data, full 5-split ladder, xgboost, one seed.
    tier2 = 8 * _experiment_seconds("plm", "xgboost")

    # Tier 3: top 3 configs, 3 seeds, full ladder, for the error bars that go in the pitch.
    tier3 = 3 * 3 * _experiment_seconds("plm", "xgboost")

    n = screen_configs + 8 + 9
    note = (
        f"tier1 {screen_configs} screens @ {tier1_per:.0f}s = {tier1 / 60:.0f} min | "
        f"tier2 8 @ {_experiment_seconds('plm', 'xgboost') / 60:.0f} min = {tier2 / 3600:.1f} h | "
        f"tier3 9 runs = {tier3 / 3600:.1f} h"
    )
    return n, tier1 + tier2 + tier3, note


def fmt(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def main() -> None:
    total_reps = sum(REPRESENTATIONS.values())
    print(f"grid: {total_reps} representations x interactions x {len(PREDICTORS)} predictors "
          f"x {TARGETS} targets x {SEEDS} seeds")
    print(f"each experiment = {N_SPLITS_IN_LADDER} splits x 5 folds = "
          f"{N_SPLITS_IN_LADDER * 5} model fits\n")

    n, secs = full_factorial()
    print(f"FULL FACTORIAL      {n:>7,} experiments   {n * 25:>9,} fits   {fmt(secs):>10}")
    print(f"  ... on 10 parallel workers{'':>24}{fmt(secs / 10):>10}")

    n, secs = ofat()
    print(f"OFAT from anchor    {n:>7,} experiments   {n * 25:>9,} fits   {fmt(secs):>10}")

    n, secs, note = tiered()
    print(f"TIERED (screen+fix) {n:>7,} experiments   {'':>9}   {fmt(secs):>10}")
    print(f"  {note}")

    print("\nlevers that change these numbers:")
    print("  - SVD-256 on embeddings before the tree ensemble: expect a 3-5x cut on PLM+XGB")
    print("  - 3 folds instead of 5 during screening: 0.6x")
    print("  - 8k-row stratified subsample during screening: 0.28x")
    print("  - hgb instead of xgboost during screening: 5.5x on PLM features")
    print("  - the embedding passes themselves are NOT the bottleneck: they are cached, "
          "one-off, and minutes")


if __name__ == "__main__":
    main()
