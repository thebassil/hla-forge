"""Where is the crossover? Starve every allele of training data and watch the two
representations trade places.

The observational version of this (scripts/allele_tail.py) found the foundation model's deficit
shrinking to -0.013 on alleles with ~20 measurements, against -0.20 on data-rich ones. But only
six alleles are that sparse, so the comparison is thin. Here we control it instead: cap the
training data at k rows per allele and sweep k. If pretraining buys anything, it buys it at
small k, and the curves should converge or cross.

    python scripts/data_curve.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hlaforge.data import load_raw
from hlaforge.experiment import ExperimentConfig, build_features
from hlaforge.models import build_model
from hlaforge.splits import make_folds

OUT = Path("artifacts/data_curve.json")

ARMS: dict[str, dict] = {
    "R1_blosum_xgb": dict(
        representation={"kind": "cheap", "peptide": ["blosum", "physchem"], "hla": ["blosum"]},
        interaction="concat",
        model={"kind": "xgboost"},
    ),
    "R2_esm2t12_flatten": dict(
        representation={"kind": "plm",
                        "plm": {"model": "esm2_t12", "pooling_peptide": "flatten",
                                "pooling_hla": "mean"}},
        interaction="concat",
        model={"kind": "xgboost", "reduce": 256},
    ),
    "R2_esm2t33_flatten": dict(
        representation={"kind": "plm",
                        "plm": {"model": "esm2_t33", "pooling_peptide": "flatten",
                                "pooling_hla": "mean"}},
        interaction="concat",
        model={"kind": "xgboost", "reduce": 256},
    ),
    # The configuration R2F actually found best: flatten BOTH sides, no compression.
    # Pooling the groove costs ~0.22, so the arms above understate what ESM can do.
    "R2_esm2t12_bothflat_full": dict(
        representation={"kind": "plm",
                        "plm": {"model": "esm2_t12", "pooling": "flatten"}},
        interaction="concat",
        model={"kind": "xgboost"},
    ),
}

CAPS = [10, 25, 50, 100, 200, 400, None]  # None = all available


def cap_training(df: pd.DataFrame, train_idx: np.ndarray, cap: int | None,
                 rng: np.random.Generator) -> np.ndarray:
    """Keep at most `cap` training rows per allele, chosen at random."""
    if cap is None:
        return train_idx
    sub = df.iloc[train_idx]
    keep: list[int] = []
    for _, group in sub.groupby("allele", sort=False):
        positions = group.index.to_numpy()
        if len(positions) > cap:
            positions = rng.choice(positions, cap, replace=False)
        keep.extend(positions.tolist())
    return np.array(sorted(keep))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="peptide")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-allele-n", type=int, default=20)
    ap.add_argument("--arms", nargs="*", default=None, help="subset of ARMS to run")
    ap.add_argument("--caps", nargs="*", type=int, default=None,
                    help="training-row caps; 0 means uncapped")
    args = ap.parse_args()

    global ARMS, CAPS
    if args.arms:
        ARMS = {k: v for k, v in ARMS.items() if k in args.arms}
    if args.caps:
        CAPS = [c if c > 0 else None for c in args.caps]

    df = load_raw()
    y = df["y"].to_numpy(dtype=float)
    alleles = df["allele"].to_numpy()
    rng = np.random.default_rng(args.seed)
    folds = make_folds(df, args.split, n_splits=args.folds, seed=args.seed)

    feature_cache = {name: build_features(df, ExperimentConfig(name=name, **spec))
                     for name, spec in ARMS.items()}
    for name, X in feature_cache.items():
        print(f"{name}: {X.shape[1]} features")

    rows = []
    for cap in CAPS:
        label = str(cap) if cap else "all"
        for name, spec in ARMS.items():
            X = feature_cache[name]
            pred = np.full(len(df), np.nan)
            n_train_total = 0
            for train_idx, test_idx in folds:
                tr = cap_training(df, train_idx, cap, rng)
                n_train_total += len(tr)
                model = build_model(spec["model"].get("kind"), spec["model"].get("params"),
                                    args.seed, reduce=spec["model"].get("reduce"))
                model.fit(X[tr], y[tr])
                pred[test_idx] = model.predict(X[test_idx])

            per = []
            for allele in np.unique(alleles):
                m = alleles == allele
                if m.sum() < args.min_allele_n or np.std(y[m]) == 0 or np.std(pred[m]) == 0:
                    continue
                per.append(float(spearmanr(y[m], pred[m]).statistic))
            overall = float(spearmanr(y[~np.isnan(pred)], pred[~np.isnan(pred)]).statistic)
            rows.append({"cap": label, "arm": name, "per_allele": float(np.mean(per)),
                         "overall": overall, "mean_train_rows": n_train_total // len(folds)})
            print(f"  cap={label:>4s} {name:22s} per-allele={np.mean(per):.3f} "
                  f"overall={overall:.3f}  (train n={n_train_total // len(folds)})")

    table = pd.DataFrame(rows).pivot(index="cap", columns="arm", values="per_allele")
    table = table.reindex([str(c) if c else "all" for c in CAPS])
    print("\nmean within-allele Spearman vs training rows per allele\n")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    plm_cols = [c for c in table.columns if c.startswith("R2")]
    if "R1_blosum_xgb" in table.columns and plm_cols:
        print("\nbest PLM minus BLOSUM, per cap:")
        for cap in table.index:
            gap = table.loc[cap, plm_cols].max() - table.loc[cap, "R1_blosum_xgb"]
            print(f"  cap={cap:>4s}: {gap:+.3f}  {'*** PLM WINS ***' if gap > 0 else ''}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "settings": vars(args)}, indent=2, default=float))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
