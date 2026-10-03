"""Does the foundation model earn its keep on DATA-POOR alleles?

This is the pan-specific promise: you have thousands of measurements for HLA-A*02:01 and seven
for HLA-B*13:02, and the hope is that a model pretrained on millions of proteins generalises to
the sparse tail where a supervised encoder cannot. Overall Spearman hides this completely --
the common alleles dominate the average.

So: hold out peptides, collect out-of-fold predictions, score each allele separately, and bucket
the alleles by how much training data they had.

    python scripts/allele_tail.py
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

OUT = Path("artifacts/allele_tail.json")

CONFIGS: dict[str, dict] = {
    "R1_blosum_xgb": dict(
        representation={"kind": "cheap", "peptide": ["blosum", "physchem"], "hla": ["blosum"]},
        interaction="concat",
        model={"kind": "xgboost"},
    ),
    "R1_netmhcstabpan_arch": dict(
        representation={"kind": "cheap", "peptide": ["blosum"], "hla": ["blosum"]},
        interaction="concat",
        model={"kind": "mlp", "params": {"hidden_layer_sizes": [56], "max_iter": 200}},
    ),
    "R2_esm2t33_mean": dict(
        representation={"kind": "plm", "plm": {"model": "esm2_t33", "pooling": "mean"}},
        interaction="concat",
        model={"kind": "xgboost", "reduce": 256},
    ),
    "R2_esm2t12_flatten": dict(
        representation={"kind": "plm", "plm": {"model": "esm2_t12", "pooling": "flatten"}},
        interaction="concat",
        model={"kind": "xgboost", "reduce": 256},
    ),
}

BUCKETS = [(0, 50), (50, 150), (150, 400), (400, 10_000)]


def out_of_fold(df: pd.DataFrame, spec: dict, split: str, folds: int, seed: int) -> np.ndarray:
    cfg = ExperimentConfig(name="oof", splits=[split], n_splits=folds, seed=seed, **spec)
    X = build_features(df, cfg)
    y = df["y"].to_numpy(dtype=np.float32)
    pred = np.full(len(df), np.nan)
    for train_idx, test_idx in make_folds(df, split, n_splits=folds, seed=seed):
        model = build_model(
            cfg.model.get("kind"), cfg.model.get("params"), seed, reduce=cfg.model.get("reduce")
        )
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="peptide")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-allele-n", type=int, default=15)
    args = ap.parse_args()

    df = load_raw()
    y = df["y"].to_numpy(dtype=float)
    counts = df["allele"].value_counts()

    per_allele: dict[str, dict[str, float]] = {}
    for name, spec in CONFIGS.items():
        print(f"running {name} ...", flush=True)
        pred = out_of_fold(df, spec, args.split, args.folds, args.seed)
        scores = {}
        for allele, n in counts.items():
            if n < args.min_allele_n:
                continue
            m = (df["allele"] == allele).to_numpy()
            if np.std(y[m]) == 0 or np.std(pred[m]) == 0:
                continue
            scores[allele] = float(spearmanr(y[m], pred[m]).statistic)
        per_allele[name] = scores
        print(f"  scored {len(scores)} alleles, mean rho {np.mean(list(scores.values())):.3f}")

    names = list(CONFIGS)
    rows = []
    for lo, hi in BUCKETS:
        alleles = [a for a, n in counts.items() if lo <= n < hi and a in per_allele[names[0]]]
        if not alleles:
            continue
        row = {"bucket": f"{lo}-{hi if hi < 10_000 else '+'}", "n_alleles": len(alleles),
               "median_rows": int(np.median([counts[a] for a in alleles]))}
        for name in names:
            vals = [per_allele[name][a] for a in alleles if a in per_allele[name]]
            row[name] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)

    table = pd.DataFrame(rows)
    print(f"\nmean within-allele Spearman by training-set size ({args.split} holdout)\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\ngap (best PLM minus best sequence model), per bucket:")
    for _, r in table.iterrows():
        best_seq = max(r["R1_blosum_xgb"], r["R1_netmhcstabpan_arch"])
        best_plm = max(r["R2_esm2t33_mean"], r["R2_esm2t12_flatten"])
        verdict = "PLM WINS" if best_plm > best_seq else ""
        print(f"  {r['bucket']:>8s} ({r['n_alleles']:2d} alleles, ~{r['median_rows']:4d} rows): "
              f"{best_plm - best_seq:+.3f}  {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"per_allele": per_allele, "buckets": rows,
                               "settings": vars(args)}, indent=2, default=float))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
