"""Re-score the affinity-transfer result on peptides the transfer source never saw.

88.5% of our peptides appear in MHCflurry's affinity training data, so the headline transfer
gain is measured partly on sequences the feature already knows. This restricts scoring to the
peptides MHCflurry has never seen -- training proceeds normally, only the evaluation subset
changes -- which is the honest estimate of what transfer buys on genuinely new peptides.

    python scripts/clean_transfer_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from hlaforge.data import load_raw
from hlaforge.experiment import ExperimentConfig, build_features
from hlaforge.models import build_model
from hlaforge.splits import make_folds

CURATED = Path(os.path.expanduser(
    "~/Library/Application Support/mhcflurry/4/2.3.0/data_curated/"
    "curated_training_data.affinity.csv.bz2"))
MASS_SPEC = CURATED.parent / "curated_training_data.mass_spec.csv.bz2"
OUT = Path("artifacts/clean_transfer.json")

CHEAP = {"kind": "cheap", "peptide": ["blosum", "physchem"], "hla": ["blosum"]}
ARMS = {
    "R1_blosum_xgb": dict(representation=CHEAP, interaction="concat",
                          model={"kind": "xgboost"}, target="rank"),
    "R6_affinity_plus_R1": dict(representation={**CHEAP, "affinity": {}}, interaction="concat",
                                model={"kind": "xgboost"}, target="rank"),
    "R6_affinity_alone": dict(representation={"kind": "affinity"}, interaction="concat",
                              model={"kind": "xgboost"}, target="rank"),
}


def mhcflurry_seen_peptides() -> set[str]:
    seen: set[str] = set()
    for path in [CURATED, MASS_SPEC]:
        if path.exists():
            mf = pd.read_csv(path, usecols=lambda c: c == "peptide")
            seen |= set(mf["peptide"].astype(str).str.upper())
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["peptide", "allele", "strict"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_raw()
    seen = mhcflurry_seen_peptides()
    is_novel = ~df["peptide"].isin(seen)
    print(f"peptides MHCflurry never saw: {df.loc[is_novel, 'peptide'].nunique()} "
          f"of {df['peptide'].nunique()}")
    print(f"rows on novel peptides      : {int(is_novel.sum())} of {len(df)} "
          f"({is_novel.mean():.1%})\n")

    y = df["y"].to_numpy(dtype=float)
    yb = df["y_binary"].to_numpy()
    novel = is_novel.to_numpy()

    from hlaforge.experiment import _fit_target

    alleles = df["allele"].to_numpy()
    rows = []
    for name, spec in ARMS.items():
        cfg = ExperimentConfig(name=name, **spec)
        X = build_features(df, cfg)
        for split in args.splits:
            pred = np.full(len(df), np.nan)
            for tr, te in make_folds(df, split, n_splits=args.folds, seed=args.seed):
                model = build_model(spec["model"]["kind"], spec["model"].get("params"),
                                    args.seed, reduce=spec["model"].get("reduce"))
                model.fit(X[tr], _fit_target(y, tr, alleles, cfg.target))
                pred[te] = model.predict(X[te])

            scored = ~np.isnan(pred)
            for label, mask in [("all", scored), ("novel_only", scored & novel)]:
                if mask.sum() < 50:
                    continue
                rho = float(spearmanr(y[mask], pred[mask]).statistic)
                auc = (float(roc_auc_score(yb[mask], pred[mask]))
                       if len(np.unique(yb[mask])) == 2 else float("nan"))
                per = []
                for a in np.unique(alleles[mask]):
                    m = mask & (alleles == a)
                    if m.sum() >= 20 and np.std(y[m]) > 0 and np.std(pred[m]) > 0:
                        per.append(float(spearmanr(y[m], pred[m]).statistic))
                rows.append({"arm": name, "split": split, "subset": label,
                             "n": int(mask.sum()), "spearman": rho, "auc": auc,
                             "per_allele": float(np.mean(per)) if per else float("nan")})
                print(f"  {name:22s} {split:8s} {label:11s} n={mask.sum():6d} "
                      f"rho={rho:.3f} per_allele={rows[-1]['per_allele']:.3f} auc={auc:.3f}",
                      flush=True)

    table = pd.DataFrame(rows)
    print("\nSpearman by subset\n")
    print(table.pivot(index=["split", "subset"], columns="arm",
                      values="spearman").to_string(float_format=lambda v: f"{v:.3f}"))

    print("\ntransfer gain over BLOSUM alone:")
    for split in args.splits:
        for subset in ["all", "novel_only"]:
            sel = table[(table.split == split) & (table.subset == subset)]
            if len(sel) < 2:
                continue
            base = sel[sel.arm == "R1_blosum_xgb"]["spearman"].iloc[0]
            best = sel[sel.arm == "R6_affinity_plus_R1"]["spearman"].iloc[0]
            print(f"  {split:8s} {subset:11s}: {best - base:+.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
