"""Fine-tune ESM2 end to end on the same splits every frozen experiment used.

    python scripts/run_finetune.py --model esm2_t12 --splits peptide allele strict
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hlaforge.data import load_raw
from hlaforge.evaluate import aggregate_folds, compute_metrics
from hlaforge.finetune import FineTuneConfig, fit_predict
from hlaforge.splits import make_folds

OUT = Path("artifacts/results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t12")
    ap.add_argument("--splits", nargs="*", default=["peptide", "allele", "strict"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--hla-field", default="hla_pseudoseq")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_raw(limit=args.limit or None)
    cfg = FineTuneConfig(
        model=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        lora_rank=args.lora_rank, hla_field=args.hla_field, seed=args.seed,
    )
    name = args.name or (
        f"R5_finetune_{args.model}" + (f"_lora{args.lora_rank}" if args.lora_rank else "_full")
    )
    print(f"{name}: {len(df)} rows, {args.folds} folds, {args.epochs} epochs", flush=True)

    y = df["y"].to_numpy(dtype=float)
    yb = df["y_binary"].to_numpy()
    alleles = df["allele"].to_numpy()

    records = []
    for split in args.splits:
        folds = make_folds(df, split, n_splits=args.folds, seed=args.seed)
        fold_metrics = []
        t0 = time.time()
        for k, (train_idx, test_idx) in enumerate(folds):
            t1 = time.time()
            pred = fit_predict(df, train_idx, test_idx, cfg, verbose=False)
            m = compute_metrics(y[test_idx], pred, alleles=alleles[test_idx],
                                y_binary=yb[test_idx])
            fold_metrics.append(m)
            print(f"  {split} fold {k + 1}/{len(folds)}: rho={m['spearman']:.3f} "
                  f"[{time.time() - t1:.0f}s]", flush=True)
        agg = aggregate_folds(fold_metrics)
        records.append({
            "name": name, "split": split, "metrics": agg, "per_fold": fold_metrics,
            "n_rows": len(df), "n_folds": len(folds), "n_features": -1,
            "fit_seconds": round(time.time() - t0, 1),
            "config": {"finetune": vars(cfg) | {"folds": args.folds}}, "git_sha": "finetune",
        })
        print(f"  {split}: rho={agg['spearman']:.3f}+-{agg.get('spearman_std', 0):.3f} "
              f"per_allele={agg.get('spearman_per_allele_mean', float('nan')):.3f} "
              f"auc={agg.get('auc', float('nan')):.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}__{int(time.time())}.json"
    path.write_text(json.dumps(records, indent=2, default=float))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
