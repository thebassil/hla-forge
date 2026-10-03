"""Run the ablation queue in order of expected information gain, and measure GROWTH RATE.

The point is not to find the best model. It is to measure, per reference system, how fast the
best-so-far improves per experiment -- so we can extrapolate whether that reference has any
chance of reaching a target before the deadline, and kill the ones that do not.

Screening tier: subsampled rows, 3 folds, hard splits only, SVD on embeddings. Each experiment
lands in tens of seconds, so a growth curve exists within the hour.

    python scripts/run_queue.py --limit 10000 --folds 3
    python scripts/run_queue.py --families R1 R2 --max-minutes 45
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hlaforge.data import load_raw
from hlaforge.experiment import ExperimentConfig, run_experiment

OUT = Path("artifacts/growth.json")
SCREEN_SPLITS = ["peptide", "allele", "strict"]
PRIMARY = "peptide"  # most stable of the honest splits; strict has 2-3x the fold variance


def queue_r1() -> list[tuple[str, dict]]:
    """R1 sequence reference: BLOSUM peptide + BLOSUM pseudoseq -> concat -> small net."""
    base = dict(
        representation={"kind": "cheap", "peptide": ["blosum"], "hla": ["blosum"]},
        interaction="concat",
        model={"kind": "mlp", "params": {"hidden_layer_sizes": [56], "max_iter": 200}},
        target="log1p",
    )
    out: list[tuple[str, dict]] = [("R1_reference", base)]

    # swap the predictor
    for kind in ["ridge", "hgb", "xgboost", "rf"]:
        out.append((f"R1_pred_{kind}", {**base, "model": {"kind": kind}}))
    # swap the encoding
    for enc in [["onehot"], ["physchem"], ["composition"], ["blosum", "physchem"],
                ["blosum", "physchem", "composition"]]:
        out.append(
            (f"R1_enc_{'+'.join(enc)}",
             {**base, "model": {"kind": "xgboost"},
              "representation": {"kind": "cheap", "peptide": enc, "hla": enc}})
        )
    # swap the glue
    for inter in ["peptide_only", "hla_only"]:
        out.append((f"R1_glue_{inter}", {**base, "model": {"kind": "xgboost"},
                                         "interaction": inter}))
    # swap the target
    out.append(("R1_target_raw", {**base, "model": {"kind": "xgboost"}, "target": "raw"}))
    return out


def queue_r2() -> list[tuple[str, dict]]:
    """R2 embedding reference: frozen ESM2 both sides -> concat -> trees."""
    def rep(model="esm2_t12", pooling="mean", layer=-1, hla_field="hla_pseudoseq"):
        return {"kind": "plm", "plm": {"model": model, "pooling": pooling, "layer": layer,
                                       "hla_field": hla_field}}

    base = dict(
        representation=rep(),
        interaction="concat",
        model={"kind": "xgboost", "reduce": 256},
        target="log1p",
    )
    out: list[tuple[str, dict]] = [("R2_reference", base)]

    for m in ["esm2_t6", "esm2_t30", "esm2_t33"]:
        out.append((f"R2_size_{m}", {**base, "representation": rep(model=m)}))
    for pool in ["cls", "max", "flatten"]:
        out.append((f"R2_pool_{pool}", {**base, "representation": rep(pooling=pool)}))
    for layer in [-2, -4]:
        out.append((f"R2_layer_{layer}", {**base, "representation": rep(layer=layer)}))
    out.append(("R2_hla_fullseq", {**base, "representation": rep(hla_field="hla_seq")}))
    for inter in ["product", "absdiff", "all"]:
        out.append((f"R2_glue_{inter}", {**base, "interaction": inter}))
    for kind in ["ridge", "hgb", "mlp"]:
        out.append((f"R2_pred_{kind}", {**base, "model": {"kind": kind, "reduce": 256}}))
    out.append(("R2_noreduce", {**base, "model": {"kind": "xgboost"}}))
    return out


def queue_r3() -> list[tuple[str, dict]]:
    """R3 scorer reference: masked-LM log-probs of the peptide inside the groove."""
    def rep(model="esm2_t12", mode="wt"):
        return {"kind": "likelihood", "likelihood": {"model": model, "mode": mode}}

    base = dict(
        representation=rep(),
        interaction="concat",
        model={"kind": "xgboost"},
        target="log1p",
    )
    out: list[tuple[str, dict]] = [("R3_reference", base)]
    for m in ["esm2_t6", "esm2_t30"]:
        out.append((f"R3_size_{m}", {**base, "representation": rep(model=m)}))
    for kind in ["ridge", "hgb"]:
        out.append((f"R3_pred_{kind}", {**base, "model": {"kind": kind}}))
    # the interesting one: likelihood bolted onto the R1 reference
    out.append(
        ("R3_plus_R1",
         {**base,
          "representation": {"kind": "cheap", "peptide": ["blosum", "physchem"],
                             "hla": ["blosum"],
                             "likelihood": {"model": "esm2_t12", "mode": "wt"}}})
    )
    return out


QUEUES = {"R1": queue_r1, "R2": queue_r2, "R3": queue_r3}


def growth_summary(history: list[dict]) -> dict:
    """Best-so-far curve per family, plus improvement rate per experiment and per hour."""
    out: dict[str, dict] = {}
    for family in sorted({h["family"] for h in history}):
        runs = [h for h in history if h["family"] == family and not np.isnan(h["score"])]
        if not runs:
            continue
        best_curve = []
        best = -np.inf
        for r in runs:
            best = max(best, r["score"])
            best_curve.append(best)
        total_minutes = sum(r["seconds"] for r in runs) / 60
        first, last = best_curve[0], best_curve[-1]
        tail = best_curve[-5:]
        out[family] = {
            "n_experiments": len(runs),
            "minutes_spent": round(total_minutes, 1),
            "best": round(last, 4),
            "best_config": max(runs, key=lambda r: r["score"])["name"],
            "gain_total": round(last - first, 4),
            "gain_per_experiment": round((last - first) / max(1, len(runs) - 1), 4),
            "gain_per_hour": round((last - first) / max(total_minutes / 60, 1e-6), 4),
            "gain_last_5": round(tail[-1] - tail[0], 4),
            "saturating": bool(len(tail) >= 4 and (tail[-1] - tail[0]) < 0.005),
            "curve": [round(v, 4) for v in best_curve],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=["R1", "R2", "R3"])
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--metric", default="spearman")
    args = ap.parse_args()

    df = load_raw(limit=args.limit)
    print(f"screening on {len(df)} rows, {args.folds} folds, splits={SCREEN_SPLITS}\n")

    jobs: list[tuple[str, str, dict]] = []
    for fam in args.families:
        for name, spec in QUEUES[fam]():
            jobs.append((fam, name, spec))

    history: list[dict] = []
    t_start = time.time()
    for i, (fam, name, spec) in enumerate(jobs, 1):
        if (time.time() - t_start) / 60 > args.max_minutes:
            print(f"\n[budget] stopping after {i - 1}/{len(jobs)} jobs")
            break
        cfg = ExperimentConfig(
            name=name, splits=SCREEN_SPLITS, n_splits=args.folds, seed=0, notes=fam, **spec
        )
        t0 = time.time()
        try:
            recs = run_experiment(df, cfg, save=True, verbose=False)
            by_split = {r["split"]: r["metrics"].get(args.metric, float("nan")) for r in recs}
            score = by_split.get(PRIMARY, float("nan"))
        except Exception as exc:
            print(f"  [{i}/{len(jobs)}] {name:28s} FAILED  {type(exc).__name__}: {exc}")
            history.append({"family": fam, "name": name, "score": float("nan"),
                            "seconds": time.time() - t0, "error": str(exc)})
            continue
        elapsed = time.time() - t0
        history.append({"family": fam, "name": name, "score": float(score),
                        "by_split": by_split, "seconds": elapsed})
        best = max(h["score"] for h in history
                   if h["family"] == fam and not np.isnan(h["score"]))
        flag = "  <-- new best" if score >= best and not np.isnan(score) else ""
        print(f"  [{i}/{len(jobs)}] {name:28s} {PRIMARY}={score:.3f}  "
              f"strict={by_split.get('strict', float('nan')):.3f}  "
              f"[{elapsed:.0f}s]{flag}")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(
            {"history": history, "growth": growth_summary(history),
             "settings": vars(args)}, indent=2, default=float))

    print("\n================ GROWTH RATE ================")
    g = growth_summary(history)
    hdr = f"{'family':8s} {'n':>3s} {'min':>6s} {'best':>7s} {'+/exp':>7s} {'+/hour':>7s} {'last5':>7s}"
    print(hdr)
    for fam, s in g.items():
        print(f"{fam:8s} {s['n_experiments']:>3d} {s['minutes_spent']:>6.1f} "
              f"{s['best']:>7.3f} {s['gain_per_experiment']:>+7.4f} "
              f"{s['gain_per_hour']:>+7.4f} {s['gain_last_5']:>+7.4f}"
              f"{'   SATURATED' if s['saturating'] else ''}")
        print(f"         best={s['best_config']}  curve={s['curve']}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
