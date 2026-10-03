"""Turn a queue result file (local or Modal) into the growth-rate table.

    python scripts/growth_report.py artifacts/growth_modal.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PRIMARY = "peptide"
METRIC = "spearman"


def load(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    rows = blob.get("results") or blob.get("history") or []
    out = []
    for r in rows:
        if "error" in r:
            continue
        by_split = r.get("by_split", {})
        # Modal returns full metric dicts per split; the local runner returns bare floats.
        def pick(split: str) -> float:
            v = by_split.get(split)
            if isinstance(v, dict):
                return float(v.get(METRIC, np.nan))
            return float(v) if v is not None else float("nan")

        out.append({
            "family": r["family"], "name": r["name"], "seconds": r.get("seconds", 0.0),
            "peptide": pick("peptide"), "allele": pick("allele"), "strict": pick("strict"),
        })
    return out


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/growth_modal.json")
    rows = load(path)
    if not rows:
        raise SystemExit(f"no usable results in {path}")

    print(f"{'family':7s} {'experiment':30s} {'peptide':>8s} {'allele':>8s} {'strict':>8s} {'s':>6s}")
    for fam in sorted({r["family"] for r in rows}):
        best = -np.inf
        for r in [x for x in rows if x["family"] == fam]:
            mark = ""
            if not np.isnan(r[PRIMARY]) and r[PRIMARY] > best:
                best, mark = r[PRIMARY], "  <-- best"
            print(f"{fam:7s} {r['name']:30s} {r['peptide']:8.3f} {r['allele']:8.3f} "
                  f"{r['strict']:8.3f} {r['seconds']:6.0f}{mark}")
        print()

    print("================ GROWTH RATE ================")
    print(f"{'family':7s} {'n':>3s} {'ref':>7s} {'best':>7s} {'gain':>7s} {'+/exp':>7s} "
          f"{'last5':>7s}  best config")
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam and not np.isnan(r[PRIMARY])]
        if not sub:
            continue
        curve, best = [], -np.inf
        for r in sub:
            best = max(best, r[PRIMARY])
            curve.append(best)
        ref = sub[0][PRIMARY]
        tail = curve[-5:]
        champ = max(sub, key=lambda r: r[PRIMARY])
        sat = " SATURATED" if len(tail) >= 4 and tail[-1] - tail[0] < 0.005 else ""
        print(f"{fam:7s} {len(sub):>3d} {ref:>7.3f} {curve[-1]:>7.3f} "
              f"{curve[-1] - ref:>+7.3f} {(curve[-1] - ref) / max(1, len(sub) - 1):>+7.4f} "
              f"{tail[-1] - tail[0]:>+7.4f}  {champ['name']}{sat}")


if __name__ == "__main__":
    main()
