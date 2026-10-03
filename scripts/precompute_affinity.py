"""Compute and cache MHCflurry predictions in a process of their own.

MHCflurry pulls in a deep-learning backend; XGBoost pulls in its own OpenMP runtime. Loading
both into one process segfaults. Every other heavy feature source in this project is imported
lazily behind its cache for the same reason, but here the first computation must happen in
isolation. After this runs once, nothing downstream ever imports mhcflurry again.

    python scripts/precompute_affinity.py
"""

from __future__ import annotations

import numpy as np

from hlaforge.affinity import FEATURE_NAMES, predict_affinity
from hlaforge.data import load_raw


def main() -> None:
    df = load_raw()
    print(f"predicting affinity for {len(df)} peptide-allele pairs", flush=True)
    feat = predict_affinity(df)
    missing = int(np.isnan(feat[:, 0]).sum())
    print(f"cached {feat.shape[0]} x {feat.shape[1]}; {missing} rows unsupported by MHCflurry")

    from scipy.stats import spearmanr

    y = df["y"].to_numpy()
    ok = ~np.isnan(feat[:, 0])
    print("\nzero-shot correlation with measured half-life:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:26s} {spearmanr(y[ok], feat[ok, i]).statistic:+.3f}")


if __name__ == "__main__":
    main()
