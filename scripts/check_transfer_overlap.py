"""How much of our evaluation set did the transfer source already see?

The affinity-transfer result is the strongest in this project, and it rests on MHCflurry being
a legitimately external predictor. It is not trained on half-lives -- but it is trained on
binding affinities, and if it was trained on these same peptide sequences then our peptide
holdout does not hold them out from the feature, only from the final model.

This measures the overlap directly against MHCflurry's own published training data.

    python scripts/check_transfer_overlap.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from hlaforge.affinity import _normalise_allele
from hlaforge.data import load_raw

CURATED = Path(
    os.path.expanduser(
        "~/Library/Application Support/mhcflurry/4/2.3.0/data_curated/"
        "curated_training_data.affinity.csv.bz2"
    )
)
MASS_SPEC = CURATED.parent / "curated_training_data.mass_spec.csv.bz2"


def main() -> None:
    df = load_raw()
    ours_pep = set(df["peptide"])
    ours_pairs = {
        (_normalise_allele(a), p) for a, p in zip(df["allele"], df["peptide"], strict=True)
    }
    print(f"our data: {len(df)} rows, {len(ours_pep)} unique peptides, "
          f"{len(ours_pairs)} unique (allele, peptide) pairs\n")

    for label, path in [("affinity", CURATED), ("mass spec", MASS_SPEC)]:
        if not path.exists():
            print(f"{label}: {path.name} not found, skipping")
            continue
        mf = pd.read_csv(path, usecols=lambda c: c in {"allele", "peptide"})
        mf["peptide"] = mf["peptide"].astype(str).str.upper()
        mf_pep = set(mf["peptide"])
        mf_pairs = set(zip(mf["allele"].astype(str), mf["peptide"], strict=True))

        pep_overlap = ours_pep & mf_pep
        pair_overlap = ours_pairs & mf_pairs
        rows_affected = df["peptide"].isin(pep_overlap).sum()

        print(f"MHCflurry {label} training data: {len(mf)} rows, {len(mf_pep)} peptides")
        print(f"  peptides also in our set : {len(pep_overlap)} / {len(ours_pep)} "
              f"({len(pep_overlap) / len(ours_pep):.1%})")
        print(f"  our rows affected        : {rows_affected} / {len(df)} "
              f"({rows_affected / len(df):.1%})")
        print(f"  exact (allele, peptide)  : {len(pair_overlap)} / {len(ours_pairs)} "
              f"({len(pair_overlap) / len(ours_pairs):.1%})")
        print()

    print("Reading: a high peptide overlap means our peptide-holdout split is honest for the")
    print("final model but not for the transfer feature -- MHCflurry saw those sequences with")
    print("affinity labels. It is still a different measurement of a different quantity, and it")
    print("is how transfer is done in this field, but the number belongs in the write-up.")


if __name__ == "__main__":
    main()
