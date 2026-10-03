"""Dataset loading and schema validation for the Rasmussen peptide-HLA stability data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["allele", "peptide", "thalf_hours", "hla_seq", "hla_pseudoseq"]
DEFAULT_RAW = Path("data/raw/rasmussen_stability.csv")

#: Half-life (hours) above which a complex is treated as "stable" for classification metrics.
#: Rasmussen et al. report a 1h threshold as a practical correlate of immunogenicity.
STABILITY_THRESHOLD_HOURS = 1.0


@dataclass(frozen=True)
class DatasetReport:
    n_rows: int
    n_alleles: int
    n_peptides: int
    n_pseudoseqs: int
    peptide_lengths: dict[int, int]
    n_zero_thalf: int
    thalf_min: float
    thalf_median: float
    thalf_max: float
    n_duplicate_pairs: int

    def render(self) -> str:
        lines = [
            f"rows                {self.n_rows}",
            f"alleles             {self.n_alleles}",
            f"unique peptides     {self.n_peptides}",
            f"unique pseudoseqs   {self.n_pseudoseqs}",
            f"peptide lengths     {self.peptide_lengths}",
            f"thalf == 0 (censored) {self.n_zero_thalf} "
            f"({100 * self.n_zero_thalf / self.n_rows:.1f}%)",
            f"thalf min/med/max   {self.thalf_min} / {self.thalf_median} / {self.thalf_max}",
            f"duplicate (allele, peptide) pairs  {self.n_duplicate_pairs}",
        ]
        return "\n".join(lines)


def load_raw(path: str | Path = DEFAULT_RAW, limit: int | None = None) -> pd.DataFrame:
    """Load the challenge CSV, validate its schema, and attach derived target columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the challenge CSV there "
            "(see scripts/prepare_data.py)."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["peptide"] = df["peptide"].str.strip().str.upper()
    df["hla_pseudoseq"] = df["hla_pseudoseq"].str.strip().str.upper()
    df["hla_seq"] = df["hla_seq"].str.strip().str.upper()
    df["thalf_hours"] = df["thalf_hours"].astype(float)

    if df["thalf_hours"].isna().any():
        raise ValueError("thalf_hours contains NaN values")
    if (df["thalf_hours"] < 0).any():
        raise ValueError("thalf_hours contains negative values")

    # Targets. Half-lives span 0-257h with ~20% exact zeros (below assay detection),
    # so a log1p transform is the regression target and a 1h cut gives the binary label.
    df["y"] = np.log1p(df["thalf_hours"])
    df["y_binary"] = (df["thalf_hours"] >= STABILITY_THRESHOLD_HOURS).astype(int)
    df["censored"] = (df["thalf_hours"] == 0.0).astype(int)

    if limit is not None:
        # Stratify the subsample by allele so small smoke runs still see many alleles.
        df = (
            df.groupby("allele", group_keys=False, sort=False)
            .apply(lambda g: g.head(max(1, limit // df["allele"].nunique())), include_groups=True)
            .head(limit)
            .reset_index(drop=True)
        )
    return df.reset_index(drop=True)


def profile(df: pd.DataFrame) -> DatasetReport:
    pairs = df.groupby(["allele", "peptide"]).size()
    return DatasetReport(
        n_rows=len(df),
        n_alleles=df["allele"].nunique(),
        n_peptides=df["peptide"].nunique(),
        n_pseudoseqs=df["hla_pseudoseq"].nunique(),
        peptide_lengths=df["peptide"].str.len().value_counts().to_dict(),
        n_zero_thalf=int((df["thalf_hours"] == 0).sum()),
        thalf_min=float(df["thalf_hours"].min()),
        thalf_median=float(df["thalf_hours"].median()),
        thalf_max=float(df["thalf_hours"].max()),
        n_duplicate_pairs=int((pairs > 1).sum()),
    )


def synthetic(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic dataset with the real schema, for tests that must not touch data/."""
    rng = np.random.default_rng(seed)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    alleles = [f"HLA-X*{i:02d}:01" for i in range(5)]
    pseudo = {a: "".join(rng.choice(aa, 34)) for a in alleles}
    full = {a: "".join(rng.choice(aa, 182)) for a in alleles}
    peptides = ["".join(rng.choice(aa, 9)) for _ in range(n // 2)]
    rows = []
    for i in range(n):
        a = alleles[i % len(alleles)]
        p = peptides[i % len(peptides)]
        # Signal: anchor residues at P2/P9 plus an allele offset, so models can learn something.
        signal = (p[1] in "LMIV") * 1.5 + (p[8] in "VLIK") * 1.0 + alleles.index(a) * 0.3
        rows.append(
            {
                "allele": a,
                "peptide": p,
                "thalf_hours": max(0.0, round(float(np.expm1(signal + rng.normal(0, 0.4))), 1)),
                "hla_seq": full[a],
                "hla_pseudoseq": pseudo[a],
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["allele", "peptide"]).reset_index(drop=True)
    df["y"] = np.log1p(df["thalf_hours"])
    df["y_binary"] = (df["thalf_hours"] >= STABILITY_THRESHOLD_HOURS).astype(int)
    df["censored"] = (df["thalf_hours"] == 0.0).astype(int)
    return df
