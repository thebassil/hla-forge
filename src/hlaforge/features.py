"""Cheap, PLM-free sequence featurisers.

These exist so the foundation-model results have an honest comparator. If a 300MB protein
language model cannot beat BLOSUM one-hot encoding of a 9-mer plus a 34-residue pseudosequence,
that is the finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
UNK = len(AA)

# Kyte-Doolittle hydropathy, side-chain charge at pH 7, molecular weight (Da), volume (A^3).
PHYSCHEM = {
    "A": (1.8, 0.0, 89.1, 88.6), "C": (2.5, 0.0, 121.2, 108.5),
    "D": (-3.5, -1.0, 133.1, 111.1), "E": (-3.5, -1.0, 147.1, 138.4),
    "F": (2.8, 0.0, 165.2, 189.9), "G": (-0.4, 0.0, 75.1, 60.1),
    "H": (-3.2, 0.1, 155.2, 153.2), "I": (4.5, 0.0, 131.2, 166.7),
    "K": (-3.9, 1.0, 146.2, 168.6), "L": (3.8, 0.0, 131.2, 166.7),
    "M": (1.9, 0.0, 149.2, 162.9), "N": (-3.5, 0.0, 132.1, 114.1),
    "P": (-1.6, 0.0, 115.1, 112.7), "Q": (-3.5, 0.0, 146.2, 143.8),
    "R": (-4.5, 1.0, 174.2, 173.4), "S": (-0.8, 0.0, 105.1, 89.0),
    "T": (-0.7, 0.0, 119.1, 116.1), "V": (4.2, 0.0, 117.1, 140.0),
    "W": (-0.9, 0.0, 204.2, 227.8), "Y": (-1.3, 0.0, 181.2, 193.6),
}

# BLOSUM62 in canonical NCBI row/column order, reordered to AA on load.
_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_RAW = """
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def _blosum_matrix() -> np.ndarray:
    """Parse the NCBI-ordered table and reorder rows/cols into alphabetical AA order."""
    raw = {}
    for line in _BLOSUM62_RAW.strip().split("\n"):
        parts = line.split()
        raw[parts[0]] = [int(v) for v in parts[1:]]
    ncbi = {a: i for i, a in enumerate(_BLOSUM62_ORDER)}
    mat = np.zeros((20, 20), dtype=np.float32)
    for i, a in enumerate(AA):
        for j, b in enumerate(AA):
            mat[i, j] = raw[a][ncbi[b]]
    return mat


BLOSUM62 = _blosum_matrix()


def _indices(seqs: pd.Series | list[str], length: int) -> np.ndarray:
    out = np.full((len(seqs), length), UNK, dtype=np.int64)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:length]):
            out[i, j] = AA_INDEX.get(c, UNK)
    return out


def onehot(seqs: pd.Series | list[str], length: int) -> np.ndarray:
    """Position-specific one-hot encoding, flattened to length * 21."""
    idx = _indices(seqs, length)
    out = np.zeros((len(idx), length, len(AA) + 1), dtype=np.float32)
    np.put_along_axis(out, idx[:, :, None], 1.0, axis=2)
    return out.reshape(len(idx), -1)


def blosum(seqs: pd.Series | list[str], length: int) -> np.ndarray:
    """Position-specific BLOSUM62 row encoding, flattened to length * 20."""
    idx = _indices(seqs, length)
    table = np.vstack([BLOSUM62, np.zeros((1, 20), dtype=np.float32)])
    return table[idx].reshape(len(idx), -1)


def physchem(seqs: pd.Series | list[str], length: int) -> np.ndarray:
    """Per-position physicochemical descriptors plus whole-sequence aggregates."""
    default = (0.0, 0.0, 0.0, 0.0)
    per_pos = np.array(
        [[PHYSCHEM.get(c, default) for c in s[:length].ljust(length, "X")] for s in seqs],
        dtype=np.float32,
    )
    flat = per_pos.reshape(len(per_pos), -1)
    agg = np.concatenate([per_pos.mean(axis=1), per_pos.sum(axis=1), per_pos.std(axis=1)], axis=1)
    return np.concatenate([flat, agg], axis=1)


def composition(seqs: pd.Series | list[str], length: int) -> np.ndarray:
    """Position-independent amino-acid composition counts (bag of residues)."""
    idx = _indices(seqs, length)
    out = np.zeros((len(idx), len(AA) + 1), dtype=np.float32)
    for i, row in enumerate(idx):
        np.add.at(out[i], row, 1.0)
    return out


FEATURISERS = {
    "onehot": onehot,
    "blosum": blosum,
    "physchem": physchem,
    "composition": composition,
}

PEPTIDE_LENGTH = 9
PSEUDOSEQ_LENGTH = 34


def encode_side(df: pd.DataFrame, side: str, kinds: list[str]) -> np.ndarray:
    """Encode one side of the pair ('peptide' or 'hla') with one or more cheap featurisers."""
    if side == "peptide":
        seqs, length = df["peptide"].tolist(), PEPTIDE_LENGTH
    elif side == "hla":
        seqs, length = df["hla_pseudoseq"].tolist(), PSEUDOSEQ_LENGTH
    else:
        raise ValueError(f"unknown side: {side}")
    blocks = []
    for kind in kinds:
        if kind not in FEATURISERS:
            raise ValueError(f"unknown featuriser '{kind}'. Known: {list(FEATURISERS)}")
        blocks.append(FEATURISERS[kind](seqs, length))
    return np.concatenate(blocks, axis=1).astype(np.float32)
