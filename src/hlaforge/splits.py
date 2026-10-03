"""Evaluation splits.

The scientific point of this harness is that a peptide-HLA model's apparent skill depends
almost entirely on how the data is split. A peptide appears under up to 36 different alleles
in this dataset, so a random row split puts near-identical examples on both sides. We define a
ladder of progressively harder regimes and report every model under all of them.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

Fold = tuple[np.ndarray, np.ndarray]

SPLIT_NAMES = ["random", "peptide", "allele", "peptide_cluster", "strict"]


def _encode_peptides(peptides: np.ndarray) -> np.ndarray:
    return np.array([[ord(c) for c in p] for p in peptides], dtype=np.uint8)


def cluster_peptides(peptides: list[str], identity: float = 0.78, chunk: int = 512) -> np.ndarray:
    """Single-linkage cluster equal-length peptides by fractional sequence identity.

    identity=0.78 means >=7/9 matching positions merges two 9-mers. Uses union-find over
    chunked pairwise comparisons so 5.6k peptides cluster in a couple of seconds.
    """
    enc = _encode_peptides(np.asarray(peptides))
    n, L = enc.shape
    min_matches = int(np.ceil(identity * L))
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for start in range(0, n, chunk):
        block = enc[start : start + chunk]
        matches = (block[:, None, :] == enc[None, :, :]).sum(axis=2)
        rows, cols = np.nonzero(matches >= min_matches)
        for r, c in zip(rows, cols, strict=True):
            i = start + int(r)
            j = int(c)
            if i < j:
                union(i, j)

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def _group_key(df: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "peptide":
        return df["peptide"].to_numpy()
    if kind == "allele":
        # Group on the pseudosequence: engineered constructs share a parent allele's groove,
        # and two alleles with an identical pseudoseq are not independent test material.
        return df["hla_pseudoseq"].to_numpy()
    if kind == "peptide_cluster":
        peps = df["peptide"].unique().tolist()
        labels = cluster_peptides(peps)
        mapping = dict(zip(peps, labels, strict=True))
        return df["peptide"].map(mapping).to_numpy()
    raise ValueError(f"unknown group kind: {kind}")


def make_folds(
    df: pd.DataFrame, split: str, n_splits: int = 5, seed: int = 0
) -> list[Fold]:
    """Return a list of (train_idx, test_idx) index arrays for the named split regime."""
    idx = np.arange(len(df))

    if split == "random":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in kf.split(idx)]

    if split in ("peptide", "allele", "peptide_cluster"):
        groups = _group_key(df, split)
        n_groups = len(np.unique(groups))
        if n_groups < n_splits:
            raise ValueError(
                f"split '{split}' has only {n_groups} groups, need >= n_splits={n_splits}"
            )
        gkf = GroupKFold(n_splits=n_splits)
        return [(tr, te) for tr, te in gkf.split(idx, groups=groups)]

    if split == "strict":
        return _strict_folds(df, n_splits=n_splits, seed=seed)

    raise ValueError(f"unknown split: {split}. Known: {SPLIT_NAMES}")


def _strict_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> list[Fold]:
    """Doubly-blocked folds: the test fold holds out peptide clusters AND alleles jointly.

    Training rows that share either the held-out peptide cluster or the held-out allele are
    dropped, so the test set contains only unseen peptides presented by unseen grooves. This is
    the hardest regime here and the one that most resembles designing for a new patient allele.
    """
    rng = np.random.default_rng(seed)
    clusters = _group_key(df, "peptide_cluster")
    alleles = _group_key(df, "allele")

    uniq_clusters = np.unique(clusters)
    uniq_alleles = np.unique(alleles)
    rng.shuffle(uniq_clusters)
    rng.shuffle(uniq_alleles)

    cluster_blocks = np.array_split(uniq_clusters, n_splits)
    allele_blocks = np.array_split(uniq_alleles, n_splits)

    folds: list[Fold] = []
    for cb, ab in zip(cluster_blocks, allele_blocks, strict=True):
        test_mask = np.isin(clusters, cb) & np.isin(alleles, ab)
        train_mask = ~np.isin(clusters, cb) & ~np.isin(alleles, ab)
        test_idx = np.nonzero(test_mask)[0]
        train_idx = np.nonzero(train_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        folds.append((train_idx, test_idx))
    if not folds:
        raise ValueError("strict split produced no usable folds")
    return folds


def describe_folds(df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Leakage audit: how much peptide/allele overlap does each fold actually have?"""
    rows = []
    for k, (tr, te) in enumerate(folds):
        tr_pep, te_pep = set(df["peptide"].iloc[tr]), set(df["peptide"].iloc[te])
        tr_al, te_al = set(df["hla_pseudoseq"].iloc[tr]), set(df["hla_pseudoseq"].iloc[te])
        rows.append(
            {
                "fold": k,
                "n_train": len(tr),
                "n_test": len(te),
                "test_peptides_seen_in_train": (
                    len(te_pep & tr_pep) / len(te_pep) if te_pep else 0.0
                ),
                "test_alleles_seen_in_train": len(te_al & tr_al) / len(te_al) if te_al else 0.0,
            }
        )
    return pd.DataFrame(rows)


def iter_named_splits(
    df: pd.DataFrame, names: list[str], n_splits: int = 5, seed: int = 0
) -> Iterator[tuple[str, list[Fold]]]:
    for name in names:
        yield name, make_folds(df, name, n_splits=n_splits, seed=seed)
