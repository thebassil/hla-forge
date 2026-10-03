import numpy as np

from hlaforge import data, splits


def test_peptide_split_has_no_peptide_overlap():
    df = data.synthetic(n=400, seed=2)
    folds = splits.make_folds(df, "peptide", n_splits=3)
    desc = splits.describe_folds(df, folds)
    assert (desc["test_peptides_seen_in_train"] == 0).all()


def test_allele_split_has_no_allele_overlap():
    df = data.synthetic(n=400, seed=3)
    folds = splits.make_folds(df, "allele", n_splits=3)
    desc = splits.describe_folds(df, folds)
    assert (desc["test_alleles_seen_in_train"] == 0).all()


def test_random_split_covers_all_rows_once():
    df = data.synthetic(n=200, seed=4)
    folds = splits.make_folds(df, "random", n_splits=4)
    test_all = np.concatenate([te for _, te in folds])
    assert sorted(test_all) == list(range(len(df)))


def test_strict_split_blocks_both_axes():
    df = data.synthetic(n=400, seed=5)
    folds = splits.make_folds(df, "strict", n_splits=3)
    desc = splits.describe_folds(df, folds)
    assert (desc["test_peptides_seen_in_train"] == 0).all()
    assert (desc["test_alleles_seen_in_train"] == 0).all()


def test_clustering_merges_near_identical_peptides():
    peps = ["AAAAAAAAA", "AAAAAAAAC", "WWWWWWWWW"]
    labels = splits.cluster_peptides(peps)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
