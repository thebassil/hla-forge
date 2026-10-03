import numpy as np

from hlaforge import data, features


def test_blosum_matrix_is_symmetric():
    assert features.BLOSUM62.shape == (20, 20)
    assert np.array_equal(features.BLOSUM62, features.BLOSUM62.T)
    # Known diagonal values: W scores 11 with itself, C scores 9.
    assert features.BLOSUM62[features.AA_INDEX["W"], features.AA_INDEX["W"]] == 11
    assert features.BLOSUM62[features.AA_INDEX["C"], features.AA_INDEX["C"]] == 9


def test_encoders_shapes():
    df = data.synthetic(n=50)
    assert features.encode_side(df, "peptide", ["onehot"]).shape == (len(df), 9 * 21)
    assert features.encode_side(df, "peptide", ["blosum"]).shape == (len(df), 9 * 20)
    assert features.encode_side(df, "hla", ["blosum"]).shape == (len(df), 34 * 20)
    assert features.encode_side(df, "peptide", ["composition"]).shape == (len(df), 21)


def test_onehot_is_one_per_position():
    df = data.synthetic(n=10)
    x = features.encode_side(df, "peptide", ["onehot"]).reshape(len(df), 9, 21)
    assert np.allclose(x.sum(axis=2), 1.0)
