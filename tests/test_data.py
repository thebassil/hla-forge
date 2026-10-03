import numpy as np

from hlaforge import data


def test_synthetic_schema():
    df = data.synthetic(n=120, seed=1)
    for col in data.REQUIRED_COLUMNS + ["y", "y_binary", "censored"]:
        assert col in df.columns
    assert (df["peptide"].str.len() == 9).all()
    assert (df["hla_pseudoseq"].str.len() == 34).all()
    assert np.allclose(df["y"], np.log1p(df["thalf_hours"]))


def test_profile_runs():
    rep = data.profile(data.synthetic(n=100))
    assert rep.n_rows == 100 or rep.n_rows > 0
    assert rep.n_duplicate_pairs == 0
    assert "rows" in rep.render()
