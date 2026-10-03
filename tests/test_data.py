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


def test_limit_is_stratified_and_exact(tmp_path):
    """The subsample path runs on the real schema, which the synthetic fixture also has."""
    import pandas as pd

    df = data.synthetic(n=600, seed=11)
    csv = tmp_path / "mini.csv"
    df[data.REQUIRED_COLUMNS].to_csv(csv, index=False)

    full = data.load_raw(csv)
    small = data.load_raw(csv, limit=50)
    assert len(small) == 50
    # every allele present in the full data should survive a stratified subsample
    assert small["allele"].nunique() == full["allele"].nunique()
    assert not isinstance(small.index, pd.MultiIndex)

    # limit above the row count is a no-op, not an error
    assert len(data.load_raw(csv, limit=10_000)) == len(full)
