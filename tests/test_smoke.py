"""End-to-end smoke test: the whole pipeline on a tiny synthetic dataset, no network, no PLM."""

from hlaforge import data
from hlaforge.experiment import ExperimentConfig, run_experiment


def test_full_pipeline_learns_synthetic_signal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = data.synthetic(n=600, seed=7)

    baseline = ExperimentConfig(
        name="mean", model={"kind": "mean"}, splits=["random"], n_splits=3
    )
    learned = ExperimentConfig(
        name="ridge",
        representation={"kind": "cheap", "peptide": ["blosum"], "hla": ["blosum"]},
        model={"kind": "ridge"},
        splits=["random", "peptide"],
        n_splits=3,
    )

    base_rec = run_experiment(df, baseline, save=False, verbose=False)
    ridge_rec = run_experiment(df, learned, save=False, verbose=False)

    assert ridge_rec[0]["metrics"]["rmse"] < base_rec[0]["metrics"]["rmse"]
    assert ridge_rec[0]["metrics"]["spearman"] > 0.3
    for rec in ridge_rec:
        assert rec["metrics"]["n"] > 0
        assert rec["n_features"] == 9 * 20 + 34 * 20


def test_results_are_saved_with_config_and_sha(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = data.synthetic(n=200, seed=8)
    cfg = ExperimentConfig(name="saved", model={"kind": "ridge"}, splits=["random"], n_splits=2)
    recs = run_experiment(df, cfg, save=True, verbose=False)
    assert recs[0]["config"]["name"] == "saved"
    assert "git_sha" in recs[0]
    assert list((tmp_path / "artifacts" / "results").glob("*.json"))
