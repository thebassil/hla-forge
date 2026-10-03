"""Contracts for the heavyweight feature sources, without loading any heavyweight model.

Each of these modules imports torch, mhcflurry or ProteinMPNN only after its cache check, so
that downstream experiments never pull a deep-learning backend into the same process as
XGBoost. These tests pin that behaviour and the pure-python parts around it.
"""

import numpy as np
import pytest

from hlaforge import affinity, embeddings, likelihood, structure
from hlaforge.data import synthetic


def test_feature_name_lists_match_their_widths():
    assert len(likelihood.FEATURE_NAMES) == 14
    assert len(structure.FEATURE_NAMES) == 13
    assert len(affinity.FEATURE_NAMES) == 5


def test_engineered_constructs_fall_back_to_the_parent_allele():
    assert affinity._normalise_allele("HLA-B*14:01(C67S)") == "HLA-B*14:01"
    assert affinity._normalise_allele("HLA-A*02:01") == "HLA-A*02:01"


def test_embedding_cache_path_is_deterministic_and_content_addressed():
    a = embeddings._cache_path("esm2_t12", "mean", ["AAA", "BBB"])
    b = embeddings._cache_path("esm2_t12", "mean", ["AAA", "BBB"])
    c = embeddings._cache_path("esm2_t12", "mean", ["AAA", "CCC"])
    d = embeddings._cache_path("esm2_t12", "max", ["AAA", "BBB"])
    assert a == b
    assert a != c, "different sequences must not share a cache entry"
    assert a != d, "different pooling must not share a cache entry"


def test_unknown_model_and_pooling_are_rejected_before_any_download():
    with pytest.raises(ValueError, match="unknown model"):
        embeddings.embed_sequences(["AAA"], model="not_a_model")
    with pytest.raises(ValueError, match="unknown pooling"):
        embeddings.embed_sequences(["AAA"], model="esm2_t6", pooling="not_a_pooling")


def test_feature_sources_do_not_import_torch_at_module_scope():
    """The segfault guard: importing these modules must not drag in a DL backend."""
    import sys

    for module in (likelihood, structure, affinity):
        assert "torch" not in [n for n in dir(module)], f"{module.__name__} exposes torch"
    # embeddings/structure may legitimately have torch in sys.modules if another test loaded it;
    # what matters is that the module body itself does not import it.
    src = (structure.__file__, likelihood.__file__, affinity.__file__)
    for path in src:
        with open(path) as fh:
            body = fh.read()
        header = body.split("def ", 1)[0]
        assert "import torch" not in header, f"{path} imports torch at module scope"
    assert sys is not None


@pytest.mark.skipif(not structure.TEMPLATE.exists(), reason="template not built")
def test_template_matches_the_dataset_schema():
    tmpl = structure._template()
    assert len(tmpl["pep_seq"]) == 9, "template peptide must be a 9-mer like the dataset"
    assert len(tmpl["hla_seq"]) >= 182
    assert tmpl["hla_coords"].shape[1:] == (4, 3), "N, CA, C, O backbone coordinates"
    assert len(tmpl["hla_seq"]) == len(tmpl["hla_coords"])


@pytest.mark.skipif(not structure.TEMPLATE.exists(), reason="template not built")
def test_row_encoding_covers_all_three_chains():
    tmpl = structure._template()
    df = synthetic(n=4, seed=3)
    df["hla_seq"] = tmpl["hla_seq"][:182]
    rows = structure._encode_rows(df, tmpl)
    expected = len(tmpl["hla_seq"]) + len(tmpl["b2m_seq"]) + 9
    assert rows.shape == (len(df), expected)
    assert rows.max() < len(structure.ALPHABET)


def test_row_encoding_rejects_a_wrong_length_groove():
    tmpl = structure._template() if structure.TEMPLATE.exists() else None
    if tmpl is None:
        pytest.skip("template not built")
    df = synthetic(n=2, seed=4)
    df["hla_seq"] = "A" * 100
    with pytest.raises(ValueError, match="182-residue groove"):
        structure._encode_rows(df, tmpl)


def test_rank_targets_are_derived_from_training_rows_only():
    """A rank depends on every other row, so deriving it over the full dataset would leak."""
    from hlaforge.experiment import _fit_target

    df = synthetic(n=60, seed=9)          # drops duplicate pairs, so len(df) <= 60
    y = df["y"].to_numpy()
    alleles = df["allele"].to_numpy()
    n_train = int(len(df) * 0.7)
    train_idx = np.arange(n_train)

    out = _fit_target(y, train_idx, alleles, "rank")
    assert len(out) == len(train_idx), "must return targets for the training rows only"
    assert out.min() > 0 and out.max() <= 1.0
    # the transform must not depend on rows outside the training fold
    y_tampered = y.copy()
    y_tampered[n_train:] = 1e6
    assert np.allclose(out, _fit_target(y_tampered, train_idx, alleles, "rank"))
