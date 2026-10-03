"""Every queued experiment must be a valid config before anyone spends compute on it."""

import pytest

from hlaforge.experiment import INTERACTIONS, ExperimentConfig
from hlaforge.models import build_model
from hlaforge.queues import QUEUES, all_jobs


def test_every_family_builds_valid_configs():
    for family, builder in QUEUES.items():
        jobs = builder()
        assert jobs, f"{family} produced no jobs"
        for name, spec in jobs:
            cfg = ExperimentConfig(name=name, notes=family, **spec)
            assert cfg.interaction in INTERACTIONS
            assert cfg.target in ("log1p", "raw", "binary", "rank", "rank_allele")
            assert "kind" in cfg.model


def test_job_names_are_unique():
    names = [name for _, name, _ in all_jobs()]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate experiment names would overwrite results: {duplicates}"


def test_every_model_kind_in_the_queues_is_constructible():
    kinds = {spec["model"]["kind"] for _, _, spec in all_jobs() if "model" in spec}
    for kind in kinds:
        build_model(kind, None, 0)


@pytest.mark.parametrize("family", sorted(QUEUES))
def test_family_has_a_reference_first(family):
    """The first job in a family is its anchor; the rest are one-slot swaps off it."""
    first = QUEUES[family]()[0][0]
    assert "reference" in first.lower(), f"{family} should start with its reference, got {first}"
