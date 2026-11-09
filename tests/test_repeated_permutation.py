"""Check fixed-control isolation and complete, prespecified summaries."""

import pytest

from competition import fit_sequence_model
from repeated_permutation import SEEDS, condition_by_name, conditions, summarize


def test_fixed_epoch_never_constructs_genuine_stop_data():
    class Model:
        def fit(self, training, **kwargs):
            assert training == "permuted"
            assert kwargs == {"epochs": 1, "shuffle": False, "verbose": 2}
            return "fitted"

    def stop_data():
        raise AssertionError("Genuine stopping labels must not be accessed")

    assert fit_sequence_model(Model(), "permuted", stop_data, {}, 1) == "fitted"
    with pytest.raises(ValueError):
        fit_sequence_model(Model(), "permuted", stop_data, {}, 2)


def test_condition_lookup_and_unknown_condition_fails_closed():
    assert condition_by_name("headline_original") == (None, None)
    assert condition_by_name("headline_one_epoch") == (None, 1)
    assert condition_by_name("permutation_1055_original") == (1055, None)
    assert condition_by_name("permutation_1060_one_epoch") == (1060, 1)
    with pytest.raises(ValueError, match="Unknown condition"):
        condition_by_name("permutation_9999_original")


def test_plan_conditions_and_incomplete_results_fail_closed():
    plan = conditions()
    assert len(plan) == 22
    assert [seed for _, seed, epochs in plan if seed is not None and epochs is None] == list(SEEDS)
    assert [seed for _, seed, epochs in plan if seed is not None and epochs == 1] == list(SEEDS)
    with pytest.raises(ValueError, match="incomplete"):
        summarize({})
