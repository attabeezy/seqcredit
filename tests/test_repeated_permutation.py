"""Check fixed-control isolation and complete, prespecified summaries."""

import pytest

from competition import fit_sequence_model
from repeated_permutation import (SEEDS, condition_by_name, conditions, parse_only,
                                 select_conditions, summarize)


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


def test_batch_selection_preserves_plan_order_and_fails_closed():
    assert [item[0] for item in select_conditions(parse_only("headline_original,permutation_1051_one_epoch"))] == [
        "headline_original", "permutation_1051_one_epoch"]
    assert select_conditions(None) == conditions()
    with pytest.raises(ValueError, match="non-empty"):
        select_conditions([])
    with pytest.raises(ValueError, match="unique"):
        select_conditions(["headline_original", "headline_original"])
    with pytest.raises(ValueError, match="Unknown condition"):
        select_conditions(["permutation_9999_original"])


def test_plan_conditions_and_incomplete_results_fail_closed():
    plan = conditions()
    assert len(plan) == 22
    assert [seed for _, seed, epochs in plan if seed is not None and epochs is None] == list(SEEDS)
    assert [seed for _, seed, epochs in plan if seed is not None and epochs == 1] == list(SEEDS)
    with pytest.raises(ValueError, match="incomplete"):
        summarize({})
