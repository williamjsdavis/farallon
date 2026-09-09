"""Restart diversity and scene invariants, without target fitting or model calls."""
from copy import deepcopy
import json

import numpy as np
import pytest

from geology.history import HistoryError, PARAM_SPECS, parse_history, validate_history
from geology.restarts import restart_history


BASELINE = parse_history("strata(levels=[-1.2,-1.05,-.92,-.78,-.5,-.2],units=[1,2,3,4,5,6,7])\nerode(level=0)")
BEST = parse_history("strata(levels=[-1.2,-1.05,-.92,-.78,-.5,-.2],units=[1,2,3,4,5,6,7])\nanticline(x=3,y=1.5,uplift=1.5,plunge_nw=0,plunge_se=0)\nerode(level=0)")
BOUNDS = {"xmin": 0., "xmax": 6., "ymin": 0., "ymax": 4.}


def assert_scene(history):
    assert validate_history(history) == history
    assert len(history["events"]) <= 8
    assert history["events"][0]["units"] == list(range(1, 8))
    assert np.all(np.diff(history["events"][0]["levels"]) > 0)
    assert history["events"][-1] == {"type": "erode", "level": 0.}
    for event in history["events"][1:]:
        for field, spec in PARAM_SPECS[event["type"]].items():
            assert spec["min"] <= event[field] <= spec["max"]


@pytest.mark.parametrize("index", [1, 2, 3, 6])
def test_repeatable_fresh_outputs_without_input_mutation(index):
    best, baseline, bounds = deepcopy(BEST), deepcopy(BASELINE), deepcopy(BOUNDS)
    original = deepcopy((best, baseline, bounds))
    first, reason = restart_history(best, baseline, bounds, index, 42)
    second, second_reason = restart_history(best, baseline, bounds, index, 42)
    assert first == second and reason == second_reason
    assert first is not second and first["events"][0] is not second["events"][0]
    assert (best, baseline, bounds) == original
    assert_scene(first)
    json.dumps(first, allow_nan=False)


def test_odd_restarts_are_undeformed_even_if_baseline_contains_a_fold():
    candidate, reason = restart_history(BEST, BEST, BOUNDS, 1, 7)
    assert [event["type"] for event in candidate["events"]] == ["strata", "erode"]
    assert candidate["events"][0]["levels"] != BEST["events"][0]["levels"]
    assert "undeformed" in reason


def test_seed_and_index_diversity_is_spatially_meaningful():
    candidates = [restart_history(BEST, BASELINE, BOUNDS, 2, seed)[0] for seed in range(12)]
    assert len({json.dumps(candidate, sort_keys=True) for candidate in candidates}) == 12
    folds = [candidate["events"][1] for candidate in candidates]
    assert np.ptp([fold["x"] for fold in folds]) > BOUNDS["xmax"] * .25
    assert np.ptp([fold["y"] for fold in folds]) > BOUNDS["ymax"] * .25
    assert np.ptp([fold["azimuth"] for fold in folds]) > 30
    assert any(fold["plunge_se"] > 0 for fold in folds)
    assert restart_history(BEST, BASELINE, BOUNDS, 2, 42)[0] != restart_history(BEST, BASELINE, BOUNDS, 4, 42)[0]


def test_multiple_events_stay_bounded_and_an_extra_event_can_be_removed():
    best = deepcopy(BEST)
    best["events"][1:-1] = [
        {"type": "anticline", "x": 100, "y": -100, "uplift": 10, "dip_ne": 85, "dip_sw": 0},
        {"type": "anticline", "hinge": 5, "nw_length": 100, "se_length": 0},
        {"type": "tilt", "angle": 85},
        {"type": "fault", "slip": -10, "dip": 90},
        {"type": "tilt", "angle": -85},
        {"type": "fault", "slip": 10, "dip": 1},
    ]
    best = validate_history(best)
    for index in (2, 4, 6, 8, 12):
        candidate, reason = restart_history(best, BASELINE, BOUNDS, index, 11)
        assert_scene(candidate)
        assert len(candidate["events"]) == (7 if index % 6 == 0 else 8)
        assert ("removed one extra" in reason) == (index % 6 == 0)
        for event in candidate["events"][1:-1]:
            assert -1.5 <= event["x"] <= 7.5
            assert -1 <= event["y"] <= 5


@pytest.mark.parametrize("levels", [[-100, -99, -50, 0, 99, 100], [99.9995, 99.9996, 99.9997, 99.9998, 99.9999, 100]])
def test_extreme_contacts_keep_positive_thicknesses_without_crossing_bounds(levels):
    baseline = deepcopy(BASELINE)
    baseline["events"][0]["levels"] = levels
    for seed in range(10):
        candidate, _ = restart_history(BEST, baseline, BOUNDS, 1, seed)
        assert_scene(candidate)


@pytest.mark.parametrize("best", [{}, None, BASELINE])
def test_unusable_or_undeformed_best_falls_back_to_baseline(best):
    candidate, reason = restart_history(best, BASELINE, BOUNDS, 2, 12)
    assert [event["type"] for event in candidate["events"]] == ["strata", "erode"]
    assert "baseline fallback" in reason
    assert_scene(candidate)


@pytest.mark.parametrize("bounds", [None, {}, {**BOUNDS, "xmin": 6}, {**BOUNDS, "ymax": 0}, {**BOUNDS, "xmax": np.inf}, {**BOUNDS, "ymin": np.nan}, {**BOUNDS, "xmin": True}, {**BOUNDS, "xmax": 101}])
def test_invalid_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        restart_history(BEST, BASELINE, bounds, 1, 0)


@pytest.mark.parametrize("index,seed", [(0, 0), (-1, 0), (True, 0), (1.5, 0), (1, -1), (1, True), (1, 1.5)])
def test_invalid_restart_index_or_seed_is_rejected(index, seed):
    with pytest.raises(ValueError):
        restart_history(BEST, BASELINE, BOUNDS, index, seed)


def test_invalid_baseline_cannot_change_scene_identities_or_terrain():
    changed_ids, changed_erosion = deepcopy(BASELINE), deepcopy(BASELINE)
    changed_ids["events"][0]["units"][0] = 9
    changed_erosion["events"][-1]["level"] = 1
    for baseline in (changed_ids, changed_erosion):
        with pytest.raises(HistoryError):
            restart_history(BEST, baseline, BOUNDS, 1, 0)
