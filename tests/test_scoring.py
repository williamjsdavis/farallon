from copy import deepcopy

import numpy as np
import pytest

from geology.scoring import MapScorer, mismatch_rgba, prepare_boundary_mask, score_map


def split_map(size=24, split=12):
    labels = np.ones((size, size), dtype=np.uint8)
    labels[:, split:] = 2
    return labels


def test_identical_map_and_missing_class():
    target = split_map()
    perfect = score_map(target, target)
    assert perfect["miou"] == perfect["accuracy"] == perfect["score"] == 1.0
    assert perfect["boundary_error_px"] == 0
    missing = score_map(np.ones_like(target), target)
    assert missing["per_unit"] == {"1": 0.5, "2": 0.0}
    assert missing["miou"] == 0.25
    assert np.isfinite(missing["boundary_error_px"])
    assert missing["boundary_error_px"] > 0


def test_predicted_extra_class_is_not_dropped():
    target = split_map()
    prediction = target.copy()
    prediction[:, :3] = 7
    metrics = score_map(prediction, target)
    assert metrics["per_unit"]["7"] == 0
    assert len(metrics["per_unit"]) == 3
    assert metrics["miou"] < metrics["accuracy"]


def test_shifted_contact_has_nonzero_distance_and_lower_score():
    target = split_map()
    shifted = score_map(split_map(split=15), target)
    assert 0 < shifted["boundary_error_px"] < 4
    assert shifted["score"] < 1
    assert shifted["accuracy"] == 21 / 24


def test_geometrically_identical_wrong_contact_pair_is_penalized():
    target = split_map()
    prediction = target.copy()
    prediction[prediction == 2] = 3
    metrics = score_map(prediction, target)
    assert metrics["boundary_error_px"] > 20


def test_masked_values_and_mask_perimeter_do_not_create_contacts():
    target = np.ones((20, 20), dtype=np.uint8)
    prediction = target.copy()
    mask = np.ones_like(target, dtype=bool)
    mask[6:14, 6:14] = False
    prediction[~mask] = 9
    target[~mask] = 8
    result = score_map(prediction, target, mask)
    assert result["miou"] == 1
    assert result["target_contact_pixels"] == result["prediction_contact_pixels"] == 0
    assert result["boundary_error_px"] == 0
    assert np.all(mismatch_rgba(prediction, target, mask)[~mask] == 0)


def test_narrow_printed_contact_gap_can_be_recovered_without_changing_iou_mask():
    complete = split_map()
    target = complete.copy()
    mask = np.ones_like(target, dtype=bool)
    mask[:, 11:13] = False
    target[~mask] = 0
    reasons = np.where(mask, 0, 3).astype(np.uint8)
    boundary_mask = prepare_boundary_mask(mask, reasons)
    strict = score_map(complete, target, mask)
    recovered = score_map(complete, target, mask, boundary_mask=boundary_mask)
    assert strict["target_contact_pixels"] == 0
    assert recovered["target_contact_pixels"] > 0
    assert recovered["miou"] == 1
    assert recovered["boundary_error_px"] == 0
    assert recovered["evaluated_pixels"] == int(mask.sum())


def test_cover_gap_is_never_bridged():
    mask = np.ones((24, 24), dtype=bool)
    mask[:, 11:13] = False
    reasons = np.where(mask, 0, 2).astype(np.uint8)
    boundary_mask = prepare_boundary_mask(mask, reasons)
    assert not boundary_mask[:, 10:14].any()


def test_empty_mask_and_misaligned_maps_rejected():
    target = split_map()
    with pytest.raises(ValueError, match="at least one"):
        score_map(target, target, np.zeros_like(target, dtype=bool))
    with pytest.raises(ValueError, match="same shape"):
        MapScorer(target)(target[:-1])


def test_refinement_improves_known_position_error_and_preserves_structure():
    from geology.engine import render_map
    from geology.history import validate_history
    from geology.search import refine_history

    truth = validate_history({"events": [
        {"type": "strata", "levels": [-0.8, -0.5, -0.2], "units": [1, 2, 3, 4]},
        {"type": "anticline", "x": 2.0, "y": 2.0, "azimuth": 135.0, "uplift": 1.0,
         "dip_ne": 35.0, "dip_sw": 35.0, "nw_length": 1.0, "se_length": 1.0,
         "plunge_nw": 0.0, "plunge_se": 0.0},
        {"type": "erode", "level": 0.0},
    ]})
    xs, ys = np.linspace(0, 4, 48), np.linspace(4, 0, 48)
    target = render_map(truth, xs, ys)  # Also warm the renderer before the deadline.
    initial = deepcopy(truth)
    initial["events"][1]["x"] = 1.5
    before = score_map(render_map(initial, xs, ys), target)
    fitted, after, evaluations, elapsed_ms = refine_history(initial, target, xs, ys, None, budget=40, seconds=2, allowed_fields=["anticline.x"])
    assert after["score"] > before["score"]
    assert evaluations <= 40
    assert elapsed_ms < 2500
    assert initial["events"][1]["x"] == 1.5
    for key in ("plunge_nw", "plunge_se", "dip_ne", "dip_sw", "nw_length", "se_length"):
        assert fitted["events"][1][key] == initial["events"][1][key]


def test_default_search_parameters_preserve_zeros_equal_limbs_and_material_ids():
    from geology.history import PARAM_SPECS, validate_history
    from geology.search import _parameters

    history = validate_history({"events": [
        {"type": "strata", "levels": [-1, -0.5], "units": [1, 2, 3]},
        {"type": "anticline", "dip_ne": 30, "dip_sw": 30, "nw_length": 1, "se_length": 1, "plunge_nw": 0, "plunge_se": 0},
        {"type": "intrusion", "unit": 4},
        {"type": "erode"},
    ]})
    parameters = _parameters(history, np.linspace(0, 4, 20), np.linspace(4, 0, 20), None, PARAM_SPECS)
    fields = [parameter.fields for parameter in parameters]
    assert ("dip_ne", "dip_sw") in fields
    assert ("nw_length", "se_length") in fields
    assert not any(field in {"plunge_nw", "plunge_se", "unit", "level"} for group in fields for field in group)


def test_coarse_search_returns_full_resolution_metrics_and_obeys_small_budgets():
    from geology.engine import render_map
    from geology.history import validate_history
    from geology.search import refine_history

    truth = validate_history({"events": [
        {"type": "strata", "levels": [-0.8, -0.4], "units": [1, 2, 3]},
        {"type": "anticline", "x": 2, "y": 2, "plunge_nw": 0, "plunge_se": 0},
        {"type": "erode"},
    ]})
    xs, ys = np.linspace(0, 4, 144), np.linspace(4, 0, 144)
    target = render_map(truth, xs, ys)
    initial = deepcopy(truth)
    initial["events"][1]["x"] = 1.7
    baseline = score_map(render_map(initial, xs, ys), target)
    for budget in (1, 2, 12):
        fitted, metrics, evaluations, _ = refine_history(initial, target, xs, ys, None, budget=budget, seconds=2, allowed_fields=["x"])
        measured = score_map(render_map(fitted, xs, ys), target)
        assert metrics == measured
        assert metrics["evaluated_pixels"] == 144 ** 2
        assert metrics["score"] >= baseline["score"]
        assert evaluations <= budget
