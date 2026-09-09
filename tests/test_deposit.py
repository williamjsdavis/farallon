"""Young unconformable deposits are geometry, not a painted surface mask."""
from copy import deepcopy
import json

import numpy as np
import pytest

from geology.cartography import fold_axes
from geology.engine import evaluate_points, render_map, render_volume
from geology.history import HistoryError, PARAM_SPECS, history_to_program, parse_history, validate_history
from geology.restarts import restart_history
from geology.scoring import score_map
from geology.search import _parameters, refine_history


STRATA = "strata(levels=[-1.2,-1.05,-.92,-.78,-.5,-.2],units=[1,2,3,4,5,6,7])"
BOUNDS = {"xmin": -2., "xmax": 2., "ymin": -1.5, "ymax": 1.5}


def history(*events, erode=True):
    return parse_history("\n".join([STRATA, *events, *(["erode(level=0)"] if erode else [])]))


def test_defaults_roundtrip_and_multiple_patches_keep_material_identity():
    model = history("deposit()", "deposit(x=2,curvature_along=1)")
    assert parse_history(history_to_program(model)) == model
    assert [event["unit"] for event in model["events"][1:-1]] == [8, 8]
    assert model["events"][0]["units"] == list(range(1, 8))
    assert model["events"][1]["curvature_along"] == 0
    json.dumps(model, allow_nan=False)


@pytest.mark.parametrize("arguments", [
    "unit=7", "unit=0", "unit=8.5", "unit=255", "unit=True",
    "curvature_cross=-.1", "curvature_along=-1", "curvature_cross=100.1",
    "base=101", "slope=5.1", "azimuth=-1", "x=1e309", "thickness=.2",
])
def test_invalid_parameters_and_bedrock_reuse_are_rejected(arguments):
    with pytest.raises(HistoryError):
        history(f"deposit({arguments})")


def test_intrusive_unit_cannot_be_reused_as_deposit_or_reverse():
    for events in [("intrusion(unit=8)", "deposit(unit=8)"),
                   ("deposit(unit=8)", "intrusion(unit=8)")]:
        with pytest.raises(HistoryError):
            history(*events)
    # A generic history can name another new young material; the scene-level
    # validator, rather than the numerical language, restricts deposits to 8.
    assert history("deposit(unit=9)")["events"][1]["unit"] == 9


def test_flat_deposit_has_real_thickness_and_exact_base_is_young():
    model = history("deposit(base=-.1,curvature_cross=0,curvature_along=0)")
    points = np.array([[0, 0, -.100001], [0, 0, -.1], [0, 0, -.05], [0, 0, .01]])
    assert evaluate_points(model, points).tolist() == [7, 8, 8, 0]
    below = np.array([[0, 0, -.6], [3, -2, -.8], [-1, 5, -1.1]])
    np.testing.assert_array_equal(evaluate_points(model, below), evaluate_points(history(), below))


def test_bowl_ellipse_and_zero_along_curvature_valley_strip():
    bowl = history("deposit(azimuth=0,base=-.4,curvature_cross=.4,curvature_along=.1)")
    points = [[0, 0, 0], [.9, 0, 0], [1.1, 0, 0], [0, 1.9, 0], [0, 2.1, 0]]
    assert evaluate_points(bowl, points).tolist() == [8, 8, 7, 8, 7]
    strip = history("deposit(azimuth=0,base=-.4,curvature_cross=.4,curvature_along=0)")
    assert evaluate_points(strip, [[0, 50, 0], [1.1, 50, 0]]).tolist() == [8, 7]


def test_planar_slope_follows_clockwise_from_north_azimuth():
    north = history("deposit(azimuth=0,base=0,curvature_cross=0,curvature_along=0,slope=.2)")
    east = history("deposit(azimuth=90,base=0,curvature_cross=0,curvature_along=0,slope=.2)")
    points = [[0, -1, -.1], [0, 1, -.1], [-1, 0, -.1], [1, 0, -.1]]
    assert evaluate_points(north, points).tolist() == [8, 7, 7, 7]
    assert evaluate_points(east, points).tolist() == [7, 7, 8, 7]


def test_disjoint_same_unit_deposits_are_a_union_without_painting_the_gap():
    left = "deposit(x=-1,y=0,base=-.2,curvature_cross=1,curvature_along=1)"
    right = "deposit(x=1,y=0,base=-.2,curvature_cross=1,curvature_along=1)"
    points = [[-1, 0, -.1], [1, 0, -.1], [0, 0, -.1], [-1, 0, -.3]]
    model = history(left, right)
    assert evaluate_points(model, points).tolist() == [8, 8, 7, 6]
    np.testing.assert_array_equal(evaluate_points(model, points), evaluate_points(history(right, left), points))


@pytest.mark.parametrize("kind,point", [("anticline", [0, 0, 0]), ("syncline", [0, 0, -.5])])
def test_deposit_is_deformed_only_by_younger_folds(kind, point):
    deposit = "deposit(base=-.3,curvature_cross=0,curvature_along=0)"
    fold = f"{kind}(x=0,y=0,uplift=.6,plunge_nw=0,plunge_se=0)"
    young_cover = evaluate_points(history(fold, deposit), [point])[0]
    folded_cover = evaluate_points(history(deposit, fold), [point])[0]
    if kind == "anticline":
        assert young_cover == 8 and folded_cover != 8
    else:
        assert young_cover != 8 and folded_cover == 8


@pytest.mark.parametrize("event,point", [
    ("fault(azimuth=0,dip=90,slip=.5)", [1, 0, -.4]),
    ("tilt(azimuth=0,angle=30)", [np.sqrt(3) / 2 + .05, 0, -.5 + np.sqrt(3) * .05]),
])
def test_later_fault_and_tilt_move_real_deposit_points(event, point):
    deposit = "deposit(base=0,curvature_cross=0,curvature_along=0)"
    assert evaluate_points(history(deposit, event, erode=False), [point])[0] == 8
    assert evaluate_points(history(event, deposit, erode=False), [point])[0] != 8


def test_map_and_volume_share_the_basal_surface_and_fixed_terrain():
    model = history("deposit(x=.3,y=-.2,azimuth=35,base=-.2,curvature_cross=.5,curvature_along=.2,slope=.05)")
    nx, ny, nz = 9, 7, 11
    xs = -2 + (np.arange(nx) + .5) * 4 / nx
    ys = -1.5 + (np.arange(ny) + .5) * 3 / ny
    # Avoid coincident stratum contacts; those have their own exact-boundary
    # test and a differently rounded coordinate can legitimately cross one.
    zs = -.8 + (np.arange(nz) + .5) * 1.21 / nz
    xx, yy = np.meshgrid(xs, ys)
    terrain = .12 + .05 * xx + .03 * yy
    original_terrain = terrain.copy()
    bare = {**model, "events": model["events"][:-1]}
    surface_points = np.column_stack([xx.ravel(), yy.ravel(), (terrain - 1e-8).ravel()])
    expected_map = evaluate_points(bare, surface_points).reshape(ny, nx)
    np.testing.assert_array_equal(render_map(model, xs, ys[::-1], terrain=terrain[::-1]), expected_map[::-1])
    zz, y3, x3 = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.column_stack([x3.ravel(), y3.ravel(), zz.ravel()])
    expected_volume = evaluate_points(bare, points).reshape(nz, ny, nx)
    expected_volume[zz > terrain[None, :, :]] = 0
    volume = render_volume(model, [-2, 2, -1.5, 1.5, -.8, .41], (nx, ny, nz), terrain=terrain)
    np.testing.assert_array_equal(volume, expected_volume)
    assert {0, 7, 8} <= set(np.unique(volume))
    np.testing.assert_array_equal(terrain, original_terrain)


def test_base_refinement_improves_a_cover_patch_without_changing_ids_or_terrain():
    truth = history("deposit(base=-.3,curvature_cross=1,curvature_along=1)")
    initial = deepcopy(truth)
    initial["events"][1]["base"] = -.15
    xs, ys = np.linspace(-1, 1, 48), np.linspace(1, -1, 48)
    target = render_map(truth, xs, ys, terrain=.1)
    before = score_map(render_map(initial, xs, ys, terrain=.1), target)
    fitted, after, evaluations, _ = refine_history(
        initial, target, xs, ys, None, budget=40, seconds=2,
        allowed_fields=["deposit.base"], terrain=.1)
    assert after["score"] > before["score"] and evaluations <= 40
    assert initial["events"][1]["base"] == -.15
    changed = deepcopy(fitted)
    changed["events"][1]["base"] = initial["events"][1]["base"]
    assert changed == initial


def test_search_preserves_planar_structure_unless_curvature_explicitly_enabled():
    model = history("deposit(curvature_cross=0,curvature_along=0)")
    axes = np.linspace(-2, 2, 12)
    default = _parameters(model, axes, axes, None, PARAM_SPECS)
    fields = {field for p in default for field in p.fields}
    assert {"base", "slope", "x", "y", "azimuth"} <= fields
    assert not {"curvature_cross", "curvature_along", "unit"} & fields
    explicit = _parameters(model, axes, axes, {"deposit.curvature_along"}, PARAM_SPECS)
    assert len(explicit) == 1 and explicit[0].fields == ("curvature_along",)
    assert explicit[0].lower == 0 and explicit[0].upper == 100


def test_restarts_preserve_and_perturb_young_deposits_without_mutating_inputs():
    best = history("deposit(x=.3,y=.1,base=-.2,curvature_cross=1,curvature_along=0)")
    baseline = history()
    original = deepcopy(best)
    first, reason = restart_history(best, baseline, BOUNDS, 2, 42)
    assert first == restart_history(best, baseline, BOUNDS, 2, 42)[0]
    assert first != restart_history(best, baseline, BOUNDS, 2, 43)[0]
    assert best == original and first["events"][1]["unit"] == 8
    assert first["events"][1]["curvature_along"] == 0
    assert first["events"][1] != original["events"][1]
    assert "fallback" not in reason and validate_history(first) == first
    fresh, _ = restart_history(best, baseline, BOUNDS, 1, 42)
    assert [event["type"] for event in fresh["events"]] == ["strata", "erode"]


def test_restarts_can_drop_a_cover_and_keep_remaining_events_valid():
    best = history("anticline()", "deposit(base=-.1)", "deposit(x=1,base=-.2)")
    removed_deposit = False
    for seed in range(12):
        candidate, reason = restart_history(best, history(), BOUNDS, 6, seed)
        assert len(candidate["events"]) == len(best["events"]) - 1
        assert validate_history(candidate) == candidate
        assert candidate["events"][0]["units"] == list(range(1, 8))
        assert candidate["events"][-1] == {"type": "erode", "level": 0.}
        assert all(event["unit"] == 8 for event in candidate["events"] if event["type"] == "deposit")
        removed_deposit |= "removed one extra deposit" in reason
    assert removed_deposit


def test_construction_axis_can_be_inferred_beneath_later_cover():
    fold = "syncline(azimuth=0,uplift=.5,plunge_nw=0,plunge_se=0)"
    cover = "deposit(base=-.2)"
    uncovered = fold_axes(history(fold), BOUNDS)[0]
    covered = fold_axes(history(fold, cover), BOUNDS)[0]
    assert covered["polyline"] == uncovered["polyline"]
    assert covered["kind"] == "syncline" and covered["arrow_direction"] == "toward"
    assert covered["status"] == "visible" and covered["inferred_under_cover"]
    assert "younger cover may conceal" in covered["diagnostic"]
    assert not fold_axes(history(cover, fold), BOUNDS)[0].get("inferred_under_cover", False)
    assert fold_axes(history(fold, cover, "tilt(angle=5)"), BOUNDS)[0]["status"] == "unavailable"
    assert fold_axes(history(cover), BOUNDS) == []
