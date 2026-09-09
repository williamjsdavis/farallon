"""World-coordinate fold marker contracts, independent of any target map."""
from copy import deepcopy
import json

import numpy as np
import pytest

from geology.cartography import fold_axes
from geology.history import parse_history


STRATA = "strata(levels=[-1,0,1],units=[1,2,3,4])"
BOX = {"xmin": -1., "xmax": 1., "ymin": -1., "ymax": 1.}


@pytest.mark.parametrize("azimuth,expected", [
    (0, [[0, -1], [0, 1]]),
    (90, [[-1, 0], [1, 0]]),
    (135, [[-1, 1], [1, -1]]),
    (270, [[1, 0], [-1, 0]]),
])
def test_clockwise_from_north_orientation_and_rectangle_clipping(azimuth, expected):
    program = STRATA + f"\nanticline(azimuth={azimuth},plunge_nw=0,plunge_se=0)\nerode(level=0)"
    feature, = fold_axes(program, BOX)
    np.testing.assert_allclose(feature["polyline"], expected, atol=1e-12)
    direction = np.diff(np.asarray(feature["polyline"]), axis=0)[0]
    assert np.dot(direction, feature["normal"]) == pytest.approx(0, abs=1e-12)
    assert feature["kind"] == "anticline" and feature["arrow_direction"] == "away"
    assert feature["source_event_index"] == 1
    assert feature["status"] == "visible"
    assert "construction axis" in feature["label"]
    json.dumps(feature, allow_nan=False)


def test_anticline_only_histories_never_invent_synclines_between_events():
    history = parse_history(STRATA + "\nanticline(x=-.4,azimuth=0)\nanticline(x=.4,azimuth=0)")
    original = deepcopy(history)
    features = fold_axes(history, BOX)
    assert [feature["kind"] for feature in features] == ["anticline", "anticline"]
    assert [feature["source_event_index"] for feature in features] == [1, 2]
    assert all("combined model" in feature["diagnostic"] for feature in features)
    assert history == original


def test_actual_syncline_uses_converging_arrows_on_its_own_axis():
    features = fold_axes(STRATA + "\nanticline(x=-.4,azimuth=0)\nsyncline(x=.4,azimuth=0)", BOX)
    assert [feature["kind"] for feature in features] == ["anticline", "syncline"]
    assert [feature["arrow_direction"] for feature in features] == ["away", "toward"]
    assert np.allclose(np.asarray(features[1]["polyline"])[:, 0], .4)


@pytest.mark.parametrize("later", ["tilt(angle=10)", "fault(slip=.2)"])
def test_later_deformation_reports_unavailable_instead_of_stale_axis(later):
    feature, = fold_axes(STRATA + "\nanticline()\n" + later + "\nerode(level=0)", BOX)
    assert feature["status"] == "unavailable"
    assert feature["polyline"] == []
    assert later.split("(")[0] in feature["diagnostic"]
    assert "intersection" in feature["diagnostic"]


def test_earlier_deformation_and_later_identity_events_do_not_hide_new_axes():
    features = fold_axes(STRATA + "\ntilt(angle=10)\nsyncline()\nfault(slip=0)\ntilt(angle=0)\nerode(level=0)", BOX)
    assert len(features) == 1 and features[0]["status"] == "visible"
    assert features[0]["source_event_index"] == 2


def test_finite_fold_noses_end_at_zero_displacement():
    box = {"xmin": -20., "xmax": 20., "ymin": -20., "ymax": 20.}
    feature, = fold_axes(STRATA + "\nanticline(azimuth=0,uplift=1,hinge=.2,nw_length=2,se_length=3,plunge_nw=45,plunge_se=0)", box)
    np.testing.assert_allclose(feature["polyline"], [[0, -3.1], [0, 20]], atol=1e-12)
    feature, = fold_axes(STRATA + "\nsyncline(azimuth=0,uplift=.01,hinge=.2,nw_length=0,se_length=0,plunge_nw=45,plunge_se=45)", box)
    distance = np.sqrt(.004)
    np.testing.assert_allclose(feature["polyline"], [[0, -distance], [0, distance]], atol=1e-12)


def test_inactive_offscreen_and_transversely_flat_events_are_not_false_markers():
    assert fold_axes(STRATA, BOX) == []
    assert fold_axes(STRATA + "\nsyncline(uplift=0)", BOX) == []
    assert fold_axes(STRATA + "\nanticline(x=2,azimuth=0)", BOX) == []
    feature, = fold_axes(STRATA + "\nanticline(dip_ne=0,dip_sw=0)", BOX)
    assert feature["status"] == "unavailable" and feature["polyline"] == []


@pytest.mark.parametrize("bounds", [None, {}, {**BOX, "xmax": -1}, {**BOX, "ymin": 1}, {**BOX, "xmin": np.nan}, {**BOX, "ymax": float("inf")}, {**BOX, "xmax": True}])
def test_bad_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        fold_axes(STRATA + "\nanticline()", bounds)
