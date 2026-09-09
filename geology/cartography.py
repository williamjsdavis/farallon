"""Truthful model-event fold axes for cartographic overlays.

USGS/FGDC uses diverging arrows for anticlines and converging arrows for
synclines. These features identify construction axes of actual model events;
they are neither field observations nor extracted hinges of the combined model.
See https://pubs.usgs.gov/of/2015/1041/pdf/ofr2015-1041_sheet8.pdf and
https://ngmdb.usgs.gov/fgdc_gds/geolsymstd/fgdc-geolsym-sec05.pdf.
"""
from __future__ import annotations

import math
from typing import Any

from .history import validate_history


def _box(bounds: Any) -> tuple[float, float, float, float]:
    if not isinstance(bounds, dict):
        raise ValueError("bounds requires xmin, xmax, ymin and ymax")
    values = []
    for key in ("xmin", "xmax", "ymin", "ymax"):
        value = bounds.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("bounds coordinates must be finite numbers")
        try:
            value = float(value)
        except (ValueError, OverflowError):
            raise ValueError("bounds coordinates must be finite numbers") from None
        if not math.isfinite(value):
            raise ValueError("bounds coordinates must be finite numbers")
        values.append(value)
    if values[0] >= values[1] or values[2] >= values[3]:
        raise ValueError("bounds must have positive x and y extents")
    return tuple(values)


def _end_distance(magnitude: float, plunge: float, hinge: float) -> float:
    """Invert the engine's smoothed end ramp at zero fold displacement."""
    if plunge == 0:
        return math.inf
    ramp = magnitude / math.tan(math.radians(plunge))
    return math.sqrt(2 * hinge * ramp) if ramp < hinge / 2 else ramp + hinge / 2


def _segment(event: dict, box: tuple[float, float, float, float]) -> list[list[float]]:
    azimuth = math.radians(event["azimuth"])
    dx, dy = math.sin(azimuth), math.cos(azimuth)
    lower = -event["nw_length"] - _end_distance(event["uplift"], event["plunge_nw"], event["hinge"])
    upper = event["se_length"] + _end_distance(event["uplift"], event["plunge_se"], event["hinge"])
    for origin, direction, minimum, maximum in (
        (event["x"], dx, box[0], box[1]), (event["y"], dy, box[2], box[3]),
    ):
        if abs(direction) < 1e-12:
            if not minimum <= origin <= maximum:
                return []
        else:
            a, b = (minimum - origin) / direction, (maximum - origin) / direction
            lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
    if upper - lower <= 1e-10:
        return []
    # Clamp only roundoff at rectangle boundaries, not the line's geometry.
    return [[min(box[1], max(box[0], event["x"] + t * dx)),
             min(box[3], max(box[2], event["y"] + t * dy))] for t in (lower, upper)]


def fold_axes(history: Any, bounds: dict) -> list[dict]:
    """Return JSON features for active anticline/syncline construction axes.

    Coordinates are world (east, north) kilometres. source_event_index is
    zero-based in the canonical event list. The two-point polyline runs from
    negative to positive local axial coordinate; azimuth is clockwise from
    north. normal points toward the engine's positive transverse coordinate.
    arrow_direction says whether transverse arrows point away from or toward
    the axis. Renderers should draw only features whose status is `visible`.

    Axes end where that event's displacement vanishes, or at map boundaries.
    A later active tilt/fault can rotate or displace an axial surface; without
    its intersection with the fixed DEM there is no truthful projected trace.
    Such events return empty geometry with status `unavailable` and a diagnostic.
    Superposed folds retain their separate *construction* axes; no net syncline
    is inferred between uplift events. A younger deposit retains an older
    construction axis with an inferred-under-cover diagnostic; that axis does
    not assert that the cover is folded or that bedrock is exposed along it.
    In an isolated upright stratigraphy,
    anticlines expose an older core and synclines a younger core at a flat cut,
    when enough stratigraphic levels are exposed. Terrain/overprinting can
    change that outcrop pattern, so labels never assert it as observed evidence.
    """
    canonical = validate_history(history)
    box = _box(bounds)
    events = canonical["events"]
    features = []
    for index, event in enumerate(events):
        kind = event["type"]
        if kind not in {"anticline", "syncline"} or event["uplift"] == 0:
            continue
        azimuth = math.radians(event["azimuth"])
        feature = {
            "kind": kind,
            "polyline": [],
            "source_event_index": index,
            "azimuth": event["azimuth"],
            "normal": [-math.cos(azimuth), math.sin(azimuth)],
            "arrow_direction": "away" if kind == "anticline" else "toward",
            "label": f"{kind.capitalize()} construction axis · event {index + 1}",
            "diagnostic": "Model-event construction axis; not an observed trace or a hinge extracted from the combined model.",
            "status": "visible",
        }
        later = [e["type"] for e in events[index + 1:] if (
            (e["type"] == "tilt" and e["angle"] != 0) or
            (e["type"] == "fault" and e["slip"] != 0))]
        if later:
            feature["status"] = "unavailable"
            feature["diagnostic"] = "Axis omitted: later " + "/".join(dict.fromkeys(later)) + " changes its position; a transformed axial-surface/terrain intersection is required."
        elif event["dip_ne"] == 0 and event["dip_sw"] == 0:
            feature["status"] = "unavailable"
            feature["diagnostic"] = "Axis omitted: both transverse limb dips are zero, so this event has no transverse fold hinge."
        else:
            feature["polyline"] = _segment(event, box)
            if not feature["polyline"]:
                continue  # No nonzero-length axis segment in this viewport.
            if any(e["type"] == "deposit" for e in events[index + 1:]):
                feature["diagnostic"] = "Model fold axis; younger cover may conceal structure. Inferred under cover, not an assertion that the younger deposit is folded or that this trace is observed."
                feature["inferred_under_cover"] = True
        features.append(feature)
    return features
