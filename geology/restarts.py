"""Deterministic broad starting points for independent geological branches.

No target labels, fitted examples or recorded histories enter this module.
Restart proposals are deliberately unscored; the controller owns evaluation
and must retain its overall best independently of a restarted branch.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .history import HistoryError, PARAM_SPECS, validate_history


_DEFORMATION = {"anticline", "syncline", "tilt", "fault"}


def _scene_history(history: Any) -> dict:
    result = validate_history(history)
    events = result["events"]
    if (events[0]["units"] != list(range(1, 8)) or len(events) > 8
            or events[-1] != {"type": "erode", "level": 0.0}
            or any(event["type"] not in _DEFORMATION for event in events[1:-1])):
        raise HistoryError("Restarts require seven original units, at most eight events, and final erode(level=0)")
    return result


def _bounds(bounds: Any) -> tuple[float, float, float, float]:
    if not isinstance(bounds, dict):
        raise ValueError("bounds must contain xmin, xmax, ymin and ymax")
    values = []
    for key in ("xmin", "xmax", "ymin", "ymax"):
        value = bounds.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
            raise ValueError("bounds must contain finite numeric xmin, xmax, ymin and ymax")
        values.append(float(value))
    xmin, xmax, ymin, ymax = values
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("bounds must have positive x and y extents")
    for lower, upper, field in ((xmin, xmax, "x"), (ymin, ymax, "y")):
        spec = PARAM_SPECS["anticline"][field]
        if lower < spec["min"] or upper > spec["max"]:
            raise ValueError("bounds must lie within the DSL's supported coordinate range")
    return xmin, xmax, ymin, ymax


def _perturb_strata(event: dict, rng: np.random.Generator, span: float) -> None:
    levels = np.asarray(event["levels"], dtype=np.float64)
    # Perturb positive intervals, then rebuild the contacts. Independent
    # clipping/sorting of contact heights could collapse thin units.
    gaps = np.maximum(np.diff(levels) * rng.uniform(.65, 1.55)
                      * np.exp(rng.normal(0, .35, len(levels) - 1)), 1e-4)
    if gaps.sum() > 180:
        gaps *= 180 / gaps.sum()
    relative = np.r_[0., np.cumsum(gaps)]
    relative -= relative[-1] / 2
    scale = max(.1, .10 * span, .25 * float(np.ptp(levels)))
    center = float((levels[0] + levels[-1]) / 2 + rng.normal(0, scale))
    center = float(np.clip(center, -100 - relative[0], 100 - relative[-1]))
    event["levels"] = (center + relative).tolist()


def _perturb_event(event: dict, rng: np.random.Generator, box: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = box
    width, height = xmax - xmin, ymax - ymin
    span = max(width, height)
    for field, spec in PARAM_SPECS[event["type"]].items():
        value = event[field]
        lower, upper = spec["min"], spec["max"]
        if field in {"x", "y"}:
            a, b, extent = (xmin, xmax, width) if field == "x" else (ymin, ymax, height)
            lower, upper = max(lower, a - .25 * extent), min(upper, b + .25 * extent)
            value = (rng.uniform(lower, upper) if rng.random() < .25
                     else value + rng.normal(0, .25 * extent))
        elif field == "azimuth":
            value = (value + rng.uniform(-65, 65)) % 360
        elif field == "uplift":
            value = value * rng.uniform(.55, 1.65) + rng.normal(0, .12 * span)
        elif field.startswith("dip") or field == "angle":
            value += rng.normal(0, 18)
        elif field.startswith("plunge"):
            value = rng.uniform(0, 35) if rng.random() < .35 else value + rng.normal(0, 12)
        elif field == "hinge":
            value = value * np.exp(rng.normal(0, .5)) + .005 * span
        elif field.endswith("_length"):
            value = value * np.exp(rng.normal(0, .5)) + rng.normal(0, .15 * span)
        elif field == "slip":
            value += rng.normal(0, .20 * span)
        else:  # Vertical pivot/reference positions of tilt and fault events.
            value += rng.normal(0, .10 * span)
        event[field] = float(np.clip(value, lower, upper))


def restart_history(
    best_history: dict,
    baseline_history: dict,
    bounds: dict,
    restart_index: int,
    seed: int,
) -> tuple[dict, str]:
    """Return a fresh valid history and a plain-language restart description.

    Odd indices start with undeformed baseline strata, varying thicknesses and
    vertical position. Even indices broadly vary the overall best's existing
    deformation and strata. Every third even restart removes one deformation
    event when several exist. Invalid/undeformed best histories fall back to
    baseline strata; the supplied baseline itself must satisfy scene invariants.

    Source histories are never mutated. All distances and bounds use km;
    ``bounds`` supplies xmin/xmax/ymin/ymax. Contact intervals remain positive,
    unit IDs remain 1..7 and the fixed terrain offset remains erode(level=0).
    """
    for name, value, minimum in (("restart_index", restart_index, 1), ("seed", seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    box = _bounds(bounds)
    baseline = _scene_history(baseline_history)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(restart_index)]))
    fresh = bool(restart_index % 2)
    fallback = False
    candidate = baseline
    if not fresh:
        try:
            candidate = _scene_history(best_history)
            fresh = not any(event["type"] in _DEFORMATION for event in candidate["events"])
        except (HistoryError, TypeError, ValueError):
            fresh = True
        fallback = fresh
    if fresh:
        candidate = baseline
        candidate["events"] = [candidate["events"][0], candidate["events"][-1]]
        description = "Fresh undeformed strata with new thicknesses and vertical position"
        if fallback:
            description += " (baseline fallback: no usable deformed best history)"
    else:
        description = "Broad perturbation of the overall best history's geometry and strata"
        deformation = [i for i, event in enumerate(candidate["events"]) if event["type"] in _DEFORMATION]
        if restart_index % 6 == 0 and len(deformation) > 1:
            removed = candidate["events"].pop(int(rng.choice(deformation)))
            description += f"; removed one extra {removed['type']} event"
        for event in candidate["events"][1:-1]:
            _perturb_event(event, rng, box)
    _perturb_strata(candidate["events"][0], rng, max(box[1] - box[0], box[3] - box[2]))
    return _scene_history(candidate), description + "."
