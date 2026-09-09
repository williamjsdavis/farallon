"""Bounded numerical refinement within a proposed geological history.

The model chooses event structure. This search changes existing numerical
parameters, preserves inactive plunges and symmetric limbs by default, and
retains the original history unless a full-resolution measurement improves.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Iterable

import numpy as np

from .scoring import MapScorer


@dataclass(frozen=True)
class _Parameter:
    event: int
    fields: tuple[str, ...]
    step: float
    lower: float
    upper: float
    level: int | None = None


_INACTIVE_ZERO = {"plunge_nw", "plunge_se", "uplift", "amplitude", "throw", "slip", "angle",
                  "curvature_cross", "curvature_along"}


def _enabled(allowed: set[str] | None, index: int, kind: str, field: str) -> bool:
    if allowed is None:
        return True
    return any(key in allowed for key in (field, kind, f"{kind}.{field}", f"events.{index}.{field}", f"{index}.{field}"))


def _parameters(history: dict, xs: np.ndarray, ys: np.ndarray, allowed: set[str] | None, specs: dict) -> list[_Parameter]:
    width = max(float(np.ptp(xs)), 0.01)
    height = max(float(np.ptp(ys)), 0.01)
    span = max(width, height)
    parameters: list[_Parameter] = []
    for index, event in enumerate(history["events"]):
        kind = event["type"]
        if kind == "strata":
            levels = event.get("levels", [])
            if levels and _enabled(allowed, index, kind, "levels"):
                step = max(0.01, min(span * 0.025, max(float(np.ptp(levels)), 0.2) * 0.06))
                parameters.append(_Parameter(index, ("levels",), step, -span * 3, span * 3, -1))
                parameters.extend(_Parameter(index, ("levels",), step * 0.6, -span * 3, span * 3, level) for level in range(len(levels)))
            continue
        # The observation surface is fixed; local fitting may not move erosion.
        if kind == "erode":
            continue
        event_specs = specs.get(kind, {})
        skip: set[str] = set()
        for field, value in event.items():
            if field in skip or field == "unit" or isinstance(value, bool) or not isinstance(value, (float, int)):
                continue
            if not _enabled(allowed, index, kind, field):
                continue
            if allowed is None and field in _INACTIVE_ZERO and abs(value) < 1e-12:
                continue
            # Do not optimize IDs, event indexes, or other incidental numbers.
            if field not in event_specs and field not in {"x", "y", "azimuth", "uplift", "dip_ne", "dip_sw", "hinge", "nw_length", "se_length", "plunge_nw", "plunge_se"}:
                continue
            config = event_specs.get(field, {})
            lower, upper = float(config.get("min", -math.inf)), float(config.get("max", math.inf))
            if field == "x":
                step, lower, upper = width * 0.06, max(lower, float(np.min(xs)) - width * 0.5), min(upper, float(np.max(xs)) + width * 0.5)
            elif field == "y":
                step, lower, upper = height * 0.06, max(lower, float(np.min(ys)) - height * 0.5), min(upper, float(np.max(ys)) + height * 0.5)
            elif field == "azimuth":
                step = 5.0
            elif field.startswith("dip"):
                step = 5.0
            elif field.startswith("plunge"):
                step = 3.5
            elif field in {"nw_length", "se_length"}:
                step = max(0.06 * span, abs(value) * 0.15)
            elif field == "hinge":
                step = max(0.02, abs(value) * 0.25)
            elif field in {"uplift", "amplitude"}:
                step = max(0.03, abs(value) * 0.12)
            elif kind == "deposit" and field == "base":
                step = max(0.01, min(0.1, 0.02 * span))
            elif kind == "deposit" and field.startswith("curvature_"):
                # Curvature has units km^-1; scale the step to the viewport.
                step = max(0.02, abs(value) * 0.25, 0.20 / span)
            elif kind == "deposit" and field == "slope":
                step = max(0.02, abs(value) * 0.15)
            else:
                step = max(float(config.get("step", 0.0)), abs(value) * 0.1, 0.01)
            fields = (field,)
            if allowed is None:
                partner = {"dip_ne": "dip_sw", "dip_sw": "dip_ne", "nw_length": "se_length", "se_length": "nw_length"}.get(field)
                if partner is not None and math.isclose(float(event.get(partner, math.inf)), value, abs_tol=1e-10):
                    fields = (field, partner)
                    skip.add(partner)
                    partner_spec = event_specs.get(partner, {})
                    lower = max(lower, float(partner_spec.get("min", -math.inf)))
                    upper = min(upper, float(partner_spec.get("max", math.inf)))
            if lower < upper:
                parameters.append(_Parameter(index, fields, step, lower, upper))
    return parameters


def _perturb(history: dict, parameter: _Parameter, delta: float) -> dict | None:
    candidate = deepcopy(history)
    event = candidate["events"][parameter.event]
    if parameter.level is not None:
        levels = np.asarray(event["levels"], dtype=float)
        if parameter.level == -1:
            levels += delta
        else:
            level = parameter.level
            lower = levels[level - 1] + 1e-4 if level > 0 else parameter.lower
            upper = levels[level + 1] - 1e-4 if level + 1 < levels.size else parameter.upper
            levels[level] = np.clip(levels[level] + delta, lower, upper)
        if np.any(np.diff(levels) <= 0) or levels.min() < parameter.lower or levels.max() > parameter.upper:
            return None
        event["levels"] = levels.tolist()
    else:
        for field in parameter.fields:
            event[field] = float(np.clip(event[field] + delta, parameter.lower, parameter.upper))
    return None if candidate == history else candidate


def refine_history(
    history: dict,
    target: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    mask: np.ndarray | None,
    budget: int = 128,
    seconds: float = 2.0,
    seed: int = 0,
    allowed_fields: Iterable[str] | None = None,
    terrain: np.ndarray | float | None = None,
    boundary_mask: np.ndarray | None = None,
) -> tuple[dict, dict, int, float]:
    """Return ``(history, full_map_metrics, evaluations, elapsed_ms)``.

    At most ``budget`` maps are rendered, including the initial map and final
    checks. Raster search uses at most 128 samples per axis. Up to three best
    coarse candidates are verified on the original grid. The deadline is
    checked between evaluations; an individual render cannot be interrupted.

    ``allowed_fields`` nominates fields by ``plunge_nw``, ``anticline.x``, or
    ``events.1.x``. An explicit nomination opts out of the default structural
    symmetry/zero preservation for those fields. Erosion and event existence
    always remain fixed. A zero deposit curvature stays zero by default,
    retaining the proposed planar/valley-strip geometry until explicitly freed.
    """
    # Delay the engine import so scoring remains usable independently.
    from .engine import render_map
    from .history import PARAM_SPECS, validate_history

    started = perf_counter()
    if int(budget) < 1 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("budget must be positive and seconds must be finite and nonnegative")
    budget = int(budget)
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    target = np.asarray(target)
    if xs.ndim != 1 or ys.ndim != 1 or target.shape != (ys.size, xs.size):
        raise ValueError("target must have shape (len(ys), len(xs)) for one-dimensional axes")
    canonical = validate_history(history)
    best_history = deepcopy(canonical)
    full_scorer = MapScorer(target, mask, boundary_mask)
    baseline_map = render_map(best_history, xs, ys, terrain=terrain)
    best_metrics = full_scorer(baseline_map)
    evaluations = 1

    def result() -> tuple[dict, dict, int, float]:
        return best_history, best_metrics, evaluations, (perf_counter() - started) * 1000.0

    if budget == 1 or seconds == 0 or perf_counter() - started >= seconds:
        return result()
    allowed = None if allowed_fields is None else ({allowed_fields} if isinstance(allowed_fields, str) else set(allowed_fields))
    parameters = _parameters(canonical, xs, ys, allowed, PARAM_SPECS)
    if not parameters:
        return result()
    ix = np.linspace(0, xs.size - 1, min(xs.size, 128)).round().astype(int)
    iy = np.linspace(0, ys.size - 1, min(ys.size, 128)).round().astype(int)
    downsampled = ix.size != xs.size or iy.size != ys.size
    search_xs, search_ys = xs[ix], ys[iy]
    if downsampled:
        search_target = target[np.ix_(iy, ix)]
        search_mask = full_scorer.mask[np.ix_(iy, ix)]
        # A very sparse fixed mask may miss every coarse sample.
        if not np.any(search_mask):
            return result()
        scorer = MapScorer(search_target, search_mask, full_scorer.boundary_mask[np.ix_(iy, ix)])
        search_terrain = terrain if terrain is None or np.ndim(terrain) == 0 else np.asarray(terrain)[np.ix_(iy, ix)]
        current_metrics = scorer(render_map(canonical, search_xs, search_ys, terrain=search_terrain))
        evaluations += 1
    else:
        scorer, search_terrain, current_metrics = full_scorer, terrain, best_metrics
    current = deepcopy(canonical)
    pool: list[tuple[float, dict]] = [(current_metrics["score"], deepcopy(current))]
    rng = np.random.default_rng(seed)
    deadline = started + seconds
    search_deadline = started + seconds * (0.75 if downsampled else 1.0)
    # Reserve final checks even when the caller gives a tiny evaluation cap.
    reserved = min(3, max(1, (budget - evaluations) // 5)) if downsampled else 0
    search_cap = budget - reserved

    def evaluate(candidate: dict | None) -> None:
        nonlocal evaluations, current, current_metrics, best_history, best_metrics
        if candidate is None or evaluations >= search_cap or perf_counter() >= search_deadline:
            return
        try:
            predicted = render_map(candidate, search_xs, search_ys, terrain=search_terrain)
        except ValueError:
            return
        evaluations += 1
        metrics = scorer(predicted)
        if all(candidate != existing for _, existing in pool):
            pool.append((metrics["score"], deepcopy(candidate)))
            pool.sort(key=lambda item: item[0], reverse=True)
            del pool[4:]
        if metrics["score"] > current_metrics["score"] + 1e-12:
            current, current_metrics = candidate, metrics
        if not downsampled and metrics["score"] > best_metrics["score"] + 1e-12:
            best_history, best_metrics = deepcopy(candidate), metrics

    sweep = 0
    while evaluations < search_cap and perf_counter() < search_deadline:
        previous_evaluations = evaluations
        scale = (1.0, 0.6, 0.3, 0.15)[min(sweep, 3)]
        for parameter_index in rng.permutation(len(parameters)):
            parameter = parameters[int(parameter_index)]
            base = deepcopy(current)
            for direction in rng.permutation((-1.0, 1.0)):
                evaluate(_perturb(base, parameter, float(direction) * parameter.step * scale))
            if evaluations >= search_cap or perf_counter() >= search_deadline:
                break
            # Joint moves give strike/location/depth a chance to improve together.
            if evaluations % 8 < 2 and len(parameters) > 1:
                base = deepcopy(pool[int(rng.integers(min(2, len(pool))))][1])
                for chosen in rng.choice(len(parameters), size=min(3, len(parameters)), replace=False):
                    changed = _perturb(base, parameters[int(chosen)], float(rng.normal()) * parameters[int(chosen)].step * scale)
                    if changed is not None:
                        base = changed
                evaluate(base)
        sweep += 1
        if evaluations == previous_evaluations:
            break

    if downsampled:
        for _, candidate in pool:
            if candidate == canonical:
                continue
            if evaluations >= budget or perf_counter() >= deadline:
                break
            metrics = full_scorer(render_map(candidate, xs, ys, terrain=terrain))
            evaluations += 1
            if metrics["score"] > best_metrics["score"] + 1e-12:
                best_history, best_metrics = deepcopy(candidate), metrics
    return result()
