"""A small, non-executing language for chronological geological histories.

Coordinates and distances are kilometres; angles are degrees. World axes are
x east, y north, z up. Programs contain only calls with literal keyword values.
The parser never compiles or executes user-provided Python.
"""

from __future__ import annotations

import ast
import math
from typing import Any


class HistoryError(ValueError):
    """An invalid or unsupported geological program."""


def _spec(low: float, high: float, default: float, step: float) -> dict[str, float]:
    return {"min": low, "max": high, "default": default, "step": step}


# Direct event dictionaries are intentionally convenient for bounded numerical
# refinement. Defaults and bounds are shared by parsing and engine validation.
PARAM_SPECS: dict[str, dict[str, dict[str, float]]] = {
    "anticline": {
        "x": _spec(-100, 100, 0, 0.10),
        "y": _spec(-100, 100, 0, 0.10),
        "azimuth": _spec(0, 360, 135, 3),
        "uplift": _spec(0, 10, 1, 0.10),
        "dip_ne": _spec(0, 85, 55, 4),
        "dip_sw": _spec(0, 85, 25, 3),
        "hinge": _spec(0.001, 5, 0.15, 0.025),
        "nw_length": _spec(0, 100, 0.6, 0.15),
        "se_length": _spec(0, 100, 4, 0.15),
        "plunge_nw": _spec(0, 60, 20, 3),
        "plunge_se": _spec(0, 60, 0, 3),
    },
    "tilt": {
        "x": _spec(-100, 100, 0, 0.1),
        "y": _spec(-100, 100, 0, 0.1),
        "z": _spec(-100, 100, 0, 0.1),
        "azimuth": _spec(0, 360, 135, 3),
        "angle": _spec(-85, 85, 0, 3),
    },
    "fault": {
        "x": _spec(-100, 100, 0, 0.1),
        "y": _spec(-100, 100, 0, 0.1),
        "z": _spec(-100, 100, 0, 0.1),
        "azimuth": _spec(0, 360, 135, 3),
        "dip": _spec(1, 90, 60, 3),
        "slip": _spec(-10, 10, 0, 0.1),
    },
    "intrusion": {
        "x": _spec(-100, 100, 0, 0.1),
        "y": _spec(-100, 100, 0, 0.1),
        "z": _spec(-100, 100, -1, 0.1),
        "rx": _spec(0.001, 10, 0.3, 0.05),
        "ry": _spec(0.001, 10, 0.3, 0.05),
        "rz": _spec(0.001, 10, 0.3, 0.05),
        "unit": _spec(1, 254, 8, 1),
    },
    "erode": {"level": _spec(-100, 100, 0, 0.1)},
}

MAX_EVENTS = 24
MAX_PROGRAM_LENGTH = 24_000
MAX_UNITS = 64


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HistoryError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise HistoryError(f"{where} must be a finite number") from None
    if not math.isfinite(result):
        raise HistoryError(f"{where} must be finite")
    return result


def _unit(value: Any, where: str) -> int:
    number = _number(value, where)
    if number != int(number) or not 1 <= number <= 254:
        raise HistoryError(f"{where} must be an integer from 1 to 254 (0 is air)")
    return int(number)


def _canonical_event(event: Any, index: int) -> dict[str, Any]:
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise HistoryError(f"Event {index + 1} must have a supported type")
    kind = event["type"]
    if kind == "strata":
        if set(event) != {"type", "levels", "units"}:
            raise HistoryError("strata requires only levels and units")
        if not isinstance(event["levels"], (list, tuple)) or not isinstance(event["units"], (list, tuple)):
            raise HistoryError("strata levels and units must be lists")
        if not 1 <= len(event["units"]) <= MAX_UNITS:
            raise HistoryError(f"strata requires 1–{MAX_UNITS} units")
        if len(event["levels"]) + 1 != len(event["units"]):
            raise HistoryError("strata requires one more unit than contact levels")
        levels = [_number(v, "strata level") for v in event["levels"]]
        if any(abs(v) > 100 for v in levels):
            raise HistoryError("strata levels must be between -100 and 100 km")
        if any(a >= b for a, b in zip(levels, levels[1:])):
            raise HistoryError("strata levels must be strictly increasing")
        units = [_unit(v, "strata unit") for v in event["units"]]
        if len(set(units)) != len(units):
            raise HistoryError("strata unit IDs must be unique")
        return {"type": kind, "levels": levels, "units": units}
    if kind not in PARAM_SPECS:
        raise HistoryError(f"Unsupported event {kind!r}")
    specs = PARAM_SPECS[kind]
    unknown = set(event) - {"type", *specs}
    if unknown:
        raise HistoryError(f"Unknown {kind} parameter(s): {', '.join(sorted(unknown))}")
    result: dict[str, Any] = {"type": kind}
    for name, spec in specs.items():
        value = _number(event.get(name, spec["default"]), f"{kind}.{name}")
        if not spec["min"] <= value <= spec["max"]:
            raise HistoryError(f"{kind}.{name} must be in [{spec['min']}, {spec['max']}]")
        result[name] = _unit(value, "intrusion.unit") if name == "unit" else value
    return result


def validate_history(history: Any) -> dict[str, Any]:
    """Return a fresh canonical JSON-serializable history or raise HistoryError.

    A string program, a canonical dictionary, or an event list is accepted.
    Erosion may only be the final event in this bounded language: unconformity
    deposition and intrusion into an already-eroded air region are not implied.
    """
    if isinstance(history, str):
        return parse_history(history)
    if isinstance(history, dict):
        if set(history) - {"version", "units", "events"}:
            raise HistoryError("Unknown history metadata")
        if history.get("version", 1) != 1:
            raise HistoryError("Unsupported history version")
        if history.get("units", "km") != "km":
            raise HistoryError("History distances must use km")
        events = history.get("events")
    else:
        events = history
    if not isinstance(events, (list, tuple)) or not 1 <= len(events) <= MAX_EVENTS:
        raise HistoryError(f"History requires 1–{MAX_EVENTS} events")
    result = [_canonical_event(event, i) for i, event in enumerate(events)]
    if result[0]["type"] != "strata" or any(e["type"] == "strata" for e in result[1:]):
        raise HistoryError("History must begin with exactly one strata event")
    if any(e["type"] == "erode" for e in result[:-1]):
        raise HistoryError("erode must be the final event; younger deposition is not supported")
    existing = set(result[0]["units"])
    for event in result[1:]:
        if event["type"] == "intrusion":
            if event["unit"] in existing:
                raise HistoryError("Each intrusion must create a new, distinct unit ID")
            existing.add(event["unit"])
    return {"version": 1, "units": "km", "events": result}


def parse_history(program: str) -> dict[str, Any]:
    """Parse literal event calls, never arbitrary Python execution.

    Examples: strata(levels=[-1, 0], units=[1, 2, 3]); anticline(...);
    erode(level=0). Comments and semicolon-separated calls are accepted.
    Imports, assignment, attributes, expressions, splats, and positional
    arguments are rejected. Only numeric constants and numeric lists exist.
    """
    if not isinstance(program, str) or len(program) > MAX_PROGRAM_LENGTH:
        raise HistoryError(f"Program must be text shorter than {MAX_PROGRAM_LENGTH} characters")
    try:
        tree = ast.parse(program, mode="exec")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise HistoryError(f"Invalid history syntax: {exc}") from None
    if sum(1 for _ in ast.walk(tree)) > 2000:
        raise HistoryError("History program is too complex")
    events = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise HistoryError("Only geological event calls are allowed")
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.args:
            raise HistoryError("Events require a simple name and keyword arguments")
        values: dict[str, Any] = {"type": call.func.id}
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg == "type" or keyword.arg in values:
                raise HistoryError("Expanded, duplicate, and reserved keyword arguments are forbidden")
            # literal_eval alone permits strings, dicts and complex arithmetic;
            # this explicit whitelist narrows the grammar to numerical literals.
            for node in ast.walk(keyword.value):
                if not isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.Load,
                                         ast.UnaryOp, ast.USub, ast.UAdd)):
                    raise HistoryError("Parameter values must be numeric literals or lists")
                if isinstance(node, ast.Constant) and (
                    isinstance(node.value, bool) or not isinstance(node.value, (int, float))
                ):
                    raise HistoryError("Only numeric parameter literals are allowed")
            try:
                values[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError, RecursionError):
                raise HistoryError("Parameter values must be numeric literals or lists") from None
        events.append(values)
    return validate_history(events)


def history_to_program(history: Any) -> str:
    """Serialize a canonical history into reproducible literal event calls."""
    canonical = validate_history(history)
    return "\n".join(
        f"{event['type']}(" + ", ".join(
            f"{key}={value!r}" for key, value in event.items() if key != "type"
        ) + ")"
        for event in canonical["events"]
    )
