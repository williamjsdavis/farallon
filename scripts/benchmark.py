"""Measure the local baseline after JIT warmup; never makes model/API calls.

Run: .venv/bin/python scripts/benchmark.py [--trials 20] [--json]
Includes validation, packing, allocation, and (for payloads) scoring/encoding.
Excludes imports, JIT warmup, HTTP transfer, and browser/WebGL rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numba
import numpy as np

from geology.engine import render_map, render_volume
from server import BASELINE, META, XS, YS, canonical, render_result


def measure(operation, trials: int) -> dict[str, float]:
    samples = []
    for _ in range(trials):
        started = perf_counter()
        operation()
        samples.append((perf_counter() - started) * 1000)
    return {
        "median_ms": round(float(np.median(samples)), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=96, choices=range(32, 129), metavar="32..128")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args()
    if not 1 <= args.trials <= 1000:
        parser.error("--trials must be between 1 and 1000")

    history = canonical(BASELINE)
    b = META["bounds"]
    bounds = [b["xmin"], b["xmax"], b["ymin"], b["ymax"], -1.8, 0.0]

    def payload():
        return render_result(history, resolution=args.resolution)

    def payload_json():
        return json.dumps(payload(), allow_nan=False, separators=(",", ":"))

    # One complete render warms every measured numerical signature and encoder.
    # This can load an existing Numba disk cache or compile it on a new machine.
    warmup_start = perf_counter()
    sample_json = payload_json()
    warmup_ms = (perf_counter() - warmup_start) * 1000

    operations = {
        "map": lambda: render_map(history, XS, YS),
        "volume": lambda: render_volume(history, bounds, args.resolution),
        "server_payload": payload,
        "server_payload_with_json": payload_json,
    }
    report = {
        "trials": args.trials,
        "history_sha256": hashlib.sha256(
            json.dumps(history, sort_keys=True).encode()
        ).hexdigest(),
        "events": len(history["events"]),
        "map_shape_yx": [len(YS), len(XS)],
        "volume_shape_zyx": [args.resolution] * 3,
        "payload_json_bytes": len(sample_json.encode()),
        "warmup_ms": round(warmup_ms, 3),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "timings": {name: measure(operation, args.trials) for name, operation in operations.items()},
        "excludes": "Imports, JIT warmup, HTTP transfer, browser/WebGL rendering, and model latency.",
    }
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print(f"Farallon warm benchmark: {args.trials} trials, {len(XS)}×{len(YS)} map, {args.resolution}³ volume")
    print(f"Python {platform.python_version()} / NumPy {np.__version__} / Numba {numba.__version__} / {platform.machine()}")
    print(f"Initial warmup: {warmup_ms:.1f} ms (excluded; may load a disk cache or compile)")
    print(f"{'Operation':<28} {'Median (ms)':>12} {'p95 (ms)':>12}")
    for name, timing in report["timings"].items():
        print(f"{name:<28} {timing['median_ms']:>12.3f} {timing['p95_ms']:>12.3f}")
    print(f"JSON payload: {report['payload_json_bytes']:,} bytes; baseline events: {report['events']}")
    print(f"History SHA-256: {report['history_sha256']}")
    print(f"Excluded: {report['excludes']}")


if __name__ == "__main__":
    main()
