"""Run 1–3 genuine paid visual proposals through the application's own API.

Usage: .venv/bin/python scripts/rehearse.py --calls 3
Each completed iteration is saved by /api/iterate in data/runs, including its
actual volume, metrics, program, model response ID and token usage. Rejected
candidates remain in the record; subsequent rounds continue the best history.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from time import perf_counter

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


async def rehearse(calls: int) -> int:
    if not 1 <= calls <= 3:
        raise ValueError("A rehearsal permits one to three model calls")
    transport = httpx.ASGITransport(app=server.app)
    records: list[dict] = []
    async with server.lifespan(server.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://farallon.local", timeout=120) as client:
            health = (await client.get("/api/health")).json()
            if not health["api_key_available"]:
                print("OPENAI_API_KEY is unavailable; no model calls made.", flush=True)
                return 2
            bootstrap = (await client.get("/api/bootstrap")).json()
            current = bootstrap["baseline"]
            scene_id = bootstrap.get("scene_id", current.get("scene_id"))
            baseline_id = bootstrap.get("baseline_id", current.get("baseline_id"))
            baseline_metrics = current["metrics"]
            print(json.dumps({"type": "baseline", "model": health["model"], "metrics": baseline_metrics,
                              "timings": current["timings"], "warmup_ms": health["warmup_ms"],
                              "scene_id": scene_id, "baseline_id": baseline_id,
                              "event_types": [event["type"] for event in current["history"]["events"]]}), flush=True)
            previous: list[dict] = []
            for index in range(calls):
                print(json.dumps({"type": "call_started", "call": index + 1, "maximum_calls": calls}), flush=True)
                start = perf_counter()
                result = None
                async with client.stream("POST", "/api/iterate", json={
                    "program": current["program"], "previous": previous, "refine": True,
                    **({"scene_id": scene_id} if scene_id is not None else {}),
                    **({"baseline_id": baseline_id} if baseline_id is not None else {}),
                }) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        message = json.loads(line)
                        if message["type"] in {"status", "proposal", "error"}:
                            print(json.dumps(message), flush=True)
                        if message["type"] == "error":
                            print(json.dumps({"type": "stopped", "calls_attempted": index + 1,
                                              "reason": "Application reported an error; no automatic paid retry."}), flush=True)
                            return 2
                        if message["type"] == "result":
                            result = message
                if result is None:
                    print("Iteration stream ended without a result; no automatic paid retry.", flush=True)
                    return 2
                generated = result["result"]
                if scene_id is not None and result.get("scene_id", generated.get("scene_id")) != scene_id:
                    raise RuntimeError("Iteration belongs to a different or unidentified observation scene")
                if baseline_id is not None and result.get("baseline_id", generated.get("baseline_id")) != baseline_id:
                    raise RuntimeError("Iteration belongs to a different or unidentified starting history")
                if "volume" not in generated or not generated["volume"]["data"]:
                    raise RuntimeError("Recorded iteration lacks the actual replay volume")
                # Verify saved replay follows the same contract as the live payload.
                replay = (await client.get(f"/api/runs/{generated['id']}")).json()
                if replay["result"]["metrics"] != generated["metrics"]:
                    raise RuntimeError("Replay metrics do not match the live result")
                if scene_id is not None and replay.get("scene_id", replay["result"].get("scene_id")) != scene_id:
                    raise RuntimeError("Saved replay belongs to a different observation scene")
                if baseline_id is not None and replay.get("baseline_id", replay["result"].get("baseline_id")) != baseline_id:
                    raise RuntimeError("Saved replay belongs to a different starting history")
                records.append({"id": generated["id"], "accepted": result["accepted"],
                                "scene_id": scene_id, "baseline_id": baseline_id,
                                "metrics": generated["metrics"], "model_ms": result["model_ms"],
                                "search": result["search"], "generation": generated["timings"],
                                "wall_ms": (perf_counter() - start) * 1000})
                print(json.dumps({"type": "round_completed", "call": index + 1, **records[-1]}), flush=True)
                previous.append({"headline": result["proposal"]["headline"],
                                 "scene_id": scene_id, "baseline_id": baseline_id,
                                 "observation": result["proposal"]["observation"],
                                 "expected_effect": result["proposal"]["expected_effect"],
                                 "program": generated["program"], "accepted": result["accepted"],
                                 "metrics": generated["metrics"]})
                if result["accepted"]:
                    current = generated
            print(json.dumps({"type": "rehearsal_completed", "calls": calls,
                              "scene_id": scene_id, "baseline_id": baseline_id,
                              "baseline_miou": baseline_metrics["miou"], "best_miou": current["metrics"]["miou"],
                              "baseline_score": baseline_metrics["score"], "best_score": current["metrics"]["score"],
                              "record_ids": [record["id"] for record in records]}), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, choices=(1, 2, 3), default=1)
    arguments = parser.parse_args()
    raise SystemExit(asyncio.run(rehearse(arguments.calls)))
