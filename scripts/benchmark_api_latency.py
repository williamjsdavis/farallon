"""Compare at most four real production-shaped Responses calls, without changing the demo.

Run only while the interactive demo is idle. The same five-image request is
captured from server.propose without network access, then sent with streaming
to measure the first visible JSON token as well as full completion latency.
The request order is default, fast, default, fast. This is a tiny diagnostic,
not a statistically powered throughput benchmark or a speed guarantee.
"""

import argparse
import asyncio
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


async def capture_request():
    captured = {}

    class CaptureClient:
        def __init__(self, **kwargs):
            self.responses = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(status="completed", output_text="{}", usage=None,
                                   id="capture-only-no-request", service_tier=None)

    request = server.IterationRequest(program=server.BASELINE, previous=[], refine=False,
                                     scene_id=server.SCENE_ID, baseline_id=server.BASELINE_ID)
    current = server.render_result(server.canonical(server.BASELINE), volume=False)
    with patch.object(server, "AsyncOpenAI", CaptureClient):
        await server.propose(request, current)
    captured.pop("service_tier", None)
    assert captured["model"] == "gpt-6-astra"
    assert captured["reasoning"] == {"effort": "low"}
    content = [item for message in captured["input"] for item in message["content"]]
    assert sum(item["type"] == "input_image" for item in content) == 5
    assert captured["text"]["format"]["strict"] is True
    return captured


async def timed_call(parameters, tier):
    started = perf_counter()
    first_event_ms = first_text_ms = None
    completed = None
    event_count = text_delta_count = text_characters = 0
    async with server.AsyncOpenAI(timeout=75.0, max_retries=0) as client:
        stream = await client.responses.create(**parameters, service_tier=tier, stream=True)
        async with stream:
            async for event in stream:
                elapsed_ms = (perf_counter() - started) * 1000
                event_count += 1
                if first_event_ms is None:
                    first_event_ms = elapsed_ms
                if event.type == "response.output_text.delta" and event.delta:
                    if first_text_ms is None:
                        first_text_ms = elapsed_ms
                    text_delta_count += 1
                    text_characters += len(event.delta)
                if event.type in {"response.completed", "response.incomplete", "response.failed"}:
                    completed = event.response
    wall_ms = (perf_counter() - started) * 1000
    if completed is None:
        raise RuntimeError("Stream ended without a terminal response event")
    usage = completed.usage.model_dump() if completed.usage else None
    program_valid = False
    if completed.status == "completed":
        try:
            proposal = json.loads(completed.output_text)
            server.canonical(proposal["program"])
            program_valid = True
        except (ValueError, KeyError):
            pass
    return {"response_id": completed.id, "requested_service_tier": tier,
            "service_tier": completed.service_tier, "status": completed.status,
            "wall_ms": wall_ms, "first_event_ms": first_event_ms,
            "first_visible_text_ms": first_text_ms,
            "text_generation_ms": None if first_text_ms is None else wall_ms - first_text_ms,
            "event_count": event_count, "text_delta_count": text_delta_count,
            "text_characters": text_characters, "usage": usage, "program_valid": program_valid}


async def benchmark(args):
    parameters = await capture_request()
    prompt_bytes = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "model": parameters["model"], "reasoning": parameters["reasoning"],
              "scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID,
              "prompt_sha256": sha256(prompt_bytes).hexdigest(),
              "request_bytes": len(prompt_bytes), "image_count": 5,
              "prompt_description": "Production baseline request: full map, observed crop, predicted surface, mismatch, fixed terrain; strict JSON proposal schema.",
              "transport": "Responses streaming, fresh client per call, retries disabled",
              "order": ["default", "fast"] * args.pairs,
              "timing_note": "First visible text is first nonempty output_text.delta, excluding metadata events and hidden reasoning. Wall includes client setup/network/API/stream consumption, excludes simulator and numerical search.",
              "samples": []}
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    # Never overwrite a prior diagnostic: failed requests can still have incurred cost.
    with args.output.open("x") as destination:
        json.dump(report, destination, indent=2)
    for index, tier in enumerate(report["order"]):
        sample = {"index": index + 1, "pair": index // 2 + 1,
                  "requested_service_tier": tier, "state": "started",
                  "started_at": datetime.now(timezone.utc).isoformat()}
        report["samples"].append(sample)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({"type": "started", **sample}), flush=True)
        try:
            sample.update(await timed_call(parameters, tier))
            sample["state"] = "completed"
        except Exception as exc:
            sample.update(state="error", error_type=type(exc).__name__)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps({"type": "error", **sample}), flush=True)
            return 1  # No retry and no further paid calls after an error.
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({"type": "completed", **sample}), flush=True)
        if sample["status"] != "completed":
            return 1
    print(json.dumps({"type": "benchmark_completed", "output": str(args.output),
                      "calls": len(report["samples"])}), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dry-run", action="store_true", help="Inspect captured metadata without model calls")
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/farallon-api-latency.json"))
    raise SystemExit(asyncio.run(benchmark(parser.parse_args())))
