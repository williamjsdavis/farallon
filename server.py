"""Local simulator and visual hypothesis loop. API credentials stay here."""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field

from geology.engine import render_map, render_volume
from geology.history import HistoryError, history_to_program, parse_history, PARAM_SPECS
from geology.scoring import MapScorer, mismatch_rgba, prepare_boundary_mask
from geology.search import refine_history

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-astra")
EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
BASELINE = """# A simple starting hypothesis: symmetric limbs, no plunge.
strata(levels=[-1.2, -1.05, -0.92, -0.78, -0.5, -0.2], units=[1, 2, 3, 4, 5, 6, 7])
anticline(x=3.2, y=1.0, azimuth=135, uplift=1.3, dip_ne=30, dip_sw=30, hinge=0.15, nw_length=1.0, se_length=3.0, plunge_nw=0, plunge_se=0)
erode(level=0)
"""
TARGET = np.load(ROOT / "data/target.npz")
META = json.loads((ROOT / "data/target.json").read_text())
LABELS, MASK = TARGET["labels"], TARGET["mask"]
XS, YS = TARGET["xs"], TARGET["ys"]
BOUNDARY_MASK = prepare_boundary_mask(MASK, TARGET["exclusion_reasons"])
SCORER = MapScorer(LABELS, MASK, BOUNDARY_MASK)
PALETTE = np.zeros((256, 4), dtype=np.uint8)
for entry in META["palette"]:
    PALETTE[entry["id"]] = [*entry["rgb"], 255]
PALETTE[0] = [0, 0, 0, 0]
WARMUP_MS = 0.0
ITERATION_LOCK = asyncio.Lock()


def png_url(rgba: np.ndarray) -> str:
    output = BytesIO()
    Image.fromarray(rgba).save(output, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def file_url(path: Path) -> str:
    suffix = "webp" if path.suffix == ".webp" else "png"
    return f"data:image/{suffix};base64," + base64.b64encode(path.read_bytes()).decode()


def canonical(program: str) -> dict:
    history = parse_history(program)
    # Unit identities and the observation surface are data, not fit parameters.
    if history["events"][0]["units"] != list(range(1, 8)):
        raise HistoryError("Keep the seven observed unit IDs in their original order.")
    if history["events"][-1]["type"] != "erode" or history["events"][-1]["level"] != 0:
        raise HistoryError("The observation surface is fixed: end with erode(level=0).")
    if len(history["events"]) > 8:
        raise HistoryError("Use at most eight events for this demonstration.")
    if any(e["type"] == "intrusion" for e in history["events"]):
        raise HistoryError("This target has seven sedimentary packages; adding intrusive units is outside its scope.")
    return history


def render_result(history: dict, volume: bool = True, resolution: int = 96) -> dict:
    started = perf_counter()
    prediction = render_map(history, XS, YS)
    map_ms = (perf_counter() - started) * 1000
    score_start = perf_counter()
    metrics = SCORER(prediction)
    score_ms = (perf_counter() - score_start) * 1000
    payload: dict = {
        "id": uuid4().hex[:12], "program": history_to_program(history),
        "history": history, "metrics": metrics,
    }
    volume_ms = 0.0
    if volume:
        b = META["bounds"]
        bounds = [b["xmin"], b["xmax"], b["ymin"], b["ymax"], -1.8, 0.0]
        volume_start = perf_counter()
        labels = render_volume(history, bounds, resolution)
        volume_ms = (perf_counter() - volume_start) * 1000
        payload["volume"] = {"bounds": bounds, "shape": list(labels.shape),
                             "data": base64.b64encode(labels.tobytes()).decode()}
    encode_start = perf_counter()
    rgba = PALETTE[prediction].copy()
    payload["map_image"] = png_url(rgba)
    rgba[~MASK, 3] = 0
    payload["observed_map_image"] = png_url(rgba)
    payload["error_image"] = png_url(mismatch_rgba(prediction, LABELS, MASK))
    payload["timings"] = {"map_ms": map_ms, "volume_ms": volume_ms, "score_ms": score_ms,
                          "encode_ms": (perf_counter() - encode_start) * 1000,
                          "generation_ms": map_ms + volume_ms,
                          "total_ms": (perf_counter() - started) * 1000}
    return payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    global WARMUP_MS
    start = perf_counter()
    await asyncio.to_thread(render_result, canonical(BASELINE))
    WARMUP_MS = (perf_counter() - start) * 1000
    yield


app = FastAPI(title="Farallon", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class RenderRequest(BaseModel):
    program: str = Field(max_length=24000)
    volume: bool = True
    resolution: int = Field(default=96, ge=32, le=128)


class IterationRequest(BaseModel):
    program: str = Field(max_length=24000)
    previous: list[dict] = Field(default_factory=list, max_length=12)
    refine: bool = True


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL, "reasoning_effort": EFFORT,
            "api_key_available": bool(os.getenv("OPENAI_API_KEY")), "warmup_ms": WARMUP_MS}


@app.get("/api/bootstrap")
async def bootstrap():
    return {"target": META, "model": MODEL, "api_key_available": bool(os.getenv("OPENAI_API_KEY")),
            "baseline": await asyncio.to_thread(render_result, canonical(BASELINE))}


@app.get("/api/image/{name}")
def image_file(name: str):
    files = {"target": ROOT / "data/target.png", "source": ROOT / "data/target_source.png",
             "full": ROOT / META["sourceImage"], "mask": ROOT / "data/target_mask.png"}
    if name not in files:
        raise HTTPException(404, "Unknown image")
    return FileResponse(files[name])


@app.post("/api/render")
async def render(request: RenderRequest):
    try:
        history = canonical(request.program)
        return await asyncio.to_thread(render_result, history, request.volume, request.resolution)
    except HistoryError as exc:
        raise HTTPException(422, str(exc)) from None


@app.post("/api/refine")
async def refine(request: IterationRequest):
    try:
        history = canonical(request.program)
        result, _, evaluations, elapsed_ms = await asyncio.to_thread(
            refine_history, history, LABELS, XS, YS, MASK, budget=160, seconds=3.0,
            boundary_mask=BOUNDARY_MASK)
        payload = await asyncio.to_thread(render_result, result)
        payload["search"] = {"evaluations": evaluations, "elapsed_ms": elapsed_ms}
        return payload
    except HistoryError as exc:
        raise HTTPException(422, str(exc)) from None


SYSTEM_PROMPT = """You infer a compact geological history from an observed geological map.
Write an executable program in the supplied geological DSL. Your program actually drives
a deterministic 3D simulator; its fixed terrain intersection is compared with observed unit IDs.
Use the images to locate mismatches, then propose a coherent revision that can improve the fit.
Distinguish visual evidence from geological inference. Be specific about WHERE a feature occurs.
Prefer one meaningful hypothesis revision at a time. You may adjust parameters, add a second
fold, or revise an event order when justified. Do not just repeat an earlier rejected proposal.
Do not invent a visible fault from fold-axis arrows or the river. Do not repaint or relabel units.
Map exposure does not uniquely determine the subsurface. The terrain is flat/schematic;
erosional irregularities and extremely thin source units may remain unexplained.

DSL: one call per event, oldest to youngest, only literal keyword arguments, no imports,
assignments, loops, expressions, or function definitions. Begin with strata and end with
erode(level=0). Units must remain [1,2,3,4,5,6,7], oldest to youngest. Preserve all seven.
strata(levels=[six strictly increasing contact elevations in km], units=[1,2,3,4,5,6,7])
anticline(x=..., y=..., azimuth=..., uplift=..., dip_ne=..., dip_sw=..., hinge=...,
          nw_length=..., se_length=..., plunge_nw=..., plunge_se=...)
tilt(x=...,y=...,z=...,azimuth=...,angle=...)
fault(x=...,y=...,z=...,azimuth=...,dip=...,slip=...)
erode(level=0)
At most 8 events. Strata extend infinitely before final erosion. Effective VERTICAL contact
intervals are fitted, not measured normal bed thicknesses. Oldest unit occupies z below
the lowest level; youngest above the highest. Keep the highest level below zero when needed.

Anticline semantics: x east, y north, z up, distances km. Azimuth clockwise from north is
the SE-directed axis, typically in the vicinity of135 degrees. Local v points along this axis;
local u points to its NE side. Center(x,y) is center of the axial plateau, not the NW nose.
Cross-axis displacement F=max(0, uplift - tan(dip_side)*(sqrt(u*u+hinge*hinge)-hinge)
 - tan(plunge_nw)*smooth_positive(-v-nw_length)
 - tan(plunge_se)*smooth_positive(v-se_length)). The actual smoothing scale is hinge.
The program lifts previously created rocks by F. Large plunge closes the ends of bands.
With plunge=0, that end never closes regardless of length. Increasing limb dip narrows
its outcrop. A smaller hinge tightens curvature. Reverse coordinates classify z-F.
A second anticline adds its uplift to the first. Parameters bounds are supplied below.

Return JSON following the schema. `program` must contain the entire executable revised
history. `headline` is a short human-readable change; `observation` is at most2 sentences
of grounded visual evidence; `expected_effect` is a short testable effect. Set
`parameters_to_refine` to up to10 field names such as x,y,azimuth,uplift,levels,dip_ne,
dip_sw,hinge,nw_length,plunge_nw that a local search should tune after your proposal.
Numerical refinements and actual measurements, not your own assessment, decide acceptance.
"""


async def propose(request: IterationRequest, current: dict) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in the local .env file, then restart the server.")
    info = {"bounds_km": META["bounds"], "image_axes": "top=north/right=east",
            "legend": [{k: e[k] for k in ("id", "name", "color", "age")} for e in META["palette"]],
            "observed_fraction": META["labeledFraction"], "current_program": request.program,
            "current_metrics": current["metrics"], "parameter_bounds": PARAM_SPECS,
            "recent_attempts": request.previous[-4:]}
    images = [
        ("Original map with legend; only the NW crop is scored:", file_url(ROOT / META["sourceImage"])),
        ("Clean observed NW target; transparent gaps are unobserved. Same bounds as prediction:", file_url(ROOT / "data/target.png")),
        ("Current predicted surface, same north-up frame and geological colors:", current["map_image"]),
        ("Disagreement overlay: red is mismatch; green is agreement; transparent is unobserved:", current["error_image"]),
    ]
    content: list[dict] = [{"type": "input_text", "text": json.dumps(info)}]
    for label, url in images:
        content += [{"type": "input_text", "text": label},
                    {"type": "input_image", "image_url": url, "detail": "high"}]
    schema = {"type": "object", "properties": {
        "headline": {"type": "string"}, "observation": {"type": "string"},
        "expected_effect": {"type": "string"}, "program": {"type": "string"},
        "parameters_to_refine": {"type": "array", "items": {"type": "string"}}},
        "required": ["headline", "observation", "expected_effect", "program", "parameters_to_refine"],
        "additionalProperties": False}
    async with AsyncOpenAI(timeout=75.0, max_retries=0) as client:
        response = await client.responses.create(
            model=MODEL, reasoning={"effort": EFFORT}, max_output_tokens=3500,
            instructions=SYSTEM_PROMPT, input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "geological_proposal", "strict": True, "schema": schema}})
    if response.status != "completed":
        raise RuntimeError("The model did not finish its proposal. Try the iteration again.")
    proposal = json.loads(response.output_text)
    proposal["usage"] = response.usage.model_dump() if response.usage else None
    proposal["response_id"] = response.id
    return proposal


@app.post("/api/iterate")
async def iterate(request: IterationRequest):
    try:
        original = canonical(request.program)
    except HistoryError as exc:
        raise HTTPException(422, str(exc)) from None
    if ITERATION_LOCK.locked():
        raise HTTPException(409, "An iteration is already running.")

    async def events():
        async with ITERATION_LOCK:
            started = perf_counter()
            def event(kind, **values):
                return json.dumps({"type": kind, **values}) + "\n"
            try:
                yield event("status", stage="observing", message="Reading the map and the current mismatch")
                current = await asyncio.to_thread(render_result, original, False)
                llm_start = perf_counter()
                proposal = await propose(request, current)
                llm_ms = (perf_counter() - llm_start) * 1000
                candidate = canonical(proposal["program"])
                yield event("proposal", **proposal, model_ms=llm_ms)
                yield event("status", stage="testing", message="Generating the hypothesis and tuning its parameters")
                evaluations, search_ms = 1, 0.0
                if request.refine:
                    candidate, _, evaluations, search_ms = await asyncio.to_thread(
                        refine_history, candidate, LABELS, XS, YS, MASK, budget=192, seconds=3.0,
                        seed=len(request.previous), allowed_fields=proposal["parameters_to_refine"][:10],
                        boundary_mask=BOUNDARY_MASK)
                result = await asyncio.to_thread(render_result, candidate)
                accepted = result["metrics"]["score"] > current["metrics"]["score"] + 1e-6
                record = {"proposal": proposal, "result": result, "accepted": accepted,
                          "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                          "before": current["metrics"], "model_ms": llm_ms,
                          "search": {"evaluations": evaluations, "elapsed_ms": search_ms},
                          "elapsed_ms": (perf_counter() - started) * 1000, "model": MODEL}
                run_dir = ROOT / "data/runs"
                run_dir.mkdir(exist_ok=True)
                (run_dir / f"{result['id']}.json").write_text(json.dumps(record))
                yield event("result", **record)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # OpenAI error messages do not contain keys; avoid returning request bodies.
                message = str(exc) if isinstance(exc, (HistoryError, RuntimeError)) else f"{type(exc).__name__}: proposal could not complete. Your current model is unchanged."
                yield event("error", message=message)
    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/runs")
def runs():
    records = []
    for path in (ROOT / "data/runs").glob("*.json"):
        record = json.loads(path.read_text())
        try:
            recorded_at = datetime.fromisoformat(record["recorded_at"])
            if recorded_at.tzinfo is None:
                raise ValueError("A recording timestamp must include its time zone")
            timestamp = recorded_at.timestamp()
        except (KeyError, TypeError, ValueError, OverflowError):
            timestamp = path.stat().st_mtime  # Older local recordings lack timestamps.
        summary = {"id": path.stem, "headline": record["proposal"]["headline"],
                   "accepted": record["accepted"], "miou": record["result"]["metrics"]["miou"]}
        records.append((timestamp, path.stem, summary))
    return [summary for _, _, summary in sorted(records, key=lambda row: (row[0], row[1]))]


@app.get("/api/runs/{run_id}")
def replay(run_id: str):
    if len(run_id) != 12 or not all(c in "0123456789abcdef" for c in run_id):
        raise HTTPException(404)
    path = ROOT / "data/runs" / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(404)
    return json.loads(path.read_text())
