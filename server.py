"""Local simulator and visual hypothesis loop. API credentials stay here."""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from secrets import randbits
from time import perf_counter
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field

from geology.engine import render_map, render_volume
from geology.history import HistoryError, history_to_program, parse_history, PARAM_SPECS
from geology.scoring import MapScorer, mismatch_rgba, prepare_boundary_mask
from geology.search import refine_history
from geology.restarts import restart_history
from geology.terrain import TerrainGrid

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-astra")
EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
SERVICE_TIER = os.getenv("OPENAI_SERVICE_TIER", "auto")
BASELINE = """# Start before deformation: seven horizontal sedimentary packages.
strata(levels=[-1.2, -1.05, -0.92, -0.78, -0.5, -0.2], units=[1, 2, 3, 4, 5, 6, 7])
erode(level=0)
"""
TARGET = np.load(ROOT / "data/target.npz")
META = json.loads((ROOT / "data/target.json").read_text())
LABELS, MASK = TARGET["labels"], TARGET["mask"]
XS, YS = TARGET["xs"], TARGET["ys"]
TERRAIN_GRID = TerrainGrid.from_npz(ROOT / "data/terrain.npz")
TERRAIN_META = json.loads((ROOT / "data/terrain.json").read_text())
HEIGHTS = TERRAIN_GRID.sample(XS, YS)
HEIGHTS.setflags(write=False)
# Observations, terrain and model semantics define a comparable scoring scene.
# Replays from the original flat experiment remain archived, not rescored.
scene_hash = sha256(b"farallon-terrain-v2-paired-contacts-v1")
for array in (LABELS, MASK, TARGET["exclusion_reasons"], XS, YS, HEIGHTS):
    scene_hash.update(np.ascontiguousarray(array).tobytes())
SCENE_ID = "sheep-" + scene_hash.hexdigest()[:16]
BASELINE_ID = sha256(json.dumps(parse_history(BASELINE), sort_keys=True).encode()).hexdigest()[:16]
META["surface"] = {"type": "USGS 3DEP terrain", "units": "km", "fixed": True,
                   "datum_m": TERRAIN_META["datum_m"], "scene_id": SCENE_ID}
BOUNDARY_MASK = prepare_boundary_mask(MASK, TARGET["exclusion_reasons"])
SCORER = MapScorer(LABELS, MASK, BOUNDARY_MASK)
PALETTE = np.zeros((256, 4), dtype=np.uint8)
for entry in META["palette"]:
    PALETTE[entry["id"]] = [*entry["rgb"], 255]
PALETTE[0] = [0, 0, 0, 0]
WARMUP_MS = 0.0
ITERATION_LOCK = asyncio.Lock()
AUTO_RUNS: dict[str, dict] = {}
AUTO_PATIENCE = 4
AUTO_RESTART_EVERY = 10


def png_url(rgba: np.ndarray) -> str:
    output = BytesIO()
    Image.fromarray(rgba).save(output, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def file_url(path: Path) -> str:
    suffix = "webp" if path.suffix == ".webp" else "png"
    return f"data:image/{suffix};base64," + base64.b64encode(path.read_bytes()).decode()


def terrain_image() -> str:
    height = (HEIGHTS - HEIGHTS.min()) / max(float(np.ptp(HEIGHTS)), 1e-9)
    low, high = np.array([47, 71, 78]), np.array([238, 219, 173])
    rgb = low[None, None, :] + height[:, :, None] * (high - low)[None, None, :]
    dy, dx = np.gradient(HEIGHTS, YS, XS)
    light = np.clip((1 + 0.5 * dx - 0.5 * dy) / np.sqrt(1 + dx * dx + dy * dy) / np.sqrt(1.5), 0, 1)
    rgb *= (0.5 + 0.5 * light[:, :, None])
    return png_url(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), np.full(HEIGHTS.shape, 255, np.uint8)]))


TERRAIN_IMAGE = terrain_image()


@lru_cache(maxsize=8)
def volume_terrain(resolution: int) -> tuple[np.ndarray, list[float], dict]:
    b = META["bounds"]
    xs = np.linspace(b["xmin"], b["xmax"], resolution + 1)
    ys = np.linspace(b["ymin"], b["ymax"], resolution + 1)
    centers = TERRAIN_GRID.sample((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2)
    vertices = TERRAIN_GRID.sample(xs, ys).astype("<f4")
    bounds = [b["xmin"], b["xmax"], b["ymin"], b["ymax"], -1.8,
              max(0.1, float(HEIGHTS.max()) + 0.03)]
    surface = {"shape": list(vertices.shape), "encoding": "float32-le", "order": "south-to-north",
               "data": base64.b64encode(vertices.tobytes()).decode()}
    return centers, bounds, surface


def check_scene(scene_id: str | None, baseline_id: str | None = None):
    if ((scene_id is not None and scene_id != SCENE_ID) or
            (baseline_id is not None and baseline_id != BASELINE_ID)):
        raise HTTPException(409, "The terrain or starting history has changed. Reload the demo to use the current scene.")


def canonical(program: str) -> dict:
    history = parse_history(program)
    # Unit identities and the observation surface are data, not fit parameters.
    if history["events"][0]["units"] != list(range(1, 8)):
        raise HistoryError("Keep the seven observed unit IDs in their original order.")
    if history["events"][-1]["type"] != "erode" or history["events"][-1]["level"] != 0:
        raise HistoryError("The measured terrain is fixed: end with erode(level=0) for zero terrain offset.")
    if len(history["events"]) > 8:
        raise HistoryError("Use at most eight events for this demonstration.")
    if any(e["type"] == "intrusion" for e in history["events"]):
        raise HistoryError("This target has seven sedimentary packages; adding intrusive units is outside its scope.")
    return history


def render_result(history: dict, volume: bool = True, resolution: int = 96) -> dict:
    started = perf_counter()
    prediction = render_map(history, XS, YS, terrain=HEIGHTS)
    map_ms = (perf_counter() - started) * 1000
    score_start = perf_counter()
    metrics = SCORER(prediction)
    score_ms = (perf_counter() - score_start) * 1000
    payload: dict = {
        "id": uuid4().hex[:12], "scene_id": SCENE_ID, "baseline_id": BASELINE_ID,
        "program": history_to_program(history),
        "history": history, "metrics": metrics,
    }
    volume_ms = 0.0
    if volume:
        terrain, bounds, surface = volume_terrain(resolution)
        volume_start = perf_counter()
        labels = render_volume(history, bounds, resolution, terrain=terrain)
        volume_ms = (perf_counter() - volume_start) * 1000
        payload["volume"] = {"bounds": bounds, "shape": list(labels.shape), "surface": surface,
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
    scene_id: str | None = None
    baseline_id: str | None = None


class IterationRequest(BaseModel):
    program: str = Field(max_length=24000)
    previous: list[dict] = Field(default_factory=list, max_length=12)
    refine: bool = True
    scene_id: str | None = None
    baseline_id: str | None = None
    auto_context: dict | None = None


class AutoRequest(IterationRequest):
    max_iterations: int = Field(default=20, strict=True, ge=1, le=100)
    seed: int | None = Field(default=None, strict=True, ge=0, le=2**32 - 1)


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL, "reasoning_effort": EFFORT,
            "requested_service_tier": SERVICE_TIER,
            "scene_id": SCENE_ID, "baseline_id": BASELINE_ID,
            "api_key_available": bool(os.getenv("OPENAI_API_KEY")), "warmup_ms": WARMUP_MS}


@app.get("/api/bootstrap")
async def bootstrap():
    return {"target": META, "model": MODEL, "api_key_available": bool(os.getenv("OPENAI_API_KEY")),
            "requested_service_tier": SERVICE_TIER,
            "scene_id": SCENE_ID, "baseline_id": BASELINE_ID, "terrain": TERRAIN_META,
            "baseline_name": "Undeformed sedimentary layers",
            "baseline": await asyncio.to_thread(render_result, canonical(BASELINE))}


@app.get("/api/image/{name}")
def image_file(name: str):
    if name == "terrain":
        return Response(base64.b64decode(TERRAIN_IMAGE.split(",", 1)[1]), media_type="image/png")
    files = {"target": ROOT / "data/target.png", "source": ROOT / "data/target_source.png",
             "full": ROOT / META["sourceImage"], "mask": ROOT / "data/target_mask.png"}
    if name not in files:
        raise HTTPException(404, "Unknown image")
    return FileResponse(files[name])


@app.post("/api/render")
async def render(request: RenderRequest):
    check_scene(request.scene_id, request.baseline_id)
    try:
        history = canonical(request.program)
        return await asyncio.to_thread(render_result, history, request.volume, request.resolution)
    except HistoryError as exc:
        raise HTTPException(422, str(exc)) from None


@app.post("/api/refine")
async def refine(request: IterationRequest):
    check_scene(request.scene_id, request.baseline_id)
    try:
        history = canonical(request.program)
        result, _, evaluations, elapsed_ms = await asyncio.to_thread(
            refine_history, history, LABELS, XS, YS, MASK, budget=160, seconds=3.0,
            boundary_mask=BOUNDARY_MASK, terrain=HEIGHTS)
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
Map exposure does not uniquely determine the subsurface. The supplied fixed USGS elevation
grid is real topography, approximately registered from printed map coordinates. Its valleys
and ridges affect the exposed units. Extremely thin source units may remain unexplained.
The starting program has only undeformed strata: add missing deformation if the observed
outcrop geometry calls for it. A numerical optimizer cannot add a geological event for you.

DSL: one call per event, oldest to youngest, only literal keyword arguments, no imports,
assignments, loops, expressions, or function definitions. Begin with strata and end with
erode(level=0). This final event intersects rocks with the fixed measured terrain; level=0
means ZERO OFFSET to that terrain, not a horizontal plane. Units must remain
[1,2,3,4,5,6,7], oldest to youngest. Preserve all seven.
strata(levels=[six strictly increasing contact elevations in km], units=[1,2,3,4,5,6,7])
anticline(x=..., y=..., azimuth=..., uplift=..., dip_ne=..., dip_sw=..., hinge=...,
          nw_length=..., se_length=..., plunge_nw=..., plunge_se=...)
tilt(x=...,y=...,z=...,azimuth=...,angle=...)
fault(x=...,y=...,z=...,azimuth=...,dip=...,slip=...)
erode(level=0)
At most 8 events. Strata extend infinitely before final erosion. Effective VERTICAL contact
intervals are fitted, not measured normal bed thicknesses. Oldest unit occupies z below
the lowest level; youngest above the highest. Elevations are km relative to the terrain's
minimum elevation, provided as datum_m. Terrain lies at or above z=0. Raise fold uplift
enough to expose older units at the actual nonzero ground elevation. Do not move the terrain.

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
            "terrain": {"source": TERRAIN_META["source"], "datum_m": TERRAIN_META["datum_m"],
                        "min_z_km": float(HEIGHTS.min()), "max_z_km": float(HEIGHTS.max()),
                        "height_samples_5x5_north_up_km": HEIGHTS[np.ix_(np.linspace(0,len(YS)-1,5).astype(int), np.linspace(0,len(XS)-1,5).astype(int))].round(4).tolist()},
            "recent_attempts": [p for p in request.previous if p.get("scene_id") == SCENE_ID and p.get("baseline_id") == BASELINE_ID][-4:]}
    if request.auto_context is not None:
        info["auto"] = request.auto_context
    images = [
        ("Original map with legend; only the NW crop is scored:", file_url(ROOT / META["sourceImage"])),
        ("Clean observed NW target; transparent gaps are unobserved. Same bounds as prediction:", file_url(ROOT / "data/target.png")),
        ("Current predicted surface, same north-up frame and geological colors:", current["map_image"]),
        ("Disagreement overlay: red is mismatch; green is agreement; transparent is unobserved:", current["error_image"]),
        ("Fixed USGS terrain in the same frame: dark teal is low elevation, pale yellow is high; shading indicates relief:", TERRAIN_IMAGE),
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
            service_tier=SERVICE_TIER,
            instructions=SYSTEM_PROMPT, input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "geological_proposal", "strict": True, "schema": schema}})
    if response.status != "completed":
        raise RuntimeError("The model did not finish its proposal. Try the iteration again.")
    proposal = json.loads(response.output_text)
    proposal["usage"] = response.usage.model_dump() if response.usage else None
    proposal["response_id"] = response.id
    proposal["requested_service_tier"] = SERVICE_TIER
    proposal["service_tier"] = response.service_tier
    return proposal


def validate_iteration(request: IterationRequest) -> dict:
    check_scene(request.scene_id, request.baseline_id)
    try:
        return canonical(request.program)
    except HistoryError as exc:
        raise HTTPException(422, str(exc)) from None


async def claim_iteration():
    """Claim before returning a response; a second stream must never queue."""
    if ITERATION_LOCK.locked():
        raise HTTPException(409, "An iteration is already running.")
    await ITERATION_LOCK.acquire()
    released = False

    def release():
        nonlocal released
        if not released:
            released = True
            ITERATION_LOCK.release()

    return release


class IterationStream(StreamingResponse):
    """Close the generator and its lock even when a transport send fails."""

    def __init__(self, content, cleanup):
        self.cleanup = cleanup
        super().__init__(content, media_type="application/x-ndjson",
                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                self.cleanup()


def ndjson(kind: str, **values) -> str:
    return json.dumps({"type": kind, **values}) + "\n"


def proposal_error(exc: Exception) -> str:
    return (str(exc) if isinstance(exc, (HistoryError, RuntimeError)) else
            f"{type(exc).__name__}: proposal could not complete. Your current model is unchanged.")


def save_record(record: dict):
    run_dir = ROOT / "data/runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{record['result']['id']}.json").write_text(json.dumps(record))


async def attempt_events(request: IterationRequest, current: dict | None = None, *, search_seed: int | None = None):
    """One proposal and measured candidate; callers choose how to retain it."""
    started = perf_counter()
    original = canonical(request.program)
    yield {"type": "status", "stage": "observing", "message": "Reading the map and the current mismatch"}
    if current is None:
        current = await asyncio.to_thread(render_result, original, False)
    llm_start = perf_counter()
    proposal = await propose(request, current)
    llm_ms = (perf_counter() - llm_start) * 1000
    candidate = canonical(proposal["program"])
    yield {"type": "proposal", **proposal, "model_ms": llm_ms}
    yield {"type": "status", "stage": "testing", "message": "Generating the hypothesis and tuning its parameters"}
    evaluations, search_ms = 1, 0.0
    if request.refine:
        candidate, _, evaluations, search_ms = await asyncio.to_thread(
            refine_history, candidate, LABELS, XS, YS, MASK, budget=192, seconds=3.0,
            seed=len(request.previous) if search_seed is None else search_seed,
            allowed_fields=proposal["parameters_to_refine"][:10],
            boundary_mask=BOUNDARY_MASK, terrain=HEIGHTS)
    result = await asyncio.to_thread(render_result, candidate)
    accepted = result["metrics"]["score"] > current["metrics"]["score"] + 1e-6
    yield {"type": "result", "proposal": proposal, "result": result, "accepted": accepted,
           "scene_id": SCENE_ID, "baseline_id": BASELINE_ID,
           "before_program": history_to_program(original),
           "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
           "before": current["metrics"], "model_ms": llm_ms,
           "search": {"evaluations": evaluations, "elapsed_ms": search_ms},
           "elapsed_ms": (perf_counter() - started) * 1000, "model": MODEL,
           "requested_service_tier": proposal.get("requested_service_tier", SERVICE_TIER),
           "service_tier": proposal.get("service_tier")}


@app.post("/api/iterate")
async def iterate(request: IterationRequest):
    validate_iteration(request)
    release = await claim_iteration()

    async def events():
        try:
            async for item in attempt_events(request):
                if item["type"] == "result":
                    save_record({key: value for key, value in item.items() if key != "type"})
                yield json.dumps(item) + "\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield ndjson("error", message=proposal_error(exc))
        finally:
            release()

    return IterationStream(events(), release)


@app.post("/api/auto/{run_id}/stop")
async def stop_auto(run_id: str):
    state = AUTO_RUNS.get(run_id)
    if state is None:
        raise HTTPException(404, "This auto run is not active.")
    state["stop_requested"] = True
    return {"run_id": run_id, "stop_requested": True}


@app.post("/api/auto")
async def auto(request: AutoRequest):
    original = validate_iteration(request)
    release = await claim_iteration()
    run_id = uuid4().hex[:12]
    seed = randbits(32) if request.seed is None else request.seed
    state = {"stop_requested": False}
    AUTO_RUNS[run_id] = state

    def cleanup():
        AUTO_RUNS.pop(run_id, None)
        release()

    try:
        initial = await asyncio.to_thread(render_result, original)
    except BaseException:
        cleanup()
        raise

    async def events():
        global_best = branch_best = initial
        previous = list(request.previous)
        completed, restarts, branch_iteration, stalls = 0, 0, 0, 0
        branch = 1
        restart_reason = None
        reason = "limit"
        try:
            yield ndjson("started", run_id=run_id, max_iterations=request.max_iterations,
                         seed=seed, patience=AUTO_PATIENCE, restart_every=AUTO_RESTART_EVERY)
            while completed < request.max_iterations:
                if state["stop_requested"]:
                    reason = "stopped"
                    break
                if stalls >= AUTO_PATIENCE or branch_iteration >= AUTO_RESTART_EVERY:
                    restart_reason = "stalled" if stalls >= AUTO_PATIENCE else "branch_limit"
                    restarted, description = restart_history(
                        canonical(global_best["program"]), canonical(BASELINE), META["bounds"],
                        restart_index=restarts + 1, seed=seed)
                    # The generator is trusted code, but the public scene constraints still apply.
                    restarted = canonical(history_to_program(restarted))
                    branch_best = await asyncio.to_thread(render_result, restarted)
                    branch_best["restart"] = {"branch": branch + 1, "description": description}
                    global_improved = branch_best["metrics"]["score"] > global_best["metrics"]["score"] + 1e-6
                    if global_improved:
                        global_best = branch_best
                    restarts += 1
                    branch += 1
                    branch_iteration = stalls = 0
                    previous = []
                    yield ndjson("restart", branch=branch, reason=restart_reason,
                                 description=description, result=branch_best, global_best=global_best,
                                 global_improved=global_improved)
                    if state["stop_requested"]:
                        reason = "stopped"
                        break
                iteration = completed + 1
                metadata = {"run_id": run_id, "iteration": iteration,
                            "max_iterations": request.max_iterations, "seed": seed,
                            "branch": branch, "branch_iteration": branch_iteration + 1}
                context = {**metadata, "stall_count": stalls, "restart_reason": restart_reason,
                           "global_best": {"id": global_best["id"], "metrics": global_best["metrics"]},
                           "purpose": "Explore this branch; its starting score may be lower than the retained global best."}
                attempt = IterationRequest(
                    program=branch_best["program"], previous=previous[-12:], refine=request.refine,
                    scene_id=SCENE_ID, baseline_id=BASELINE_ID, auto_context=context)
                async for item in attempt_events(attempt, branch_best, search_seed=(seed + iteration) % 2**32):
                    if item["type"] != "result":
                        item["auto"] = metadata
                        if item["type"] == "status":
                            item["message"] = f"Iteration {iteration}/{request.max_iterations} · branch {branch}: {item['message']}"
                        yield json.dumps(item) + "\n"
                        continue
                    candidate = item["result"]
                    global_improved = candidate["metrics"]["score"] > global_best["metrics"]["score"] + 1e-6
                    if item["accepted"]:
                        branch_best = candidate
                    if global_improved:
                        global_best = candidate
                    stalls = 0 if global_improved else stalls + 1
                    completed += 1
                    branch_iteration += 1
                    record = {key: value for key, value in item.items() if key != "type"}
                    record["auto"] = {**metadata, "global_improved": global_improved,
                                      "global_best_id": global_best["id"]}
                    save_record(record)
                    previous.append({"scene_id": SCENE_ID, "baseline_id": BASELINE_ID,
                                     "headline": item["proposal"]["headline"],
                                     "observation": item["proposal"].get("observation", ""),
                                     "expected_effect": item["proposal"].get("expected_effect", ""),
                                     "program": candidate["program"], "accepted": item["accepted"],
                                     "global_improved": global_improved, "metrics": candidate["metrics"]})
                    yield ndjson("result", **record, global_best=global_best, branch_best=branch_best)
            if state["stop_requested"]:
                reason = "stopped"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = "error"
            yield ndjson("error", message=proposal_error(exc), run_id=run_id)
        finally:
            cleanup()
        yield ndjson("complete", reason=reason, run_id=run_id, completed_iterations=completed,
                     restarts=restarts, best=global_best)

    return IterationStream(events(), cleanup)


@app.get("/api/runs")
def runs():
    records = []
    for path in (ROOT / "data/runs").glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("scene_id") != SCENE_ID or record.get("baseline_id") != BASELINE_ID:
            continue
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
    record = json.loads(path.read_text())
    if record.get("scene_id") != SCENE_ID or record.get("baseline_id") != BASELINE_ID:
        raise HTTPException(409, "This recording belongs to an earlier terrain or starting history. Use the current scene's recordings.")
    return record


def _read_replay_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        raise HTTPException(404, f"{description} is missing.") from None
    except (OSError, ValueError):
        raise HTTPException(409, f"{description} could not be read as valid JSON.") from None
    if not isinstance(value, dict):
        raise HTTPException(409, f"{description} must be a JSON object.")
    return value


@app.get("/api/replay")
def replay_playlist():
    """One explicitly ordered investigation, separate from the full archive."""
    manifest = _read_replay_json(ROOT / "data/replay.json", "Replay playlist")
    if manifest.get("scene_id") != SCENE_ID or manifest.get("baseline_id") != BASELINE_ID:
        raise HTTPException(409, "Replay playlist belongs to another terrain or starting history.")
    run_ids = manifest.get("run_ids")
    title = manifest.get("title")
    if (manifest.get("version") != 1 or not isinstance(title, str) or not title.strip()
            or not isinstance(run_ids, list) or not run_ids
            or any(not isinstance(rid, str) or len(rid) != 12
                   or any(c not in "0123456789abcdef" for c in rid) for rid in run_ids)
            or len(set(run_ids)) != len(run_ids)):
        raise HTTPException(409, "Replay playlist has an invalid title, version, or ordered record list.")
    incumbent = canonical(BASELINE)
    summaries = []
    for run_id in run_ids:
        record = _read_replay_json(ROOT / "data/runs" / f"{run_id}.json", f"Replay record {run_id}")
        result = record.get("result")
        if (record.get("scene_id") != SCENE_ID or record.get("baseline_id") != BASELINE_ID
                or not isinstance(result, dict) or result.get("scene_id") != SCENE_ID
                or result.get("baseline_id") != BASELINE_ID):
            raise HTTPException(409, f"Replay record {run_id} belongs to another terrain or starting history.")
        try:
            before = canonical(record["before_program"])
            candidate = canonical(result["program"])
            accepted = record["accepted"]
            headline = record["proposal"]["headline"]
            miou = result["metrics"]["miou"]
            if (result.get("id") != run_id or not isinstance(accepted, bool)
                    or not isinstance(headline, str) or isinstance(miou, bool)
                    or not isinstance(miou, (int, float)) or not 0 <= miou <= 1):
                raise ValueError("Invalid recording fields")
        except (KeyError, TypeError, ValueError):
            raise HTTPException(409, f"Replay record {run_id} has invalid or incomplete history data.") from None
        if before != incumbent:
            raise HTTPException(409, f"Replay record {run_id} does not continue the playlist's accepted history.")
        summaries.append({"id": run_id, "headline": headline, "accepted": accepted, "miou": miou})
        if accepted:
            incumbent = candidate
    return {"title": title, "scene_id": SCENE_ID, "baseline_id": BASELINE_ID, "runs": summaries}
