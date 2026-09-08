# Farallon

GPT-6 reads a geological map, writes an executable geological history, and tests the resulting 3D world against the observed outcrops. It starts with undeformed sedimentary layers and must add missing deformation. The demo uses the northwest nose of Sheep Mountain, Wyoming, from the supplied Fiore Allwardt et al. (2007) map, intersected with real USGS 3DEP topography.

## Run locally

Dependencies are already installed on this laptop. From this directory:

```sh
./run.sh
```

Open **http://127.0.0.1:3000**. Ctrl+C stops both services. If the app is already running, just open the link. Ports 3000 and 8000 must otherwise be free.

For a fresh installation, use Node 22.13+ and uv:

```sh
uv sync
npm --prefix web ci
```

The backend reads `OPENAI_API_KEY` from the environment or a root `.env` file. See `.env.example`; do not place a key in the frontend. Defaults: `OPENAI_MODEL=gpt-6-astra`, `OPENAI_REASONING_EFFORT=low`. Restart after changing environment settings. Without a key, manual simulation and saved replay still work. Each investigation button click makes one paid API call; no model calls run automatically on page load.

## Present

See [DEMO.md](DEMO.md) for a three-minute script. Start with **Let GPT-6 investigate**. While it reasons, orbit the model or move the cutaway. Open **History** to show the executable program. The **Mismatch** tab highlights disagreement; the same fixed observation mask is used for every model. The predicted map is shown in full, so gaps in the source cannot masquerade as predicted geological contacts.

Three genuine GPT-6 terrain rehearsal iterations are included under `data/runs/`, with recorded timestamps that preserve their order after checkout. **Replay next** displays these recorded programs, images, volumes, timings and measurements, with a visible replay label. It makes no API call. **Reset** returns to the undeformed baseline. New live results are saved automatically and ignored by Git; refreshing clears the displayed session but preserves recordings. Three earlier flat-surface recordings remain archived, but are excluded from this scene. Scene and baseline IDs prevent mixing incompatible measurements.

The **Terrain** tab shows the measured elevations. **Vertical 1× / 2×** changes display exaggeration only; it does not change the terrain supplied to the simulator or any scores. The cutaway follows the measured ground surface. See [terrain provenance](data/TERRAIN.md) for calibration, source data and offline reproduction.

## What runs

- **`geology/history.py`** validates a small event language: strata, anticline, tilt, fault, intrusion and erosion. The Sheep Mountain endpoint restricts it to the seven observed sedimentary packages. Programs are parsed as constrained syntax, never passed to Python `exec`.
- **`geology/engine.py`** classifies points through inverse analytic event transformations in one Numba kernel. A 256² surface and a 96³ categorical volume are generated independently. No Gaussian processes or mesh reconstruction are used.
- **`geology/terrain.py`** bilinearly samples a fixed cached 256² USGS elevation grid. The 416.10 m relief is retained directly, without fitting a parametric approximation. Terrain samples and 97² rendering vertices are cached per volume resolution. Final `erode(level=0)` means zero offset to this measured terrain, not a horizontal plane.
- **`geology/scoring.py`** measures mean unit IoU and matched contact-pair distance. Acceptance uses `0.8 × mIoU + 0.2 × exp(-contact_error / scale)`. The categorical mask is fixed; narrow uncertain line gaps are bridged only for the approximate boundary metric.
- **`geology/search.py`** tunes numerical parameters within each model's proposed structure, with a 192-evaluation / 3-second maximum in the live loop. GPT chooses the history revision and tunable fields; numerical search tests parameter values. Final acceptance compares full-resolution measurements with the incumbent.
- **`server.py`** keeps credentials local and uses the Responses API with five images: original map, extracted target, current prediction, mismatch and measured terrain. It streams status, the structured proposal and the measured result to the UI. Rejected proposals are recorded; the best history is retained.
- **`web/`** contains the React/Sites interface and a Three.js viewer. A shaded terrain mesh uses the exact geological surface prediction as its texture. The cut walls share the terrain mesh's edge coordinates and show actual volume samples. At partial top voxels, wall colors extend the nearest sampled rock label to seal the voxel-scale seam. The app runs locally, with `/api` proxied to Python. `npm run build` validates a frontend build; this demo uses the local development launch path.

## Verified rehearsal

| Stage | Candidate mean unit overlap | Contact error | Model proposal time | Decision |
|---|---:|---:|---:|---|
| Undeformed layers | 3.11% | No predicted contacts | — | Starting hypothesis |
| GPT-6 adds an anticline | 29.38% | 18.58 px | 19.27 s | Accepted |
| GPT-6 revises the upper contact using terrain | 42.71% | 22.46 px | 13.13 s | Accepted |
| GPT-6 adjusts the older core | 42.88% | 23.70 px | 13.12 s | Rejected: combined fit worsened |

The retained best has **42.71% mean unit overlap** and **80.33% pixel agreement** within the observed mask. The third proposal's small overlap gain did not offset its contact-error penalty, so the second remains best. From that best model, removing plunge reduces mean overlap to **30.51%**; making the limbs symmetric reduces it to **32.92%**. These are direct feature tests, without reoptimization.

Warm map plus 96³ terrain-clipped volume generation measured **7.32–7.39 ms**, full Python render payload **27.3–28.1 ms**, and each parameter search **0.79–0.90 s**. These exclude HTTP transport and browser rendering. The three genuine proposals and their simulations took **48.17 seconds** in this rehearsal; API times vary. The initial boundary-error value is a missing-contact penalty, not a measured contact displacement, and the UI displays “No predicted contacts.”

Run `.venv/bin/python scripts/benchmark.py --history fold` for a reproducible warm benchmark with a nontrivial fold, or omit `--history fold` to time the undeformed baseline. Add `--json` for machine-readable timings and environment details. The benchmark fold is a timing fixture, not a recovered or preloaded solution.

To run another bounded rehearsal (makes paid API calls):

```sh
.venv/bin/python scripts/rehearse.py --calls 3
```

## Checks

```sh
.venv/bin/python -m pytest -q
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run build
```

Application lint excludes unmodified generated Shadcn primitives and the starter mobile hook. Type checking covers the project. Python tests cover history validation, geological transformations, array orientation, fixed masks, scoring, search, acceptance/rejection, streaming failures and replay; tests mock the model and make no paid calls.

## Interpretation limits

This is a schematic inverse-modeling demonstration, not a recovered unique history. Topography comes from the official USGS 3DEP service, approximately registered using the figure's printed geographic ticks. The 6.059 × 4.139 km extent supersedes the initial rough scale-bar estimate. The source figure's datum and positional accuracy are unresolved; tick-fit residuals are not an external accuracy estimate. Cached elevations are used relative to their minimum, 1129.736 m. The source is a compressed raster; 66.5% of the crop is retained as labeled observations, with Quaternary cover and uncertain pixels excluded. Thin Madison and Amsden outcrops are poorly resolved.

Analytic fold shear fits effective vertical intervals, not preserved normal bed thicknesses. Multiple additive anticline events commute, so changing only their order cannot establish event timing. Fold-axis arrows are not surface faults. A good map fit constrains exposure patterns more strongly than subsurface geometry.

Source: [Fiore Allwardt et al. (2007), Geosphere, Figure 1](https://doi.org/10.1130/GES00088.1). The supplied source figure is unchanged; crop, palette, mask, approximate scale and provenance are recorded in `data/target.json`. See [RESEARCH.md](RESEARCH.md) for the design research.
