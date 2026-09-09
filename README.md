# Farallon

GPT-6 reads a geological map, writes an executable geological history, and tests the resulting 3D world against the observed outcrops. It starts with undeformed sedimentary layers and can add deformation and young Quaternary deposits. All eight mapped units, including yellow cover, contribute to the fit. The demo uses the northwest nose of Sheep Mountain, Wyoming, from the supplied Fiore Allwardt et al. (2007) map, intersected with real USGS 3DEP topography.

<img width="1430" height="680" alt="Screenshot 2026-09-08 at 4 00 43 PM" src="https://github.com/user-attachments/assets/e97c5f26-9eae-43b9-844f-89bcfcc67ab0" />


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

The backend reads `OPENAI_API_KEY` from the environment or a root `.env` file. See `.env.example`; do not place a key in the frontend. Without a key, manual simulation and saved replay still work. **Test next hypothesis** makes one paid proposal call. **Start auto** runs up to the chosen number of iterations, with one paid proposal per iteration. No model calls run automatically on page load or when the dropdown changes.

The **Model** dropdown offers four fixed presets, ordered from speed to capability:

| Preset | Model | Reasoning | Processing |
|---|---|---|---|
| Speed | GPT-5.6 Luna | None | Fast |
| Balanced | GPT-5.6 Terra | Low | Fast |
| Strong | GPT-5.6 Sol | Low | Fast |
| Strongest (default) | GPT-6 Astra | Low | Fast |

The selected preset applies to both single iterations and the entire next auto run, including restarts. You can switch models between runs while retaining the best geological history. The dropdown is locked during running work and replay; returning to live preserves your selection. Run labels retain the model that actually produced each proposal. This ordering describes preset intent, not a measured latency guarantee for this map.

All four UI presets request Fast mode while preserving the five images and output schema. Scripts and API requests that omit `model_preset` retain the environment defaults: `OPENAI_MODEL=gpt-6-astra`, `OPENAI_REASONING_EFFORT=low`, and `OPENAI_SERVICE_TIER=auto` (set to `fast` in this laptop's ignored `.env`). For those callers, set the service tier to `default` for Standard processing, or `auto` to follow the API project's setting, then restart. Health/bootstrap report these legacy defaults and bootstrap separately lists the UI presets. New recordings capture the selected preset, effective model and reasoning effort, plus requested and actual service tiers, including any downgrade. GPT-6 Astra Fast mode uses 2× the applicable Standard token rates, has no latency SLA, and is unavailable with EU data residency. See the [official Fast mode guide](https://developers.openai.com/api/docs/guides/fast-mode) and [Astra pricing](https://developers.openai.com/api/docs/models/gpt-6-astra). The archived seven-unit rehearsal predates this setting.

The UI distinguishes **Fast requested** from the response's **Fast confirmed**, and reports proposal and numerical tuning times separately. A September 8 local audit found all 23 instrumented recordings actually received Fast processing. Four identical-input diagnostic calls averaged **16.10 s Standard / 11.42 s Fast**, about **1.41×** for full completion; pair ratios were 1.10× and 1.96×. Reasoning/output lengths differed, so this small test is not a speed guarantee. The guide's “up to 2.5×” explanation concerns GPT-5.6 Sol, not an Astra end-to-end guarantee. See [the measured results](data/api_latency_audit.json). `scripts/benchmark_api_latency.py --dry-run` checks the request without paid calls; omitting `--dry-run` makes at most four paid calls and should only be done while the interactive demo is idle.

The fixed source map, observed crop, terrain and reference metadata now precede an explicit 30-minute cache breakpoint; each changing history, prediction and mismatch follows it. Two real validation calls both completed on Fast, and the second reused **3,935 of 4,917 input tokens** despite the changed geology. Their 13.23 s and 12.84 s response times do not establish an additional speedup. [Validation metadata](data/api_cache_validation.json) and the [official caching guide](https://developers.openai.com/api/docs/guides/prompt-caching) document the change. No model settings or images were removed to obtain cache reuse.

## Present

See [DEMO.md](DEMO.md) for a three-minute script. Start with **Test next hypothesis**. While it reasons, orbit the model or move the cutaway. Open **History** to show the executable program. The **Mismatch** tab highlights disagreement; the same fixed observation mask is used for every model. The predicted map is shown in full, so gaps in the source cannot masquerade as predicted geological contacts.

Three genuine GPT-6 iterations fitting bedrock and Quaternary are included under `data/runs/`. The explicit sequence in `data/replay.json` keeps this rehearsal separate from later investigations. **Replay next** displays the recorded programs, images, volumes, timings and measurements, with a visible replay label and step count. It stops after step three; it never wraps or appends unrelated live runs. Replay makes no model call. Your live investigation is preserved while replay is open; **Return to live** restores it. During replay, **Reset** restarts the rehearsal from undeformed layers; otherwise it resets the live investigation. New live results are saved automatically and ignored by Git; refreshing clears the displayed session but preserves recordings. Three earlier flat-surface and three bedrock-only terrain recordings remain archived, but are excluded from this scene; the previous playlist is preserved in `data/replay-bedrock-v1.json`. Scene and baseline IDs prevent mixing incompatible measurements.

The **Terrain** tab shows the measured elevations. **Vertical 1× / 2×** changes display exaggeration only; it does not change the terrain supplied to the simulator or any scores. The cutaway follows the measured ground surface. See [terrain provenance](data/TERRAIN.md) for calibration, source data and offline reproduction.

## Automatic investigation

Choose a **Model**, set **Iterations** to an integer from **1–100** (default **20**), then click **Start auto**. It begins from the current best live geology and makes sequential proposal → simulation → measurement calls. The server enforces the 100-iteration cap and retains the chosen model preset across every branch and restart. In replay, first click **Return to live**.

The run keeps a best model for the current branch and a separate best overall, both ranked by the combined fit score below. It restarts after **four consecutive attempts without an overall improvement**, or **ten attempts in one branch**. Odd-numbered restarts use fresh undeformed strata with varied layer intervals and elevation; even-numbered restarts broadly perturb the overall best's existing geometry, deposits and strata, falling back to undeformed strata when needed. No prebuilt fold or cover footprint is inserted. All branches retain bedrock IDs 1–7, optional deposit ID 8, the eight-unit observation mask and measured terrain.

While running, the viewer shows the current branch, which can score below the retained overall best. The progress line reports completed iterations, restarts and best-overall overlap. Completion, **Stop auto**, or an error returns the display to the best overall model received.

**Stop auto** finishes an in-flight iteration, then starts no further proposal. Stopping during startup aborts the request. Closing or refreshing the page cancels the run; it does not resume automatically. Errors stop the run without an automatic paid retry. Completed proposals remain in `data/runs/` with auto-run, branch and iteration provenance. They do not alter the original three-step replay sequence. The displayed session is not restored automatically after a refresh.

## What runs

- **`geology/history.py`** validates a small event language: strata, anticline, syncline, tilt, fault, deposit, intrusion and erosion. The Sheep Mountain endpoint permits seven bedrock packages plus yellow Quaternary deposits (unit 8), with at most eight events. Programs are parsed as constrained syntax, never passed to Python `exec`. Syncline shares anticline's parameters; `uplift` is a nonnegative displacement magnitude, applied downward for a syncline.
- **`geology/engine.py`** classifies points through inverse analytic event transformations in one Numba kernel. A 256² surface and a 96³ categorical volume are generated independently. No Gaussian processes or mesh reconstruction are used.
- **`geology/terrain.py`** bilinearly samples a fixed cached 256² USGS elevation grid. The 416.10 m relief is retained directly, without fitting a parametric approximation. Terrain samples and 97² rendering vertices are cached per volume resolution. Final `erode(level=0)` means zero offset to this measured terrain, not a horizontal plane.
- **`geology/scoring.py`** measures mean unit IoU and matched contact-pair distance. Acceptance uses `0.8 × mIoU + 0.2 × exp(-contact_error / scale)`. The categorical mask is fixed; narrow uncertain line gaps are bridged only for the approximate boundary metric.
- **`geology/search.py`** tunes numerical parameters within each model's proposed structure, with a 192-evaluation / 3-second maximum in the live loop. GPT chooses the history revision and tunable fields; numerical search tests parameter values. Final acceptance compares full-resolution measurements with the incumbent.
- **`server.py`** keeps credentials local and uses the Responses API with five images: original map, extracted target, current prediction, mismatch and measured terrain. It streams status, the structured proposal and the measured result to the UI. Rejected proposals are recorded; the best history is retained.
- **`geology/presentation.py`** draws source-style maps independently of scoring. The Units view uses exact legend colors at all eight classified observations, retaining the supplied map's dark linework elsewhere. No missing bedrock is guessed. Yellow is generated by deposition and scored against classified yellow observations. Prediction contacts are thin continuous black lines, preserving narrow units. The raw categorical texture used by the simulator and 3D viewer remains unchanged; compatible recordings receive current cartography without rewriting their scores or timings.
- **`geology/cartography.py`** marks actual model anticline/syncline event axes with outward/inward transverse arrows. Dashed traces distinguish inferred construction axes from source ink. Axes clip to active fold extents and map bounds; older axes affected by subsequent tilt/fault are omitted with a diagnostic. No syncline is invented in an anticline-only history. These are event axes, not hinges extracted from the combined deformed geology. Arrow conventions follow [USGS](https://pubs.usgs.gov/of/2015/1041/pdf/ofr2015-1041_sheet8.pdf).
- **`web/`** contains the React/Sites interface and a Three.js viewer. A shaded terrain mesh uses the exact geological surface prediction as its texture. The cut walls share the terrain mesh's edge coordinates and show actual volume samples. At partial top voxels, wall colors extend the nearest sampled rock label to seal the voxel-scale seam. The app runs locally, with `/api` proxied to Python. `npm run build` validates a frontend build; this demo uses the local development launch path.

## Quaternary deposition

The model may add one or more calls such as `deposit(unit=8, x=3, y=2, azimuth=135, base=-0.1, curvature_cross=1, curvature_along=0, slope=0)`. This example is a geometry illustration, not a fitted or preloaded solution. Each deposit replaces older material above a quadratic basal surface; the fixed terrain clips its top. Cross-axis curvature creates a valley strip, along-axis curvature closes a basin, and a slope tilts the base. Multiple calls can form separate lobes of the same unit. Parameters are in km, degrees, km⁻¹ curvature and dimensionless slope; no map mask is painted into the simulator.

A syncline before deposition bends the older bedrock and may be buried; one after deposition also bends the young deposit. Dashed construction axes can continue beneath cover, with an explicit diagnostic. The mapped axis across Quaternary cover does not establish that the young deposits themselves were folded. A fully hidden syncline may leave the scored surface unchanged, so unit/contact fit alone cannot favor that interpretation. This is a simplified valley/basin fill, not a sediment-transport model.

## Current eight-unit rehearsal

Three real GPT-6 Astra Fast proposals, from the unchanged flat bedrock baseline:

| Stage | Mean unit overlap | Yellow overlap | Contact error | Decision |
|---|---:|---:|---:|---|
| Undeformed bedrock | 2.33% | 0% | No predicted contacts | Starting hypothesis |
| Proposal 1 | 39.60% | 58.11% | 18.96 px | Accepted |
| Proposal 2 | 50.27% | 67.03% | 13.00 px | Accepted |
| Proposal 3 | 50.58% | 67.03% | 12.99 px | Accepted |

The final model agrees with 77.15% of observed pixels. All three responses confirmed Fast processing, with proposal times 16.13, 14.35 and 11.08 seconds. Map plus 96³ volume generation took 8.77–11.44 ms; numerical tuning took 0.70–0.88 seconds. These are one rehearsal's measurements, not latency guarantees. No target footprint was inserted into the history. Replay uses these original programs, scores, volumes and timings.

## Archived seven-unit rehearsal

The following measurements belong to the earlier scene that excluded Quaternary. Its three recordings remain unchanged; these scores are not comparable with the current eight-unit fit.

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
npm --prefix web run test:stream
npm --prefix web run build
```

Application lint excludes unmodified generated Shadcn primitives and the starter mobile hook. Type checking covers the project. `test:stream` runs the stream reader tests with Node's `--experimental-strip-types`. Python tests cover history validation, geological transformations, array orientation, fixed masks, scoring, search, acceptance/rejection, streaming failures and replay; tests mock the model and make no paid calls.

## Interpretation limits

This is a schematic inverse-modeling demonstration, not a recovered unique history. Topography comes from the official USGS 3DEP service, approximately registered using the figure's printed geographic ticks. The 6.059 × 4.139 km extent supersedes the initial rough scale-bar estimate. The source figure's datum and positional accuracy are unresolved; tick-fit residuals are not an external accuracy estimate. Cached elevations are used relative to their minimum, 1129.736 m. The source is a compressed raster; 50,882 pixels (77.64% of the crop) are labeled observations, including 7,290 yellow Quaternary pixels. Outside and uncertain pixels remain excluded. Thin Madison and Amsden outcrops are poorly resolved.

Analytic fold shear fits effective vertical intervals, not preserved normal bed thicknesses. Multiple additive anticline/syncline events commute, so changing only their order cannot establish event timing. Fold-axis arrows are not surface faults. A good map fit constrains exposure patterns more strongly than subsurface geometry.

Source: [Fiore Allwardt et al. (2007), Geosphere, Figure 1](https://doi.org/10.1130/GES00088.1). The supplied source figure is unchanged; crop, palette, mask, approximate scale and provenance are recorded in `data/target.json`. See [RESEARCH.md](RESEARCH.md) for the design research.

The [palette re-export and audit](data/palette_audit.json) samples all eight supplied WEBP legend interiors: every RGB equals its source median exactly, including Quaternary `#fbf08d`. The PDF legend differs by only 1–2 RGB channel values from the compressed WEBP. `scripts/audit_palette.py` reproduces the sampling and compares three gap/contact treatments using an explicitly archived prediction for display only. The 3D view applies terrain lighting to these same colors, so shaded rock faces appear darker than flat map swatches.
