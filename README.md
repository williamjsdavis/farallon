# Farallon

GPT-6 reads a geological map, writes an executable geological history, and tests the resulting 3D world against the observed outcrops. The demo uses the northwest nose of Sheep Mountain, Wyoming, from the supplied Fiore Allwardt et al. (2007) map.

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

See [DEMO.md](DEMO.md) for a three-minute script. Start with **Let GPT-6 investigate**. While it reasons, orbit the model or move the cutaway. Open **History** to show the executable program. The **Mismatch** tab highlights disagreement; the same fixed observation mask is used for every model.

Three genuine GPT-6 rehearsal iterations are included in the repository under `data/runs/`, with recorded timestamps that preserve their order after checkout. **Replay next** displays these recorded programs, images, volumes, timings and measurements, with a visible replay label. It makes no API call. **Reset** returns to the original baseline. New live results are saved automatically and ignored by Git; refreshing clears the displayed session but preserves recordings.

## What runs

- **`geology/history.py`** validates a small event language: strata, anticline, tilt, fault, intrusion and erosion. The Sheep Mountain endpoint restricts it to the seven observed sedimentary packages. Programs are parsed as constrained syntax, never passed to Python `exec`.
- **`geology/engine.py`** classifies points through inverse analytic event transformations in one Numba kernel. A 256² surface and a 96³ categorical volume are generated independently. No Gaussian processes or mesh reconstruction are used.
- **`geology/scoring.py`** measures mean unit IoU and matched contact-pair distance. Acceptance uses `0.8 × mIoU + 0.2 × exp(-contact_error / scale)`. The categorical mask is fixed; narrow uncertain line gaps are bridged only for the approximate boundary metric.
- **`geology/search.py`** tunes numerical parameters within each model's proposed structure, with a 192-evaluation / 3-second maximum in the live loop. GPT chooses the history revision and tunable fields; numerical search tests parameter values. Final acceptance compares full-resolution measurements with the incumbent.
- **`server.py`** keeps credentials local and uses the Responses API with four images: original map, extracted target, current prediction and mismatch. It streams status, the structured proposal and the measured result to the UI. Rejected proposals are recorded; the best history is retained.
- **`web/`** contains the React/Sites interface and a Three.js viewer. The block's top uses the exact surface image; its sides and movable cut use actual volume samples. The app runs locally, with `/api` proxied to Python. `npm run build` validates a frontend build; this demo uses the local development launch path, not a public cloud deployment.

## Verified rehearsal

| Stage | Mean unit overlap | Contact error | Model proposal time |
|---|---:|---:|---:|
| Starting hypothesis | 4.17% | 53.50 px | — |
| GPT-6 iteration 1 | 29.91% | 23.77 px | 28.55 s |
| GPT-6 iteration 2 | 30.26% | 21.93 px | 22.90 s |
| GPT-6 iteration 3 | 33.63% | 19.70 px | 18.33 s |

Warm map plus 96³ volume generation was approximately **9 ms**, full Python render payload approximately **27 ms**, and each parameter search approximately **0.7 s**. A separate five-request check through the UI proxy measured **34.75 ms median / 37.45 ms maximum** for rendering, HTTP transport and client JSON decoding. These timings exclude browser rendering. API times are measurements of this rehearsal, not latency guarantees. The three real iterations completed in approximately 72 seconds.

Run `.venv/bin/python scripts/benchmark.py` for a reproducible warm engine benchmark, or add `--json` for machine-readable timings and environment details.

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

This is a schematic inverse-modeling demonstration, not a recovered unique history. Terrain is a flat erosion surface. The source is a compressed raster; 66.5% of the crop is retained as labeled observations, with Quaternary cover and uncertain pixels excluded. Thin Madison and Amsden outcrops are poorly resolved. The final rehearsal still does not expose Madison, despite that being the model's stated intent in its third proposal.

Analytic fold shear fits effective vertical intervals, not preserved normal bed thicknesses. Multiple additive anticline events commute, so changing only their order cannot establish event timing. Fold-axis arrows are not surface faults. A good map fit constrains exposure patterns more strongly than subsurface geometry.

Source: [Fiore Allwardt et al. (2007), Geosphere, Figure 1](https://doi.org/10.1130/GES00088.1). The supplied source figure is unchanged; crop, palette, mask, approximate scale and provenance are recorded in `data/target.json`. See [RESEARCH.md](RESEARCH.md) for the design research.
