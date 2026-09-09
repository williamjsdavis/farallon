# Farallon: three-minute presenter aid

Open **http://localhost:3000**. The story is **visual evidence → executable history → 3D simulation → measurement → revised explanation**.

## Before going on stage

- Click **Reset**. Confirm the starting history has only `strata(...)` and `erode(level=0)`: seven horizontal packages, **no fold**. Its surface prediction is Cretaceous throughout.
- Leave the map on **Source**, the lower panel on **Reasoning**, cutaway near zero and **Vertical 1×**. Confirm **Replay next (1/3)** is available for the current terrain scene.
- **Terrain** shows the actual cached USGS DEM: **416.1 m relief** across **6.059 × 4.139 km**. Printed map ticks provide approximate registration; details are in [data/TERRAIN.md](data/TERRAIN.md).
- Optional full-map context: `http://localhost:3000/api/image/full`. **Source** shows the original crop; **Units** shows the fixed classified observations.
- **Generate**, feature tests and 3D controls use local computation. They do not make another model API call.
- Choose single-step investigation or the optional Auto sequence below. Auto starts only after **Start auto** is clicked; it never launches calls on page load.

## Talking script and clicks

| Time | Click / show | Say |
| --- | --- | --- |
| 0:00–0:20 | **Source**, then **Terrain**; point to the fold nose. | “This is a geological map of Sheep Mountain, Wyoming, paired with real terrain. Can GPT-6 write a geological history whose exposed rocks explain these colored bands?” |
| 0:20–0:45 | Click **Let GPT-6 investigate**. Show **History**, then **Units**; gently orbit. | “We start with horizontal sedimentary layers and no fold. GPT sees the map, terrain, prediction and mismatch. It must propose the missing geological events and write the executable history.” |
| 0:45–1:10 | Read the first actual result; start the second investigation. Show the new history. | “In our rehearsal it added an asymmetric, plunging anticline. That changes the kind of model. A separate local search tunes the parameters, and the simulator measures the resulting outcrops.” |
| 1:10–1:35 | Start a third investigation when ready. Show **Terrain**, **Mismatch** and the generation timer. | “The terrain stays fixed. The second proposal used the lower northeastern ground to expose the Jurassic package. The warm volume and surface simulation takes about seven milliseconds here; model reasoning takes seconds.” |
| 1:35–1:55 | Show the investigation rows; click **Restore best** if visible. | “Our rehearsal improved mean unit overlap from 3.1% to 42.7%. A third proposal slightly improved overlap but worsened the contact score, so it was rejected. The measurement can disagree with the hypothesis.” |
| 1:55–2:20 | On the retained best, click **Remove plunge**, compare the northwest nose and score, then **Restore best**. | “We can remove one feature and regenerate immediately. In this rehearsed model, removing plunge drops overlap from 42.7% to 30.5%. That feature helps explain the map.” |
| 2:20–2:40 | Move **Cut into the mountain** to about 35%; orbit; toggle **Vertical 2×**, then return to 1×. | “The terrain surface and underground layers come from the same 3D model. This toggle exaggerates height only for viewing; the geology and score are unchanged.” |
| 2:40–3:00 | End on the best block, **Units** and **History**. | “This is a possible history, not a unique reconstruction. The figure is only approximately registered, and considerable mismatch remains. GPT connects visual interpretation to executable hypotheses that we can inspect, measure and challenge.” |

Live hypotheses and timing vary: describe the result actually shown. Read **Expected** as a prediction, not an accomplished effect. Start the next call promptly; if running behind, stop after two proposals and preserve time for the feature test and cutaway.

## Optional Auto sequence

For the same three-minute structure, set **Iterations** to **3** and click **Start auto** at 0:20. Skip the manual second and third investigation clicks: Auto runs them sequentially from the current best. Continue narrating the evidence, history and measured results. To end early, click **Stop auto** and wait for the current iteration to finish before the feature test or replay. The best overall model is restored when the run completes, stops or errors. Auto uses the current Fast/service-tier setting; the older rehearsal timings below are not an Auto performance guarantee.

For a longer investigation, the control accepts **1–100 iterations**, defaults to **20**, and the server enforces the cap. Each iteration makes one paid proposal call. A three-iteration run shows the automatic sequence; use a larger limit to show restarts. After four attempts without an overall combined-score improvement, or ten attempts in one branch, it starts a new branch. Odd restarts use fresh undeformed layers; even restarts broadly vary the best history already found, with an undeformed fallback. It does not insert a prebuilt fold. Seven rock packages and the terrain stay fixed.

Say: **“The viewer follows the current branch, while we retain the best explanation found across every branch. A restart lets us explore another starting point without losing that best model.”** Branch acceptance and overall improvement use the combined score, not overlap alone.

Stopping before startup completes aborts the request. Closing or refreshing cancels the run; errors stop it without automatic paid retries. Completed proposals are saved with Auto provenance, but there is no automatic session resume. The original three-record replay stays unchanged. Auto is available in the live investigation; use **Return to live** first if replay is open.

## Transparent replay fallback

Say: **“I’ll switch to our saved rehearsal. These are genuine GPT-6 proposals and their recorded simulations; this sequence is replay.”**

When idle, click **Replay next** three times. The app identifies replay visibly and shows the step count. It plays only the explicit three-step rehearsal in `data/replay.json`, with the current terrain, starting history and accepted-model continuity checked before playback. Other investigations never enter this sequence. After step three, **Replay complete** is disabled; there is no automatic wraparound. **Reset** restarts the rehearsal from undeformed layers. **Return to live** restores the investigation you had open before replay, including the selected model and history editor. Original flat-surface experiments remain archived and are not comparable or offered for this scene. A busy request must finish or error before replay is available.

Replayed timing is recorded timing, not a fresh benchmark. Orbit and cutaway stay interactive; feature tests perform new local renders. The third recorded candidate is rejected, leaving the second model as best. Replay displays the candidate for inspection, so click **Restore best** after the third replay before the feature demonstration.

## Verified terrain rehearsal

All three records use scene `sheep-023fcbb758811c9a`, baseline `9a229952fd019ab7`, and were recorded September 8, 2026. Overlap below is each **candidate's** macro IoU; the retained best ends at **42.71%**.

| Stage / record | Recorded UTC | Overlap | Contact error | Decision | Iteration time |
| --- | --- | ---: | ---: | --- | ---: |
| Undeformed starting strata | — | 3.11% | No predicted contacts | Starting model | — |
| Add asymmetric plunging anticline · `256e033e51d6` | 21:39:08.734 | 29.38% | 18.58 px | Accept | 20.10 s |
| Expose Jurassic on lower NE terrain · `33e081b4fb01` | 21:39:22.838 | **42.71%** | **22.46 px** | **Retained best** | 14.08 s |
| Restrict the Madison core · `e7837223f163` | 21:39:36.842 | 42.88% | 23.70 px | Reject | 13.98 s |

Three proposals totalled **48.17 s**. Model calls took **19.27 / 13.13 / 13.12 s**; each used **192 local parameter trials**, taking 0.79–0.90 s. Recorded warm generation of the **96³ volume plus 256² terrain-intersection map** took **7.32–7.39 ms**. That excludes scoring, encoding, API latency and browser rendering; full server render/score/encode took 27–28 ms. Final-best observed-pixel accuracy was 80.33%, a different metric from 42.71% mean unit overlap.

## Feature test

Use **Remove plunge on retained best `33e081b4fb01`** for the stage demonstration. These measurements reproduce each UI edit independently from that model, without parameter refitting or model API calls.

| Edit | Mean unit overlap | Contact error | Observed pixels changed |
| --- | ---: | ---: | ---: |
| Original retained best | 42.71% | 22.46 px | — |
| **Remove plunge** | **30.51%** | **31.99 px** | **18.54%** |
| Make symmetric | 32.92% | 21.81 px | 32.84% |

Removing plunge worsens both reported fit measures. The symmetry test changes more pixels and reduces overlap, but slightly improves contact error; its combined score still worsens. Each button starts from the same original parent history: clicking **Make symmetric** after **Remove plunge** does **not** stack both edits. **Restore best** returns to the accepted best model. These tests establish a feature's role in this fitted model, not exact parameters for the real mountain.

## Facts to keep straight

- GPT writes a restricted executable geological DSL using existing procedural operators; it does not invent a numerical operator or use Lemmalog. Local parameter optimization is separate from its hypothesis proposal.
- The measured DEM is a fixed observation surface. `erode(level=0)` means **zero offset from that terrain**, not a flat plane. The terrain is neither generated nor adjusted by GPT.
- Mean unit overlap is macro IoU across seven packages, evaluated only on the fixed **66.52% observed bedrock mask**. Contact error compares matching unit-pair boundaries; narrow printed-line gaps are recovered for this approximate contact metric alone.
- Improved fit does not establish correct geological dips, thicknesses or event history. Thin Madison/Amsden bands are poorly resolved in the source figure; do not claim precise recovery.
- The fold-axis arrows and river are not evidence of an exposed fault. Let the actual evidence and proposals determine the story.
