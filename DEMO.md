# Farallon: three-minute presenter aid

Open **http://localhost:3000**. The story is **visual evidence → executable history → 3D simulation → measurement → revised explanation**.

## Before going on stage

- Click **Reset**. Confirm the starting history has only `strata(...)` and `erode(level=0)`: seven horizontal bedrock packages, **no fold or young deposit**. Its surface prediction is Cretaceous throughout; mean unit overlap is **2.33%** against the eight-unit target.
- Set **Model** to **GPT-6 Astra · Strongest** for the launch story. The dropdown also offers **GPT-5.6 Luna · Speed** (no reasoning), **Terra · Balanced** and **Sol · Strong** (low reasoning); Astra uses low reasoning. All four request Fast mode, and the choice applies to single and Auto runs. Actual returned service tier is shown per result; these labels are not measured guarantees of latency or quality for this task.
- Leave the map on **Source**, the lower panel on **Reasoning**, cutaway near zero and **Vertical 1×**. Confirm **Replay next (1/3)** is available for the current terrain scene.
- **Terrain** shows the actual cached USGS DEM: **416.1 m relief** across **6.059 × 4.139 km**. Printed map ticks provide approximate registration; details are in [data/TERRAIN.md](data/TERRAIN.md).
- Optional full-map context: `http://localhost:3000/api/image/full`. **Source** shows the original crop; **Units** shows the fixed classified observations.
- **Generate**, **Tune parameters**, feature tests and 3D controls use local computation. They do not make another model API call.
- Choose single-step investigation or the optional Auto sequence below. Auto starts only after **Start auto** is clicked; it never launches calls on page load.

## Talking script and clicks

| Time | Click / show | Say |
| --- | --- | --- |
| 0:00–0:20 | **Source**, then **Terrain**; point to the fold nose and yellow corridor. | “This is a geological map of Sheep Mountain, Wyoming, paired with real terrain. Can GPT-6 write a history that explains both the folded rock bands and these young surface deposits?” |
| 0:20–0:45 | Click **Test next hypothesis**. Show **History**, then **Units**; gently orbit. | “We start with horizontal sedimentary layers. GPT sees the map, terrain, prediction and mismatch. It must propose the missing geological events and write the executable history.” |
| 0:45–1:10 | Read the first actual result; click **Test next hypothesis** again. Show the new history. | “In our rehearsal it added a plunging anticline, then deposited young material along the northeastern corridor. A separate local search tunes the parameters, and the simulator measures the resulting outcrops.” |
| 1:10–1:35 | Start a third investigation when ready. Show **Terrain**, **Mismatch** and the generation timer. | “The next proposals reshape the narrow oldest-rock core. The terrain stays fixed, and the yellow deposit comes after folding. Warm volume and surface generation took about nine to eleven milliseconds in the rehearsal; model calls took seconds.” |
| 1:35–1:55 | Show the investigation rows and retained best. | “Our rehearsal improved mean unit overlap from 2.3% to 39.6%, then 50.3% and 50.6%. All three proposals were accepted. Yellow cover reached 67% overlap. The last gain is small: improvement does not mean we have recovered the true history.” |
| 1:55–2:20 | On the retained best, click **Remove plunge**, compare the northwest nose and score, then **Restore best**. | “We can remove a feature and regenerate immediately. In the rehearsed model, removing plunge drops overlap from 50.6% to 39.9%. Yellow stays unchanged because it was deposited after folding.” |
| 2:20–2:40 | Move **Cut into the mountain** to about 35%; orbit; toggle **Vertical 2×**, then return to 1×. | “The terrain surface and underground layers come from the same 3D model. This toggle exaggerates height only for viewing; the geology and score are unchanged.” |
| 2:40–3:00 | End on the best block, **Units** and **History**. | “This is a possible history, not a unique reconstruction. The figure is only approximately registered, and considerable mismatch remains. GPT connects visual interpretation to executable hypotheses that we can inspect, measure and challenge.” |

Live hypotheses and timing vary: describe the result actually shown. Read **Expected** as a prediction, not an accomplished effect. Start the next call promptly; if running behind, stop after two proposals and preserve time for the feature test and cutaway.

## Optional Auto sequence

For the same three-minute structure, select **GPT-6 Astra**, set **Iterations** to **3** and click **Start auto** at 0:20. Skip the manual second and third investigation clicks: Auto runs them sequentially from the current branch's best. Continue narrating the evidence, history and measured results. To end early, click **Stop auto**; **Finishing iteration…** means the current call will finish and its result will be retained, with no next call. Wait before the feature test or replay. The best overall model is restored when the run completes, stops or errors. Auto keeps the selected model for the entire run; the rehearsal timings below are not an Auto performance guarantee.

For a longer investigation, the control accepts **1–100 iterations**, defaults to **20**, and the server enforces the cap. Each iteration makes one paid proposal call. A three-iteration run shows the automatic sequence; use a larger limit to show restarts. After four attempts without an overall combined-score improvement, or ten attempts in one branch, it starts a new branch. Odd restarts use fresh undeformed layers; even restarts broadly vary the best history already found, including any young deposit, with an undeformed fallback. It does not insert a prebuilt fold. The seven bedrock identities, Quaternary identity, target observations and terrain stay fixed.

Say: **“The viewer follows the current branch, while we retain the best explanation found across every branch. A restart lets us explore another starting point without losing that best model.”** Branch acceptance and overall improvement use the combined score, not overlap alone.

Stopping before startup completes aborts the request. Closing or refreshing cancels the run; errors stop it without automatic paid retries. Completed proposals are saved with Auto provenance, but there is no automatic session resume. Auto does not change the explicit three-record replay. Auto is available in the live investigation; use **Return to live** first if replay is open.

## Transparent replay fallback

Say: **“I’ll switch to our saved rehearsal. These are genuine GPT-6 proposals and their recorded simulations; this sequence is replay.”**

When idle, click **Replay next** three times. The app identifies replay visibly and shows the step count. The playlist is **Sheep Mountain: folds and Quaternary cover**, defined in `data/replay.json`; the current observations, terrain, starting history and accepted-model continuity are checked before playback. Other investigations never enter this sequence. After step three, **Replay complete** is disabled; there is no automatic wraparound. **Reset** restarts the rehearsal from undeformed layers. **Return to live** restores the investigation you had open before replay, including the selected model and history editor. Previous seven-unit and flat-surface experiments remain archived; their scores are not comparable or offered for this scene. A busy request must finish or error before replay is available.

Replayed timing is recorded timing, not a fresh benchmark. Orbit and cutaway stay interactive; feature tests perform new local renders. All three recorded candidates were accepted; after the third replay, the displayed third model is also the retained best. **Restore best** is needed after inspecting an earlier model or running a feature test.

## Verified eight-unit terrain rehearsal

All three records use scene `sheep-18ee492e408ac79e`, baseline `9a229952fd019ab7`, and were recorded **September 9, 2026 UTC** (September 8 in California). They used **GPT-6 Astra, low reasoning**, and each API response confirmed **Fast** service. Overlap below is macro IoU across all eight observed units; the retained best ends at **50.58%**.

| Stage / record | Recorded UTC | Overlap | Contact error | Decision | Iteration time |
| --- | --- | ---: | ---: | --- | ---: |
| Undeformed starting strata | — | 2.33% | No predicted contacts | Starting model | — |
| Add plunging anticline and young valley fill · `3982d29f03e0` | 06:00:26.035 | 39.60% | 18.96 px | Accept | 16.89 s |
| Add localized folding to expose the oldest core · `03a5d833007e` | 06:00:41.359 | 50.27% | 13.00 px | Accept | 15.29 s |
| Extend the older-rock core northwest · `f5de8726020c` | 06:00:53.444 | **50.58%** | **12.99 px** | **Accept; retained best** | 12.05 s |

Three proposals totalled **44.23 s**. Model calls took **16.13 / 14.35 / 11.08 s**; each used **192 local parameter trials**, taking 0.70–0.88 s. Recorded warm generation of the **96³ volume plus 256² terrain-intersection map** took **8.77–11.44 ms**. That excludes scoring, encoding, API latency and browser rendering; full server render/score/encode took **41.96–49.15 ms**. Fast confirmation describes the actual service tier; it does not establish a fixed speedup or promise these latencies on stage.

Final-best observed-pixel accuracy was **77.15%**, a different metric from **50.58% mean unit overlap**. Quaternary's individual IoU was **67.03%**. The model uses two bedrock-fold events followed by one `deposit(unit=8, ...)` event; it does not contain a syncline.

## Feature test

Use **Remove plunge on retained best `f5de8726020c`** for the stage demonstration. These measurements reproduce each UI edit independently from that model, using the current fixed DEM and eight-unit scorer, without parameter refitting or model API calls.

| Edit | Mean unit overlap | Contact error | Observed pixels changed |
| --- | ---: | ---: | ---: |
| Original retained best | 50.58% | 12.99 px | — |
| **Remove plunge** | **39.87%** | **26.75 px** | **17.75%** |
| Make symmetric | 50.44% | 13.06 px | 1.09% |

Removing plunge sets both plunge parameters to zero in every fold event and worsens both reported fit measures. Making the limbs symmetric has only a small effect because this fitted model's limb dips are already similar; it is a weaker stage demonstration. Quaternary IoU remains **67.03%** in both tests because young deposition follows these folds. Each button starts from the same original parent history: clicking **Make symmetric** after **Remove plunge** does **not** stack both edits. **Restore best** returns to the accepted best model. These tests establish a feature's role in this fitted model, not exact parameters for the real mountain. For a different live result, report its actual change—even if the fit stays unchanged or improves.

## Facts to keep straight

- GPT writes a restricted executable geological DSL using existing procedural operators; it does not invent a numerical operator or use Lemmalog. Local parameter optimization is separate from its hypothesis proposal.
- The measured DEM is a fixed observation surface. `erode(level=0)` means **zero offset from that terrain**, not a flat plane. The terrain is neither generated nor adjusted by GPT.
- Mean unit overlap is macro IoU across **seven bedrock packages plus Quaternary unit 8**, evaluated on the fixed **77.64% observation mask**: 50,882 of 65,536 pixels. All 7,290 classified yellow pixels are scored. Transparent/outside and uncertain ink or blended colors remain excluded. Contact error compares matching unit-pair boundaries; narrow printed-line gaps are recovered for this approximate contact metric alone.
- `deposit(unit=8, ...)` generates young cover from an analytic basal surface; it is not a painted copy of the target. Event order matters: in the rehearsal, folding precedes deposition. The final terrain determines which material is exposed.
- Improved fit does not establish correct geological dips, thicknesses or event history. Thin Madison/Amsden bands are poorly resolved in the source figure; do not claim precise recovery.
- The source legend distinguishes anticline and syncline axes. A trace crossing yellow indicates mapped structural context; it does not establish that the Quaternary itself was folded. A fully buried syncline can leave the surface colors unchanged, so those colors alone do not constrain its geometry. Printed axis symbols are contextual evidence, not an independently scored subsurface reconstruction.
- Dashed predicted axes come from the proposed fold events; the accepted rehearsal has anticlines but no syncline. Neither the fold-axis arrows nor the river is evidence of an exposed fault. Let the actual evidence and proposals determine the story.
