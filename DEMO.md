# Farallon: three-minute presenter aid

Open **http://localhost:3000**. The story is **visual evidence → executable geological history → 3D simulation → independent measurement → revised hypothesis**.

## Before going on stage

- Confirm the starting model and live model indicator load. Click **Reset**; leave the map on **Source**, the lower panel on **Reasoning**, and the cutaway near zero.
- Confirm **Replay next (3)** or a larger saved-run count is available. The three original rehearsal records are listed below. Replay order follows saved-file modification time; avoid touching those files before presenting.
- If useful, open `http://localhost:3000/api/image/full` in a second tab for full-map context. **Source** inside the app shows the unclassified crop, not the full figure.
- Do not run another paid rehearsal merely to check the buttons. **Generate**, feature tests, orbit and cutaway use the local simulator.

## Talking script and clicks

| Time | Click / show | Say |
|---|---|---|
| 0:00–0:15 | **Source**, point at the northwest tip and contrasting band widths. | “This is a published geological map of Sheep Mountain, Wyoming. We see rocks at the surface. Can GPT-6 write a geological history that generates a 3D world consistent with those observations?” |
| 0:15–0:40 | Click **Let GPT-6 investigate**. While it works, switch to **Units**, gently orbit the block. | “We start with a deliberately poor fold: symmetric, with no plunge. GPT sees the original crop, our predicted surface and the mismatch. It proposes a specific geological change, then writes the program that produces it.” |
| 0:40–1:05 | When the first result arrives, point to its observation and overlap. Click **Let GPT-6 investigate** again. Open **History**. | “These are executable events: deposit the strata, deform them, then erode to expose the surface. GPT chooses the hypothesis and parameters to examine. A separate local numerical search tunes those parameters; the measured fit decides whether to retain it.” |
| 1:05–1:30 | When ready, start a third investigation. Show **Mismatch** and the generation timer; move **Cut into the mountain** to about 35%. | “Each candidate produces a real volume and a surface prediction. The warm simulation takes roughly nine milliseconds here. The model’s visual reasoning takes seconds, which gives us time to inspect the geology.” |
| 1:30–1:50 | Read the actual result; show **Units** and the investigation rows. | “The retained model now agrees better with the mapped units. In our recorded rehearsal, mean unit overlap rose from 4.2% to 33.6% in three proposals. That is a substantial improvement, with substantial mismatch still visible.” |
| 1:50–2:15 | On the best result, click **Remove plunge**, compare the northwest end and score, then **Restore best**. | “We can test a feature directly. Removing plunge changes the surface prediction immediately. In the rehearsed best model, overlap falls from 33.6% to 27.0%. Restoring the same history restores the fit.” |
| 2:15–2:40 | Orbit the block; adjust the cutaway. Return to **History** and optionally click **Generate** once. | “The surface and these underground layers come from the same short program. We can inspect it, change it and regenerate it. The loop connects visual interpretation to code and then to a measurable consequence.” |
| 2:40–3:00 | End on the best block and observed/predicted maps. | “This is a possible history. We hold terrain flat, whereas real erosion shapes these contacts, and one surface map cannot uniquely determine the subsurface. The useful capability is proposing and testing explanations—and showing us where they still fail.” |

Live timings and proposals vary. Start the next call as soon as a result arrives. If running behind, stop after two proposals and reserve 30 seconds for the feature test and block reveal. Read **Expected** as the model's prediction, not an accomplished result.

## Transparent replay fallback

If a live call errors or cannot finish, say: **“I’ll switch to the saved run from our rehearsal. These are genuine GPT-6 proposals and their recorded simulations; this next sequence is replay.”**

When the app is idle, click **Reset**, then **Replay next** to step through the original three records. The app visibly changes to **RECORDED RUN / REPLAY**. Orbit, cutaway and feature tests remain interactive; a feature test performs a new local render. **Restore best** returns to the best loaded model. If a request is still busy, wait for completion/error; the replay button is disabled during an active call.

Do not describe the displayed replay generation time as a new benchmark: it is stored with that record. The three original records are:

| Stage | Record | Mean unit overlap | Contact error | Model time | Total iteration |
|---|---|---:|---:|---:|---:|
| Starting hypothesis | `server.BASELINE` | 4.17% | 53.50 px | — | — |
| Narrow fold with NW closure | `772c9fcdb2cc` | 29.91% | 23.77 px | 28.55 s | 29.28 s |
| Reposition and extend the nose | `e57cfef0bd45` | 30.26% | 21.93 px | 22.90 s | 23.67 s |
| Revise uplift and layer intervals | `049306ce6f1f` | 33.63% | 19.70 px | 18.33 s | 19.08 s |

Records live in `data/runs/*.json`. All three were accepted under the combined score and totalled **72.04 seconds**, with **192 local numerical trials per proposal**. Recorded warm generation was **8.85–9.06 ms** for the 96³ volume plus 256² surface. This excludes scoring, encoding, API latency and browser rendering; the recorded server render/score/encode totals were about 27–37 ms.

## Which feature button is strongest?

These measurements reproduce the UI's exact edits using `server.render_result`; no model API calls were made. Each edit begins from the specified unmodified record.

| Selected model / edit | Overlap before → after | Contact error before → after | Observed pixels changed |
|---|---:|---:|---:|
| **Final best → Remove plunge** | **33.63% → 27.01%** | **19.70 → 27.85 px** | **13.87%** |
| Final best → Make symmetric | 33.63% → 33.68% | 19.70 → 20.09 px | 2.05% |
| First accepted → Remove plunge | 29.91% → 25.48% | 23.77 → 29.61 px | 12.97% |
| First accepted → Make symmetric | 29.91% → 25.38% | 23.77 → 25.34 px | 24.12% |

Use **Remove plunge on the final best** in the main presentation. For an optional asymmetry demonstration, select the **first accepted investigation row**, then click **Make symmetric**. The final fit already has nearly equal limb dips, so its symmetry test is visually weak and slightly improves overlap while worsening the combined score. Always restore/reselect the original model between tests; the buttons edit the currently selected history.

## Facts to keep straight

- GPT-6 writes a restricted, executable geological DSL using fixed procedural operators. This run does not generate a new numerical operator or use Lemmalog. Local parameter search is separate from GPT's proposal.
- **Mean unit overlap** is macro IoU across seven classes, not overall pixel accuracy. Only the fixed **66.52% observed bedrock mask** is scored. Cover and unreliable source pixels remain excluded. Contact error is approximate because narrow printed-line gaps are reconstructed for that metric alone.
- The third proposal **expected to expose Madison**, but its final refined result still predicts **zero Madison pixels in the observed area**. Its improvement comes elsewhere. Do not claim that expected effect was achieved. The first two accepted models also miss Madison.
- The final fitted limb dips are about **38.0° and 39.1°**. This simplified map fit does not recover the documented strong real-world limb asymmetry or establish correct geological dips/thicknesses.
- Fold-axis arrows and the river are not evidence of an exposed fault. The demo's actual story is plunge, fold shape, placement and layer intervals. Do not force a fault-discovery narrative onto this target.
- Geometry uses a smooth, explicit procedural deformation with a flat observation surface. Smooth contacts will not reproduce every erosional scallop. Extra hypotheses may improve fit, but neither improvement nor a particular next discovery is guaranteed.
