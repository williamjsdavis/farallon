#!/usr/bin/env python3
"""Read-only palette analysis and display-style QA; never changes target labels.

Run from the repository with its venv. Optional --pdf-render accepts a 1500-pixel
wide render of Fiore et al.'s printed page 409 for a separate color cross-check.
All fills and contours below are presentation experiments, never scoring inputs.
"""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_target import DEFAULT_SOURCE, PALETTE, SOURCE_CROP, classify_source

CURRENT_SCENE = "sheep-023fcbb758811c9a"
NEAR_BLACK = [25, 29, 34]
COVER = [251, 240, 141]
PALE_UNKNOWN = [232, 232, 225]


def color_stats(values: np.ndarray) -> dict:
    values = np.asarray(values).reshape(-1, 3)
    if not len(values):
        return {"count": 0, "medianRGB": None, "modeRGB": None}
    unique, counts = np.unique(values, axis=0, return_counts=True)
    k = int(counts.argmax())
    return {"count": len(values), "medianRGB": np.median(values, axis=0).tolist(),
            "modeRGB": unique[k].tolist(), "modeCount": int(counts[k]),
            "p10RGB": np.percentile(values, 10, axis=0).tolist(),
            "p90RGB": np.percentile(values, 90, axis=0).tolist()}


def boundaries(labels: np.ndarray) -> np.ndarray:
    """One-sided raster contours: one pixel on each changed row/column edge."""
    edge = np.zeros(labels.shape, bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return edge


def checker(rgba: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(rgba.shape[:2])
    value = np.where((xx // 8 + yy // 8) % 2, 235, 248).astype(np.uint8)
    bg = np.repeat(value[..., None], 3, axis=-1)
    alpha = rgba[..., 3:4].astype(float) / 255
    return np.uint8(rgba[..., :3] * alpha + bg * (1 - alpha))


def cartographic_observation(labels: np.ndarray, mask: np.ndarray,
                             source_rgba: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Canonical observed colors over original unscored cartographic context.

    This returns a separate RGB display image. It does not fill geology beneath
    cover, alter the mask, or turn source annotations into scored observations.
    """
    h, w = labels.shape
    source = np.array(Image.fromarray(source_rgba).resize((w, h), Image.Resampling.NEAREST))
    display = source[..., :3].copy()
    display[source[..., 3] == 0] = PALE_UNKNOWN
    display[mask] = lut[labels[mask]]
    return display


def cartographic_prediction(labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Full predicted units with antialiased half-pixel contact strokes."""
    h, w = labels.shape
    scale = 4
    strokes = Image.new("L", (w * scale, h * scale), 0)
    pen = ImageDraw.Draw(strokes)
    for y, x in np.argwhere(labels[1:] != labels[:-1]):
        pen.line((int(x * scale), int((y + 1) * scale), int((x + 1) * scale), int((y + 1) * scale)), fill=255, width=2)
    for y, x in np.argwhere(labels[:, 1:] != labels[:, :-1]):
        pen.line((int((x + 1) * scale), int(y * scale), int((x + 1) * scale), int((y + 1) * scale)), fill=255, width=2)
    alpha = np.array(strokes.resize((w, h), Image.Resampling.LANCZOS), dtype=float)[..., None] / 255
    return np.uint8(np.rint(lut[labels] * (1 - alpha) + np.array([44, 43, 42]) * alpha))


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def choose_record(record_path: Path | None) -> tuple[Path, dict]:
    if record_path:
        return record_path, json.loads(record_path.read_text())
    records = [(p, json.loads(p.read_text())) for p in (ROOT / "data/runs").glob("*.json")]
    compatible = [(p, d) for p, d in records if d.get("scene_id") == CURRENT_SCENE]
    if not compatible:
        raise ValueError("No recorded prediction exists for the fixed-terrain scene.")
    return max(compatible, key=lambda item: item[1]["result"]["metrics"]["score"])


def audit(output: Path, audit_json: Path, record_path: Path | None, pdf_render: Path | None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source_bytes = DEFAULT_SOURCE.read_bytes()
    target_hash_before = sha256((ROOT / "data/target.npz").read_bytes()).hexdigest()
    rgba = np.array(Image.open(BytesIO(source_bytes)).convert("RGBA"))
    crop = rgba[SOURCE_CROP[1]:SOURCE_CROP[3], SOURCE_CROP[0]:SOURCE_CROP[2]]
    native_labels, _ = classify_source(crop)
    with np.load(ROOT / "data/target.npz") as target:
        labels, mask, reasons = target["labels"], target["mask"], target["exclusion_reasons"]
    target_meta = json.loads((ROOT / "data/target.json").read_text())
    source_resized = np.array(Image.fromarray(crop).resize((256, 256), Image.Resampling.NEAREST))

    pdf = None
    if pdf_render:
        pdf = np.array(Image.open(pdf_render).convert("RGB"))
        if pdf.shape[1] != 1500:
            raise ValueError("PDF legend rectangles refer to a 1500-pixel-wide page render.")

    audit_units = []
    for entry in PALETTE:
        x0, y0, x1, y1 = entry["legendSample"]
        swatch = rgba[y0:y1, x0:x1]
        legend = color_stats(swatch[swatch[..., 3] == 255, :3])
        native_mask = native_labels == entry["id"]
        interior = distance_transform_edt(native_mask) > 1.5
        item = {"id": entry["id"], "name": entry["name"], "age": entry["age"],
                "currentRGB": entry["rgb"], "recommendedRGB": entry["rgb"],
                "recommendedHex": entry["color"], "legendSampleLTRB": entry["legendSample"],
                "legend": legend,
                "currentMinusLegendMedian": (np.array(entry["rgb"]) - legend["medianRGB"]).tolist(),
                "nativeCropObservedPixels": int(native_mask.sum()),
                "nativeCropInterior": color_stats(crop[interior, :3]),
                "targetPixels": int((labels == entry["id"]).sum()),
                "mapStatsCaveat": "Conditional on the existing color classifier; >1.5 native pixels from each class edge. This is a compression consistency check, not independent class validation."}
        if pdf is not None:
            middle = {1: 930, 2: 901, 3: 871, 4: 841, 5: 811, 6: 781, 7: 751}[entry["id"]]
            box = [122, middle - 3, 141, middle + 4]
            sample = color_stats(pdf[box[1]:box[3], box[0]:box[2]])
            item["renderedPDF"] = {"sampleLTRB": box, **sample,
                "currentMinusMedian": (np.array(entry["rgb"]) - sample["medianRGB"]).tolist(),
                "euclideanDifference": float(np.linalg.norm(np.array(entry["rgb"]) - sample["medianRGB"]))}
        audit_units.append(item)

    record_path, record = choose_record(record_path)
    result = record["result"]
    raw = np.array(Image.open(BytesIO(base64.b64decode(result["map_image"].split(",", 1)[1]))).convert("RGB"))
    palette = np.array([entry["rgb"] for entry in PALETTE], np.int16)
    distances = ((raw[..., None, :].astype(np.int16) - palette).astype(np.int32)**2).sum(axis=-1)
    if distances.min(axis=-1).max() != 0:
        raise ValueError("Recorded map has non-palette colors; do not infer its labels silently.")
    prediction = (distances.argmin(axis=-1) + 1).astype(np.uint8)
    lut = np.array([[0, 0, 0]] + [entry["rgb"] for entry in PALETTE], np.uint8)
    target_rgba = np.dstack((lut[labels], mask.astype(np.uint8) * 255))

    # A: all unobserved gaps deliberately dark; predicted contours are geological
    # class transitions only and never use the target's missing-data mask.
    observed_a = lut[labels].copy()
    observed_a[~mask] = NEAR_BLACK
    pred_a = raw.copy()
    pred_edges = boundaries(prediction)
    pred_a[pred_edges] = NEAR_BLACK

    # B: no inference enters the observation arrays. Fill only one-pixel reason3
    # display gaps away from cover/outside; keep their original mask separately.
    distance, nearest = distance_transform_edt(~mask, return_indices=True)
    protected = (reasons == 1) | (reasons == 2)
    away_from_cover = distance_transform_edt(~protected) > 1.5
    fill = (reasons == 3) & (distance <= 1.0) & away_from_cover
    display_labels = labels.copy()
    display_labels[fill] = labels[nearest[0][fill], nearest[1][fill]]
    observed_b = lut[display_labels].copy()
    observed_b[display_labels == 0] = NEAR_BLACK
    observed_b[reasons == 2] = COVER
    observed_b[reasons == 1] = PALE_UNKNOWN
    pred_b = raw.copy()  # Unoutlined bands preserve narrow predicted units.

    # C: original source-color gaps retain Quaternary yellow and dark linework.
    # Predicted contacts are drawn independently, at half the A stroke width.
    observed_c = cartographic_observation(labels, mask, crop, lut)
    pred_c = cartographic_prediction(prediction, lut)

    panels = {
        "original_crop": source_resized,
        "current_target": target_rgba, "current_prediction": raw,
        "style_a_observed": observed_a, "style_a_prediction": pred_a,
        "style_b_observed": observed_b, "style_b_prediction": pred_b,
        "style_c_observed": observed_c, "style_c_prediction": pred_c,
    }
    for name, array in panels.items():
        Image.fromarray(array).save(output / f"{name}.png")
    Image.fromarray(fill.astype(np.uint8) * 255).save(output / "style_b_display_fill_mask.png")

    rows = [
        ("Reference: supplied source crop / frozen classified target", "original_crop", "current_target"),
        ("Current: fixed observation gaps / full raw prediction", "current_target", "current_prediction"),
        ("A: near-black gaps / 1-pixel dark predicted contacts", "style_a_observed", "style_a_prediction"),
        ("B: small display-only fill + yellow cover / unoutlined prediction", "style_b_observed", "style_b_prediction"),
        ("C: source-like yellow cover + linework / half-pixel predicted contacts", "style_c_observed", "style_c_prediction"),
    ]
    for width, filename in ((520, "contact_sheet.png"), (320, "contact_sheet_app_size.png")):
        height = round(width * target_meta["bounds"]["ymax"] / target_meta["bounds"]["xmax"])
        margin, gap, row_header = 22, 22, 45
        sheet = Image.new("RGB", (margin * 2 + width * 2 + gap, 92 + len(rows) * (height + row_header + gap)), "#f5f3ed")
        draw = ImageDraw.Draw(sheet)
        draw.text((margin, 15), "SHEEP MOUNTAIN  /  PALETTE + GAP STYLE REVIEW", font=font(24 if width == 520 else 19), fill="#17222b")
        draw.text((margin, 49), "Display only — fixed labels, mask and scores. Full prediction.", font=font(15 if width == 520 else 12), fill="#374953")
        for i, (title, left, right) in enumerate(rows):
            y = 85 + i * (height + row_header + gap)
            draw.text((margin, y), title, font=font(18 if width == 520 else 13), fill="#17222b")
            for j, name in enumerate((left, right)):
                panel = panels[name]
                if panel.shape[-1] == 4:
                    panel = checker(panel)
                method = Image.Resampling.LANCZOS if name == "style_c_prediction" else Image.Resampling.NEAREST
                image = Image.fromarray(panel).resize((width, height), method)
                sheet.paste(image, (margin + j * (width + gap), y + row_header))
        sheet.save(output / filename)

    thin_units = []
    for entry in PALETTE:
        unit = prediction == entry["id"]
        count = int(unit.sum())
        thin_units.append({"id": entry["id"], "predictedPixels": count,
                          "aContourPixels": int((pred_edges & unit).sum()),
                          "aContourFraction": float((pred_edges & unit).sum() / count) if count else None})
    report = {
        "schemaVersion": 1, "source": str(DEFAULT_SOURCE.relative_to(ROOT)),
        "sourceSHA256": sha256(source_bytes).hexdigest(), "sourceImageDimensions": [rgba.shape[1], rgba.shape[0]],
        "sourceCropLTRB": list(SOURCE_CROP), "units": audit_units,
        "recommendation": "Retain the existing RGB palette: every entry equals its source WEBP legend median. Display gaps and contact width dominate the visual difference; do not change labels or masks to improve appearance.",
        "recommendedDisplayStyle": "C: canonical observed interiors with source linework/Quaternary context, and independently antialiased predicted unit contacts. Keep raw predicted images and scoring arrays separate.",
        "displayFindings": [
            "A paints all missing observations near-black, including yellow Quaternary, and heavily obscures the narrow mapped core.",
            f"B fills {int(fill.sum())} one-pixel display gaps ({100 * float(fill.mean()):.2f}% of the raster) but does not recover reliable observations of the thin units. Its benefit remains modest at 320-pixel app-panel width.",
            "C most closely matches the supplied cartography without inventing covered geology. The source-like image contains context that is not part of the score; disclose that distinction.",
            "Render the antialiased cartographic prediction with normal image sampling, not CSS image-rendering:pixelated, to keep subpixel contacts continuous at actual panel size.",
            f"Madison has {audit_units[0]['targetPixels']} and Amsden {audit_units[1]['targetPixels']} scored target pixels. Their native crop interiors (>1.5px from class edges) contain only {audit_units[0]['nativeCropInterior']['count']} and {audit_units[1]['nativeCropInterior']['count']} pixels; apparent width is not reliable evidence for precise geological thickness.",
        ],
        "prediction": {"record": str(record_path.relative_to(ROOT)), "sceneId": record.get("scene_id"),
                       "program": result["program"], "metrics": result["metrics"],
                       "selection": "Highest combined score among saved records in this scene, unless --record is provided."},
        "styles": {
            "A": {"gapRGB": NEAR_BLACK, "predictedContactWidthTargetPixels": 1.0,
                  "caveat": "Black missing-data gaps resemble geological linework and can conceal very thin units. This treatment merges cover and uncertainty visually."},
            "B": {"displayFillPixels": int(fill.sum()), "displayFillFractionOfRaster": float(fill.mean()),
                  "displayFillRule": "Only reason3, nearest observed pixel <=1 raster pixel, >1.5 pixels from Quaternary/outside; nearest unit for display only.",
                  "coverRGB": COVER, "predictedContacts": "none",
                  "caveat": "Interpolated color is not a geological observation. Keep scoring mask and explicit display-fill note."},
            "C": {"gapRule": "Source WEBP colors retained where target unobserved; pale outside, source yellow cover and dark printed linework.",
                  "predictedContactRGB": [44, 43, 42], "predictedContactWidthTargetPixels": 0.5,
                  "contactRendering": "Continuous class-edge segments at 4x resolution, 2-pixel strokes, Lanczos-downsampled coverage blended over full raw prediction.",
                  "caveat": "Source annotations and antialiasing remain contextual imagery, not scored observations; predicted lines denote unit boundaries only."},
        },
        "thinUnitContourAudit": thin_units,
        "fixedObservationPixels": int(mask.sum()), "fixedObservationFraction": float(mask.mean()),
        "targetNPZSHA256": target_hash_before,
        "targetInputsUnchanged": sha256((ROOT / "data/target.npz").read_bytes()).hexdigest() == target_hash_before,
        "labelsSHA256": sha256(labels.tobytes()).hexdigest(), "maskSHA256": sha256(mask.tobytes()).hexdigest(),
        "sourceUnchanged": sha256(DEFAULT_SOURCE.read_bytes()).hexdigest() == sha256(source_bytes).hexdigest(),
        "pdfComparison": None if pdf is None else {"renderPath": str(pdf_render),
             "renderSHA256": sha256(pdf_render.read_bytes()).hexdigest(),
             "renderDimensions": [pdf.shape[1], pdf.shape[0]],
             "caveat": "PDFKit rendered page 409 at 1500px width. Rendered RGB depends on color conversion; this is not extraction of original vector color operators."},
        "reviewDirectory": str(output), "contactSheet": str(output / "contact_sheet.png"),
        "appSizeContactSheet": str(output / "contact_sheet_app_size.png"),
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(report, indent=2) + "\n")
    (output / "palette_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/farallon-map-review"))
    parser.add_argument("--audit-json", type=Path, default=ROOT / "data/palette_audit.json")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--pdf-render", type=Path)
    args = parser.parse_args()
    report = audit(args.output_dir, args.audit_json, args.record, args.pdf_render)
    print(json.dumps({"recommendation": report["recommendation"], "prediction": report["prediction"]["record"],
                      "styleBFillPixels": report["styles"]["B"]["displayFillPixels"],
                      "contactSheet": report["contactSheet"]}, indent=2))
