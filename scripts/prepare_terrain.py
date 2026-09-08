#!/usr/bin/env python3
"""Prepare real USGS terrain using the map's approximate printed-tick registration.

Default execution is offline and uses data/terrain_source.tif plus its saved
export response. Pass --download to refresh those official, public source files.
No geological labels are used to align, fit or change the DEM.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image

try:
    from scripts.prepare_target import ROOT, build_target, map_registration
except ModuleNotFoundError:
    from prepare_target import ROOT, build_target, map_registration


SERVICE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
SIZE = 256


def export_parameters(registration: dict) -> dict:
    b = registration["geographicBounds"]
    return {
        "bbox": ",".join(str(b[k]) for k in ("west", "south", "east", "north")),
        "bboxSR": "4326", "imageSR": "4326", "size": f"{SIZE},{SIZE}",
        "format": "tiff", "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation",
        "adjustAspectRatio": "false", "f": "pjson",
    }


def download_sources(output_dir: Path, registration: dict) -> None:
    url = SERVICE + "/exportImage?" + urlencode(export_parameters(registration))
    with urlopen(url, timeout=60) as response:
        export = json.load(response)
    if "error" in export or "href" not in export:
        raise ValueError(f"USGS export failed: {export}")
    with urlopen(export["href"], timeout=60) as response:
        (output_dir / "terrain_source.tif").write_bytes(response.read())
    export["retrievedDateUTC"] = datetime.now(timezone.utc).date().isoformat()
    (output_dir / "terrain_export.json").write_text(json.dumps(export, indent=2) + "\n")


def build_terrain(output_dir: Path = ROOT / "data", download: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    registration = map_registration()
    if download:
        download_sources(output_dir, registration)
    source_path = output_dir / "terrain_source.tif"
    export = json.loads((output_dir / "terrain_export.json").read_text())
    expected = registration["geographicBounds"]
    for source_key, expected_key in (("xmin", "west"), ("ymin", "south"), ("xmax", "east"), ("ymax", "north")):
        if not np.isclose(export["extent"][source_key], expected[expected_key], rtol=0, atol=1e-10):
            raise ValueError("Cached DEM extent differs from current map registration; refresh with --download.")
    elevation_m = np.array(Image.open(source_path), dtype=np.float32)
    if elevation_m.shape != (SIZE, SIZE):
        raise ValueError(f"Expected {(SIZE, SIZE)} DEM, found {elevation_m.shape}.")
    if not np.isfinite(elevation_m).all() or np.any((elevation_m < -500) | (elevation_m > 9000)):
        raise ValueError("DEM contains nodata or implausible elevations; no invented fill is permitted.")

    # Regenerate only the coordinate contract; the existing frozen color rules
    # and observation mask remain identical to the pre-terrain target.
    target_meta = build_target(output_dir=output_dir, size=SIZE)
    with np.load(output_dir / "target.npz") as target:
        xs, ys = target["xs"], target["ys"]
    datum_m = float(elevation_m.min())
    maximum_m = float(elevation_m.max())
    heights_km = (elevation_m - np.float32(datum_m)) / np.float32(1000)
    np.savez_compressed(output_dir / "terrain.npz", heights_km=heights_km, xs=xs, ys=ys, elevation_m=elevation_m)

    # Scientific QA render only: normalized Lambertian hillshade, never used
    # to change the DEM or to decide which observations are scored.
    dy, dx = np.gradient(heights_km, ys, xs)
    light = np.array([-0.5, 0.5, 0.70710678])
    shade = (-dx * light[0] - dy * light[1] + light[2]) / np.sqrt(1 + dx*dx + dy*dy)
    grayscale = np.uint8(np.clip(0.25 + 0.75 * shade, 0, 1) * 255)
    Image.fromarray(grayscale).save(output_dir / "terrain_hillshade.png")

    metadata = {
        "schemaVersion": 1,
        "title": "Sheep Mountain — actual USGS 3DEP terrain",
        "datum_m": datum_m,
        "elevation_min_m": datum_m,
        "elevation_max_m": maximum_m,
        "relief_m": maximum_m - datum_m,
        "heightUnits": "km above minimum elevation in cached DEM",
        "verticalDatum": "Source vertical datum not independently identified for this composite-service export; absolute service elevations are meters. The model uses only relief above the cached minimum.",
        "source": {
            "name": "USGS 3D Elevation Program (3DEP) Bare Earth DEM Dynamic Image Service",
            "url": SERVICE,
            "productPage": "https://www.usgs.gov/3d-elevation-program/about-3dep-products-services",
            "exportRequest": SERVICE + "/exportImage?" + urlencode(export_parameters(registration)),
            "exportParameters": export_parameters(registration),
            "exportResponseFile": "data/terrain_export.json",
            "cachedRaster": "data/terrain_source.tif",
            "cachedRasterSHA256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "retrievedDateUTC": export.get("retrievedDateUTC", "2026-09-08"),
            "access": "Public official service; no API key. USGS 3DEP products are freely available; original geological figure retains its separate provenance.",
        },
        "registration": registration,
        "geographicBounds": expected,
        "bounds": target_meta["bounds"],
        "boundsArray": target_meta["boundsArray"],
        "raster": {
            "width": SIZE, "height": SIZE, "pixelType": "float32",
            "order": "rows north to south; columns west to east",
            "sampleLocations": "cell centers; xs ascending, ys descending, identical to target.npz",
            "cellWidthM": registration["widthKm"] * 1000 / SIZE,
            "cellHeightM": registration["heightKm"] * 1000 / SIZE,
            "resampling": "USGS server-side bilinear elevation sampling; no geological fitting or analytical terrain approximation",
            "effectiveResolutionNote": "Export pixel spacing is not a claim about native DEM source resolution or vertical accuracy.",
        },
        "notes": [
            "Terrain covers the complete crop, including pixels whose geological label is unobserved.",
            "No flat-earth replacement, synthetic valley, target-label fitting, nearest fill, or terrain smoothing was used.",
            "Printed coordinate calibration supersedes the initial rough scale-bar metric extent. Historical flat-surface recordings are a different scene and their scores are not comparable.",
            "The original map datum and survey accuracy are unknown. Interpret residual contact mismatch with this registration uncertainty in mind.",
            "A surface match still does not uniquely constrain subsurface structure or geological history.",
        ],
        "files": {"array": "data/terrain.npz", "metadata": "data/terrain.json", "hillshade": "data/terrain_hillshade.png", "provenance": "data/TERRAIN.md"},
    }
    (output_dir / "terrain.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--download", action="store_true", help="Refresh the official cached USGS export; requires network access.")
    args = parser.parse_args()
    metadata = build_terrain(args.output_dir, args.download)
    print(json.dumps({k: metadata[k] for k in ("bounds", "geographicBounds", "datum_m", "elevation_max_m", "relief_m")}, indent=2))


if __name__ == "__main__":
    main()
