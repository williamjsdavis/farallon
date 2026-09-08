#!/usr/bin/env python3
"""Prepare the fixed, bedrock-only observation raster from the supplied map.

This classifies source pixels by their distance from the seven legend colors.
It does not fill contact lines, infer covered bedrock, or alter the source image.
Printed geographic ticks provide approximate registration to a real DEM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "assets" / "Geological-map-of-Sheep-Mountain-anticline-The-Bighorn-River-dissects-the-fold.webp"
SOURCE_CROP = (20, 50, 286, 250)  # left, top, right, bottom; right/bottom exclusive
SCALE_BAR = ((438, 650), (490, 650))
PIXELS_PER_KM = 52.0
COLOR_TOLERANCE = 25.0  # Euclidean distance in source 8-bit sRGB

# Manually read the actual tick strokes, not the adjacent text centers.
# Coordinates are source-image pixel coordinates, measured from the top left.
LONGITUDE_TICKS = [(64, -108 - 12/60), (179, -108 - 10/60),
                   (294, -108 - 8/60), (410, -108 - 6/60),
                   (529, -108 - 4/60)]
LATITUDE_TICKS = [(156, 44 + 38/60), (335, 44 + 36/60)]


def map_registration() -> dict:
    """Approximate north-up geographic calibration, independent of geology fit."""
    lon_pixels, lon_degrees = np.array(LONGITUDE_TICKS).T
    lat_pixels, lat_degrees = np.array(LATITUDE_TICKS).T
    lon_slope, lon_intercept = np.polyfit(lon_pixels, lon_degrees, 1)
    lat_slope, lat_intercept = np.polyfit(lat_pixels, lat_degrees, 1)
    x0, y0, x1, y1 = SOURCE_CROP
    west, east = lon_slope * np.array([x0, x1]) + lon_intercept
    north, south = lat_slope * np.array([y0, y1]) + lat_intercept
    # WGS84 local radii of curvature; adequate for this roughly 6 km crop.
    phi = np.deg2rad((north + south) / 2)
    flattening = 1 / 298.257223563
    eccentricity2 = flattening * (2 - flattening)
    denom = 1 - eccentricity2 * np.sin(phi)**2
    prime_vertical_radius = 6378137 / np.sqrt(denom)
    meridional_radius = 6378137 * (1 - eccentricity2) / denom**1.5
    width_km = prime_vertical_radius * np.cos(phi) * np.deg2rad(east - west) / 1000
    height_km = meridional_radius * np.deg2rad(north - south) / 1000
    residual_pixels = (lon_slope * lon_pixels + lon_intercept - lon_degrees) / lon_slope
    return {
        "method": "Manual printed-coordinate tick strokes; independent linear longitude(x) and latitude(y) fits; north-up assumed from map arrow.",
        "longitudeTicks": [{"pixelX": x, "longitude": lon} for x, lon in LONGITUDE_TICKS],
        "latitudeTicks": [{"pixelY": y, "latitude": lat} for y, lat in LATITUDE_TICKS],
        "longitudeDegreesPerPixel": float(lon_slope),
        "longitudeAtPixelX0": float(lon_intercept),
        "latitudeDegreesPerPixel": float(lat_slope),
        "latitudeAtPixelY0": float(lat_intercept),
        "longitudeFitResidualPixels": residual_pixels.tolist(),
        "longitudeFitMaxResidualPixels": float(np.max(np.abs(residual_pixels))),
        "geographicBounds": {"west": float(west), "south": float(south), "east": float(east), "north": float(north)},
        "widthKm": float(width_km), "heightKm": float(height_km),
        "horizontalCRS": "EPSG:4326 assumed for DEM request; original figure horizontal datum is unspecified",
        "localMetricMethod": "WGS84 ellipsoid radii of curvature at crop midpoint latitude; x east and y north from southwest crop corner",
        "confidence": "Approximate figure registration, not surveyed control. Tick reading, figure distortion, cartographic generalization and unknown original datum remain. Tick-fit residuals do not measure external positional accuracy.",
        "visualCheck": "Full-map USGS context hillshade has the corresponding NW-SE ridge and Bighorn River canyon; this is a qualitative check, not an independent accuracy measurement.",
        "legacyScaleBarEstimate": {"pixelsPerKm": PIXELS_PER_KM, "barEndpointsPixels": [list(p) for p in SCALE_BAR], "used": False, "reason": "Replaced with independent horizontal and vertical printed-coordinate calibration; the rough bar estimate is inconsistent with those ticks."},
    }

# Frozen colors sampled from the centers of the source legend swatches.
# These are categorical identifiers, not estimates of physical rock color.
PALETTE = [
    {"id": 1, "name": "Madison", "color": "#893f7c", "rgb": [137, 63, 124], "age": "Mississippian", "legendSample": [13, 628, 32, 636]},
    {"id": 2, "name": "Amsden", "color": "#214b8a", "rgb": [33, 75, 138], "age": "Pennsylvanian", "legendSample": [13, 605, 32, 613]},
    {"id": 3, "name": "Tensleep", "color": "#2e6faa", "rgb": [46, 111, 170], "age": "Pennsylvanian", "legendSample": [13, 581, 32, 589]},
    {"id": 4, "name": "Phosphoria", "color": "#729ac8", "rgb": [114, 154, 200], "age": "Permian", "legendSample": [13, 557, 32, 565]},
    {"id": 5, "name": "Triassic", "color": "#b1cbe6", "rgb": [177, 203, 230], "age": "Triassic", "legendSample": [13, 534, 32, 542]},
    {"id": 6, "name": "Jurassic", "color": "#d4dfe4", "rgb": [212, 223, 228], "age": "Jurassic", "legendSample": [13, 510, 32, 518]},
    {"id": 7, "name": "Cretaceous", "color": "#93c89b", "rgb": [147, 200, 155], "age": "Cretaceous", "legendSample": [13, 485, 32, 493]},
]
QUATERNARY_RGB = [251, 240, 141]


def classify_source(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 labels and exclusion reasons at the native source resolution.

    Reasons: 0 observed bedrock, 1 transparent/outside, 2 Quaternary cover,
    3 unmatched color (linework, annotations, or uncertain blended source pixel).
    """
    colors = np.array([entry["rgb"] for entry in PALETTE] + [QUATERNARY_RGB], dtype=np.float32)
    distances = ((rgba[..., None, :3].astype(np.float32) - colors) ** 2).sum(axis=-1)
    nearest = distances.argmin(axis=-1)
    close = distances.min(axis=-1) <= COLOR_TOLERANCE**2
    opaque = rgba[..., 3] == 255
    labels = np.where(opaque & close & (nearest < 7), nearest + 1, 0).astype(np.uint8)
    reasons = np.full(labels.shape, 3, dtype=np.uint8)
    reasons[opaque & close & (nearest == 7)] = 2
    reasons[labels > 0] = 0
    reasons[~opaque] = 1
    return labels, reasons


def build_target(source_path: Path = DEFAULT_SOURCE, output_dir: Path = ROOT / "data", size: int = 256) -> dict:
    if size < 32:
        raise ValueError("Target size must be at least 32 pixels.")
    source = Image.open(source_path).convert("RGBA")
    if source.size != (697, 671):
        raise ValueError(f"Crop and scale calibration require the supplied 697×671 map, got {source.size}.")
    x0, y0, x1, y1 = SOURCE_CROP
    source_crop = source.crop(SOURCE_CROP)
    native_labels, native_reasons = classify_source(np.asarray(source_crop))
    # Nearest-neighbor resampling never invents a blended geological class.
    labels = np.array(Image.fromarray(native_labels).resize((size, size), Image.Resampling.NEAREST))
    reasons = np.array(Image.fromarray(native_reasons).resize((size, size), Image.Resampling.NEAREST))
    mask = labels > 0
    registration = map_registration()
    width_km = registration["widthKm"]
    height_km = registration["heightKm"]
    xs = (np.arange(size, dtype=np.float64) + 0.5) * width_km / size
    ys = height_km - (np.arange(size, dtype=np.float64) + 0.5) * height_km / size
    bounds = {"xmin": 0.0, "xmax": width_km, "ymin": 0.0, "ymax": height_km}

    lut = np.zeros((8, 4), dtype=np.uint8)
    for entry in PALETTE:
        lut[entry["id"]] = [*entry["rgb"], 255]
    target_rgba = lut[labels]
    class_counts = {str(entry["id"]): int((labels == entry["id"]).sum()) for entry in PALETTE}
    excluded_counts = {
        "transparentOrOutside": int((reasons == 1).sum()),
        "quaternaryCover": int((reasons == 2).sum()),
        "lineworkAnnotationsOrUncertainColor": int((reasons == 3).sum()),
    }
    total_pixels = int(labels.size)
    metadata = {
        "schemaVersion": 1,
        "title": "Sheep Mountain — northwestern fold nose",
        "description": "Fixed categorical observations from a crop of the supplied geological map, north of the Bighorn River crossing. Seven ordered bedrock packages are retained. Quaternary cover, transparent/outside pixels and colors inconsistent with the legend are unobserved.",
        "bounds": bounds,
        "boundsArray": [0.0, width_km, 0.0, height_km],
        "coordinateSystem": {"x": "east", "y": "north", "z": "up", "units": "km", "georeferenced": True, "registrationAccuracy": "approximate printed-figure ticks", "origin": "southwest corner of source crop"},
        "surface": {"type": "terrain", "file": "data/terrain.npz", "units": "km", "heightDatum": "minimum cached USGS DEM elevation", "registration": "approximate"},
        "sourceImage": str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
        "sourceImageSHA256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "sourceImageDimensions": {"width": source.width, "height": source.height},
        "sourceCrop": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
        "sourceCropLTRB": list(SOURCE_CROP),
        "scaleCalibration": registration,
        "geographicBounds": registration["geographicBounds"],
        "raster": {"width": size, "height": size, "order": "rows north to south; columns west to east", "sampleLocations": "cell centers", "cellWidthKm": width_km / size, "cellHeightKm": height_km / size, "resampling": "nearest neighbor"},
        "palette": PALETTE,
        "unobserved": {"id": 0, "name": "Unobserved / masked", "color": "#00000000"},
        "labeledFraction": float(mask.mean()),
        "labeledPixels": int(mask.sum()),
        "totalPixels": total_pixels,
        "classCounts": class_counts,
        "classFractions": {key: count / total_pixels for key, count in class_counts.items()},
        "excludedCounts": excluded_counts,
        "classification": {"method": "Nearest fixed legend color in 8-bit sRGB", "maxColorDistance": COLOR_TOLERANCE, "quaternaryRGB": QUATERNARY_RGB, "requireOpaqueSource": True, "maskFixedAcrossCandidates": True, "interpolateAcrossMasks": False, "exclusionReasonCodes": {"0": "observed bedrock", "1": "transparent/outside", "2": "Quaternary cover", "3": "linework, annotation, or uncertain compressed color"}},
        "citations": [{"authors": "Fiore Allwardt et al.", "year": 2007, "publication": "Geosphere", "figure": "Figure 1, printed page 409", "doi": "10.1130/GES00088.1", "url": "https://doi.org/10.1130/GES00088.1", "localPaper": "papers/Fiore-et-al-Geosphere-2007.pdf", "role": "Supplied geological map; target observations"}],
        "provenance": "Source figure supplied by the user. This target was rasterized from that figure, not from the NPS BICA GeoPackage or the Rioux USGS map. No new license is asserted for the source figure.",
        "warnings": [
            "This is an illustrative map interpretation, not dense ground samples or a surveyed reconstruction.",
            "Real USGS 3DEP terrain is registered approximately using printed coordinate ticks. Original map datum, distortion and generalized contacts can cause positional mismatch; see data/TERRAIN.md.",
            "Madison and Amsden occupy very few resolved source pixels in this crop; do not claim precise widths or thicknesses for them.",
            "The source is a compressed raster. A fixed color tolerance masks uncertain, antialiased and annotated pixels; it does not reconstruct geology beneath them.",
            "Triassic, Jurassic and Cretaceous are grouped map packages, not single formations.",
            "The mapped arrowed axes represent folds, not evidence of an exposed surface fault.",
            "A good surface fit does not uniquely constrain the 3D geometry or geological history.",
        ],
        "files": {"array": "data/target.npz", "image": "data/target.png", "mask": "data/target_mask.png", "sourceCrop": "data/target_source.png", "metadata": "data/target.json"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "target.npz", labels=labels, mask=mask, xs=xs, ys=ys, exclusion_reasons=reasons)
    Image.fromarray(target_rgba).save(output_dir / "target.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / "target_mask.png")
    source_crop.save(output_dir / "target_source.png")
    (output_dir / "target.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    metadata = build_target(args.source.resolve(), args.output_dir, args.size)
    print(json.dumps({"bounds": metadata["bounds"], "labeledFraction": metadata["labeledFraction"], "classCounts": metadata["classCounts"], "excludedCounts": metadata["excludedCounts"]}, indent=2))


if __name__ == "__main__":
    main()
