"""Checks for observation masking and the map-to-world coordinate contract."""

import hashlib
import json

import numpy as np
from PIL import Image

from scripts.prepare_target import DEFAULT_SOURCE, PALETTE, build_target, classify_source


def test_quaternary_is_observed_but_annotations_and_transparency_remain_masked():
    pixels = np.array([[
        [251, 240, 141, 255],  # Young surface deposits have a distinct unit ID.
        [0, 0, 0, 255],       # Contact/annotation line.
        [255, 255, 255, 255], # White annotation/background.
        [137, 63, 124, 0],    # A geological RGB with no observed source pixel.
        [137, 63, 124, 255],  # An observed oldest-unit pixel.
    ]], dtype=np.uint8)
    labels, reasons = classify_source(pixels)
    assert labels.tolist() == [[8, 0, 0, 0, 1]]
    assert reasons.tolist() == [[0, 3, 3, 1, 0]]


def test_export_preserves_north_up_geometry_and_fixed_observations(tmp_path):
    source_hash = hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest()
    metadata = build_target(output_dir=tmp_path)
    arrays = np.load(tmp_path / "target.npz")
    labels, mask = arrays["labels"], arrays["mask"]
    xs, ys = arrays["xs"], arrays["ys"]
    rgba = np.asarray(Image.open(tmp_path / "target.png"))
    assert labels.shape == mask.shape == (256, 256)
    assert labels.dtype == np.uint8 and mask.dtype == np.bool_
    assert np.array_equal(mask, labels > 0)
    assert np.array_equal(mask, rgba[..., 3] == 255)
    assert set(np.unique(labels)) == set(range(9))
    # Image rows run down while world northing runs up, with samples at centers.
    assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) < 0)
    assert np.isclose(xs[0], metadata["raster"]["cellWidthKm"] / 2)
    assert np.isclose(ys[-1], metadata["raster"]["cellHeightKm"] / 2)
    assert np.isclose(xs[-1] + xs[0], metadata["bounds"]["xmax"])
    assert np.isclose(ys[0] + ys[-1], metadata["bounds"]["ymax"])
    assert metadata["bounds"]["xmax"] > metadata["bounds"]["ymax"]
    assert int(mask.sum()) == 50882
    assert metadata["classCounts"] == {
        "1": 147, "2": 195, "3": 1911, "4": 3484,
        "5": 4541, "6": 23832, "7": 9482, "8": 7290,
    }
    assert metadata["excludedCounts"] == {
        "transparentOrOutside": 100,
        "quaternaryCover": 0,
        "lineworkAnnotationsOrUncertainColor": 14554,
    }
    assert np.all(arrays["exclusion_reasons"][labels == 8] == 0)
    # The extension must preserve every prior bedrock label and coordinate.
    # These are hashes of the original seven-unit fixed observation export.
    legacy_labels = np.where(labels == 8, 0, labels).astype(np.uint8)
    assert hashlib.sha256(legacy_labels.tobytes()).hexdigest() == "3f4e465a4764b623f2fcae26a300679030a0bba48e9f1a9b14d081be29170477"
    assert hashlib.sha256(xs.tobytes()).hexdigest() == "8b40ccadd8360aa0a20e94ae8df0926195755863987331f8cc354c590b68076f"
    assert hashlib.sha256(ys.tobytes()).hexdigest() == "cf045d4dec22bab1af74055ad3711b205e2ffc8c39cc8eedde5a3cb612894c78"
    assert sum(metadata["classCounts"].values()) == int(mask.sum())
    assert sum(metadata["excludedCounts"].values()) + int(mask.sum()) == labels.size
    for entry in PALETTE:
        assert np.all(rgba[labels == entry["id"], :3] == entry["rgb"])
    assert json.loads((tmp_path / "target.json").read_text())["sourceImageSHA256"] == source_hash
    assert hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest() == source_hash


def test_all_eight_palette_colors_match_original_legend_swatch_medians():
    source = np.asarray(Image.open(DEFAULT_SOURCE).convert("RGBA"))
    assert [entry["id"] for entry in PALETTE] == list(range(1, 9))
    assert PALETTE[-1]["rgb"] == [251, 240, 141]
    assert PALETTE[-1]["color"] == "#fbf08d"
    for entry in PALETTE:
        x0, y0, x1, y1 = entry["legendSample"]
        swatch = source[y0:y1, x0:x1]
        opaque_rgb = swatch[swatch[..., 3] == 255, :3]
        assert np.array_equal(np.median(opaque_rgb, axis=0), entry["rgb"])
