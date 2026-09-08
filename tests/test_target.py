"""Checks for observation masking and the map-to-world coordinate contract."""

import hashlib
import json

import numpy as np
from PIL import Image

from scripts.prepare_target import DEFAULT_SOURCE, PALETTE, build_target, classify_source


def test_cover_annotations_and_transparency_are_not_bedrock():
    pixels = np.array([[
        [251, 240, 141, 255],  # Quaternary cover must never become deep bedrock.
        [0, 0, 0, 255],       # Contact/annotation line.
        [255, 255, 255, 255], # White annotation/background.
        [137, 63, 124, 0],    # A geological RGB with no observed source pixel.
        [137, 63, 124, 255],  # An observed oldest-unit pixel.
    ]], dtype=np.uint8)
    labels, reasons = classify_source(pixels)
    assert labels.tolist() == [[0, 0, 0, 0, 1]]
    assert reasons.tolist() == [[2, 3, 3, 1, 0]]


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
    assert set(np.unique(labels)) == set(range(8))
    # Image rows run down while world northing runs up, with samples at centers.
    assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) < 0)
    assert np.isclose(xs[0], metadata["raster"]["cellWidthKm"] / 2)
    assert np.isclose(ys[-1], metadata["raster"]["cellHeightKm"] / 2)
    assert np.isclose(xs[-1] + xs[0], metadata["bounds"]["xmax"])
    assert np.isclose(ys[0] + ys[-1], metadata["bounds"]["ymax"])
    assert metadata["bounds"]["xmax"] > metadata["bounds"]["ymax"]
    assert 0.60 < metadata["labeledFraction"] < 0.75
    assert sum(metadata["classCounts"].values()) == int(mask.sum())
    assert sum(metadata["excludedCounts"].values()) + int(mask.sum()) == labels.size
    for entry in PALETTE:
        assert np.all(rgba[labels == entry["id"], :3] == entry["rgb"])
    assert json.loads((tmp_path / "target.json").read_text())["sourceImageSHA256"] == source_hash
    assert hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest() == source_hash
