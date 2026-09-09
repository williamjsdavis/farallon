"""Cartography must improve legibility without changing the geological evidence."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image

import server
from geology.cartography import fold_axes
from geology.presentation import INK, OUTSIDE, observed_cartography, predicted_cartography
from geology.restarts import restart_history


def test_server_palette_matches_resampled_source_legend_and_export():
    audit = json.loads((server.ROOT / "data/palette_audit.json").read_text())
    with Image.open(server.ROOT / audit["source"]) as source:
        rgb = np.asarray(source.convert("RGB"))
    for unit in audit["units"]:
        left, top, right, bottom = unit["legendSampleLTRB"]
        sampled = np.median(rgb[top:bottom, left:right].reshape(-1, 3), axis=0)
        np.testing.assert_array_equal(sampled, unit["recommendedRGB"])
        np.testing.assert_array_equal(server.PALETTE[unit["id"], :3], sampled)
        assert server.META["palette"][unit["id"] - 1]["color"] == unit["recommendedHex"]


def test_observed_display_retains_source_context_and_exact_observed_colors():
    labels, mask = server.LABELS.copy(), server.MASK.copy()
    with Image.open(server.ROOT / "data/target_source.png") as source:
        source_rgba = np.array(source.convert("RGBA").resize((256, 256), Image.Resampling.NEAREST))
        shown = observed_cartography(source, labels, mask, server.PALETTE)
    np.testing.assert_array_equal(shown[mask], server.PALETTE[labels[mask]])
    source_context = ~mask & (source_rgba[..., 3] > 0)
    np.testing.assert_array_equal(shown[source_context, :3], source_rgba[source_context, :3])
    assert np.all(shown[..., 3] == 255)
    assert np.all(shown[~mask & (source_rgba[..., 3] == 0)] == OUTSIDE)
    np.testing.assert_array_equal(labels, server.LABELS)
    np.testing.assert_array_equal(mask, server.MASK)
    # Covered areas remain their source yellow; they are not reclassified.
    cover = server.TARGET["exclusion_reasons"] == 2
    assert np.all(~mask[cover])
    np.testing.assert_array_equal(shown[cover, :3], source_rgba[cover, :3])


def test_contacts_are_continuous_between_pixels_without_erasing_thin_units():
    labels = np.array([[1, 2, 3], [1, 2, 3]], np.uint8)
    original = labels.copy()
    shown = predicted_cartography(labels, server.PALETTE, [],
                                   {"xmin": 0, "xmax": 3, "ymin": 0, "ymax": 2})
    for x, unit in ((2, 1), (6, 2), (10, 3)):
        np.testing.assert_array_equal(shown[:, x], np.tile(server.PALETTE[unit], (8, 1)))
    assert np.all(shown[:, 4] == INK) and np.all(shown[:, 8] == INK)
    np.testing.assert_array_equal(labels, original)


def test_true_fold_kinds_generate_different_arrow_directions_only_in_display():
    labels = np.full((64, 64), 6, np.uint8)
    bounds = {"xmin": 0, "xmax": 6, "ymin": 0, "ymax": 6}
    images = []
    for kind in ("anticline", "syncline"):
        history = server.canonical(server.BASELINE.replace("erode(level=0)",
            f"{kind}(x=3,y=3,azimuth=0,plunge_nw=0,plunge_se=0)\nerode(level=0)"))
        axes = fold_axes(history, bounds)
        assert axes[0]["kind"] == kind and axes[0]["status"] == "visible"
        images.append(predicted_cartography(labels, server.PALETTE, axes, bounds))
    assert not np.array_equal(*images)
    assert np.all(labels == 6)


def test_cartographic_image_preserves_world_aspect_for_fold_symbols():
    labels = np.full((64, 64), 6, np.uint8)
    bounds = {"xmin": 0, "xmax": 8, "ymin": 0, "ymax": 4}
    shown = predicted_cartography(labels, server.PALETTE, [], bounds)
    assert shown.shape == (128, 256, 4)


def test_replay_refreshes_cartography_without_rewriting_scores_or_recording(tmp_path, monkeypatch):
    rendered = server.render_result(server.canonical(server.BASELINE), volume=False)
    rendered.pop("cartographic_map_image")
    rendered.pop("fold_axes")
    rendered.pop("cartographic_style")
    record = {"scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID,
              "result": rendered, "model_ms": 1234, "service_tier": "default"}
    run_dir = tmp_path / "data/runs"
    run_dir.mkdir(parents=True)
    path = run_dir / f"{rendered['id']}.json"
    before = json.dumps(record)
    path.write_text(before)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    refreshed = server.replay(rendered["id"])
    assert refreshed["result"]["cartographic_map_image"].startswith("data:image/png;base64,")
    assert refreshed["result"]["fold_axes"] == []
    for key, value in rendered.items():
        assert refreshed["result"][key] == value
    assert refreshed["model_ms"] == 1234 and refreshed["service_tier"] == "default"
    assert path.read_text() == before


def test_syncline_is_preserved_by_perturbed_restart():
    baseline = server.canonical(server.BASELINE)
    best = server.canonical(server.BASELINE.replace("erode(level=0)",
        "syncline(x=3,y=2,uplift=1)\nerode(level=0)"))
    original = deepcopy(best)
    restarted, reason = restart_history(best, baseline, server.META["bounds"], 2, 42)
    assert best == original
    assert restarted["events"][1]["type"] == "syncline"
    assert "fallback" not in reason
