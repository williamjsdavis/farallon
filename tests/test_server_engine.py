"""Check the browser/server numerical contract without making model API calls."""

import base64
from copy import deepcopy
from io import BytesIO
import json

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

import server
from geology.engine import evaluate_points, render_map
from geology.history import parse_history


def _png(url):
    assert url.startswith("data:image/png;base64,")
    return np.asarray(Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))).convert("RGBA"))


@pytest.fixture(scope="module")
def rendered():
    # The baseline is uniformly undeformed. A fold makes x/y orientation and
    # surface-height mistakes observable in both colors and subsurface units.
    program = server.BASELINE.replace(
        "erode(level=0)",
        "anticline(x=2.7,y=1.5,azimuth=135,uplift=1.7,dip_ne=55,dip_sw=25)\nerode(level=0)",
    )
    return server.render_result(server.canonical(program), resolution=32)


def test_payload_volume_is_uint8_zyx_at_ascending_world_voxel_centres(rendered):
    # Decode exactly as a browser would. Checking raw array length alone would
    # not catch x/y/z permutation or north-up image order leaking into volume.
    json.dumps(rendered, allow_nan=False)
    volume = rendered["volume"]
    data = base64.b64decode(volume["data"], validate=True)
    assert len(data) == int(np.prod(volume["shape"]))
    labels = np.frombuffer(data, dtype=np.uint8).reshape(volume["shape"])
    nz, ny, nx = labels.shape
    xmin, xmax, ymin, ymax, zmin, zmax = volume["bounds"]
    k, j, i = np.indices(labels.shape)
    points = np.column_stack([
        (xmin + (i.ravel() + 0.5) * (xmax - xmin) / nx),
        (ymin + (j.ravel() + 0.5) * (ymax - ymin) / ny),
        (zmin + (k.ravel() + 0.5) * (zmax - zmin) / nz),
    ])
    uneroded = deepcopy(rendered["history"])
    assert uneroded["events"].pop()["type"] == "erode"
    expected = evaluate_points(uneroded, points).reshape(labels.shape)
    xs = xmin + (np.arange(nx) + .5) * (xmax - xmin) / nx
    ys = ymin + (np.arange(ny) + .5) * (ymax - ymin) / ny
    terrain = server.TERRAIN_GRID.sample(xs, ys)
    above_surface = points[:, 2].reshape(labels.shape) > terrain[None, :, :]
    expected[above_surface] = 0
    np.testing.assert_array_equal(labels, expected)
    assert np.any(above_surface)
    assert np.any((labels > 0) & (points[:, 2].reshape(labels.shape) > 0))
    assert np.all(labels[above_surface] == 0)
    assert parse_history(rendered["program"]) == rendered["history"]


def test_cap_vertices_decode_to_fixed_dem_at_ascending_world_edges(rendered):
    volume = rendered["volume"]
    _, ny, nx = volume["shape"]
    xmin, xmax, ymin, ymax, _, _ = volume["bounds"]
    surface = volume["surface"]
    assert surface["encoding"] == "float32-le"
    assert surface["order"] == "south-to-north"
    assert surface["shape"] == [ny + 1, nx + 1]
    raw = base64.b64decode(surface["data"], validate=True)
    assert len(raw) == 4 * (ny + 1) * (nx + 1)
    heights = np.frombuffer(raw, dtype="<f4").reshape(surface["shape"])
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    expected = server.TERRAIN_GRID.sample(xs, ys).astype("<f4")
    np.testing.assert_array_equal(heights, expected)
    assert np.ptp(heights) > 0


def test_map_png_and_mask_preserve_actual_north_up_categorical_prediction(rendered):
    prediction = render_map(rendered["history"], server.XS, server.YS, terrain=server.HEIGHTS)
    assert np.all(np.diff(server.XS) > 0)
    assert np.all(np.diff(server.YS) < 0)
    image = _png(rendered["map_image"])
    masked = _png(rendered["observed_map_image"])
    error = _png(rendered["error_image"])
    np.testing.assert_array_equal(image, server.PALETTE[prediction])
    np.testing.assert_array_equal(masked[server.MASK], image[server.MASK])
    assert np.all(masked[~server.MASK, 3] == 0)
    assert np.all(error[~server.MASK, 3] == 0)
    assert rendered["metrics"] == server.SCORER(prediction)
    assert rendered["metrics"]["evaluated_pixels"] == int(server.MASK.sum())


def test_fixed_dem_mask_and_scene_survive_a_different_history(rendered):
    terrain_before = server.HEIGHTS.copy()
    mask_before = server.MASK.copy()
    source_before = server.TERRAIN_GRID.heights_km.copy()
    baseline = server.render_result(server.canonical(server.BASELINE), resolution=32)
    assert baseline["map_image"] != rendered["map_image"]
    assert baseline["volume"]["surface"] == rendered["volume"]["surface"]
    assert baseline["volume"]["bounds"] == rendered["volume"]["bounds"]
    assert baseline["scene_id"] == rendered["scene_id"] == server.SCENE_ID
    assert baseline["baseline_id"] == rendered["baseline_id"] == server.BASELINE_ID
    assert baseline["metrics"]["evaluated_pixels"] == rendered["metrics"]["evaluated_pixels"]
    np.testing.assert_array_equal(server.HEIGHTS, terrain_before)
    np.testing.assert_array_equal(server.MASK, mask_before)
    np.testing.assert_array_equal(server.TERRAIN_GRID.heights_km, source_before)
    np.testing.assert_array_equal(server.HEIGHTS, server.TERRAIN_GRID.sample(server.XS, server.YS))
    assert not server.HEIGHTS.flags.writeable


def test_render_endpoint_rejects_changes_to_observation_and_semantic_labels():
    with TestClient(server.app) as client:
        changed_terrain = server.BASELINE.replace("erode(level=0)", "erode(level=0.1)")
        changed_ids = server.BASELINE.replace("units=[1, 2, 3, 4, 5, 6, 7]", "units=[2, 1, 3, 4, 5, 6, 7]")
        for program in (changed_terrain, changed_ids, "import os"):
            response = client.post("/api/render", json={"program": program, "volume": False})
            assert response.status_code == 422
        response = client.post("/api/render", json={"program": server.BASELINE, "volume": False})
        assert response.status_code == 200
        payload = response.json()
        assert "volume" not in payload
        assert payload["metrics"]["evaluated_pixels"] == int(server.MASK.sum())


def test_iteration_rejects_no_improvement_and_saves_the_measured_program(monkeypatch, tmp_path):
    # A deterministic local substitute is essential: this test must never
    # consume API credits or depend on a model's response latency/content.
    calls = []

    async def local_proposal(request, current):
        calls.append(current["metrics"])
        return {"headline": "Unchanged control", "observation": "Control proposal.",
                "expected_effect": "No improvement.", "program": request.program,
                "parameters_to_refine": []}

    monkeypatch.setattr(server, "propose", local_proposal)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    with TestClient(server.app) as client:
        response = client.post("/api/iterate", json={"program": server.BASELINE, "refine": False})
        assert response.status_code == 200
        messages = [json.loads(line) for line in response.text.splitlines()]
    assert len(calls) == 1
    assert not any(message["type"] == "error" for message in messages)
    result = next(message for message in messages if message["type"] == "result")
    assert result["accepted"] is False
    assert result["before"] == result["result"]["metrics"]
    measured = render_map(result["result"]["history"], server.XS, server.YS, terrain=server.HEIGHTS)
    assert result["result"]["metrics"] == server.SCORER(measured)
    records = list((tmp_path / "data" / "runs").glob("*.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text())
    assert parse_history(saved["result"]["program"]) == saved["result"]["history"]
    assert saved["accepted"] is False
