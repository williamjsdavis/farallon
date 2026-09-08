"""Check the browser/server numerical contract without making model API calls."""

import base64
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
    return server.render_result(server.canonical(server.BASELINE), resolution=32)


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
    expected = evaluate_points(rendered["history"], points)
    np.testing.assert_array_equal(labels.ravel(), expected)
    assert parse_history(rendered["program"]) == rendered["history"]


def test_map_png_and_mask_preserve_actual_north_up_categorical_prediction(rendered):
    prediction = render_map(rendered["history"], server.XS, server.YS)
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
    measured = render_map(result["result"]["history"], server.XS, server.YS)
    assert result["result"]["metrics"] == server.SCORER(measured)
    records = list((tmp_path / "data" / "runs").glob("*.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text())
    assert parse_history(saved["result"]["program"]) == saved["result"]["history"]
    assert saved["accepted"] is False
