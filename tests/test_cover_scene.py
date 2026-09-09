"""Yellow cover is generated geology and fixed observed evidence, end to end."""
import base64
from io import BytesIO

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

import server
from geology.engine import render_map


def test_quaternary_is_scored_with_bedrock_and_contacts():
    yellow = server.LABELS == 8
    assert yellow.sum() == 7290
    assert server.MASK.sum() == 50882
    assert np.all(server.MASK[yellow])
    assert list(server.SCORER.target_units) == list(range(1, 9))
    assert any(8 in pair for pair in server.SCORER.target_contacts)
    exact = server.LABELS.copy()
    exact_score = server.SCORER(exact)
    missing = exact.copy()
    missing[yellow] = 7
    false_cover = exact.copy()
    false_cover[server.MASK & ~yellow] = 8
    for prediction in (missing, false_cover):
        measured = server.SCORER(prediction)
        assert measured["score"] < exact_score["score"]
        assert measured["accuracy"] < exact_score["accuracy"]
        assert measured["per_unit"]["8"] < 1
        assert measured["evaluated_pixels"] == 50882


def test_deposit_surface_volume_colors_and_fold_axis_share_one_history():
    program = server.BASELINE.replace("erode(level=0)",
        "syncline(x=3,y=2,uplift=0.2)\n"
        "deposit(unit=8,x=3,y=2,base=-0.2,curvature_cross=0.4,curvature_along=0.4)\n"
        "erode(level=0)")
    history = server.canonical(program)
    before_mask, before_heights = server.MASK.copy(), server.HEIGHTS.copy()
    result = server.render_result(history, resolution=32)
    prediction = render_map(history, server.XS, server.YS, terrain=server.HEIGHTS)
    assert np.any(prediction == 8) and np.any(prediction != 8)
    raw = base64.b64decode(result["map_image"].split(",", 1)[1])
    image = np.array(Image.open(BytesIO(raw)))
    np.testing.assert_array_equal(image, server.PALETTE[prediction])
    assert np.all(image[prediction == 8] == [251, 240, 141, 255])
    volume = np.frombuffer(base64.b64decode(result["volume"]["data"]), dtype=np.uint8)
    assert 8 in volume and 7 in volume and 0 in volume
    assert result["metrics"] == server.SCORER(prediction)
    assert result["fold_axes"][0]["kind"] == "syncline"
    assert result["fold_axes"][0]["inferred_under_cover"] is True
    np.testing.assert_array_equal(server.MASK, before_mask)
    np.testing.assert_array_equal(server.HEIGHTS, before_heights)


def test_scene_accepts_yellow_deposition_but_cannot_relabel_bedrock():
    with TestClient(server.app) as client:
        for unit, status in ((8, 200), (9, 422), (7, 422)):
            program = server.BASELINE.replace("erode(level=0)",
                f"deposit(unit={unit},x=3,y=2)\nerode(level=0)")
            response = client.post("/api/render", json={"program": program, "volume": False})
            assert response.status_code == status
        # The old seven-unit scores are incompatible even on identical terrain.
        response = client.post("/api/render", json={"program": server.BASELINE,
            "scene_id": "sheep-023fcbb758811c9a", "volume": False})
        assert response.status_code == 409
