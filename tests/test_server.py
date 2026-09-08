"""Controller contract checks with no paid model requests."""

import json
from datetime import datetime, timezone
import os
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

import server


def scene_request(**values):
    return {"scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID, **values}


def saved_fixture(**overrides):
    return {"scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID,
            "proposal": {"headline": "Mock recording"}, "accepted": True,
            "result": {"metrics": {"miou": 0.2}}, **overrides}


@pytest.fixture
def isolated_controller(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(server, "ROOT", tmp_path)

    def rendered(history, volume=True, resolution=96):
        if history == server.canonical(server.BASELINE):
            score = 0.5
        else:
            x = next(event["x"] for event in history["events"] if event["type"] == "anticline")
            score = 1 - x / 10
        payload = {
            "id": uuid4().hex[:12], "history": history,
            "scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID,
            "program": server.history_to_program(history),
            "metrics": {"miou": score, "accuracy": score, "score": score, "boundary_error_px": 1},
            "map_image": "data:image/png;base64,mocked", "error_image": "data:image/png;base64,mocked",
            "timings": {"generation_ms": 1, "total_ms": 1},
        }
        if volume:
            payload["volume"] = {"data": "AQ==", "shape": [1, 1, 1], "bounds": [0, 1, 0, 1, -1, 0]}
        return payload

    monkeypatch.setattr(server, "render_result", rendered)
    with TestClient(server.app) as client:
        yield client, tmp_path


@pytest.mark.parametrize("candidate_x,accepted", [(2.0, True), (9.5, False)])
def test_measured_acceptance_and_saved_replay(isolated_controller, monkeypatch, candidate_x, accepted):
    client, path = isolated_controller
    candidate_program = f"""strata(levels=[-1.2,-1.05,-0.92,-0.78,-0.5,-0.2], units=[1,2,3,4,5,6,7])
anticline(x={candidate_x},y=1,azimuth=135,uplift=1.3,dip_ne=30,dip_sw=30,plunge_nw=0,plunge_se=0)
erode(level=0)
"""

    async def proposal(request, current):
        return {"headline": "Test candidate", "observation": "Mocked model evidence.",
                "expected_effect": "Test the independent acceptance measurement.",
                "program": candidate_program, "parameters_to_refine": ["x"], "response_id": "mock-only"}

    monkeypatch.setattr(server, "propose", proposal)
    started = datetime.now(timezone.utc)
    response = client.post("/api/iterate", json=scene_request(program=server.BASELINE, refine=False))
    finished = datetime.now(timezone.utc)
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["status", "proposal", "status", "result"]
    result = events[-1]
    assert result["accepted"] is accepted
    assert result["scene_id"] == result["result"]["scene_id"] == server.SCENE_ID
    assert result["baseline_id"] == result["result"]["baseline_id"] == server.BASELINE_ID
    assert result["before_program"] == server.history_to_program(server.canonical(server.BASELINE))
    recorded_at = datetime.fromisoformat(result["recorded_at"])
    assert recorded_at.utcoffset().total_seconds() == 0
    assert started <= recorded_at <= finished
    assert "volume" in result["result"]
    record_id = result["result"]["id"]
    replay = client.get(f"/api/runs/{record_id}")
    assert replay.status_code == 200
    assert replay.json()["recorded_at"] == result["recorded_at"]
    assert replay.json()["result"]["metrics"] == result["result"]["metrics"]
    assert len(list((path / "data/runs").glob("*.json"))) == 1


def test_recorded_replay_order_survives_scrambled_and_equal_mtimes(isolated_controller):
    client, path = isolated_controller
    run_dir = path / "data/runs"
    run_dir.mkdir()
    expected = ["ffffffffffff", "777777777777", "000000000000"]
    copied = []
    for index, run_id in enumerate(expected):
        destination = run_dir / f"{run_id}.json"
        destination.write_text(json.dumps(saved_fixture(recorded_at=f"2026-09-08T12:00:0{index}Z")))
        # A checkout may give files arbitrary or equal mtimes. Neither can
        # replace the explicitly recorded sequence.
        os.utime(destination, (1000 + len(expected) - index,) * 2)
        copied.append(destination)
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert [record["id"] for record in response.json()] == expected
    for destination in copied:
        os.utime(destination, (1000, 1000))
    assert [record["id"] for record in client.get("/api/runs").json()] == expected


def test_current_scene_replay_without_timestamp_uses_mtime(isolated_controller):
    client, path = isolated_controller
    run_dir = path / "data/runs"
    run_dir.mkdir()
    for run_id, mtime in (("ffffffffffff", 1000), ("000000000000", 2000)):
        record = saved_fixture()
        destination = run_dir / f"{run_id}.json"
        destination.write_text(json.dumps(record))
        os.utime(destination, (mtime, mtime))
    assert [record["id"] for record in client.get("/api/runs").json()] == [
        "ffffffffffff", "000000000000"]


def test_invalid_history_rejected_before_model_call(isolated_controller, monkeypatch):
    client, _ = isolated_controller

    async def should_not_call(*args):
        pytest.fail("Invalid programs must never reach the model")

    monkeypatch.setattr(server, "propose", should_not_call)
    response = client.post("/api/iterate", json=scene_request(program="import os", refine=False))
    assert response.status_code == 422


def test_provider_error_does_not_save_a_fake_result(isolated_controller, monkeypatch):
    client, path = isolated_controller

    async def unavailable(*args):
        raise RuntimeError("Test provider unavailable")

    monkeypatch.setattr(server, "propose", unavailable)
    response = client.post("/api/iterate", json=scene_request(program=server.BASELINE, refine=False))
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["status", "error"]
    assert events[-1]["message"] == "Test provider unavailable"
    assert not list((path / "data").rglob("*.json"))


@pytest.mark.parametrize("endpoint", ["render", "refine", "iterate"])
@pytest.mark.parametrize("stale_field", ["scene_id", "baseline_id"])
def test_stale_scene_requests_rejected_before_work(isolated_controller, monkeypatch, endpoint, stale_field):
    client, _ = isolated_controller

    async def should_not_propose(*args):
        pytest.fail("Stale scene requests must never reach the model")

    monkeypatch.setattr(server, "propose", should_not_propose)
    response = client.post(f"/api/{endpoint}", json=scene_request(program=server.BASELINE, **{stale_field: "old-scene"}))
    assert response.status_code == 409
    assert "Reload" in response.json()["detail"]


@pytest.mark.parametrize("kind", ["legacy", "old-terrain", "old-baseline"])
def test_legacy_or_stale_replays_are_excluded_and_cannot_load(isolated_controller, kind):
    client, path = isolated_controller
    run_dir = path / "data/runs"
    run_dir.mkdir()
    old = saved_fixture()
    if kind == "legacy":
        old.pop("scene_id")
        old.pop("baseline_id")
    elif kind == "old-terrain":
        old["scene_id"] = "legacy-flat-terrain"
    else:
        old["baseline_id"] = "old-symmetric-fold"
    (run_dir / "000000000000.json").write_text(json.dumps(old))
    (run_dir / "111111111111.json").write_text(json.dumps(saved_fixture()))
    summaries = client.get("/api/runs").json()
    assert [record["id"] for record in summaries] == ["111111111111"]
    assert client.get("/api/runs/000000000000").status_code == 409
    assert client.get("/api/runs/111111111111").status_code == 200


@pytest.fixture
def playlist_case(isolated_controller):
    client, path = isolated_controller
    run_dir = path / "data/runs"
    run_dir.mkdir()
    base = server.history_to_program(server.canonical(server.BASELINE))

    def folded(x):
        history = server.canonical(server.BASELINE)
        history["events"].insert(-1, {"type": "anticline", "x": x})
        return server.history_to_program(history)

    first, rejected, third = folded(2), folded(3), folded(4)
    ids = ["ffffffff1111", "000000001111", "777777773333"]
    specifications = [(base, first, True), (first, rejected, False), (first, third, True)]
    for index, (run_id, (before, candidate, accepted)) in enumerate(zip(ids, specifications)):
        record = saved_fixture(before_program=before, accepted=accepted,
                               recorded_at=f"2026-09-08T12:00:0{3-index}Z",
                               result={"id": run_id, "scene_id": server.SCENE_ID,
                                       "baseline_id": server.BASELINE_ID, "program": candidate,
                                       "metrics": {"miou": 0.2 + index * 0.1}})
        (run_dir / f"{run_id}.json").write_text(json.dumps(record))
    manifest = {"version": 1, "title": "A single investigation", "scene_id": server.SCENE_ID,
                "baseline_id": server.BASELINE_ID, "run_ids": ids}
    (path / "data/replay.json").write_text(json.dumps(manifest))
    return client, path, ids


def test_playlist_keeps_explicit_order_and_excludes_unrelated_archive_runs(playlist_case):
    client, path, ids = playlist_case
    (path / "data/runs/999999999999.json").write_text(json.dumps(saved_fixture()))
    response = client.get("/api/replay")
    assert response.status_code == 200
    assert response.json()["title"] == "A single investigation"
    assert response.json()["scene_id"] == server.SCENE_ID
    assert response.json()["baseline_id"] == server.BASELINE_ID
    assert [row["id"] for row in response.json()["runs"]] == ids
    assert [row["accepted"] for row in response.json()["runs"]] == [True, False, True]
    # The archive stays complete and timestamp ordered independently.
    archive = client.get("/api/runs").json()
    assert {row["id"] for row in archive} == {*ids, "999999999999"}
    assert [row["id"] for row in archive if row["id"] in ids] == list(reversed(ids))


@pytest.mark.parametrize("location", ["manifest", "record", "result"])
@pytest.mark.parametrize("identity", ["scene_id", "baseline_id"])
def test_playlist_rejects_stale_identity_at_every_level(playlist_case, location, identity):
    client, path, ids = playlist_case
    file = path / "data/replay.json" if location == "manifest" else path / "data/runs" / f"{ids[0]}.json"
    value = json.loads(file.read_text())
    (value["result"] if location == "result" else value)[identity] = "stale"
    file.write_text(json.dumps(value))
    assert client.get("/api/replay").status_code == 409


def test_playlist_missing_record_is_visible_and_does_not_fall_back_to_archive(playlist_case):
    client, path, _ = playlist_case
    file = path / "data/replay.json"
    manifest = json.loads(file.read_text())
    manifest["run_ids"][1] = "123456789abc"
    file.write_text(json.dumps(manifest))
    response = client.get("/api/replay")
    assert response.status_code == 404
    assert "123456789abc" in response.json()["detail"]


def test_playlist_rejects_continuation_from_a_rejected_candidate(playlist_case):
    client, path, ids = playlist_case
    rejected = json.loads((path / "data/runs" / f"{ids[1]}.json").read_text())
    file = path / "data/runs" / f"{ids[2]}.json"
    continuation = json.loads(file.read_text())
    continuation["before_program"] = rejected["result"]["program"]
    file.write_text(json.dumps(continuation))
    response = client.get("/api/replay")
    assert response.status_code == 409
    assert "accepted history" in response.json()["detail"]


def test_playlist_rejects_reordering_that_breaks_history_continuity(playlist_case):
    client, path, _ = playlist_case
    file = path / "data/replay.json"
    manifest = json.loads(file.read_text())
    manifest["run_ids"].reverse()
    file.write_text(json.dumps(manifest))
    assert client.get("/api/replay").status_code == 409


def test_missing_playlist_is_visible_even_if_archive_has_records(isolated_controller):
    client, path = isolated_controller
    (path / "data/runs").mkdir()
    (path / "data/runs/999999999999.json").write_text(json.dumps(saved_fixture()))
    assert client.get("/api/replay").status_code == 404
