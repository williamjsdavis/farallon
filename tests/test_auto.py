"""Bounded auto investigations, using fake proposals and numerical scores only."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient
import httpx
import pytest

import server


def program(score):
    return f"""strata(levels=[-1.2,-1.05,-0.92,-0.78,-0.5,-0.2], units=[1,2,3,4,5,6,7])
anticline(x={score},y=1,azimuth=135,uplift=1)
erode(level=0)
"""


def proposal(history_program):
    return {"headline": "Mock proposal", "observation": "Mock evidence.",
            "expected_effect": "Measure a candidate.", "program": history_program,
            "parameters_to_refine": [], "response_id": "mock-only",
            "requested_service_tier": "fast", "service_tier": "default"}


def request_body(**extra):
    return {"program": server.BASELINE, "refine": False, "seed": 7,
            "scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID, **extra}


def read_events(response):
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines()]


@pytest.fixture
def isolated_auto(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "ITERATION_LOCK", asyncio.Lock())
    monkeypatch.setattr(server, "AUTO_RUNS", {})
    (tmp_path / "data").mkdir()
    captured, restart_calls = [], []

    def rendered(history, volume=True, resolution=96):
        fold = next((event for event in history["events"] if event["type"] == "anticline"), None)
        score = 0.2 if fold is None else fold["x"]
        result = {"id": uuid4().hex[:12], "history": deepcopy(history),
                  "scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID,
                  "program": server.history_to_program(history),
                  "metrics": {"score": score, "miou": score, "accuracy": score, "boundary_error_px": 1},
                  "map_image": "data:image/png;base64,mocked", "error_image": "data:image/png;base64,mocked"}
        if volume:
            result["volume"] = {"data": "AQ==", "shape": [1, 1, 1]}
        return result

    async def unchanged(request, current):
        captured.append(request.model_copy(deep=True))
        return proposal(request.program)

    def restarted(best, baseline, bounds, restart_index, seed):
        restart_calls.append({"best": deepcopy(best), "baseline": deepcopy(baseline),
                              "bounds": bounds, "restart_index": restart_index, "seed": seed})
        return server.canonical(program(0.1)), "A mocked exploratory restart"

    monkeypatch.setattr(server, "render_result", rendered)
    monkeypatch.setattr(server, "propose", AsyncMock(side_effect=unchanged))
    monkeypatch.setattr(server, "restart_history", restarted)
    return SimpleNamespace(client=TestClient(server.app), path=tmp_path,
                           captured=captured, restarts=restart_calls)


def scripted_proposals(monkeypatch, scores, captured=None):
    scores = iter(scores)

    async def scripted(request, current):
        if captured is not None:
            captured.append(request.model_copy(deep=True))
        score = next(scores)
        if isinstance(score, Exception):
            raise score
        return proposal(program(score))

    mock = AsyncMock(side_effect=scripted)
    monkeypatch.setattr(server, "propose", mock)
    return mock


@pytest.mark.parametrize("budget", [1, 100])
def test_exact_iteration_budget_and_saved_attempts(isolated_auto, budget):
    env = isolated_auto
    events = read_events(env.client.post("/api/auto", json=request_body(max_iterations=budget)))
    started, complete = events[0], events[-1]
    assert started["type"] == "started"
    assert started["max_iterations"] == budget
    assert started["seed"] == 7
    assert started["patience"] == 4 and started["restart_every"] == 10
    results = [event for event in events if event["type"] == "result"]
    assert len(results) == server.propose.await_count == budget
    assert [event["auto"]["iteration"] for event in results] == list(range(1, budget + 1))
    assert complete["type"] == "complete" and complete["reason"] == "limit"
    assert complete["completed_iterations"] == budget
    assert "volume" in complete["best"]
    files = list((env.path / "data/runs").glob("*.json"))
    assert len(files) == budget
    assert {json.loads(path.read_text())["auto"]["run_id"] for path in files} == {started["run_id"]}
    assert not server.ITERATION_LOCK.locked() and not server.AUTO_RUNS
    # Reaching the limit never generates a restart for an iteration that cannot run.
    assert events[-2]["type"] == "result"


@pytest.mark.parametrize("field,value", [
    ("max_iterations", 0), ("max_iterations", 101), ("max_iterations", True),
    ("max_iterations", 1.0), ("max_iterations", "1"),
    ("seed", -1), ("seed", 2**32), ("seed", True), ("seed", 1.5), ("seed", "2"),
])
def test_invalid_budget_or_seed_rejected_before_model(isolated_auto, field, value):
    response = isolated_auto.client.post("/api/auto", json=request_body(**{field: value}))
    assert response.status_code == 422
    server.propose.assert_not_awaited()
    assert not server.ITERATION_LOCK.locked() and not server.AUTO_RUNS


def test_default_budget_and_generated_seed_are_bounded(isolated_auto):
    assert server.AutoRequest(program=server.BASELINE).max_iterations == 20
    body = request_body(max_iterations=1)
    body.pop("seed")
    events = read_events(isolated_auto.client.post("/api/auto", json=body))
    assert isinstance(events[0]["seed"], int) and 0 <= events[0]["seed"] < 2**32


@pytest.mark.parametrize("override,status", [
    ({"program": "import os"}, 422),
    ({"scene_id": "stale"}, 409),
    ({"baseline_id": "stale"}, 409),
])
def test_invalid_scene_or_history_never_claims_lock(isolated_auto, override, status):
    response = isolated_auto.client.post("/api/auto", json=request_body(**override))
    assert response.status_code == status
    server.propose.assert_not_awaited()
    assert not server.ITERATION_LOCK.locked() and not server.AUTO_RUNS


def test_restart_keeps_global_best_while_accepting_worse_branch(isolated_auto, monkeypatch):
    env = isolated_auto
    scripted_proposals(monkeypatch, [0.9, 0.8, 0.8, 0.8, 0.8, 0.3], env.captured)
    events = read_events(env.client.post("/api/auto", json=request_body(max_iterations=6)))
    results = [event for event in events if event["type"] == "result"]
    restart = next(event for event in events if event["type"] == "restart")
    assert restart["branch"] == 2 and restart["reason"] == "stalled"
    assert restart["result"]["metrics"]["score"] == 0.1
    assert restart["result"]["restart"]["branch"] == 2
    assert restart["global_best"]["metrics"]["score"] == 0.9
    final_attempt = results[-1]
    assert final_attempt["accepted"] is True
    assert final_attempt["auto"]["global_improved"] is False
    assert final_attempt["branch_best"]["metrics"]["score"] == 0.3
    assert final_attempt["global_best"]["id"] == results[0]["result"]["id"]
    assert events[-1]["best"]["id"] == results[0]["result"]["id"]
    assert env.captured[-1].previous == []
    assert env.captured[-1].auto_context["branch"] == 2
    assert env.captured[-1].auto_context["global_best"]["metrics"]["score"] == 0.9
    assert env.restarts[0]["restart_index"] == 1 and env.restarts[0]["seed"] == 7
    assert env.restarts[0]["best"] == server.canonical(results[0]["result"]["program"])
    saved = json.loads((env.path / f"data/runs/{final_attempt['result']['id']}.json").read_text())
    assert saved["auto"] == final_attempt["auto"]
    assert "global_best" not in saved  # Archive avoids repeating the full retained volume.


def test_local_branch_improvements_do_not_reset_global_stall(isolated_auto, monkeypatch):
    scripted_proposals(monkeypatch, [0.2] * 4 + [0.2, 0.3, 0.4, 0.5, 0.2])
    events = read_events(isolated_auto.client.post(
        "/api/auto", json=request_body(program=program(0.9), max_iterations=9)))
    results = [event for event in events if event["type"] == "result"]
    assert all(event["accepted"] for event in results[4:8])
    assert all(not event["auto"]["global_improved"] for event in results)
    assert [event["branch"] for event in events if event["type"] == "restart"] == [2, 3]
    assert results[-1]["auto"]["branch"] == 3
    assert events[-1]["best"]["metrics"]["score"] == 0.9


def test_branch_limit_restarts_even_when_each_attempt_improves(isolated_auto, monkeypatch):
    scripted_proposals(monkeypatch, [0.25 + 0.05 * index for index in range(11)])
    events = read_events(isolated_auto.client.post("/api/auto", json=request_body(max_iterations=11)))
    restarts = [event for event in events if event["type"] == "restart"]
    results = [event for event in events if event["type"] == "result"]
    assert len(restarts) == 1 and restarts[0]["reason"] == "branch_limit"
    assert results[9]["auto"]["branch_iteration"] == 10
    assert results[10]["auto"]["branch"] == 2
    assert all(event["auto"]["global_improved"] for event in results)


def test_scored_restart_can_be_global_best_with_explicit_provenance(isolated_auto, monkeypatch):
    monkeypatch.setattr(server, "restart_history", lambda *args, **kwargs: (
        server.canonical(program(0.95)), "Mock random seed improvement"))
    scripted_proposals(monkeypatch, [0.2] * 5)
    events = read_events(isolated_auto.client.post("/api/auto", json=request_body(max_iterations=5)))
    restart = next(event for event in events if event["type"] == "restart")
    assert restart["global_improved"] is True
    assert restart["global_best"]["id"] == restart["result"]["id"]
    assert events[-1]["best"]["metrics"]["score"] == 0.95
    assert events[-1]["best"]["restart"] == {
        "branch": 2, "description": "Mock random seed improvement"}


def test_stop_finishes_current_attempt_without_another_call(isolated_auto, monkeypatch):
    async def scenario():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def blocked(request, current):
            entered.set()
            await finish.wait()
            return proposal(program(0.8))

        mocked = AsyncMock(side_effect=blocked)
        monkeypatch.setattr(server, "propose", mocked)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            running = asyncio.create_task(client.post("/api/auto", json=request_body(max_iterations=10)))
            await asyncio.wait_for(entered.wait(), 2)
            run_id = next(iter(server.AUTO_RUNS))
            stopped = await client.post(f"/api/auto/{run_id}/stop")
            assert stopped.status_code == 200 and stopped.json()["stop_requested"] is True
            assert (await client.post(f"/api/auto/{run_id}/stop")).status_code == 200
            finish.set()
            events = read_events(await asyncio.wait_for(running, 2))
            assert events[-1]["reason"] == "stopped"
            assert events[-1]["completed_iterations"] == 1
            assert events[-1]["best"]["metrics"]["score"] == 0.8
            assert not any(event["type"] == "restart" for event in events)
            mocked.assert_awaited_once()
            assert (await client.post(f"/api/auto/{run_id}/stop")).status_code == 404
        assert not server.ITERATION_LOCK.locked() and not server.AUTO_RUNS

    asyncio.run(scenario())


@pytest.mark.parametrize("scores,count,best", [
    ([RuntimeError("Mock provider unavailable")], 0, 0.2),
    ([0.9, RuntimeError("Mock provider unavailable")], 1, 0.9),
])
def test_provider_error_preserves_best_and_releases_lock(isolated_auto, monkeypatch, scores, count, best):
    mocked = scripted_proposals(monkeypatch, scores)
    events = read_events(isolated_auto.client.post("/api/auto", json=request_body(max_iterations=10)))
    assert events[-2]["type"] == "error" and "Mock provider unavailable" in events[-2]["message"]
    assert events[-1]["type"] == "complete" and events[-1]["reason"] == "error"
    assert events[-1]["completed_iterations"] == count
    assert events[-1]["best"]["metrics"]["score"] == best
    assert mocked.await_count == count + 1
    assert len(list((isolated_auto.path / "data/runs").glob("*.json"))) == count
    assert not server.ITERATION_LOCK.locked() and not server.AUTO_RUNS


@pytest.mark.parametrize("owner", ["auto", "iterate"])
def test_manual_and_auto_share_one_nonqueuing_lock(isolated_auto, monkeypatch, owner):
    async def scenario():
        entered, finish = asyncio.Event(), asyncio.Event()

        async def blocked(request, current):
            entered.set()
            await finish.wait()
            return proposal(request.program)

        mocked = AsyncMock(side_effect=blocked)
        monkeypatch.setattr(server, "propose", mocked)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            running = asyncio.create_task(client.post(f"/api/{owner}", json=request_body(max_iterations=1)))
            await asyncio.wait_for(entered.wait(), 2)
            for contender in ("iterate", "auto"):
                response = await asyncio.wait_for(client.post(
                    f"/api/{contender}", json=request_body(max_iterations=1)), 1)
                assert response.status_code == 409
            finish.set()
            assert (await running).status_code == 200
            mocked.assert_awaited_once()
        assert not server.ITERATION_LOCK.locked() and not server.AUTO_RUNS

    asyncio.run(scenario())


def test_disconnect_cancels_pending_proposal_and_cleans_up(isolated_auto, monkeypatch):
    async def scenario():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def blocked(request, current):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        mocked = AsyncMock(side_effect=blocked)
        monkeypatch.setattr(server, "propose", mocked)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            running = asyncio.create_task(client.post("/api/auto", json=request_body(max_iterations=10)))
            await asyncio.wait_for(entered.wait(), 2)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        assert cancelled.is_set()
        mocked.assert_awaited_once()
        assert not server.ITERATION_LOCK.locked() and not server.AUTO_RUNS
        assert not list((isolated_auto.path / "data/runs").glob("*.json"))

    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["auto", "iterate"])
def test_transport_failure_before_first_event_releases_claim(isolated_auto, endpoint):
    async def scenario():
        request = (server.AutoRequest(**request_body(max_iterations=1)) if endpoint == "auto"
                   else server.IterationRequest(**request_body()))
        response = await getattr(server, endpoint)(request)
        assert server.ITERATION_LOCK.locked()

        async def disconnected_send(message):
            raise OSError("Mock client disconnected before response headers")

        async def receive():
            return {"type": "http.disconnect"}

        with pytest.raises(Exception):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, disconnected_send)
        assert not server.ITERATION_LOCK.locked() and not server.AUTO_RUNS
        server.propose.assert_not_awaited()

    asyncio.run(scenario())
