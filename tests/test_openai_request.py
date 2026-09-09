"""Exercise the real proposal request while replacing the network client."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
import pytest

import server


@pytest.fixture
def provider_stub(monkeypatch):
    """No OpenAI client is constructed and no credentials leave the process."""
    payload = {
        "headline": "A mocked geological proposal",
        "observation": "The observed map contains curved contacts.",
        "expected_effect": "Test the request and response metadata.",
        "program": server.BASELINE,
        "parameters_to_refine": ["levels"],
    }
    response = SimpleNamespace(
        status="completed", output_text=json.dumps(payload), id="resp_mock_only",
        usage=None, service_tier="default",
    )
    create = AsyncMock(return_value=response)

    class Client:
        responses = SimpleNamespace(create=create)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    constructor = Mock(return_value=Client())
    monkeypatch.setattr(server, "AsyncOpenAI", constructor)
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-never-sent")
    monkeypatch.setattr(server, "file_url", lambda path: f"data:image/png;base64,{path.name}")
    return SimpleNamespace(response=response, create=create, constructor=constructor, payload=payload)


@pytest.mark.parametrize("requested,returned", [
    ("fast", "fast"),
    ("fast", "default"),
    ("default", "default"),
])
def test_propose_requests_configured_tier_and_reports_actual_tier(
    monkeypatch, provider_stub, requested, returned,
):
    monkeypatch.setattr(server, "SERVICE_TIER", requested)
    provider_stub.response.service_tier = returned
    previous = {"scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID,
                "headline": "Earlier observed result"}
    request = server.IterationRequest(program=server.BASELINE, previous=[
        {**previous, "scene_id": "stale-scene"}, previous,
    ])
    current = {"metrics": {"miou": 0.03, "score": 0.025},
               "map_image": "data:image/png;base64,prediction",
               "error_image": "data:image/png;base64,mismatch"}

    proposal = asyncio.run(server.propose(request, current))

    provider_stub.constructor.assert_called_once_with(timeout=75.0, max_retries=0)
    provider_stub.create.assert_awaited_once()
    call = provider_stub.create.call_args.kwargs
    assert call["service_tier"] == requested
    assert call["model"] == server.MODEL
    assert call["reasoning"] == {"effort": server.EFFORT}
    assert call["instructions"] == server.SYSTEM_PROMPT
    assert call["max_output_tokens"] == 3500

    content = call["input"][0]["content"]
    assert call["input"][0]["role"] == "user"
    context = json.loads(content[0]["text"])
    assert context["current_program"] == request.program
    assert context["current_metrics"] == current["metrics"]
    assert context["recent_attempts"] == [previous]
    images = [item for item in content if item["type"] == "input_image"]
    assert [item["image_url"] for item in images] == [
        f"data:image/png;base64,{server.META['sourceImage'].split('/')[-1]}",
        "data:image/png;base64,target.png", current["map_image"],
        current["error_image"], server.TERRAIN_IMAGE,
    ]
    assert all(item["detail"] == "high" for item in images)
    output_format = call["text"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["name"] == "geological_proposal"
    assert output_format["strict"] is True
    schema = output_format["schema"]
    assert set(schema["required"]) == set(provider_stub.payload)
    assert set(schema["properties"]) == set(provider_stub.payload)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["program"] == {"type": "string"}
    assert schema["properties"]["parameters_to_refine"] == {
        "type": "array", "items": {"type": "string"},
    }
    assert {key: proposal[key] for key in provider_stub.payload} == provider_stub.payload
    assert proposal["response_id"] == "resp_mock_only"
    assert proposal["requested_service_tier"] == requested
    assert proposal["service_tier"] == returned


def test_health_and_bootstrap_expose_requested_tier(monkeypatch):
    monkeypatch.setattr(server, "SERVICE_TIER", "default")
    monkeypatch.setattr(server, "render_result", lambda *args, **kwargs: {"id": "baseline"})
    client = TestClient(server.app)
    for endpoint in ("health", "bootstrap"):
        response = client.get(f"/api/{endpoint}")
        assert response.status_code == 200
        assert response.json()["requested_service_tier"] == "default"
        # No provider request has occurred, so this metadata cannot claim an actual tier.
        assert "service_tier" not in response.json()


def test_iteration_record_preserves_requested_and_returned_tiers(
    monkeypatch, tmp_path, provider_stub,
):
    monkeypatch.setattr(server, "SERVICE_TIER", "fast")
    provider_stub.response.service_tier = "default"
    monkeypatch.setattr(server, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()

    def rendered(history, *args, **kwargs):
        return {"id": "aaaaaaaaaaaa", "scene_id": server.SCENE_ID,
                "baseline_id": server.BASELINE_ID,
                "program": server.history_to_program(history),
                "metrics": {"miou": 0.03, "score": 0.025},
                "map_image": "data:image/png;base64,prediction",
                "error_image": "data:image/png;base64,mismatch"}

    monkeypatch.setattr(server, "render_result", rendered)
    client = TestClient(server.app)
    response = client.post("/api/iterate", json={
        "program": server.BASELINE, "refine": False,
        "scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID,
    })
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["status", "proposal", "status", "result"]
    record = json.loads((tmp_path / "data/runs/aaaaaaaaaaaa.json").read_text())
    for value in (events[1], events[-1], record, record["proposal"]):
        assert value["requested_service_tier"] == "fast"
        assert value["service_tier"] == "default"
    provider_stub.create.assert_awaited_once()
