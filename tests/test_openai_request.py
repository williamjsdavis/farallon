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

    assert [message["role"] for message in call["input"]] == ["user", "user"]
    reference, dynamic = [message["content"] for message in call["input"]]
    context = json.loads(dynamic[0]["text"])
    assert context["current_program"] == request.program
    assert context["current_metrics"] == current["metrics"]
    assert context["recent_attempts"] == [previous]
    fixed = json.loads(reference[-1]["text"])
    assert set(fixed) == {"bounds_km", "image_axes", "legend", "observed_fraction", "parameter_bounds", "terrain"}
    assert fixed["bounds_km"] == server.META["bounds"]
    assert fixed["image_axes"] == "top=north/right=east"
    assert fixed["legend"] == [{key: entry[key] for key in ("id", "name", "color", "age")}
                               for entry in server.META["palette"]]
    assert fixed["observed_fraction"] == server.META["labeledFraction"]
    assert fixed["parameter_bounds"] == server.PARAM_SPECS
    assert fixed["terrain"]["source"] == server.TERRAIN_META["source"]
    assert fixed["terrain"]["datum_m"] == server.TERRAIN_META["datum_m"]
    assert fixed["terrain"]["min_z_km"] == float(server.HEIGHTS.min())
    assert fixed["terrain"]["max_z_km"] == float(server.HEIGHTS.max())
    assert fixed["terrain"]["height_samples_5x5_north_up_km"] == server.HEIGHTS[server.np.ix_(
        server.np.linspace(0, len(server.YS) - 1, 5).astype(int),
        server.np.linspace(0, len(server.XS) - 1, 5).astype(int))].round(4).tolist()
    assert call["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert server.SCENE_ID in call["prompt_cache_key"]
    content = reference + dynamic
    breakpoints = [index for index, item in enumerate(content) if "prompt_cache_breakpoint" in item]
    assert breakpoints == [len(reference) - 1]
    assert reference[-1]["type"] == "input_text"
    assert reference[-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    images = [item for item in content if item["type"] == "input_image"]
    assert [item["image_url"] for item in images] == [
        f"data:image/png;base64,{server.META['sourceImage'].split('/')[-1]}",
        "data:image/png;base64,target.png", server.TERRAIN_IMAGE,
        current["map_image"], current["error_image"],
    ]
    assert all(item["detail"] == "high" for item in images)
    labels = [content[index - 1]["text"] for index, item in enumerate(content) if item["type"] == "input_image"]
    assert labels == [
        "Original map with legend; only the NW crop is scored:",
        "Clean observed NW target; transparent gaps are unobserved. Same bounds as prediction:",
        "Fixed USGS terrain in the same frame: dark teal is low elevation, pale yellow is high; shading indicates relief:",
        "Current predicted surface, same north-up frame and geological colors:",
        "Disagreement overlay: red is mismatch; green is agreement; transparent is unobserved:",
    ]
    assert "stream" not in call
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


def test_cached_reference_prefix_survives_changed_iteration_context(provider_stub):
    first_request = server.IterationRequest(program=server.BASELINE)
    first_current = {"metrics": {"miou": 0.03, "score": 0.025},
                     "map_image": "data:image/png;base64,first-prediction",
                     "error_image": "data:image/png;base64,first-error"}
    asyncio.run(server.propose(first_request, first_current))
    first = provider_stub.create.call_args.kwargs
    previous = [{"scene_id": server.SCENE_ID, "baseline_id": server.BASELINE_ID,
                 "headline": f"Attempt {index}"} for index in range(6)]
    second_request = server.IterationRequest(
        program=server.BASELINE.replace("-1.2", "-1.3"),
        previous=[*previous, {"scene_id": "stale", "baseline_id": server.BASELINE_ID}],
        auto_context={"iteration": 7, "branch": 2, "stall_count": 1})
    second_current = {"metrics": {"miou": 0.4, "score": 0.35},
                      "map_image": "data:image/png;base64,new-prediction",
                      "error_image": "data:image/png;base64,new-error"}
    asyncio.run(server.propose(second_request, second_current))
    second = provider_stub.create.call_args.kwargs

    assert first["input"][0] == second["input"][0]
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert first["instructions"] == second["instructions"]
    assert first["text"] == second["text"]
    assert first["input"][1] != second["input"][1]
    dynamic = second["input"][1]["content"]
    assert json.loads(dynamic[0]["text"]) == {
        "current_program": second_request.program, "current_metrics": second_current["metrics"],
        "recent_attempts": previous[-4:], "auto": second_request.auto_context,
    }
    assert [item["image_url"] for item in dynamic if item["type"] == "input_image"] == [
        second_current["map_image"], second_current["error_image"],
    ]
    assert not any("prompt_cache_breakpoint" in item for item in dynamic)


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
