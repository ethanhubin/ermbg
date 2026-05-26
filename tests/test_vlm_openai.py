"""Tests for the optional OpenAI VLM planner client."""

from __future__ import annotations

import json

import numpy as np

from ermbg.planner import RiskRegion
from ermbg.vlm_openai import (
    OpenAIVLMPlannerClient,
    build_openai_responses_payload,
    extract_openai_output_text,
)
from ermbg.vlm_payload import build_vlm_planner_request


def _request():
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    rgba = np.zeros((12, 12, 4), dtype=np.uint8)
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3:9] = True
    region = RiskRegion(id="same_bg_0", kind="same_bg_enclosed_region", mask=mask)
    return build_vlm_planner_request(
        image_srgb=image,
        base_rgba=rgba,
        regions=[region],
        background_color=(255, 255, 255),
        max_region_crops=1,
        thumbnail_max_side=16,
        crop_max_side=8,
    )


def test_build_openai_responses_payload_uses_images_and_structured_output():
    payload = build_openai_responses_payload(_request(), model="gpt-4o-mini")

    assert payload["model"] == "gpt-4o-mini"
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["properties"]["candidates"]["maxItems"] == 4
    user_content = payload["input"][1]["content"]
    assert user_content[0]["type"] == "input_text"
    image_items = [item for item in user_content if item["type"] == "input_image"]
    assert image_items
    assert image_items[0]["image_url"].startswith("data:image/png;base64,")


def test_extract_openai_output_text_handles_response_output_shape():
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"candidates":[]}',
                    }
                ],
            }
        ]
    }

    assert extract_openai_output_text(payload) == '{"candidates":[]}'


def test_openai_client_posts_and_parses_candidate_json(monkeypatch):
    request = _request()
    response_text = json.dumps(
        {
            "candidates": [
                {
                    "id": "keep",
                    "label": "Keep transparent",
                    "confidence": 0.7,
                    "selected": True,
                    "operations": [{"tool": "preserve_hole", "region_id": "same_bg_0"}],
                    "reason": "Mocked response.",
                }
            ]
        }
    )

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"output_text": response_text}

    def fake_post(url, headers, json, timeout):
        assert url == "https://api.openai.com/v1/responses"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["text"]["format"]["schema"]["required"] == ["candidates"]
        assert timeout == 3.0
        return FakeResponse()

    import ermbg.vlm_openai as vlm_openai

    monkeypatch.setattr(vlm_openai.requests, "post", fake_post)
    client = OpenAIVLMPlannerClient(api_key="test-key", timeout=3.0)

    plans = client.plan_request(request)

    assert [plan.id for plan in plans] == ["keep"]
    assert client.last_raw_response == {"output_text": response_text}
