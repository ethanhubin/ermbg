"""Tests for VLM semantic-prior region classification."""

from __future__ import annotations

import json

import cv2
import numpy as np

from ermbg import io
from ermbg.planner import RiskRegion
from ermbg.vlm_semantic import (
    OpenAIVLMSemanticPriorClient,
    build_openai_semantic_payload,
    build_vlm_semantic_request,
    extract_shadow_candidate_regions,
    extract_subject_material_candidate_regions,
    parse_semantic_prior_payload,
    parse_qwen_json_text,
)


def _green_panel_case():
    image = np.full((80, 100, 3), [0, 200, 0], dtype=np.uint8)
    image[20:60, 18:82] = [30, 120, 70]
    alpha = np.zeros((80, 100), dtype=np.float32)
    alpha[20:60, 18:82] = 1.0
    return image, alpha, (0, 200, 0)


def _green_shadow_case():
    h, w = 80, 120
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.broadcast_to(bg, (h, w, 3)).copy()
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[20:48, 36:82] = 1.0
    shadow = np.zeros((h, w), dtype=np.float32)
    hard = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(hard, (62, 55), (34, 10), 0.0, 0.0, 360.0, 1.0, -1)
    shadow = cv2.GaussianBlur(hard, (11, 11), sigmaX=3.0) * 0.45
    shadow[alpha > 0] = 0.0
    B = io.srgb_to_linear(bg.reshape(1, 1, 3))[0, 0]
    C = (1.0 - shadow[..., None]) * B
    image = io.linear_to_srgb_u8(C)
    image[alpha > 0] = [220, 30, 30]
    return image, alpha, tuple(int(c) for c in bg), shadow


def test_extract_subject_material_candidate_regions_finds_green_subject_region():
    image, alpha, bg = _green_panel_case()

    regions = extract_subject_material_candidate_regions(image, alpha, bg, min_area_ratio=0.001)

    assert regions
    assert regions[0].kind == "subject_material_candidate"
    assert regions[0].mask[30:50, 30:70].mean() > 0.8


def test_extract_shadow_candidate_regions_finds_measured_shadow_support():
    image, alpha, bg, shadow = _green_shadow_case()

    regions = extract_shadow_candidate_regions(
        image,
        alpha,
        bg,
        shadow_alpha=shadow,
        min_area_ratio=0.001,
    )

    assert regions
    assert regions[0].kind == "owned_shadow_candidate"
    assert regions[0].mask[52:60, 45:78].mean() > 0.5


def test_parse_semantic_prior_payload_builds_material_protect_mask():
    image, alpha, bg = _green_panel_case()
    regions = extract_subject_material_candidate_regions(image, alpha, bg, min_area_ratio=0.001)
    payload = {
        "shadow_allowed": False,
        "regions": [
            {
                "region_id": regions[0].id,
                "role": "subject_material",
                "confidence": 0.9,
                "reason": "green panel fill belongs to the object",
            }
        ],
    }

    prior = parse_semantic_prior_payload(payload, regions, alpha.shape)

    assert prior.subject_material_mask is not None
    assert prior.subject_material_mask[30:50, 30:70].mean() > 0.8
    assert prior.subject_mask is not None
    assert prior.shadow_allowed is True
    assert prior.to_dict()["subject_material_pixels"] > 1000


def test_parse_semantic_prior_payload_can_disallow_shadow_candidates():
    shape = (40, 60)
    mask = np.zeros(shape, dtype=bool)
    mask[24:32, 18:46] = True
    regions = [RiskRegion(id="shadow_candidate_0", kind="owned_shadow_candidate", mask=mask)]
    payload = {
        "shadow_allowed": False,
        "regions": [
            {
                "region_id": "shadow_candidate_0",
                "role": "background",
                "confidence": 0.9,
                "reason": "not an owned shadow",
            }
        ],
    }

    prior = parse_semantic_prior_payload(payload, regions, shape)

    assert prior.shadow_allowed is False
    assert prior.shadow_ownership_mask is None


def test_parse_semantic_prior_payload_accepts_owned_shadow_mask():
    shape = (40, 60)
    mask = np.zeros(shape, dtype=bool)
    mask[24:32, 18:46] = True
    regions = [RiskRegion(id="shadow_candidate_0", kind="owned_shadow_candidate", mask=mask)]
    payload = {
        "shadow_allowed": True,
        "regions": [
            {
                "region_id": "shadow_candidate_0",
                "role": "shadow",
                "confidence": 0.9,
                "reason": "owned contact shadow",
            }
        ],
    }

    prior = parse_semantic_prior_payload(payload, regions, shape)

    assert prior.shadow_allowed is True
    assert prior.shadow_ownership_mask is not None
    assert int(prior.shadow_ownership_mask.sum()) == int(mask.sum())


def test_build_openai_semantic_payload_uses_region_schema_and_images():
    image, alpha, bg = _green_panel_case()
    regions = extract_subject_material_candidate_regions(image, alpha, bg, min_area_ratio=0.001)
    request = build_vlm_semantic_request(
        image_srgb=image,
        subject_alpha=alpha,
        background_color=bg,
        regions=regions[:1],
        thumbnail_max_side=32,
        crop_max_side=24,
    )
    payload = build_openai_semantic_payload(request, model="gpt-4o-mini")

    assert payload["model"] == "gpt-4o-mini"
    assert payload["text"]["format"]["schema"]["required"] == ["regions", "shadow_allowed"]
    region_enum = payload["text"]["format"]["schema"]["properties"]["regions"]["items"]["properties"]["region_id"]["enum"]
    assert region_enum == [regions[0].id]
    image_items = [item for item in payload["input"][1]["content"] if item["type"] == "input_image"]
    assert image_items
    assert image_items[0]["image_url"].startswith("data:image/png;base64,")


def test_openai_semantic_client_posts_and_parses(monkeypatch):
    image, alpha, bg = _green_panel_case()
    regions = extract_subject_material_candidate_regions(image, alpha, bg, min_area_ratio=0.001)
    request = build_vlm_semantic_request(
        image_srgb=image,
        subject_alpha=alpha,
        background_color=bg,
        regions=regions[:1],
        thumbnail_max_side=32,
        crop_max_side=24,
    )
    response_text = json.dumps(
        {
            "shadow_allowed": True,
            "regions": [
                {
                    "region_id": regions[0].id,
                    "role": "subject_material",
                    "confidence": 0.85,
                    "reason": "mocked",
                }
            ],
        }
    )

    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"output_text": response_text}

    def fake_post(url, headers, json, timeout):
        assert headers["Authorization"] == "Bearer test-key"
        assert json["text"]["format"]["name"] == "ermbg_semantic_prior"
        assert timeout == 3.0
        return FakeResponse()

    import ermbg.vlm_semantic as vlm_semantic

    monkeypatch.setattr(vlm_semantic.requests, "post", fake_post)
    client = OpenAIVLMSemanticPriorClient(api_key="test-key", timeout=3.0)

    prior = client.classify_request(request, regions[:1], alpha.shape)

    assert prior.subject_material_mask is not None
    assert client.last_raw_response == {"output_text": response_text}


def test_parse_qwen_json_text_handles_previewany_wrapped_string():
    wrapped = json.dumps(['{"shadow_allowed": true, "regions": []}'])
    assert parse_qwen_json_text(wrapped) == {"shadow_allowed": True, "regions": []}
