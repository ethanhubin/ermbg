"""Tests for VLM planner request packaging."""

from __future__ import annotations

import base64
import json
from io import BytesIO

import numpy as np
from PIL import Image

from ermbg.planner import RiskRegion
from ermbg.vlm_payload import build_vlm_planner_request


def _decode_png(data_base64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(data_base64)))


def test_build_vlm_planner_request_packages_json_and_visual_context():
    image = np.full((24, 32, 3), 255, dtype=np.uint8)
    image[6:18, 6:22] = (210, 20, 20)
    rgba = np.zeros((24, 32, 4), dtype=np.uint8)
    rgba[6:18, 6:22, :3] = image[6:18, 6:22]
    rgba[6:18, 6:22, 3] = 255

    same_bg_mask = np.zeros((24, 32), dtype=bool)
    same_bg_mask[9:14, 11:16] = True
    hard_edge_mask = np.zeros((24, 32), dtype=bool)
    hard_edge_mask[6:18, 6:8] = True
    regions = [
        RiskRegion(id="same_bg_0", kind="same_bg_enclosed_region", mask=same_bg_mask),
        RiskRegion(id="hard_edge_0", kind="hard_edge_candidate", mask=hard_edge_mask),
    ]

    request = build_vlm_planner_request(
        image_srgb=image,
        base_rgba=rgba,
        regions=regions,
        background_color=(255, 255, 255),
        max_region_crops=1,
        thumbnail_max_side=20,
        crop_max_side=12,
        crop_padding_px=2,
    )
    payload = request.to_dict()

    json.dumps(payload)
    assert payload["planner_bundle"]["regions"][0]["evidence_kind"] == "same_bg_low_alpha_enclosed"
    assert payload["planner_bundle"]["tools"][0]["name"] == "preserve_hole"
    assert payload["response_schema"]["properties"]["candidates"]["maxItems"] == 4
    op_schema = payload["response_schema"]["properties"]["candidates"]["items"]["properties"]["operations"]
    assert "snap_hard_edge" in op_schema["items"]["properties"]["tool"]["enum"]
    assert op_schema["items"]["properties"]["region_id"]["enum"] == ["same_bg_0", "hard_edge_0"]

    attachment_ids = [attachment["id"] for attachment in payload["attachments"]]
    assert attachment_ids == [
        "original_thumbnail",
        "base_on_checker",
        "base_on_black",
        "base_on_white",
        "evidence_overlay",
        "region_crop_0",
    ]
    thumb = _decode_png(payload["attachments"][0]["data_base64"])
    assert max(thumb.size) <= 20
    crop = payload["attachments"][-1]
    assert crop["region_id"] == "same_bg_0"
    assert crop["metadata"]["evidence_kind"] == "same_bg_low_alpha_enclosed"
    assert max(_decode_png(crop["data_base64"]).size) <= 12


def test_build_vlm_planner_request_rejects_mismatched_region_shape():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    bad_region = RiskRegion(
        id="bad",
        kind="same_bg_enclosed_region",
        mask=np.zeros((7, 8), dtype=bool),
    )

    try:
        build_vlm_planner_request(image_srgb=image, base_rgba=rgba, regions=[bad_region])
    except ValueError as e:
        assert "mask shape" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected shape validation failure")
