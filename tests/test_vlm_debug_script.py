"""Smoke tests for the local VLM planner debug script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "07_vlm_planner_debug.py"
    spec = importlib.util.spec_from_file_location("vlm_planner_debug_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_debug_inputs(tmp_path: Path) -> tuple[Path, Path]:
    h, w = 64, 64
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - 28) ** 2 + (xx - 28) ** 2)
    ring = (r <= 18) & (r >= 7)
    hole = r < 7
    image[ring] = (220, 20, 20)
    rgba[ring, :3] = image[ring]
    rgba[ring, 3] = 255
    rgba[hole, :3] = 255
    rgba[hole, 3] = 0

    # A low-alpha black stroke near opaque support gives the fixture a second
    # region type for combined preserve/fill + hard-edge plans.
    image[48, 16:48] = (10, 10, 10)
    rgba[48, 16:48, :3] = image[48, 16:48]
    rgba[48, 16:48, 3] = 64
    rgba[49:52, 16:48, :3] = (220, 20, 20)
    rgba[49:52, 16:48, 3] = 255

    input_path = tmp_path / "input.png"
    rgba_path = tmp_path / "base_rgba.png"
    Image.fromarray(image, mode="RGB").save(input_path)
    Image.fromarray(rgba, mode="RGBA").save(rgba_path)
    return input_path, rgba_path


def test_vlm_debug_script_exports_request_and_executes_fixture(tmp_path):
    module = _load_script_module()
    input_path, rgba_path = _write_debug_inputs(tmp_path)
    out_dir = tmp_path / "vlm_debug"

    request_summary = module.run(
        input_path,
        rgba_path,
        (255, 255, 255),
        out_dir,
        coalesce=False,
        max_region_crops=2,
    )
    request_payload = json.loads((out_dir / "vlm_request.json").read_text())
    region_ids = [region["id"] for region in request_payload["planner_bundle"]["regions"]]
    assert "same_bg_0" in region_ids
    hard_edge_id = next(region_id for region_id in region_ids if region_id.startswith("hard_edge_"))
    assert request_summary["attachment_count"] == len(request_payload["attachments"])
    assert (out_dir / "attachments" / "evidence_overlay.png").exists()

    fixture_path = tmp_path / "fixture_response.json"
    fixture_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "transparent_plus_edge",
                        "label": "Transparent + edge",
                        "confidence": 0.7,
                        "selected": True,
                        "operations": [
                            {"tool": "preserve_hole", "region_id": "same_bg_0"},
                            {
                                "tool": "snap_hard_edge",
                                "region_id": hard_edge_id,
                                "parameters": {"alpha_floor": 0.95},
                            },
                        ],
                        "reason": "Fixture combined interpretation.",
                    },
                    {
                        "id": "fill_plus_edge",
                        "label": "Fill + edge",
                        "confidence": 0.6,
                        "operations": [
                            {
                                "tool": "fill_same_color_region",
                                "region_id": "same_bg_0",
                                "parameters": {"alpha_floor": 1.0},
                            },
                            {
                                "tool": "snap_hard_edge",
                                "region_id": hard_edge_id,
                                "parameters": {"alpha_floor": 0.95},
                            },
                        ],
                        "reason": "Fixture alternate interpretation.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    fixture_summary = module.run(
        input_path,
        rgba_path,
        (255, 255, 255),
        out_dir,
        fixture_path=fixture_path,
        coalesce=False,
        max_region_crops=2,
    )

    assert fixture_summary["candidate_count"] == 2
    assert (out_dir / "candidate_plans.json").exists()
    assert (out_dir / "candidates" / "transparent_plus_edge.png").exists()
    assert (out_dir / "candidates" / "fill_plus_edge.png").exists()


def test_vlm_debug_script_executes_mocked_openai_provider(tmp_path, monkeypatch):
    module = _load_script_module()
    input_path, rgba_path = _write_debug_inputs(tmp_path)
    out_dir = tmp_path / "openai_debug"

    class FakeOpenAIClient:
        def __init__(self, model, timeout, env_path):
            self.model = model
            self.timeout = timeout
            self.env_path = env_path
            self.last_request_payload = {"model": model, "mock": True}
            self.last_raw_response = {"output_text": '{"candidates":[]}'}

        def plan_request(self, request):
            del request
            from ermbg.planner import CandidatePlan, PlanOperation

            return [
                CandidatePlan(
                    id="mock_openai",
                    label="Mock OpenAI",
                    selected=True,
                    operations=[PlanOperation(tool="preserve_hole", region_id="same_bg_0")],
                    reason="Mock provider response.",
                )
            ]

    monkeypatch.setattr(module, "OpenAIVLMPlannerClient", FakeOpenAIClient)

    summary = module.run(
        input_path,
        rgba_path,
        (255, 255, 255),
        out_dir,
        provider="openai",
        openai_model="gpt-4o-mini",
        openai_timeout=2.0,
        env_path=tmp_path / ".env",
        coalesce=False,
        max_region_crops=1,
    )

    assert summary["provider"] == "openai"
    assert summary["candidate_count"] == 1
    assert (out_dir / "openai_request.json").exists()
    assert (out_dir / "vlm_raw_response.json").exists()
    assert (out_dir / "candidates" / "mock_openai.png").exists()
