from __future__ import annotations

import argparse
import base64
import io
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_corridorkey_game_eval.py"
    spec = importlib.util.spec_from_file_location("run_corridorkey_game_eval", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_game_eval_direct_worker_writes_standard_outputs(monkeypatch, tmp_path):
    module = _load_script_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    image_path = tmp_path / "samples" / "case" / "green.png"
    image_path.parent.mkdir(parents=True)
    rgb = np.full((8, 10, 3), (0, 200, 0), dtype=np.uint8)
    rgb[2:6, 3:7] = (220, 40, 40)
    Image.fromarray(rgb, mode="RGB").save(image_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "case_green",
                        "sample_id": "B001",
                        "category": "button",
                        "screen": "green",
                        "input": "samples/case/green.png",
                        "green": "samples/case/green.png",
                        "backgrounds": {"green": [0, 200, 0]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_direct_worker(image, **kwargs):
        captured["image"] = image
        captured.update(kwargs)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
        trimap_buf = io.BytesIO()
        Image.fromarray(np.full(rgb.shape[:2], 128, dtype=np.uint8), mode="L").save(trimap_buf, format="PNG")
        hint_buf = io.BytesIO()
        Image.fromarray(np.full(rgb.shape[:2], 255, dtype=np.uint8), mode="L").save(hint_buf, format="PNG")
        return SimpleNamespace(
            rgba=rgba,
            alpha=np.ones(rgb.shape[:2], dtype=np.float32),
            foreground_srgb=rgb,
            debug={
                "timings": {"server_elapsed_sec": 0.42},
                "direct_worker": {
                    "execution_backend": "direct-pymatting-known-b",
                    "trimap_png_base64": base64.b64encode(trimap_buf.getvalue()).decode("ascii"),
                    "corridorkey_hint_png_base64": base64.b64encode(hint_buf.getvalue()).decode("ascii"),
                },
            },
        )

    monkeypatch.setattr(module, "matte_image_direct_worker", fake_direct_worker)

    out_dir = tmp_path / "out" / "direct_worker"
    args = argparse.Namespace(
        manifest=manifest_path,
        out_dir=out_dir,
        backend="direct-worker",
        sample_id="B001",
        category="",
        comfy_url="http://unused",
        direct_worker_url="http://worker.test",
        subject_threshold=35.0,
        corridorkey_preset="detail_safe",
    )

    summary = module.run(args)

    case_dir = out_dir / "B001_case_green_green"
    assert summary["ok_count"] == 1
    assert summary["artifact_manifest"] == "out/direct_worker/manifest.json"
    assert summary["runs"][0]["backend"] == "direct-worker"
    assert summary["runs"][0]["timings"]["server_elapsed_sec"] == 0.42
    assert summary["timing_summary"]["overall"]["timings"]["server_elapsed_sec"]["avg"] == 0.42
    assert captured["image"] == image_path
    assert captured["direct_worker_url"] == "http://worker.test"
    assert captured["corridorkey_preset"] == "detail_safe"
    assert "corridorkey_hard_ui_hint_mode" not in captured
    assert (case_dir / "rgba.png").exists()
    assert (case_dir / "alpha.png").exists()
    assert (case_dir / "foreground.png").exists()
    assert (case_dir / "trimap.png").exists()
    assert (case_dir / "corridorkey_hint.png").exists()
    assert (case_dir / "contact_sheet.png").exists()
    case_summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    assert case_summary["status"] == "ok"
    assert case_summary["artifact_manifest"] == "out/direct_worker/B001_case_green_green/manifest.json"
    case_manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    assert case_manifest["schema"] == "ermbg.run.v1"
    assert case_manifest["request"]["backend"] == "direct-worker"
    assert case_manifest["request"]["effective_backend"] == "direct-worker"
    assert case_manifest["outputs"]["rgba"] == "rgba.png"
    assert case_manifest["outputs"]["alpha"] == "alpha.png"
    assert case_manifest["outputs"]["foreground"] == "foreground.png"
    assert case_manifest["outputs"]["trimap"] == "trimap.png"
    assert case_manifest["outputs"]["hint"] == "corridorkey_hint.png"
    assert case_manifest["runtime"]["backend"] == "direct-worker"
    batch_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert batch_manifest["schema"] == "ermbg.run.v1"
    assert batch_manifest["runtime"]["kind"] == "game-eval"
    assert batch_manifest["request"]["backend"] == "direct-worker"
    assert batch_manifest["outputs"]["summary"] == "summary.json"
    assert batch_manifest["extra"]["case_manifests"] == [
        "out/direct_worker/B001_case_green_green/manifest.json"
    ]
