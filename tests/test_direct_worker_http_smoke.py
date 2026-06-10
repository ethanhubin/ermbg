from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_direct_worker_http.py"
    spec = importlib.util.spec_from_file_location("smoke_direct_worker_http", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _png_base64(rgb: np.ndarray) -> str:
    import io

    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_direct_worker_http_smoke_writes_standard_manifests(monkeypatch, tmp_path):
    module = _load_script_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    rgb = np.full((4, 5, 3), (0, 200, 0), dtype=np.uint8)
    image_path = tmp_path / "samples" / "case" / "green.png"
    image_path.parent.mkdir(parents=True)
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
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, timeout):
        del url, timeout
        return FakeResponse({"status": "ok", "git_sha": "test"})

    def fake_post(url, files, data, timeout):
        del url, files, data, timeout
        return FakeResponse(
            {
                "status": "ok",
                "case_count": 1,
                "ok_count": 1,
                "runs": [
                    {
                        "status": "ok",
                        "filename": "B001_case_green_green.png",
                        "algorithm": "pymatting_known_b",
                        "execution_backend": "direct-pymatting-known-b",
                        "route": "pymatting_known_b",
                        "asset_kind": "button",
                        "parameter_profile": "known_b",
                        "execution_profile": "pymatting-hard-button",
                        "rgba_png_base64": _png_base64(rgb),
                    }
                ],
            }
        )

    monkeypatch.setattr(module.requests, "get", fake_get)
    monkeypatch.setattr(module.requests, "post", fake_post)

    args = argparse.Namespace(
        base_url="http://worker.test",
        manifest=manifest_path,
        out_dir=tmp_path / "out" / "smoke",
        sample_id="B001",
        all=False,
        category="",
        shadow_mode="auto",
        corridorkey_screen_mode="auto",
        corridorkey_preset="auto",
        fallback_bg_color=(0, 200, 0),
        timeout=1.0,
        write_images=True,
    )

    summary = module.run(args)

    batch_manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    case_dir = args.out_dir / "B001_case_green_green"
    case_manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    case_summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["artifact_manifest"] == "out/smoke/manifest.json"
    assert batch_manifest["schema"] == "ermbg.run.v1"
    assert batch_manifest["outputs"]["summary"] == "summary.json"
    assert batch_manifest["extra"]["case_manifests"] == ["out/smoke/B001_case_green_green/manifest.json"]
    assert case_manifest["schema"] == "ermbg.run.v1"
    assert case_manifest["outputs"]["rgba"] == "rgba.png"
    assert case_manifest["report"] == "summary.json"
    assert case_summary["fixed_backend"] == "auto"
    assert case_summary["actual_execution_backend"] == "direct-pymatting-known-b"
