from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_direct_worker_path.py"
    spec = importlib.util.spec_from_file_location("benchmark_direct_worker_path", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_worker_benchmark_run_writes_summary_and_compare(monkeypatch, tmp_path):
    module = _load_script_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    image_path = tmp_path / "samples" / "case" / "green.png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8), mode="RGB").save(image_path)

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
                        "image_size": [3, 2],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    compare_path = tmp_path / "comfy_summary.json"
    compare_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "case": "B001_case_green_green",
                        "backend": "comfy-pymatting-known-b",
                        "elapsed_sec_client": 4.0,
                        "timings": {"remote_total_sec": 3.5},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_direct_matte(rgb, **kwargs):
        result = SimpleNamespace(
            rgba=np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)]),
            debug={"array": np.asarray([1, 2, 3], dtype=np.float32)},
        )
        timings = {"route_sec": 0.25, "backend_sec": 1.75}
        metadata = {
            "selected_backend": "comfy-pymatting-known-b",
            "execution_backend": "direct-pymatting-known-b",
            "route": "pymatting_known_b",
            "asset_kind": "button",
            "parameter_profile": "opaque_hard_ui_no_shadow",
            "execution_profile": "pymatting-hard-button",
        }
        return result, timings, metadata

    monkeypatch.setattr(module, "_direct_matte", fake_direct_matte)
    monkeypatch.setattr(module, "_runtime_info", lambda: {"python": "test"})

    out_dir = tmp_path / "out" / "direct"
    args = argparse.Namespace(
        manifest=manifest_path,
        out_dir=out_dir,
        sample_id="B001",
        warmup_sample_id="",
        all=False,
        category="",
        compare_summary=compare_path,
        shadow_mode="on",
        corridorkey_screen_mode="auto",
        corridorkey_preset="auto",
        corridorkey_hard_ui_hint_mode="bbox_2px",
        fallback_bg_color=(0, 200, 0),
        include_debug=False,
        write_images=True,
    )

    summary = module.run(args)

    assert summary["case_count"] == 1
    assert summary["ok_count"] == 1
    assert summary["runtime"] == {"python": "test"}
    row = summary["runs"][0]
    assert row["status"] == "ok"
    assert row["selected_backend"] == "comfy-pymatting-known-b"
    assert row["execution_backend"] == "direct-pymatting-known-b"
    assert row["execution_profile"] == "pymatting-hard-button"
    assert row["debug"] is None
    assert row["compare"]["speedup_vs_comfy_client"] > 0.0
    assert row["compare"]["saved_sec_vs_comfy_client"] > 0.0
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "B001_case_green_green" / "summary.json").exists()
    assert (out_dir / "B001_case_green_green" / "rgba.png").exists()
    assert "compare_speedup_vs_comfy_client" in summary["timing_summary"]["overall"]


def test_load_comfy_compare_rejects_missing_runs(tmp_path):
    module = _load_script_module()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"rows": []}), encoding="utf-8")

    try:
        module._load_comfy_compare(path)
    except ValueError as exc:
        assert "runs list" in str(exc)
    else:
        raise AssertionError("expected ValueError")
