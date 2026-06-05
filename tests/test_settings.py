from __future__ import annotations

import json

import pytest

from ermbg import settings


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    monkeypatch.delenv("ERMBG_DIRECT_URL", raising=False)
    monkeypatch.setattr(settings, "_dotenv_paths", lambda: ())


def test_local_config_overrides_tracked_config(monkeypatch, tmp_path):
    base = tmp_path / "ermbg.config.json"
    local = tmp_path / "ermbg.local.json"
    base.write_text(
        json.dumps(
            {
                "services": {
                    "direct_worker_url": "http://base-worker:7871",
                    "comfy_url": "http://base-comfy:8000",
                },
                "web": {
                    "auto_backend": "direct-worker",
                    "auto_fallback_backend": "pymatting-known-b",
                    "enable_comfy": False,
                },
            }
        ),
        encoding="utf-8",
    )
    local.write_text(
        json.dumps(
            {
                "services": {"comfy_url": "http://local-comfy:8000"},
                "web": {"enable_comfy": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "CONFIG_PATH", base)
    monkeypatch.setattr(settings, "LOCAL_CONFIG_PATH", local)

    assert settings.get_direct_worker_url() == "http://base-worker:7871"
    assert settings.get_comfy_url() == "http://local-comfy:8000"
    assert settings.get_setting("web.auto_backend") == "direct-worker"
    assert settings.get_bool_setting("web.enable_comfy") is True


def test_environment_overrides_local_config(monkeypatch, tmp_path):
    base = tmp_path / "ermbg.config.json"
    local = tmp_path / "ermbg.local.json"
    base.write_text(json.dumps({"services": {"comfy_url": "http://base-comfy:8000"}}), encoding="utf-8")
    local.write_text(json.dumps({"services": {"comfy_url": "http://local-comfy:8000"}}), encoding="utf-8")
    monkeypatch.setattr(settings, "CONFIG_PATH", base)
    monkeypatch.setattr(settings, "LOCAL_CONFIG_PATH", local)
    monkeypatch.setenv("COMFY_URL", "http://env-comfy:8000")

    assert settings.get_comfy_url() == "http://env-comfy:8000"


def test_missing_config_uses_project_fallbacks(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "missing.config.json")
    monkeypatch.setattr(settings, "LOCAL_CONFIG_PATH", tmp_path / "missing.local.json")
    monkeypatch.setattr(settings, "_dotenv_paths", lambda: ())
    monkeypatch.delenv("COMFY_URL", raising=False)
    monkeypatch.delenv("ERMBG_DIRECT_URL", raising=False)

    assert settings.get_comfy_url() == settings.DEFAULT_COMFY_URL
    assert settings.get_direct_worker_url() == settings.DEFAULT_DIRECT_WORKER_URL


def test_direct_worker_endpoints_keep_default_and_local_override(monkeypatch, tmp_path):
    base = tmp_path / "ermbg.config.json"
    local = tmp_path / "ermbg.local.json"
    base.write_text(
        json.dumps({"services": {"direct_worker_url": "http://127.0.0.1:7871"}}),
        encoding="utf-8",
    )
    local.write_text(
        json.dumps({"services": {"direct_worker_url": "http://192.168.0.8:7871"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "CONFIG_PATH", base)
    monkeypatch.setattr(settings, "LOCAL_CONFIG_PATH", local)
    monkeypatch.delenv("ERMBG_DIRECT_URL", raising=False)

    assert settings.get_direct_worker_url() == "http://192.168.0.8:7871"
    assert settings.get_direct_worker_endpoints() == {
        "local": "http://127.0.0.1:7871",
        "remote": "http://192.168.0.8:7871",
    }
