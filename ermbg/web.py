"""Small web UI for ERMBG.

The service keeps the browser flow intentionally narrow: upload one image,
run ``matte_image``, preview the returned RGBA PNG, and download it.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Any
from urllib.parse import quote, unquote

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, Response
except ImportError as e:  # pragma: no cover - exercised only without web extra
    raise ImportError('Install the web extra with `uv pip install -e ".[web]"`.') from e

from . import io as ermbg_io
from .analyze import analyze_candidates
from .api import MatteResponse, classify_image_route, matte_image
from .artifacts import build_run_manifest, route_from_response, runtime_from_response, write_run_manifest
from .candidates import MatteCandidate, generate_matte_candidates
from .direct_worker_client import DEFAULT_DIRECT_WORKER_URL, matte_image_direct_worker
from .local_ownership import generate_local_ownership_candidate
from .pipeline_contracts import (
    AnalyzeResult,
    ExecutionRequest,
    PreprocessDecision,
    SemanticDecision,
    UserMaskDecision,
    semantic_manifest_summary,
)
from .preprocess import (
    BACKGROUND_REPAIR,
    analyze_input_preprocess,
    apply_input_preprocess,
    checkerboard_info_from_decision,
    repair_known_background_preprocess,
)
from .runtime_capabilities import collect_runtime_capabilities
from .settings import get_bool_setting, get_direct_worker_endpoints, get_direct_worker_servers, get_setting
from .slicer import (
    SliceBox,
    SliceResult,
    classify_ui_slice,
    crop_slice,
    find_slice_boxes,
    merge_overlapping_slice_boxes,
    pad_slice_box,
    slice_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


WEB_AUTO_BACKEND = get_setting("web.auto_backend", env="ERMBG_WEB_AUTO_BACKEND", default="direct-worker")
WEB_AUTO_FALLBACK_BACKEND = get_setting(
    "web.auto_fallback_backend",
    env="ERMBG_WEB_AUTO_FALLBACK_BACKEND",
    default="pymatting-known-b",
)
WEB_DIRECT_WORKER_URL = get_setting(
    "services.direct_worker_url",
    env="ERMBG_DIRECT_URL",
    default=DEFAULT_DIRECT_WORKER_URL,
).rstrip("/")
WEB_DIRECT_WORKER_ENDPOINTS = get_direct_worker_endpoints()
WEB_DIRECT_WORKER_SERVERS = get_direct_worker_servers()
WEB_ENABLE_COMFY = get_bool_setting("web.enable_comfy", env="ERMBG_ENABLE_COMFY", default=False)

ALLOWED_BACKENDS = {
    "auto",
    "auto-local",
    "pymatting-known-b",
    "direct-worker",
    "direct-pymatting-known-b",
    "direct-corridorkey",
    "direct-known-bg-glow",
    "corridorkey",
    "known-bg-glow",
    "known_bg_glow",
    "pymatting_known_b",
    "passthrough",
}
REMOTE_DIRECT_BACKENDS = {
    "passthrough",
    "corridorkey",
    "pymatting_known_b",
    "known-bg-glow",
    "known_bg_glow",
    "pymatting-known-b",
    "direct-worker",
    "direct-pymatting-known-b",
    "direct-corridorkey",
    "direct-known-bg-glow",
}
WEB_SHADOW_MODE = "on"
DEFAULT_GAME_EVAL_ROOT = PROJECT_ROOT / "out" / "local_ownership_full_20260527"
GAME_SAMPLE_REL = Path("samples") / "corridorkey_semantic"
LOCAL_OWNERSHIP_EVAL_PREFIX = "local_ownership_"
SOLID_GRAPHIC_EVAL_PREFIX = "solid_graphic_"
AUTO_EVAL_PREFIX = "auto_"
CORRIDORKEY_EVAL_PREFIX = "corridorkey_"
RMBG_EVAL_PREFIX = "rmbg_"
DIRECT_WORKER_EVAL_PREFIX = "direct_worker_"
WEB_MATTE_RUN_PREFIX = "web_matte_runs_"
LEGACY_MATTE_CANDIDATES_COMPAT = {
    "endpoint": "/api/matte-candidates",
    "status": "compatibility_layer",
    "deprecated": True,
    "replacement_flow": "Preprocess -> Analyze -> Decide -> Execute",
    "quality_validation_entrypoint": False,
}


@dataclass(frozen=True)
class WebExecutionRequest:
    """Internal Web form adapter for the Execute boundary."""

    file: UploadFile
    corridorkey_hint_mask: UploadFile | None = None
    user_keep_mask: UploadFile | None = None
    user_remove_mask: UploadFile | None = None
    backend: str = "auto"
    parameter_source: str = "auto"
    shadow_mode: str | None = None
    shadow_enabled: bool | None = None
    corridorkey_gamma_space: str = "sRGB"
    corridorkey_despill_strength: float = 1.0
    corridorkey_refiner_strength: float = 1.0
    corridorkey_auto_despeckle: str = "On"
    corridorkey_despeckle_size: int = 400
    corridorkey_auto_mask: bool = False
    corridorkey_screen_mode: str = "auto"
    corridorkey_preset: str = "auto"
    pymatting_method: str = "cf"
    pymatting_image_space: str = "linear"
    pymatting_bg_source: str = "auto"
    pymatting_bg_color: str = "0,200,0"
    pymatting_bg_threshold: float = 3.5
    pymatting_fg_threshold: float = 24.0
    pymatting_boundary_band_px: int = 2
    pymatting_adapt_bg_threshold: bool = False
    pymatting_adapt_fg_threshold: bool = True
    pymatting_adapt_boundary_band: bool = True
    pymatting_cg_maxiter: int = 1000
    pymatting_cg_rtol: float = 1e-6
    known_bg_glow_material_strength: float = 1.0
    background_repair: bool = False
    semantic_decision: str | None = None
    analysis_payload: dict[str, Any] | None = None
    execution_request_payload: dict[str, Any] | None = None


GAME_EVAL_RUN_PREFIXES = (
    LOCAL_OWNERSHIP_EVAL_PREFIX,
    SOLID_GRAPHIC_EVAL_PREFIX,
    AUTO_EVAL_PREFIX,
    CORRIDORKEY_EVAL_PREFIX,
    RMBG_EVAL_PREFIX,
    DIRECT_WORKER_EVAL_PREFIX,
)
FAST_GAME_EVAL_SAMPLE_IDS = ("B001", "B016", "B031", "B046", "I011", "I019", "C004", "C009")
GAME_EVAL_SCREENS = ("green", "blue")
# Fallback only applies in tests or broken installs where the manifest is not
# available. It mirrors the current B/I/C semantic manifest so progress does not
# silently drift back to the retired 78-sample set.
FALLBACK_GAME_EVAL_EXPECTED_TOTAL = 88
DEFAULT_GAME_EVAL_TEST_PATH = "auto"
GAME_EVAL_TEST_PATHS = {
    "auto": {
        "label": "Auto",
        "backend": "auto",
        "prefix": AUTO_EVAL_PREFIX,
    },
    "corridorkey": {
        "label": "CorridorKey",
        "backend": "corridorkey",
        "prefix": CORRIDORKEY_EVAL_PREFIX,
    },
    "rmbg": {
        "label": "RMBG",
        "backend": "rmbg",
        "prefix": RMBG_EVAL_PREFIX,
    },
    "direct-worker": {
        "label": "Direct Worker",
        "backend": "direct-worker",
        "prefix": DIRECT_WORKER_EVAL_PREFIX,
    },
}
SERVABLE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REGION_BOX_COLORS = {
    "same_bg_enclosed_region": (0, 153, 255, 235),
    "alpha_keyer_disagreement": (179, 92, 255, 235),
    "hard_edge_candidate": (255, 160, 0, 235),
}
REGION_FILL_COLORS = {
    "same_bg_enclosed_region": (0, 153, 255, 28),
    "alpha_keyer_disagreement": (179, 92, 255, 24),
    "hard_edge_candidate": (255, 160, 0, 24),
}

app = FastAPI(title="ERMBG Web", version="0.1.0")
_GAME_EVAL_JOBS: dict[str, dict[str, object]] = {}
_GAME_EVAL_JOBS_LOCK = Lock()
_SLICE_CACHE_MAX = 4
_SLICE_WEB_MAX_PIXELS = 4_000_000
_SLICE_CACHE: OrderedDict[tuple[str, int, int], SliceResult] = OrderedDict()
_SLICE_CACHE_LOCK = Lock()


def _direct_backend_base(backend: str) -> str:
    if backend.startswith("direct-worker:"):
        return "direct-worker"
    if backend.startswith("direct-pymatting-known-b:"):
        return "direct-pymatting-known-b"
    if backend.startswith("direct-corridorkey:"):
        return "direct-corridorkey"
    if backend.startswith("direct-known-bg-glow:"):
        return "direct-known-bg-glow"
    return backend


def _direct_backend_endpoint_name(backend: str) -> str | None:
    if ":" not in backend:
        return None
    base, name = backend.split(":", 1)
    if base not in {"direct-worker", "direct-pymatting-known-b", "direct-corridorkey", "direct-known-bg-glow"}:
        return None
    name = name.strip()
    return name or None


def _direct_worker_url_for_backend(backend: str) -> tuple[str, str | None]:
    endpoint_name = _direct_backend_endpoint_name(backend)
    if endpoint_name is None:
        return WEB_DIRECT_WORKER_URL, None
    try:
        return WEB_DIRECT_WORKER_ENDPOINTS[endpoint_name], endpoint_name
    except KeyError as exc:
        raise ValueError(f"unknown Direct Worker endpoint {endpoint_name!r}") from exc


def _direct_worker_servers_for_backend(backend: str) -> list[dict[str, Any]]:
    endpoint_name = _direct_backend_endpoint_name(backend)
    if endpoint_name is not None:
        url, name = _direct_worker_url_for_backend(backend)
        return [{"name": name or endpoint_name, "url": url, "priority": 0}]
    servers = [dict(server) for server in WEB_DIRECT_WORKER_SERVERS]
    primary_url = WEB_DIRECT_WORKER_URL.rstrip("/")
    if primary_url:
        servers = [server for server in servers if str(server.get("url", "")).rstrip("/") != primary_url]
        servers.insert(0, {"name": "primary", "url": primary_url, "priority": 0})
    return servers or [{"name": "primary", "url": primary_url, "priority": 0}]


def _execution_backend_for_algorithm(backend: str) -> str | None:
    base = _direct_backend_base(backend)
    return {
        "direct-worker": "auto",
        "corridorkey": "direct-corridorkey",
        "direct-corridorkey": "direct-corridorkey",
        "pymatting_known_b": "direct-pymatting-known-b",
        "direct-pymatting-known-b": "direct-pymatting-known-b",
        "known-bg-glow": "direct-known-bg-glow",
        "known_bg_glow": "direct-known-bg-glow",
        "direct-known-bg-glow": "direct-known-bg-glow",
    }.get(base)


def _is_allowed_backend(backend: str) -> bool:
    base = _direct_backend_base(backend)
    if base not in ALLOWED_BACKENDS:
        return False
    if base in {"direct-worker", "direct-pymatting-known-b", "direct-corridorkey", "direct-known-bg-glow"} and _direct_backend_endpoint_name(backend) is not None:
        return _direct_backend_endpoint_name(backend) in WEB_DIRECT_WORKER_ENDPOINTS
    return backend in ALLOWED_BACKENDS


def _allowed_backend_names() -> list[str]:
    names = sorted(ALLOWED_BACKENDS)
    for endpoint_name in sorted(WEB_DIRECT_WORKER_ENDPOINTS):
        names.append(f"direct-worker:{endpoint_name}")
        names.append(f"direct-pymatting-known-b:{endpoint_name}")
        names.append(f"direct-corridorkey:{endpoint_name}")
        names.append(f"direct-known-bg-glow:{endpoint_name}")
    return names


def _backend_options_html(*, selected: str = "auto") -> str:
    def option(value: str, label: str) -> str:
        selected_attr = " selected" if value == selected else ""
        return f'<option value="{html.escape(value)}"{selected_attr}>{html.escape(label)}</option>'

    rows = [
        option("auto", "Auto Route"),
        option("corridorkey", "CorridorKey"),
        option("pymatting_known_b", "PyMatting Known-B"),
        option("known-bg-glow", "Known-B Glow"),
        option("passthrough", "Passthrough"),
    ]
    return "".join(rows)


def _direct_worker_endpoint_options_html(*, runtime_labels: bool = False) -> str:
    primary_url = WEB_DIRECT_WORKER_URL.rstrip("/")
    prefix = "Direct · " if runtime_labels else ""

    def option(value: str, label: str, title: str = "") -> str:
        title_attr = f' title="{html.escape(title)}"' if title else ""
        return f'<option value="{html.escape(value)}"{title_attr}>{html.escape(label)}</option>'

    rows = [option("", f"{prefix}Auto primary", primary_url)]
    seen_urls: set[str] = set()
    for name, url in WEB_DIRECT_WORKER_ENDPOINTS.items():
        url = url.rstrip("/")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        marker = " · primary" if url == primary_url else ""
        rows.append(option(name, f"{prefix}{name}{marker}", url))
    return "".join(rows)


def _direct_worker_endpoint_form_select_html() -> str:
    has_remote = any(
        "127.0.0.1" not in url.lower() and "localhost" not in url.lower() and "[::1]" not in url.lower()
        for url in WEB_DIRECT_WORKER_ENDPOINTS.values()
    )
    if not has_remote and len(WEB_DIRECT_WORKER_ENDPOINTS) <= 1:
        return ""
    return (
        '<label class="direct-endpoint-field">Direct Worker'
        '<select id="direct-endpoint">'
        f'{_direct_worker_endpoint_options_html()}'
        '</select></label>'
    )


def _direct_worker_endpoint_runtime_select_html() -> str:
    return (
        '<span class="runtime-pill runtime-endpoint-pill" data-runtime="direct">'
        '<span class="runtime-dot" aria-hidden="true"></span>'
        '<select id="direct-endpoint" class="runtime-endpoint-select" aria-label="Direct Worker">'
        f'{_direct_worker_endpoint_options_html(runtime_labels=True)}'
        '</select></span>'
    )


def _inject_backend_options(html: str) -> str:
    old = (
        '<option value="auto" selected>Auto</option>'
        '<option value="direct-worker">direct-worker</option>'
        '<option value="direct-corridorkey">Direct Worker CorridorKey</option>'
        '<option value="direct-known-bg-glow">Direct Worker Known-B Glow</option>'
        '<option value="pymatting-known-b">pymatting-known-b</option>'
    )
    return html.replace(old, _backend_options_html()).replace(
        "__BACKEND_OPTIONS__",
        _backend_options_html(),
    ).replace(
        "__DIRECT_WORKER_RUNTIME_ENDPOINT__",
        _direct_worker_endpoint_runtime_select_html(),
    ).replace(
        "__DIRECT_WORKER_ENDPOINT_FIELD__",
        _direct_worker_endpoint_form_select_html(),
    )


def _game_sample_root() -> Path:
    return PROJECT_ROOT / GAME_SAMPLE_REL


def _game_sample_manifest() -> Path:
    return _game_sample_root() / "manifest.json"


def _encode_png(rgba: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _image_from_upload_bytes(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from e

    has_alpha = image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)
    return image.convert("RGBA" if has_alpha else "RGB")


def _load_upload_image(upload: UploadFile) -> Image.Image:
    return _image_from_upload_bytes(upload.file.read())


def _upload_mask_array(upload: UploadFile | None) -> np.ndarray | None:
    if upload is None:
        return None
    position = upload.file.tell()
    data = upload.file.read()
    upload.file.seek(position)
    if not data:
        return None
    mask = np.asarray(Image.open(BytesIO(data)).convert("L"), dtype=np.uint8)
    return mask > 127


def _user_mask_upload_summary(
    *,
    keep_mask: UploadFile | None,
    remove_mask: UploadFile | None,
) -> dict[str, object]:
    keep = _upload_mask_array(keep_mask)
    remove = _upload_mask_array(remove_mask)
    shape = keep.shape if keep is not None else (remove.shape if remove is not None else None)
    total_pixels = int(shape[0] * shape[1]) if shape is not None else 0
    keep_pixels = int(keep.sum()) if keep is not None else 0
    remove_pixels = int(remove.sum()) if remove is not None else 0
    conflict_pixels = int((keep & remove).sum()) if keep is not None and remove is not None and keep.shape == remove.shape else 0
    return {
        "keep_mask_provided": keep_mask is not None,
        "remove_mask_provided": remove_mask is not None,
        "keep_pixels": keep_pixels,
        "remove_pixels": remove_pixels,
        "conflict_pixels": conflict_pixels,
        "total_pixels": total_pixels,
        "keep_coverage": (keep_pixels / total_pixels) if total_pixels else 0.0,
        "remove_coverage": (remove_pixels / total_pixels) if total_pixels else 0.0,
        "keep_empty": keep_mask is not None and keep_pixels == 0,
        "remove_empty": remove_mask is not None and remove_pixels == 0,
        "keep_full": total_pixels > 0 and keep_pixels == total_pixels,
        "remove_full": total_pixels > 0 and remove_pixels == total_pixels,
        "high_risk_full_mask": total_pixels > 0 and (keep_pixels == total_pixels or remove_pixels == total_pixels),
        "conflict_policy": "remove_overrides_keep",
    }


def _load_upload_image_with_digest(upload: UploadFile) -> tuple[Image.Image, str]:
    data = upload.file.read()
    return _image_from_upload_bytes(data), hashlib.sha256(data).hexdigest()


def _source_alpha_mask(image: Image.Image) -> np.ndarray | None:
    if image.mode != "RGBA":
        return None
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    if not np.any(alpha < 255):
        return None
    return alpha


def _source_alpha_skip_info(*, requested: bool, stage: str) -> dict[str, object]:
    return {
        "requested": bool(requested),
        "applied": False,
        "skipped": True,
        "reason": "source_alpha_transparent",
        "stage": stage,
    }


def _preprocess_background_repair_image(image: Image.Image, enabled: bool) -> tuple[Image.Image, dict[str, object]]:
    if _source_alpha_mask(image) is not None:
        return image, _source_alpha_skip_info(requested=enabled, stage="background_repair")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    result = apply_input_preprocess(
        rgb,
        selected=[BACKGROUND_REPAIR] if enabled else [],
    )
    info = checkerboard_info_from_decision(result.decision)
    info["analysis"] = result.analysis.to_dict()
    info["decision"] = result.decision.to_dict()
    info["background_repair"] = dict(result.decision.metadata.get("background_repair") or {})
    if not info.get("applied", False):
        return image, info
    return Image.fromarray(result.image_srgb, mode="RGB"), info


def _image_digest(image: Image.Image) -> str:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def _attachment_content_disposition(filename: str) -> str:
    fallback = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not fallback:
        suffix = Path(filename).suffix if Path(filename).suffix.isascii() else ""
        fallback = f"download{suffix or '.bin'}"
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runtime-capabilities")
def runtime_capabilities(
    include_comfy: bool = Query(
        WEB_ENABLE_COMFY,
        description="When true, query the optional ComfyUI adapter runtime.",
    ),
    include_object_info: bool = Query(
        True,
        description="When true, query Comfy /object_info to verify ERMBG custom node availability.",
    ),
    timeout: float = Query(3.0, ge=0.1, le=30.0),
) -> dict[str, Any]:
    payload = collect_runtime_capabilities(
        timeout=timeout,
        include_object_info=include_object_info,
        include_comfy=include_comfy,
        direct_worker_url=WEB_DIRECT_WORKER_URL,
    )
    payload["web"] = {
        "auto_backend": WEB_AUTO_BACKEND,
        "auto_fallback_backend": WEB_AUTO_FALLBACK_BACKEND,
        "direct_worker_url": WEB_DIRECT_WORKER_URL,
        "direct_worker_endpoints": dict(WEB_DIRECT_WORKER_ENDPOINTS),
        "enable_comfy": WEB_ENABLE_COMFY,
    }
    return payload


@app.post("/api/checkerboard-background")
def checkerboard_background_endpoint(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    image = _load_upload_image(file)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    analysis = analyze_input_preprocess(rgb)
    checkerboard = analysis.debug.get("checkerboard", {})
    return {
        "accepted": bool(checkerboard.get("accepted", False)),
        "recommended": bool(checkerboard.get("accepted", False)),
        "analysis": checkerboard,
        "preprocess": analysis.to_dict(),
    }


@app.post("/api/preprocess-analysis")
def preprocess_analysis_endpoint(
    file: Annotated[UploadFile, File()],
    background_repair: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    image = _load_upload_image(file)
    source_alpha = _source_alpha_mask(image)
    if source_alpha is not None:
        skip_info = _source_alpha_skip_info(requested=background_repair, stage="preprocess_analysis")
        return _json_safe_debug(
            {
                "schema": "ermbg.preprocess_analysis.v1",
                "image_digest": _image_digest(image),
                "width": int(image.width),
                "height": int(image.height),
                "selected": [],
                "applied": [],
                "preprocess": skip_info,
                "analysis": {"items": [], "skipped": True, "reason": "source_alpha_transparent"},
                "checkerboard": skip_info,
                "next_endpoint": "/api/analyze-candidates",
            }
        )
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    result = apply_input_preprocess(
        rgb,
        selected=[BACKGROUND_REPAIR] if background_repair else [],
    )
    checkerboard = checkerboard_info_from_decision(result.decision)
    return _json_safe_debug(
        {
            "schema": "ermbg.preprocess_analysis.v1",
            "image_digest": _image_digest(image),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "selected": result.decision.selected,
            "applied": result.decision.applied,
            "preprocess": result.decision.to_dict(),
            "analysis": result.analysis.to_dict(),
            "checkerboard": checkerboard,
            "next_endpoint": "/api/analyze-candidates",
        }
    )


@app.post("/api/analyze-candidates")
def analyze_candidates_endpoint(
    file: Annotated[UploadFile, File()],
    background_repair: Annotated[bool, Form()] = False,
    corridorkey_screen_mode: Annotated[str, Form()] = "auto",
    corridorkey_preset: Annotated[str, Form()] = "auto",
    fallback_bg_color: Annotated[str, Form()] = "0,200,0",
) -> dict[str, object]:
    if corridorkey_screen_mode not in {"auto", "green", "blue"}:
        raise HTTPException(status_code=400, detail="corridorkey_screen_mode must be auto, green, or blue")
    if corridorkey_preset not in {"auto", "detail_safe", "spill_safe", "manual"}:
        raise HTTPException(status_code=400, detail="corridorkey_preset must be auto, detail_safe, spill_safe, or manual")
    fallback_bg = _parse_rgb_triplet(fallback_bg_color)
    image = _load_upload_image(file)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    source_alpha = _source_alpha_mask(image)
    preprocess = apply_input_preprocess(
        rgb,
        selected=[] if source_alpha is not None else ([BACKGROUND_REPAIR] if background_repair else []),
    )
    analyze_image = Image.fromarray(preprocess.image_srgb, mode="RGB")
    effective_preprocess = preprocess.decision
    known_b_preprocess_info: dict[str, Any] = {"requested": bool(background_repair), "applied": False}
    if source_alpha is not None:
        known_b_preprocess_info = _source_alpha_skip_info(requested=background_repair, stage="known_b_background_normalization")
    elif background_repair:
        pymatting_params = {
            "pymatting_bg_source": "auto",
            "pymatting_bg_color": fallback_bg,
            "pymatting_bg_threshold": 3.5,
            "pymatting_fg_threshold": 24.0,
        }
        route = _known_b_route_for_web_matte(
            analyze_image,
            backend="auto-local",
            pymatting_params=pymatting_params,
        )
        if route is not None:
            route_params = route.get("params") if isinstance(route.get("params"), dict) else {}
            analyze_image, known_b_preprocess_info, _execution_params, known_b_decision = _apply_known_b_background_repair(
                analyze_image,
                route_params=route_params,
                pymatting_params=pymatting_params,
            )
            if known_b_decision is not None:
                effective_preprocess = _merge_web_preprocess_decisions(effective_preprocess, known_b_decision)
        else:
            known_b_preprocess_info = {"requested": True, "applied": False, "reason": "route_is_not_known_b"}
    analyze_rgb = np.asarray(analyze_image.convert("RGB"), dtype=np.uint8)
    result = analyze_candidates(
        analyze_rgb,
        preprocess=effective_preprocess,
        screen_mode=corridorkey_screen_mode,
        preset=corridorkey_preset,
        fallback_background_color=fallback_bg,
    )
    payload = result.to_dict()
    payload["preprocess_analysis"] = preprocess.analysis.to_dict()
    payload["preprocess_analysis"]["known_background_normalization"] = known_b_preprocess_info
    payload["ambiguities"] = payload.get("ambiguity_regions", [])
    return _json_safe_debug(payload)


def _json_form_object(value: str | None, field_name: str) -> dict[str, Any]:
    if value is None or not str(value).strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object") from e
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object")
    return payload


def _semantic_candidate_payload(
    analysis_payload: dict[str, Any],
    selected_candidate_id: str,
) -> tuple[dict[str, Any], float | None]:
    candidate = _semantic_candidate_record(analysis_payload, selected_candidate_id)
    if candidate is not None:
        confidence = candidate.get("confidence")
        return (
            dict(candidate.get("decision") or {}),
            float(confidence) if isinstance(confidence, (int, float)) else None,
        )
    return ({"policy": "auto_default"}, None)


def _semantic_candidate_record(
    analysis_payload: dict[str, Any],
    selected_candidate_id: str,
) -> dict[str, Any] | None:
    for candidate in analysis_payload.get("candidates", []):
        if isinstance(candidate, dict) and candidate.get("id") == selected_candidate_id:
            return candidate
    return None


def _selected_route_payload_from_analysis(
    analysis_payload: dict[str, Any],
    selected_candidate_id: str,
) -> dict[str, Any]:
    route_candidates = analysis_payload.get("route_candidates")
    candidate = _semantic_candidate_record(analysis_payload, selected_candidate_id)
    route_candidate_id = None
    if isinstance(candidate, dict):
        route_candidate_id = candidate.get("route_candidate_id")
        decision = candidate.get("decision")
        if route_candidate_id is None and isinstance(decision, dict):
            route_candidate_id = decision.get("route_candidate_id")
    if route_candidate_id is None:
        route_candidate_id = analysis_payload.get("default_route_candidate_id")
    if isinstance(route_candidates, list) and route_candidate_id is not None:
        for route_candidate in route_candidates:
            if isinstance(route_candidate, dict) and route_candidate.get("id") == route_candidate_id:
                return dict(route_candidate)
    return _route_payload_from_contract(analysis_payload)


def _selected_preprocess_ids_from_analysis(analysis_payload: dict[str, Any]) -> list[str]:
    preprocess = analysis_payload.get("preprocess")
    if not isinstance(preprocess, dict):
        return []
    selected = preprocess.get("selected")
    return [str(item) for item in selected] if isinstance(selected, list) else []


def _selected_preprocess_ids_from_contract(contract_payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(contract_payload, dict):
        return []
    preprocess = contract_payload.get("preprocess")
    if not isinstance(preprocess, dict):
        return []
    selected = preprocess.get("selected")
    return [str(item) for item in selected] if isinstance(selected, list) else []


def _execute_background_repair_from_contract(analysis_payload: dict[str, Any], fallback: bool) -> bool:
    selected = _selected_preprocess_ids_from_analysis(analysis_payload)
    if selected:
        return BACKGROUND_REPAIR in selected
    if isinstance(analysis_payload.get("preprocess"), dict):
        return False
    return fallback


def _route_payload_from_contract(contract_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(contract_payload, dict):
        return {}
    route = contract_payload.get("route")
    return dict(route) if isinstance(route, dict) else {}


def _analysis_route_params(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    route = _route_payload_from_contract(analysis_payload)
    params = route.get("params")
    return dict(params) if isinstance(params, dict) else {}


def _analysis_algorithm(analysis_payload: dict[str, Any]) -> str:
    route = _route_payload_from_contract(analysis_payload)
    return str(route.get("algorithm") or route.get("route") or "")


def _analysis_explicit_algorithm(analysis_payload: dict[str, Any]) -> str:
    route = _route_payload_from_contract(analysis_payload)
    return str(route.get("algorithm") or "")


def _analysis_route_decision_payload(analysis_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(analysis_payload, dict):
        return None
    route = _route_payload_from_contract(analysis_payload)
    if not route:
        return None
    params = route.get("params")
    analysis = route.get("analysis")
    payload = {
        "route": route.get("route") or route.get("algorithm"),
        "algorithm": route.get("algorithm") or route.get("backend") or route.get("route"),
        "backend": route.get("backend") or route.get("algorithm") or route.get("route"),
        "asset_kind": route.get("asset_kind"),
        "parameter_profile": route.get("parameter_profile"),
        "execution_profile": route.get("execution_profile"),
        "confidence": route.get("confidence"),
        "reasons": route.get("reasons") if isinstance(route.get("reasons"), list) else [],
        "params": params if isinstance(params, dict) else {},
        "analysis": analysis if isinstance(analysis, dict) else {},
    }
    corridorkey_analysis = route.get("corridorkey_analysis")
    if isinstance(corridorkey_analysis, dict):
        payload["corridorkey_analysis"] = corridorkey_analysis
    return payload


def _rgb_color_from_payload(value: Any) -> tuple[int, int, int] | None:
    if not (isinstance(value, (list, tuple)) and len(value) == 3):
        return None
    return tuple(int(np.clip(c, 0, 255)) for c in value)  # type: ignore[return-value]


def _merge_web_preprocess_decisions(
    current: PreprocessDecision | None,
    addition: PreprocessDecision,
) -> PreprocessDecision:
    selected = list(current.selected) if current is not None else []
    applied = list(current.applied) if current is not None else []
    metadata = dict(current.metadata) if current is not None else {}
    for item in addition.selected:
        if item not in selected:
            selected.append(item)
    for item in addition.applied:
        if item not in applied:
            applied.append(item)
    metadata.update(addition.metadata)
    return PreprocessDecision(
        selected=selected,
        applied=applied,
        metadata=metadata,
        background_model=addition.background_model or (current.background_model if current is not None else None),
    )


def _known_b_background_from_params(
    image: Image.Image,
    route_params: dict[str, Any],
    pymatting_params: dict[str, object],
) -> tuple[tuple[int, int, int] | None, dict[str, Any]]:
    background = _rgb_color_from_payload(route_params.get("pymatting_bg_color"))
    if background is None:
        background = _rgb_color_from_payload(pymatting_params.get("pymatting_bg_color"))
    bg_source = str(route_params.get("pymatting_bg_source", pymatting_params.get("pymatting_bg_source", "custom"))).strip().lower()
    if background is not None and bg_source in {"", "custom"}:
        return background, {"source": "known_b_params", "background_color": list(background)}
    if bg_source == "green":
        return (0, 200, 0), {"source": "preset_green", "background_color": [0, 200, 0]}
    if bg_source == "blue":
        return (0, 0, 200), {"source": "preset_blue", "background_color": [0, 0, 200]}
    if background is not None and bg_source != "auto":
        return background, {"source": "known_b_params", "background_color": list(background)}

    from .pymatting_refine import estimate_stable_background_color

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    estimated, info = estimate_stable_background_color(rgb)
    selected = tuple(int(c) for c in estimated)
    return selected, {"source": "preprocess_estimate_stable_background_color", **info}


def _apply_known_b_background_repair(
    image: Image.Image,
    *,
    route_params: dict[str, Any],
    pymatting_params: dict[str, object],
) -> tuple[Image.Image, dict[str, Any], dict[str, Any], PreprocessDecision | None]:
    if _source_alpha_mask(image) is not None:
        return image, _source_alpha_skip_info(requested=True, stage="known_b_background_normalization"), {}, None
    background, background_info = _known_b_background_from_params(image, route_params, pymatting_params)
    if background is None:
        return image, {"skipped": True, "reason": "missing_known_background_color"}, {}, None

    bg_threshold = float(route_params.get("pymatting_bg_threshold", pymatting_params.get("pymatting_bg_threshold", 3.5)))
    fg_threshold = float(route_params.get("pymatting_fg_threshold", pymatting_params.get("pymatting_fg_threshold", 24.0)))
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    normalized, decision = repair_known_background_preprocess(
        rgb,
        background,
        bg_threshold=bg_threshold,
        fg_threshold=fg_threshold,
        adaptive=False,
    )
    normalization = dict(decision.metadata.get("known_background_normalization") or {})
    normalization["background_model"] = background_info
    execution_params = {
        "pymatting_bg_source": "custom",
        "pymatting_bg_color": background,
        "pymatting_input_preprocessed": True,
        "pymatting_bg_threshold": bg_threshold,
        "pymatting_fg_threshold": fg_threshold,
        "pymatting_adapt_bg_threshold": bool(
            route_params.get("pymatting_adapt_bg_threshold", pymatting_params.get("pymatting_adapt_bg_threshold", False))
        ),
        "pymatting_adapt_fg_threshold": bool(
            route_params.get("pymatting_adapt_fg_threshold", pymatting_params.get("pymatting_adapt_fg_threshold", True))
        ),
        "pymatting_adapt_boundary_band": bool(
            route_params.get(
                "pymatting_adapt_boundary_band",
                pymatting_params.get("pymatting_adapt_boundary_band", True),
            )
        ),
    }
    for key in (
        "pymatting_method",
        "pymatting_image_space",
        "pymatting_boundary_band_px",
        "pymatting_cg_maxiter",
        "pymatting_cg_rtol",
        "pymatting_trimap_mode",
        "pymatting_unknown_grow_px",
        "pymatting_input_preprocessed",
    ):
        if key in route_params:
            execution_params[key] = route_params[key]
        elif key in pymatting_params:
            execution_params[key] = pymatting_params[key]
    info = {
        "selected": True,
        "requested": True,
        "applied": BACKGROUND_REPAIR in decision.applied,
        "background_color": [int(c) for c in background],
        "background_model": background_info,
        "known_background_normalization": normalization,
        "decision": decision.to_dict(),
    }
    return Image.fromarray(normalized, mode="RGB"), info, execution_params, decision


def _known_b_preprocess_from_contract(
    image: Image.Image,
    analysis_payload: dict[str, Any] | None,
    pymatting_params: dict[str, object],
) -> tuple[Image.Image, dict[str, Any], dict[str, Any]]:
    """Apply Analyze-selected Known-B normalization before Execute.

    The selected item is contract-driven. The executor receives the normalized
    image plus metadata that tells it to skip its legacy private normalization.
    """

    analysis = analysis_payload or {}
    selected = _selected_preprocess_ids_from_contract(analysis)
    if BACKGROUND_REPAIR not in selected or _analysis_algorithm(analysis) != "pymatting_known_b":
        return image, {}, {}

    route_params = _analysis_route_params(analysis)
    background = _rgb_color_from_payload(route_params.get("pymatting_bg_color"))
    if background is None:
        background = _rgb_color_from_payload(pymatting_params.get("pymatting_bg_color"))
    if background is None:
        return image, {"skipped": True, "reason": "missing_known_background_color"}, {}
    route_params = {**route_params, "pymatting_bg_source": "custom", "pymatting_bg_color": background}
    normalized, info, execution_params, _decision = _apply_known_b_background_repair(
        image,
        route_params=route_params,
        pymatting_params=pymatting_params,
    )
    return normalized, info, execution_params


def _execute_backend_from_analysis(requested_backend: str, analysis_payload: dict[str, Any] | None) -> str:
    if requested_backend != "auto":
        return requested_backend
    algorithm = _analysis_explicit_algorithm(analysis_payload or {})
    if algorithm in {"pymatting_known_b", "corridorkey", "known_bg_glow"}:
        return algorithm
    if algorithm == "rgba_passthrough":
        return "passthrough"
    return requested_backend


def _route_decision_algorithm(route_decision: dict[str, Any] | None) -> str:
    if not isinstance(route_decision, dict):
        return ""
    return str(
        route_decision.get("algorithm")
        or route_decision.get("backend")
        or route_decision.get("route")
        or ""
    )


def _merge_known_b_execution_params_into_route_decision(
    route_decision: dict[str, Any] | None,
    pymatting_params: dict[str, object],
) -> dict[str, Any] | None:
    if route_decision is None or _route_decision_algorithm(route_decision) != "pymatting_known_b":
        return route_decision
    params = route_decision.get("params")
    return {
        **route_decision,
        "params": {
            **(params if isinstance(params, dict) else {}),
            **pymatting_params,
        },
    }


def _known_b_route_for_web_matte(
    image: Image.Image,
    *,
    backend: str,
    pymatting_params: dict[str, object],
) -> dict[str, Any] | None:
    base = _direct_backend_base(backend)
    if base in {"pymatting_known_b", "pymatting-known-b", "direct-pymatting-known-b"}:
        return {"algorithm": "pymatting_known_b", "params": dict(pymatting_params)}
    if base not in {"auto", "auto-local", "direct-worker"}:
        return None

    fallback = _rgb_color_from_payload(pymatting_params.get("pymatting_bg_color")) or (0, 200, 0)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    route = classify_image_route(rgb, bg_color=fallback).to_dict()
    algorithm = str(route.get("algorithm") or route.get("route") or "")
    if algorithm != "pymatting_known_b":
        return None
    return route


def _apply_web_matte_known_b_background_repair(
    image: Image.Image,
    *,
    backend: str,
    background_repair: bool,
    pymatting_params: dict[str, object],
) -> tuple[Image.Image, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    if not background_repair:
        return image, {"requested": False, "applied": False}, {}, None
    route = _known_b_route_for_web_matte(image, backend=backend, pymatting_params=pymatting_params)
    if route is None:
        return image, {"requested": True, "applied": False, "reason": "route_is_not_known_b"}, {}, None
    route_params = route.get("params") if isinstance(route.get("params"), dict) else {}
    normalized, info, execution_params, _decision = _apply_known_b_background_repair(
        image,
        route_params=route_params,
        pymatting_params=pymatting_params,
    )
    if execution_params:
        route = {
            **route,
            "params": {
                **route_params,
                **execution_params,
            },
        }
    return normalized, info, execution_params, route


def _semantic_execution_summary(
    *,
    analysis_payload: dict[str, Any],
    selected_candidate_id: str,
    semantic_decision_payload: dict[str, Any],
    user_mask: UserMaskDecision | None = None,
) -> dict[str, Any]:
    analyze = AnalyzeResult.from_dict(analysis_payload) if analysis_payload else None
    selected = selected_candidate_id.strip() or (
        analyze.default_candidate_id if analyze is not None else "auto_default"
    ) or "auto_default"
    candidate_decision, candidate_confidence = _semantic_candidate_payload(analysis_payload, selected)
    decision_payload = semantic_decision_payload or candidate_decision
    selected_route = _selected_route_payload_from_analysis(analysis_payload, selected)
    source = "web_user" if analyze is not None and analyze.status == "needs_decision" else "auto_default"
    semantic_decision = SemanticDecision(
        candidate_id=selected,
        decision=decision_payload,
        source=source,
        confidence=candidate_confidence,
    )
    preprocess = analyze.preprocess if analyze is not None and analyze.preprocess is not None else PreprocessDecision()
    execution_request = ExecutionRequest(
        analysis_id=analyze.analysis_id if analyze is not None else None,
        preprocess=preprocess,
        route=selected_route,
        selected_candidate_id=selected,
        semantic_decision=semantic_decision,
        user_mask=user_mask,
        metadata={
            "schema": "ermbg.execution_request.summary.v1",
            "source": "web_execute_candidate",
            "selected_route_candidate_id": selected_route.get("id") or selected_route.get("route_candidate_id"),
        },
    )
    summary = semantic_manifest_summary(
        analyze=analyze,
        preprocess=preprocess,
        selected_candidate_id=selected,
        semantic_decision=semantic_decision,
        user_mask=user_mask,
    )
    summary["execution_request"] = execution_request.to_dict()
    return summary


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _attach_semantic_execution_metadata(
    payload: dict[str, object],
    *,
    analysis_payload: dict[str, Any],
    selected_candidate_id: str,
    semantic_decision_payload: dict[str, Any],
    user_mask: UserMaskDecision | None = None,
) -> dict[str, object]:
    summary = _semantic_execution_summary(
        analysis_payload=analysis_payload,
        selected_candidate_id=selected_candidate_id,
        semantic_decision_payload=semantic_decision_payload,
        user_mask=user_mask,
    )
    semantic = summary["semantic"]
    payload["preprocess"] = summary["preprocess"]
    payload["semantic"] = semantic
    payload["analysis_status"] = semantic.get("analysis_status")
    payload["default_candidate_id"] = semantic.get("default_candidate_id")
    payload["selected_candidate_id"] = semantic.get("selected_candidate_id")
    payload["semantic_decision"] = semantic.get("semantic_decision")
    payload["execution_request"] = summary["execution_request"]
    debug = payload.get("debug")
    if not isinstance(debug, dict):
        debug = {}
        payload["debug"] = debug
    debug["semantic_execution"] = {
        "mode": "stage6_front_loaded_decision_with_user_mask_metadata",
        "selected_candidate_id": semantic.get("selected_candidate_id"),
        "analysis_status": semantic.get("analysis_status"),
        "user_mask_used": semantic.get("user_mask_used"),
        "execution_request_schema": summary["execution_request"].get("metadata", {}).get("schema"),
    }

    artifact_value = payload.get("artifact_manifest")
    if isinstance(artifact_value, str) and artifact_value:
        manifest_path = _resolve_project_path(artifact_value)
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            extra = manifest.setdefault("extra", {})
            if isinstance(extra, dict):
                extra["pipeline"] = summary
            request_payload = manifest.setdefault("request", {})
            if isinstance(request_payload, dict):
                request_payload["selected_candidate_id"] = semantic.get("selected_candidate_id")
            manifest_path.write_text(json.dumps(_json_safe_debug(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
            report_value = manifest.get("report")
            if isinstance(report_value, str) and report_value:
                report_path = manifest_path.parent / report_value
                if report_path.exists():
                    report = _load_json(report_path)
                    report["preprocess"] = summary["preprocess"]
                    report["semantic"] = semantic
                    report["execution_request"] = summary["execution_request"]
                    report_path.write_text(json.dumps(_json_safe_debug(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


@app.get("/api/artifacts")
def artifacts_index(
    limit: int = Query(200, ge=1, le=1000),
    type: str | None = Query(None),
    route: str | None = Query(None),
    execution_backend: str | None = Query(None),
    analysis_status: str | None = Query(None),
    selected_candidate_id: str | None = Query(None),
    user_mask_used: bool | None = Query(None),
) -> dict[str, Any]:
    items = _list_artifacts(limit=1000)
    filters = {
        "type": type,
        "route": route,
        "execution_backend": execution_backend,
        "analysis_status": analysis_status,
        "selected_candidate_id": selected_candidate_id,
    }
    for key, value in filters.items():
        if value:
            items = [item for item in items if str(item.get(key) or "") == value]
    if user_mask_used is not None:
        items = [item for item in items if bool(item.get("user_mask_used")) is bool(user_mask_used)]
    items = items[:limit]
    return {
        "schema": "ermbg.artifacts.index.v1",
        "count": len(items),
        "items": items,
    }


@app.get("/api/artifacts/{artifact_id:path}")
def artifact_detail(artifact_id: str) -> dict[str, Any]:
    path = _artifact_path_from_id(artifact_id)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    manifest = _load_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema") != "ermbg.run.v1":
        raise HTTPException(status_code=404, detail="Artifact manifest is not an ERMBG run manifest.")
    summary = _artifact_summary(path)
    return {
        "summary": summary,
        "manifest": manifest,
    }


@app.get("/artifacts", response_class=HTMLResponse)
def artifacts_page() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG Artifacts</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #18211d; background: #f5f7f4; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; }
    header { min-height: 56px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 24px; border-bottom: 1px solid #d9dfd7; background: #ffffff; }
    h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    nav { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    nav a { color: #196f5a; font-size: 13px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    main { width: min(1440px, 100%); margin: 0 auto; padding: 18px 24px 28px; }
    .toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; color: #5a665f; font-size: 13px; flex-wrap: wrap; }
    .filters { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    select, input { min-height: 34px; border: 1px solid #cfd8cc; border-radius: 6px; padding: 0 8px; background: #ffffff; color: #18211d; font: inherit; }
    input { width: 188px; }
    .status { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    button { min-height: 34px; padding: 0 12px; border: 0; border-radius: 6px; background: #196f5a; color: #ffffff; font: inherit; font-weight: 800; cursor: pointer; }
    .table-wrap { overflow: auto; border: 1px solid #d8e0d5; border-radius: 8px; background: #ffffff; }
    table { width: 100%; min-width: 1320px; border-collapse: separate; border-spacing: 0; table-layout: fixed; }
    th, td { border-bottom: 1px solid #e3e9e0; padding: 10px; vertical-align: top; text-align: left; }
    th { position: sticky; top: 0; z-index: 1; background: #fbfcfa; color: #53615a; font-size: 12px; white-space: nowrap; }
    tr:last-child td { border-bottom: 0; }
    .type-col { width: 132px; }
    .pipeline-col { width: 190px; }
    .candidate-col { width: 210px; }
    .backend-col { width: 210px; }
    .route-col { width: 190px; }
    .outputs-col { width: 190px; }
    .manifest-col { width: 280px; }
    .pill { display: inline-flex; align-items: center; min-height: 24px; padding: 0 8px; border-radius: 999px; background: #edf4ef; color: #245f53; font-size: 12px; font-weight: 900; white-space: nowrap; }
    .muted { color: #68746e; font-size: 12px; line-height: 1.35; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }
    .links { display: flex; flex-wrap: wrap; gap: 6px; }
    .links a, .manifest-link { color: #196f5a; font-size: 12px; font-weight: 800; text-decoration: none; }
    .empty { min-height: 220px; display: grid; place-items: center; color: #68746e; background: #ffffff; border: 1px dashed #cfd8cc; border-radius: 8px; }
    @media (max-width: 760px) { header { align-items: flex-start; flex-direction: column; padding: 12px 16px; } main { padding: 14px 12px 24px; } .toolbar { align-items: flex-start; flex-direction: column; } }
  </style>
</head>
<body>
  <header>
    <h1>ERMBG Artifacts</h1>
    <nav>
      <a href="/">上传页</a>
      <a href="/artifacts">Artifacts</a>
      <a href="/eval/game">Game Eval</a>
    </nav>
  </header>
  <main>
    <div class="toolbar">
      <span class="status" id="status">读取 artifacts</span>
      <div class="filters" aria-label="artifact filters">
        <select id="analysis-filter" aria-label="analysis status">
          <option value="">全部状态</option>
          <option value="ready">ready</option>
          <option value="needs_decision">needs_decision</option>
          <option value="unsupported">unsupported</option>
        </select>
        <select id="mask-filter" aria-label="mask usage">
          <option value="">全部 mask</option>
          <option value="true">mask used</option>
          <option value="false">no mask</option>
        </select>
        <input id="route-filter" type="search" placeholder="route">
        <button type="button" id="refresh">刷新</button>
      </div>
    </div>
    <div class="table-wrap" id="table-wrap" hidden>
      <table aria-label="artifact list">
        <thead>
          <tr>
            <th class="type-col">类型</th>
            <th class="pipeline-col">Pipeline</th>
            <th class="candidate-col">Candidate / Mask</th>
            <th class="backend-col">后端</th>
            <th class="route-col">Route</th>
            <th class="outputs-col">输出</th>
            <th class="manifest-col">Manifest</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div class="empty" id="empty" hidden>没有发现 ermbg.run.v1 artifact</div>
  </main>
  <script>
    const rows = document.getElementById("rows");
    const statusEl = document.getElementById("status");
    const tableWrap = document.getElementById("table-wrap");
    const empty = document.getElementById("empty");
    const refreshButton = document.getElementById("refresh");
    const analysisFilter = document.getElementById("analysis-filter");
    const maskFilter = document.getElementById("mask-filter");
    const routeFilter = document.getElementById("route-filter");

    function text(value) {
      return value === null || value === undefined || value === "" ? "—" : String(value);
    }

    function esc(value) {
      return text(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
    }

    function shortPath(value) {
      const raw = text(value);
      return raw.length > 84 ? "…" + raw.slice(-83) : raw;
    }

    function listText(value) {
      return Array.isArray(value) && value.length ? value.join(", ") : "—";
    }

    function maskSummary(item) {
      if (!item.user_mask_used) return "mask: no";
      const summary = item.user_mask_summary && typeof item.user_mask_summary === "object" ? item.user_mask_summary : {};
      const keep = Number(summary.keep_pixels || 0);
      const remove = Number(summary.remove_pixels || 0);
      const risk = summary.high_risk_full_mask ? " · full-risk" : "";
      return `mask: keep ${keep} / remove ${remove}${risk}`;
    }

    function renderOutputs(item) {
      const urls = item.urls && typeof item.urls === "object" ? item.urls : {};
      const entries = Object.entries(urls).filter(([, url]) => typeof url === "string" && url);
      if (!entries.length) return '<span class="muted">—</span>';
      return '<span class="links">' + entries.map(([key, url]) => `<a href="${url}" target="_blank" rel="noreferrer">${key}</a>`).join("") + '</span>';
    }

    function render(items) {
      rows.innerHTML = "";
      tableWrap.hidden = items.length === 0;
      empty.hidden = items.length !== 0;
      items.forEach((item) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><span class="pill">${esc(item.type)}</span><div class="muted">${esc(new Date((Number(item.mtime) || 0) * 1000).toLocaleString())}</div></td>
          <td><div>${esc(item.analysis_status)}</div><div class="muted">pre: ${esc(listText(item.preprocess && item.preprocess.selected))}</div></td>
          <td><div>${esc(item.selected_candidate_id)}</div><div class="muted">${esc(maskSummary(item))}</div></td>
          <td><div>${esc(item.execution_backend || item.backend)}</div><div class="muted">${esc(item.execution_server_url || item.requested_backend)}</div></td>
          <td><div>${esc(item.route)}</div><div class="muted">${esc(item.execution_profile)}</div></td>
          <td>${renderOutputs(item)}</td>
          <td><a class="manifest-link mono" href="/api/artifacts/${esc(item.id)}" target="_blank" rel="noreferrer">${esc(shortPath(item.manifest))}</a></td>
        `;
        rows.appendChild(tr);
      });
    }

    async function loadArtifacts() {
      statusEl.textContent = "读取 artifacts";
      refreshButton.disabled = true;
      try {
        const params = new URLSearchParams({ limit: "200" });
        if (analysisFilter.value) params.set("analysis_status", analysisFilter.value);
        if (maskFilter.value) params.set("user_mask_used", maskFilter.value);
        if (routeFilter.value.trim()) params.set("route", routeFilter.value.trim());
        const response = await fetch(`/api/artifacts?${params.toString()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        const items = Array.isArray(payload.items) ? payload.items : [];
        render(items);
        statusEl.textContent = `${items.length} artifacts`;
      } catch (error) {
        rows.innerHTML = "";
        tableWrap.hidden = true;
        empty.hidden = false;
        empty.textContent = "读取 artifacts 失败";
        statusEl.textContent = error.message;
      } finally {
        refreshButton.disabled = false;
      }
    }

    refreshButton.addEventListener("click", loadArtifacts);
    [analysisFilter, maskFilter].forEach((control) => control.addEventListener("change", loadArtifacts));
    routeFilter.addEventListener("keydown", (event) => { if (event.key === "Enter") loadArtifacts(); });
    loadArtifacts();
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _matte_page_html()


def _matte_page_html() -> str:
    return _inject_backend_options("""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1c2320; background: #f5f7f4; }
    * { box-sizing: border-box; }
    body { margin: 0; height: 100vh; display: grid; grid-template-rows: auto 1fr; overflow: hidden; }
    .app-header { min-height: 56px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 24px; border-bottom: 1px solid #d9dfd7; background: #ffffff; }
    .primary-tabs { min-width: 0; display: flex; align-items: center; justify-content: flex-start; gap: 4px; overflow-x: auto; }
    .nav-tab { display: inline-flex; align-items: center; justify-content: center; min-height: 34px; padding: 0 12px; border-radius: 6px; color: #47524c; font-size: 13px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    .nav-tab.is-active { background: #196f5a; color: #ffffff; }
    .header-right { margin-left: auto; display: flex; align-items: center; justify-content: flex-end; gap: 12px; min-width: 0; }
    .eval-link { color: #196f5a; font-size: 13px; font-weight: 900; text-decoration: none; white-space: nowrap; }
    .runtime-status { min-width: 0; display: inline-flex; align-items: center; gap: 6px; overflow: hidden; }
    .runtime-pill { display: inline-flex; align-items: center; gap: 5px; min-height: 24px; padding: 0 8px; border: 1px solid #d2dad0; border-radius: 999px; background: #f7faf6; color: #5d6862; font-size: 12px; font-weight: 800; white-space: nowrap; }
    .runtime-pill::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #9aa59e; }
    .runtime-pill.is-ok::before { background: #23855f; }
    .runtime-pill.is-error::before { background: #b94a42; }
    .runtime-pill.is-warn::before { background: #b57b18; }
    .runtime-endpoint-pill { position: relative; padding: 0 6px 0 8px; gap: 0; }
    .runtime-endpoint-pill::before { display: none; }
    .runtime-dot { flex: 0 0 7px; width: 7px; height: 7px; border-radius: 50%; background: #9aa59e; }
    .runtime-endpoint-pill.is-ok .runtime-dot { background: #23855f; }
    .runtime-endpoint-pill.is-error .runtime-dot { background: #b94a42; }
    .runtime-endpoint-pill.is-warn .runtime-dot { background: #b57b18; }
    .runtime-endpoint-select { width: auto; min-width: 120px; max-width: 190px; min-height: 22px; padding: 0 22px 0 6px; border: 0; background: transparent; color: inherit; font: inherit; font-size: 12px; font-weight: 900; cursor: pointer; }
    .runtime-endpoint-select:disabled { cursor: wait; opacity: 0.72; }
    main { width: min(1120px, 100%); height: calc(100vh - 56px); min-height: 0; margin: 0 auto; padding: 16px 24px; display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 24px; align-items: stretch; overflow: hidden; }
    form, .preview { background: #ffffff; border: 1px solid #d9dfd7; border-radius: 8px; }
    form { min-width: 0; min-height: 0; max-height: 100%; padding: 16px; display: grid; gap: 12px; align-content: start; overflow-y: auto; }
    label { display: grid; gap: 8px; font-size: 13px; font-weight: 600; color: #47524c; }
    .inline-label { display: grid; grid-template-columns: 76px minmax(0, 1fr); align-items: center; gap: 10px; }
    input, select, button { width: 100%; min-height: 40px; border-radius: 6px; border: 1px solid #b8c1b7; background: #ffffff; color: #1c2320; font: inherit; }
    input[type="file"] { padding: 8px; }
    button, a.download { display: inline-flex; align-items: center; justify-content: center; min-height: 42px; border: 0; border-radius: 6px; background: #196f5a; color: #ffffff; text-decoration: none; font-weight: 700; cursor: pointer; }
    button:disabled, a.download[aria-disabled="true"] { opacity: 0.55; cursor: not-allowed; pointer-events: none; }
    .settings { display: none; border: 1px solid #d9dfd7; border-radius: 6px; background: #fbfcfa; }
    .settings.is-visible { display: block; }
    .settings summary { min-height: 38px; display: flex; align-items: center; padding: 0 10px; color: #196f5a; font-size: 13px; font-weight: 800; cursor: pointer; user-select: none; }
    .settings-grid { display: grid; gap: 12px; padding: 0 10px 10px; }
    .settings-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .settings label { font-size: 12px; gap: 6px; }
    .check-label { display: flex; align-items: center; justify-content: space-between; gap: 10px; min-height: 38px; }
    .check-label input { width: 18px; min-height: 18px; }
    .color-range { display: grid; gap: 8px; padding: 8px 0 2px; }
    .range-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 12px; font-weight: 700; color: #47524c; }
    .range-value { min-width: 58px; text-align: right; color: #196f5a; }
    .dual-range { --range-low: 16%; --range-high: 33%; position: relative; height: 34px; }
    .range-rail, .range-fill { position: absolute; left: 0; right: 0; top: 15px; height: 6px; border-radius: 999px; pointer-events: none; }
    .range-rail { background: linear-gradient(90deg, #00c853 0%, #c8d28c 45%, #f2bc24 72%, #d84646 100%); opacity: 0.78; }
    .range-fill { left: var(--range-low); right: calc(100% - var(--range-high)); background: rgba(25, 111, 90, 0.58); box-shadow: 0 0 0 1px rgba(25, 111, 90, 0.18); }
    .dual-range input[type="range"] { position: absolute; inset: 0; width: 100%; min-height: 34px; margin: 0; padding: 0; appearance: none; -webkit-appearance: none; background: transparent; border: 0; pointer-events: none; }
    .dual-range input[type="range"]::-webkit-slider-runnable-track { height: 6px; background: transparent; border: 0; }
    .dual-range input[type="range"]::-webkit-slider-thumb { appearance: none; -webkit-appearance: none; width: 18px; height: 18px; margin-top: -6px; border: 2px solid #ffffff; border-radius: 50%; background: #196f5a; box-shadow: 0 1px 4px rgba(12, 17, 15, 0.28); pointer-events: auto; cursor: ew-resize; }
    .dual-range input[type="range"]::-moz-range-track { height: 6px; background: transparent; border: 0; }
    .dual-range input[type="range"]::-moz-range-thumb { width: 18px; height: 18px; border: 2px solid #ffffff; border-radius: 50%; background: #196f5a; box-shadow: 0 1px 4px rgba(12, 17, 15, 0.28); pointer-events: auto; cursor: ew-resize; }
    .range-labels { display: flex; justify-content: space-between; color: #6a746f; font-size: 11px; font-weight: 700; }
    .source-preview { display: none; gap: 8px; }
    .source-preview.is-visible { display: grid; }
    .mask-stage { width: 100%; height: 100%; min-width: 0; min-height: 0; max-width: 100%; max-height: 100%; grid-template-rows: minmax(0, 1fr); overflow: hidden; }
    .canvas:not(.is-mask-mode) .mask-stage { display: none; }
    .canvas.is-mask-mode .mask-stage { display: grid; }
    .preview-stage { width: 100%; height: 100%; min-width: 0; min-height: 0; max-width: 100%; max-height: 100%; display: grid; place-items: center; overflow: hidden; }
    .canvas.is-mask-mode .preview-stage { display: none; }
    .source-frame { position: relative; width: 100%; aspect-ratio: 4 / 3; max-height: 360px; min-height: 148px; display: grid; place-items: center; border: 1px solid #d9dfd7; border-radius: 6px; background-color: #eef2ec; background-image: linear-gradient(45deg, #d7dfd4 25%, transparent 25%), linear-gradient(-45deg, #d7dfd4 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d7dfd4 75%), linear-gradient(-45deg, transparent 75%, #d7dfd4 75%); background-position: 0 0, 0 10px, 10px -10px, -10px 0; background-size: 20px 20px; overflow: hidden; }
    .source-frame img { position: absolute; z-index: 1; left: 50%; top: 50%; display: block; width: auto; height: auto; max-width: 100%; max-height: 100%; object-fit: contain; object-position: center; transform: translate(-50%, -50%) scale(1); transform-origin: center center; will-change: transform; }
    .mask-stage .source-frame { height: 100%; min-height: 0; max-height: none; aspect-ratio: auto; }
    .mask-overlay { position: absolute; z-index: 2; display: none; touch-action: none; cursor: crosshair; opacity: 0.62; image-rendering: pixelated; transform-origin: center center; will-change: transform; }
    .source-frame.has-mask .mask-overlay { display: block; }
    .preview-statuses { min-width: 0; flex: 1 1 auto; overflow: hidden; }
    .preview-statuses .status { display: block; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .mask-toolbar { display: none; align-items: center; justify-content: flex-start; gap: 8px; flex-wrap: nowrap; min-height: 42px; padding: 5px 16px; border-bottom: 1px solid #d9dfd7; background: #fbfcfa; overflow: hidden; }
    .preview.is-mask-mode .mask-toolbar { display: flex; }
    .mask-toolbar label { display: flex; align-items: center; font-size: 12px; gap: 6px; white-space: nowrap; }
    .mask-toolbar button { width: auto; min-height: 32px; padding: 0 10px; white-space: nowrap; }
    .mask-tools { min-width: 0; display: flex; align-items: center; justify-content: flex-start; gap: 8px; flex-wrap: nowrap; }
    .mask-tools > label { width: auto; flex: 0 0 auto; }
    .mask-mode-toggle { display: inline-flex; align-items: center; gap: 4px; padding: 3px; border: 1px solid #cfd7cc; border-radius: 6px; background: #f7f9f6; }
    .mask-mode-button { min-height: 28px; border: 0; border-radius: 4px; background: transparent; color: #47524c; font-size: 12px; font-weight: 800; }
    .mask-mode-button[aria-pressed="true"] { background: #196f5a; color: #ffffff; }
    #mask-brush-size { width: 104px; min-height: 32px; }
    .mask-actions { display: flex; gap: 8px; flex: 0 0 auto; }
    .preview { min-height: 0; height: 100%; display: grid; grid-template-rows: 48px auto minmax(0, 1fr) 56px; overflow: hidden; }
    .preview > .preview-bar { grid-row: 1; }
    .preview > .mask-toolbar { grid-row: 2; }
    .preview > .canvas { grid-row: 3; }
    .preview > .preview-actions { grid-row: 4; }
    .preview-bar, .preview-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 16px; border-bottom: 1px solid #d9dfd7; }
    .preview-actions { height: 56px; min-height: 56px; max-height: 56px; border-top: 1px solid #d9dfd7; border-bottom: 0; overflow: hidden; }
    .preview-actions a.download { flex: 0 0 auto; min-width: 128px; padding: 0 18px; white-space: nowrap; }
    .confirm-matte { flex: 0 0 112px; width: 112px; min-width: 112px; max-width: 112px; height: 36px; min-height: 36px; max-height: 36px; padding: 0; display: grid; place-items: center; background: #1f5f9d; white-space: nowrap; }
    .confirm-matte[hidden] { display: none; }
    .tabs { display: inline-flex; align-items: center; gap: 4px; padding: 3px; border: 1px solid #cfd7cc; border-radius: 6px; background: #f7f9f6; flex-shrink: 0; }
    .tab { width: auto; min-height: 30px; padding: 0 10px; border: 0; border-radius: 4px; background: transparent; color: #47524c; font-size: 12px; font-weight: 700; }
    .tab[aria-selected="true"] { background: #196f5a; color: #ffffff; }
    .status { font-size: 13px; color: #5d6862; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .canvas, .candidate-thumb { background-color: #e9eee6; background-image: linear-gradient(45deg, #d3dbd0 25%, transparent 25%), linear-gradient(-45deg, #d3dbd0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d3dbd0 75%), linear-gradient(-45deg, transparent 75%, #d3dbd0 75%); }
    .canvas { width: 100%; height: 100%; min-width: 0; min-height: 0; max-width: 100%; max-height: 100%; align-self: stretch; justify-self: stretch; display: grid; place-items: stretch; padding: 16px; overflow: hidden; contain: layout paint; touch-action: none; background-position: 0 0, 0 12px, 12px -12px, -12px 0; background-size: 24px 24px; }
    .canvas.is-mask-mode { padding: 0; }
    .canvas.is-mask-mode .source-frame { border: 0; border-radius: 0; }
    .canvas.has-image { cursor: grab; }
    .canvas.is-dragging { cursor: grabbing; }
    .canvas.bg-white { background: #ffffff; }
    .canvas.bg-black { background: #111514; }
    .canvas.bg-gray { background: #aeb7b1; }
    .canvas.bg-green { background: #00c853; }
    .canvas.bg-blue { background: #4aa3ff; }
    img { max-width: 100%; max-height: 68vh; object-fit: contain; image-rendering: auto; }
    .canvas img { max-width: 100%; max-height: 100%; -webkit-user-drag: none; user-select: none; }
    .result-image { width: 100%; height: 100%; object-fit: contain; transform-origin: center center; user-select: none; pointer-events: none; will-change: transform; align-self: center; justify-self: center; }
    .empty { color: #6a746f; font-size: 14px; }
    .candidate-panel { min-height: 0; display: grid; grid-template-columns: 1fr; align-items: stretch; gap: 8px; padding: 10px; border: 1px solid #d9dfd7; border-radius: 6px; background: #fbfcfa; overflow: hidden; }
    .candidate-panel[hidden] { display: none; }
    .preview.is-mask-mode { grid-template-rows: 48px auto minmax(0, 1fr) 56px; }
    .candidate-title { font-size: 12px; font-weight: 800; color: #47524c; white-space: nowrap; }
    .candidate-list { min-width: 0; display: grid; grid-template-columns: 1fr; grid-auto-rows: 64px; gap: 8px; max-height: var(--candidate-list-max-height, 50vh); overflow-y: auto; padding: 2px; align-content: start; }
    .candidate-tab { width: 100%; min-width: 0; height: 64px; min-height: 64px; max-height: 64px; display: grid; grid-template-columns: 56px minmax(0, 1fr); align-items: center; gap: 8px; padding: 5px; border: 1px solid #cfd7cc; border-radius: 6px; background: #ffffff; color: #47524c; cursor: pointer; text-align: left; overflow: hidden; }
    .candidate-tab[aria-selected="true"] { border-color: #196f5a; box-shadow: 0 0 0 2px rgba(25, 111, 90, 0.18); color: #1c2320; }
    .semantic-candidate-tab .candidate-name { white-space: nowrap; line-height: 1.2; }
    .semantic-candidate-tab .candidate-copy { min-width: 0; display: grid; gap: 3px; overflow: hidden; }
    .candidate-subtitle { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #68746e; font-size: 10px; font-weight: 700; line-height: 1.15; }
    .candidate-thumb { width: 56px; height: 56px; display: grid; place-items: center; overflow: hidden; border-radius: 4px; background-position: 0 0, 0 6px, 6px -6px, -6px 0; background-size: 12px 12px; }
    .candidate-thumb img { width: 56px; height: 56px; object-fit: contain; display: block; }
    .candidate-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; font-weight: 800; line-height: 1.1; }
    @media (max-width: 760px) { body { height: auto; min-height: 100vh; overflow: auto; } header { padding: 0 16px; } main { height: auto; min-height: 0; grid-template-columns: 1fr; padding: 16px; overflow: visible; } form { max-height: none; } .candidate-list { max-height: none; overflow-y: visible; } .preview { min-height: 620px; height: min(720px, calc(100vh - 32px)); grid-template-rows: auto auto minmax(0, 1fr) 56px; } .preview-bar { min-height: 84px; align-items: stretch; flex-direction: column; justify-content: center; padding: 10px 16px; } .tabs { width: 100%; overflow-x: auto; } .canvas { min-height: 0; height: 100%; } .source-frame { aspect-ratio: 16 / 10; max-height: 340px; } .mask-stage .source-frame { height: 100%; min-height: 0; max-height: none; aspect-ratio: auto; } }
  </style>
</head>
<body>
  <header class="app-header">
    <nav class="primary-tabs" aria-label="主导航">
      <a class="nav-tab" href="/slice">切图</a>
      <a class="nav-tab is-active" href="/" aria-current="page">抠图</a>
      <a class="nav-tab" href="/batch">批量抠图</a>
    </nav>
    <div class="header-right">
      <span class="runtime-status" id="runtime-status" aria-live="polite">
        <span class="runtime-pill" data-runtime="local">Local</span>
        __DIRECT_WORKER_RUNTIME_ENDPOINT__
      </span>
      <span class="status" id="strategy">就绪</span>
      <a class="eval-link" href="/eval/game" target="_blank" rel="noreferrer">Game Eval</a>
    </div>
  </header>
  <main>
    <form id="matte-form">
      <label>图片<input id="file" name="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" required></label>
      <label class="check-label"><span>背景修复</span><input id="background-repair" name="background_repair" type="checkbox" checked></label>
      <label class="inline-label">后端<select id="backend" name="backend"><option value="auto" selected>Auto</option><option value="direct-worker">direct-worker</option><option value="direct-corridorkey">Direct Worker CorridorKey</option><option value="direct-known-bg-glow">Direct Worker Known-B Glow</option><option value="pymatting-known-b">pymatting-known-b</option></select></label>
      <div class="candidate-panel" aria-label="语义候选">
        <span class="candidate-title">候选</span>
        <div class="candidate-list" id="candidate-list" role="tablist" aria-label="语义候选缩略图"><span class="empty">上传后生成候选</span></div>
      </div>
      <button id="submit" type="submit" disabled>抠图</button>
      <details class="settings" id="corridorkey-settings" open>
        <summary>[设置]</summary>
        <div class="settings-grid">
          <input id="ck-screen-mode" name="corridorkey_screen_mode" type="hidden" value="auto">
          <label class="inline-label">预设<select id="ck-preset" name="corridorkey_preset"><option value="auto" selected>自动</option><option value="detail_safe">细节保护</option><option value="spill_safe">强去溢色</option><option value="manual">手动参数</option></select></label>
          <label class="inline-label">色彩空间<select id="ck-gamma-space" name="corridorkey_gamma_space"><option value="sRGB" selected>sRGB</option><option value="Linear">Linear</option></select></label>
          <div class="settings-row">
            <label>去溢色<input id="ck-despill" name="corridorkey_despill_strength" type="number" min="0" max="1" step="0.01" value="1"></label>
            <label>精修强度<input id="ck-refiner" name="corridorkey_refiner_strength" type="number" min="0" max="4" step="0.1" value="1"></label>
          </div>
          <div class="settings-row">
            <label>去斑点<select id="ck-auto-despeckle" name="corridorkey_auto_despeckle"><option value="On" selected>开启</option><option value="Off">关闭</option></select></label>
            <label>斑点尺寸<input id="ck-despeckle-size" name="corridorkey_despeckle_size" type="number" min="0" max="4096" step="1" value="400"></label>
          </div>
          <label class="check-label"><span>自动 Mask</span><input id="ck-auto-mask" name="corridorkey_auto_mask" type="checkbox"></label>
        </div>
      </details>
      <details class="settings" id="known-bg-glow-settings" open>
        <summary>[Known-B Glow]</summary>
        <div class="settings-grid">
          <label>Background removal<input id="glow-material-strength" name="known_bg_glow_material_strength" type="range" min="0" max="2" step="0.05" value="1"></label>
          <output class="range-value" id="glow-material-strength-value">1.00</output>
        </div>
      </details>
      <details class="settings" id="pymatting-settings" open>
        <summary>[PyMatting]</summary>
        <div class="settings-grid">
          <div class="settings-row">
            <label>算法<select id="pm-method" name="pymatting_method"><option value="cf" selected>closed form</option><option value="knn">KNN</option><option value="lbdm">learning based</option><option value="lkm">large kernel</option><option value="rw">random walk</option><option value="sm">shared matting</option></select></label>
            <label>色彩空间<select id="pm-image-space" name="pymatting_image_space"><option value="linear" selected>linear</option><option value="sRGB">sRGB</option></select></label>
          </div>
          <div class="settings-row">
            <label>背景<select id="pm-bg-source" name="pymatting_bg_source"><option value="auto" selected>auto</option><option value="green">green 0,200,0</option><option value="blue">blue 0,0,200</option><option value="custom">custom</option></select></label>
            <label>自定义 RGB<input id="pm-bg-color" name="pymatting_bg_color" type="text" value="0,200,0" inputmode="numeric"></label>
          </div>
          <details class="settings" id="pymatting-advanced">
            <summary>[高级]</summary>
            <div class="settings-grid">
              <div class="settings-row">
                <label>unknown 宽度<input id="pm-boundary-band" name="pymatting_boundary_band_px" type="number" min="0" max="16" step="1" value="2"></label>
                <label>CG maxiter<input id="pm-cg-maxiter" name="pymatting_cg_maxiter" type="number" min="100" max="10000" step="100" value="1000"></label>
              </div>
              <div class="settings-row">
                <label>背景阈值<input id="pm-bg-threshold" name="pymatting_bg_threshold" type="number" min="0" max="32" step="0.1" value="3.5"></label>
                <label>前景阈值<input id="pm-fg-threshold" name="pymatting_fg_threshold" type="number" min="0" max="96" step="0.5" value="24"></label>
              </div>
              <label>CG rtol<input id="pm-cg-rtol" name="pymatting_cg_rtol" type="number" min="0.00000001" max="0.01" step="any" value="0.000001"></label>
            </div>
          </details>
          <label>阴影策略<select id="shadow-mode" name="shadow_mode"><option value="auto" selected>自动</option><option value="on">保留</option><option value="off">关闭</option></select></label>
        </div>
      </details>
    </form>
    <section class="preview" id="preview-panel" aria-label="result preview">
      <div class="preview-bar">
        <strong>PNG 预览</strong>
        <div class="tabs" role="tablist" aria-label="预览背景">
          <button class="tab" type="button" role="tab" aria-selected="true" data-view="mask">遮罩</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="checker">棋盘</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="white">白底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="black">黑底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="gray">灰底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="green">绿幕</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="blue">蓝底</button>
        </div>
      </div>
      <div class="mask-toolbar" id="mask-toolbar" aria-label="遮罩工具栏">
        <div class="mask-tools" id="mask-tools">
          <button id="sam-mask-button" type="button">Sam3</button>
          <span class="mask-mode-toggle" role="group" aria-label="画笔模式">
            <button class="mask-mode-button" type="button" aria-pressed="true" data-mask-mode="keep">保留</button>
            <button class="mask-mode-button" type="button" aria-pressed="false" data-mask-mode="remove">移除</button>
            <button class="mask-mode-button" type="button" aria-pressed="false" data-mask-mode="erase">擦除</button>
          </span>
          <label>尺寸<input id="mask-brush-size" type="range" min="4" max="96" step="1" value="28"></label>
          <div class="mask-actions">
            <button id="mask-clear-button" type="button">清空</button>
          </div>
        </div>
      </div>
      <div class="canvas" id="canvas">
        <div class="preview-stage" id="preview-stage"><span class="empty">结果会显示在这里</span></div>
        <div class="source-preview mask-stage" id="source-preview" aria-live="polite">
          <div class="source-frame" id="source-frame"><span class="empty">选择图片后显示预览</span></div>
        </div>
      </div>
      <div class="preview-actions">
        <span class="preview-statuses">
          <span class="status" id="status">等待上传</span>
        </span>
        <button class="confirm-matte" id="confirm-matte" type="button" disabled hidden>确定抠图</button>
        <a class="download" id="download" aria-disabled="true" download="ermbg_rgba.png">下载 PNG</a>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById("matte-form");
    const file = document.getElementById("file");
    const backend = document.getElementById("backend");
    const directEndpoint = document.getElementById("direct-endpoint");
    const backgroundRepair = document.getElementById("background-repair");
    const submit = document.getElementById("submit");
    const statusEl = document.getElementById("status");
    const strategyEl = document.getElementById("strategy");
    const runtimeStatus = document.getElementById("runtime-status");
    const previewPanel = document.getElementById("preview-panel");
    const canvas = document.getElementById("canvas");
    const previewStage = document.getElementById("preview-stage");
    const download = document.getElementById("download");
    const confirmMatte = document.getElementById("confirm-matte");
    const candidatePanel = document.querySelector(".candidate-panel");
    const candidateList = document.getElementById("candidate-list");
    const sourcePreview = document.getElementById("source-preview");
    const sourceFrame = document.getElementById("source-frame");
    const corridorSettings = document.getElementById("corridorkey-settings");
    const knownBgGlowSettings = document.getElementById("known-bg-glow-settings");
    const pymattingSettings = document.getElementById("pymatting-settings");
    const corridorSettingControls = Array.from(document.querySelectorAll("[name^='corridorkey_']"));
    const knownBgGlowSettingControls = Array.from(document.querySelectorAll("[name^='known_bg_glow_']"));
    const pymattingSettingControls = Array.from(document.querySelectorAll("[name^='pymatting_']"));
    const shadowMode = document.getElementById("shadow-mode");
    const autoMask = document.getElementById("ck-auto-mask");
    const samMaskButton = document.getElementById("sam-mask-button");
    const metaEl = statusEl;
    const sourceMeta = statusEl;
    const maskStatus = statusEl;
    const maskBrushModeButtons = Array.from(document.querySelectorAll("[data-mask-mode]"));
    const maskBrushSize = document.getElementById("mask-brush-size");
    const maskClearButton = document.getElementById("mask-clear-button");
    const glowMaterialStrength = document.getElementById("glow-material-strength");
    const glowMaterialStrengthValue = document.getElementById("glow-material-strength-value");
    const backgroundTabs = Array.from(document.querySelectorAll("[data-bg]"));
    const viewTabs = Array.from(document.querySelectorAll("[data-view]"));
    const maskToolbarControls = Array.from(document.querySelectorAll("#mask-toolbar input, #mask-toolbar select, #mask-toolbar button"));
    let sourceUrl = null;
    let candidates = [];
    let activeCandidateIndex = -1;
    let activeView = "mask";
    let activeBackground = "checker";
    let resultImage = null;
    let previewScale = 1;
    let previewPanX = 0;
    let previewPanY = 0;
    let dragStart = null;
    let sourceImage = null;
    let maskCanvas = null;
    let maskCtx = null;
    let maskDirty = false;
    let maskPainting = false;
    let samMaskRequestId = 0;
    let maskBrushMode = "keep";
    let maskScale = 1;
    let maskPanX = 0;
    let maskPanY = 0;
    let pendingAnalyzePayload = null;
    let pendingExecuteFormData = null;
    let selectedSemanticCandidate = null;
    let semanticPreviewRenderSeq = 0;
    let analyzeRequestId = 0;

    function humanSize(bytes) { if (bytes < 1024) return `${bytes} B`; if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`; return `${(bytes / 1024 / 1024).toFixed(2)} MB`; }
    function formatElapsed(ms) { return `${(ms / 1000).toFixed(2)}s`; }
    function completionBackendLabel(payload) {
      const debug = payload && payload.debug ? payload.debug : {};
      const directWorker = debug.direct_worker || {};
      const autoRoute = debug.auto_route || {};
      const executionBackend = payload.execution_backend || directWorker.execution_backend || autoRoute.execution_backend;
      const backendLabel = executionBackend || payload.backend || effectiveBackendValue();
      const route = payload.route || directWorker.route || autoRoute.route;
      const profile = payload.execution_profile || directWorker.execution_profile || autoRoute.execution_profile || payload.parameter_profile || directWorker.parameter_profile || autoRoute.parameter_profile;
      return [backendLabel, route, profile].filter((part, index, parts) => part && parts.indexOf(part) === index).join(" · ");
    }
    function completionStatusText(payload, elapsed, serverElapsed) {
      const details = completionBackendLabel(payload);
      const base = serverElapsed ? `完成 · client ${elapsed} · server ${serverElapsed}` : `完成 · ${elapsed}`;
      return details ? `${base} · ${details}` : base;
    }
    function isAutoRouteMode() { return backend.value.split(":")[0] === "auto"; }
    function syncCandidateListHeight() {
      if (candidatePanel.hidden) return;
      if (window.matchMedia("(max-width: 760px)").matches) {
        candidateList.style.removeProperty("--candidate-list-max-height");
        return;
      }
      const rect = candidateList.getBoundingClientRect();
      if (rect.top <= 0) return;
      const bottomInset = 24;
      const height = Math.max(160, Math.floor(window.innerHeight - rect.top - bottomInset));
      candidateList.style.setProperty("--candidate-list-max-height", `${height}px`);
    }
    function scheduleCandidateListHeightSync() { requestAnimationFrame(syncCandidateListHeight); }
    function setRuntimePill(kind, label, state, title) { const pill = runtimeStatus.querySelector(`[data-runtime="${kind}"]`); if (!pill) return; const select = pill.querySelector("select"); if (select) select.title = title || label; else pill.textContent = label; pill.classList.remove("is-ok", "is-error", "is-warn"); if (state) pill.classList.add(`is-${state}`); pill.title = title || label; }
    async function refreshRuntimeStatus() { try { const response = await fetch("/api/runtime-capabilities?include_comfy=false&include_object_info=false&timeout=1.5"); if (!response.ok) throw new Error(`HTTP ${response.status}`); const payload = await response.json(); setRuntimePill("local", "Local", payload.local && payload.local.status === "ok" ? "ok" : "error", `ERMBG ${payload.local && payload.local.version ? payload.local.version : ""}`); const dw = payload.direct_worker || {}; const directOk = dw.status === "ok"; const directLabel = dw.location ? `Direct · ${dw.location}` : "Direct"; setRuntimePill("direct", directLabel, directOk ? "ok" : "error", directOk ? dw.url : (dw.error || "Direct Worker unavailable")); } catch (error) { setRuntimePill("local", "Local", "warn", "capability check failed"); setRuntimePill("direct", "Direct", "warn", "capability check failed"); } }
    function setBusy(isBusy) { submit.disabled = isBusy || !file.files.length || (!!pendingAnalyzePayload && !selectedSemanticCandidate); file.disabled = isBusy; backend.disabled = isBusy; if (directEndpoint) directEndpoint.disabled = isBusy; backgroundRepair.disabled = isBusy; corridorSettingControls.forEach((control) => { control.disabled = isBusy; }); knownBgGlowSettingControls.forEach((control) => { control.disabled = isBusy; }); pymattingSettingControls.forEach((control) => { control.disabled = isBusy; }); shadowMode.disabled = isBusy; maskToolbarControls.forEach((control) => { control.disabled = isBusy; }); confirmMatte.disabled = true; submit.textContent = isBusy ? "处理中" : "抠图"; }
    function effectiveBackendValue() { const endpoint = directEndpoint ? directEndpoint.value : ""; const baseBackend = backend.value.split(":")[0]; if (!endpoint) return backend.value; if (baseBackend === "auto" || baseBackend === "direct-worker") return `direct-worker:${endpoint}`; if (baseBackend === "corridorkey" || baseBackend === "direct-corridorkey") return `direct-corridorkey:${endpoint}`; if (baseBackend === "pymatting_known_b" || baseBackend === "pymatting-known-b" || baseBackend === "direct-pymatting-known-b") return `direct-pymatting-known-b:${endpoint}`; if (baseBackend === "known-bg-glow" || baseBackend === "known_bg_glow" || baseBackend === "direct-known-bg-glow") return `direct-known-bg-glow:${endpoint}`; return backend.value; }
    function syncBackendSettings() { const baseBackend = backend.value.split(":")[0]; corridorSettings.classList.toggle("is-visible", baseBackend === "corridorkey" || baseBackend === "direct-corridorkey"); knownBgGlowSettings.classList.toggle("is-visible", baseBackend === "known-bg-glow" || baseBackend === "known_bg_glow" || baseBackend === "direct-known-bg-glow"); pymattingSettings.classList.toggle("is-visible", baseBackend === "pymatting_known_b" || baseBackend === "pymatting-known-b"); }
    function syncCandidatePanelVisibility() { const autoRoute = isAutoRouteMode(); candidatePanel.hidden = !autoRoute; candidatePanel.setAttribute("aria-hidden", String(!autoRoute)); if (!autoRoute) { analyzeRequestId += 1; pendingAnalyzePayload = null; pendingExecuteFormData = null; selectedSemanticCandidate = null; semanticPreviewRenderSeq += 1; candidateList.innerHTML = ""; candidateList.style.removeProperty("--candidate-list-max-height"); confirmMatte.hidden = true; confirmMatte.disabled = true; return; } if (!candidateList.children.length) candidateList.innerHTML = '<span class="empty">上传后生成候选</span>'; scheduleCandidateListHeightSync(); }
    function syncGlowMaterialStrength() { glowMaterialStrengthValue.textContent = Number(glowMaterialStrength.value || 1).toFixed(2); }
    function syncPreviewMode() { const maskMode = activeView === "mask"; previewPanel.classList.toggle("is-mask-mode", maskMode); canvas.classList.toggle("is-mask-mode", maskMode); viewTabs.forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.view === activeView))); backgroundTabs.forEach((tab) => tab.setAttribute("aria-selected", String(activeView === "preview" && tab.dataset.bg === activeBackground))); if (maskMode) layoutMaskCanvas(); }
    function setPreviewView(view) { activeView = view; syncPreviewMode(); }
    function setPreviewBackground(mode) { activeBackground = mode; activeView = "preview"; canvas.classList.remove("bg-white", "bg-black", "bg-gray", "bg-green", "bg-blue"); if (mode !== "checker") canvas.classList.add(`bg-${mode}`); syncPreviewMode(); }
    function resetPreviewTransform() { previewScale = 1; previewPanX = 0; previewPanY = 0; dragStart = null; applyPreviewTransform(); }
    function applyPreviewTransform() { if (resultImage) resultImage.style.transform = `translate(${previewPanX}px, ${previewPanY}px) scale(${previewScale})`; }
    function resetMaskTransform() { maskScale = 1; maskPanX = 0; maskPanY = 0; applyMaskTransform(); }
    function maskTransformCss() { return `translate(-50%, -50%) translate(${maskPanX}px, ${maskPanY}px) scale(${maskScale})`; }
    function applyMaskTransform() { const transform = maskTransformCss(); if (sourceImage) sourceImage.style.transform = transform; if (maskCanvas) maskCanvas.style.transform = transform; }
    function resetResult() { candidates.forEach((candidate) => { if (candidate.revoke) URL.revokeObjectURL(candidate.url); }); candidates = []; activeCandidateIndex = -1; resultImage = null; pendingAnalyzePayload = null; pendingExecuteFormData = null; selectedSemanticCandidate = null; semanticPreviewRenderSeq += 1; resetPreviewTransform(); previewStage.innerHTML = '<span class="empty">结果会显示在这里</span>'; canvas.classList.remove("has-image", "is-dragging"); candidateList.innerHTML = '<span class="empty">上传后生成候选</span>'; scheduleCandidateListHeightSync(); metaEl.textContent = "RGBA PNG"; download.removeAttribute("href"); download.setAttribute("aria-disabled", "true"); confirmMatte.hidden = true; confirmMatte.disabled = true; }
    function clearMaskState() { maskDirty = false; maskPainting = false; if (maskCanvas) { maskCanvas.remove(); maskCanvas = null; maskCtx = null; } sourceFrame.classList.remove("has-mask"); resetMaskTransform(); }
    function layoutMaskCanvas() { if (!sourceImage || sourceImage.naturalWidth <= 0 || sourceImage.naturalHeight <= 0) return; const frameRect = sourceFrame.getBoundingClientRect(); if (frameRect.width <= 0 || frameRect.height <= 0) { requestAnimationFrame(layoutMaskCanvas); return; } const fit = Math.min(frameRect.width / sourceImage.naturalWidth, frameRect.height / sourceImage.naturalHeight); if (!Number.isFinite(fit) || fit <= 0) return; const displayWidth = Math.max(1, sourceImage.naturalWidth * fit); const displayHeight = Math.max(1, sourceImage.naturalHeight * fit); const transform = maskTransformCss(); sourceImage.style.width = `${displayWidth}px`; sourceImage.style.height = `${displayHeight}px`; sourceImage.style.maxWidth = "none"; sourceImage.style.maxHeight = "none"; sourceImage.style.left = "50%"; sourceImage.style.top = "50%"; sourceImage.style.transform = transform; if (maskCanvas) { maskCanvas.style.left = "50%"; maskCanvas.style.top = "50%"; maskCanvas.style.width = `${displayWidth}px`; maskCanvas.style.height = `${displayHeight}px`; maskCanvas.style.transform = transform; } }
    function ensureMaskCanvas(width, height) { if (!maskCanvas) { maskCanvas = document.createElement("canvas"); maskCanvas.className = "mask-overlay"; sourceFrame.appendChild(maskCanvas); maskCanvas.addEventListener("pointerdown", beginMaskPaint); maskCanvas.addEventListener("pointermove", paintMask); maskCanvas.addEventListener("pointerup", endMaskPaint); maskCanvas.addEventListener("pointercancel", endMaskPaint); } maskCanvas.width = width; maskCanvas.height = height; maskCanvas.style.display = "block"; sourceFrame.classList.add("has-mask"); maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true }); layoutMaskCanvas(); }
    function exportMaskLayerFile(width, height, pixels, mode) { return new Promise((resolve) => { const exportCanvas = document.createElement("canvas"); exportCanvas.width = width; exportCanvas.height = height; const exportCtx = exportCanvas.getContext("2d"); const out = exportCtx.createImageData(width, height); let count = 0; for (let i = 0; i < pixels.data.length; i += 4) { const active = pixels.data[i + 3] > 8; const removePixel = pixels.data[i] > pixels.data[i + 2]; const include = active && (mode === "remove" ? removePixel : !removePixel); const value = include ? 255 : 0; if (include) count += 1; out.data[i] = value; out.data[i + 1] = value; out.data[i + 2] = value; out.data[i + 3] = 255; } if (count === 0) { resolve(null); return; } exportCtx.putImageData(out, 0, 0); exportCanvas.toBlob((blob) => { resolve(blob ? new File([blob], `user_${mode}_mask.png`, { type: "image/png" }) : null); }, "image/png"); }); }
    async function exportUserMaskFiles() { if (!maskCanvas || !maskCtx || !maskDirty) return { keep: null, remove: null }; const width = maskCanvas.width; const height = maskCanvas.height; const pixels = maskCtx.getImageData(0, 0, width, height); const keep = await exportMaskLayerFile(width, height, pixels, "keep"); const remove = await exportMaskLayerFile(width, height, pixels, "remove"); return { keep, remove }; }
    function waitForSourceImage() { return new Promise((resolve, reject) => { if (!sourceImage) { reject(new Error("请先选择图片")); return; } if (sourceImage.complete && sourceImage.naturalWidth > 0 && sourceImage.naturalHeight > 0) { resolve(sourceImage); return; } const done = () => { cleanup(); if (sourceImage.naturalWidth > 0 && sourceImage.naturalHeight > 0) resolve(sourceImage); else reject(new Error("图片预览尚未载入")); }; const fail = () => { cleanup(); reject(new Error("图片预览载入失败")); }; const cleanup = () => { sourceImage.removeEventListener("load", done); sourceImage.removeEventListener("error", fail); }; sourceImage.addEventListener("load", done, { once: true }); sourceImage.addEventListener("error", fail, { once: true }); }); }
    function loadMaskOverlay(dataUrl) { return new Promise((resolve, reject) => { const img = new Image(); img.onload = async () => { try { setPreviewView("mask"); sourcePreview.classList.add("is-visible"); await waitForSourceImage(); const width = sourceImage && sourceImage.naturalWidth > 0 ? sourceImage.naturalWidth : img.naturalWidth; const height = sourceImage && sourceImage.naturalHeight > 0 ? sourceImage.naturalHeight : img.naturalHeight; if (width <= 0 || height <= 0) throw new Error("mask 尺寸无效"); ensureMaskCanvas(width, height); maskCtx.clearRect(0, 0, width, height); maskCtx.globalCompositeOperation = "source-over"; maskCtx.drawImage(img, 0, 0, width, height); const pixels = maskCtx.getImageData(0, 0, width, height); const data = pixels.data; for (let i = 0; i < data.length; i += 4) { const value = data[i]; data[i] = 0; data[i + 1] = 190; data[i + 2] = 255; data[i + 3] = value > 8 ? 255 : 0; } maskCtx.putImageData(pixels, 0, 0); maskDirty = true; requestAnimationFrame(() => { layoutMaskCanvas(); maskCanvas.style.display = "block"; sourceFrame.classList.add("has-mask"); }); maskStatus.textContent = "已生成 Sam3 mask"; resolve(); } catch (error) { reject(error); } }; img.onerror = () => reject(new Error("Sam3 mask 载入失败")); img.src = dataUrl; }); }
    function canvasPoint(event) { const rect = maskCanvas.getBoundingClientRect(); return { x: ((event.clientX - rect.left) / rect.width) * maskCanvas.width, y: ((event.clientY - rect.top) / rect.height) * maskCanvas.height }; }
    function setMaskBrushMode(mode) { maskBrushMode = ["keep", "remove", "erase"].includes(mode) ? mode : "keep"; maskBrushModeButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.maskMode === maskBrushMode))); }
    function drawMaskBrush(event) { if (!maskCanvas || !maskCtx) return; const p = canvasPoint(event); const radius = Number(maskBrushSize.value || 28); maskCtx.save(); maskCtx.globalCompositeOperation = maskBrushMode === "erase" ? "destination-out" : "source-over"; maskCtx.fillStyle = maskBrushMode === "remove" ? "rgba(255,80,72,1)" : "rgba(0,190,255,1)"; maskCtx.beginPath(); maskCtx.arc(p.x, p.y, radius, 0, Math.PI * 2); maskCtx.fill(); maskCtx.restore(); maskDirty = true; maskStatus.textContent = "edited mask"; }
    function beginMaskPaint(event) { if (!maskCanvas) return; event.preventDefault(); event.stopPropagation(); maskPainting = true; maskCanvas.setPointerCapture(event.pointerId); drawMaskBrush(event); }
    function paintMask(event) { if (!maskPainting) return; event.preventDefault(); event.stopPropagation(); drawMaskBrush(event); }
    function endMaskPaint(event) { if (!maskPainting) return; event.preventDefault(); event.stopPropagation(); maskPainting = false; try { maskCanvas.releasePointerCapture(event.pointerId); } catch (_) {} }
    function renderCandidateTabs() { candidateList.innerHTML = ""; if (!candidates.length) { candidateList.innerHTML = '<span class="empty">上传后生成候选</span>'; scheduleCandidateListHeightSync(); return; } candidates.forEach((candidate, index) => { const button = document.createElement("button"); button.className = "candidate-tab"; button.type = "button"; button.role = "tab"; button.setAttribute("aria-selected", String(index === activeCandidateIndex)); button.dataset.index = String(index); button.title = candidate.label; const thumb = document.createElement("span"); thumb.className = "candidate-thumb"; const img = document.createElement("img"); img.src = candidate.url; img.alt = `${candidate.label} 缩略图`; thumb.appendChild(img); const label = document.createElement("span"); label.className = "candidate-name"; label.textContent = candidate.label; button.appendChild(thumb); button.appendChild(label); button.addEventListener("click", () => setActiveCandidate(index)); candidateList.appendChild(button); }); scheduleCandidateListHeightSync(); }
    function setActiveCandidate(index) { if (index < 0 || index >= candidates.length) return; const candidate = candidates[index]; activeCandidateIndex = index; resetPreviewTransform(); previewStage.innerHTML = ""; const img = document.createElement("img"); img.src = candidate.url; img.alt = candidate.label; img.draggable = false; img.className = "result-image"; resultImage = img; canvas.classList.add("has-image"); previewStage.appendChild(img); applyPreviewTransform(); download.href = candidate.url; download.download = candidate.downloadName; download.setAttribute("aria-disabled", "false"); metaEl.textContent = candidate.meta; renderCandidateTabs(); setPreviewView("preview"); }
    function setCandidatePayloads(payload, name) { const stem = name.replace(/\\.[^.]+$/, ""); candidates = (payload.candidates || []).map((candidate, index) => ({ url: candidate.rgba, revoke: false, label: candidate.label || `结果 ${index + 1}`, selected: candidate.selected === true, meta: `结果 ${index + 1} / ${payload.candidates.length} · ${candidate.kind || "RGBA PNG"}`, downloadName: candidate.filename || `${stem}_${candidate.id || `candidate_${index + 1}`}.png` })); if (!candidates.length) throw new Error("没有可显示的执行结果"); const selectedIndex = candidates.findIndex((candidate) => candidate.selected); const index = selectedIndex >= 0 ? selectedIndex : 0; const candidate = candidates[index]; activeCandidateIndex = index; resetPreviewTransform(); previewStage.innerHTML = ""; const img = document.createElement("img"); img.src = candidate.url; img.alt = candidate.label; img.draggable = false; img.className = "result-image"; resultImage = img; canvas.classList.add("has-image"); previewStage.appendChild(img); applyPreviewTransform(); download.href = candidate.url; download.download = candidate.downloadName; download.setAttribute("aria-disabled", "false"); metaEl.textContent = candidate.meta; setPreviewView("preview"); }
    function dataUrlToFile(dataUrl, filename) { const [header, base64] = dataUrl.split(","); const mime = (header.match(/data:(.*);base64/) || [])[1] || "image/png"; const binary = atob(base64); const bytes = new Uint8Array(binary.length); for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i); return new File([bytes], filename, { type: mime }); }
    async function generateSamMask() { if (!file.files.length) { setPreviewView("mask"); maskStatus.textContent = "请先选择图片"; return; } const requestId = samMaskRequestId + 1; samMaskRequestId = requestId; setPreviewView("mask"); const formData = new FormData(); formData.append("file", file.files[0]); setBusy(true); maskStatus.textContent = "Sam3 生成中"; try { await waitForSourceImage(); const response = await fetch("/api/sam-mask", { method: "POST", body: formData }); if (!response.ok) { let message = "Sam3 mask 失败"; try { const payload = await response.json(); message = payload.detail || message; } catch (_) {} throw new Error(message); } const payload = await response.json(); if (requestId !== samMaskRequestId) return; await loadMaskOverlay(payload.mask); const elapsed = typeof payload.server_elapsed_sec === "number" ? formatElapsed(payload.server_elapsed_sec * 1000) : ""; maskStatus.textContent = elapsed ? `已生成 Sam3 mask · ${elapsed}` : "已生成 Sam3 mask"; } catch (error) { if (requestId === samMaskRequestId) { clearMaskState(); maskStatus.textContent = error.message; } } finally { if (requestId === samMaskRequestId) setBusy(false); } }
    function loadPendingSlice() { const raw = sessionStorage.getItem("ermbgPendingSlice"); if (!raw) return; sessionStorage.removeItem("ermbgPendingSlice"); try { const pending = JSON.parse(raw); const sliceFile = dataUrlToFile(pending.rgb, pending.filename || "slice.png"); const transfer = new DataTransfer(); transfer.items.add(sliceFile); file.files = transfer.files; backend.value = "auto"; backgroundRepair.checked = true; setPreviewView("mask"); sourcePreview.classList.add("is-visible"); sourceFrame.innerHTML = ""; const img = document.createElement("img"); sourceImage = img; resetMaskTransform(); img.alt = "切图预览"; img.onload = () => { resetMaskTransform(); layoutMaskCanvas(); analyzeCurrentUpload(); }; img.onerror = () => { sourceMeta.textContent = "切图预览载入失败"; }; sourceFrame.appendChild(img); img.src = pending.rgb; sourceMeta.textContent = `${sliceFile.name} · ${pending.meta || "来自切图"}`; statusEl.textContent = "正在生成候选"; strategyEl.textContent = effectiveBackendValue(); setBusy(false); } catch (error) { statusEl.textContent = "切图载入失败"; } }

    file.addEventListener("change", () => { if (!file.files.length) return; resetResult(); clearMaskState(); setPreviewView("mask"); backgroundRepair.checked = true; statusEl.textContent = "正在生成候选"; strategyEl.textContent = effectiveBackendValue(); if (sourceUrl) URL.revokeObjectURL(sourceUrl); const selected = file.files[0]; sourceUrl = URL.createObjectURL(selected); sourcePreview.classList.add("is-visible"); sourceFrame.innerHTML = ""; const img = document.createElement("img"); sourceImage = img; resetMaskTransform(); img.alt = "上传图片预览"; img.onload = () => { sourceMeta.textContent = `${img.naturalWidth}x${img.naturalHeight} · ${humanSize(selected.size)}`; resetMaskTransform(); layoutMaskCanvas(); analyzeCurrentUpload(); }; img.onerror = () => { sourceMeta.textContent = `无法预览 · ${humanSize(selected.size)}`; }; sourceFrame.appendChild(img); img.src = sourceUrl; });
    backend.addEventListener("change", () => { strategyEl.textContent = effectiveBackendValue(); syncBackendSettings(); });
    if (directEndpoint) directEndpoint.addEventListener("change", () => { strategyEl.textContent = effectiveBackendValue(); });
    glowMaterialStrength.addEventListener("input", syncGlowMaterialStrength);
    samMaskButton.addEventListener("click", () => generateSamMask());
    maskBrushModeButtons.forEach((button) => button.addEventListener("click", () => setMaskBrushMode(button.dataset.maskMode)));
    maskClearButton.addEventListener("click", () => { if (!maskCanvas || !maskCtx) return; maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height); maskDirty = false; maskStatus.textContent = "mask cleared"; });
    window.addEventListener("resize", () => { layoutMaskCanvas(); syncCandidateListHeight(); });
    viewTabs.forEach((tab) => tab.addEventListener("click", () => setPreviewView(tab.dataset.view)));
    backgroundTabs.forEach((tab) => tab.addEventListener("click", () => setPreviewBackground(tab.dataset.bg)));
    [canvas, previewStage, sourceFrame, sourcePreview].forEach((element) => { ["dragstart", "dragover", "drop"].forEach((type) => element.addEventListener(type, (event) => event.preventDefault())); });
    canvas.addEventListener("wheel", (event) => { if (activeView === "mask") { if (!sourceImage) return; event.preventDefault(); const rect = sourceFrame.getBoundingClientRect(); const centerX = rect.left + rect.width / 2; const centerY = rect.top + rect.height / 2; const pointerX = event.clientX - centerX; const pointerY = event.clientY - centerY; const previousScale = maskScale; const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12; maskScale = Math.min(8, Math.max(1, maskScale * factor)); maskPanX = pointerX - ((pointerX - maskPanX) * maskScale) / previousScale; maskPanY = pointerY - ((pointerY - maskPanY) * maskScale) / previousScale; if (maskScale === 1) { maskPanX = 0; maskPanY = 0; } applyMaskTransform(); return; } if (!resultImage) return; event.preventDefault(); const rect = canvas.getBoundingClientRect(); const centerX = rect.left + rect.width / 2; const centerY = rect.top + rect.height / 2; const pointerX = event.clientX - centerX; const pointerY = event.clientY - centerY; const previousScale = previewScale; const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12; previewScale = Math.min(8, Math.max(0.2, previewScale * factor)); previewPanX = pointerX - ((pointerX - previewPanX) * previewScale) / previousScale; previewPanY = pointerY - ((pointerY - previewPanY) * previewScale) / previousScale; applyPreviewTransform(); }, { passive: false });
    canvas.addEventListener("pointerdown", (event) => { if (activeView === "mask" || !resultImage) return; dragStart = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX: previewPanX, panY: previewPanY }; canvas.setPointerCapture(event.pointerId); canvas.classList.add("is-dragging"); });
    canvas.addEventListener("pointermove", (event) => { if (!dragStart || dragStart.pointerId !== event.pointerId) return; previewPanX = dragStart.panX + event.clientX - dragStart.x; previewPanY = dragStart.panY + event.clientY - dragStart.y; applyPreviewTransform(); });
    function endDrag(event) { if (!dragStart || dragStart.pointerId !== event.pointerId) return; dragStart = null; canvas.classList.remove("is-dragging"); }
    canvas.addEventListener("pointerup", endDrag); canvas.addEventListener("pointercancel", endDrag); canvas.addEventListener("dblclick", () => { if (activeView !== "mask") resetPreviewTransform(); });
    async function buildMatteFormData() {
      const formData = new FormData();
      formData.append("file", file.files[0]);
      formData.append("backend", effectiveBackendValue());
      formData.append("background_repair", backgroundRepair.checked ? "true" : "false");
      formData.append("shadow_mode", shadowMode.value);
      formData.append("parameter_source", backend.value === "auto" ? "auto" : "manual");
      corridorSettingControls.forEach((control) => {
        if (control.type === "checkbox") formData.append(control.name, control.checked ? "true" : "false");
        else formData.append(control.name, control.value);
      });
      knownBgGlowSettingControls.forEach((control) => {
        formData.append(control.name, control.value);
      });
      pymattingSettingControls.forEach((control) => {
        if (control.type === "checkbox") formData.append(control.name, control.checked ? "true" : "false");
        else formData.append(control.name, control.value);
      });
      const userMasks = await exportUserMaskFiles();
      if (userMasks.keep) formData.append("user_keep_mask", userMasks.keep);
      if (userMasks.remove) formData.append("user_remove_mask", userMasks.remove);
      const shouldUseCustomMask = backend.value === "corridorkey" && !autoMask.checked && userMasks.keep;
      if (shouldUseCustomMask) formData.append("corridorkey_hint_mask", userMasks.keep);
      return formData;
    }
    function appendAnalyzeControls(formData) { formData.append("background_repair", backgroundRepair.checked ? "true" : "false"); formData.append("corridorkey_screen_mode", document.getElementById("ck-screen-mode").value || "auto"); formData.append("corridorkey_preset", document.querySelector("[name='corridorkey_preset']").value || "auto"); }
    function semanticRegionsForCandidate(candidate) { const ids = Array.isArray(candidate && candidate.regions) ? candidate.regions : []; const regions = Array.isArray(pendingAnalyzePayload && pendingAnalyzePayload.ambiguity_regions) ? pendingAnalyzePayload.ambiguity_regions : []; return ids.length ? regions.filter((region) => ids.includes(region.id)) : regions; }
    function serverSemanticPreviewAsset(candidate, mode) { const assets = pendingAnalyzePayload && pendingAnalyzePayload.preview_assets; const refs = candidate && candidate.preview && candidate.preview.assets; const key = refs && refs[mode]; const asset = assets && key ? assets[key] : null; return asset && asset.data_url ? asset.data_url : null; }
    function semanticPreviewKind(candidate) { const refs = candidate && candidate.preview && candidate.preview.assets ? candidate.preview.assets : {}; if (refs.trimap) return "trimap"; if (refs.hint) return "hint"; if (refs.overlay) return "overlay"; return "regions"; }
    function semanticCandidateDisplayLabel(candidate, fallback) { const raw = (candidate && (candidate.label || candidate.id) ? String(candidate.label || candidate.id) : fallback).trim(); return raw === "Auto default" || raw === "auto_default" ? "默认" : raw; }
    function executionAnalysisPayload(payload, selectedCandidate) {
      if (!payload || typeof payload !== "object") return {};
      const slim = { ...payload };
      const selectedId = selectedCandidate && selectedCandidate.id ? String(selectedCandidate.id) : String(payload.default_candidate_id || "auto_default");
      const selectedRefs = selectedCandidate && selectedCandidate.preview && selectedCandidate.preview.assets ? selectedCandidate.preview.assets : {};
      const selectedTrimapRef = selectedRefs && selectedRefs.trimap ? String(selectedRefs.trimap) : "";
      const selectedTrimapAsset = selectedTrimapRef && payload.preview_assets ? payload.preview_assets[selectedTrimapRef] : null;
      // Analyze includes full-size PNG data URLs for UI previews. Execute only
      // needs the selected candidate trimap so local semantic assembly survives
      // without exceeding Starlette's 1MB text-part parser limit.
      slim.preview_assets = selectedTrimapAsset ? { [selectedTrimapRef]: selectedTrimapAsset } : {};
      if (Array.isArray(slim.candidates)) {
        slim.candidates = slim.candidates.map((candidate) => {
          if (!candidate || typeof candidate !== "object") return candidate;
          const copy = { ...candidate };
          if (String(copy.id || "") === selectedId && selectedTrimapRef) {
            copy.preview = { assets: { trimap: selectedTrimapRef } };
          } else {
            delete copy.preview;
          }
          return copy;
        });
      }
      return slim;
    }
    function routeAssetKindLabel(payload) { const route = payload && payload.route ? payload.route : {}; const raw = route.asset_kind || route.asset_type || ""; const labels = { asset: "资产", button: "按钮", character: "角色", game_ui: "游戏 UI", icon: "图标" }; return raw ? labels[raw] || raw : ""; }
    function loadPreviewImage(src) { return new Promise((resolve, reject) => { const img = new Image(); img.onload = () => resolve(img); img.onerror = () => reject(new Error("候选预览载入失败")); img.src = src; }); }
    async function hintOverlayDataUrl(hintDataUrl, maxSide = 0) {
      await waitForSourceImage();
      const sourceWidth = sourceImage.naturalWidth;
      const sourceHeight = sourceImage.naturalHeight;
      if (sourceWidth <= 0 || sourceHeight <= 0) throw new Error("原图尺寸无效");
      const scale = maxSide > 0 ? Math.min(1, maxSide / Math.max(sourceWidth, sourceHeight)) : 1;
      const width = Math.max(1, Math.round(sourceWidth * scale));
      const height = Math.max(1, Math.round(sourceHeight * scale));
      const previewCanvas = document.createElement("canvas");
      previewCanvas.width = width;
      previewCanvas.height = height;
      const ctx = previewCanvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(sourceImage, 0, 0, width, height);
      const hintImage = await loadPreviewImage(hintDataUrl);
      const hintCanvas = document.createElement("canvas");
      hintCanvas.width = width;
      hintCanvas.height = height;
      const hintCtx = hintCanvas.getContext("2d", { willReadFrequently: true });
      hintCtx.drawImage(hintImage, 0, 0, width, height);
      const hintPixels = hintCtx.getImageData(0, 0, width, height).data;
      const output = ctx.getImageData(0, 0, width, height);
      const data = output.data;
      for (let i = 0; i < data.length; i += 4) {
        const value = hintPixels[i] / 255;
        if (value <= 0.004) continue;
        const alpha = Math.min(0.58, value * 0.55);
        data[i] = Math.round(data[i] * (1 - alpha) + 255 * alpha);
        data[i + 1] = Math.round(data[i + 1] * (1 - alpha) + 72 * alpha);
        data[i + 2] = Math.round(data[i + 2] * (1 - alpha) + 64 * alpha);
      }
      ctx.putImageData(output, 0, 0);
      return previewCanvas.toDataURL("image/png");
    }
    async function applySemanticThumbnailPreview(img, candidate) {
      const kind = semanticPreviewKind(candidate);
      const dataUrl = serverSemanticPreviewAsset(candidate, kind);
      if (!dataUrl) return;
      if (kind !== "hint") {
        img.src = dataUrl;
        return;
      }
      if (sourceImage && sourceImage.src) img.src = sourceImage.src;
      try {
        img.src = await hintOverlayDataUrl(dataUrl, 128);
      } catch (_) {
        if (!img.src) img.src = dataUrl;
      }
    }
    function regionBox(region, width, height) { const raw = Array.isArray(region && region.bbox_xyxy) ? region.bbox_xyxy : [0, 0, 0, 0]; const x1 = Math.max(0, Math.min(width, Number(raw[0]) || 0)); const y1 = Math.max(0, Math.min(height, Number(raw[1]) || 0)); const x2 = Math.max(x1, Math.min(width, Number(raw[2]) || 0)); const y2 = Math.max(y1, Math.min(height, Number(raw[3]) || 0)); return { x: x1, y: y1, w: Math.max(1, x2 - x1), h: Math.max(1, y2 - y1) }; }
    function presentSemanticPreview(dataUrl, candidate, label) {
      previewStage.innerHTML = "";
      const img = document.createElement("img");
      img.src = dataUrl;
      img.alt = candidate.label || "语义候选预览";
      img.draggable = false;
      img.className = "result-image";
      resultImage = img;
      canvas.classList.add("has-image");
      previewStage.appendChild(img);
      resetPreviewTransform();
      setPreviewView("preview");
      download.removeAttribute("href");
      download.setAttribute("aria-disabled", "true");
      metaEl.textContent = `${candidate.label || candidate.id || "语义候选"} · ${label}`;
    }
    async function renderSemanticPreview(candidate) {
      const renderSeq = ++semanticPreviewRenderSeq;
      const previewKind = semanticPreviewKind(candidate);
      const serverPreview = serverSemanticPreviewAsset(candidate, previewKind);
      if (previewKind === "hint" && serverPreview) {
        try {
          const overlayPreview = await hintOverlayDataUrl(serverPreview);
          if (renderSeq !== semanticPreviewRenderSeq) return;
          presentSemanticPreview(overlayPreview, candidate, "hint overlay");
        } catch (_) {
          presentSemanticPreview(serverPreview, candidate, "hint");
        }
        return;
      }
      if (!sourceImage || sourceImage.naturalWidth <= 0 || sourceImage.naturalHeight <= 0) {
        try {
          await waitForSourceImage();
        } catch (_) {
          previewStage.innerHTML = '<span class="empty">预览载入中</span>';
          return;
        }
      }
      const width = sourceImage.naturalWidth;
      const height = sourceImage.naturalHeight;
      const previewCanvas = document.createElement("canvas");
      previewCanvas.width = width;
      previewCanvas.height = height;
      const ctx = previewCanvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(sourceImage, 0, 0, width, height);
      if (previewKind === "trimap" && serverPreview) {
        try {
          const trimapImage = await loadPreviewImage(serverPreview);
          if (renderSeq !== semanticPreviewRenderSeq) return;
          const trimapCanvas = document.createElement("canvas");
          trimapCanvas.width = width;
          trimapCanvas.height = height;
          const trimapCtx = trimapCanvas.getContext("2d", { willReadFrequently: true });
          trimapCtx.drawImage(trimapImage, 0, 0, width, height);
          const trimapPixels = trimapCtx.getImageData(0, 0, width, height).data;
          const output = ctx.getImageData(0, 0, width, height);
          const data = output.data;
          const unknownAlpha = 0.42;
          const sureFgAlpha = 0.28;
          for (let i = 0; i < data.length; i += 4) {
            const value = trimapPixels[i];
            if (value >= 64 && value <= 191) {
              data[i] = Math.round(data[i] * (1 - unknownAlpha) + 255 * unknownAlpha);
              data[i + 1] = Math.round(data[i + 1] * (1 - unknownAlpha) + 72 * unknownAlpha);
              data[i + 2] = Math.round(data[i + 2] * (1 - unknownAlpha) + 64 * unknownAlpha);
            } else if (value > 191) {
              data[i] = Math.round(data[i] * (1 - sureFgAlpha) + 18 * sureFgAlpha);
              data[i + 1] = Math.round(data[i + 1] * (1 - sureFgAlpha) + 24 * sureFgAlpha);
              data[i + 2] = Math.round(data[i + 2] * (1 - sureFgAlpha) + 22 * sureFgAlpha);
            }
          }
          ctx.putImageData(output, 0, 0);
        } catch (_) {}
      }
      if (renderSeq !== semanticPreviewRenderSeq) return;
      presentSemanticPreview(previewCanvas.toDataURL("image/png"), candidate, previewKind === "trimap" ? "trimap unknown" : previewKind);
    }
    function setSelectedSemanticCandidate(candidate) { selectedSemanticCandidate = candidate; candidateList.querySelectorAll(".semantic-candidate-tab").forEach((button) => button.setAttribute("aria-selected", String(button.dataset.candidateId === (candidate.id || "")))); confirmMatte.hidden = true; confirmMatte.disabled = true; renderSemanticPreview(candidate); statusEl.textContent = "候选已选择，点击抠图执行"; setBusy(false); }
    async function executeSemanticCandidate(candidate) {
      if (!candidate) return;
      if (!pendingExecuteFormData || !pendingAnalyzePayload) return;
      const executeFormData = new FormData();
      for (const [key, value] of pendingExecuteFormData.entries()) executeFormData.append(key, value);
      executeFormData.append("selected_candidate_id", candidate.id || pendingAnalyzePayload.default_candidate_id || "auto_default");
      executeFormData.append("semantic_decision", JSON.stringify(candidate.decision || {}));
      executeFormData.append("analysis_payload", JSON.stringify(executionAnalysisPayload(pendingAnalyzePayload, candidate)));
      setBusy(true);
      statusEl.textContent = "正在执行决策";
      const startedAt = performance.now();
      try {
        const response = await fetch("/api/execute-candidate", { method: "POST", body: executeFormData });
        if (!response.ok) {
          let message = "处理失败";
          try {
            const payload = await response.json();
            message = payload.detail || message;
          } catch (_) {}
          throw new Error(message);
        }
        const payload = await response.json();
        const elapsed = formatElapsed(performance.now() - startedAt);
        const serverElapsed = typeof payload.server_elapsed_sec === "number" ? formatElapsed(payload.server_elapsed_sec * 1000) : null;
        setCandidatePayloads(payload, file.files[0].name);
        const strategy = payload.strategy || "done";
        const bg = Array.isArray(payload.background) ? payload.background.join(",") : "";
        statusEl.textContent = completionStatusText(payload, elapsed, serverElapsed);
        strategyEl.textContent = bg ? `${strategy} · ${bg}` : strategy;
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    }
    function renderSemanticCandidates(payload) {
      pendingAnalyzePayload = payload;
      selectedSemanticCandidate = null;
      previewStage.innerHTML = '<span class="empty">选择语义候选</span>';
      canvas.classList.remove("has-image", "is-dragging");
      download.removeAttribute("href");
      download.setAttribute("aria-disabled", "true");
      confirmMatte.hidden = true;
      confirmMatte.disabled = true;
      candidateList.innerHTML = "";
      const semanticCandidates = payload.candidates || [];
      const assetKindLabel = routeAssetKindLabel(payload);
      semanticCandidates.forEach((candidate, index) => {
        const displayLabel = semanticCandidateDisplayLabel(candidate, `候选 ${index + 1}`);
        const button = document.createElement("button");
        button.className = "candidate-tab semantic-candidate-tab";
        button.type = "button";
        button.role = "tab";
        button.setAttribute("aria-selected", "false");
        button.dataset.candidateId = candidate.id || "";
        button.title = [displayLabel, assetKindLabel, candidate.intent].filter(Boolean).join(" · ") || "semantic candidate";
        const thumb = document.createElement("span");
        thumb.className = "candidate-thumb";
        const img = document.createElement("img");
        img.alt = `${displayLabel} 预览`;
        thumb.appendChild(img);
        applySemanticThumbnailPreview(img, candidate);
        const copy = document.createElement("span");
        copy.className = "candidate-copy";
        const label = document.createElement("span");
        label.className = "candidate-name";
        label.textContent = displayLabel;
        copy.appendChild(label);
        if (assetKindLabel) {
          const subtitle = document.createElement("span");
          subtitle.className = "candidate-subtitle";
          subtitle.textContent = assetKindLabel;
          copy.appendChild(subtitle);
        }
        button.appendChild(thumb);
        button.appendChild(copy);
        button.addEventListener("click", () => setSelectedSemanticCandidate(candidate));
        candidateList.appendChild(button);
      });
      if (!semanticCandidates.length) {
        candidateList.innerHTML = '<span class="empty">没有可执行候选</span>';
        confirmMatte.disabled = true;
      }
      scheduleCandidateListHeightSync();
      const defaultCandidate = semanticCandidates.find((candidate) => candidate.id === payload.default_candidate_id) || semanticCandidates.find((candidate) => candidate.default === true) || semanticCandidates[0];
      if (defaultCandidate) setSelectedSemanticCandidate(defaultCandidate);
      setPreviewView("preview");
      if (!defaultCandidate) metaEl.textContent = `语义候选 ${semanticCandidates.length}`;
    }
    async function analyzeCurrentUpload() {
      if (!file.files.length) return;
      if (!isAutoRouteMode()) { statusEl.textContent = "手动路径就绪，点击抠图执行"; setBusy(false); return; }
      const requestId = analyzeRequestId + 1;
      analyzeRequestId = requestId;
      setBusy(true);
      statusEl.textContent = "正在分析";
      strategyEl.textContent = effectiveBackendValue();
      const analyzeFormData = new FormData();
      analyzeFormData.append("file", file.files[0]);
      appendAnalyzeControls(analyzeFormData);
      try {
        pendingExecuteFormData = await buildMatteFormData();
        const response = await fetch("/api/analyze-candidates", { method: "POST", body: analyzeFormData });
        if (!response.ok) {
          let message = "分析失败";
          try {
            const payload = await response.json();
            message = payload.detail || message;
          } catch (_) {}
          throw new Error(message);
        }
        const payload = await response.json();
        if (requestId !== analyzeRequestId) return;
        pendingAnalyzePayload = payload;
        renderSemanticCandidates(payload);
        statusEl.textContent = selectedSemanticCandidate ? "候选已就绪，点击抠图执行" : "请选择候选后抠图";
        strategyEl.textContent = payload.route && payload.route.route ? payload.route.route : (payload.status || "ready");
      } catch (error) {
        if (requestId === analyzeRequestId) statusEl.textContent = error.message;
      } finally {
        if (requestId === analyzeRequestId) setBusy(false);
      }
    }
    async function executeManualMatte() {
      if (!file.files.length) return;
      syncCandidatePanelVisibility();
      const executeFormData = await buildMatteFormData();
      executeFormData.append("selected_candidate_id", "auto_default");
      executeFormData.append("semantic_decision", "{}");
      executeFormData.append("analysis_payload", "{}");
      setBusy(true);
      statusEl.textContent = "正在执行手动路径";
      const startedAt = performance.now();
      try {
        const response = await fetch("/api/execute-candidate", { method: "POST", body: executeFormData });
        if (!response.ok) {
          let message = "处理失败";
          try {
            const payload = await response.json();
            message = payload.detail || message;
          } catch (_) {}
          throw new Error(message);
        }
        const payload = await response.json();
        const elapsed = formatElapsed(performance.now() - startedAt);
        const serverElapsed = typeof payload.server_elapsed_sec === "number" ? formatElapsed(payload.server_elapsed_sec * 1000) : null;
        setCandidatePayloads(payload, file.files[0].name);
        statusEl.textContent = completionStatusText(payload, elapsed, serverElapsed);
        strategyEl.textContent = payload.strategy || completionBackendLabel(payload) || effectiveBackendValue();
        syncCandidatePanelVisibility();
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    }
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!file.files.length) return;
      if (!isAutoRouteMode()) { await executeManualMatte(); return; }
      if (!pendingAnalyzePayload || !selectedSemanticCandidate) {
        await analyzeCurrentUpload();
      }
      if (selectedSemanticCandidate) await executeSemanticCandidate(selectedSemanticCandidate);
    });
    confirmMatte.addEventListener("click", () => executeSemanticCandidate(selectedSemanticCandidate));
    syncGlowMaterialStrength();
    syncBackendSettings();
    syncCandidatePanelVisibility();
    syncPreviewMode();
    scheduleCandidateListHeightSync();
    refreshRuntimeStatus();
    loadPendingSlice();
  </script>
</body>
</html>""")

    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1c2320;
      background: #f5f7f4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .app-header { min-height: 56px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 24px; border-bottom: 1px solid #d9dfd7; background: #ffffff; }
    .primary-tabs { min-width: 0; display: flex; align-items: center; justify-content: flex-start; gap: 4px; overflow-x: auto; }
    .nav-tab { display: inline-flex; align-items: center; justify-content: center; min-height: 34px; padding: 0 12px; border-radius: 6px; color: #47524c; font-size: 13px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    .nav-tab.is-active { background: #196f5a; color: #ffffff; }
    .header-right { margin-left: auto; display: flex; align-items: center; justify-content: flex-end; gap: 12px; min-width: 0; }
    .eval-link { color: #196f5a; font-size: 13px; font-weight: 900; text-decoration: none; white-space: nowrap; }
    main {
      width: min(1120px, 100%);
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
      align-items: start;
    }
    form, .preview {
      background: #ffffff;
      border: 1px solid #d9dfd7;
      border-radius: 8px;
    }
    form {
      padding: 18px;
      display: grid;
      gap: 16px;
    }
    label, .field {
      display: grid;
      gap: 8px;
      font-size: 13px;
      font-weight: 600;
      color: #47524c;
    }
    input, select, button {
      width: 100%;
      min-height: 40px;
      border-radius: 6px;
      border: 1px solid #b8c1b7;
      background: #ffffff;
      color: #1c2320;
      font: inherit;
    }
    input[type="file"] { padding: 8px; }
    button, a.download {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      background: #196f5a;
      color: #ffffff;
      text-decoration: none;
      font-weight: 700;
      cursor: pointer;
    }
    a.mode-button {
      text-decoration: none;
    }
    button:disabled, a.download[aria-disabled="true"] {
      opacity: 0.55;
      cursor: not-allowed;
      pointer-events: none;
    }
    .mode-switch {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      padding: 4px;
      border: 1px solid #cfd7cc;
      border-radius: 6px;
      background: #f7f9f6;
    }
    .mode-button {
      width: 100%;
      min-height: 34px;
      border: 0;
      background: transparent;
      color: #47524c;
      font-size: 13px;
    }
    .mode-button[aria-pressed="true"] {
      background: #196f5a;
      color: #ffffff;
    }
    .slice-settings {
      display: none;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .slice-settings.is-visible {
      display: grid;
    }
    .slice-settings input {
      min-height: 38px;
      padding: 0 10px;
    }
    .source-preview {
      display: none;
      gap: 10px;
    }
    .source-preview.is-visible {
      display: grid;
    }
    .source-frame {
      width: 100%;
      aspect-ratio: 4 / 3;
      min-height: 148px;
      display: grid;
      place-items: center;
      border: 1px solid #d9dfd7;
      border-radius: 6px;
      background-color: #eef2ec;
      background-image:
        linear-gradient(45deg, #d7dfd4 25%, transparent 25%),
        linear-gradient(-45deg, #d7dfd4 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d7dfd4 75%),
        linear-gradient(-45deg, transparent 75%, #d7dfd4 75%);
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
      background-size: 20px 20px;
    }
    .source-frame img {
      display: block;
      width: auto;
      height: auto;
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      object-position: center;
    }
    .source-meta {
      min-height: auto;
      font-size: 12px;
      line-height: 1.4;
      color: #5d6862;
      overflow-wrap: anywhere;
    }
    .preview {
      min-height: 520px;
      display: grid;
      grid-template-rows: 48px 1fr 104px 56px;
      overflow: hidden;
    }
    .preview-bar, .preview-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 16px;
      border-bottom: 1px solid #d9dfd7;
    }
    .preview-actions {
      border-top: 1px solid #d9dfd7;
      border-bottom: 0;
    }
    .preview-actions button {
      width: auto;
      min-width: 108px;
      padding: 0 14px;
    }
    .tabs {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px;
      border: 1px solid #cfd7cc;
      border-radius: 6px;
      background: #f7f9f6;
      flex-shrink: 0;
    }
    .tab {
      width: auto;
      min-height: 30px;
      padding: 0 10px;
      border: 0;
      border-radius: 4px;
      background: transparent;
      color: #47524c;
      font-size: 12px;
      font-weight: 700;
    }
    .tab[aria-selected="true"] {
      background: #196f5a;
      color: #ffffff;
    }
    .status {
      font-size: 13px;
      color: #5d6862;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .canvas {
      min-height: 416px;
      display: grid;
      place-items: center;
      padding: 16px;
      overflow: hidden;
      touch-action: none;
      background-color: #e9eee6;
      background-image:
        linear-gradient(45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(-45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d3dbd0 75%),
        linear-gradient(-45deg, transparent 75%, #d3dbd0 75%);
      background-position: 0 0, 0 12px, 12px -12px, -12px 0;
      background-size: 24px 24px;
    }
    .canvas.has-image {
      cursor: grab;
    }
    .canvas.is-dragging {
      cursor: grabbing;
    }
    .canvas.bg-white { background: #ffffff; }
    .canvas.bg-black { background: #111514; }
    .canvas.bg-gray { background: #aeb7b1; }
    .canvas.bg-green { background: #00c853; }
    .canvas.bg-blue { background: #4aa3ff; }
    img {
      max-width: 100%;
      max-height: 68vh;
      object-fit: contain;
      image-rendering: auto;
    }
    .result-image {
      transform-origin: center center;
      user-select: none;
      pointer-events: none;
      will-change: transform;
    }
    .empty {
      color: #6a746f;
      font-size: 14px;
    }
    .candidate-panel {
      min-height: 104px;
      display: grid;
      grid-template-columns: auto 1fr;
      align-items: center;
      gap: 12px;
      padding: 12px 16px;
      border-top: 1px solid #d9dfd7;
      background: #fbfcfa;
    }
    .candidate-title {
      font-size: 12px;
      font-weight: 800;
      color: #47524c;
      white-space: nowrap;
    }
    .candidate-list {
      min-width: 0;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 2px;
    }
    .candidate-tab {
      width: 92px;
      min-width: 92px;
      min-height: 76px;
      display: grid;
      grid-template-rows: 48px auto;
      gap: 5px;
      padding: 5px;
      border: 1px solid #cfd7cc;
      border-radius: 6px;
      background: #ffffff;
      color: #47524c;
      cursor: pointer;
    }
    .candidate-tab[aria-selected="true"] {
      border-color: #196f5a;
      box-shadow: 0 0 0 2px rgba(25, 111, 90, 0.18);
      color: #1c2320;
    }
    .candidate-thumb {
      width: 100%;
      height: 48px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 4px;
      background-color: #e9eee6;
      background-image:
        linear-gradient(45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(-45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d3dbd0 75%),
        linear-gradient(-45deg, transparent 75%, #d3dbd0 75%);
      background-position: 0 0, 0 6px, 6px -6px, -6px 0;
      background-size: 12px 12px;
    }
    .candidate-thumb img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .candidate-name {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
      font-weight: 800;
      line-height: 1.1;
    }
    .slice-list {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      max-height: 232px;
      overflow-y: auto;
      overflow-x: hidden;
    }
    .slice-row {
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr) auto;
      align-items: center;
      gap: 10px;
      padding: 8px;
      border: 1px solid #cfd7cc;
      border-radius: 6px;
      background: #ffffff;
    }
    .slice-thumb {
      width: 72px;
      height: 56px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 4px;
      background-color: #e9eee6;
      background-image:
        linear-gradient(45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(-45deg, #d3dbd0 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d3dbd0 75%),
        linear-gradient(-45deg, transparent 75%, #d3dbd0 75%);
      background-position: 0 0, 0 6px, 6px -6px, -6px 0;
      background-size: 12px 12px;
    }
    .slice-thumb img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .slice-info {
      min-width: 0;
      display: grid;
      gap: 3px;
    }
    .slice-label {
      font-size: 13px;
      font-weight: 800;
      color: #1c2320;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .slice-meta {
      font-size: 12px;
      color: #5d6862;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .slice-row button {
      width: auto;
      min-height: 34px;
      padding: 0 12px;
      white-space: nowrap;
    }
    .slice-actions {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .slice-download {
      min-height: 34px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0 12px;
      border-radius: 6px;
      background: #edf4ef;
      color: #196f5a;
      font-size: 13px;
      font-weight: 800;
      text-decoration: none;
      white-space: nowrap;
    }
    @media (max-width: 760px) {
      .app-header { padding: 0 16px; }
      main {
        grid-template-columns: 1fr;
        padding: 16px;
      }
      .preview {
        min-height: 420px;
        grid-template-rows: auto 1fr 104px 56px;
      }
      .preview-bar {
        min-height: 84px;
        align-items: stretch;
        flex-direction: column;
        justify-content: center;
        padding: 10px 16px;
      }
      .tabs {
        width: 100%;
        overflow-x: auto;
      }
      .canvas { min-height: 312px; }
      .candidate-panel {
        grid-template-columns: 1fr;
        align-items: stretch;
        gap: 8px;
      }
    }
  </style>
</head>
<body>
  <header class="app-header">
    <nav class="primary-tabs" aria-label="主导航">
      <a class="nav-tab" href="/slice">切图</a>
      <a class="nav-tab is-active" href="/" aria-current="page">抠图</a>
      <a class="nav-tab" href="/batch">批量抠图</a>
    </nav>
    <div class="header-right">
      <span class="status" id="strategy">就绪</span>
      <a class="eval-link" href="/eval/game" target="_blank" rel="noreferrer">Game Eval</a>
    </div>
  </header>
  <main>
    <form id="matte-form">
      <label>
        图片
        <input id="file" name="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" required>
      </label>
      <div class="source-preview" id="source-preview" aria-live="polite">
        <div class="source-frame" id="source-frame">
          <span class="empty">选择图片后显示预览</span>
        </div>
        <div class="source-meta" id="source-meta">未选择图片</div>
      </div>
      <label class="checkbox-field">
        <input id="background-repair" name="background_repair" type="checkbox" checked>
        <span>背景修复</span>
      </label>
      <div class="field">
        任务
        <span class="mode-switch" role="group" aria-label="任务">
          <a class="mode-button" href="/" data-task="matte" role="button" aria-pressed="true">抠图</a>
          <a class="mode-button" href="/slice" data-task="slice" role="button" aria-pressed="false">切图</a>
        </span>
        <select id="task" name="task" hidden>
          <option value="matte" selected>抠图</option>
          <option value="slice">切图</option>
        </select>
      </div>
      <label>
        后端
        <select id="backend" name="backend">
          <option value="auto" selected>Auto</option>
          <option value="direct-worker">direct-worker</option>
          <option value="direct-corridorkey">Direct Worker CorridorKey</option>
          <option value="direct-known-bg-glow">Direct Worker Known-B Glow</option>
          <option value="pymatting-known-b">pymatting-known-b</option>
        </select>
      </label>
      <div class="slice-settings" id="slice-settings">
        <label>
          最小面积
          <input id="slice-min-area" name="min_area" type="number" min="1" step="1" value="64">
        </label>
        <label>
          边距
          <input id="slice-padding" name="padding" type="number" min="0" step="1" value="4">
        </label>
      </div>
      <button id="submit" type="submit">抠图</button>
    </form>
    <section class="preview" aria-label="result preview">
      <div class="preview-bar">
        <strong>PNG 预览</strong>
        <div class="tabs" role="tablist" aria-label="预览背景">
          <button class="tab" type="button" role="tab" aria-selected="true" data-bg="checker">棋盘</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="white">白底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="black">黑底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="gray">灰底</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="green">绿幕</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-bg="blue">蓝底</button>
        </div>
        <span class="status" id="status">等待上传</span>
      </div>
      <div class="canvas" id="canvas">
        <span class="empty">结果会显示在这里</span>
      </div>
      <div class="candidate-panel" aria-label="切图与执行结果">
        <span class="candidate-title">切图/结果</span>
        <div class="candidate-list" id="candidate-list" role="tablist" aria-label="切图与执行结果缩略图">
          <span class="empty">切图或结果会显示在这里</span>
        </div>
      </div>
      <div class="preview-actions">
        <span class="status" id="meta">RGBA PNG</span>
        <button id="confirm-slices" type="button" disabled hidden>生成切图</button>
        <a class="download" id="download" aria-disabled="true" download="ermbg_rgba.png">下载 PNG</a>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById("matte-form");
    const file = document.getElementById("file");
    const backend = document.getElementById("backend");
    const task = document.getElementById("task");
    const modeButtons = Array.from(document.querySelectorAll(".mode-button"));
    const sliceSettings = document.getElementById("slice-settings");
    const sliceMinArea = document.getElementById("slice-min-area");
    const slicePadding = document.getElementById("slice-padding");
    const backgroundRepair = document.getElementById("background-repair");
    const submit = document.getElementById("submit");
    const confirmSlices = document.getElementById("confirm-slices");
    const statusEl = document.getElementById("status");
    const strategyEl = document.getElementById("strategy");
    const canvas = document.getElementById("canvas");
    const download = document.getElementById("download");
    const candidateList = document.getElementById("candidate-list");
    const metaEl = document.getElementById("meta");
    const sourcePreview = document.getElementById("source-preview");
    const sourceFrame = document.getElementById("source-frame");
    const sourceMeta = document.getElementById("source-meta");
    const tabs = Array.from(document.querySelectorAll(".tab"));
    let sourceUrl = null;
    let candidates = [];
    let activeCandidateIndex = -1;
    let resultImage = null;
    let previewScale = 1;
    let previewPanX = 0;
    let previewPanY = 0;
    let dragStart = null;
    let slicePreviewPayload = null;
    let sourceMetaBase = "未选择图片";
    let sourceCheckerMeta = "";

    function humanSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    }

    function formatElapsed(ms) {
      return `${(ms / 1000).toFixed(2)}s`;
    }

    function completionBackendLabel(payload) {
      const debug = payload && payload.debug ? payload.debug : {};
      const directWorker = debug.direct_worker || {};
      const autoRoute = debug.auto_route || {};
      const executionBackend = payload.execution_backend || directWorker.execution_backend || autoRoute.execution_backend;
      const backendLabel = executionBackend || payload.backend || backend.value;
      const route = payload.route || directWorker.route || autoRoute.route;
      const profile = payload.execution_profile || directWorker.execution_profile || autoRoute.execution_profile || payload.parameter_profile || directWorker.parameter_profile || autoRoute.parameter_profile;
      return [backendLabel, route, profile].filter((part, index, parts) => part && parts.indexOf(part) === index).join(" · ");
    }

    function completionStatusText(payload, elapsed, serverElapsed) {
      const details = completionBackendLabel(payload);
      const base = serverElapsed ? `完成 · client ${elapsed} · server ${serverElapsed}` : `完成 · ${elapsed}`;
      return details ? `${base} · ${details}` : base;
    }

    function setBusy(isBusy) {
      submit.disabled = isBusy;
      file.disabled = isBusy;
      backend.disabled = isBusy || task.value === "slice";
      task.disabled = isBusy;
      modeButtons.forEach((button) => {
        button.setAttribute("aria-disabled", String(isBusy));
      });
      sliceMinArea.disabled = isBusy;
      slicePadding.disabled = isBusy;
      backgroundRepair.disabled = isBusy;
      confirmSlices.disabled = isBusy || !slicePreviewPayload;
      submit.textContent = isBusy ? "处理中" : (task.value === "slice" ? "自动标注" : "抠图");
    }

    function setTaskMode(mode) {
      task.value = mode;
      modeButtons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.task === mode));
      });
      sliceSettings.classList.toggle("is-visible", mode === "slice");
      backend.disabled = mode === "slice";
      submit.textContent = mode === "slice" ? "自动标注" : "抠图";
      statusEl.textContent = mode === "slice" ? "等待切图标注" : "等待抠图";
      confirmSlices.hidden = mode !== "slice";
      confirmSlices.disabled = !slicePreviewPayload;
      metaEl.textContent = mode === "slice" ? "切图预览" : "RGBA PNG";
      if (mode !== "slice") {
        slicePreviewPayload = null;
      }
    }

    function setPreviewBackground(mode) {
      canvas.classList.remove("bg-white", "bg-black", "bg-gray", "bg-green", "bg-blue");
      if (mode !== "checker") canvas.classList.add(`bg-${mode}`);
      tabs.forEach((tab) => {
        tab.setAttribute("aria-selected", String(tab.dataset.bg === mode));
      });
    }

    function resetResult() {
      candidates.forEach((candidate) => {
        if (candidate.revoke) URL.revokeObjectURL(candidate.url);
      });
      candidates = [];
      activeCandidateIndex = -1;
      resultImage = null;
      slicePreviewPayload = null;
      resetPreviewTransform();
      canvas.innerHTML = '<span class="empty">结果会显示在这里</span>';
      canvas.classList.remove("has-image", "is-dragging");
      candidateList.className = "candidate-list";
      candidateList.innerHTML = '<span class="empty">切图或结果会显示在这里</span>';
      metaEl.textContent = "RGBA PNG";
      confirmSlices.disabled = true;
      download.removeAttribute("href");
      download.setAttribute("aria-disabled", "true");
    }

    function clampScale(value) {
      return Math.min(8, Math.max(0.2, value));
    }

    function applyPreviewTransform() {
      if (!resultImage) return;
      resultImage.style.transform = `translate(${previewPanX}px, ${previewPanY}px) scale(${previewScale})`;
    }

    function resetPreviewTransform() {
      previewScale = 1;
      previewPanX = 0;
      previewPanY = 0;
      dragStart = null;
      applyPreviewTransform();
    }

    function renderCandidateTabs() {
      candidateList.innerHTML = "";
      if (!candidates.length) {
        candidateList.innerHTML = '<span class="empty">结果会显示在这里</span>';
        return;
      }
      candidates.forEach((candidate, index) => {
        const button = document.createElement("button");
        button.className = "candidate-tab";
        button.type = "button";
        button.role = "tab";
        button.setAttribute("aria-selected", String(index === activeCandidateIndex));
        button.dataset.index = String(index);
        button.title = candidate.label;

        const thumb = document.createElement("span");
        thumb.className = "candidate-thumb";
        const img = document.createElement("img");
        img.src = candidate.url;
        img.alt = `${candidate.label} 缩略图`;
        thumb.appendChild(img);

        const label = document.createElement("span");
        label.className = "candidate-name";
        label.textContent = candidate.label;

        button.appendChild(thumb);
        button.appendChild(label);
        button.addEventListener("click", () => setActiveCandidate(index));
        candidateList.appendChild(button);
      });
    }

    function setActiveCandidate(index) {
      if (index < 0 || index >= candidates.length) return;
      const candidate = candidates[index];
      activeCandidateIndex = index;
      resetPreviewTransform();
      canvas.innerHTML = "";
      const img = document.createElement("img");
      img.src = candidate.url;
      img.alt = candidate.label;
      img.draggable = false;
      img.className = "result-image";
      resultImage = img;
      canvas.classList.add("has-image");
      canvas.appendChild(img);
      applyPreviewTransform();
      download.href = candidate.url;
      download.download = candidate.downloadName;
      download.setAttribute("aria-disabled", "false");
      metaEl.textContent = candidate.meta;
      renderCandidateTabs();
    }

    function setDownload(blob, name) {
      resetResult();
      const url = URL.createObjectURL(blob);
      const stem = name.replace(/\\.[^.]+$/, "");
      candidates = [{
        url,
        revoke: true,
        label: "自动结果",
        meta: "结果 1 / 1 · RGBA PNG",
        downloadName: `${stem}_rgba.png`,
      }];
      setActiveCandidate(0);
    }

    function setCandidatePayloads(payload, name) {
      resetResult();
      const stem = name.replace(/\\.[^.]+$/, "");
      candidates = (payload.candidates || []).map((candidate, index) => ({
        url: candidate.rgba,
        revoke: false,
        label: candidate.label || `结果 ${index + 1}`,
        selected: candidate.selected === true,
        meta: `结果 ${index + 1} / ${payload.candidates.length} · ${candidate.kind || "RGBA PNG"}`,
        downloadName: candidate.filename || `${stem}_${candidate.id || `candidate_${index + 1}`}.png`,
      }));
      if (!candidates.length) {
        throw new Error("没有可显示的执行结果");
      }
      const selectedIndex = candidates.findIndex((candidate) => candidate.selected);
      setActiveCandidate(selectedIndex >= 0 ? selectedIndex : 0);
    }

    function setSlicePreviewPayload(payload) {
      resetResult();
      slicePreviewPayload = payload;
      canvas.innerHTML = "";
      const img = document.createElement("img");
      img.src = payload.annotated;
      img.alt = "自动切图标注预览";
      img.draggable = false;
      img.className = "result-image";
      resultImage = img;
      canvas.classList.add("has-image");
      canvas.appendChild(img);
      applyPreviewTransform();
      candidateList.innerHTML = '<span class="empty">确认标注后生成切图列表</span>';
      confirmSlices.hidden = false;
      confirmSlices.disabled = !payload.count;
      metaEl.textContent = `检测到 ${payload.count || 0} 个矩形`;
      download.removeAttribute("href");
      download.setAttribute("aria-disabled", "true");
    }

    function dataUrlToBlob(dataUrl) {
      const [header, base64] = dataUrl.split(",");
      const mime = (header.match(/data:(.*);base64/) || [])[1] || "image/png";
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      return new Blob([bytes], { type: mime });
    }

    function defaultSemanticCandidate(payload) {
      const candidates = payload.candidates || [];
      return candidates.find((candidate) => candidate.id === payload.default_candidate_id) || candidates[0] || { id: "auto_default", decision: { policy: "auto_default" } };
    }

    function executionAnalysisPayload(payload, selectedCandidate) {
      if (!payload || typeof payload !== "object") return {};
      const slim = { ...payload };
      const selectedId = selectedCandidate && selectedCandidate.id ? String(selectedCandidate.id) : String(payload.default_candidate_id || "auto_default");
      const selectedRefs = selectedCandidate && selectedCandidate.preview && selectedCandidate.preview.assets ? selectedCandidate.preview.assets : {};
      const selectedTrimapRef = selectedRefs && selectedRefs.trimap ? String(selectedRefs.trimap) : "";
      const selectedTrimapAsset = selectedTrimapRef && payload.preview_assets ? payload.preview_assets[selectedTrimapRef] : null;
      // Preview assets are base64 PNGs for UI display. Keep only the selected
      // candidate trimap because Execute consumes it as the explicit Known-B
      // trimap contract for local semantic decisions such as keep-hole.
      slim.preview_assets = selectedTrimapAsset ? { [selectedTrimapRef]: selectedTrimapAsset } : {};
      if (Array.isArray(slim.candidates)) {
        slim.candidates = slim.candidates.map((candidate) => {
          if (!candidate || typeof candidate !== "object") return candidate;
          const copy = { ...candidate };
          if (String(copy.id || "") === selectedId && selectedTrimapRef) {
            copy.preview = { assets: { trimap: selectedTrimapRef } };
          } else {
            delete copy.preview;
          }
          return copy;
        });
      }
      return slim;
    }

    async function executeDefaultDecision(fileObj, filename, fields) {
      const analyzeFormData = new FormData();
      analyzeFormData.append("file", fileObj, filename);
      analyzeFormData.append("background_repair", fields.background_repair || "false");
      const analyzeResponse = await fetch("/api/analyze-candidates", { method: "POST", body: analyzeFormData });
      if (!analyzeResponse.ok) {
        let message = "分析失败";
        try {
          const payload = await analyzeResponse.json();
          message = payload.detail || message;
        } catch (_) {}
        throw new Error(message);
      }
      const analysisPayload = await analyzeResponse.json();
      const semanticCandidate = defaultSemanticCandidate(analysisPayload);
      const executeFormData = new FormData();
      executeFormData.append("file", fileObj, filename);
      Object.entries(fields).forEach(([key, value]) => executeFormData.append(key, value));
      executeFormData.append("selected_candidate_id", semanticCandidate.id || analysisPayload.default_candidate_id || "auto_default");
      executeFormData.append("semantic_decision", JSON.stringify(semanticCandidate.decision || {}));
      executeFormData.append("analysis_payload", JSON.stringify(executionAnalysisPayload(analysisPayload, semanticCandidate)));
      const response = await fetch("/api/execute-candidate", { method: "POST", body: executeFormData });
      if (!response.ok) {
        let message = "处理失败";
        try {
          const payload = await response.json();
          message = payload.detail || message;
        } catch (_) {}
        throw new Error(message);
      }
      return response.json();
    }

    async function matteSlice(crop) {
      const cropBlob = dataUrlToBlob(crop.rgb);
      setBusy(true);
      statusEl.textContent = `正在抠图 · ${crop.label}`;
      strategyEl.textContent = crop.label;
      const startedAt = performance.now();
      try {
        const payload = await executeDefaultDecision(cropBlob, crop.filename, {
          backend: backend.value,
          background_repair: "false",
        });
        const elapsed = formatElapsed(performance.now() - startedAt);
        const serverElapsed = typeof payload.server_elapsed_sec === "number" ? formatElapsed(payload.server_elapsed_sec * 1000) : null;
        setCandidatePayloads(payload, crop.filename);
        const strategy = payload.strategy || "done";
        const bg = Array.isArray(payload.background) ? payload.background.join(",") : "";
        statusEl.textContent = completionStatusText(payload, elapsed, serverElapsed);
        strategyEl.textContent = bg ? `${crop.label} · ${strategy} · ${bg}` : `${crop.label} · ${strategy}`;
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    }

    function renderSliceCrops(payload) {
      resetResult();
      candidateList.className = "candidate-list slice-list";
      if (!payload.crops || !payload.crops.length) {
        candidateList.innerHTML = '<span class="empty">没有检测到可切割主体</span>';
        return;
      }
      canvas.innerHTML = '<span class="empty">选择下方切图进入抠图流程</span>';
      payload.crops.forEach((crop) => {
        const row = document.createElement("div");
        row.className = "slice-row";

        const thumb = document.createElement("span");
        thumb.className = "slice-thumb";
        const img = document.createElement("img");
        img.src = crop.rgb;
        img.alt = `${crop.label} 预览`;
        thumb.appendChild(img);

        const info = document.createElement("span");
        info.className = "slice-info";
        const label = document.createElement("span");
        label.className = "slice-label";
        label.textContent = crop.label;
        const meta = document.createElement("span");
        meta.className = "slice-meta";
        meta.textContent = crop.meta || "";
        info.appendChild(label);
        info.appendChild(meta);

        const actions = document.createElement("span");
        actions.className = "slice-actions";
        const downloadCrop = document.createElement("a");
        downloadCrop.className = "slice-download";
        downloadCrop.href = crop.rgb;
        downloadCrop.download = crop.filename || `${crop.id || crop.label || "slice"}_rgb.png`;
        downloadCrop.textContent = "下载";
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "抠图";
        button.addEventListener("click", () => matteSlice(crop));
        actions.appendChild(downloadCrop);
        actions.appendChild(button);

        row.appendChild(thumb);
        row.appendChild(info);
        row.appendChild(actions);
        candidateList.appendChild(row);
      });
      confirmSlices.disabled = true;
      metaEl.textContent = `已生成 ${payload.count || payload.crops.length} 张切图`;
      statusEl.textContent = "切图完成";
    }

    file.addEventListener("change", () => {
      if (!file.files.length) return;
      resetResult();
      statusEl.textContent = task.value === "slice" ? "等待切图标注" : "等待抠图";
      strategyEl.textContent = backend.value;
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);

      const selected = file.files[0];
      sourceMetaBase = `${selected.name} · ${humanSize(selected.size)}`;
      sourceCheckerMeta = "";
      sourceUrl = URL.createObjectURL(selected);
      sourcePreview.classList.add("is-visible");
      sourceFrame.innerHTML = "";
      const img = document.createElement("img");
      img.src = sourceUrl;
      img.alt = "上传图片预览";
      img.onload = () => {
        sourceMetaBase = `${selected.name} · ${img.naturalWidth}x${img.naturalHeight} · ${humanSize(selected.size)}`;
        sourceMeta.textContent = `${sourceMetaBase}${sourceCheckerMeta}`;
      };
      img.onerror = () => {
        sourceMetaBase = `${selected.name} · 无法预览 · ${humanSize(selected.size)}`;
        sourceMeta.textContent = `${sourceMetaBase}${sourceCheckerMeta}`;
      };
      sourceFrame.appendChild(img);
      backgroundRepair.checked = true;
      setBusy(false);
    });

    backgroundRepair.addEventListener("change", () => {
      if (task.value !== "slice") return;
      slicePreviewPayload = null;
      confirmSlices.disabled = true;
      candidateList.innerHTML = '<span class="empty">预处理已修改，重新自动标注</span>';
    });

    modeButtons.forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        resetResult();
        setTaskMode(button.dataset.task);
        const nextPath = button.dataset.task === "slice" ? "/slice" : "/";
        if (window.location.pathname !== nextPath) {
          window.history.replaceState(null, "", nextPath);
        }
      });
    });
    setTaskMode(window.location.pathname === "/slice" ? "slice" : task.value);

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => setPreviewBackground(tab.dataset.bg));
    });

    canvas.addEventListener("wheel", (event) => {
      if (!resultImage) return;
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const pointerX = event.clientX - centerX;
      const pointerY = event.clientY - centerY;
      const previousScale = previewScale;
      const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
      previewScale = clampScale(previewScale * factor);
      previewPanX = pointerX - ((pointerX - previewPanX) * previewScale) / previousScale;
      previewPanY = pointerY - ((pointerY - previewPanY) * previewScale) / previousScale;
      applyPreviewTransform();
    }, { passive: false });

    canvas.addEventListener("pointerdown", (event) => {
      if (!resultImage) return;
      dragStart = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        panX: previewPanX,
        panY: previewPanY,
      };
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("is-dragging");
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!dragStart || dragStart.pointerId !== event.pointerId) return;
      previewPanX = dragStart.panX + event.clientX - dragStart.x;
      previewPanY = dragStart.panY + event.clientY - dragStart.y;
      applyPreviewTransform();
    });

    function endDrag(event) {
      if (!dragStart || dragStart.pointerId !== event.pointerId) return;
      dragStart = null;
      canvas.classList.remove("is-dragging");
    }

    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("dblclick", () => resetPreviewTransform());

    confirmSlices.addEventListener("click", async () => {
      if (!file.files.length || !slicePreviewPayload) return;
      const formData = new FormData();
      formData.append("file", file.files[0]);
      formData.append("background_repair", backgroundRepair.checked ? "true" : "false");
      formData.append("min_area", sliceMinArea.value || "64");
      formData.append("padding", slicePadding.value || "4");
      setBusy(true);
      statusEl.textContent = "正在生成切图";
      try {
        const response = await fetch("/api/slice-crops", { method: "POST", body: formData });
        if (!response.ok) {
          let message = "处理失败";
          try {
            const payload = await response.json();
            message = payload.detail || message;
          } catch (_) {}
          throw new Error(message);
        }
        renderSliceCrops(await response.json());
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!file.files.length) return;
      const formData = new FormData();
      formData.append("file", file.files[0]);
      formData.append("background_repair", backgroundRepair.checked ? "true" : "false");
      if (task.value === "slice") {
        formData.append("min_area", sliceMinArea.value || "64");
        formData.append("padding", slicePadding.value || "4");
      } else {
        formData.append("backend", backend.value);
      }
      setBusy(true);
      statusEl.textContent = task.value === "slice" ? "正在自动标注" : "正在抠图";
      strategyEl.textContent = backend.value;
      const startedAt = performance.now();
      try {
        if (task.value === "slice") {
          const response = await fetch("/api/slice-preview", { method: "POST", body: formData });
          if (!response.ok) {
            let message = "处理失败";
            try {
              const payload = await response.json();
              message = payload.detail || message;
            } catch (_) {}
            throw new Error(message);
          }
          const payload = await response.json();
          setSlicePreviewPayload(payload);
          const bg = Array.isArray(payload.background_color) ? payload.background_color.join(",") : "";
          statusEl.textContent = "标注完成";
          strategyEl.textContent = bg ? `slice · ${bg}` : "slice";
        } else {
          const payload = await executeDefaultDecision(file.files[0], file.files[0].name, {
            backend: backend.value,
            background_repair: backgroundRepair.checked ? "true" : "false",
          });
          const elapsed = formatElapsed(performance.now() - startedAt);
          const serverElapsed = typeof payload.server_elapsed_sec === "number" ? formatElapsed(payload.server_elapsed_sec * 1000) : null;
          setCandidatePayloads(payload, file.files[0].name);
          const strategy = payload.strategy || "done";
          const bg = Array.isArray(payload.background) ? payload.background.join(",") : "";
          statusEl.textContent = completionStatusText(payload, elapsed, serverElapsed);
          strategyEl.textContent = bg ? `${strategy} · ${bg}` : strategy;
        }
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setBusy(false);
      }
    });
  </script>
</body>
</html>"""


@app.get("/slice", response_class=HTMLResponse)
def slice_page() -> str:
    return _slice_page_html()


@app.get("/batch", response_class=HTMLResponse)
def batch_page() -> str:
    return _batch_page_html()


@app.post("/api/batch-results.zip")
def batch_results_zip_endpoint(payload: Annotated[dict[str, Any], Body()]) -> Response:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise HTTPException(status_code=400, detail="items must contain at least one successful result")
    items = [item for item in raw_items if isinstance(item, dict) and item.get("rgba")]
    if not items:
        raise HTTPException(status_code=400, detail="items must contain at least one rgba result")
    batch_dir, content, filename = _write_batch_result_artifacts(items)
    rel_dir = str(batch_dir.relative_to(PROJECT_ROOT)) if _is_relative_to(batch_dir, PROJECT_ROOT) else str(batch_dir)
    return Response(
        content=content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-ERMBG-Batch-Dir": rel_dir,
            "X-ERMBG-Batch-Count": str(len(items)),
        },
    )


def _batch_page_html() -> str:
    return _inject_backend_options("""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG Batch Matte</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1c2320; background: #f5f7f4; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    .app-header { min-height: 56px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 24px; border-bottom: 1px solid #d9dfd7; background: #ffffff; }
    .primary-tabs { min-width: 0; display: flex; align-items: center; justify-content: flex-start; gap: 4px; overflow-x: auto; }
    .nav-tab { display: inline-flex; align-items: center; justify-content: center; min-height: 34px; padding: 0 12px; border-radius: 6px; color: #47524c; font-size: 13px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    .nav-tab.is-active { background: #196f5a; color: #ffffff; }
    .header-right { margin-left: auto; display: flex; align-items: center; justify-content: flex-end; gap: 12px; min-width: 0; }
    .eval-link { color: #196f5a; font-size: 13px; font-weight: 900; text-decoration: none; white-space: nowrap; }
    main { width: min(1220px, 100%); margin: 0 auto; padding: 20px 24px 28px; display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 20px; align-items: start; }
    .panel { min-width: 0; border: 1px solid #d9dfd7; border-radius: 8px; background: #ffffff; overflow: hidden; }
    .controls { padding: 16px; display: grid; gap: 12px; }
    label { display: grid; gap: 8px; color: #47524c; font-size: 13px; font-weight: 800; }
    input, select, button { min-height: 40px; border-radius: 6px; border: 1px solid #b8c1b7; background: #ffffff; color: #1c2320; font: inherit; }
    input[type="file"] { padding: 8px; }
    .checkbox-field {
      min-height: 40px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid #cfd7cc;
      border-radius: 6px;
      background: #fbfcfa;
      color: #47524c;
      font-size: 13px;
      font-weight: 800;
    }
    .checkbox-field input {
      width: 16px;
      min-width: 16px;
      height: 16px;
      min-height: 16px;
      margin: 0;
    }
    button { border: 0; background: #196f5a; color: #ffffff; font-weight: 800; cursor: pointer; }
    button.secondary { border: 1px solid #b8c1b7; background: #f7faf6; color: #196f5a; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .dropzone { min-height: 112px; display: grid; place-items: center; padding: 14px; border: 1px dashed #9fb0a7; border-radius: 8px; background: #f8fbf7; color: #5d6862; text-align: center; font-size: 13px; font-weight: 700; }
    .dropzone.is-over { border-color: #196f5a; background: #eaf4ef; }
    .toolbar { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .status { min-width: 0; color: #5d6862; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .queue { display: grid; grid-template-rows: 48px minmax(420px, calc(100vh - 152px)); }
    .bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 16px; border-bottom: 1px solid #d9dfd7; }
    .list { min-height: 0; overflow: auto; }
    .row { min-width: 720px; display: grid; grid-template-columns: 64px minmax(160px, 1fr) 116px 150px 104px; gap: 10px; align-items: center; padding: 8px 12px; border-bottom: 1px solid #edf1eb; }
    .row:last-child { border-bottom: 0; }
    .thumb { width: 56px; height: 56px; display: grid; place-items: center; overflow: hidden; border-radius: 4px; border: 0; padding: 0; cursor: zoom-in; background-color: #e9eee6; background-image: linear-gradient(45deg, #d3dbd0 25%, transparent 25%), linear-gradient(-45deg, #d3dbd0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d3dbd0 75%), linear-gradient(-45deg, transparent 75%, #d3dbd0 75%); background-size: 12px 12px; background-position: 0 0, 0 6px, 6px -6px, -6px 0; }
    .thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .thumb:focus-visible { outline: 3px solid rgba(25, 111, 90, 0.28); outline-offset: 2px; }
    .info { min-width: 0; display: grid; gap: 3px; }
    .name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; font-weight: 800; }
    .meta { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #5d6862; font-size: 12px; }
    .pill { width: max-content; max-width: 110px; padding: 4px 8px; border-radius: 999px; background: #edf1eb; color: #47524c; font-size: 12px; font-weight: 800; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pill.done { background: #dff2e7; color: #196f5a; }
    .pill.failed { background: #ffe8e1; color: #a33a22; }
    .actions { display: flex; gap: 6px; justify-content: flex-end; }
    .actions a, .actions button { min-width: 44px; min-height: 32px; display: inline-flex; align-items: center; justify-content: center; padding: 0 9px; border-radius: 6px; font-size: 12px; }
    .empty { padding: 18px; color: #6a746f; font-size: 14px; }
    .lightbox { position: fixed; inset: 0; z-index: 20; display: none; grid-template-rows: 52px minmax(0, 1fr); background: rgba(17, 21, 20, 0.86); }
    .lightbox.is-open { display: grid; }
    .lightbox-bar { min-width: 0; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 18px; color: #ffffff; }
    .lightbox-title { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; font-weight: 800; }
    .lightbox-close { width: 40px; min-width: 40px; min-height: 36px; border: 1px solid rgba(255,255,255,0.28); background: rgba(255,255,255,0.12); color: #ffffff; }
    .lightbox-stage { min-width: 0; min-height: 0; display: grid; place-items: center; padding: 16px; background-color: #e9eee6; background-image: linear-gradient(45deg, #d3dbd0 25%, transparent 25%), linear-gradient(-45deg, #d3dbd0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d3dbd0 75%), linear-gradient(-45deg, transparent 75%, #d3dbd0 75%); background-size: 24px 24px; background-position: 0 0, 0 12px, 12px -12px, -12px 0; }
    .lightbox-stage img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
    @media (max-width: 860px) { .app-header { padding: 0 16px; } main { grid-template-columns: 1fr; padding: 16px; } .queue { grid-template-rows: 48px 520px; } }
  </style>
</head>
<body>
  <header class="app-header">
    <nav class="primary-tabs" aria-label="主导航">
      <a class="nav-tab" href="/slice">切图</a>
      <a class="nav-tab" href="/">抠图</a>
      <a class="nav-tab is-active" href="/batch" aria-current="page">批量抠图</a>
    </nav>
    <div class="header-right">
      <a class="eval-link" href="/eval/game" target="_blank" rel="noreferrer">Game Eval</a>
    </div>
  </header>
  <main>
    <section class="panel controls">
      <label>上传图片<input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" multiple></label>
      <div class="dropzone" id="dropzone">拖拽多张图片到这里，或使用上方文件选择</div>
      <label>后端<select id="backend">__BACKEND_OPTIONS__</select></label>
      __DIRECT_WORKER_ENDPOINT_FIELD__
      <label class="checkbox-field background-repair-field"><input id="background-repair" name="background_repair" type="checkbox" checked><span>背景修复</span></label>
      <div class="toolbar">
        <button id="start" type="button" disabled>开始全部</button>
        <button id="retry" class="secondary" type="button" disabled>重试失败</button>
        <button id="zip" type="button" disabled>下载 ZIP</button>
        <button id="clear" class="secondary" type="button" disabled>清空队列</button>
      </div>
      <span class="status" id="status">等待添加图片</span>
    </section>
    <section class="panel queue">
      <div class="bar"><strong>队列</strong><span class="status" id="count">0 项</span></div>
      <div class="list" id="list"><div class="empty">批量队列会显示在这里</div></div>
    </section>
  </main>
  <div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-labelledby="lightbox-title" hidden>
    <div class="lightbox-bar">
      <span class="lightbox-title" id="lightbox-title"></span>
      <button class="lightbox-close" id="lightbox-close" type="button" aria-label="关闭">×</button>
    </div>
    <div class="lightbox-stage" id="lightbox-stage"></div>
  </div>
  <script>
    const file = document.getElementById("file");
    const dropzone = document.getElementById("dropzone");
    const backend = document.getElementById("backend");
    const directEndpoint = document.getElementById("direct-endpoint");
    const backgroundRepair = document.getElementById("background-repair");
    const startButton = document.getElementById("start");
    const retryButton = document.getElementById("retry");
    const zipButton = document.getElementById("zip");
    const clearButton = document.getElementById("clear");
    const statusEl = document.getElementById("status");
    const countEl = document.getElementById("count");
    const list = document.getElementById("list");
    const lightbox = document.getElementById("lightbox");
    const lightboxTitle = document.getElementById("lightbox-title");
    const lightboxStage = document.getElementById("lightbox-stage");
    const lightboxClose = document.getElementById("lightbox-close");
    let queue = [];
    let running = false;
    let nextId = 1;
    const BATCH_QUEUE_STORAGE_KEY = "ermbgBatchQueue";
    const BATCH_QUEUE_DB_NAME = "ermbgBatchQueueDb";
    const BATCH_QUEUE_STORE_NAME = "queues";

    function dataUrlToFile(dataUrl, filename) {
      const [header, base64] = dataUrl.split(",");
      const mime = (header.match(/data:(.*);base64/) || [])[1] || "image/png";
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      return new File([bytes], filename, { type: mime });
    }

    function openBatchQueueDb() {
      return new Promise((resolve, reject) => {
        if (!window.indexedDB) {
          reject(new Error("IndexedDB unavailable"));
          return;
        }
        const request = indexedDB.open(BATCH_QUEUE_DB_NAME, 1);
        request.onupgradeneeded = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains(BATCH_QUEUE_STORE_NAME)) db.createObjectStore(BATCH_QUEUE_STORE_NAME);
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error("IndexedDB open failed"));
      });
    }

    function getBatchQueueDb(db) {
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(BATCH_QUEUE_STORE_NAME, "readwrite");
        const store = transaction.objectStore(BATCH_QUEUE_STORE_NAME);
        const request = store.get(BATCH_QUEUE_STORAGE_KEY);
        request.onsuccess = () => {
          const payload = request.result || null;
          if (payload) store.delete(BATCH_QUEUE_STORAGE_KEY);
          resolve(payload);
        };
        request.onerror = () => reject(request.error || new Error("IndexedDB read failed"));
      });
    }

    async function loadBatchQueuePayload() {
      if (window.indexedDB) {
        try {
          const db = await openBatchQueueDb();
          const payload = await getBatchQueueDb(db);
          db.close();
          if (payload) {
            sessionStorage.removeItem(BATCH_QUEUE_STORAGE_KEY);
            return payload;
          }
        } catch (error) {
          console.warn("IndexedDB batch queue read failed; falling back to sessionStorage", error);
        }
      }
      const raw = sessionStorage.getItem(BATCH_QUEUE_STORAGE_KEY);
      if (!raw) return null;
      sessionStorage.removeItem(BATCH_QUEUE_STORAGE_KEY);
      return JSON.parse(raw);
    }

    function addFiles(files, source = "upload") {
      Array.from(files || []).forEach((item) => {
        const fileObj = item instanceof File ? item : dataUrlToFile(item.rgb, item.filename || "slice.png");
        const previewUrl = item.rgb || URL.createObjectURL(fileObj);
        queue.push({
          id: `item-${nextId++}`,
          source,
          file: fileObj,
          name: item.name || fileObj.name,
          filename: fileObj.name,
          previewUrl,
          meta: item.meta || source,
          status: "queued",
          message: "等待",
          result: null,
        });
      });
      render();
    }

    function selectedCandidate(payload) {
      const candidates = payload.candidates || [];
      return candidates.find((candidate) => candidate.selected) || candidates[0] || null;
    }

    function effectiveBackendValue() {
      const endpoint = directEndpoint ? directEndpoint.value : "";
      const baseBackend = backend.value.split(":")[0];
      if (!endpoint) return backend.value || "auto";
      if (baseBackend === "auto" || baseBackend === "direct-worker") return `direct-worker:${endpoint}`;
      if (baseBackend === "corridorkey" || baseBackend === "direct-corridorkey") return `direct-corridorkey:${endpoint}`;
      if (baseBackend === "pymatting_known_b" || baseBackend === "pymatting-known-b" || baseBackend === "direct-pymatting-known-b") return `direct-pymatting-known-b:${endpoint}`;
      if (baseBackend === "known-bg-glow" || baseBackend === "known_bg_glow" || baseBackend === "direct-known-bg-glow") return `direct-known-bg-glow:${endpoint}`;
      return backend.value || "auto";
    }

    function defaultSemanticCandidate(payload) {
      const candidates = payload.candidates || [];
      return candidates.find((candidate) => candidate.id === payload.default_candidate_id) || candidates[0] || { id: "auto_default", decision: { policy: "auto_default" } };
    }

    function statusLabel(item) {
      if (item.status === "running") return "处理中";
      if (item.status === "done") return "成功";
      if (item.status === "failed") return "失败";
      return "等待";
    }

    function itemPreviewUrl(item) {
      return item.result && item.result.rgba ? item.result.rgba : item.previewUrl;
    }

    function executionAnalysisPayload(payload, selectedCandidate) {
      if (!payload || typeof payload !== "object") return {};
      const slim = { ...payload };
      const selectedId = selectedCandidate && selectedCandidate.id ? String(selectedCandidate.id) : String(payload.default_candidate_id || "auto_default");
      const selectedRefs = selectedCandidate && selectedCandidate.preview && selectedCandidate.preview.assets ? selectedCandidate.preview.assets : {};
      const selectedTrimapRef = selectedRefs && selectedRefs.trimap ? String(selectedRefs.trimap) : "";
      const selectedTrimapAsset = selectedTrimapRef && payload.preview_assets ? payload.preview_assets[selectedTrimapRef] : null;
      slim.preview_assets = selectedTrimapAsset ? { [selectedTrimapRef]: selectedTrimapAsset } : {};
      if (Array.isArray(slim.candidates)) {
        slim.candidates = slim.candidates.map((candidate) => {
          if (!candidate || typeof candidate !== "object") return candidate;
          const copy = { ...candidate };
          if (String(copy.id || "") === selectedId && selectedTrimapRef) {
            copy.preview = { assets: { trimap: selectedTrimapRef } };
          } else {
            delete copy.preview;
          }
          return copy;
        });
      }
      return slim;
    }

    function openLightbox(item) {
      const img = document.createElement("img");
      img.src = itemPreviewUrl(item);
      img.alt = item.name;
      lightboxTitle.textContent = item.result ? `${item.name} · 结果` : `${item.name} · 原图`;
      lightboxStage.innerHTML = "";
      lightboxStage.appendChild(img);
      lightbox.hidden = false;
      lightbox.classList.add("is-open");
      lightboxClose.focus();
    }

    function closeLightbox() {
      lightbox.classList.remove("is-open");
      lightbox.hidden = true;
      lightboxStage.innerHTML = "";
    }

    function render() {
      const done = queue.filter((item) => item.status === "done").length;
      const failed = queue.filter((item) => item.status === "failed").length;
      countEl.textContent = `${queue.length} 项 · ${done} 成功 · ${failed} 失败`;
      startButton.disabled = running || !queue.some((item) => item.status === "queued");
      retryButton.disabled = running || !queue.some((item) => item.status === "failed");
      zipButton.disabled = running || !done;
      clearButton.disabled = running || !queue.length;
      backend.disabled = running;
      if (directEndpoint) directEndpoint.disabled = running;
      backgroundRepair.disabled = running;
      if (!queue.length) {
        list.innerHTML = '<div class="empty">批量队列会显示在这里</div>';
        statusEl.textContent = "等待添加图片";
        return;
      }
      list.innerHTML = "";
      queue.forEach((item) => {
        const row = document.createElement("div");
        row.className = "row";
        const thumb = document.createElement("button");
        thumb.className = "thumb";
        thumb.type = "button";
        thumb.title = "查看大图";
        thumb.setAttribute("aria-label", `查看 ${item.name} 大图`);
        const img = document.createElement("img");
        img.src = itemPreviewUrl(item);
        img.alt = item.name;
        thumb.appendChild(img);
        thumb.addEventListener("click", () => openLightbox(item));
        const info = document.createElement("span");
        info.className = "info";
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = item.name;
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = item.message || item.meta || "";
        info.appendChild(name);
        info.appendChild(meta);
        const state = document.createElement("span");
        state.className = `pill ${item.status}`;
        state.textContent = statusLabel(item);
        const backendText = document.createElement("span");
        backendText.className = "meta";
        backendText.textContent = item.result ? [item.result.algorithm, item.result.execution_backend, item.result.execution_profile].filter(Boolean).join(" · ") : item.source;
        const actions = document.createElement("span");
        actions.className = "actions";
        if (item.status === "done" && item.result) {
          const download = document.createElement("a");
          download.href = item.result.rgba;
          download.download = item.result.filename || `${item.name.replace(/\\.[^.]+$/, "")}_rgba.png`;
          download.textContent = "下载";
          actions.appendChild(download);
        }
        if (item.status === "failed") {
          const retry = document.createElement("button");
          retry.type = "button";
          retry.className = "secondary";
          retry.textContent = "重试";
          retry.addEventListener("click", () => { item.status = "queued"; item.message = "等待重试"; render(); processQueue(); });
          actions.appendChild(retry);
        }
        row.appendChild(thumb);
        row.appendChild(info);
        row.appendChild(state);
        row.appendChild(backendText);
        row.appendChild(actions);
        list.appendChild(row);
      });
    }

    async function processItem(item) {
      item.status = "running";
      item.message = "正在分析";
      render();
      const startedAt = performance.now();
      const analyzeFormData = new FormData();
      analyzeFormData.append("file", item.file);
      analyzeFormData.append("background_repair", backgroundRepair.checked ? "true" : "false");
      const analyzeResponse = await fetch("/api/analyze-candidates", { method: "POST", body: analyzeFormData });
      if (!analyzeResponse.ok) {
        let message = "处理失败";
        try { message = (await analyzeResponse.json()).detail || message; } catch (_) {}
        throw new Error(message);
      }
      const analysisPayload = await analyzeResponse.json();
      const semanticCandidate = defaultSemanticCandidate(analysisPayload);
      item.message = `正在执行 ${semanticCandidate.label || semanticCandidate.id || "默认决策"}`;
      render();
      const executeFormData = new FormData();
      executeFormData.append("file", item.file);
      executeFormData.append("backend", effectiveBackendValue());
      executeFormData.append("background_repair", backgroundRepair.checked ? "true" : "false");
      executeFormData.append("shadow_mode", "auto");
      executeFormData.append("parameter_source", "auto");
      executeFormData.append("selected_candidate_id", semanticCandidate.id || analysisPayload.default_candidate_id || "auto_default");
      executeFormData.append("semantic_decision", JSON.stringify(semanticCandidate.decision || {}));
      executeFormData.append("analysis_payload", JSON.stringify(executionAnalysisPayload(analysisPayload, semanticCandidate)));
      const response = await fetch("/api/execute-candidate", { method: "POST", body: executeFormData });
      if (!response.ok) {
        let message = "处理失败";
        try { message = (await response.json()).detail || message; } catch (_) {}
        throw new Error(message);
      }
      const payload = await response.json();
      const candidate = selectedCandidate(payload);
      if (!candidate || !candidate.rgba) throw new Error("没有可下载的 RGBA 结果");
      item.status = "done";
      item.message = `${((performance.now() - startedAt) / 1000).toFixed(2)}s`;
      item.result = {
        source: item.source,
        filename: item.filename,
        name: item.name,
        rgba: candidate.rgba,
        requested_backend: payload.requested_backend || effectiveBackendValue(),
        algorithm: payload.algorithm,
        route: payload.route,
        parameter_profile: payload.parameter_profile,
        execution_profile: payload.execution_profile,
        execution_backend: payload.execution_backend,
        execution_url: payload.execution_url,
        server_elapsed_sec: payload.server_elapsed_sec,
        analysis_status: payload.analysis_status,
        selected_candidate_id: payload.selected_candidate_id,
      };
    }

    async function processQueue() {
      if (running) return;
      running = true;
      render();
      try {
        for (const item of queue) {
          if (item.status !== "queued") continue;
          try {
            await processItem(item);
          } catch (error) {
            item.status = "failed";
            item.message = error.message;
          }
          render();
        }
      } finally {
        running = false;
        const done = queue.filter((item) => item.status === "done").length;
        const failed = queue.filter((item) => item.status === "failed").length;
        statusEl.textContent = `完成 ${done} 项${failed ? ` · ${failed} 失败` : ""}`;
        render();
      }
    }

    async function downloadZip() {
      const items = queue.filter((item) => item.status === "done" && item.result).map((item) => item.result);
      if (!items.length) return;
      zipButton.disabled = true;
      statusEl.textContent = "正在打包 ZIP";
      try {
        const response = await fetch("/api/batch-results.zip", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ items }),
        });
        if (!response.ok) throw new Error((await response.json()).detail || "打包失败");
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "ermbg-batch-results.zip";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
        statusEl.textContent = `已打包 ${items.length} 项`;
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        render();
      }
    }

    file.addEventListener("change", () => addFiles(file.files, "upload"));
    startButton.addEventListener("click", processQueue);
    retryButton.addEventListener("click", () => { queue.forEach((item) => { if (item.status === "failed") { item.status = "queued"; item.message = "等待重试"; } }); processQueue(); });
    zipButton.addEventListener("click", downloadZip);
    clearButton.addEventListener("click", () => { queue = []; render(); });
    lightboxClose.addEventListener("click", closeLightbox);
    lightbox.addEventListener("click", (event) => { if (event.target === lightbox) closeLightbox(); });
    window.addEventListener("keydown", (event) => { if (event.key === "Escape" && !lightbox.hidden) closeLightbox(); });
    ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("is-over"); }));
    dropzone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files, "upload"));

    async function restorePendingBatchQueue() {
      try {
        const pending = await loadBatchQueuePayload();
        if (pending && Array.isArray(pending.items)) {
          addFiles(pending.items, pending.source || "slicer");
          statusEl.textContent = `已接收来自切图的 ${pending.items.length} 项`;
        } else {
          render();
        }
      } catch (error) {
        statusEl.textContent = `无法读取批量队列: ${error.message}`;
        render();
      }
    }
    restorePendingBatchQueue();
  </script>
</body>
</html>""")


def _slice_page_html() -> str:
    return _inject_backend_options("""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG Slice</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1c2320; background: #f5f7f4; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    .app-header { min-height: 56px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 24px; border-bottom: 1px solid #d9dfd7; background: #ffffff; }
    .primary-tabs { min-width: 0; display: flex; align-items: center; justify-content: flex-start; gap: 4px; overflow-x: auto; }
    .nav-tab { display: inline-flex; align-items: center; justify-content: center; min-height: 34px; padding: 0 12px; border-radius: 6px; color: #47524c; font-size: 13px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    .nav-tab.is-active { background: #196f5a; color: #ffffff; }
    .header-right { margin-left: auto; display: flex; align-items: center; justify-content: flex-end; gap: 12px; min-width: 0; }
    .eval-link { color: #196f5a; font-size: 13px; font-weight: 900; text-decoration: none; white-space: nowrap; }
    main { width: min(1120px, 100%); margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 320px 1fr; gap: 24px; align-items: start; }
    form, .workspace { background: #ffffff; border: 1px solid #d9dfd7; border-radius: 8px; }
    form { min-width: 0; min-height: 640px; max-height: 640px; padding: 16px; display: grid; grid-template-rows: auto auto auto minmax(0, 1fr) auto; gap: 12px; overflow: hidden; }
    label { display: grid; gap: 8px; font-size: 13px; font-weight: 700; color: #47524c; }
    input, button { width: 100%; min-height: 40px; border-radius: 6px; border: 1px solid #b8c1b7; background: #ffffff; color: #1c2320; font: inherit; }
    input[type="file"] { padding: 8px; }
    .checkbox-field { min-height: 40px; display: flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid #cfd7cc; border-radius: 6px; background: #fbfcfa; color: #47524c; font-size: 13px; font-weight: 800; }
    .checkbox-field input { width: 16px; min-width: 16px; height: 16px; min-height: 16px; margin: 0; }
    button { border: 0; background: #196f5a; color: #ffffff; font-weight: 800; cursor: pointer; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .settings { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .background-repair-field { grid-column: 1; }
    .slice-actions-main { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; align-items: start; }
    .slice-actions-main button { height: 40px; min-height: 40px; align-self: start; }
    .slice-actions-main .wide-action { grid-column: 1 / -1; }
    .preview, .thumb { background-color: #e9eee6; background-image: linear-gradient(45deg, #d3dbd0 25%, transparent 25%), linear-gradient(-45deg, #d3dbd0 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d3dbd0 75%), linear-gradient(-45deg, transparent 75%, #d3dbd0 75%); background-size: 24px 24px; background-position: 0 0, 0 12px, 12px -12px, -12px 0; }
    .workspace { min-height: 640px; display: grid; grid-template-rows: 48px 1fr; overflow: hidden; }
    .bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 16px; border-bottom: 1px solid #d9dfd7; }
    .status { color: #5d6862; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .preview { min-height: 420px; display: grid; place-items: center; padding: 16px; overflow: hidden; touch-action: none; cursor: grab; }
    .preview.is-dragging { cursor: grabbing; }
    .preview img { max-width: 100%; max-height: 72vh; object-fit: contain; transform-origin: center center; user-select: none; pointer-events: none; will-change: transform; }
    .empty { color: #6a746f; font-size: 14px; }
    .left-list { min-width: 0; min-height: 0; height: 100%; max-height: 100%; display: block; overflow-y: auto; overflow-x: hidden; border: 1px solid #cfd7cc; border-radius: 6px; background: #ffffff; scrollbar-gutter: stable; }
    .row { width: 100%; min-width: 0; height: 72px; display: grid; grid-template-columns: 64px minmax(0, 1fr) 96px; gap: 8px; align-items: center; padding: 4px 6px; border: 0; border-bottom: 1px solid #d9dfd7; border-radius: 0; background: #ffffff; text-align: left; cursor: pointer; }
    .row:last-child { border-bottom: 0; }
    .row:hover { background: #f3f7f1; }
    .row[aria-selected="true"] { background: #d7eadf; }
    .thumb { width: 64px; height: 64px; display: grid; place-items: center; overflow: hidden; border-radius: 4px; background-size: 12px 12px; background-position: 0 0, 0 6px, 6px -6px, -6px 0; }
    .thumb img { display: block; width: 100%; height: 100%; max-width: 100%; max-height: 100%; object-fit: contain; object-position: center; }
    .info { min-width: 0; display: grid; gap: 2px; align-content: center; overflow: hidden; }
    .name { min-width: 0; font-size: 12px; line-height: 1.25; font-weight: 800; color: #1c2320; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .meta { min-width: 0; font-size: 11px; line-height: 1.25; font-weight: 600; color: #5d6862; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .row-actions { width: 96px; min-width: 96px; display: grid; grid-template-columns: 1fr 1fr; gap: 4px; align-items: center; }
    .row-action { width: 100%; min-width: 0; height: 30px; min-height: 30px; display: inline-flex; align-items: center; justify-content: center; padding: 0; border-radius: 6px; font-size: 12px; line-height: 1; visibility: hidden; }
    .row-download { background: #edf4ef; color: #196f5a; text-decoration: none; font-weight: 800; }
    .row:hover .row-action, .row:focus-within .row-action { visibility: visible; }
    .selected-actions { display: none; gap: 8px; }
    .selected-actions.is-visible { display: grid; }
    @media (max-width: 760px) { .app-header { padding: 0 16px; } main { grid-template-columns: 1fr; padding: 16px; } form { min-height: 520px; } .workspace { min-height: 520px; } .preview { min-height: 320px; } }
  </style>
</head>
<body>
  <header class="app-header">
    <nav class="primary-tabs" aria-label="主导航">
      <a class="nav-tab is-active" href="/slice" aria-current="page">切图</a>
      <a class="nav-tab" href="/">抠图</a>
      <a class="nav-tab" href="/batch">批量抠图</a>
    </nav>
    <div class="header-right">
      <a class="eval-link" href="/eval/game" target="_blank" rel="noreferrer">Game Eval</a>
    </div>
  </header>
  <main>
    <form id="slice-form">
      <label>图片<input id="file" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" required></label>
      <div class="settings">
        <label>最小面积<input id="min-area" type="number" min="1" step="1" value="64"></label>
        <label>边距<input id="padding" type="number" min="0" step="1" value="4"></label>
        <label class="checkbox-field background-repair-field"><input id="background-repair" name="background_repair" type="checkbox" checked><span>背景修复</span></label>
      </div>
      <div class="slice-actions-main">
        <button id="preview-button" type="button" disabled>预览</button>
        <button id="confirm" type="button" disabled>切图</button>
        <button id="download-slices" class="wide-action" type="button" disabled>批量下载切图</button>
      </div>
      <div class="left-list" id="list"><span class="empty">切图列表会显示在这里</span></div>
      <div class="selected-actions" id="selected-actions">
        <button id="batch-all" type="button">批量抠图</button>
      </div>
    </form>
    <section class="workspace" aria-label="slice workspace">
      <div class="bar"><strong>切图预览</strong><span class="status" id="status">等待上传</span></div>
      <div class="preview" id="preview"><span class="empty">自动标注会显示在这里</span></div>
    </section>
  </main>
  <script>
    const form = document.getElementById("slice-form");
    const file = document.getElementById("file");
    const minArea = document.getElementById("min-area");
    const padding = document.getElementById("padding");
    const backgroundRepair = document.getElementById("background-repair");
    const previewButton = document.getElementById("preview-button");
    const confirmButton = document.getElementById("confirm");
    const downloadSlices = document.getElementById("download-slices");
    const batchAll = document.getElementById("batch-all");
    const selectedActions = document.getElementById("selected-actions");
    const statusEl = document.getElementById("status");
    const preview = document.getElementById("preview");
    const list = document.getElementById("list");
    let hasPreview = false;
    let currentCrops = [];
    let selectedCrop = null;
    let previewRequestId = 0;
    let lastPreviewSettingsKey = "";
    let uploadedPreviewUrl = null;
    const SLICE_STATE_KEY = "ermbgSliceWorkspace";
    const BATCH_QUEUE_STORAGE_KEY = "ermbgBatchQueue";
    const BATCH_QUEUE_DB_NAME = "ermbgBatchQueueDb";
    const BATCH_QUEUE_STORE_NAME = "queues";

    function setSliceBusy(isBusy) {
      previewButton.disabled = isBusy || !file.files.length;
      confirmButton.disabled = isBusy || !hasPreview;
      downloadSlices.disabled = isBusy || !file.files.length;
      file.disabled = isBusy;
      backgroundRepair.disabled = isBusy;
    }

    function setTransferBusy(isBusy) {
      batchAll.disabled = isBusy || !currentCrops.length;
      Array.from(list.querySelectorAll(".row-action")).forEach((control) => {
        control.disabled = isBusy;
        control.setAttribute("aria-disabled", String(isBusy));
      });
    }

    function formData() {
      const data = new FormData();
      data.append("file", file.files[0]);
      data.append("min_area", minArea.value || "64");
      data.append("padding", padding.value || "4");
      data.append("background_repair", backgroundRepair.checked ? "true" : "false");
      return data;
    }

    function currentSettings() {
      return {
        minArea: minArea.value || "64",
        padding: padding.value || "4",
        backgroundRepair: backgroundRepair.checked ? "true" : "false",
      };
    }

    function saveSliceState(patch) {
      let current = {};
      try {
        current = JSON.parse(sessionStorage.getItem(SLICE_STATE_KEY) || "{}");
      } catch (_) {}
      const next = { ...current, ...patch, settings: currentSettings() };
      try {
        sessionStorage.setItem(SLICE_STATE_KEY, JSON.stringify(next));
      } catch (_) {
        const slim = { ...next };
        delete slim.crops;
        if (slim.preview) {
          slim.preview = {
            count: slim.preview.count,
            background_color: slim.preview.background_color,
            overlap_count: slim.preview.overlap_count,
            overlaps: slim.preview.overlaps,
            boxes: slim.preview.boxes,
            raw_boxes: slim.preview.raw_boxes,
          };
        }
        try {
          sessionStorage.setItem(SLICE_STATE_KEY, JSON.stringify(slim));
        } catch (_) {}
      }
    }

    function clearSliceState() {
      sessionStorage.removeItem(SLICE_STATE_KEY);
    }

    function openBatchQueueDb() {
      return new Promise((resolve, reject) => {
        if (!window.indexedDB) {
          reject(new Error("IndexedDB unavailable"));
          return;
        }
        const request = indexedDB.open(BATCH_QUEUE_DB_NAME, 1);
        request.onupgradeneeded = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains(BATCH_QUEUE_STORE_NAME)) db.createObjectStore(BATCH_QUEUE_STORE_NAME);
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error("IndexedDB open failed"));
      });
    }

    function putBatchQueueDb(db, payload) {
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(BATCH_QUEUE_STORE_NAME, "readwrite");
        transaction.objectStore(BATCH_QUEUE_STORE_NAME).put(payload, BATCH_QUEUE_STORAGE_KEY);
        transaction.oncomplete = resolve;
        transaction.onerror = () => reject(transaction.error || new Error("IndexedDB write failed"));
      });
    }

    async function saveBatchQueuePayload(payload) {
      if (window.indexedDB) {
        try {
          const db = await openBatchQueueDb();
          await putBatchQueueDb(db, payload);
          db.close();
          sessionStorage.removeItem(BATCH_QUEUE_STORAGE_KEY);
          return;
        } catch (error) {
          console.warn("IndexedDB batch queue write failed; falling back to sessionStorage", error);
        }
      }
      sessionStorage.setItem(BATCH_QUEUE_STORAGE_KEY, JSON.stringify(payload));
    }

    function createImagePanZoomViewport(viewport, options = {}) {
      const minScale = options.minScale || 0.25;
      const maxScale = options.maxScale || 8;
      let image = null;
      let transform = { scale: 1, x: 0, y: 0 };
      let drag = null;
      function apply() {
        if (!image) return;
        image.style.transform = `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`;
      }
      function reset() {
        transform = { scale: 1, x: 0, y: 0 };
        apply();
      }
      function setImage(src, alt) {
        viewport.innerHTML = "";
        image = document.createElement("img");
        image.src = src;
        image.alt = alt;
        image.draggable = false;
        viewport.appendChild(image);
        reset();
      }
      function clear(html) {
        image = null;
        drag = null;
        viewport.classList.remove("is-dragging");
        viewport.innerHTML = html || "";
      }
      viewport.addEventListener("wheel", (event) => {
        if (!image) return;
        event.preventDefault();
        const rect = viewport.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const pointerX = event.clientX - centerX;
        const pointerY = event.clientY - centerY;
        const previousScale = transform.scale;
        const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
        transform.scale = Math.max(minScale, Math.min(maxScale, transform.scale * factor));
        transform.x = pointerX - ((pointerX - transform.x) * transform.scale) / previousScale;
        transform.y = pointerY - ((pointerY - transform.y) * transform.scale) / previousScale;
        apply();
      }, { passive: false });
      viewport.addEventListener("pointerdown", (event) => {
        if (!image) return;
        event.preventDefault();
        drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, startX: transform.x, startY: transform.y };
        viewport.classList.add("is-dragging");
        viewport.setPointerCapture(event.pointerId);
      });
      viewport.addEventListener("pointermove", (event) => {
        if (!drag || drag.pointerId !== event.pointerId) return;
        transform.x = drag.startX + event.clientX - drag.x;
        transform.y = drag.startY + event.clientY - drag.y;
        apply();
      });
      function endDrag(event) {
        if (!drag || drag.pointerId !== event.pointerId) return;
        drag = null;
        viewport.classList.remove("is-dragging");
        try { viewport.releasePointerCapture(event.pointerId); } catch (_) {}
      }
      viewport.addEventListener("pointerup", endDrag);
      viewport.addEventListener("pointercancel", endDrag);
      viewport.addEventListener("dblclick", reset);
      viewport.addEventListener("dragstart", (event) => event.preventDefault());
      return { setImage, clear, reset, hasImage: () => Boolean(image) };
    }

    const previewViewport = createImagePanZoomViewport(preview);

    function renderPreviewImage(src, alt) {
      previewViewport.setImage(src, alt);
    }

    function showPreview(payload) {
      hasPreview = Boolean(payload.count);
      renderPreviewImage(payload.annotated, "自动标注预览");
      list.innerHTML = '<span class="empty">确认标注后生成切图列表</span>';
      const overlapCount = payload.overlap_count || 0;
      statusEl.textContent = overlapCount
        ? `标注完成 · ${payload.count || 0} 个矩形 · ${overlapCount} 处边距重叠`
        : `标注完成 · ${payload.count || 0} 个矩形`;
      confirmButton.disabled = !hasPreview;
      saveSliceState({ preview: payload, crops: null });
    }

    function sendToMatte(crop) {
      setTransferBusy(true);
      saveSliceState({ selectedCropId: crop.id || crop.filename });
      sessionStorage.setItem("ermbgPendingSlice", JSON.stringify(crop));
      window.location.href = "/";
    }

    async function sendToBatch(crops) {
      const items = (crops || []).filter(Boolean).map((crop) => ({
        source: "slicer",
        filename: crop.filename || `${crop.id || crop.label || "slice"}_rgb.png`,
        name: crop.label || crop.id || crop.filename || "slice",
        rgb: crop.rgb,
        meta: crop.meta || "来自切图",
      }));
      if (!items.length) return;
      setTransferBusy(true);
      saveSliceState({ selectedCropId: selectedCrop ? (selectedCrop.id || selectedCrop.filename) : null });
      try {
        await saveBatchQueuePayload({ source: "slicer", items });
        window.location.href = "/batch";
      } catch (error) {
        statusEl.textContent = `无法保存批量队列: ${error.message}`;
        setTransferBusy(false);
      }
    }

    function selectCrop(crop) {
      selectedCrop = crop;
      selectedActions.classList.add("is-visible");
      Array.from(list.querySelectorAll(".row")).forEach((row) => {
        row.setAttribute("aria-selected", String(row.dataset.cropId === crop.id));
      });
      renderPreviewImage(crop.rgb, `${crop.label} 预览`);
      statusEl.textContent = `已选择 ${crop.label}`;
      saveSliceState({ selectedCropId: crop.id || crop.filename });
    }

    function renderCrops(payload) {
      list.innerHTML = "";
      currentCrops = payload.crops || [];
      selectedCrop = null;
      selectedActions.classList.remove("is-visible");
      if (!payload.crops || !payload.crops.length) {
        list.innerHTML = '<span class="empty">没有检测到可切割主体</span>';
        return;
      }
      payload.crops.forEach((crop) => {
        const row = document.createElement("div");
        row.className = "row";
        row.dataset.cropId = crop.id;
        row.setAttribute("aria-selected", "false");
        const thumb = document.createElement("span");
        thumb.className = "thumb";
        const img = document.createElement("img");
        img.src = crop.rgb;
        img.alt = `${crop.label} 预览`;
        thumb.appendChild(img);
        const info = document.createElement("span");
        info.className = "info";
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = crop.label;
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = crop.meta || "";
        info.appendChild(name);
        info.appendChild(meta);
        const actions = document.createElement("span");
        actions.className = "row-actions";
        const downloadCrop = document.createElement("a");
        downloadCrop.className = "row-action row-download";
        downloadCrop.href = crop.rgb;
        downloadCrop.download = crop.filename || `${crop.id || crop.label || "slice"}_rgb.png`;
        downloadCrop.textContent = "下载";
        downloadCrop.addEventListener("click", (event) => {
          event.stopPropagation();
        });
        const action = document.createElement("button");
        action.className = "row-action";
        action.type = "button";
        action.textContent = "抠图";
        action.addEventListener("click", (event) => {
          event.stopPropagation();
          sendToMatte(crop);
        });
        actions.appendChild(downloadCrop);
        actions.appendChild(action);
        row.appendChild(thumb);
        row.appendChild(info);
        row.appendChild(actions);
        row.addEventListener("click", () => selectCrop(crop));
        list.appendChild(row);
      });
      statusEl.textContent = `切图完成 · ${payload.count || payload.crops.length} 张`;
      confirmButton.disabled = true;
      selectedActions.classList.add("is-visible");
      batchAll.disabled = !currentCrops.length;
      saveSliceState({ crops: payload });
      if (payload.crops.length === 1) {
        selectCrop(payload.crops[0]);
      }
    }

    file.addEventListener("change", () => {
      if (!file.files.length) return;
      hasPreview = false;
      currentCrops = [];
      selectedCrop = null;
      selectedActions.classList.remove("is-visible");
      clearSliceState();
      previewButton.disabled = false;
      confirmButton.disabled = true;
      downloadSlices.disabled = false;
      if (uploadedPreviewUrl) URL.revokeObjectURL(uploadedPreviewUrl);
      uploadedPreviewUrl = URL.createObjectURL(file.files[0]);
      renderPreviewImage(uploadedPreviewUrl, "上传图片预览");
      list.innerHTML = '<span class="empty">切图列表会显示在这里</span>';
      backgroundRepair.checked = true;
      setSliceBusy(false);
      statusEl.textContent = "已载入图片，正在自动标注";
      runAnnotate();
    });

    function invalidatePreview() {
      if (!file.files.length) return;
      hasPreview = false;
      currentCrops = [];
      selectedCrop = null;
      selectedActions.classList.remove("is-visible");
      confirmButton.disabled = true;
      batchAll.disabled = true;
      lastPreviewSettingsKey = "";
      list.innerHTML = '<span class="empty">参数已修改，正在重新标注</span>';
      statusEl.textContent = "参数已修改，正在自动标注";
      runAnnotate();
    }

    minArea.addEventListener("change", invalidatePreview);
    padding.addEventListener("change", invalidatePreview);
    backgroundRepair.addEventListener("change", invalidatePreview);

    async function runAnnotate() {
      if (!file.files.length) return;
      const settingsKey = JSON.stringify(currentSettings());
      if (settingsKey === lastPreviewSettingsKey && hasPreview) return;
      const requestId = ++previewRequestId;
      setSliceBusy(true);
      statusEl.textContent = "正在自动标注";
      try {
        const response = await fetch("/api/slice-preview", { method: "POST", body: formData() });
        if (!response.ok) throw new Error((await response.json()).detail || "标注失败");
        const payload = await response.json();
        if (requestId === previewRequestId) {
          lastPreviewSettingsKey = settingsKey;
          showPreview(payload);
        }
      } catch (error) {
        if (requestId === previewRequestId) {
          statusEl.textContent = error.message;
        }
      } finally {
        if (requestId === previewRequestId) {
          setSliceBusy(false);
        }
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      runAnnotate();
    });

    previewButton.addEventListener("click", () => {
      runAnnotate();
    });

    downloadSlices.addEventListener("click", async () => {
      if (!file.files.length) return;
      setSliceBusy(true);
      statusEl.textContent = "正在打包切图";
      try {
        const response = await fetch("/api/slice", { method: "POST", body: formData() });
        if (!response.ok) {
          let detail = "下载失败";
          try {
            detail = (await response.json()).detail || detail;
          } catch (_) {}
          throw new Error(detail);
        }
        const blob = await response.blob();
        const stem = file.files[0].name.replace(/\\.[^.]+$/, "") || "ermbg";
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${stem}_slices.zip`;
        document.body.appendChild(link);
        link.click();
        setTimeout(() => {
          link.remove();
          URL.revokeObjectURL(url);
        }, 30000);
        statusEl.textContent = `已下载 ${response.headers.get("X-ERMBG-Slice-Count") || ""} 张切图`.trim();
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setSliceBusy(false);
      }
    });

    confirmButton.addEventListener("click", async () => {
      if (!file.files.length || !hasPreview) return;
      setSliceBusy(true);
      statusEl.textContent = "正在生成切图";
      try {
        const response = await fetch("/api/slice-crops", { method: "POST", body: formData() });
        if (!response.ok) throw new Error((await response.json()).detail || "切图失败");
        renderCrops(await response.json());
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        setSliceBusy(false);
      }
    });

    batchAll.addEventListener("click", () => {
      sendToBatch(currentCrops);
    });

    function restoreSliceState() {
      let state = null;
      try {
        state = JSON.parse(sessionStorage.getItem(SLICE_STATE_KEY) || "null");
      } catch (_) {
        state = null;
      }
      if (!state) return;
      if (state.settings) {
        minArea.value = state.settings.minArea || minArea.value;
        padding.value = state.settings.padding || padding.value;
      }
      if (state.preview) {
        showPreview(state.preview);
      }
      if (state.crops) {
        renderCrops(state.crops);
        const crop = currentCrops.find((item) => item.id === state.selectedCropId);
        if (crop) {
          selectCrop(crop);
        }
        statusEl.textContent = state.selectedCropId ? "已返回切图列表" : "切图已恢复";
      }
    }

    restoreSliceState();
  </script>
</body>
</html>""")


def _png_data_url(rgba: np.ndarray) -> str:
    encoded = base64.b64encode(_encode_png(rgba)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _rgb_png_data_url(rgb: np.ndarray) -> str:
    encoded = base64.b64encode(_encode_rgb_png(rgb)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mask_png_data_url(mask: np.ndarray) -> str:
    arr = np.clip(mask.astype(np.float32), 0.0, 1.0)
    u8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _json_safe_debug(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        summary: dict[str, object] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size:
            summary.update(
                {
                    "min": float(np.min(value)),
                    "max": float(np.max(value)),
                    "mean": float(np.mean(value)),
                }
            )
        return summary
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe_debug(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_debug(v) for v in value]
    return value


def _candidate_payload(candidate: MatteCandidate, stem: str) -> dict[str, object]:
    debug = _json_safe_debug(candidate.debug)
    return {
        "id": candidate.id,
        "label": candidate.label,
        "kind": candidate.kind,
        "filename": f"{stem}_{candidate.id}.png",
        "rgba": _png_data_url(candidate.rgba),
        "selected": candidate.selected,
        "plan": debug.get("plan"),
        "regions": debug.get("regions", []),
        "operation_results": debug.get("operation_results", []),
        "debug": debug,
    }


def _safe_zip_name(value: str, default: str = "item") -> str:
    name = Path(value or default).name.strip() or default
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or default


def _decode_png_data_url(data_url: str) -> np.ndarray:
    if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
        raise HTTPException(status_code=400, detail="rgba must be a PNG data URL")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail="rgba must contain readable PNG data") from e
    return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _decode_png_data_url_gray(data_url: str, *, field: str) -> np.ndarray:
    if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
        raise HTTPException(status_code=400, detail=f"{field} must be a PNG data URL")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        image = Image.open(BytesIO(raw))
        image.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{field} must contain readable PNG data") from e
    return np.asarray(image.convert("L"), dtype=np.uint8)


def _explicit_trimap_from_analysis(
    analysis_payload: dict[str, Any] | None,
    *,
    selected_candidate_id: str,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    if not isinstance(analysis_payload, dict) or _analysis_algorithm(analysis_payload) != "pymatting_known_b":
        return None
    candidates = analysis_payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    asset_ref = ""
    for candidate in candidates:
        if not isinstance(candidate, dict) or str(candidate.get("id") or "") != selected_candidate_id:
            continue
        preview = candidate.get("preview")
        assets = preview.get("assets") if isinstance(preview, dict) else None
        if isinstance(assets, dict):
            asset_ref = str(assets.get("trimap") or "")
        break
    if not asset_ref:
        asset_ref = f"candidate:{selected_candidate_id}:trimap"
    preview_assets = analysis_payload.get("preview_assets")
    asset = preview_assets.get(asset_ref) if isinstance(preview_assets, dict) else None
    if not isinstance(asset, dict):
        return None
    if asset.get("execution_role") not in {None, "pymatting_explicit_trimap"}:
        return None
    data_url = asset.get("data_url")
    if not isinstance(data_url, str):
        return None
    trimap = _decode_png_data_url_gray(data_url, field="candidate trimap")
    if trimap.shape != image_shape:
        raise HTTPException(status_code=400, detail="candidate trimap shape must match image shape")
    out = np.full(trimap.shape, 128, dtype=np.uint8)
    out[trimap < 64] = 0
    out[trimap > 191] = 255
    return out


def _web_batch_root() -> Path:
    return PROJECT_ROOT / "out" / f"web_batch_matte_{datetime.now().strftime('%Y%m%d')}"


def _write_batch_result_artifacts(items: list[dict[str, Any]]) -> tuple[Path, bytes, str]:
    batch_id = datetime.now().strftime("batch_%H%M%S_%f")
    batch_dir = _web_batch_root() / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    case_summaries: list[dict[str, Any]] = []
    used_zip_names: set[str] = set()
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, item in enumerate(items, start=1):
            source_name = str(item.get("filename") or item.get("name") or f"item_{index:03d}.png")
            stem = Path(_safe_zip_name(source_name, f"item_{index:03d}.png")).stem or f"item_{index:03d}"
            case_name = f"{index:03d}_{stem}"
            case_dir = batch_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            rgba = _decode_png_data_url(str(item.get("rgba") or ""))
            rgba_path = case_dir / "rgba.png"
            summary_path = case_dir / "summary.json"
            ermbg_io.save_rgba(rgba_path, rgba)
            summary = {
                "status": "ok",
                "source": item.get("source") or "batch",
                "filename": source_name,
                "fixed_backend": item.get("requested_backend") or "auto",
                "requested_backend": item.get("requested_backend") or "auto",
                "actual_execution_backend": item.get("execution_backend"),
                "algorithm": item.get("algorithm"),
                "route": item.get("route"),
                "execution_profile": item.get("execution_profile"),
                "execution_server_url": item.get("execution_url") or item.get("execution_server_url"),
                "server_elapsed_sec": item.get("server_elapsed_sec"),
                "outputs": {"rgba": "rgba.png"},
            }
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            manifest = build_run_manifest(
                run_dir=case_dir,
                outputs={"rgba": rgba_path},
                request={
                    "backend": item.get("requested_backend") or "auto",
                    "filename": source_name,
                    "source": item.get("source") or "batch",
                },
                route={
                    "algorithm": item.get("algorithm"),
                    "route": item.get("route"),
                    "parameter_profile": item.get("parameter_profile"),
                    "execution_profile": item.get("execution_profile"),
                },
                runtime={
                    "requested_backend": item.get("requested_backend") or "auto",
                    "backend": item.get("execution_backend"),
                    "execution_server_url": item.get("execution_url") or item.get("execution_server_url"),
                    "server_elapsed_sec": item.get("server_elapsed_sec"),
                },
                report_path=summary_path,
                extra={"batch_id": batch_id, "index": index},
            )
            manifest_path = write_run_manifest(case_dir / "manifest.json", manifest)
            case_summary = {
                **summary,
                "case_manifest": str(manifest_path.relative_to(batch_dir)),
                "output_files": {"rgba": str(rgba_path.relative_to(batch_dir))},
            }
            case_summaries.append(case_summary)
            zip_name = f"{stem}_rgba.png"
            if zip_name in used_zip_names:
                zip_name = f"{stem}_{index:03d}_rgba.png"
            used_zip_names.add(zip_name)
            zf.write(rgba_path, zip_name)

        batch_summary = {
            "status": "ok",
            "schema": "ermbg.batch.summary.v1",
            "batch_id": batch_id,
            "fixed_backend": "auto",
            "requested_backend": "auto",
            "count": len(case_summaries),
            "success_count": len(case_summaries),
            "error_count": 0,
            "cases": case_summaries,
        }
        batch_summary_path = batch_dir / "summary.json"
        batch_summary_path.write_text(json.dumps(batch_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        batch_manifest = build_run_manifest(
            run_dir=batch_dir,
            outputs={"summary": batch_summary_path},
            request={"backend": "auto", "source": "web-batch"},
            route={"algorithm": "mixed"},
            runtime={"kind": "web-batch"},
            report_path=batch_summary_path,
            extra={"case_manifests": [case["case_manifest"] for case in case_summaries]},
        )
        write_run_manifest(batch_dir / "manifest.json", batch_manifest)

    return batch_dir, zip_buf.getvalue(), f"{batch_id}.zip"


def _slice_annotated_preview(image_rgb: np.ndarray, boxes: list[SliceBox]) -> np.ndarray:
    preview = Image.fromarray(image_rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for x0, y0, x1, y1 in _slice_box_overlaps(boxes):
        overlay_draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(255, 0, 0, 110), outline=(255, 0, 0, 255), width=2)
    preview = Image.alpha_composite(preview, overlay)
    draw = ImageDraw.Draw(preview)
    for box in boxes:
        x, y, w, h = box.bbox
        color = (255, 160, 0, 255)
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=1)
        label = f"{box.id}"
        text_box = draw.textbbox((x, y), label)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        draw.rectangle((x, y, x + tw + 8, y + th + 6), fill=(25, 111, 90, 235))
        draw.text((x + 4, y + 3), label, fill=(255, 255, 255, 255))
    preview.putalpha(255)
    return np.asarray(preview, dtype=np.uint8)


def _slice_display_boxes(result: SliceResult, image_shape: tuple[int, int]) -> list[SliceBox]:
    return [pad_slice_box(box, image_shape, result.padding) for box in result.boxes]


def _slice_box_overlaps(boxes: list[SliceBox]) -> list[tuple[int, int, int, int]]:
    overlaps: list[tuple[int, int, int, int]] = []
    for i, a in enumerate(boxes):
        ax, ay, aw, ah = a.bbox
        for b in boxes[i + 1 :]:
            bx, by, bw, bh = b.bbox
            x0 = max(ax, bx)
            y0 = max(ay, by)
            x1 = min(ax + aw, bx + bw)
            y1 = min(ay + ah, by + bh)
            if x0 < x1 and y0 < y1:
                overlaps.append((x0, y0, x1, y1))
    return overlaps


def _cached_slice_result(
    image_rgb: np.ndarray,
    image_digest: str,
    *,
    min_area: int,
    padding: int,
) -> SliceResult:
    padding = max(0, int(padding))
    key = (image_digest, int(min_area))
    with _SLICE_CACHE_LOCK:
        cached = _SLICE_CACHE.get(key)
        if cached is not None:
            _SLICE_CACHE.move_to_end(key)
            return SliceResult(
                background_color=cached.background_color,
                foreground_mask=cached.foreground_mask,
                boxes=cached.boxes,
                padding=padding,
            )

    h, w = image_rgb.shape[:2]
    pixels = h * w
    if pixels > _SLICE_WEB_MAX_PIXELS:
        # Web interaction only needs crop rectangles. For very large sheets,
        # full-resolution OKLab masking dominates latency; detect boxes on a
        # bounded preview image and map them back. This protects 4K/6K sheets
        # from multi-second duplicate work while keeping CLI/core slicing exact.
        scale = (_SLICE_WEB_MAX_PIXELS / float(pixels)) ** 0.5
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small = cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_min_area = max(1, int(round(min_area * scale * scale)))
        small_result = slice_image(small, min_area=small_min_area, padding=0)
        boxes: list[SliceBox] = []
        mask = np.zeros((h, w), dtype=np.float32)
        for box in small_result.boxes:
            x, y, bw, bh = box.bbox
            x0 = max(0, int(np.floor(x / scale)))
            y0 = max(0, int(np.floor(y / scale)))
            x1 = min(w, int(np.ceil((x + bw) / scale)))
            y1 = min(h, int(np.ceil((y + bh) / scale)))
            mask[y0:y1, x0:x1] = 1.0
            boxes.append(SliceBox(id=box.id, bbox=(x0, y0, x1 - x0, y1 - y0), area=int(box.area / max(scale * scale, 1e-6))))
        boxes = merge_overlapping_slice_boxes(boxes)
        result = SliceResult(background_color=small_result.background_color, foreground_mask=mask, boxes=boxes, padding=0)
    else:
        result = slice_image(image_rgb, min_area=min_area, padding=0)

    with _SLICE_CACHE_LOCK:
        _SLICE_CACHE[key] = result
        _SLICE_CACHE.move_to_end(key)
        while len(_SLICE_CACHE) > _SLICE_CACHE_MAX:
            _SLICE_CACHE.popitem(last=False)
    return SliceResult(
        background_color=result.background_color,
        foreground_mask=result.foreground_mask,
        boxes=result.boxes,
        padding=padding,
    )


def _slice_result_from_source_alpha(image: Image.Image, *, min_area: int, padding: int) -> tuple[np.ndarray, SliceResult] | None:
    alpha = _source_alpha_mask(image)
    if alpha is None:
        return None
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = alpha > 0
    boxes = find_slice_boxes(mask, min_area=min_area, padding=0, exterior_background_mask=None)
    return image_rgb, SliceResult(
        background_color=(0, 0, 0),
        foreground_mask=(alpha.astype(np.float32) / 255.0),
        boxes=boxes,
        padding=max(0, int(padding)),
    )


def _slice_source_and_result(
    image: Image.Image,
    image_digest: str,
    *,
    min_area: int,
    padding: int,
) -> tuple[np.ndarray, SliceResult]:
    source_alpha_result = _slice_result_from_source_alpha(image, min_area=min_area, padding=padding)
    if source_alpha_result is not None:
        return source_alpha_result
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return image_rgb, _cached_slice_result(image_rgb, image_digest, min_area=min_area, padding=padding)


def _slice_preview_payload(image_rgb: np.ndarray, stem: str, result: SliceResult) -> dict[str, object]:
    display_boxes = _slice_display_boxes(result, image_rgb.shape[:2])
    overlaps = _slice_box_overlaps(display_boxes)
    annotated = _slice_annotated_preview(image_rgb, display_boxes)
    payload = result.to_dict()
    payload.update(
        {
            "stem": stem,
            "annotated": _png_data_url(annotated),
            "boxes": [box.to_dict() for box in display_boxes],
            "raw_boxes": [box.to_dict() for box in result.boxes],
            "overlap_count": len(overlaps),
            "overlaps": [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in overlaps],
        }
    )
    return payload


def _slice_crop_payloads(image_rgb: np.ndarray, stem: str, result: SliceResult) -> dict[str, object]:
    crops = []
    kind_counts: dict[str, int] = {}
    for box in result.boxes:
        padded_box = pad_slice_box(box, image_rgb.shape[:2], result.padding)
        crop = crop_slice(image_rgb, result.foreground_mask, box, padding=result.padding, transparent=False)
        prediction = classify_ui_slice(crop, padded_box, image_rgb.shape[:2], result.foreground_mask)
        kind = prediction.kind if prediction.confidence >= 0.6 else "asset"
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        name = f"{kind}_{kind_counts[kind]:03d}"
        x, y, w, h = padded_box.bbox
        crops.append(
            {
                "id": name,
                "label": name,
                "kind": kind,
                "confidence": prediction.confidence,
                "filename": f"{name}_rgb.png",
                "rgb": _rgb_png_data_url(crop),
                "bbox": [x, y, w, h],
                "meta": f"{kind} {prediction.confidence:.2f} · {w}x{h}",
                "features": prediction.features,
            }
        )
    return {
        "background": list(result.background_color),
        "count": len(crops),
        "crops": crops,
    }


@app.post("/api/sam-mask")
def sam_mask_endpoint(
    file: Annotated[UploadFile, File()],
    threshold: Annotated[float, Form()] = 0.5,
    refine_iterations: Annotated[int, Form()] = 2,
) -> dict[str, object]:
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")
    if not 0 <= refine_iterations <= 5:
        raise HTTPException(status_code=400, detail="refine_iterations must be between 0 and 5")

    image = _load_upload_image(file)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    server_started_at = time.perf_counter()
    try:
        from .probe.comfyui_sam3_mask import ComfyUISAM3MaskClient

        result = ComfyUISAM3MaskClient().mask(
            image_rgb,
            threshold=threshold,
            refine_iterations=refine_iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM3 mask failed: {e}") from e

    return {
        "backend": "comfy-sam3",
        "server_elapsed_sec": time.perf_counter() - server_started_at,
        "mask": _mask_png_data_url(result.mask),
        "debug": _json_safe_debug(result.debug),
    }


def _effective_backend(requested_backend: str, result: MatteResponse) -> str:
    requested_base = _direct_backend_base(requested_backend)
    if isinstance(result.debug, dict):
        direct_worker = result.debug.get("direct_worker")
        if isinstance(direct_worker, dict) and isinstance(direct_worker.get("execution_backend"), str):
            return str(direct_worker.get("execution_backend"))
        if result.debug.get("backend") in {"direct-worker", "direct-corridorkey", "direct-known-bg-glow"}:
            return str(result.debug.get("backend"))
    if requested_base == "auto" and isinstance(result.debug, dict):
        fallback = result.debug.get("web_auto_fallback_backend")
        if isinstance(fallback, str) and fallback:
            return fallback
    auto_route = result.debug.get("auto_route") if isinstance(result.debug, dict) else None
    if isinstance(auto_route, dict) and auto_route.get("requested_backend") in {"direct-worker", "direct-corridorkey", "direct-known-bg-glow"}:
        return str(auto_route.get("requested_backend"))
    return requested_base


def _response_backend(requested_backend: str) -> str:
    return _direct_backend_base(requested_backend)


def _route_metadata(result: MatteResponse) -> dict[str, Any]:
    auto_route = result.debug.get("auto_route") if isinstance(result.debug, dict) else None
    if not isinstance(auto_route, dict):
        return {}
    reasons = auto_route.get("reasons")
    if not isinstance(reasons, list):
        reason = auto_route.get("reason")
        reasons = [reason] if isinstance(reason, str) and reason else []
    execution_server_url = result.debug.get("execution_server_url") if isinstance(result.debug, dict) else None
    return {
        "algorithm": auto_route.get("algorithm") or auto_route.get("route"),
        "route": auto_route.get("route"),
        "asset_kind": auto_route.get("asset_kind"),
        "parameter_profile": auto_route.get("parameter_profile"),
        "execution_profile": auto_route.get("execution_profile"),
        "execution_backend": auto_route.get("execution_backend"),
        "execution_server": result.debug.get("execution_server") if isinstance(result.debug, dict) else None,
        "execution_url": execution_server_url,
        "execution_server_url": execution_server_url,
        "parameter_source": result.debug.get("parameter_source") if isinstance(result.debug, dict) else None,
        "server_fallback_chain": result.debug.get("server_fallback_chain") if isinstance(result.debug, dict) else None,
        "route_confidence": auto_route.get("confidence"),
        "route_reasons": reasons,
    }


def _web_matte_batch_root() -> Path:
    return PROJECT_ROOT / "out" / f"{WEB_MATTE_RUN_PREFIX}{datetime.now().strftime('%Y%m%d')}"


def _write_web_matte_artifacts(
    *,
    image_rgb: np.ndarray,
    selected_rgba: np.ndarray,
    result: MatteResponse,
    filename: str,
    requested_backend: str,
    effective_backend: str,
    shadow_mode: str,
    server_elapsed_sec: float,
) -> Path:
    stem = Path(filename or "ermbg").stem or "ermbg"
    digest = hashlib.sha256(image_rgb.tobytes()).hexdigest()[:10]
    run_id = f"{datetime.now().strftime('%H%M%S_%f')}_{digest}"
    run_dir = _web_matte_batch_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.png"
    output_path = run_dir / "output.png"
    alpha_path = run_dir / "alpha.png"
    foreground_path = run_dir / "foreground.png"
    report_path = run_dir / "summary.json"

    ermbg_io.save_rgb(input_path, image_rgb)
    ermbg_io.save_rgba(output_path, selected_rgba)
    ermbg_io.save_mask(alpha_path, selected_rgba[..., 3].astype(np.float32) / 255.0)
    ermbg_io.save_rgb(foreground_path, result.foreground_srgb)
    summary = {
        "status": "ok",
        "filename": filename,
        "backend": _response_backend(requested_backend),
        "requested_backend": requested_backend,
        "strategy": result.strategy_name,
        "background": list(result.background_color),
        "server_elapsed_sec": server_elapsed_sec,
        **_route_metadata(result),
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = build_run_manifest(
        run_dir=run_dir,
        input_path=input_path,
        outputs={
            "rgba": output_path,
            "alpha": alpha_path,
            "foreground": foreground_path,
        },
        request={
            "backend": requested_backend,
            "effective_backend": effective_backend,
            "shadow_mode": shadow_mode,
            "filename": filename,
        },
        route=route_from_response(result),
        runtime={
            **runtime_from_response(result, requested_backend=requested_backend),
            "backend": effective_backend,
            "server_elapsed_sec": server_elapsed_sec,
        },
        report_path=report_path,
        result=result,
        extra={"stem": stem},
    )
    return write_run_manifest(run_dir / "manifest.json", manifest)


def _run_web_backend(
    image: Image.Image,
    *,
    backend: str,
    shadow_mode: str,
    corridorkey_screen_mode: str = "auto",
    corridorkey_preset: str = "auto",
    parameter_source: str = "auto",
    **kwargs: Any,
) -> MatteResponse:
    local_kwargs = dict(kwargs)
    # Direct Worker-only controls are posted by the unified Web form. If the
    # primary direct-worker path is unavailable and Web falls back to a local
    # backend, do not leak those controls into the public local API.
    local_kwargs.pop("known_bg_glow_material_strength", None)
    local_kwargs.pop("route_decision", None)
    execution_backend = backend
    requested_auto = backend == "auto"
    if backend == "auto":
        configured = WEB_AUTO_BACKEND.strip().lower()
        if configured in {"direct-worker", "direct_worker", "direct"}:
            execution_backend = "direct-worker"
        elif configured in {"auto-local", "local"}:
            execution_backend = "auto-local"
        else:
            raise ValueError(
                "ERMBG_WEB_AUTO_BACKEND must be direct-worker or auto-local"
            )
    execution_base = _direct_backend_base(execution_backend)
    direct_execution_backend = _execution_backend_for_algorithm(execution_backend)
    if direct_execution_backend is not None:
        servers = _direct_worker_servers_for_backend(execution_backend)
        has_route_decision = isinstance(kwargs.get("route_decision"), dict)
        manual_params = (
            str(parameter_source or "auto").strip().lower() == "manual"
            or _direct_backend_base(execution_backend) == "direct-corridorkey"
            or (not requested_auto and direct_execution_backend != "auto")
        )
        corridorkey_overrides: dict[str, Any] = {}
        if not has_route_decision or manual_params:
            corridorkey_overrides.update(
                {
                    "corridorkey_screen_mode": corridorkey_screen_mode,
                    "corridorkey_preset": corridorkey_preset,
                }
            )
        if kwargs.get("corridorkey_hint_mask") is not None:
            corridorkey_overrides["corridorkey_hint_mask"] = kwargs["corridorkey_hint_mask"]
        if isinstance(kwargs.get("semantic_decision"), dict):
            corridorkey_overrides["semantic_decision"] = kwargs["semantic_decision"]
        if kwargs.get("user_keep_mask") is not None:
            corridorkey_overrides["user_keep_mask"] = kwargs["user_keep_mask"]
        if kwargs.get("user_remove_mask") is not None:
            corridorkey_overrides["user_remove_mask"] = kwargs["user_remove_mask"]
        if has_route_decision:
            corridorkey_overrides["route_decision"] = kwargs["route_decision"]
        if direct_execution_backend == "direct-corridorkey" and manual_params:
            corridorkey_overrides.update(
                {
                    "corridorkey_gamma_space": str(kwargs.get("corridorkey_gamma_space", "sRGB")),
                    "corridorkey_despill_strength": float(kwargs.get("corridorkey_despill_strength", 1.0)),
                    "corridorkey_refiner_strength": float(kwargs.get("corridorkey_refiner_strength", 1.0)),
                    "corridorkey_auto_despeckle": str(kwargs.get("corridorkey_auto_despeckle", "On")),
                    "corridorkey_despeckle_size": int(kwargs.get("corridorkey_despeckle_size", 400)),
                    "corridorkey_auto_mask": bool(kwargs.get("corridorkey_auto_mask", False)),
                }
            )
        if direct_execution_backend == "direct-known-bg-glow" and manual_params:
            corridorkey_overrides["known_bg_glow_material_strength"] = float(
                kwargs.get("known_bg_glow_material_strength", 1.0)
            )
        known_b_keys = (
            "pymatting_method",
            "pymatting_image_space",
            "pymatting_bg_source",
            "pymatting_bg_color",
            "pymatting_bg_threshold",
            "pymatting_fg_threshold",
            "pymatting_boundary_band_px",
            "pymatting_adapt_bg_threshold",
            "pymatting_adapt_fg_threshold",
            "pymatting_adapt_boundary_band",
            "pymatting_cg_maxiter",
            "pymatting_cg_rtol",
            "pymatting_trimap_mode",
            "pymatting_unknown_grow_px",
            "pymatting_explicit_trimap",
            "pymatting_input_preprocessed",
        )
        route_payload = kwargs.get("route_decision") if isinstance(kwargs.get("route_decision"), dict) else {}
        route_params = route_payload.get("params") if isinstance(route_payload.get("params"), dict) else {}
        corridorkey_route_keys = {
            "corridorkey_gamma_space",
            "corridorkey_despill_strength",
            "corridorkey_refiner_strength",
            "corridorkey_auto_despeckle",
            "corridorkey_despeckle_size",
            "corridorkey_auto_mask",
            "corridorkey_screen_mode",
            "corridorkey_preset",
        }
        if direct_execution_backend == "direct-corridorkey" and route_params:
            for key in corridorkey_route_keys:
                if key in route_params:
                    corridorkey_overrides.setdefault(key, route_params[key])
        if kwargs.get("pymatting_explicit_trimap") is not None:
            corridorkey_overrides["pymatting_explicit_trimap"] = kwargs["pymatting_explicit_trimap"]
        if direct_execution_backend == "direct-pymatting-known-b":
            for key in known_b_keys:
                if key in route_params:
                    corridorkey_overrides[key] = route_params[key]
            for key in known_b_keys:
                if key in kwargs:
                    corridorkey_overrides[key] = kwargs[key]
        fallback_chain: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for server in servers:
            direct_worker_url = str(server["url"]).rstrip("/")
            endpoint_name = str(server.get("name") or "")
            try:
                result = matte_image_direct_worker(
                    image,
                    direct_worker_url=direct_worker_url,
                    execution_backend=direct_execution_backend,
                    shadow_mode=shadow_mode,
                    **corridorkey_overrides,
                )
                fallback_chain.append({"name": endpoint_name, "url": direct_worker_url, "status": "selected"})
                result.debug.setdefault("execution_server", endpoint_name)
                result.debug.setdefault("execution_server_url", direct_worker_url)
                result.debug.setdefault("server_fallback_chain", fallback_chain)
                result.debug.setdefault("parameter_source", "manual" if manual_params else "auto")
                result.debug.setdefault("web_direct_worker_url", direct_worker_url)
                if endpoint_name:
                    result.debug.setdefault("web_direct_worker_endpoint", endpoint_name)
                return result
            except Exception as exc:
                last_exc = exc
                fallback_chain.append({"name": endpoint_name, "url": direct_worker_url, "status": "error", "error": str(exc)})
        if last_exc is not None:
            exc = last_exc
            fallback = WEB_AUTO_FALLBACK_BACKEND.strip().lower()
            if not requested_auto or fallback in {"", "none", "off", "disabled"}:
                raise exc
            if fallback not in ALLOWED_BACKENDS or _execution_backend_for_algorithm(fallback) is not None:
                raise ValueError(
                    "ERMBG_WEB_AUTO_FALLBACK_BACKEND must be a non-direct backend or disabled"
                ) from exc
            result = matte_image(
                image,
                backend=fallback,
                qa=False,
                shadow_mode=shadow_mode,
                corridorkey_screen_mode=corridorkey_screen_mode,
                corridorkey_preset=corridorkey_preset,
                **local_kwargs,
            )
            result.debug.setdefault("web_auto_primary_error", str(exc))
            result.debug.setdefault("web_auto_primary_backend", execution_base)
            result.debug.setdefault("server_fallback_chain", fallback_chain)
            result.debug.setdefault("web_auto_fallback_backend", fallback)
            return result
    return matte_image(
        image,
        backend=execution_backend,
        qa=False,
        shadow_mode=shadow_mode,
        corridorkey_screen_mode=corridorkey_screen_mode,
        corridorkey_preset=corridorkey_preset,
        **local_kwargs,
    )


def _parse_rgb_triplet(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise HTTPException(status_code=400, detail="pymatting_bg_color must be R,G,B")
    try:
        rgb = tuple(int(part) for part in parts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="pymatting_bg_color must contain integers") from e
    if any(c < 0 or c > 255 for c in rgb):
        raise HTTPException(status_code=400, detail="pymatting_bg_color channels must be between 0 and 255")
    return rgb  # type: ignore[return-value]


def _pymatting_kwargs(
    *,
    pymatting_method: str,
    pymatting_image_space: str,
    pymatting_bg_source: str,
    pymatting_bg_color: str,
    pymatting_bg_threshold: float,
    pymatting_fg_threshold: float,
    pymatting_boundary_band_px: int,
    pymatting_cg_maxiter: int,
    pymatting_cg_rtol: float,
) -> dict[str, object]:
    method = pymatting_method.strip().lower()
    if method not in {"cf", "knn", "lbdm", "lkm", "rw", "sm"}:
        raise HTTPException(status_code=400, detail="pymatting_method must be cf, knn, lbdm, lkm, rw, or sm")
    if pymatting_image_space not in {"linear", "sRGB"}:
        raise HTTPException(status_code=400, detail="pymatting_image_space must be linear or sRGB")
    bg_source = pymatting_bg_source.strip().lower()
    if bg_source not in {"auto", "green", "blue", "custom"}:
        raise HTTPException(status_code=400, detail="pymatting_bg_source must be auto, green, blue, or custom")
    if not 0.0 <= pymatting_bg_threshold < pymatting_fg_threshold:
        raise HTTPException(status_code=400, detail="pymatting_bg_threshold must be >= 0 and less than pymatting_fg_threshold")
    if not 0 <= pymatting_boundary_band_px <= 16:
        raise HTTPException(status_code=400, detail="pymatting_boundary_band_px must be between 0 and 16")
    if not 1 <= pymatting_cg_maxiter <= 10000:
        raise HTTPException(status_code=400, detail="pymatting_cg_maxiter must be between 1 and 10000")
    if not 0.0 < pymatting_cg_rtol <= 0.01:
        raise HTTPException(status_code=400, detail="pymatting_cg_rtol must be between 0 and 0.01")
    return {
        "pymatting_method": method,
        "pymatting_image_space": pymatting_image_space,
        "pymatting_bg_source": bg_source,
        "pymatting_bg_color": _parse_rgb_triplet(pymatting_bg_color) if bg_source == "custom" else None,
        "pymatting_bg_threshold": pymatting_bg_threshold,
        "pymatting_fg_threshold": pymatting_fg_threshold,
        "pymatting_boundary_band_px": pymatting_boundary_band_px,
        "pymatting_adapt_bg_threshold": False,
        "pymatting_adapt_fg_threshold": True,
        "pymatting_adapt_boundary_band": True,
        "pymatting_cg_maxiter": pymatting_cg_maxiter,
        "pymatting_cg_rtol": pymatting_cg_rtol,
    }


def _shadow_mode_from_form(shadow_mode: str | None, shadow_enabled: bool | None) -> str:
    if shadow_mode is not None and str(shadow_mode).strip():
        mode = str(shadow_mode).strip().lower()
        if mode not in {"auto", "on", "off"}:
            raise HTTPException(status_code=400, detail="shadow_mode must be auto, on, or off")
        return mode
    if shadow_enabled is None:
        return "auto"
    return "on" if shadow_enabled else "off"


def _shadow_mode_from_semantic_decision(
    current_shadow_mode: str | None,
    semantic_decision: dict[str, Any],
) -> str | None:
    """Let semantic candidates override only explicit execution knobs.

    Shadow ownership used to be a candidate dimension. In the BG-seed outline
    flow it is only boundary evidence inside the trimap builder, so semantic
    decisions may pass a literal shadow_mode but no longer infer one from a
    shadow ownership policy.
    """

    mode = semantic_decision.get("shadow_mode")
    if isinstance(mode, str) and mode.strip():
        normalized = mode.strip().lower()
        if normalized not in {"auto", "on", "off"}:
            raise HTTPException(status_code=400, detail="semantic shadow_mode must be auto, on, or off")
        return normalized
    return current_shadow_mode


def _semantic_decision_requires_explicit_trimap(semantic_decision: dict[str, Any]) -> bool:
    if not semantic_decision:
        return False
    policy = semantic_decision.get("policy")
    if isinstance(policy, str) and policy == "auto_default" and len(semantic_decision) == 1:
        return False
    if (
        policy == "same_key_opaque_outline"
        and semantic_decision.get("button_body_policy") == "opaque_subject"
        and semantic_decision.get("pymatting_trimap_mode") == "same_key_opaque_body_outline"
        and set(semantic_decision).issubset(
            {
                "policy",
                "button_body_policy",
                "pymatting_trimap_mode",
                "pymatting_unknown_grow_px",
            }
        )
    ):
        return False
    # Route-only auto-default execution should let the executor build the fresh
    # Known-B trimap from the normalized image. Semantic ownership choices such
    # as hole cut/keep still need the Analyze-provided trimap overlay contract.
    # Same-key opaque outline is also route/mode evidence: the executor must build
    # the trimap itself so its opaque edge solver and color-restore masks share the
    # same measured contour instead of replaying the Analyze preview trimap.
    return True


def _known_b_execution_overrides_from_semantic_decision(semantic_decision: dict[str, Any]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    trimap_mode = semantic_decision.get("pymatting_trimap_mode")
    if isinstance(trimap_mode, str) and trimap_mode.strip():
        normalized = trimap_mode.strip()
        if normalized not in {"standard", "same_key_opaque_body_outline"}:
            raise HTTPException(status_code=400, detail="semantic pymatting_trimap_mode is not supported")
        overrides["pymatting_trimap_mode"] = normalized
    unknown_grow = semantic_decision.get("pymatting_unknown_grow_px")
    if isinstance(unknown_grow, (int, float)):
        value = int(unknown_grow)
        if not 0 <= value <= 16:
            raise HTTPException(status_code=400, detail="semantic pymatting_unknown_grow_px must be between 0 and 16")
        overrides["pymatting_unknown_grow_px"] = value
    return overrides


@app.post("/api/matte")
def matte_endpoint(
    file: Annotated[UploadFile, File()],
    backend: Annotated[str, Form()] = "auto",
    parameter_source: Annotated[str, Form()] = "auto",
    shadow_mode: Annotated[str | None, Form()] = None,
    shadow_enabled: Annotated[bool | None, Form()] = None,
    pymatting_method: Annotated[str, Form()] = "cf",
    pymatting_image_space: Annotated[str, Form()] = "linear",
    pymatting_bg_source: Annotated[str, Form()] = "auto",
    pymatting_bg_color: Annotated[str, Form()] = "0,200,0",
    pymatting_bg_threshold: Annotated[float, Form()] = 3.5,
    pymatting_fg_threshold: Annotated[float, Form()] = 24.0,
    pymatting_boundary_band_px: Annotated[int, Form()] = 2,
    pymatting_cg_maxiter: Annotated[int, Form()] = 1000,
    pymatting_cg_rtol: Annotated[float, Form()] = 1e-6,
    background_repair: Annotated[bool, Form()] = False,
) -> Response:
    if not _is_allowed_backend(backend):
        raise HTTPException(status_code=400, detail=f"backend must be one of {_allowed_backend_names()}")
    if parameter_source not in {"auto", "manual"}:
        raise HTTPException(status_code=400, detail="parameter_source must be auto or manual")

    image = _load_upload_image(file)
    image, preprocess_info = _preprocess_background_repair_image(image, background_repair)
    shadow_mode = _shadow_mode_from_form(shadow_mode, shadow_enabled)
    pymatting_params = _pymatting_kwargs(
        pymatting_method=pymatting_method,
        pymatting_image_space=pymatting_image_space,
        pymatting_bg_source=pymatting_bg_source,
        pymatting_bg_color=pymatting_bg_color,
        pymatting_bg_threshold=pymatting_bg_threshold,
        pymatting_fg_threshold=pymatting_fg_threshold,
        pymatting_boundary_band_px=pymatting_boundary_band_px,
        pymatting_cg_maxiter=pymatting_cg_maxiter,
        pymatting_cg_rtol=pymatting_cg_rtol,
    )
    image, known_b_preprocess_info, known_b_execution_params, route_decision = _apply_web_matte_known_b_background_repair(
        image,
        backend=backend,
        background_repair=background_repair,
        pymatting_params=pymatting_params,
    )
    if known_b_execution_params:
        pymatting_params = {**pymatting_params, **known_b_execution_params}
    route_kwargs = {"route_decision": route_decision} if route_decision is not None else {}
    try:
        result = _run_web_backend(
            image,
            backend=backend,
            shadow_mode=shadow_mode,
            parameter_source=parameter_source,
            **route_kwargs,
            **pymatting_params,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matting failed: {e}") from e
    result.debug["input_preprocess"] = {
        "background_repair": preprocess_info,
        "known_background_normalization": known_b_preprocess_info,
    }

    effective_backend = _effective_backend(backend, result)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    selected_rgba = result.rgba
    local_ownership_used = False
    try:
        local_candidate = generate_local_ownership_candidate(
            image_rgb,
            result.rgba,
            result.background_color,
            backend=effective_backend,
            soft_mask=result.debug.get("soft_mask"),
            shadow_mode=shadow_mode,
        ) if effective_backend not in REMOTE_DIRECT_BACKENDS else None
    except Exception:
        local_candidate = None
    if local_candidate is not None:
        selected_rgba = local_candidate.rgba
        local_ownership_used = True

    png = _encode_png(selected_rgba)
    filename = (file.filename or "ermbg").rsplit(".", 1)[0] + "_rgba.png"
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-ERMBG-Strategy": result.strategy_name,
            "X-ERMBG-Background": ",".join(str(c) for c in result.background_color),
            "X-ERMBG-Local-Ownership": "1" if local_ownership_used else "0",
            "X-ERMBG-Background-Repair": "1" if preprocess_info.get("applied") else "0",
        },
    )


def _execute_matte_candidates_payload(
    request: WebExecutionRequest,
    *,
    include_compatibility: bool = False,
) -> dict[str, object]:
    if not _is_allowed_backend(request.backend):
        raise HTTPException(status_code=400, detail=f"backend must be one of {_allowed_backend_names()}")
    if request.parameter_source not in {"auto", "manual"}:
        raise HTTPException(status_code=400, detail="parameter_source must be auto or manual")
    if request.corridorkey_gamma_space not in {"sRGB", "Linear"}:
        raise HTTPException(status_code=400, detail="corridorkey_gamma_space must be sRGB or Linear")
    if request.corridorkey_auto_despeckle not in {"On", "Off"}:
        raise HTTPException(status_code=400, detail="corridorkey_auto_despeckle must be On or Off")
    if request.corridorkey_screen_mode not in {"auto", "green", "blue"}:
        raise HTTPException(status_code=400, detail="corridorkey_screen_mode must be auto, green, or blue")
    if request.corridorkey_preset not in {"auto", "detail_safe", "spill_safe", "manual"}:
        raise HTTPException(status_code=400, detail="corridorkey_preset must be auto, detail_safe, spill_safe, or manual")
    if not 0.0 <= request.corridorkey_despill_strength <= 1.0:
        raise HTTPException(status_code=400, detail="corridorkey_despill_strength must be between 0 and 1")
    if not 0.0 <= request.corridorkey_refiner_strength <= 4.0:
        raise HTTPException(status_code=400, detail="corridorkey_refiner_strength must be between 0 and 4")
    if not 0 <= request.corridorkey_despeckle_size <= 4096:
        raise HTTPException(status_code=400, detail="corridorkey_despeckle_size must be between 0 and 4096")
    if not 0.0 <= request.known_bg_glow_material_strength <= 2.0:
        raise HTTPException(status_code=400, detail="known_bg_glow_material_strength must be between 0 and 2")
    shadow_mode = _shadow_mode_from_form(request.shadow_mode, request.shadow_enabled)
    pymatting_params = _pymatting_kwargs(
        pymatting_method=request.pymatting_method,
        pymatting_image_space=request.pymatting_image_space,
        pymatting_bg_source=request.pymatting_bg_source,
        pymatting_bg_color=request.pymatting_bg_color,
        pymatting_bg_threshold=request.pymatting_bg_threshold,
        pymatting_fg_threshold=request.pymatting_fg_threshold,
        pymatting_boundary_band_px=request.pymatting_boundary_band_px,
        pymatting_cg_maxiter=request.pymatting_cg_maxiter,
        pymatting_cg_rtol=request.pymatting_cg_rtol,
    )
    semantic_decision_payload = _json_form_object(request.semantic_decision, "semantic_decision")
    semantic_known_b_overrides = _known_b_execution_overrides_from_semantic_decision(semantic_decision_payload)
    execution_contract = request.execution_request_payload or request.analysis_payload

    image = _load_upload_image(request.file)
    image, preprocess_info = _preprocess_background_repair_image(image, request.background_repair)
    image, known_b_preprocess_info, known_b_execution_params = _known_b_preprocess_from_contract(
        image,
        execution_contract,
        pymatting_params,
    )
    if known_b_execution_params:
        pymatting_params = {**pymatting_params, **known_b_execution_params}
    pymatting_params.update(semantic_known_b_overrides)
    explicit_route_decision = _analysis_route_decision_payload(execution_contract)
    explicit_route_decision = _merge_known_b_execution_params_into_route_decision(
        explicit_route_decision,
        pymatting_params,
    )
    hint_mask = (
        _load_upload_image(request.corridorkey_hint_mask)
        if request.corridorkey_hint_mask is not None and not request.corridorkey_auto_mask
        else None
    )
    keep_mask = _load_upload_image(request.user_keep_mask) if request.user_keep_mask is not None else None
    remove_mask = _load_upload_image(request.user_remove_mask) if request.user_remove_mask is not None else None
    execution_request = request.execution_request_payload if isinstance(request.execution_request_payload, dict) else {}
    selected_candidate_id = str(execution_request.get("selected_candidate_id") or "auto_default")
    explicit_trimap = (
        _explicit_trimap_from_analysis(
            request.analysis_payload,
            selected_candidate_id=selected_candidate_id,
            image_shape=np.asarray(image.convert("RGB"), dtype=np.uint8).shape[:2],
        )
        if _semantic_decision_requires_explicit_trimap(semantic_decision_payload)
        else None
    )
    server_started_at = time.perf_counter()
    try:
        execution_backend = (
            request.backend
            if explicit_route_decision is not None
            else _execute_backend_from_analysis(request.backend, execution_contract)
        )
        result = _run_web_backend(
            image,
            backend=execution_backend,
            shadow_mode=shadow_mode,
            corridorkey_gamma_space=request.corridorkey_gamma_space,
            corridorkey_despill_strength=request.corridorkey_despill_strength,
            corridorkey_refiner_strength=request.corridorkey_refiner_strength,
            corridorkey_auto_despeckle=request.corridorkey_auto_despeckle,
            corridorkey_despeckle_size=request.corridorkey_despeckle_size,
            corridorkey_auto_mask=request.corridorkey_auto_mask,
            corridorkey_screen_mode=request.corridorkey_screen_mode,
            corridorkey_preset=request.corridorkey_preset,
            known_bg_glow_material_strength=request.known_bg_glow_material_strength,
            parameter_source=request.parameter_source,
            corridorkey_hint_mask=hint_mask,
            route_decision=explicit_route_decision,
            semantic_decision=semantic_decision_payload or None,
            user_keep_mask=keep_mask,
            user_remove_mask=remove_mask,
            pymatting_explicit_trimap=explicit_trimap,
            **pymatting_params,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matting failed: {e}") from e
    result.debug["input_preprocess"] = {
        "background_repair": preprocess_info,
        "known_background_normalization": known_b_preprocess_info,
    }

    stem = (request.file.filename or "ermbg").rsplit(".", 1)[0]
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    effective_backend = _effective_backend(request.backend, result)
    if effective_backend in REMOTE_DIRECT_BACKENDS:
        if effective_backend == "corridorkey":
            direct_label = "CorridorKey"
        elif effective_backend in {"pymatting_known_b", "pymatting-known-b"}:
            direct_label = "PyMatting Known-B"
        elif effective_backend == "rmbg":
            direct_label = "RMBG"
        elif effective_backend == "direct-worker":
            direct_label = "Direct Worker"
        elif effective_backend == "direct-corridorkey":
            direct_label = "Direct Worker CorridorKey"
        elif effective_backend == "direct-pymatting-known-b":
            direct_label = "Direct Worker PyMatting Known-B"
        elif effective_backend == "direct-known-bg-glow":
            direct_label = "Direct Worker Known-B Glow"
        elif effective_backend == "passthrough":
            direct_label = "Passthrough"
        else:
            direct_label = "PyMatting Known-B"
        candidates = [
            MatteCandidate(
                id="auto",
                label=direct_label,
                rgba=result.rgba,
                selected=True,
                debug={"remote": result.debug},
            )
        ]
    else:
        candidates = generate_matte_candidates(image_rgb, result.rgba, result.background_color)
    try:
        local_candidate = generate_local_ownership_candidate(
            image_rgb,
            result.rgba,
            result.background_color,
            backend=effective_backend,
            soft_mask=result.debug.get("soft_mask"),
            shadow_mode=shadow_mode,
        ) if effective_backend not in REMOTE_DIRECT_BACKENDS else None
    except Exception as e:
        local_candidate = None
        for candidate in candidates:
            candidate.debug["local_ownership_error"] = str(e)
    if local_candidate is not None:
        for candidate in candidates:
            candidate.selected = False
        candidates.append(local_candidate)
    server_elapsed_sec = time.perf_counter() - server_started_at
    artifact_manifest = _write_web_matte_artifacts(
        image_rgb=image_rgb,
        selected_rgba=next((candidate.rgba for candidate in candidates if candidate.selected), result.rgba),
        result=result,
        filename=request.file.filename or "ermbg.png",
        requested_backend=request.backend,
        effective_backend=effective_backend,
        shadow_mode=shadow_mode,
        server_elapsed_sec=server_elapsed_sec,
    )
    payload: dict[str, object] = {
        "strategy": result.strategy_name,
        "background": list(result.background_color),
        "backend": _response_backend(request.backend),
        "requested_backend": request.backend,
        "pipeline_mode": "execute_candidate",
        **_route_metadata(result),
        "server_elapsed_sec": server_elapsed_sec,
        "artifact_manifest": str(artifact_manifest.relative_to(PROJECT_ROOT)) if _is_relative_to(artifact_manifest, PROJECT_ROOT) else str(artifact_manifest),
        "debug": _json_safe_debug(result.debug),
        "candidates": [_candidate_payload(candidate, stem) for candidate in candidates],
    }
    if include_compatibility:
        payload["compatibility"] = dict(LEGACY_MATTE_CANDIDATES_COMPAT)
        payload["pipeline_mode"] = "legacy_matte_candidates_compat"
    return payload


@app.post("/api/matte-candidates", deprecated=True)
def matte_candidates_endpoint(
    file: Annotated[UploadFile, File()],
    corridorkey_hint_mask: Annotated[UploadFile | None, File()] = None,
    user_keep_mask: Annotated[UploadFile | None, File()] = None,
    user_remove_mask: Annotated[UploadFile | None, File()] = None,
    backend: Annotated[str, Form()] = "auto",
    parameter_source: Annotated[str, Form()] = "auto",
    shadow_mode: Annotated[str | None, Form()] = None,
    shadow_enabled: Annotated[bool | None, Form()] = None,
    corridorkey_gamma_space: Annotated[str, Form()] = "sRGB",
    corridorkey_despill_strength: Annotated[float, Form()] = 1.0,
    corridorkey_refiner_strength: Annotated[float, Form()] = 1.0,
    corridorkey_auto_despeckle: Annotated[str, Form()] = "On",
    corridorkey_despeckle_size: Annotated[int, Form()] = 400,
    corridorkey_auto_mask: Annotated[bool, Form()] = False,
    corridorkey_screen_mode: Annotated[str, Form()] = "auto",
    corridorkey_preset: Annotated[str, Form()] = "auto",
    pymatting_method: Annotated[str, Form()] = "cf",
    pymatting_image_space: Annotated[str, Form()] = "linear",
    pymatting_bg_source: Annotated[str, Form()] = "auto",
    pymatting_bg_color: Annotated[str, Form()] = "0,200,0",
    pymatting_bg_threshold: Annotated[float, Form()] = 3.5,
    pymatting_fg_threshold: Annotated[float, Form()] = 24.0,
    pymatting_boundary_band_px: Annotated[int, Form()] = 2,
    pymatting_cg_maxiter: Annotated[int, Form()] = 1000,
    pymatting_cg_rtol: Annotated[float, Form()] = 1e-6,
    known_bg_glow_material_strength: Annotated[float, Form()] = 1.0,
    background_repair: Annotated[bool, Form()] = False,
    semantic_decision: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    return _execute_matte_candidates_payload(
        WebExecutionRequest(
            file=file,
            corridorkey_hint_mask=corridorkey_hint_mask,
            user_keep_mask=user_keep_mask,
            user_remove_mask=user_remove_mask,
            backend=backend,
            parameter_source=parameter_source,
            shadow_mode=shadow_mode,
            shadow_enabled=shadow_enabled,
            corridorkey_gamma_space=corridorkey_gamma_space,
            corridorkey_despill_strength=corridorkey_despill_strength,
            corridorkey_refiner_strength=corridorkey_refiner_strength,
            corridorkey_auto_despeckle=corridorkey_auto_despeckle,
            corridorkey_despeckle_size=corridorkey_despeckle_size,
            corridorkey_auto_mask=corridorkey_auto_mask,
            corridorkey_screen_mode=corridorkey_screen_mode,
            corridorkey_preset=corridorkey_preset,
            pymatting_method=pymatting_method,
            pymatting_image_space=pymatting_image_space,
            pymatting_bg_source=pymatting_bg_source,
            pymatting_bg_color=pymatting_bg_color,
            pymatting_bg_threshold=pymatting_bg_threshold,
            pymatting_fg_threshold=pymatting_fg_threshold,
            pymatting_boundary_band_px=pymatting_boundary_band_px,
            pymatting_cg_maxiter=pymatting_cg_maxiter,
            pymatting_cg_rtol=pymatting_cg_rtol,
            known_bg_glow_material_strength=known_bg_glow_material_strength,
            background_repair=background_repair,
            semantic_decision=semantic_decision,
        ),
        include_compatibility=True,
    )


@app.post("/api/execute-candidate")
def execute_candidate_endpoint(
    file: Annotated[UploadFile, File()],
    corridorkey_hint_mask: Annotated[UploadFile | None, File()] = None,
    user_keep_mask: Annotated[UploadFile | None, File()] = None,
    user_remove_mask: Annotated[UploadFile | None, File()] = None,
    selected_candidate_id: Annotated[str, Form()] = "auto_default",
    semantic_decision: Annotated[str, Form()] = "{}",
    analysis_payload: Annotated[str, Form()] = "{}",
    backend: Annotated[str, Form()] = "auto",
    parameter_source: Annotated[str, Form()] = "auto",
    shadow_mode: Annotated[str | None, Form()] = None,
    shadow_enabled: Annotated[bool | None, Form()] = None,
    corridorkey_gamma_space: Annotated[str, Form()] = "sRGB",
    corridorkey_despill_strength: Annotated[float, Form()] = 1.0,
    corridorkey_refiner_strength: Annotated[float, Form()] = 1.0,
    corridorkey_auto_despeckle: Annotated[str, Form()] = "On",
    corridorkey_despeckle_size: Annotated[int, Form()] = 400,
    corridorkey_auto_mask: Annotated[bool, Form()] = False,
    corridorkey_screen_mode: Annotated[str, Form()] = "auto",
    corridorkey_preset: Annotated[str, Form()] = "auto",
    pymatting_method: Annotated[str, Form()] = "cf",
    pymatting_image_space: Annotated[str, Form()] = "linear",
    pymatting_bg_source: Annotated[str, Form()] = "auto",
    pymatting_bg_color: Annotated[str, Form()] = "0,200,0",
    pymatting_bg_threshold: Annotated[float, Form()] = 3.5,
    pymatting_fg_threshold: Annotated[float, Form()] = 24.0,
    pymatting_boundary_band_px: Annotated[int, Form()] = 2,
    pymatting_cg_maxiter: Annotated[int, Form()] = 1000,
    pymatting_cg_rtol: Annotated[float, Form()] = 1e-6,
    known_bg_glow_material_strength: Annotated[float, Form()] = 1.0,
    background_repair: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    analysis = _json_form_object(analysis_payload, "analysis_payload")
    decision = _json_form_object(semantic_decision, "semantic_decision")
    if not decision:
        decision, _candidate_confidence = _semantic_candidate_payload(analysis, selected_candidate_id)
    shadow_mode = _shadow_mode_from_semantic_decision(shadow_mode, decision)
    user_mask_summary = _user_mask_upload_summary(
        keep_mask=user_keep_mask,
        remove_mask=user_remove_mask,
    )
    user_mask = UserMaskDecision(
        keep_mask="uploaded:user_keep_mask" if user_keep_mask is not None else None,
        remove_mask="uploaded:user_remove_mask" if user_remove_mask is not None else None,
        source="web_user_brush" if user_keep_mask is not None or user_remove_mask is not None else "none",
        summary=user_mask_summary,
    )
    execution_summary = _semantic_execution_summary(
        analysis_payload=analysis,
        selected_candidate_id=selected_candidate_id,
        semantic_decision_payload=decision,
        user_mask=user_mask,
    )
    payload = _execute_matte_candidates_payload(
        WebExecutionRequest(
            file=file,
            corridorkey_hint_mask=corridorkey_hint_mask,
            user_keep_mask=user_keep_mask,
            user_remove_mask=user_remove_mask,
            backend=backend,
            parameter_source=parameter_source,
            shadow_mode=shadow_mode,
            shadow_enabled=shadow_enabled,
            corridorkey_gamma_space=corridorkey_gamma_space,
            corridorkey_despill_strength=corridorkey_despill_strength,
            corridorkey_refiner_strength=corridorkey_refiner_strength,
            corridorkey_auto_despeckle=corridorkey_auto_despeckle,
            corridorkey_despeckle_size=corridorkey_despeckle_size,
            corridorkey_auto_mask=corridorkey_auto_mask,
            corridorkey_screen_mode=corridorkey_screen_mode,
            corridorkey_preset=corridorkey_preset,
            pymatting_method=pymatting_method,
            pymatting_image_space=pymatting_image_space,
            pymatting_bg_source=pymatting_bg_source,
            pymatting_bg_color=pymatting_bg_color,
            pymatting_bg_threshold=pymatting_bg_threshold,
            pymatting_fg_threshold=pymatting_fg_threshold,
            pymatting_boundary_band_px=pymatting_boundary_band_px,
            pymatting_cg_maxiter=pymatting_cg_maxiter,
            pymatting_cg_rtol=pymatting_cg_rtol,
            known_bg_glow_material_strength=known_bg_glow_material_strength,
            background_repair=_execute_background_repair_from_contract(analysis, background_repair),
            semantic_decision=json.dumps(decision),
            analysis_payload=analysis,
            execution_request_payload=execution_summary["execution_request"],
        ),
        include_compatibility=False,
    )
    return _attach_semantic_execution_metadata(
        payload,
        analysis_payload=analysis,
        selected_candidate_id=selected_candidate_id,
        semantic_decision_payload=decision,
        user_mask=user_mask,
    )


@app.post("/api/slice")
def slice_endpoint(
    file: Annotated[UploadFile, File()],
    min_area: Annotated[int, Form()] = 64,
    padding: Annotated[int, Form()] = 4,
    transparent: Annotated[bool, Form()] = False,
    background_repair: Annotated[bool, Form()] = False,
) -> Response:
    image, image_digest = _load_upload_image_with_digest(file)
    image, preprocess_info = _preprocess_background_repair_image(image, background_repair)
    if preprocess_info.get("applied", False):
        image_digest = _image_digest(image)
    try:
        image_rgb, result = _slice_source_and_result(image, image_digest, min_area=min_area, padding=padding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"slicing failed: {e}") from e

    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}.slices.json", json.dumps(result.to_dict(), indent=2))
        for box in result.boxes:
            crop = crop_slice(image_rgb, result.foreground_mask, box, padding=result.padding, transparent=transparent)
            png = _encode_png(crop) if transparent else _encode_rgb_png(crop)
            suffix = "rgba" if transparent else "rgb"
            zf.writestr(f"{stem}_{box.id:03d}_{suffix}.png", png)

    filename = f"{stem}_slices.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": _attachment_content_disposition(filename),
            "X-ERMBG-Slice-Count": str(len(result.boxes)),
            "X-ERMBG-Background": ",".join(str(c) for c in result.background_color),
            "X-ERMBG-Background-Repair": "1" if preprocess_info.get("applied") else "0",
        },
    )


@app.post("/api/slice-preview")
def slice_preview_endpoint(
    file: Annotated[UploadFile, File()],
    min_area: Annotated[int, Form()] = 64,
    padding: Annotated[int, Form()] = 4,
    background_repair: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    image, image_digest = _load_upload_image_with_digest(file)
    image, preprocess_info = _preprocess_background_repair_image(image, background_repair)
    if preprocess_info.get("applied", False):
        image_digest = _image_digest(image)
    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    try:
        image_rgb, result = _slice_source_and_result(image, image_digest, min_area=min_area, padding=padding)
        payload = _slice_preview_payload(image_rgb, stem, result)
        payload["preprocess"] = {"checkerboard": preprocess_info}
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"slice preview failed: {e}") from e


@app.post("/api/slice-crops")
def slice_crops_endpoint(
    file: Annotated[UploadFile, File()],
    min_area: Annotated[int, Form()] = 64,
    padding: Annotated[int, Form()] = 4,
    background_repair: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    image, image_digest = _load_upload_image_with_digest(file)
    image, preprocess_info = _preprocess_background_repair_image(image, background_repair)
    if preprocess_info.get("applied", False):
        image_digest = _image_digest(image)
    stem = (file.filename or "ermbg").rsplit(".", 1)[0]
    try:
        image_rgb, result = _slice_source_and_result(image, image_digest, min_area=min_area, padding=padding)
        payload = _slice_crop_payloads(image_rgb, stem, result)
        payload["preprocess"] = {"checkerboard": preprocess_info}
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"slice crops failed: {e}") from e


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Missing file: {path.relative_to(PROJECT_ROOT)}") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON: {path.relative_to(PROJECT_ROOT)}") from e


def _load_optional_game_eval_summary(path: Path) -> object | None:
    try:
        return _load_json(path)
    except HTTPException:
        # out/ is shared by ad-hoc smoke runs and web artifacts. Game Eval
        # discovery should ignore unrelated or malformed summaries instead of
        # making the whole page unavailable before a run is explicitly selected.
        return None


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _image_url(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path = _resolve_project_path(path_value)
    # The /eval/game/file endpoint only serves out/ and samples/. Mirror that
    # allow-list here so a manifest field that points elsewhere under the repo
    # (e.g. ermbg/ source, docs/assets) never gets a servable URL. Path values
    # originate from on-disk manifests/case.json that are not fully trusted.
    if not any(_is_relative_to(path, (PROJECT_ROOT / root).resolve()) for root in ("out", "samples")):
        return None
    if not path.exists() or path.suffix.lower() not in SERVABLE_IMAGE_SUFFIXES:
        return None
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return f"/eval/game/file/{quote(rel, safe='/')}"


def _artifact_id(path: Path) -> str:
    rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    return quote(rel, safe="")


def _artifact_path_from_id(artifact_id: str) -> Path:
    rel = unquote(artifact_id)
    path = (PROJECT_ROOT / rel).resolve()
    out_root = (PROJECT_ROOT / "out").resolve()
    if not _is_relative_to(path, out_root):
        raise HTTPException(status_code=404, detail="Artifact is outside output root.")
    if path.name != "manifest.json":
        raise HTTPException(status_code=404, detail="Artifact id must point to manifest.json.")
    return path


def _artifact_type(path: Path, manifest: dict[str, Any]) -> str:
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    if runtime.get("kind") == "game-eval":
        return "game-eval-batch"
    parent = path.parent
    if parent.parent.name.startswith(WEB_MATTE_RUN_PREFIX):
        return "web-matte"
    request = manifest.get("request") if isinstance(manifest.get("request"), dict) else {}
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    if request.get("source_input") or extra.get("case_metadata"):
        return "game-eval-case"
    return "run"


def _artifact_file_url(path_value: str | Path | None, manifest_path: Path) -> str | None:
    if path_value is None:
        return None
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    candidate = candidate.resolve()
    if not _is_relative_to(candidate, (PROJECT_ROOT / "out").resolve()):
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    rel = candidate.relative_to(PROJECT_ROOT).as_posix()
    if candidate.suffix.lower() in SERVABLE_IMAGE_SUFFIXES:
        return f"/eval/game/file/{quote(rel, safe='/')}"
    return None


def _artifact_summary(manifest_path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(manifest, dict) or manifest.get("schema") != "ermbg.run.v1":
        return None
    request = manifest.get("request") if isinstance(manifest.get("request"), dict) else {}
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    route = manifest.get("route") if isinstance(manifest.get("route"), dict) else {}
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    pipeline = extra.get("pipeline") if isinstance(extra.get("pipeline"), dict) else {}
    preprocess = pipeline.get("preprocess") if isinstance(pipeline.get("preprocess"), dict) else {}
    semantic = pipeline.get("semantic") if isinstance(pipeline.get("semantic"), dict) else {}
    return {
        "id": _artifact_id(manifest_path),
        "type": _artifact_type(manifest_path, manifest),
        "manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "mtime": manifest_path.stat().st_mtime,
        "backend": request.get("effective_backend") or runtime.get("backend") or request.get("backend"),
        "requested_backend": request.get("backend") or runtime.get("requested_backend"),
        "strategy": runtime.get("strategy"),
        "route": route.get("route"),
        "execution_profile": route.get("execution_profile"),
        "execution_backend": runtime.get("execution_backend") or runtime.get("backend"),
        "execution_server_url": runtime.get("execution_server_url") or runtime.get("execution_url"),
        "pipeline": pipeline,
        "preprocess": preprocess,
        "semantic": semantic,
        "analysis_status": semantic.get("analysis_status"),
        "selected_candidate_id": semantic.get("selected_candidate_id") or request.get("selected_candidate_id"),
        "default_candidate_id": semantic.get("default_candidate_id"),
        "user_mask_used": bool(semantic.get("user_mask_used", False)),
        "user_mask_summary": semantic.get("user_mask_summary") if isinstance(semantic.get("user_mask_summary"), dict) else {},
        "input": manifest.get("input"),
        "report": manifest.get("report"),
        "outputs": outputs,
        "urls": {
            key: _artifact_file_url(value, manifest_path)
            for key, value in outputs.items()
            if isinstance(key, str)
        },
    }


def _list_artifacts(limit: int = 200) -> list[dict[str, Any]]:
    out_root = PROJECT_ROOT / "out"
    if not out_root.exists():
        return []
    items = [
        item
        for path in out_root.glob("**/manifest.json")
        if (item := _artifact_summary(path)) is not None
    ]
    items.sort(key=lambda item: float(item.get("mtime", 0.0)), reverse=True)
    return items[:limit]


def _candidate_result_items(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = _load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _candidate_tools(candidate: dict[str, object], fallback: list[str]) -> list[str]:
    plan = candidate.get("plan")
    if not isinstance(plan, dict):
        return fallback
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return fallback
    tools: list[str] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        tool = operation.get("tool")
        if isinstance(tool, str) and tool not in tools:
            tools.append(tool)
    return tools or fallback


def _candidate_regions(candidate_results: list[dict[str, object]]) -> list[dict[str, object]]:
    for candidate in candidate_results:
        regions = candidate.get("regions")
        if isinstance(regions, list):
            return [region for region in regions if isinstance(region, dict)]
    return []


def _game_sample_paths(case_id: str) -> dict[str, str]:
    sample_root = _game_sample_root()
    case_path = sample_root / case_id / "case.json"
    if not case_path.exists():
        for category in ("button", "icon", "character"):
            candidate = sample_root / category / case_id / "case.json"
            if candidate.exists():
                case_path = candidate
                break
    if case_path.exists():
        payload = _load_json(case_path)
        if isinstance(payload, dict):
            paths = {
                screen: path
                for screen in GAME_EVAL_SCREENS
                if isinstance(path := payload.get(screen), str)
            }
            if paths:
                return paths
    return {screen: f"{GAME_SAMPLE_REL.as_posix()}/{case_id}/{screen}.png" for screen in GAME_EVAL_SCREENS}


def _sample_screen_from_path(path_value: object) -> str | None:
    if not isinstance(path_value, str):
        return None
    stem = Path(path_value).stem.lower()
    if stem in {"white", "green", "blue"}:
        return stem
    return None


def _game_sample_ids() -> dict[str, str]:
    manifest = _load_json(_game_sample_manifest())
    if not isinstance(manifest, dict):
        return {}
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return {}
    sample_ids: dict[str, str] = {}
    for index, item in enumerate(cases, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        fallback = f"G{index:02d}"
        sample_id = item.get("sample_id")
        sample_ids[item["id"]] = sample_id if isinstance(sample_id, str) else fallback
    return sample_ids


def _game_eval_samples() -> list[dict[str, object]]:
    if not _game_sample_manifest().exists():
        return []
    manifest = _load_json(_game_sample_manifest())
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list):
        return []
    samples: list[dict[str, object]] = []
    for index, item in enumerate(cases, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        sample_id = item.get("sample_id")
        sample_id = sample_id if isinstance(sample_id, str) else f"G{index:02d}"
        sample_paths = _game_sample_paths(item["id"])
        thumb_url = (
            _image_url(sample_paths.get("green"))
            or _image_url(sample_paths.get("white"))
            or _image_url(sample_paths.get("blue"))
        )
        samples.append(
            {
                "sampleId": sample_id,
                "caseId": item["id"],
                "category": item.get("category", ""),
                "family": item.get("family", ""),
                "screen": item.get("screen", ""),
                "primaryAmbiguity": item.get("primary_ambiguity", ""),
                "thumbnailUrl": thumb_url,
                "defaultSelected": sample_id in FAST_GAME_EVAL_SAMPLE_IDS,
            }
        )
    return samples


def _game_eval_manifest_cases() -> list[dict[str, object]]:
    if not _game_sample_manifest().exists():
        return []
    manifest = _load_json(_game_sample_manifest())
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    return [case for case in cases if isinstance(case, dict)] if isinstance(cases, list) else []


def _game_eval_case_count(
    cases: list[dict[str, object]] | None = None,
    sample_ids: list[str] | None = None,
) -> int:
    selected = cases if cases is not None else _game_eval_manifest_cases()
    if not selected and sample_ids:
        return len(sample_ids)
    if sample_ids:
        wanted = set(sample_ids)
        selected = [case for case in selected if str(case.get("sample_id", "")) in wanted]
    return sum(
        1
        for case in selected
        if isinstance(case.get("input"), str)
        or any(isinstance(case.get(screen), str) for screen in GAME_EVAL_SCREENS)
    )


def _game_report_path(root: Path) -> Path | None:
    path = root / "local_ownership" / "eval_report.json"
    if path.exists():
        return path
    return None


def _game_vlm_root(root: Path) -> Path:
    report_path = _game_report_path(root)
    if report_path is not None:
        return report_path.parent
    return root / "local_ownership"


def _game_eval_partial_summary_paths(root: Path) -> list[Path]:
    local_root = root / "local_ownership"
    if not local_root.exists():
        return []
    return sorted(
        path
        for path in local_root.glob("*/*/summary.json")
        if path.is_file()
    )


def _solid_graphic_summary_path(root: Path) -> Path | None:
    path = root / "summary.json"
    if not path.is_file():
        return None
    payload = _load_optional_game_eval_summary(path)
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    batch = str(payload.get("batch", ""))
    if root.name.startswith(SOLID_GRAPHIC_EVAL_PREFIX) or batch.startswith(f"out/{SOLID_GRAPHIC_EVAL_PREFIX}"):
        return path
    if payload.get("solid_graphic_prepass") is True or isinstance(payload.get("strategy_pairs"), dict):
        return path
    return None


def _remote_backend_summary_path(root: Path) -> Path | None:
    path = root / "summary.json"
    if not path.is_file():
        return None
    payload = _load_optional_game_eval_summary(path)
    if not isinstance(payload, dict):
        return None
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return None
    remote_summary_backends = {"direct-worker", "auto"}
    for item in runs:
        backend = str(item.get("backend", "")) if isinstance(item, dict) else ""
        if backend.startswith("comfy-") or backend.startswith("direct-") or backend in remote_summary_backends:
            return path
    return None


def _route_analyze_summary_path(root: Path) -> Path | None:
    path = root / "summary.json"
    if not path.is_file():
        return None
    payload = _load_optional_game_eval_summary(path)
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") == "ermbg.route_analyze.batch.v1" and isinstance(payload.get("runs"), list):
        return path
    return None


def _game_eval_root_has_data(root: Path) -> bool:
    if _game_report_path(root) is not None:
        return True
    if _game_eval_partial_summary_paths(root):
        return True
    if _solid_graphic_summary_path(root) is not None:
        return True
    if _remote_backend_summary_path(root) is not None:
        return True
    if _route_analyze_summary_path(root) is not None:
        return True
    return _game_matte_summary_path(root) is not None


def _game_eval_root_is_complete(root: Path) -> bool:
    report_path = _game_report_path(root)
    if report_path is None:
        solid_path = _solid_graphic_summary_path(root)
        if solid_path is None:
            comfy_path = _remote_backend_summary_path(root)
            if comfy_path is None:
                return False
            report = _load_json(comfy_path)
            if not isinstance(report, dict):
                return False
            runs = report.get("runs")
            return isinstance(runs, list) and len(runs) >= _game_eval_expected_case_count()
        report = _load_json(solid_path)
        if not isinstance(report, dict):
            return False
        rows = report.get("rows")
        case_count = report.get("case_count")
        try:
            return isinstance(rows, list) and len(rows) >= int(case_count or _game_eval_expected_case_count())
        except (TypeError, ValueError):
            return False
    report = _load_json(report_path)
    if not isinstance(report, dict):
        return False
    try:
        return int(report.get("case_count", 0)) >= 18
    except (TypeError, ValueError):
        return False


def _game_eval_root_sort_key(root: Path) -> tuple[float, str]:
    try:
        mtime = root.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, root.name)


def _game_eval_data_roots() -> list[Path]:
    out_root = PROJECT_ROOT / "out"
    roots = [
        path
        for path in out_root.iterdir()
        if path.is_dir() and _game_eval_root_has_data(path)
    ] if out_root.exists() else []
    if DEFAULT_GAME_EVAL_ROOT.exists() and DEFAULT_GAME_EVAL_ROOT not in roots and _game_eval_root_has_data(DEFAULT_GAME_EVAL_ROOT):
        roots.append(DEFAULT_GAME_EVAL_ROOT)
    return sorted(set(roots), key=_game_eval_root_sort_key, reverse=True)


def _game_eval_runs(selected_root: Path | None = None) -> list[dict[str, object]]:
    roots = _game_eval_data_roots()
    selected = (selected_root or _default_game_eval_root()).resolve()
    runs: list[dict[str, object]] = []
    for root in roots:
        runs.append(
            {
                "id": root.name,
                "label": root.name,
                "selected": root.resolve() == selected,
                "url": f"/eval/game?run={quote(root.name, safe='')}",
            }
        )
    return runs


def _is_valid_game_eval_run_id(run_id: str) -> bool:
    return bool(run_id) and "/" not in run_id and "\\" not in run_id and not run_id.startswith(".")


def _validate_game_eval_run_id(run_id: str) -> None:
    if not _is_valid_game_eval_run_id(run_id):
        raise HTTPException(status_code=404, detail="Game eval run not found.")


def _game_eval_run_path(run_id: str) -> Path:
    _validate_game_eval_run_id(run_id)
    root = (PROJECT_ROOT / "out" / run_id).resolve()
    if not _is_relative_to(root, (PROJECT_ROOT / "out").resolve()):
        raise HTTPException(status_code=404, detail="Game eval run not found.")
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Game eval run not found.")
    return root


def _next_game_eval_run_id(prefix: str = LOCAL_OWNERSHIP_EVAL_PREFIX) -> str:
    out_root = PROJECT_ROOT / "out"
    stamp = datetime.now().strftime("%Y%m%d")
    version_re = re.compile(rf"^{re.escape(prefix)}{stamp}_v(\d+)$")
    versions = []
    for path in out_root.glob(f"{prefix}{stamp}_v*"):
        match = version_re.match(path.name)
        if match:
            versions.append(int(match.group(1)))
    version = max(versions, default=0) + 1
    return f"{prefix}{stamp}_v{version:03d}"


def _game_eval_expected_case_count() -> int:
    total = _game_eval_case_count()
    return total if total > 0 else FALLBACK_GAME_EVAL_EXPECTED_TOTAL


def _game_eval_batch_progress(
    root: Path,
    report_path: Path | None,
    *,
    prefer_report_total: bool = False,
    expected_total: int | None = None,
) -> dict[str, object]:
    total = int(expected_total) if expected_total is not None and expected_total > 0 else _game_eval_expected_case_count()
    completed = 0
    ok = 0
    errors = 0
    if report_path is not None:
        report = _load_json(report_path)
        rows = report.get("rows") if isinstance(report, dict) else None
        if isinstance(rows, list):
            completed = len(rows)
            ok = sum(1 for row in rows if isinstance(row, dict) and row.get("status") == "ok")
            errors = sum(1 for row in rows if isinstance(row, dict) and row.get("status") == "error")
        runs = report.get("runs") if isinstance(report, dict) else None
        if isinstance(runs, list):
            completed = len(runs)
            errors = sum(1 for row in runs if isinstance(row, dict) and row.get("status") == "error")
            ok = completed - errors
        if isinstance(report, dict):
            try:
                if isinstance(runs, list) and isinstance(report.get("run_count"), int):
                    report_total = int(report["run_count"])
                else:
                    report_total = int(report.get("case_count", 0))
                total = report_total if prefer_report_total and report_total > 0 else max(total, report_total)
            except (TypeError, ValueError):
                pass
    else:
        # The batch script writes per-case summaries immediately, while the
        # final eval_report.json appears only after every selected input
        # finishes. Counting these partial summaries keeps the UI visibly
        # alive during long local matting/ownership runs.
        for summary_path in _game_eval_partial_summary_paths(root):
            summary = _load_json(summary_path)
            if not isinstance(summary, dict):
                continue
            completed += 1
            if summary.get("status", "ok") == "error":
                errors += 1
            else:
                ok += 1
    percent = 0 if total <= 0 else round(min(100.0, completed * 100.0 / total), 1)
    return {
        "completed": completed,
        "total": total,
        "ok": ok,
        "errors": errors,
        "percent": percent,
        "reportPath": str(report_path.relative_to(PROJECT_ROOT)) if report_path is not None else None,
    }


def _game_eval_status_report_path(root: Path) -> Path | None:
    return (
        _game_report_path(root)
        or _solid_graphic_summary_path(root)
        or _remote_backend_summary_path(root)
        or _route_analyze_summary_path(root)
        or _game_matte_summary_path(root)
    )


def _game_eval_batch_status(run_id: str) -> dict[str, object]:
    root = _game_eval_run_path(run_id)
    report_path = _game_eval_status_report_path(root)
    with _GAME_EVAL_JOBS_LOCK:
        job = _GAME_EVAL_JOBS.get(run_id)
    process = job.get("process") if isinstance(job, dict) else None
    running = isinstance(process, subprocess.Popen) and process.poll() is None
    returncode = process.poll() if isinstance(process, subprocess.Popen) and not running else None
    if running:
        status = "running"
    elif report_path is not None:
        status = "complete"
    elif returncode not in (None, 0):
        status = "error"
    else:
        status = "started" if root.exists() else "unknown"
    expected_total = job.get("expected_total") if isinstance(job, dict) else None
    progress = _game_eval_batch_progress(
        root,
        report_path,
        prefer_report_total=not running,
        expected_total=int(expected_total) if isinstance(expected_total, int) else None,
    )
    return {
        "runId": run_id,
        "status": status,
        "returnCode": returncode,
        "url": f"/eval/game?run={quote(run_id, safe='')}",
        "statusUrl": f"/eval/game/run/{quote(run_id, safe='')}/status",
        "hasReport": report_path is not None,
        "progress": progress,
    }


def _selected_game_eval_sample_ids(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    raw = payload.get("sample_ids")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="sample_ids must be a list.")
    sample_ids = [str(item).strip() for item in raw if str(item).strip()]
    known = {str(item["sampleId"]) for item in _game_eval_samples()}
    if known:
        invalid = sorted(set(sample_ids) - known)
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unknown sample_id: {', '.join(invalid)}")
    elif any(not re.fullmatch(r"[A-Z]\d{3}|G\d{2}", sample_id) for sample_id in sample_ids):
        raise HTTPException(status_code=400, detail="sample_ids must look like B001.")
    deduped: list[str] = []
    for sample_id in sample_ids:
        if sample_id not in deduped:
            deduped.append(sample_id)
    if raw and not deduped:
        raise HTTPException(status_code=400, detail="Select at least one sample.")
    return deduped


def _selected_game_eval_test_path(payload: dict[str, Any] | None) -> str:
    if not payload:
        return DEFAULT_GAME_EVAL_TEST_PATH
    raw = payload.get("test_path", payload.get("path", payload.get("backend")))
    if raw is None:
        return DEFAULT_GAME_EVAL_TEST_PATH
    selected = str(raw).strip().lower()
    backend_to_path = {
        str(config["backend"]): path_key
        for path_key, config in GAME_EVAL_TEST_PATHS.items()
    }
    selected = backend_to_path.get(selected, selected)
    if selected not in GAME_EVAL_TEST_PATHS:
        raise HTTPException(status_code=400, detail=f"Unknown test_path: {raw}")
    return selected


def _start_game_eval_batch(
    sample_ids: list[str] | None = None,
    test_path: str = DEFAULT_GAME_EVAL_TEST_PATH,
) -> dict[str, object]:
    selected_sample_ids = list(sample_ids or [])
    path_config = GAME_EVAL_TEST_PATHS.get(test_path, GAME_EVAL_TEST_PATHS[DEFAULT_GAME_EVAL_TEST_PATH])
    backend = str(path_config["backend"])
    run_id = _next_game_eval_run_id(str(path_config["prefix"]))
    out_dir = PROJECT_ROOT / "out" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "web_batch.log"
    script_path = PROJECT_ROOT / "scripts" / "run_corridorkey_game_eval.py"
    command = [
        sys.executable,
        str(script_path),
        "--out-dir",
        str(out_dir),
        "--backend",
        backend,
    ]
    if selected_sample_ids:
        command.extend(["--sample-id", ",".join(selected_sample_ids)])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch = {
        "run_id": run_id,
        "command": command,
        "log": str(log_path.relative_to(PROJECT_ROOT)),
        "pid": process.pid,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "test_path": test_path,
        "test_path_label": path_config["label"],
        "sample_ids": selected_sample_ids,
    }
    (out_dir / "web_launch.json").write_text(json.dumps(launch, indent=2, ensure_ascii=False), encoding="utf-8")
    with _GAME_EVAL_JOBS_LOCK:
        _GAME_EVAL_JOBS[run_id] = {
            "process": process,
            "log": log_path,
            "backend": backend,
            "test_path": test_path,
            "sample_ids": selected_sample_ids,
            "expected_total": (
                _game_eval_case_count(sample_ids=selected_sample_ids)
                if selected_sample_ids
                else _game_eval_expected_case_count()
            ),
        }
    return _game_eval_batch_status(run_id)


def _default_game_eval_root() -> Path:
    roots = _game_eval_data_roots()
    if roots:
        return roots[0]
    return DEFAULT_GAME_EVAL_ROOT


def _game_eval_root(run: str | None = None) -> Path:
    if not run:
        root = _default_game_eval_root()
        if root.is_dir() and _game_eval_root_has_data(root):
            return root
        raise HTTPException(status_code=404, detail="Game eval run not found.")
    root = _game_eval_run_path(run)
    if not root.is_dir() or not _game_eval_root_has_data(root):
        raise HTTPException(status_code=404, detail="Game eval run not found.")
    return root


def _game_matte_summary_path(root: Path) -> Path | None:
    matte_root = root / "matte"
    candidates = [
        matte_root / "summary_shadow_rerun.json",
        matte_root / "summary.json",
    ]
    candidates.extend(sorted(matte_root.glob("summary*.json")))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _game_report_rows(root: Path = DEFAULT_GAME_EVAL_ROOT) -> list[dict[str, object]]:
    report_path = _game_report_path(root)
    if report_path is None:
        raise HTTPException(status_code=404, detail="Game eval report not found.")
    report = _load_json(report_path)
    if not isinstance(report, dict):
        raise HTTPException(status_code=500, detail="Game eval report must be a JSON object.")

    rows = report.get("rows")
    if not isinstance(rows, list):
        raise HTTPException(status_code=500, detail="Game eval report is missing rows.")
    return [row for row in rows if isinstance(row, dict)]


def _game_case_out_dir(row: dict[str, object], root: Path = DEFAULT_GAME_EVAL_ROOT) -> Path:
    case_id = str(row.get("case_id", "unknown"))
    out_dir_value = row.get("out_dir")
    if isinstance(out_dir_value, str):
        return _resolve_project_path(out_dir_value)
    screen = row.get("sample_screen")
    if isinstance(screen, str):
        return _game_vlm_root(root) / case_id / screen
    return _game_vlm_root(root) / case_id


def _case_matte_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    for value in (summary.get("rgba"), summary.get("matte"), summary.get("output")):
        if isinstance(value, str):
            url = _image_url(value)
            if url:
                return url
    for name in (f"{sample_screen}_rgba.png", "rgba.png"):
        url = _image_url(out_dir / name)
        if url:
            return url
    matches = sorted(out_dir.glob("*_rgba.png"))
    return _image_url(matches[0]) if matches else None


def _case_alpha_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    for value in (summary.get("alpha"), summary.get("mask")):
        if isinstance(value, str):
            url = _image_url(value)
            if url:
                return url
    for name in (f"{sample_screen}_alpha.png", "alpha.png", "mask.png"):
        url = _image_url(out_dir / name)
        if url:
            return url
    matches = sorted(out_dir.glob("*_alpha.png"))
    return _image_url(matches[0]) if matches else None


def _case_artifact_url(
    out_dir: Path,
    sample_screen: str,
    summary: dict[str, object],
    summary_keys: tuple[str, ...],
    filenames: tuple[str, ...],
) -> str | None:
    for key in summary_keys:
        value = summary.get(key)
        if isinstance(value, str):
            url = _image_url(value)
            if url:
                return url
    for name in filenames:
        url = _image_url(out_dir / name)
        if url:
            return url

    stemmed_suffixes = tuple(name.removeprefix(f"{sample_screen}_") for name in filenames)
    for suffix in stemmed_suffixes:
        matches = sorted(out_dir.glob(f"*_{suffix}"))
        if matches:
            url = _image_url(matches[0])
            if url:
                return url
    return None


def _looks_like_corridorkey(*values: object) -> bool:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            if _looks_like_corridorkey(*value):
                return True
            continue
        normalized = str(value).replace("-", "").replace("_", "").lower()
        if "corridorkey" in normalized:
            return True
    return False


def _case_mask_hint_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    return _case_artifact_url(
        out_dir,
        sample_screen,
        summary,
        ("hint", "mask_hint", "corridorkey_hint"),
        (f"{sample_screen}_corridorkey_hint.png", "corridorkey_hint.png"),
    )


def _case_corridorkey_raw_alpha_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    return _case_artifact_url(
        out_dir,
        sample_screen,
        summary,
        ("raw_alpha", "corridorkey_raw_alpha"),
        (f"{sample_screen}_corridorkey_raw_alpha.png", "corridorkey_raw_alpha.png"),
    )


def _case_foreground_url(out_dir: Path, sample_screen: str, summary: dict[str, object]) -> str | None:
    return _case_artifact_url(
        out_dir,
        sample_screen,
        summary,
        ("foreground", "corridorkey_foreground"),
        (f"{sample_screen}_foreground.png", "foreground.png"),
    )


def _sibling_image_url(path_value: object, filename: str) -> str | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = _resolve_project_path(path_value)
    return _image_url(path.with_name(filename))


def _game_region_url(root: Path, case_id: str, sample_screen: str | None = None) -> str:
    base = f"/eval/game/regions/{quote(case_id, safe='')}"
    params: list[str] = []
    if sample_screen:
        params.append(f"screen={quote(sample_screen, safe='')}")
    if root.resolve() == DEFAULT_GAME_EVAL_ROOT.resolve():
        return f"{base}?{'&'.join(params)}" if params else base
    params.append(f"run={quote(root.name, safe='')}")
    return f"{base}?{'&'.join(params)}"


def _game_eval_data_from_matte_summary(root: Path) -> dict[str, object]:
    summary_path = _game_matte_summary_path(root)
    if summary_path is None:
        raise HTTPException(status_code=404, detail="Game eval summary not found.")
    payload = _load_json(summary_path)
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Game matte summary must be a JSON list.")

    sample_ids = _game_sample_ids()
    cases: list[dict[str, object]] = []
    for case_index, row in enumerate(item for item in payload if isinstance(item, dict)):
        case_id = str(row.get("id") or row.get("image") or row.get("case_id") or f"case_{case_index + 1}")
        sample_id = sample_ids.get(case_id, f"G{case_index + 1:02d}")
        out_dir = _resolve_project_path(str(row.get("out_dir"))) if isinstance(row.get("out_dir"), str) else root / "matte" / case_id
        input_path = row.get("input")
        active_screen = _sample_screen_from_path(input_path) or "green"
        sample_paths = _game_sample_paths(case_id)
        shadow_detected = bool(row.get("shadow_detected", False))
        shadow_pixels = int(row.get("shadow_pixels", 0) or 0)
        strategy = str(row.get("strategy", ""))
        is_corridorkey = _looks_like_corridorkey(strategy, row.get("backend"), row.get("execution_backend"))
        matte_url = _case_matte_url(out_dir, active_screen, row)
        alpha_url = _case_alpha_url(out_dir, active_screen, row)
        mask_hint_url = _case_mask_hint_url(out_dir, active_screen, row)
        raw_alpha_url = _case_corridorkey_raw_alpha_url(out_dir, active_screen, row)
        foreground_url = _case_foreground_url(out_dir, active_screen, row)
        candidate = {
            "id": "matte",
            "label": "matte result",
            "selected": True,
            "tools": [strategy] if strategy else [],
            "reason": f"shadow={shadow_detected}, pixels={shadow_pixels}",
            "url": matte_url,
        }

        for sample_screen, sample_path in sample_paths.items():
            is_active_run = sample_screen == active_screen
            sample_code = f"{sample_id}-{sample_screen[:1].upper()}"
            cases.append(
                {
                    "caseId": case_id,
                    "sampleId": sample_id,
                    "sampleCode": sample_code,
                    "sampleScreen": sample_screen,
                    "isCorridorKey": is_corridorkey if is_active_run else False,
                    "runStatus": "ran" if is_active_run else "not-run",
                    "category": "matte-rerun" if is_active_run else "",
                    "verdict": strategy if is_active_run else "not-run",
                    "expectedHit": shadow_detected if is_active_run else False,
                    "regionCount": shadow_pixels if is_active_run else 0,
                    "counts": {"shadow_pixels": shadow_pixels} if is_active_run else {},
                    "selectedTools": [strategy] if is_active_run and strategy else [],
                    "primaryAmbiguity": f"shadow mean={float(row.get('shadow_mean_alpha', 0.0) or 0.0):.3f}, p95={float(row.get('shadow_p95_alpha', 0.0) or 0.0):.3f}" if is_active_run else "",
                    "originalUrl": _image_url(sample_path),
                    "regionsUrl": None,
                    "alphaUrl": alpha_url if is_active_run else None,
                    "matteUrl": matte_url if is_active_run else None,
                    "maskHintUrl": mask_hint_url if is_active_run else None,
                    "corridorkeyRawAlphaUrl": raw_alpha_url if is_active_run else None,
                    "corridorkeyForegroundUrl": foreground_url if is_active_run else None,
                    "candidates": [candidate] if is_active_run and matte_url else [],
                }
            )

    return {
        "runId": root.name,
        "model": "matte rerun",
        "success": f"{sum(1 for item in cases if item.get('runStatus') == 'ran')}/{len(payload)}",
        "expectedHit": f"{sum(1 for item in payload if isinstance(item, dict) and item.get('shadow_detected'))}/{len(payload)}",
        "sampleRows": len(cases),
        "reportPath": str(summary_path.relative_to(PROJECT_ROOT)),
        "matteRoot": str((root / "matte").relative_to(PROJECT_ROOT)),
        "vlmRoot": "",
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _game_eval_data_from_partial_summaries(root: Path) -> dict[str, object]:
    summary_paths = _game_eval_partial_summary_paths(root)
    if not summary_paths:
        raise HTTPException(status_code=404, detail="Game eval summary not found.")

    sample_ids = _game_sample_ids()
    cases: list[dict[str, object]] = []
    ok_count = 0
    expected_role_hit_count = 0
    expected_role_required_count = 0
    for summary_path in summary_paths:
        row = _load_json(summary_path)
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or summary_path.parents[1].name)
        sample_screen = str(row.get("sample_screen") or summary_path.parent.name)
        sample_id = str(row.get("sample_id") or sample_ids.get(case_id, case_id))
        sample_paths = _game_sample_paths(case_id)
        sample_path = sample_paths.get(sample_screen, "")
        status = str(row.get("status", "ok"))
        is_ok = status != "error"
        is_corridorkey = _looks_like_corridorkey(row.get("backend"), row.get("execution_backend"), row.get("selected_tools"))
        ok_count += 1 if is_ok else 0
        if row.get("expected_role_hit") is not None:
            expected_role_required_count += 1
            expected_role_hit_count += 1 if row.get("expected_role_hit") is True else 0

        top_roles = [role for role in row.get("top_roles", []) if isinstance(role, str)]
        role_counts = row.get("role_counts") if isinstance(row.get("role_counts"), dict) else {}
        role_summary = ", ".join(
            f"{role}={count}"
            for role, count in sorted(role_counts.items())
            if isinstance(role, str)
        )
        preview_path = row.get("protected_rgba") or row.get("rgba")
        alpha_url = _sibling_image_url(preview_path, "alpha.png")
        artifact_dir = _resolve_project_path(preview_path).parent if isinstance(preview_path, str) and preview_path else summary_path.parent
        mask_hint_url = _case_mask_hint_url(artifact_dir, sample_screen, row)
        raw_alpha_url = _case_corridorkey_raw_alpha_url(artifact_dir, sample_screen, row)
        foreground_url = _case_foreground_url(artifact_dir, sample_screen, row)
        candidates = []
        if is_ok and preview_path:
            candidates.append(
                {
                    "id": "local_ownership",
                    "label": "local ownership",
                    "selected": True,
                    "tools": top_roles[:8],
                    "reason": role_summary or "Local ownership ranking.",
                    "url": _image_url(preview_path),
                }
            )

        cases.append(
            {
                "caseId": case_id,
                "sampleId": sample_id,
                "sampleCode": str(row.get("sample_code") or f"{sample_id}-{sample_screen[:1].upper()}"),
                "sampleScreen": sample_screen,
                "isCorridorKey": is_corridorkey if is_ok else False,
                "runStatus": "ran" if is_ok else "error",
                "category": row.get("category", ""),
                "verdict": row.get("diagnosis_verdict", status),
                "expectedHit": bool(row.get("expected_role_hit")) if is_ok else False,
                "expectedAnyHit": bool(row.get("expected_role_hit")) if is_ok else False,
                "harmfulToolSelected": False,
                "harmfulTools": [],
                "regionCount": row.get("region_count", 0) if is_ok else 0,
                "counts": row.get("role_mask_pixels", {}) if is_ok else {},
                "selectedTools": top_roles if is_ok else [],
                "primaryAmbiguity": row.get("expected_role", row.get("error", "")),
                "originalUrl": _image_url(sample_path),
                "regionsUrl": _game_region_url(root, case_id, sample_screen) if is_ok else None,
                "alphaUrl": alpha_url if is_ok else None,
                "matteUrl": _image_url(preview_path) if is_ok else None,
                "maskHintUrl": mask_hint_url if is_ok else None,
                "corridorkeyRawAlphaUrl": raw_alpha_url if is_ok else None,
                "corridorkeyForegroundUrl": foreground_url if is_ok else None,
                "candidates": candidates,
            }
        )

    progress = _game_eval_batch_progress(root, None)
    return {
        "runId": root.name,
        "model": "local ownership (running)",
        "success": f"{ok_count}/{progress['total']}",
        "expectedHit": f"{expected_role_hit_count}/{expected_role_required_count}",
        "expectedAnyHit": f"{expected_role_hit_count}/{expected_role_required_count}",
        "harmfulTools": f"0/{len(cases)}",
        "sampleRows": len(cases),
        "reportPath": None,
        "matteRoot": str((root / "matte").relative_to(PROJECT_ROOT)),
        "vlmRoot": str((root / "local_ownership").relative_to(PROJECT_ROOT)),
        "runs": _game_eval_runs(root),
        "selectedRun": root.name,
        "progress": progress,
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _solid_graphic_artifact_url(branch: dict[str, object], field: str) -> str | None:
    value = branch.get(field)
    if not isinstance(value, str):
        return None
    path = Path(value)
    if not path.is_absolute() and len(path.parts) == 1 and isinstance(branch.get("dir"), str):
        path = Path(str(branch["dir"])) / path
    return _image_url(path)


def _solid_graphic_diff_url(root: Path, row: dict[str, object]) -> str | None:
    for branch_name in ("new", "old"):
        branch = row.get(branch_name)
        if isinstance(branch, dict) and isinstance(branch.get("dir"), str):
            candidate = _resolve_project_path(str(branch["dir"])).parent / "alpha_abs_diff.png"
            if candidate.exists():
                return _image_url(candidate)
    sample_id = str(row.get("sample_id", ""))
    case_id = str(row.get("case_id", ""))
    screen = str(row.get("screen", ""))
    if sample_id and case_id and screen:
        return _image_url(root / f"{sample_id}_{case_id}_{screen}" / "alpha_abs_diff.png")
    return None


def _solid_graphic_candidate_reason(branch: dict[str, object]) -> str:
    parts: list[str] = []
    if isinstance(branch.get("solid_confidence"), (int, float)):
        parts.append(f"confidence={float(branch['solid_confidence']):.3f}")
    if isinstance(branch.get("alpha_mean"), (int, float)):
        parts.append(f"alpha_mean={float(branch['alpha_mean']):.3f}")
    if isinstance(branch.get("alpha_soft_fraction"), (int, float)):
        parts.append(f"soft={float(branch['alpha_soft_fraction']):.3f}")
    if isinstance(branch.get("elapsed_sec"), (int, float)):
        parts.append(f"{float(branch['elapsed_sec']):.2f}s")
    return ", ".join(parts)


def _solid_graphic_diff_reason(diff: dict[str, object]) -> str:
    parts: list[str] = []
    labels = (
        ("mean_abs", "mean"),
        ("p95_abs", "p95"),
        ("max_abs", "max"),
        ("gt_05_fraction", ">0.05"),
        ("gt_25_fraction", ">0.25"),
    )
    for key, label in labels:
        value = diff.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{label}={float(value):.3f}")
    return ", ".join(parts)


def _game_eval_data_from_solid_graphic_summary(root: Path, summary_path: Path) -> dict[str, object]:
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Solid graphic summary must be a JSON object.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise HTTPException(status_code=500, detail="Solid graphic summary is missing rows.")

    cases: list[dict[str, object]] = []
    ok_count = 0
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "ok"))
        is_ok = status != "error"
        ok_count += 1 if is_ok else 0
        case_id = str(item.get("case_id") or f"case_{index:02d}")
        sample_id = str(item.get("sample_id") or f"G{index:02d}")
        screen = str(item.get("screen") or _sample_screen_from_path(item.get("input")) or "green")
        sample_code = f"{sample_id}-{screen[:1].upper()}"
        new_branch = item.get("new") if isinstance(item.get("new"), dict) else {}
        old_branch = item.get("old") if isinstance(item.get("old"), dict) else {}
        if not new_branch and isinstance(item.get("outputs"), dict):
            outputs = item["outputs"]
            alpha_stats = item.get("alpha") if isinstance(item.get("alpha"), dict) else {}
            solid_graphic = item.get("solid_graphic") if isinstance(item.get("solid_graphic"), dict) else {}
            new_branch = {
                "strategy": item.get("strategy", "solid_graphic"),
                "solid_confidence": solid_graphic.get("confidence"),
                "alpha_mean": alpha_stats.get("mean"),
                "alpha_soft_fraction": alpha_stats.get("soft_fraction"),
                "elapsed_sec": item.get("elapsed_sec"),
                "dir": outputs.get("case_dir"),
                "rgba": outputs.get("rgba"),
                "ownership_counts": item.get("ownership_counts", {}),
            }
        diff = item.get("alpha_diff") if isinstance(item.get("alpha_diff"), dict) else {}
        new_strategy = str(new_branch.get("strategy", "solid_graphic"))
        old_strategy = str(old_branch.get("strategy", "fallback"))

        candidates: list[dict[str, object]] = []
        new_url = _solid_graphic_artifact_url(new_branch, "rgba")
        new_alpha_url = _solid_graphic_artifact_url(new_branch, "alpha")
        new_foreground_url = _solid_graphic_artifact_url(new_branch, "foreground")
        new_mask_hint_url = _solid_graphic_artifact_url(new_branch, "hint")
        new_raw_alpha_url = _solid_graphic_artifact_url(new_branch, "raw_alpha")
        if new_url:
            candidates.append(
                {
                    "id": "new_solid_graphic" if old_branch else "solid_graphic",
                    "label": f"new {new_strategy}" if old_branch else new_strategy,
                    "selected": True,
                    "tools": [new_strategy],
                    "reason": _solid_graphic_candidate_reason(new_branch),
                    "url": new_url,
                }
            )
        old_url = _solid_graphic_artifact_url(old_branch, "rgba")
        if old_url:
            candidates.append(
                {
                    "id": "old_fallback",
                    "label": "old fallback",
                    "selected": False,
                    "tools": [old_strategy],
                    "reason": _solid_graphic_candidate_reason(old_branch),
                    "url": old_url,
                }
            )
        diff_url = _solid_graphic_diff_url(root, item)
        if diff_url:
            candidates.append(
                {
                    "id": "alpha_abs_diff",
                    "label": "alpha diff",
                    "selected": False,
                    "tools": ["alpha_abs_diff"],
                    "reason": _solid_graphic_diff_reason(diff),
                    "url": diff_url,
                }
            )

        ownership_counts = new_branch.get("ownership_counts")
        if not isinstance(ownership_counts, dict):
            ownership_counts = {}
        verdict = f"{new_strategy} vs {old_strategy}" if old_branch else new_strategy
        primary = str(item.get("primary_ambiguity", ""))
        diff_reason = _solid_graphic_diff_reason(diff)
        if diff_reason:
            primary = f"{primary} · diff {diff_reason}" if primary else f"diff {diff_reason}"

        cases.append(
            {
                "caseId": case_id,
                "sampleId": sample_id,
                "sampleCode": sample_code,
                "sampleScreen": screen,
                "isCorridorKey": False,
                "runStatus": "ran" if is_ok else "error",
                "category": "solid-graphic-compare",
                "verdict": verdict if is_ok else status,
                "expectedHit": is_ok,
                "expectedAnyHit": is_ok,
                "harmfulToolSelected": False,
                "harmfulTools": [],
                "regionCount": sum(int(value) for value in ownership_counts.values() if isinstance(value, int)),
                "counts": ownership_counts,
                "selectedTools": [new_strategy] if is_ok else [],
                "primaryAmbiguity": primary,
                "originalUrl": _image_url(item.get("input")),
                "regionsUrl": None,
                "alphaUrl": new_alpha_url,
                "matteUrl": new_url,
                "maskHintUrl": new_mask_hint_url,
                "corridorkeyRawAlphaUrl": new_raw_alpha_url,
                "corridorkeyForegroundUrl": new_foreground_url,
                "candidates": candidates if is_ok else [],
            }
        )

    case_count = int(payload.get("case_count", len(rows)) or len(rows))
    progress = _game_eval_batch_progress(root, summary_path, prefer_report_total=True)
    return {
        "runId": root.name,
        "model": "solid graphic comparison",
        "success": f"{ok_count}/{case_count}",
        "expectedHit": "n/a",
        "expectedAnyHit": "n/a",
        "harmfulTools": "0/0",
        "sampleRows": len(cases),
        "reportPath": str(summary_path.relative_to(PROJECT_ROOT)),
        "matteRoot": str(root.relative_to(PROJECT_ROOT)),
        "vlmRoot": GAME_SAMPLE_REL.as_posix(),
        "runs": _game_eval_runs(root),
        "selectedRun": root.name,
        "progress": progress,
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _case_id_from_comfy_run(item: dict[str, object], index: int) -> tuple[str, str, str]:
    metadata = item.get("case_metadata") if isinstance(item.get("case_metadata"), dict) else {}
    input_path = item.get("input")
    item_screen = item.get("sample_screen")
    screen = str(item_screen) if isinstance(item_screen, str) and item_screen else (_sample_screen_from_path(input_path) or "green")
    sample_id = str(metadata.get("sample_id") or "")
    case_id = str(metadata.get("id") or "")
    case_label = str(item.get("case") or "")
    if not sample_id and case_label:
        parts = case_label.split("_")
        if parts and re.fullmatch(r"[A-Z]\d{3}|G\d{2}", parts[0]):
            sample_id = parts[0]
    if not case_id and isinstance(input_path, str):
        try:
            case_id = Path(input_path).parent.name
        except Exception:
            case_id = ""
    if not sample_id:
        sample_id = f"S{index:03d}"
    if not case_id:
        case_id = case_label or f"case_{index:02d}"
    return sample_id, case_id, screen


def _game_eval_data_from_comfy_ermbg_summary(root: Path, summary_path: Path) -> dict[str, object]:
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Comfy ERMBG summary must be a JSON object.")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise HTTPException(status_code=500, detail="Comfy ERMBG summary is missing runs.")

    cases: list[dict[str, object]] = []
    ok_count = 0
    for index, item in enumerate((run for run in runs if isinstance(run, dict)), start=1):
        status = str(item.get("status", "ok"))
        is_ok = status != "error"
        ok_count += 1 if is_ok else 0
        sample_id, case_id, screen = _case_id_from_comfy_run(item, index)
        metadata = item.get("case_metadata") if isinstance(item.get("case_metadata"), dict) else {}
        outputs = item.get("outputs") if isinstance(item.get("outputs"), dict) else None
        if outputs is None:
            outputs = item.get("output") if isinstance(item.get("output"), dict) else {}
        metrics = item.get("quality_metrics") if isinstance(item.get("quality_metrics"), dict) else {}
        remote_debug = item.get("remote_debug") if isinstance(item.get("remote_debug"), dict) else {}
        direct_worker_payload = remote_debug.get("direct_worker") if isinstance(remote_debug.get("direct_worker"), dict) else {}
        timings = remote_debug.get("timings") if isinstance(remote_debug.get("timings"), dict) else {}
        strategy = str(item.get("backend") or "auto")
        is_corridorkey = _looks_like_corridorkey(
            strategy,
            item.get("requested_backend"),
            item.get("execution_backend"),
            direct_worker_payload.get("algorithm"),
            direct_worker_payload.get("execution_backend"),
            direct_worker_payload.get("route"),
            remote_debug.get("auto_route"),
        )
        elapsed = item.get("elapsed_sec_client")
        alpha_mean = metrics.get("alpha_mean")
        alpha_pixels = metrics.get("alpha_nonzero_pixels")
        reason_parts = []
        if isinstance(elapsed, (int, float)):
            reason_parts.append(f"{float(elapsed):.1f}s client")
        if isinstance(timings.get("total_sec"), (int, float)):
            reason_parts.append(f"{float(timings['total_sec']):.1f}s server")
        if isinstance(alpha_mean, (int, float)):
            reason_parts.append(f"alpha_mean={float(alpha_mean):.3f}")
        if isinstance(alpha_pixels, int):
            reason_parts.append(f"alpha_px={alpha_pixels}")
        candidate_url = _image_url(outputs.get("rgba"))
        alpha_url = _image_url(outputs.get("alpha")) or _sibling_image_url(outputs.get("rgba"), "alpha.png")
        trimap_url = _image_url(outputs.get("trimap")) or _sibling_image_url(outputs.get("rgba"), "trimap.png")
        mask_hint_url = _image_url(outputs.get("hint")) or _image_url(outputs.get("corridorkey_hint")) or _sibling_image_url(outputs.get("rgba"), "corridorkey_hint.png")
        raw_alpha_url = _image_url(outputs.get("raw_alpha")) or _image_url(outputs.get("corridorkey_raw_alpha")) or _sibling_image_url(outputs.get("rgba"), "corridorkey_raw_alpha.png")
        foreground_url = _image_url(outputs.get("foreground")) or _sibling_image_url(outputs.get("rgba"), "foreground.png")
        candidates = []
        if is_ok and candidate_url:
            candidates.append(
                {
                    "id": strategy.replace("-", "_"),
                    "label": strategy,
                    "selected": True,
                    "tools": [strategy],
                    "reason": ", ".join(reason_parts),
                    "url": candidate_url,
                }
            )
        sample_paths = _game_sample_paths(case_id)
        sample_path = sample_paths.get(screen) or str(item.get("input", ""))
        cases.append(
            {
                "caseId": case_id,
                "sampleId": sample_id,
                "sampleCode": f"{sample_id}-{screen[:1].upper()}",
                "sampleScreen": screen,
                "isCorridorKey": is_corridorkey if is_ok else False,
                "runStatus": "ran" if is_ok else "error",
                "category": metadata.get("category", strategy),
                "verdict": strategy if is_ok else status,
                "expectedHit": is_ok,
                "expectedAnyHit": is_ok,
                "harmfulToolSelected": False,
                "harmfulTools": [],
                "regionCount": int(alpha_pixels) if isinstance(alpha_pixels, int) else 0,
                "counts": {"alpha_nonzero_pixels": alpha_pixels, "alpha_mean": alpha_mean},
                "selectedTools": [strategy] if is_ok else [],
                "primaryAmbiguity": metadata.get("primary_ambiguity", ""),
                "originalUrl": _image_url(sample_path),
                "regionsUrl": None,
                "alphaUrl": alpha_url if is_ok else None,
                "trimapUrl": trimap_url if is_ok else None,
                "matteUrl": candidate_url if is_ok else None,
                "maskHintUrl": mask_hint_url if is_ok else None,
                "corridorkeyRawAlphaUrl": raw_alpha_url if is_ok else None,
                "corridorkeyForegroundUrl": foreground_url if is_ok else None,
                "candidates": candidates,
            }
        )

    progress = _game_eval_batch_progress(root, summary_path, prefer_report_total=True)
    return {
        "runId": root.name,
        "model": f"{str(payload.get('backend') or (runs[0].get('backend') if runs and isinstance(runs[0], dict) else 'auto'))} remote",
        "success": f"{ok_count}/{len(runs)}",
        "expectedHit": f"{ok_count}/{len(runs)}",
        "expectedAnyHit": f"{ok_count}/{len(runs)}",
        "harmfulTools": f"0/{len(runs)}",
        "sampleRows": len(cases),
        "reportPath": str(summary_path.relative_to(PROJECT_ROOT)),
        "matteRoot": str(root.relative_to(PROJECT_ROOT)),
        "vlmRoot": GAME_SAMPLE_REL.as_posix(),
        "runs": _game_eval_runs(root),
        "selectedRun": root.name,
        "progress": progress,
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _game_eval_data_from_route_analyze_summary(root: Path, summary_path: Path) -> dict[str, object]:
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Route analyze summary must be a JSON object.")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise HTTPException(status_code=500, detail="Route analyze summary is missing runs.")

    cases: list[dict[str, object]] = []
    ok_count = 0
    for index, item in enumerate((run for run in runs if isinstance(run, dict)), start=1):
        status = str(item.get("status", "ok"))
        is_ok = status != "error"
        ok_count += 1 if is_ok else 0
        sample_id = str(item.get("sample_id") or f"S{index:03d}")
        case_id = str(item.get("id") or item.get("case") or f"case_{index:03d}")
        input_value = item.get("input")
        sample_screen = _sample_screen_from_path(input_value) or str(item.get("screen") or "green")
        default_algorithm = str(item.get("default_algorithm") or "")
        parameter_profile = str(item.get("parameter_profile") or "")
        route_candidates = [str(value) for value in item.get("route_candidate_algorithms", []) if isinstance(value, str)]
        semantic_candidates = [
            str(value)
            for value in item.get("semantic_candidate_ids", [])
            if isinstance(value, str)
        ]
        ambiguity_types = [
            str(value)
            for value in item.get("ambiguity_types", [])
            if isinstance(value, str)
        ]
        verdict_parts = [default_algorithm]
        if parameter_profile:
            verdict_parts.append(parameter_profile)
        if status == "mismatch":
            verdict_parts.append("mismatch")
        reason_parts = []
        if route_candidates:
            reason_parts.append("routes=" + ",".join(route_candidates))
        if ambiguity_types:
            reason_parts.append("ambiguity=" + ",".join(ambiguity_types))
        if item.get("default_candidate_id"):
            reason_parts.append(f"default={item.get('default_candidate_id')}")
        candidates = [
            {
                "id": "route_analyze_default",
                "label": default_algorithm or "route analyze",
                "selected": True,
                "tools": route_candidates or [default_algorithm],
                "reason": "; ".join(reason_parts),
                "url": None,
            }
        ]
        if semantic_candidates:
            candidates.append(
                {
                    "id": "semantic_candidates",
                    "label": f"{len(semantic_candidates)} semantic candidates",
                    "selected": False,
                    "tools": semantic_candidates[:8],
                    "reason": ", ".join(semantic_candidates[:12]),
                    "url": None,
                }
            )
        cases.append(
            {
                "caseId": case_id,
                "sampleId": sample_id,
                "sampleCode": f"{sample_id}-{sample_screen[:1].upper()}",
                "sampleScreen": sample_screen,
                "isCorridorKey": False,
                "runStatus": "ran" if is_ok else "error",
                "category": item.get("category", "route-analyze"),
                "verdict": " / ".join(part for part in verdict_parts if part),
                "expectedHit": bool(item.get("expected_present", is_ok)),
                "expectedAnyHit": bool(item.get("expected_present", is_ok)),
                "harmfulToolSelected": False,
                "harmfulTools": [],
                "regionCount": int(item.get("ambiguity_region_count") or 0),
                "counts": {
                    "route_candidates": len(route_candidates),
                    "semantic_candidates": len(semantic_candidates),
                    "ambiguity_regions": int(item.get("ambiguity_region_count") or 0),
                },
                "selectedTools": route_candidates or [default_algorithm],
                "primaryAmbiguity": ", ".join(ambiguity_types) or str(item.get("target_route") or ""),
                "originalUrl": _image_url(input_value),
                "regionsUrl": None,
                "alphaUrl": None,
                "matteUrl": None,
                "maskHintUrl": None,
                "corridorkeyRawAlphaUrl": None,
                "corridorkeyForegroundUrl": None,
                "candidates": candidates,
            }
        )

    progress = _game_eval_batch_progress(root, summary_path, prefer_report_total=True)
    case_count = int(payload.get("case_count", len(runs)) or len(runs))
    return {
        "runId": root.name,
        "model": "route/analyze strategy",
        "success": f"{ok_count}/{case_count}",
        "expectedHit": f"{payload.get('ok_count', ok_count)}/{case_count}",
        "expectedAnyHit": f"{payload.get('ok_count', ok_count)}/{case_count}",
        "harmfulTools": f"0/{case_count}",
        "sampleRows": len(cases),
        "reportPath": str(summary_path.relative_to(PROJECT_ROOT)),
        "matteRoot": str(root.relative_to(PROJECT_ROOT)),
        "vlmRoot": GAME_SAMPLE_REL.as_posix(),
        "runs": _game_eval_runs(root),
        "selectedRun": root.name,
        "progress": progress,
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _game_eval_data(root: Path = DEFAULT_GAME_EVAL_ROOT) -> dict[str, object]:
    report_path = _game_report_path(root)
    if report_path is None:
        if _game_eval_partial_summary_paths(root):
            return _game_eval_data_from_partial_summaries(root)
        solid_path = _solid_graphic_summary_path(root)
        if solid_path is not None:
            return _game_eval_data_from_solid_graphic_summary(root, solid_path)
        comfy_path = _remote_backend_summary_path(root)
        if comfy_path is not None:
            return _game_eval_data_from_comfy_ermbg_summary(root, comfy_path)
        route_analyze_path = _route_analyze_summary_path(root)
        if route_analyze_path is not None:
            return _game_eval_data_from_route_analyze_summary(root, route_analyze_path)
        data = _game_eval_data_from_matte_summary(root)
        data["runs"] = _game_eval_runs(root)
        data["selectedRun"] = root.name
        return data

    report = _load_json(report_path)
    if not isinstance(report, dict):
        raise HTTPException(status_code=500, detail="Game eval report must be a JSON object.")

    rows = _game_report_rows(root)
    sample_ids = _game_sample_ids()

    cases: list[dict[str, object]] = []
    for case_index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id", "unknown"))
        sample_id = sample_ids.get(case_id, f"G{case_index:02d}")
        out_dir = _game_case_out_dir(row, root)
        summary_path = out_dir / "summary.json"
        summary = _load_json(summary_path) if summary_path.exists() else {}
        if not isinstance(summary, dict):
            summary = {}

        fallback_tools = [tool for tool in row.get("selected_tools", []) if isinstance(tool, str)]
        top_roles = [role for role in row.get("top_roles", []) if isinstance(role, str)]
        if not fallback_tools and top_roles:
            fallback_tools = top_roles
        candidate_results_path = out_dir / "candidate_results.json"
        candidate_results = _candidate_result_items(candidate_results_path)
        candidate_paths = [path for path in summary.get("candidate_paths", []) if isinstance(path, str)]
        if not candidate_paths:
            selected_ids = [plan_id for plan_id in row.get("selected_plan_ids", []) if isinstance(plan_id, str)]
            candidate_paths = [str(out_dir / "candidates" / f"{plan_id}.png") for plan_id in selected_ids]
        if not candidate_paths:
            candidate_paths = [str(path) for path in sorted((out_dir / "candidates").glob("*.png"))]

        candidates: list[dict[str, object]] = []
        result_by_path: dict[str, dict[str, object]] = {}
        for candidate in candidate_results:
            candidate_path = candidate.get("path")
            if isinstance(candidate_path, str):
                result_by_path[_resolve_project_path(candidate_path).as_posix()] = candidate

        for index, candidate_path in enumerate(candidate_paths):
            resolved_candidate_path = _resolve_project_path(candidate_path)
            candidate = result_by_path.get(resolved_candidate_path.as_posix(), {})
            plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}
            candidate_id = candidate.get("id") or (plan.get("id") if isinstance(plan, dict) else None) or resolved_candidate_path.stem
            label = candidate.get("label") or (plan.get("label") if isinstance(plan, dict) else None) or str(candidate_id)
            candidates.append(
                {
                    "id": str(candidate_id),
                    "label": str(label),
                    "selected": bool(candidate.get("selected", index == 0)),
                    "tools": _candidate_tools(candidate, fallback_tools),
                    "reason": str((plan.get("reason") if isinstance(plan, dict) else "") or row.get("selected_reason", "")),
                    "url": _image_url(resolved_candidate_path),
                }
            )
        if not candidates and isinstance(row.get("ownership"), list):
            role_counts = row.get("role_counts") if isinstance(row.get("role_counts"), dict) else {}
            role_summary = ", ".join(
                f"{role}={count}"
                for role, count in sorted(role_counts.items())
                if isinstance(role, str)
            )
            candidates.append(
                {
                    "id": "local_ownership",
                    "label": "local ownership",
                    "selected": True,
                    "tools": top_roles[:8],
                    "reason": role_summary or "Local local ownership ranking.",
                    "url": _image_url(row.get("protected_rgba") or row.get("rgba")),
                }
            )

        sample_paths = _game_sample_paths(case_id)
        row_screen = row.get("sample_screen")
        active_screen = (
            row_screen
            if isinstance(row_screen, str) and row_screen in sample_paths
            else _sample_screen_from_path(summary.get("input")) or "green"
        )
        screens = [active_screen] if isinstance(row_screen, str) else list(sample_paths)
        for sample_screen in screens:
            sample_path = sample_paths.get(sample_screen, str(summary.get("input", "")))
            is_active_run = sample_screen == active_screen
            sample_code = f"{sample_id}-{sample_screen[:1].upper()}"
            alpha_url = _case_alpha_url(out_dir, active_screen, summary) if is_active_run else None
            matte_url = (
                _image_url(summary.get("rgba") or row.get("protected_rgba") or row.get("rgba") or root / "matte" / case_id / "rgba.png")
                if is_active_run
                else None
            )
            is_corridorkey = _looks_like_corridorkey(summary.get("backend"), row.get("backend"), fallback_tools, top_roles)
            mask_hint_url = _case_mask_hint_url(out_dir, active_screen, summary) if is_active_run else None
            raw_alpha_url = _case_corridorkey_raw_alpha_url(out_dir, active_screen, summary) if is_active_run else None
            foreground_url = _case_foreground_url(out_dir, active_screen, summary) if is_active_run else None
            cases.append(
                {
                    "caseId": case_id,
                    "sampleId": sample_id,
                    "sampleCode": sample_code,
                    "sampleScreen": sample_screen,
                    "isCorridorKey": is_corridorkey if is_active_run else False,
                    "runStatus": "ran" if is_active_run else "not-run",
                    "category": row.get("category", ""),
                    "verdict": row.get("diagnosis_verdict", "") if is_active_run else "not-run",
                    "expectedHit": bool(row.get("expected_hit", row.get("expected_role_hit"))) if is_active_run else False,
                    "expectedAnyHit": bool(row.get("expected_any_hit", row.get("expected_hit", row.get("expected_role_hit")))) if is_active_run else False,
                    "harmfulToolSelected": bool(row.get("harmful_tool_selected")) if is_active_run else False,
                    "harmfulTools": row.get("harmful_tools", []) if is_active_run else [],
                    "regionCount": row.get("region_count", 0) if is_active_run else 0,
                    "counts": row.get("counts", {}) if is_active_run else {},
                    "selectedTools": fallback_tools if is_active_run else [],
                    "primaryAmbiguity": row.get("primary_ambiguity", row.get("expected_role", "")),
                    "originalUrl": _image_url(sample_path),
                    "regionsUrl": _game_region_url(root, case_id, sample_screen) if is_active_run else None,
                    "alphaUrl": alpha_url,
                    "matteUrl": matte_url,
                    "maskHintUrl": mask_hint_url,
                    "corridorkeyRawAlphaUrl": raw_alpha_url,
                    "corridorkeyForegroundUrl": foreground_url,
                    "candidates": candidates if is_active_run else [],
                }
            )

    return {
        "runId": report.get("run_id", root.name),
        "model": report.get("model", ""),
        "success": f"{report.get('ok_count', 0)}/{report.get('case_count', len(cases))}",
        "expectedHit": f"{report.get('expected_tool_hit_count', report.get('expected_role_hit_count', 0))}/{report.get('case_count', len(cases))}",
        "expectedAnyHit": f"{report.get('expected_any_tool_hit_count', report.get('expected_tool_hit_count', report.get('expected_role_hit_count', 0)))}/{report.get('case_count', len(cases))}",
        "harmfulTools": f"{report.get('harmful_tool_selected_count', 0)}/{report.get('case_count', len(cases))}",
        "sampleRows": len(cases),
        "reportPath": str(report_path.relative_to(PROJECT_ROOT)),
        "matteRoot": str((root / "matte").relative_to(PROJECT_ROOT)),
        "vlmRoot": str(_game_vlm_root(root).relative_to(PROJECT_ROOT)),
        "runs": _game_eval_runs(root),
        "selectedRun": root.name,
        "progress": _game_eval_batch_progress(root, report_path, prefer_report_total=True),
        "samples": _game_eval_samples(),
        "cases": cases,
    }


def _empty_game_eval_data() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    sample_ids = _game_sample_ids()
    for index, item in enumerate(_game_eval_manifest_cases(), start=1):
        if not isinstance(item.get("id"), str):
            continue
        case_id = str(item["id"])
        sample_id = sample_ids.get(case_id, f"G{index:02d}")
        sample_paths = _game_sample_paths(case_id)
        for sample_screen, sample_path in sample_paths.items():
            cases.append(
                {
                    "caseId": case_id,
                    "sampleId": sample_id,
                    "sampleCode": f"{sample_id}-{sample_screen[:1].upper()}",
                    "sampleScreen": sample_screen,
                    "isCorridorKey": False,
                    "runStatus": "not-run",
                    "category": item.get("category", ""),
                    "verdict": "not-run",
                    "expectedHit": False,
                    "expectedAnyHit": False,
                    "harmfulToolSelected": False,
                    "harmfulTools": [],
                    "regionCount": 0,
                    "counts": {},
                    "selectedTools": [],
                    "primaryAmbiguity": item.get("primary_ambiguity", ""),
                    "originalUrl": _image_url(sample_path),
                    "regionsUrl": None,
                    "alphaUrl": None,
                    "matteUrl": None,
                    "maskHintUrl": None,
                    "corridorkeyRawAlphaUrl": None,
                    "corridorkeyForegroundUrl": None,
                    "candidates": [],
                }
            )
    return {
        "runId": "game eval",
        "model": "no run selected",
        "success": "0/0",
        "expectedHit": "0/0",
        "expectedAnyHit": "0/0",
        "harmfulTools": "0/0",
        "sampleRows": len(cases),
        "reportPath": None,
        "matteRoot": "",
        "vlmRoot": GAME_SAMPLE_REL.as_posix(),
        "runs": [],
        "selectedRun": "",
        "progress": {
            "completed": 0,
            "ok": 0,
            "errors": 0,
            "total": _game_eval_expected_case_count(),
            "percent": 0.0,
        },
        "samples": _game_eval_samples(),
        "cases": cases,
    }


@app.post("/eval/game/run")
def start_game_eval_run(payload: Annotated[dict[str, Any] | None, Body()] = None) -> dict[str, object]:
    return _start_game_eval_batch(
        sample_ids=_selected_game_eval_sample_ids(payload),
        test_path=_selected_game_eval_test_path(payload),
    )


@app.get("/eval/game/run/{run_id}/status")
def game_eval_run_status(run_id: str) -> dict[str, object]:
    return _game_eval_batch_status(run_id)


@app.get("/eval/game/file/{rel_path:path}")
def game_eval_file(rel_path: str) -> FileResponse:
    path = (PROJECT_ROOT / rel_path).resolve()
    allowed_roots = [PROJECT_ROOT / "out", PROJECT_ROOT / "samples"]
    if not any(_is_relative_to(path, root.resolve()) for root in allowed_roots):
        raise HTTPException(status_code=404, detail="File is outside eval output roots.")
    if not path.exists() or not path.is_file() or path.suffix.lower() not in SERVABLE_IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path)


def _draw_region_overlay(input_path: Path, regions: list[dict[str, object]]) -> bytes:
    image = Image.open(input_path).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    line_width = max(2, min(w, h) // 220)

    for region in regions:
        bbox = region.get("bbox_xyxy")
        kind = str(region.get("kind", "unknown"))
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1:
            x2 = min(w - 1, x1 + line_width)
        if y2 <= y1:
            y2 = min(h - 1, y1 + line_width)
        outline = REGION_BOX_COLORS.get(kind, (255, 255, 255, 235))
        fill = REGION_FILL_COLORS.get(kind, (255, 255, 255, 20))
        draw.rectangle((x1, y1, x2, y2), outline=outline, fill=fill, width=line_width)

    legend_items = [
        ("same_bg", REGION_BOX_COLORS["same_bg_enclosed_region"]),
        ("alpha_diff", REGION_BOX_COLORS["alpha_keyer_disagreement"]),
        ("hard_edge", REGION_BOX_COLORS["hard_edge_candidate"]),
    ]
    legend_pad = max(8, line_width * 3)
    row_h = 18
    legend_w = 142
    legend_h = legend_pad * 2 + row_h * len(legend_items)
    draw.rounded_rectangle(
        (legend_pad, legend_pad, legend_pad + legend_w, legend_pad + legend_h),
        radius=6,
        fill=(255, 255, 255, 210),
        outline=(18, 25, 22, 90),
        width=1,
    )
    y = legend_pad * 2
    for label, color in legend_items:
        draw.rectangle((legend_pad * 2, y + 3, legend_pad * 2 + 12, y + 15), fill=color)
        draw.text((legend_pad * 2 + 18, y), label, fill=(18, 25, 22, 255))
        y += row_h

    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/eval/game/regions/{case_id}")
def game_eval_regions(
    case_id: str,
    run: str | None = Query(default=None),
    screen: str | None = Query(default=None),
) -> Response:
    root = _game_eval_root(run)
    rows = _game_report_rows(root)
    row = next(
        (
            item
            for item in rows
            if item.get("case_id") == case_id
            and (screen is None or item.get("sample_screen") == screen)
        ),
        None,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    out_dir = _game_case_out_dir(row, root)
    summary = _load_json(out_dir / "summary.json")
    if not isinstance(summary, dict):
        raise HTTPException(status_code=500, detail="Case summary must be a JSON object.")
    input_path_value = summary.get("input")
    if isinstance(input_path_value, str):
        input_path = _resolve_project_path(input_path_value)
    else:
        sample_paths = _game_sample_paths(case_id)
        input_path = _resolve_project_path(sample_paths.get(str(row.get("sample_screen", screen or "")), ""))
    if not _is_relative_to(input_path, (PROJECT_ROOT / "samples").resolve()) or not input_path.exists():
        raise HTTPException(status_code=404, detail="Case input image not found.")
    regions = _candidate_regions(_candidate_result_items(out_dir / "candidate_results.json"))
    if not regions:
        regions = [
            item["region"]
            for item in summary.get("ownership", row.get("ownership", []))
            if isinstance(item, dict) and isinstance(item.get("region"), dict)
        ]
    png = _draw_region_overlay(input_path, regions)
    return Response(content=png, media_type="image/png")


@app.get("/eval/game", response_class=HTMLResponse)
def game_eval_page(run: str | None = Query(default=None)) -> str:
    if run:
        root = _game_eval_root(run)
        data = _game_eval_data(root)
    else:
        try:
            root = _game_eval_root(None)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            data = _empty_game_eval_data()
        else:
            data = _game_eval_data(root)
    data_json = json.dumps(data, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ERMBG Game Eval</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17201c;
      background: #f4f6f3;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      padding: 10px 20px;
      border-bottom: 1px solid #d6ddd4;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(10px);
    }}
    h1 {{ flex: 0 0 auto; margin: 0; font-size: 18px; letter-spacing: 0; white-space: nowrap; }}
    nav {{ min-width: 0; display: flex; align-items: center; justify-content: flex-start; flex-wrap: wrap; gap: 10px; font-size: 13px; color: #53615a; }}
    nav a {{ color: #196f5a; font-weight: 700; text-decoration: none; white-space: nowrap; }}
    .runtime-status {{ min-width: 0; display: inline-flex; align-items: center; gap: 6px; overflow: hidden; }}
    .runtime-pill {{ display: inline-flex; align-items: center; gap: 5px; min-height: 24px; padding: 0 8px; border: 1px solid #d1d9cf; border-radius: 999px; background: #ffffff; color: #53615a; font-size: 12px; font-weight: 900; white-space: nowrap; }}
    .runtime-pill::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: #9aa59e; }}
    .runtime-pill.is-ok::before {{ background: #23855f; }}
    .runtime-pill.is-error::before {{ background: #b94a42; }}
    .runtime-pill.is-warn::before {{ background: #b57b18; }}
    #run-id {{
      min-width: 0;
      flex: 1 1 260px;
      max-width: 420px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .run-button {{
      min-height: 34px;
      padding: 0 12px;
      border: 0;
      border-radius: 6px;
      background: #176a56;
      color: #ffffff;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      white-space: nowrap;
    }}
    .run-button:disabled {{ opacity: 0.58; cursor: progress; }}
    .run-status {{ flex: 0 1 160px; min-width: 92px; color: #53615a; font-size: 12px; font-weight: 800; }}
    .run-progress {{
      flex: 1 1 160px;
      width: 128px;
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #dce4d9;
    }}
    .run-progress-bar {{
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: #176a56;
      transition: width 180ms ease;
    }}
    .run-picker {{
      min-width: 0;
      flex: 1 1 420px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #53615a;
      font-weight: 800;
    }}
    .run-picker span {{ flex: 0 0 auto; white-space: nowrap; }}
    .run-picker select {{
      min-width: 0;
      width: 100%;
      min-height: 34px;
      padding: 0 30px 0 10px;
      border: 1px solid #c6d0c3;
      border-radius: 6px;
      background: #ffffff;
      color: #17201c;
      font: inherit;
      font-weight: 700;
    }}
    main {{ width: min(1600px, 100%); margin: 0 auto; padding: 18px 20px 28px; }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-bottom: 14px;
      color: #53615a;
      font-size: 13px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border: 1px solid #d1d9cf;
      border-radius: 999px;
      background: #ffffff;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid #d6ddd4;
      border-radius: 8px;
      background: #ffffff;
    }}
    table {{
      width: max(100%, 1280px);
      border-collapse: separate;
      border-spacing: 0;
      table-layout: fixed;
    }}
    th, td {{ border-bottom: 1px solid #e2e8df; vertical-align: top; }}
    th {{
      position: sticky;
      top: 0;
      z-index: 5;
      height: 40px;
      padding: 0 10px;
      background: #fbfcfa;
      color: #53615a;
      font-size: 12px;
      text-align: left;
      white-space: nowrap;
    }}
    td {{ padding: 10px; }}
    tr:last-child td {{ border-bottom: 0; }}
    .compare-col {{ width: 92px; }}
    .preview-col {{ width: 148px; }}
    .compare-button {{
      width: 100%;
      min-height: 34px;
      border: 1px solid #176a56;
      border-radius: 6px;
      background: #176a56;
      color: #ffffff;
      font: inherit;
      font-size: 13px;
      font-weight: 900;
      cursor: pointer;
    }}
    .compare-button:disabled {{
      border-color: #cbd5c8;
      background: #eef3ec;
      color: #758179;
      cursor: not-allowed;
    }}
    .case-name {{ font-size: 13px; font-weight: 800; overflow-wrap: anywhere; }}
    .sample-code {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      margin-bottom: 7px;
      padding: 0 8px;
      border-radius: 6px;
      color: #ffffff;
      background: #245f53;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0;
      white-space: nowrap;
    }}
    .sample-badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      margin-top: 7px;
      padding: 0 8px;
      border: 1px solid #cbd5c8;
      border-radius: 999px;
      color: #17201c;
      background: #f7f9f6;
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
    }}
    .case-meta {{ margin-top: 6px; display: grid; gap: 5px; color: #5f6c66; font-size: 12px; line-height: 1.35; }}
    .hit {{ color: #176a56; font-weight: 800; }}
    .miss {{ color: #a23d35; font-weight: 800; }}
    .pending {{ color: #6b6258; font-weight: 800; }}
    .tools {{ overflow-wrap: anywhere; }}
    .thumb-button {{
      position: relative;
      width: 100%;
      min-height: 92px;
      max-height: 220px;
      aspect-ratio: var(--thumb-ratio, 1 / 1);
      display: grid;
      place-items: center;
      padding: 6px;
      border: 1px solid #cad3c7;
      border-radius: 6px;
      cursor: zoom-in;
      overflow: hidden;
    }}
    .thumb-tag {{
      position: absolute;
      top: 6px;
      left: 6px;
      max-width: calc(100% - 12px);
      padding: 3px 6px;
      border-radius: 5px;
      background: rgba(12, 17, 15, 0.78);
      color: #ffffff;
      font-size: 11px;
      font-weight: 900;
      line-height: 1.2;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      pointer-events: none;
    }}
    .thumb-button img {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      user-select: none;
      pointer-events: none;
    }}
    .thumb-button:focus-visible {{ outline: 3px solid rgba(25, 111, 90, 0.32); outline-offset: 2px; }}
    .bg-checker {{
      background-color: #edf1ea;
      background-image:
        linear-gradient(45deg, #cad3c7 25%, transparent 25%),
        linear-gradient(-45deg, #cad3c7 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #cad3c7 75%),
        linear-gradient(-45deg, transparent 75%, #cad3c7 75%);
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
      background-size: 20px 20px;
    }}
    .bg-white {{ background: #ffffff; }}
    .bg-black {{ background: #101413; }}
    .bg-purple {{ background: #7c3aed; }}
    .bg-blue {{ background: #2563eb; }}
    /* Known green-screen reference for judging whether transparent shadows
       match the original source, without white/checker contrast bias. */
    .bg-green {{ background: #00c800; }}
    .candidate-label {{ margin-bottom: 7px; color: #53615a; font-size: 12px; font-weight: 800; overflow-wrap: anywhere; }}
    .selected-mark {{
      display: inline-flex;
      margin-left: 6px;
      color: #176a56;
      font-weight: 900;
    }}
    .empty-cell {{
      width: 100%;
      min-height: 92px;
      aspect-ratio: 1 / 1;
      display: grid;
      place-items: center;
      border: 1px dashed #cbd5c8;
      border-radius: 6px;
      color: #66736c;
      background: #f7f9f6;
      font-size: 12px;
      font-weight: 800;
    }}
    .modal {{
      position: fixed;
      inset: 0;
      z-index: 50;
      display: none;
      grid-template-rows: 56px 1fr;
      background: rgba(12, 17, 15, 0.94);
    }}
    .modal.is-open {{ display: grid; }}
    .modal-bar {{
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 0 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.14);
      color: #ffffff;
    }}
    .modal-title {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; font-weight: 800; }}
    .modal-actions {{ display: flex; align-items: center; gap: 8px; flex-shrink: 0; }}
    .swatch, .icon-button {{
      width: 34px;
      height: 34px;
      border: 1px solid rgba(255, 255, 255, 0.28);
      border-radius: 6px;
      color: #ffffff;
      background: transparent;
      cursor: pointer;
    }}
    .swatch[aria-pressed="true"], .icon-button:focus-visible {{
      outline: 2px solid #ffffff;
      outline-offset: 2px;
    }}
    .icon-button {{ font-size: 18px; line-height: 1; }}
    .modal-stage {{
      min-height: 0;
      display: grid;
      place-items: center;
      overflow: hidden;
      touch-action: none;
      cursor: grab;
    }}
    .modal-stage.is-dragging {{ cursor: grabbing; }}
    .modal-stage img {{
      max-width: 86vw;
      max-height: 82vh;
      object-fit: contain;
      transform-origin: center center;
      will-change: transform;
      user-select: none;
      pointer-events: none;
    }}
    .compare-modal {{
      position: fixed;
      inset: 0;
      z-index: 70;
      display: none;
      grid-template-rows: 64px 1fr;
      background: rgba(0, 0, 0, 0.96);
      color: #ffffff;
    }}
    .compare-modal.is-open {{ display: grid; }}
    .compare-bar {{
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 12px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.14);
    }}
    .compare-picker {{
      min-width: 0;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: #cdd5d0;
      font-size: 13px;
      font-weight: 900;
    }}
    .compare-picker select {{
      width: min(240px, 30vw);
      min-height: 36px;
      padding: 0 32px 0 10px;
      border: 1px solid rgba(255, 255, 255, 0.28);
      border-radius: 6px;
      background: #111614;
      color: #ffffff;
      font: inherit;
      font-weight: 800;
    }}
    .compare-alpha {{
      min-width: 132px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: #cdd5d0;
      font-size: 12px;
      font-weight: 900;
    }}
    .compare-alpha input {{
      width: 104px;
      accent-color: #ffffff;
      cursor: pointer;
    }}
    .compare-close {{
      position: absolute;
      top: 14px;
      right: 16px;
    }}
    .compare-stage {{
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      place-items: center;
      gap: 12px;
      overflow: hidden;
      padding: 24px;
    }}
    .compare-bg-row {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }}
    .compare-bg-button {{
      width: 34px;
      height: 34px;
      border: 1px solid rgba(255, 255, 255, 0.3);
      border-radius: 6px;
      cursor: pointer;
    }}
    .compare-bg-button[aria-pressed="true"], .compare-bg-button:focus-visible {{
      outline: 2px solid #ffffff;
      outline-offset: 2px;
    }}
    .compare-frame {{
      position: relative;
      width: min(calc(100vw - 48px), calc(100vh - 172px));
      aspect-ratio: 1 / 1;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 6px;
      background: #070908;
      cursor: ew-resize;
    }}
    .compare-frame.bg-checker {{ background-color: #edf1ea; }}
    .compare-frame.bg-white {{ background: #ffffff; }}
    .compare-frame.bg-black {{ background: #101413; }}
    .compare-frame.bg-green {{ background: #00c800; }}
    .compare-frame.bg-purple {{ background: #7c3aed; }}
    .compare-frame.bg-blue {{ background: #2563eb; }}
    .compare-frame img {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      user-select: none;
      pointer-events: none;
    }}
    .compare-img-two {{
      clip-path: inset(0 50% 0 0);
    }}
    .compare-divider {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 2px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.28);
      transform: translateX(-1px);
      pointer-events: none;
    }}
    .compare-empty {{
      position: absolute;
      inset: 0;
      display: none;
      place-items: center;
      color: #aab3ae;
      font-size: 13px;
      font-weight: 900;
      background: #070908;
    }}
    .compare-frame.is-empty .compare-empty {{ display: grid; }}
    .compare-frame.is-empty img, .compare-frame.is-empty .compare-divider {{ display: none; }}
    .eval-panel {{
      position: fixed;
      inset: 0;
      z-index: 60;
      display: none;
      place-items: center;
      padding: 20px;
      background: rgba(12, 17, 15, 0.58);
    }}
    .eval-panel.is-open {{ display: grid; }}
    .eval-dialog {{
      width: min(720px, 100%);
      max-height: min(760px, calc(100vh - 40px));
      display: grid;
      grid-template-rows: auto auto auto auto 1fr auto;
      overflow: hidden;
      border: 1px solid #d6ddd4;
      border-radius: 8px;
      background: #ffffff;
      box-shadow: 0 18px 52px rgba(20, 31, 26, 0.22);
    }}
    .eval-dialog header {{
      position: static;
      min-height: 54px;
      padding: 0 16px;
      border-bottom: 1px solid #e2e8df;
      background: #ffffff;
      backdrop-filter: none;
    }}
    .eval-dialog h2 {{ margin: 0; font-size: 16px; letter-spacing: 0; }}
    .eval-tools {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 16px;
      border-bottom: 1px solid #e2e8df;
    }}
    .eval-tools button, .eval-actions button {{
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid #c6d0c3;
      border-radius: 6px;
      background: #ffffff;
      color: #17201c;
      font: inherit;
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
    }}
    .eval-tools .selection-count {{ margin-left: auto; color: #53615a; font-size: 12px; font-weight: 800; }}
    .eval-tools .category-buttons {{ display: inline-flex; gap: 6px; flex-wrap: wrap; }}
    .path-tools, .screen-tools {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      padding: 10px 16px;
      border-bottom: 1px solid #e2e8df;
      color: #53615a;
      font-size: 13px;
      font-weight: 800;
    }}
    .path-tools label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
    }}
    .path-tools select {{
      min-width: 180px;
      min-height: 34px;
      padding: 0 10px;
      border: 1px solid #c6d0c3;
      border-radius: 6px;
      background: #ffffff;
      color: #17201c;
      font: inherit;
      font-weight: 800;
    }}
    .screen-option {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 0 10px;
      border: 1px solid #c6d0c3;
      border-radius: 6px;
      background: #ffffff;
      color: #17201c;
    }}
    .screen-option input {{ width: 15px; height: 15px; min-height: 0; margin: 0; }}
    .sample-list {{
      min-height: 0;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(218px, 1fr));
      align-content: start;
      gap: 10px;
      overflow: auto;
      padding: 12px 16px;
      background: #f7faf6;
    }}
    .sample-group {{
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 24px;
      margin-top: 4px;
      color: #53615a;
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
    }}
    .sample-group:first-child {{ margin-top: 0; }}
    .sample-option {{
      display: grid;
      grid-template-columns: 22px 54px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      min-height: 74px;
      padding: 8px;
      border: 1px solid #dde6da;
      border-radius: 7px;
      background: #ffffff;
      color: #17201c;
      font-size: 13px;
      font-weight: 700;
    }}
    .sample-option:hover {{ border-color: #b8c8b4; background: #fbfdfb; }}
    .sample-option input {{ width: 16px; height: 16px; min-height: 0; margin: 0; }}
    .sample-thumb {{
      width: 54px;
      height: 54px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border: 1px solid #cbd5c8;
      border-radius: 6px;
      background: #00c800;
    }}
    .sample-thumb img {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    .sample-meta {{ min-width: 0; display: grid; gap: 2px; }}
    .sample-code {{ margin: 0; font-weight: 900; line-height: 1.1; }}
    .sample-case, .sample-family {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #617068;
      font-size: 11px;
      line-height: 1.15;
    }}
    .eval-actions {{
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      padding: 12px 16px;
      border-top: 1px solid #e2e8df;
    }}
    .eval-actions .primary {{
      border-color: #176a56;
      background: #176a56;
      color: #ffffff;
    }}
    .eval-actions .primary:disabled {{ opacity: 0.55; cursor: not-allowed; }}
    @media (max-width: 980px) {{
      header {{
        position: static;
        padding: 10px 16px;
      }}
      nav {{ width: 100%; gap: 8px; }}
      #run-id {{ flex: 1 1 100%; max-width: 100%; }}
      .run-picker {{ flex: 1 1 320px; width: auto; min-width: 0; }}
      .run-picker select {{ width: 100%; }}
      .run-button {{ flex: 0 0 auto; }}
      .run-status {{ flex: 1 1 180px; }}
      .run-progress {{ flex: 1 1 160px; }}
      main {{ padding: 14px 12px 22px; }}
      .modal-bar {{ min-height: 92px; align-items: flex-start; flex-direction: column; padding: 10px 12px; }}
      .modal {{ grid-template-rows: auto 1fr; }}
      .compare-modal {{ grid-template-rows: auto 1fr; }}
      .compare-bar {{ justify-content: flex-start; flex-wrap: wrap; padding-right: 58px; }}
      .compare-picker select {{ width: min(220px, 62vw); }}
      .compare-alpha {{ min-width: 124px; }}
    }}
    @media (max-width: 560px) {{
      .run-picker {{ flex-basis: 100%; }}
      .run-progress, .run-status {{ flex-basis: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ERMBG Game Eval</h1>
    <nav>
      <span id="run-id"></span>
      <label class="run-picker" for="run-select">
        <span>批次</span>
        <select id="run-select" aria-label="选择测试批次"></select>
      </label>
      <button class="run-button" type="button" id="start-full-eval">启动测试</button>
      <div class="run-progress" role="progressbar" aria-label="测试进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
        <div class="run-progress-bar" id="batch-progress"></div>
      </div>
      <span class="run-status" id="batch-status" aria-live="polite"></span>
      <span class="runtime-status" id="runtime-status" aria-live="polite">
        <span class="runtime-pill" data-runtime="local">Local</span>
        <span class="runtime-pill" data-runtime="direct">Direct</span>
      </span>
      <a href="/artifacts">Artifacts</a>
      <a href="/">上传页</a>
    </nav>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <div class="table-wrap">
      <table aria-label="game eval result table">
        <thead>
          <tr>
            <th class="compare-col">比较</th>
            <th class="preview-col">原图</th>
            <th class="preview-col">alpha mask</th>
            <th class="preview-col">trimap</th>
            <th class="preview-col">白底</th>
            <th class="preview-col">黑底</th>
            <th class="preview-col">透明底</th>
            <th class="preview-col">绿底</th>
            <th class="preview-col">紫底</th>
            <th class="preview-col">蓝底</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </main>
  <div class="eval-panel" id="eval-panel" aria-hidden="true">
    <section class="eval-dialog" role="dialog" aria-modal="true" aria-labelledby="eval-dialog-title">
      <header>
        <h2 id="eval-dialog-title">选择测试样本</h2>
      </header>
      <div class="eval-tools">
        <button type="button" id="select-all-samples">全选</button>
        <button type="button" id="clear-all-samples">取消全选</button>
        <span class="category-buttons" id="category-buttons"></span>
        <span class="selection-count" id="selection-count"></span>
      </div>
      <div class="path-tools" aria-label="选择测试路径">
        <label for="eval-test-path">测试路径
          <select id="eval-test-path" name="eval-test-path">
            <option value="auto" selected>Auto</option>
            <option value="direct-worker">Direct Worker</option>
          </select>
        </label>
      </div>
      <div class="sample-list" id="sample-list"></div>
      <div class="eval-actions">
        <button type="button" id="cancel-eval-panel">取消</button>
        <button class="primary" type="button" id="confirm-start-eval">开始测试</button>
      </div>
    </section>
  </div>
  <div class="compare-modal" id="compare-modal" aria-hidden="true">
    <div class="compare-bar">
      <label class="compare-alpha" for="compare-alpha-one">Alpha 1
        <input id="compare-alpha-one" type="range" min="0" max="100" value="100" step="1">
      </label>
      <label class="compare-picker" for="compare-view-one">视图1
        <select id="compare-view-one"></select>
      </label>
      <label class="compare-picker" for="compare-view-two">视图2
        <select id="compare-view-two"></select>
      </label>
      <label class="compare-alpha" for="compare-alpha-two">Alpha 2
        <input id="compare-alpha-two" type="range" min="0" max="100" value="100" step="1">
      </label>
      <button class="icon-button compare-close" type="button" id="close-compare" title="关闭" aria-label="关闭">×</button>
    </div>
    <div class="compare-stage" id="compare-stage">
      <div class="compare-bg-row" aria-label="比较背景色">
        <button class="compare-bg-button bg-black" type="button" data-bg="black" title="黑底" aria-label="黑底"></button>
        <button class="compare-bg-button bg-white" type="button" data-bg="white" title="白底" aria-label="白底"></button>
        <button class="compare-bg-button bg-checker" type="button" data-bg="checker" title="透明底" aria-label="透明底"></button>
        <button class="compare-bg-button bg-green" type="button" data-bg="green" title="绿底" aria-label="绿底"></button>
        <button class="compare-bg-button bg-purple" type="button" data-bg="purple" title="紫底" aria-label="紫底"></button>
        <button class="compare-bg-button bg-blue" type="button" data-bg="blue" title="蓝底" aria-label="蓝底"></button>
      </div>
      <div class="compare-frame bg-black" id="compare-frame">
        <img class="compare-img-one" id="compare-img-one" alt="">
        <img class="compare-img-two" id="compare-img-two" alt="">
        <div class="compare-divider" id="compare-divider"></div>
        <div class="compare-empty" id="compare-empty">没有可比较的图片</div>
      </div>
    </div>
  </div>
  <div class="modal" id="modal" aria-hidden="true">
    <div class="modal-bar">
      <div class="modal-title" id="modal-title"></div>
      <div class="modal-actions" aria-label="preview controls">
        <button class="swatch bg-checker" type="button" data-bg="checker" title="透明底" aria-label="透明底"></button>
        <button class="swatch bg-white" type="button" data-bg="white" title="白底" aria-label="白底"></button>
        <button class="swatch bg-black" type="button" data-bg="black" title="黑底" aria-label="黑底"></button>
        <button class="swatch bg-green" type="button" data-bg="green" title="绿幕参照" aria-label="绿幕参照"></button>
        <button class="swatch bg-purple" type="button" data-bg="purple" title="紫底" aria-label="紫底"></button>
        <button class="swatch bg-blue" type="button" data-bg="blue" title="蓝底" aria-label="蓝底"></button>
        <button class="icon-button" type="button" id="reset-preview" title="重置视图" aria-label="重置视图">↺</button>
        <button class="icon-button" type="button" id="close-modal" title="关闭" aria-label="关闭">×</button>
      </div>
    </div>
    <div class="modal-stage bg-checker" id="modal-stage">
      <img id="modal-img" alt="">
    </div>
  </div>
  <script>
    const data = {data_json};
    const backgrounds = ["checker", "white", "black", "green", "purple", "blue"];
    const previewColumns = [
      {{ label: "原图", urlKey: "originalUrl", bg: "checker" }},
      {{ label: "alpha mask", urlKey: "alphaUrl", bg: "white" }},
      {{ label: "trimap", urlKey: "trimapUrl", bg: "white" }},
      {{ label: "白底", urlKey: "matteUrl", bg: "white" }},
      {{ label: "黑底", urlKey: "matteUrl", bg: "black" }},
      {{ label: "透明底", urlKey: "matteUrl", bg: "checker" }},
      {{ label: "绿底", urlKey: "matteUrl", bg: "green" }},
      {{ label: "紫底", urlKey: "matteUrl", bg: "purple" }},
      {{ label: "蓝底", urlKey: "matteUrl", bg: "blue" }},
    ];
    const compareOptions = [
      {{ label: "原图", urlKey: "originalUrl" }},
      {{ label: "Trimap", urlKey: "trimapUrl" }},
      {{ label: "Mask Hint", urlKey: "maskHintUrl" }},
      {{ label: "corridorkey Raw Alpha", urlKey: "corridorkeyRawAlphaUrl" }},
      {{ label: "corridorkey Forground", urlKey: "corridorkeyForegroundUrl" }},
      {{ label: "输出 Alpha", urlKey: "alphaUrl" }},
    ];
    const rowsEl = document.getElementById("rows");
    const summaryEl = document.getElementById("summary");
    const runIdEl = document.getElementById("run-id");
    const runSelect = document.getElementById("run-select");
    const startFullEvalButton = document.getElementById("start-full-eval");
    const batchProgress = document.getElementById("batch-progress");
    const batchProgressRoot = batchProgress.parentElement;
    const batchStatusEl = document.getElementById("batch-status");
    const runtimeStatus = document.getElementById("runtime-status");
    const evalPanel = document.getElementById("eval-panel");
    const sampleList = document.getElementById("sample-list");
    const testPathSelect = document.getElementById("eval-test-path");
    const selectAllSamplesButton = document.getElementById("select-all-samples");
    const clearAllSamplesButton = document.getElementById("clear-all-samples");
    const cancelEvalPanelButton = document.getElementById("cancel-eval-panel");
    const confirmStartEvalButton = document.getElementById("confirm-start-eval");
    const selectionCountEl = document.getElementById("selection-count");
    const categoryButtonsEl = document.getElementById("category-buttons");
    const modal = document.getElementById("modal");
    const modalStage = document.getElementById("modal-stage");
    const modalImg = document.getElementById("modal-img");
    const modalTitle = document.getElementById("modal-title");
    const closeModalButton = document.getElementById("close-modal");
    const resetPreviewButton = document.getElementById("reset-preview");
    const swatches = Array.from(document.querySelectorAll(".swatch"));
    const compareModal = document.getElementById("compare-modal");
    const compareStage = document.getElementById("compare-stage");
    const compareFrame = document.getElementById("compare-frame");
    const compareViewOne = document.getElementById("compare-view-one");
    const compareViewTwo = document.getElementById("compare-view-two");
    const compareAlphaOne = document.getElementById("compare-alpha-one");
    const compareAlphaTwo = document.getElementById("compare-alpha-two");
    const compareImgOne = document.getElementById("compare-img-one");
    const compareImgTwo = document.getElementById("compare-img-two");
    const compareDivider = document.getElementById("compare-divider");
    const closeCompareButton = document.getElementById("close-compare");
    const compareBgButtons = Array.from(document.querySelectorAll(".compare-bg-button"));
    let scale = 1;
    let panX = 0;
    let panY = 0;
    let dragStart = null;
    let activeBatchStatusUrl = "";
    let activeCompareCase = null;
    let comparePosition = 0.5;
    let compareAlphaOneValue = 1;
    let compareAlphaTwoValue = 1;

    function text(value) {{
      return value === null || value === undefined || value === "" ? "—" : String(value);
    }}

    function setBackground(element, bg) {{
      element.classList.remove(...backgrounds.map((name) => `bg-${{name}}`));
      element.classList.add(`bg-${{bg}}`);
    }}

    function countsText(counts) {{
      if (!counts || typeof counts !== "object") return "";
      return Object.entries(counts).map(([key, value]) => `${{key}}=${{value}}`).join(", ");
    }}

    function makePreview(src, label, bg, tag = "") {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb-button";
      setBackground(button, bg);
      button.title = label;
      const img = document.createElement("img");
      img.src = src;
      img.alt = label;
      img.onload = () => {{
        if (img.naturalWidth > 0 && img.naturalHeight > 0) {{
          button.style.setProperty("--thumb-ratio", img.naturalWidth + " / " + img.naturalHeight);
        }}
      }};
      button.appendChild(img);
      if (tag) {{
        const badge = document.createElement("span");
        badge.className = "thumb-tag";
        badge.textContent = tag;
        button.appendChild(badge);
      }}
      button.addEventListener("click", () => openModal(src, label, bg));
      return button;
    }}

    function availableCompareViews(caseItem) {{
      return compareOptions
        .map((option) => ({{ ...option, url: caseItem[option.urlKey] || "" }}))
        .filter((option) => option.url);
    }}

    function populateCompareSelect(select, caseItem, preferredKey, fallbackViews) {{
      select.innerHTML = "";
      compareOptions.forEach((option) => {{
        const choice = document.createElement("option");
        choice.value = option.urlKey;
        choice.textContent = option.label;
        choice.disabled = !caseItem[option.urlKey];
        select.appendChild(choice);
      }});
      const preferred = compareOptions.find((option) => option.urlKey === preferredKey && caseItem[option.urlKey]);
      select.value = preferred ? preferred.urlKey : (fallbackViews[0] ? fallbackViews[0].urlKey : compareOptions[0].urlKey);
    }}

    function updateCompareImages() {{
      if (!activeCompareCase) return;
      const first = compareOptions.find((option) => option.urlKey === compareViewOne.value);
      const second = compareOptions.find((option) => option.urlKey === compareViewTwo.value);
      const firstUrl = first ? activeCompareCase[first.urlKey] : "";
      const secondUrl = second ? activeCompareCase[second.urlKey] : "";
      compareFrame.classList.toggle("is-empty", !firstUrl || !secondUrl);
      compareImgOne.src = firstUrl || "";
      compareImgTwo.src = secondUrl || "";
      compareImgOne.alt = first ? first.label : "";
      compareImgTwo.alt = second ? second.label : "";
      updateComparePosition(comparePosition);
      updateCompareAlpha();
    }}

    function updateCompareAlpha() {{
      compareAlphaOneValue = Math.max(0, Math.min(1, Number(compareAlphaOne.value) / 100));
      compareAlphaTwoValue = Math.max(0, Math.min(1, Number(compareAlphaTwo.value) / 100));
      compareImgOne.style.opacity = String(compareAlphaOneValue);
      compareImgTwo.style.opacity = String(compareAlphaTwoValue);
    }}

    function resetCompareAlpha() {{
      compareAlphaOne.value = "100";
      compareAlphaTwo.value = "100";
      updateCompareAlpha();
    }}

    function setCompareBackground(bg) {{
      setBackground(compareFrame, bg);
      compareBgButtons.forEach((button) => {{
        button.setAttribute("aria-pressed", String(button.dataset.bg === bg));
      }});
    }}

    function updateComparePosition(value) {{
      comparePosition = Math.max(0, Math.min(1, value));
      const pct = comparePosition * 100;
      compareImgTwo.style.clipPath = `inset(0 ${{100 - pct}}% 0 0)`;
      compareDivider.style.left = `${{pct}}%`;
    }}

    function updateCompareFromPointer(event) {{
      if (!compareModal.classList.contains("is-open")) return;
      const rect = compareFrame.getBoundingClientRect();
      if (!rect.width) return;
      updateComparePosition((event.clientX - rect.left) / rect.width);
    }}

    function openCompare(caseItem) {{
      activeCompareCase = caseItem;
      const views = availableCompareViews(caseItem);
      populateCompareSelect(compareViewOne, caseItem, "maskHintUrl", views);
      populateCompareSelect(compareViewTwo, caseItem, "corridorkeyRawAlphaUrl", views.slice(1).length ? views.slice(1) : views);
      if (compareViewOne.value === compareViewTwo.value && views.length > 1) {{
        const alternate = views.find((view) => view.urlKey !== compareViewOne.value);
        if (alternate) compareViewTwo.value = alternate.urlKey;
      }}
      updateComparePosition(0.5);
      resetCompareAlpha();
      setCompareBackground("black");
      updateCompareImages();
      compareModal.classList.add("is-open");
      compareModal.setAttribute("aria-hidden", "false");
    }}

    function closeCompare() {{
      compareModal.classList.remove("is-open");
      compareModal.setAttribute("aria-hidden", "true");
      activeCompareCase = null;
      compareImgOne.removeAttribute("src");
      compareImgTwo.removeAttribute("src");
    }}

    function renderRunSelect() {{
      const runs = Array.isArray(data.runs) ? data.runs : [];
      runSelect.innerHTML = "";
      runs.forEach((run) => {{
        const option = document.createElement("option");
        option.value = run.id;
        option.textContent = run.label || run.id;
        option.selected = run.selected === true || run.id === data.selectedRun;
        runSelect.appendChild(option);
      }});
      runSelect.disabled = runs.length <= 1;
    }}

    function previewUrlForColumn(caseItem, column) {{
      if (column.urlKey === "trimapUrl" && caseItem.isCorridorKey) {{
        return caseItem.maskHintUrl || caseItem.trimapUrl || "";
      }}
      return caseItem[column.urlKey] || "";
    }}

    function previewTagForColumn(caseItem, column) {{
      if (column.urlKey === "originalUrl") return `${{caseItem.sampleCode || ""}}`;
      if (column.urlKey === "trimapUrl" && caseItem.isCorridorKey && caseItem.maskHintUrl) return "hint";
      return "";
    }}

    function renderRows() {{
      renderRunSelect();
      runIdEl.textContent = data.runId || "game eval";
      setBatchProgress(data.progress);
      if (data.progress) {{
        setBatchStatus(`当前：${{progressText(data)}}`, false);
      }}
      summaryEl.innerHTML = "";
      [
        `model: ${{text(data.model)}}`,
        `success: ${{text(data.success)}}`,
        `expected hit: ${{text(data.expectedHit)}}`,
        `any hit: ${{text(data.expectedAnyHit)}}`,
        `harmful tools: ${{text(data.harmfulTools)}}`,
        `sample rows: ${{text(data.sampleRows)}}`,
        `report: ${{text(data.reportPath)}}`,
        `samples: ${{text(data.vlmRoot)}}`,
        `matte: ${{text(data.matteRoot)}}`,
      ].forEach((item) => {{
        const pill = document.createElement("span");
        pill.className = "pill";
        pill.textContent = item;
        summaryEl.appendChild(pill);
      }});

      rowsEl.innerHTML = "";
      data.cases.forEach((caseItem) => {{
        const row = document.createElement("tr");
        const compareCell = document.createElement("td");
        const compareButton = document.createElement("button");
        const compareViews = availableCompareViews(caseItem);
        compareButton.type = "button";
        compareButton.className = "compare-button";
        compareButton.textContent = "比较";
        compareButton.disabled = compareViews.length < 2;
        compareButton.title = compareViews.length < 2 ? "至少需要两张图" : `${{caseItem.sampleCode || ""}} · ${{caseItem.caseId || ""}} · 比较`;
        compareButton.addEventListener("click", () => openCompare(caseItem));
        compareCell.appendChild(compareButton);
        row.appendChild(compareCell);
        previewColumns.forEach((column) => {{
          const cell = document.createElement("td");
          const previewUrl = previewUrlForColumn(caseItem, column);
          if (previewUrl) {{
            cell.appendChild(
              makePreview(
                previewUrl,
                `${{caseItem.sampleCode || ""}} · ${{caseItem.caseId || ""}} · ${{column.label}}`,
                column.bg,
                previewTagForColumn(caseItem, column),
              ),
            );
          }} else {{
            const empty = document.createElement("div");
            empty.className = "empty-cell";
            empty.textContent = "—";
            cell.appendChild(empty);
          }}
          row.appendChild(cell);
        }});
        rowsEl.appendChild(row);
      }});
    }}

    function setBatchStatus(message, isRunning = false) {{
      batchStatusEl.textContent = message || "";
      startFullEvalButton.disabled = isRunning;
    }}

    function setBatchProgress(progress) {{
      const pct = progress && Number.isFinite(Number(progress.percent)) ? Number(progress.percent) : 0;
      const clamped = Math.max(0, Math.min(100, pct));
      batchProgress.style.width = `${{clamped}}%`;
      batchProgressRoot.setAttribute("aria-valuenow", String(Math.round(clamped)));
    }}

    function progressText(status) {{
      const progress = status && status.progress ? status.progress : {{}};
      const completed = Number.isFinite(Number(progress.completed)) ? Number(progress.completed) : 0;
      const total = Number.isFinite(Number(progress.total)) ? Number(progress.total) : 18;
      const ok = Number.isFinite(Number(progress.ok)) ? Number(progress.ok) : 0;
      const errors = Number.isFinite(Number(progress.errors)) ? Number(progress.errors) : 0;
      return `${{completed}}/${{total}} · ok ${{ok}} · error ${{errors}}`;
    }}

    function setRuntimePill(kind, label, state, title) {{
      const pill = runtimeStatus.querySelector(`[data-runtime="${{kind}}"]`);
      if (!pill) return;
      pill.textContent = label;
      pill.classList.remove("is-ok", "is-error", "is-warn");
      if (state) pill.classList.add(`is-${{state}}`);
      pill.title = title || label;
    }}

    async function refreshRuntimeStatus() {{
      try {{
        const response = await fetch("/api/runtime-capabilities?include_comfy=false&include_object_info=false&timeout=1.5");
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const payload = await response.json();
        setRuntimePill("local", "Local", payload.local && payload.local.status === "ok" ? "ok" : "error", `ERMBG ${{payload.local && payload.local.version ? payload.local.version : ""}}`);
        const dw = payload.direct_worker || {{}};
        const directOk = dw.status === "ok";
        const directLabel = dw.location ? `Direct · ${{dw.location}}` : "Direct";
        setRuntimePill("direct", directLabel, directOk ? "ok" : "error", directOk ? dw.url : (dw.error || "Direct Worker unavailable"));
      }} catch (error) {{
        setRuntimePill("local", "Local", "warn", "capability check failed");
        setRuntimePill("direct", "Direct", "warn", "capability check failed");
      }}
    }}

    async function pollBatchStatus() {{
      if (!activeBatchStatusUrl) return;
      try {{
        const response = await fetch(activeBatchStatusUrl);
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const status = await response.json();
        setBatchProgress(status.progress);
        if (status.status === "complete" && status.hasReport && status.url) {{
          setBatchStatus(`完成：${{progressText(status)}}`, false);
          window.location.href = status.url;
          return;
        }}
        if (status.status === "error") {{
          setBatchStatus(`失败：${{progressText(status)}}`, false);
          activeBatchStatusUrl = "";
          return;
        }}
        setBatchStatus(`运行中：${{progressText(status)}}`, true);
        window.setTimeout(pollBatchStatus, 5000);
      }} catch (error) {{
        setBatchStatus("状态读取失败", false);
        activeBatchStatusUrl = "";
      }}
    }}

    function sampleCheckboxes() {{
      return Array.from(sampleList.querySelectorAll('input[type="checkbox"]'));
    }}

    function selectedSampleIds() {{
      return sampleCheckboxes().filter((input) => input.checked).map((input) => input.value);
    }}

    function selectedTestPath() {{
      return testPathSelect ? testPathSelect.value : "corridorkey";
    }}

    function updateSelectionCount() {{
      const selected = selectedSampleIds().length;
      const total = sampleCheckboxes().length;
      selectionCountEl.textContent = `${{selected}}/${{total}} samples`;
      confirmStartEvalButton.disabled = selected === 0;
    }}

    function renderSampleList() {{
      const samples = Array.isArray(data.samples) ? data.samples : [];
      sampleList.innerHTML = "";
      const labels = {{ button: "Button", icon: "Icon / Effect", character: "Character" }};
      const grouped = samples.reduce((acc, sample) => {{
        const key = sample.category || "other";
        if (!acc.has(key)) acc.set(key, []);
        acc.get(key).push(sample);
        return acc;
      }}, new Map());
      grouped.forEach((groupSamples, category) => {{
        const group = document.createElement("div");
        group.className = "sample-group";
        group.textContent = `${{labels[category] || category}} · ${{groupSamples.length}}`;
        sampleList.appendChild(group);
        groupSamples.forEach((sample) => {{
        const label = document.createElement("label");
        label.className = "sample-option";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = sample.sampleId;
        checkbox.checked = sample.defaultSelected === true;
        checkbox.addEventListener("change", updateSelectionCount);
        const thumb = document.createElement("span");
        thumb.className = "sample-thumb";
        if (sample.thumbnailUrl) {{
          const image = document.createElement("img");
          image.src = sample.thumbnailUrl;
          image.alt = sample.sampleId || "";
          thumb.appendChild(image);
        }}
        const code = document.createElement("span");
        code.className = "sample-meta";
        const codeText = document.createElement("span");
        codeText.className = "sample-code";
        codeText.textContent = `${{sample.sampleId || ""}} · ${{sample.screen || ""}}`;
        const caseText = document.createElement("span");
        caseText.className = "sample-case";
        caseText.textContent = sample.caseId || "";
        const familyText = document.createElement("span");
        familyText.className = "sample-family";
        familyText.textContent = sample.family || sample.primaryAmbiguity || "";
        code.appendChild(codeText);
        code.appendChild(caseText);
        code.appendChild(familyText);
        label.title = `${{sample.sampleId || ""}} · ${{sample.caseId || ""}}`;
        label.appendChild(checkbox);
        label.appendChild(thumb);
        label.appendChild(code);
        sampleList.appendChild(label);
      }});
      }});
      updateSelectionCount();
    }}

    function openEvalPanel() {{
      renderSampleList();
      renderCategoryButtons();
      evalPanel.classList.add("is-open");
      evalPanel.setAttribute("aria-hidden", "false");
    }}

    function closeEvalPanel() {{
      evalPanel.classList.remove("is-open");
      evalPanel.setAttribute("aria-hidden", "true");
    }}

    function setAllSamples(checked) {{
      sampleCheckboxes().forEach((input) => {{
        input.checked = checked;
      }});
      updateSelectionCount();
    }}

    function setCategorySamples(category) {{
      const samples = Array.isArray(data.samples) ? data.samples : [];
      const targetIds = new Set(
        samples
          .filter((sample) => (sample.category || "other") === category)
          .map((sample) => sample.sampleId)
      );
      sampleCheckboxes().forEach((input) => {{
        input.checked = targetIds.has(input.value);
      }});
      updateSelectionCount();
    }}

    function renderCategoryButtons() {{
      if (!categoryButtonsEl) return;
      categoryButtonsEl.innerHTML = "";
      const samples = Array.isArray(data.samples) ? data.samples : [];
      const labels = {{ button: "Button", icon: "Icon / Effect", character: "Character" }};
      const counts = new Map();
      samples.forEach((sample) => {{
        const key = sample.category || "other";
        counts.set(key, (counts.get(key) || 0) + 1);
      }});
      counts.forEach((count, category) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.category = category;
        button.textContent = `${{labels[category] || category}} (${{count}})`;
        button.title = `仅选中 ${{labels[category] || category}}`;
        button.addEventListener("click", () => setCategorySamples(category));
        categoryButtonsEl.appendChild(button);
      }});
    }}

    async function startSelectedEval() {{
      const sampleIds = selectedSampleIds();
      const testPath = selectedTestPath();
      if (!sampleIds.length) return;
      setBatchStatus("启动中", true);
      setBatchProgress({{ percent: 0 }});
      try {{
        closeEvalPanel();
        const response = await fetch("/eval/game/run", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ sample_ids: sampleIds, test_path: testPath }}),
        }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const payload = await response.json();
        activeBatchStatusUrl = payload.statusUrl || "";
        setBatchProgress(payload.progress);
        setBatchStatus(`运行中：${{progressText(payload)}}`, true);
        window.setTimeout(pollBatchStatus, 5000);
      }} catch (error) {{
        setBatchStatus("启动失败", false);
      }}
    }}

    function clampScale(value) {{
      return Math.min(16, Math.max(0.1, value));
    }}

    function applyTransform() {{
      modalImg.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
    }}

    function resetTransform() {{
      scale = 1;
      panX = 0;
      panY = 0;
      dragStart = null;
      applyTransform();
    }}

    function setModalBackground(bg) {{
      setBackground(modalStage, bg);
      swatches.forEach((swatch) => swatch.setAttribute("aria-pressed", String(swatch.dataset.bg === bg)));
    }}

    function openModal(src, label, bg) {{
      modalImg.src = src;
      modalImg.alt = label;
      modalTitle.textContent = label;
      setModalBackground(bg || "checker");
      resetTransform();
      modal.classList.add("is-open");
      modal.setAttribute("aria-hidden", "false");
    }}

    function closeModal() {{
      modal.classList.remove("is-open");
      modal.setAttribute("aria-hidden", "true");
      modalImg.removeAttribute("src");
    }}

    modalStage.addEventListener("wheel", (event) => {{
      if (!modal.classList.contains("is-open")) return;
      event.preventDefault();
      const rect = modalStage.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const pointerX = event.clientX - centerX;
      const pointerY = event.clientY - centerY;
      const previousScale = scale;
      const factor = event.deltaY < 0 ? 1.14 : 1 / 1.14;
      scale = clampScale(scale * factor);
      panX = pointerX - ((pointerX - panX) * scale) / previousScale;
      panY = pointerY - ((pointerY - panY) * scale) / previousScale;
      applyTransform();
    }}, {{ passive: false }});

    modalStage.addEventListener("pointerdown", (event) => {{
      if (!modal.classList.contains("is-open")) return;
      dragStart = {{ pointerId: event.pointerId, x: event.clientX, y: event.clientY, panX, panY }};
      modalStage.setPointerCapture(event.pointerId);
      modalStage.classList.add("is-dragging");
    }});

    modalStage.addEventListener("pointermove", (event) => {{
      if (!dragStart || dragStart.pointerId !== event.pointerId) return;
      panX = dragStart.panX + event.clientX - dragStart.x;
      panY = dragStart.panY + event.clientY - dragStart.y;
      applyTransform();
    }});

    function endDrag(event) {{
      if (!dragStart || dragStart.pointerId !== event.pointerId) return;
      dragStart = null;
      modalStage.classList.remove("is-dragging");
    }}

    modalStage.addEventListener("pointerup", endDrag);
    modalStage.addEventListener("pointercancel", endDrag);
    modalStage.addEventListener("dblclick", resetTransform);
    closeModalButton.addEventListener("click", closeModal);
    resetPreviewButton.addEventListener("click", resetTransform);
    swatches.forEach((swatch) => swatch.addEventListener("click", () => setModalBackground(swatch.dataset.bg)));
    closeCompareButton.addEventListener("click", closeCompare);
    compareViewOne.addEventListener("change", updateCompareImages);
    compareViewTwo.addEventListener("change", updateCompareImages);
    compareAlphaOne.addEventListener("input", updateCompareAlpha);
    compareAlphaTwo.addEventListener("input", updateCompareAlpha);
    compareBgButtons.forEach((button) => button.addEventListener("click", () => setCompareBackground(button.dataset.bg)));
    compareFrame.addEventListener("pointermove", updateCompareFromPointer);
    compareFrame.addEventListener("pointerdown", (event) => {{
      updateCompareFromPointer(event);
      compareFrame.setPointerCapture(event.pointerId);
    }});
    compareModal.addEventListener("click", (event) => {{
      if (event.target === compareModal) closeCompare();
    }});
    startFullEvalButton.addEventListener("click", openEvalPanel);
    selectAllSamplesButton.addEventListener("click", () => setAllSamples(true));
    clearAllSamplesButton.addEventListener("click", () => setAllSamples(false));
    if (testPathSelect) testPathSelect.addEventListener("change", updateSelectionCount);
    cancelEvalPanelButton.addEventListener("click", closeEvalPanel);
    confirmStartEvalButton.addEventListener("click", startSelectedEval);
    evalPanel.addEventListener("click", (event) => {{
      if (event.target === evalPanel) closeEvalPanel();
    }});
    runSelect.addEventListener("change", () => {{
      if (runSelect.value) {{
        window.location.href = `/eval/game?run=${{encodeURIComponent(runSelect.value)}}`;
      }}
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "Escape") closeModal();
      if (event.key === "Escape") closeCompare();
      if (event.key === "Escape") closeEvalPanel();
    }});

    renderRows();
    refreshRuntimeStatus();
  </script>
</body>
</html>"""


def main() -> None:
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover - exercised only without web extra
        raise ImportError('Install the web extra with `uv pip install -e ".[web]"`.') from e

    uvicorn.run("ermbg.web:app", host="127.0.0.1", port=7860, reload=False)


__all__ = ["app", "main"]
