"""Run CorridorKey on the remote ComfyUI server for clean green-screen assets."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image

from ermbg.comfy import DEFAULT_COMFY_URL
from ermbg.colorspace import oklab_distance, srgb_to_oklab
from ermbg.keyer import KeyerThresholds, chromatic_key_alpha
from ermbg.shadow import ShadowThresholds, exterior_scalar_darkening_mask

_DEFAULT_WORKFLOW = Path(__file__).parent / "comfyui_corridorkey.json"
_FOREGROUND_NODE = "30"
_ALPHA_NODE = "50"
_PROCESSED_NODE = "60"
_QC_NODE = "70"


@dataclass(frozen=True)
class ComfyCorridorKeyResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground_srgb: np.ndarray
    hint_alpha: np.ndarray
    raw_alpha: np.ndarray
    color_protection_alpha: np.ndarray
    debug: dict[str, Any]


def build_corridorkey_hint(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    *,
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Build CorridorKey's coarse alpha hint from known green-screen evidence.

    CorridorKey is not a generic segmenter; it expects a rough foreground
    ownership hint. For high-confidence AI green-screen assets, direct known-B
    chroma distance gives that hint without running a second neural model.
    The support is slightly eroded and blurred because the model's own docs
    prefer a soft, under-expanded hint over an exact or over-grown mask.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("build_corridorkey_hint() expects HxWx3 sRGB uint8")

    raw = chromatic_key_alpha(
        image_srgb,
        background_color,
        thresholds or KeyerThresholds(bg_max=4.0, fg_min=18.0),
    )
    h, w = raw.shape
    if not np.any(raw > 0.18):
        return raw.astype(np.float32)

    # Empirical, signal-based values: the threshold accepts pixels with clear
    # non-background chroma evidence; the one-pixel erosion protects against
    # known-B fringe leakage becoming model-owned foreground in the hint.
    support = (raw >= 0.18).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    if min(h, w) >= 24:
        support = cv2.erode(support, kernel, iterations=1)
    if not support.any():
        support = (raw >= 0.35).astype(np.uint8)
    if not support.any():
        return raw.astype(np.float32)

    blur_ksize = 7 if min(h, w) >= 96 else 5
    hint = cv2.GaussianBlur(support.astype(np.float32), (blur_ksize, blur_ksize), 0)
    return np.clip(hint, 0.0, 1.0).astype(np.float32)


def build_key_color_protection_floor(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    *,
    thresholds: KeyerThresholds | None = None,
) -> np.ndarray:
    """Build a soft alpha floor from key-color distance, not region ownership.

    Failure mode this protects against: CorridorKey can treat saturated UI
    colors such as yellow/orange as spill-like transparency even when the hint
    says foreground. The invariant is color based: pixels far outside the key
    color family should not be driven transparent by a learned green-screen
    prior. For saturated key colors we measure OKLab a/b distance only, so
    darker same-hue screen shadows remain key-colored instead of becoming an
    opaque protected component. Neutral screens fall back to full OKLab distance
    because their useful signal is mostly lightness.
    """
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("build_key_color_protection_floor() expects HxWx3 sRGB uint8")

    t = thresholds or KeyerThresholds(bg_max=8.0, fg_min=16.0)
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    bg_chroma = float(np.linalg.norm(bg_lab[1:]))
    if bg_chroma >= 0.04:
        # Empirical signal split: saturated screen shadows often differ mainly
        # in L while keeping the same a/b family. Protecting by chroma distance
        # preserves yellow/red/white UI material without turning dark green
        # screen shading into opaque subject.
        delta = lab[..., 1:] - bg_lab[1:]
        d = np.sqrt(np.sum(delta * delta, axis=-1)).astype(np.float32) * 100.0
        mode = "oklab_ab"
    else:
        d = oklab_distance(lab, bg_lab).astype(np.float32)
        mode = "oklab_full"
    x = np.clip((d - t.bg_max) / max(t.fg_min - t.bg_max, 1e-6), 0.0, 1.0)
    floor = x * x * (3.0 - 2.0 * x)
    return np.clip(floor, 0.0, 1.0).astype(np.float32)


def _shadow_safe_color_protection_floor(
    *,
    image_srgb: np.ndarray,
    raw_alpha: np.ndarray,
    background_color: tuple[int, int, int],
    floor: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Suppress color protection where pixels are measured screen darkening.

    Broad invariant: color distance alone is not ownership. A cast shadow on a
    saturated known background can move far enough from the key color to look
    like protected subject, while the pixel still satisfies the physical model
    ``C ~= scale * B``. The empirical gates below are intentionally loose and
    evidence-based: they only disable protection for scalar darkening connected
    to exterior background, and only where CorridorKey has not already assigned
    strong subject ownership.
    """
    known_bg = (floor <= 0.05) & (raw_alpha <= 0.20)
    shadow_like, shadow_info = exterior_scalar_darkening_mask(
        image_srgb,
        background_color,
        known_bg,
        ShadowThresholds(
            # Color protection runs before the dedicated shadow pass, so this
            # guard must also catch very light hard-shadow antialiasing. The
            # scalar reconstruction error and exterior flood are the stronger
            # counter-signals against suppressing real subject material.
            min_strength=0.01,
            max_reconstruction_error=0.07,
            reject_border_components=False,
        ),
    )
    subject_owned = raw_alpha >= 0.80
    blocked = shadow_like & (~subject_owned) & (floor > 0.0)
    applied_floor = floor.copy()
    applied_floor[blocked] = 0.0
    exterior_domain = known_bg | shadow_like
    distance_to_exterior = cv2.distanceTransform((~exterior_domain).astype(np.uint8), cv2.DIST_L2, 3)
    edge_antialias = (
        (raw_alpha >= 0.20)
        & (raw_alpha <= 0.88)
        & (floor > raw_alpha + 0.05)
        & (distance_to_exterior <= 2.0)
    )
    # Color protection is allowed to fill interior holes, but not to convert
    # CorridorKey's measured outer-edge antialiasing into full opacity. B023's
    # blue-screen hard UI edge exposed this: mixed yellow+screen pixels were
    # lifted to alpha 1.0 and became discrete dark edge dots on recomposite.
    applied_floor[edge_antialias] = 0.0
    stats = {
        "shadow_safe_enabled": True,
        "shadow_like_pixels": int(shadow_like.sum()),
        "shadow_known_background_pixels": int(known_bg.sum()),
        "shadow_candidate_pixels": int(shadow_info.get("candidate_pixels", 0)),
        "floor_shadow_blocked_pixels": int(blocked.sum()),
        "floor_shadow_blocked_mean": float(floor[blocked].mean()) if blocked.any() else 0.0,
        "floor_edge_antialias_blocked_pixels": int(edge_antialias.sum()),
        "floor_applied_mean": float(applied_floor.mean()),
    }
    return applied_floor.astype(np.float32), stats


def apply_key_color_protection(
    *,
    image_srgb: np.ndarray,
    foreground_srgb: np.ndarray,
    alpha: np.ndarray,
    background_color: tuple[int, int, int] = (0, 200, 0),
    thresholds: KeyerThresholds | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Lift model alpha where non-key colors prove the pixel is not screen.

    This deliberately avoids geometric protection masks. The floor comes only
    from color distance to the key color, so anti-aliased edge pixels blended
    toward the screen naturally get a lower floor instead of a hard region
    boundary.
    """
    t = thresholds or KeyerThresholds(bg_max=8.0, fg_min=16.0)
    raw_alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    floor = build_key_color_protection_floor(image_srgb, background_color, thresholds=t)
    applied_floor, shadow_safe_stats = _shadow_safe_color_protection_floor(
        image_srgb=image_srgb,
        raw_alpha=raw_alpha,
        background_color=background_color,
        floor=floor,
    )
    protected_alpha = np.maximum(raw_alpha, applied_floor).astype(np.float32)
    lift = np.clip(protected_alpha - raw_alpha, 0.0, 1.0)
    blend = lift / np.maximum(protected_alpha, 1e-6)
    protected_fg = (
        foreground_srgb.astype(np.float32) * (1.0 - blend[..., None])
        + image_srgb.astype(np.float32) * blend[..., None]
    )
    stats = {
        "source": "key_color_distance_floor",
        "mode": "auto",
        "thresholds": {
            "bg_max": float(t.bg_max),
            "fg_min": float(t.fg_min),
        },
        "floor_min": float(floor.min()),
        "floor_max": float(floor.max()),
        "floor_mean": float(floor.mean()),
        "floor_applied_min": float(applied_floor.min()),
        "floor_applied_max": float(applied_floor.max()),
        "floor_applied_mean": float(applied_floor.mean()),
        "lifted_pixels_gt_01": int((lift > 0.01).sum()),
        "alpha_mean_before": float(raw_alpha.mean()),
        "alpha_mean_after": float(protected_alpha.mean()),
        **shadow_safe_stats,
    }
    return (
        np.clip(protected_fg + 0.5, 0, 255).astype(np.uint8),
        np.clip(protected_alpha, 0.0, 1.0).astype(np.float32),
        applied_floor,
        stats,
    )


class ComfyUICorridorKeyClient:
    """Submit a CorridorKey workflow to a remote ComfyUI server."""

    def __init__(
        self,
        url: str = DEFAULT_COMFY_URL,
        workflow_path: Path | str | None = None,
        timeout: float = 600.0,
        poll_interval: float = 0.25,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.timeout = timeout
        self.poll_interval = poll_interval
        path = Path(workflow_path) if workflow_path else _DEFAULT_WORKFLOW
        self.workflow_template = json.loads(path.read_text())

    def _post(self, path: str, **kwargs):
        r = requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def _get(self, path: str, **kwargs):
        r = requests.get(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def _upload(self, image: np.ndarray, name: str) -> str:
        buf = BytesIO()
        if image.ndim == 2:
            Image.fromarray(image, mode="L").save(buf, format="PNG")
        else:
            Image.fromarray(image).save(buf, format="PNG")
        buf.seek(0)
        files = {"image": (name, buf, "image/png")}
        data = {"overwrite": "true"}
        return self._post("/upload/image", files=files, data=data).json()["name"]

    def _queue(self, workflow: dict[str, Any]) -> str:
        body = {"prompt": workflow, "client_id": self.client_id}
        result = self._post("/prompt", json=body).json()
        if "prompt_id" not in result:
            raise RuntimeError(f"Comfy /prompt rejected: {result}")
        return result["prompt_id"]

    def _wait(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            data = self._get(f"/history/{prompt_id}").json()
            if prompt_id in data:
                entry = data[prompt_id]
                status = entry.get("status", {})
                if status.get("completed", False):
                    return entry
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Comfy workflow errored: {entry.get('status')}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {self.timeout}s")

    def _download_node_image(self, history_entry: dict[str, Any], node_id: str, mode: str) -> np.ndarray:
        node_out = history_entry.get("outputs", {}).get(str(node_id), {})
        images = node_out.get("images", [])
        if not images:
            raise RuntimeError(f"No ComfyUI image output found for node {node_id}")
        img_meta = images[0]
        params = {
            "filename": img_meta["filename"],
            "subfolder": img_meta.get("subfolder", ""),
            "type": img_meta.get("type", "output"),
        }
        r = self._get("/view", params=params)
        im = Image.open(BytesIO(r.content)).convert(mode)
        return np.asarray(im, dtype=np.uint8)

    def _render_workflow(
        self,
        *,
        input_image: str,
        mask_image: str,
        gamma_space: str,
        screen_color: str,
        despill_strength: float,
        refiner_strength: float,
        auto_despeckle: str,
        despeckle_size: int,
        filename_prefix: str,
    ) -> dict[str, Any]:
        rendered = Template(json.dumps(self.workflow_template)).safe_substitute(
            input_image=input_image,
            mask_image=mask_image,
            gamma_space=gamma_space,
            screen_color=screen_color,
            despill_strength=float(despill_strength),
            refiner_strength=float(refiner_strength),
            auto_despeckle=auto_despeckle,
            despeckle_size=int(despeckle_size),
            filename_prefix=json.dumps(filename_prefix)[1:-1],
        )
        workflow = json.loads(rendered)
        workflow.pop("_comment", None)
        inputs = workflow["20"]["inputs"]
        inputs["screen_color"] = str(inputs["screen_color"])
        inputs["despill_strength"] = float(inputs["despill_strength"])
        inputs["refiner_strength"] = float(inputs["refiner_strength"])
        inputs["despeckle_size"] = int(inputs["despeckle_size"])
        return workflow

    def matte(
        self,
        image_srgb: np.ndarray,
        *,
        background_color: tuple[int, int, int] = (0, 200, 0),
        hint_alpha: np.ndarray | None = None,
        gamma_space: str = "sRGB",
        screen_color: str = "green",
        despill_strength: float = 1.0,
        refiner_strength: float = 1.0,
        auto_despeckle: str = "On",
        despeckle_size: int = 400,
        hint_source: str | None = None,
        apply_color_protection: bool = True,
        color_protection_bg_max: float = 12.0,
        color_protection_fg_min: float = 28.0,
    ) -> ComfyCorridorKeyResult:
        if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
            raise ValueError("matte() expects HxWx3 sRGB uint8")
        if gamma_space not in {"sRGB", "Linear"}:
            raise ValueError("gamma_space must be 'sRGB' or 'Linear'")
        if screen_color not in {"green", "blue", "auto"}:
            raise ValueError("screen_color must be 'green', 'blue', or 'auto'")
        if auto_despeckle not in {"On", "Off"}:
            raise ValueError("auto_despeckle must be 'On' or 'Off'")
        if color_protection_fg_min <= color_protection_bg_max:
            raise ValueError("color_protection_fg_min must be greater than color_protection_bg_max")

        h, w = image_srgb.shape[:2]
        if hint_alpha is None:
            hint_alpha = build_corridorkey_hint(image_srgb, background_color)
            hint_source = hint_source or "known_bg_chromatic_key_eroded_blur"
        else:
            hint_source = hint_source or "provided_alpha_hint"
        if hint_alpha.shape != (h, w):
            raise ValueError("hint_alpha must have shape HxW matching image_srgb")
        hint = np.clip(hint_alpha.astype(np.float32), 0.0, 1.0)
        hint_u8 = np.clip(hint * 255.0 + 0.5, 0, 255).astype(np.uint8)

        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        prefix = f"ermbg_corridorkey_{uuid.uuid4().hex[:8]}"
        step_start = time.perf_counter()
        server_image = self._upload(image_srgb, f"{prefix}.png")
        server_mask = self._upload(hint_u8, f"{prefix}_hint.png")
        timings["upload_sec"] = time.perf_counter() - step_start
        workflow = self._render_workflow(
            input_image=server_image,
            mask_image=server_mask,
            gamma_space=gamma_space,
            screen_color=screen_color,
            despill_strength=despill_strength,
            refiner_strength=refiner_strength,
            auto_despeckle=auto_despeckle,
            despeckle_size=despeckle_size,
            filename_prefix=prefix,
        )
        step_start = time.perf_counter()
        prompt_id = self._queue(workflow)
        timings["queue_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        history = self._wait(prompt_id)
        timings["wait_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        foreground = self._download_node_image(history, _FOREGROUND_NODE, "RGB")
        timings["download_foreground_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        alpha_u8 = self._download_node_image(history, _ALPHA_NODE, "L")
        timings["download_alpha_sec"] = time.perf_counter() - step_start

        if foreground.shape[:2] != (h, w):
            foreground = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if alpha_u8.shape != (h, w):
            alpha_u8 = cv2.resize(alpha_u8, (w, h), interpolation=cv2.INTER_LINEAR)

        raw_alpha = alpha_u8.astype(np.float32) / 255.0
        color_protection = np.zeros((h, w), dtype=np.float32)
        protection_debug: dict[str, Any] = {"enabled": False}
        alpha = raw_alpha
        if apply_color_protection:
            protection_thresholds = KeyerThresholds(
                bg_max=float(color_protection_bg_max),
                fg_min=float(color_protection_fg_min),
            )
            foreground, alpha, color_protection, protection_stats = apply_key_color_protection(
                image_srgb=image_srgb,
                foreground_srgb=foreground,
                alpha=raw_alpha,
                background_color=background_color,
                thresholds=protection_thresholds,
            )
            protection_debug = {"enabled": True, **protection_stats}
        alpha_u8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        rgba = np.dstack([foreground, alpha_u8]).astype(np.uint8)
        timings["total_sec"] = time.perf_counter() - total_start
        return ComfyCorridorKeyResult(
            rgba=rgba,
            alpha=np.clip(alpha, 0.0, 1.0).astype(np.float32),
            foreground_srgb=foreground.astype(np.uint8),
            hint_alpha=hint.astype(np.float32),
            raw_alpha=np.clip(raw_alpha, 0.0, 1.0).astype(np.float32),
            color_protection_alpha=color_protection.astype(np.float32),
            debug={
                "backend": "comfy-corridorkey",
                "prompt_id": prompt_id,
                "server_image": server_image,
                "server_mask": server_mask,
                "foreground_node": _FOREGROUND_NODE,
                "alpha_node": _ALPHA_NODE,
                "processed_node": _PROCESSED_NODE,
                "qc_node": _QC_NODE,
                "background_color": list(background_color),
                "settings": {
                    "gamma_space": gamma_space,
                    "screen_color": screen_color,
                    "despill_strength": float(despill_strength),
                    "refiner_strength": float(refiner_strength),
                    "auto_despeckle": auto_despeckle,
                    "despeckle_size": int(despeckle_size),
                    "apply_color_protection": bool(apply_color_protection),
                },
                "hint": {
                    "source": hint_source,
                    "min": float(hint.min()),
                    "max": float(hint.max()),
                    "mean": float(hint.mean()),
                },
                "color_protection": protection_debug,
                "timings": timings,
            },
        )


__all__ = [
    "ComfyCorridorKeyResult",
    "ComfyUICorridorKeyClient",
    "apply_key_color_protection",
    "build_key_color_protection_floor",
    "build_corridorkey_hint",
]
