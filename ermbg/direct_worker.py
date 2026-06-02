"""Direct ERMBG worker path that bypasses ComfyUI prompt execution.

This module is a test/validation path for running the same route families in a
plain Python worker process on the remote GPU host. It intentionally does not
change the production Web/API/Comfy path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import io as ermbg_io
from .api import (
    MatteResponse,
    _corridorkey_shadow_patch,
    _matte_image_passthrough,
    _matte_image_pymatting_known_b,
)
from .corridorkey_runner import LocalCorridorKeyClient
from .known_bg_glow import matte_known_bg_glow
from .qa import run_qa
from .router import RouteDecision, classify_route


@dataclass(frozen=True)
class DirectWorkerResult:
    response: MatteResponse
    timings: dict[str, float]
    metadata: dict[str, Any]


def _route_params(params: dict[str, Any], key: str, fallback: Any) -> Any:
    return params.get(key, fallback)


class DirectCorridorKeyClientFactory:
    def __init__(self) -> None:
        self._client: Any | None = None

    def get(self) -> Any:
        if self._client is None:
            self._client = DirectCorridorKeyClient()
        return self._client


class DirectCorridorKeyClient(LocalCorridorKeyClient):
    def __init__(self) -> None:
        super().__init__(
            backend_label="direct-corridorkey",
            prompt_id="direct-worker",
            prefer_loaded_node=True,
        )


def _ck_background_color(analysis: dict[str, Any], fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    raw = analysis.get("background_color")
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return tuple(int(np.clip(c, 0, 255)) for c in raw)
    return fallback


def _rgb_param(params: dict[str, Any], key: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    raw = params.get(key)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return tuple(int(np.clip(c, 0, 255)) for c in raw)
    return fallback


def matte_known_bg_glow_direct(
    rgb: np.ndarray,
    *,
    params: dict[str, Any],
    bg_color: tuple[int, int, int],
    qa: bool = False,
    auto_route: dict[str, Any] | None = None,
) -> MatteResponse:
    selected_bg_color = _rgb_param(params, "known_bg_glow_bg_color", bg_color)
    target_color = _rgb_param(params, "known_bg_glow_target_color", (255, 255, 255))
    mode = str(params.get("known_bg_glow_mode") or "single_target_line")
    result = matte_known_bg_glow(rgb, selected_bg_color, target_color, mode=mode)
    report: dict[str, Any] = {
        "diagnosis": None,
        "background_color": list(selected_bg_color),
        "despill_method": "known_bg_glow_line_solver",
        "matting_model": "none",
        "keyer": {},
        "shadow": {"mode": "off", "applied": False, "reason": "glow route has no shadow layer"},
        "semantic_prior": {},
        "strategy": {
            "name": "direct_known_bg_glow",
            "bg_type": "known_background",
            "image_type": "glow_icon",
            "keyer_mode": "known_bg_glow",
            "despill": "line_unmix",
            "passthrough": False,
            "notes": "Simple glow solved directly from a known background mixing line.",
            "extras": result.debug,
        },
    }
    if auto_route is not None:
        report["auto_route"] = auto_route
    if qa:
        report["qa"] = run_qa(
            image_srgb=rgb,
            rgba=result.rgba,
            soft_mask=result.alpha,
            background_color=selected_bg_color,
            out_dir=Path("/tmp/_ermbg_direct_worker_qa_discard"),
        )
    debug = {
        "backend": "direct-known-bg-glow",
        "known_bg_glow": result.debug,
        "strategy": report["strategy"],
        "soft_mask": result.alpha,
    }
    if auto_route is not None:
        debug["auto_route"] = auto_route
    return MatteResponse(
        rgba=result.rgba,
        alpha=result.alpha,
        foreground_srgb=result.foreground_srgb,
        strategy_name="direct_known_bg_glow",
        background_color=selected_bg_color,
        report=report,
        output_dir=None,
        debug=debug,
    )


def matte_corridorkey_direct(
    rgb: np.ndarray,
    *,
    corridorkey_analysis: dict[str, Any],
    params: dict[str, Any],
    bg_color: tuple[int, int, int],
    shadow_mode: str,
    qa: bool = False,
    auto_route: dict[str, Any] | None = None,
    hint_alpha: np.ndarray | None = None,
    hard_ui_hint_mode: str = "bbox_2px",
    execution_profile: str = "auto",
    corridorkey_client: Any | None = None,
) -> MatteResponse:
    """Execute CorridorKey directly from an already-computed route analysis.

    The production Comfy path analyzes the asset inside route selection and then
    analyzes it again inside the CorridorKey executor. This test path keeps the
    analysis single-pass and reuses the route metadata to remove that duplicate
    CPU work before the GPU refine call.
    """
    from .probe.comfyui_corridorkey import (
        build_hard_ui_boundary_corridorkey_hint,
        build_hard_ui_corridorkey_hint,
        build_hard_ui_shadow_safe_material_alpha_floor,
        build_hard_ui_shadow_safe_solid_interior_mask,
        build_hard_ui_solid_interior_mask,
    )

    hard_ui_hint_modes = {
        "all_white",
        "bbox_2px",
        "boundary_2px",
        "boundary_2px_shadow_safe",
        "boundary_2px_shadow_safe_edge_floor",
        "translucent_button",
    }
    hard_ui_hint_mode = str(params.get("corridorkey_hard_ui_hint_mode", hard_ui_hint_mode))
    if hard_ui_hint_mode not in hard_ui_hint_modes:
        raise ValueError(
            "corridorkey_hard_ui_hint_mode must be all_white, bbox_2px, boundary_2px, "
            "boundary_2px_shadow_safe, boundary_2px_shadow_safe_edge_floor, or translucent_button"
        )

    selected_bg_color = _ck_background_color(corridorkey_analysis, bg_color)
    screen_mode = str(corridorkey_analysis.get("screen_mode") or params.get("corridorkey_screen_mode") or "auto")
    parameter_profile = str(corridorkey_analysis.get("parameter_profile") or "")
    execution_profile = str(params.get("corridorkey_execution_profile") or params.get("execution_profile") or execution_profile)
    if execution_profile == "auto":
        if hard_ui_hint_mode == "translucent_button":
            execution_profile = "corridorkey-transparent-button"
        elif parameter_profile == "composite_character_corridor_only":
            execution_profile = "corridorkey-character"
        elif parameter_profile == "translucent_button":
            execution_profile = "corridorkey-transparent-button"
        elif parameter_profile == "screen_tinted_translucency":
            execution_profile = "corridorkey-effect-icon"
        else:
            execution_profile = "corridorkey-shaped-icon"
    gamma_space = str(params.get("corridorkey_gamma_space", "sRGB"))
    despill_strength = float(params.get("corridorkey_despill_strength", 1.0))
    refiner_strength = float(params.get("corridorkey_refiner_strength", 1.0))
    auto_despeckle = str(params.get("corridorkey_auto_despeckle", "On"))
    despeckle_size = int(params.get("corridorkey_despeckle_size", 400))
    apply_color_protection = bool(params.get("corridorkey_color_protection", True))
    color_protection_bg_max = float(params.get("corridorkey_protection_bg_max", 12.0))
    color_protection_fg_min = float(params.get("corridorkey_protection_fg_min", 28.0))
    auto_mask = bool(params.get("corridorkey_auto_mask", False))

    forced_translucent_hint_settings = False
    if execution_profile in {
        "corridorkey-transparent-button",
        "corridorkey-effect-icon",
    }:
        forced_translucent_hint_settings = parameter_profile not in {
            "translucent_button",
            "screen_tinted_translucency",
        }
        gamma_space = "sRGB"
        despill_strength = 1.0
        refiner_strength = 1.15
        auto_despeckle = "Off"
        despeckle_size = 64
        apply_color_protection = False
        color_protection_bg_max = 6.0
        color_protection_fg_min = 14.0
    if execution_profile == "corridorkey-character":
        gamma_space = "sRGB"
        despill_strength = 1.0
        refiner_strength = 1.0
        auto_despeckle = "Off"
        despeckle_size = 64
        apply_color_protection = False
        color_protection_bg_max = 6.0
        color_protection_fg_min = 14.0

    client = corridorkey_client if corridorkey_client is not None else DirectCorridorKeyClient()
    hint_source = None
    solid_interior_mask: np.ndarray | None = None
    material_alpha_floor: np.ndarray | None = None
    solid_interior_info: dict[str, Any] = {}
    if hint_alpha is not None:
        hint_source = "provided_corridorkey_hint_mask"
    elif not auto_mask or hard_ui_hint_mode == "all_white":
        hint_alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        hint_source = "all_white_alpha_hint"
    elif execution_profile == "corridorkey-transparent-button":
        hint_alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        hint_source = "glass_all_white_corridorkey_hint"
    elif execution_profile == "corridorkey-character":
        hint_alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        hint_source = "character_all_white_corridorkey_hint"
    elif execution_profile == "corridorkey-effect-icon":
        hint_alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        hint_source = "effect_all_white_corridorkey_hint"
    elif parameter_profile.startswith("opaque_hard_ui"):
        if hard_ui_hint_mode == "boundary_2px":
            hint_alpha = build_hard_ui_boundary_corridorkey_hint(rgb, selected_bg_color)
            solid_interior_mask = build_hard_ui_solid_interior_mask(rgb, selected_bg_color)
            hint_source = "known_bg_hard_ui_boundary_2px_hint"
        elif hard_ui_hint_mode == "boundary_2px_shadow_safe":
            hint_alpha = build_hard_ui_boundary_corridorkey_hint(rgb, selected_bg_color)
            solid_interior_mask, solid_interior_info = build_hard_ui_shadow_safe_solid_interior_mask(
                rgb,
                selected_bg_color,
            )
            hint_source = "known_bg_hard_ui_boundary_2px_shadow_safe_hint"
        elif hard_ui_hint_mode == "boundary_2px_shadow_safe_edge_floor":
            hint_alpha = build_hard_ui_boundary_corridorkey_hint(rgb, selected_bg_color)
            material_alpha_floor, solid_interior_info = build_hard_ui_shadow_safe_material_alpha_floor(
                rgb,
                selected_bg_color,
            )
            hint_source = "known_bg_hard_ui_boundary_2px_shadow_safe_edge_floor_hint"
        else:
            hint_alpha = build_hard_ui_corridorkey_hint(rgb, selected_bg_color)
            hint_source = "known_bg_hard_ui_bbox_2px_hint"

    remote = client.matte(
        rgb,
        background_color=selected_bg_color,
        hint_alpha=hint_alpha,
        hint_source=hint_source,
        gamma_space=gamma_space,
        screen_color=screen_mode if screen_mode in {"green", "blue"} else "auto",
        despill_strength=despill_strength,
        refiner_strength=refiner_strength,
        auto_despeckle=auto_despeckle,
        despeckle_size=despeckle_size,
        apply_color_protection=apply_color_protection,
        color_protection_bg_max=color_protection_bg_max,
        color_protection_fg_min=color_protection_fg_min,
        protect_hint_supported_material=parameter_profile == "key_color_material",
        execution_profile=execution_profile,
    )
    subject_alpha = remote.alpha
    subject_foreground_srgb = remote.foreground_srgb
    subject_rgba = remote.rgba
    solid_interior_pixels = 0
    material_floor_lift_pixels = 0
    if solid_interior_mask is not None and solid_interior_mask.any():
        interior = solid_interior_mask.astype(bool)
        solid_interior_pixels = int(interior.sum())
        subject_alpha = remote.alpha.copy()
        subject_foreground_srgb = remote.foreground_srgb.copy()
        subject_alpha[interior] = 1.0
        subject_foreground_srgb[interior] = rgb[interior]
    if material_alpha_floor is not None and (material_alpha_floor > 0.0).any():
        floor = np.clip(material_alpha_floor.astype(np.float32), 0.0, 1.0)
        lift = floor > (subject_alpha + 1e-4)
        material_floor_lift_pixels = int(lift.sum())
        solid_interior_pixels = int(solid_interior_info.get("solid_interior_pixels", int((floor >= 0.999).sum())))
        if lift.any():
            subject_alpha = np.maximum(subject_alpha, floor).astype(np.float32)
            subject_foreground_srgb = subject_foreground_srgb.copy()
            C_lin = ermbg_io.srgb_to_linear(rgb).astype(np.float32)
            B_lin = ermbg_io.srgb_to_linear(np.asarray(selected_bg_color, dtype=np.uint8).reshape(1, 1, 3))[
                0,
                0,
            ].astype(np.float32)
            recovered = C_lin.copy()
            a = np.maximum(floor[lift, None], 1e-3)
            recovered[lift] = (C_lin[lift] - (1.0 - floor[lift, None]) * B_lin.reshape(1, 3)) / a
            recovered = np.clip(recovered, 0.0, 1.0).astype(np.float32)
            recovered_srgb = ermbg_io.linear_to_srgb_u8(recovered)
            subject_foreground_srgb[lift] = recovered_srgb[lift]
    if (
        (solid_interior_mask is not None and solid_interior_mask.any())
        or (material_alpha_floor is not None and (material_alpha_floor > 0.0).any())
    ):
        subject_rgba = np.dstack(
            [
                subject_foreground_srgb,
                (np.clip(subject_alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
            ]
        )

    alpha, rgba_rgb_srgb, shadow_alpha, shadow_alpha_physical, shadow_info = _corridorkey_shadow_patch(
        rgb,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=subject_foreground_srgb,
        background_color=selected_bg_color,
        shadow_mode=shadow_mode,
    )
    rgba = np.dstack(
        [
            rgba_rgb_srgb,
            (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        ]
    )
    shadow_rgba = np.dstack(
        [
            np.zeros(rgb.shape, dtype=np.uint8),
            (np.clip(shadow_alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8),
        ]
    )
    bg_type = f"saturated_{screen_mode}" if screen_mode in {"green", "blue"} else "unknown_screen"
    image_type = f"ai_{screen_mode}_asset" if screen_mode in {"green", "blue"} else "ai_screen_asset"
    report: dict[str, Any] = {
        "diagnosis": None,
        "background_color": list(selected_bg_color),
        "despill_method": "direct_corridorkey",
        "matting_model": "CorridorKey",
        "corridorkey_analysis": corridorkey_analysis,
        "keyer": {
            "used": True,
            "source": remote.debug.get("hint", {}).get("source") or hint_source or "known_bg_chromatic_key_alpha_hint",
            "hint": remote.debug.get("hint", {}),
        },
        "shadow": shadow_info,
        "semantic_prior": {},
        "strategy": {
            "name": "direct_corridorkey",
            "bg_type": bg_type,
            "image_type": image_type,
            "keyer_mode": "corridorkey",
            "despill": "direct_corridorkey",
            "passthrough": False,
            "notes": "CorridorKey executed in a direct worker without ComfyUI prompt/queue overhead.",
            "extras": remote.debug,
        },
    }
    if auto_route is not None:
        report["auto_route"] = auto_route
    if qa:
        report["qa"] = run_qa(
            image_srgb=rgb,
            rgba=rgba,
            soft_mask=alpha,
            background_color=selected_bg_color,
            out_dir=Path("/tmp/_ermbg_direct_worker_qa_discard"),
        )

    debug = {
        **remote.debug,
        "strategy": report["strategy"],
        "corridorkey_analysis": corridorkey_analysis,
        "soft_mask": alpha,
        "subject_alpha": subject_alpha,
        "corridorkey_subject_rgba": subject_rgba,
        "corridorkey_hint": remote.hint_alpha,
        "corridorkey_raw_alpha": remote.raw_alpha,
        "key_color_protection": remote.color_protection_alpha,
        "shadow_alpha": shadow_alpha,
        "shadow_alpha_physical": shadow_alpha_physical,
        "shadow_layer_rgba": shadow_rgba,
        "shadow": shadow_info,
        "hard_ui_hint": {
            "mode": hard_ui_hint_mode,
            "execution_profile": execution_profile,
            "forced_translucent_settings": forced_translucent_hint_settings,
            "solid_interior_pixels": solid_interior_pixels,
            "material_floor_lift_pixels": material_floor_lift_pixels,
            **solid_interior_info,
        },
    }
    if auto_route is not None:
        debug["auto_route"] = auto_route
    return MatteResponse(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=subject_foreground_srgb,
        strategy_name="direct_corridorkey",
        background_color=selected_bg_color,
        report=report,
        output_dir=None,
        debug=debug,
    )


def direct_matte_from_decision(
    rgb: np.ndarray,
    *,
    decision: RouteDecision,
    source_alpha: np.ndarray | None = None,
    shadow_mode: str = "auto",
    corridorkey_hard_ui_hint_mode: str = "bbox_2px",
    fallback_bg_color: tuple[int, int, int] = (0, 200, 0),
    ck_factory: DirectCorridorKeyClientFactory | None = None,
    route_sec: float | None = None,
) -> DirectWorkerResult:
    """Execute a precomputed route decision in the direct worker.

    Batch CPU scheduling uses this to classify once, then run CPU-owned
    PyMatting cases in separate worker processes without sending CorridorKey
    work to multiple GPU-owning processes.
    """
    timings: dict[str, float] = {}
    if route_sec is not None:
        timings["route_sec"] = float(route_sec)
    auto_route = decision.to_dict()
    selected_backend = decision.backend
    params = dict(decision.params)
    shadow_mode = str(shadow_mode or "auto").strip().lower()
    if shadow_mode not in {"auto", "on", "off"}:
        raise ValueError("shadow_mode must be 'auto', 'on', or 'off'")

    t = time.perf_counter()
    if selected_backend == "passthrough":
        response = _matte_image_passthrough(
            rgb,
            source_alpha,
            src_path=None,
            output_dir=None,
            qa=False,
            auto_route=auto_route,
        )
        response.strategy_name = "direct_passthrough"
        response.debug["backend"] = "direct-passthrough"
        execution_backend = "direct-passthrough"
    elif selected_backend == "comfy-pymatting-known-b":
        response = _matte_image_pymatting_known_b(
            rgb,
            src_path=None,
            output_dir=None,
            qa=False,
            shadow_mode=shadow_mode,
            method=_route_params(params, "pymatting_method", "cf"),
            image_space=_route_params(params, "pymatting_image_space", "linear"),
            bg_source=_route_params(params, "pymatting_bg_source", "custom"),
            bg_color=_route_params(params, "pymatting_bg_color", fallback_bg_color),
            bg_threshold=_route_params(params, "pymatting_bg_threshold", 3.5),
            fg_threshold=_route_params(params, "pymatting_fg_threshold", 24.0),
            boundary_band_px=_route_params(params, "pymatting_boundary_band_px", 2),
            auto_adapt=_route_params(params, "pymatting_auto_adapt", True),
            cg_maxiter=_route_params(params, "pymatting_cg_maxiter", 1000),
            cg_rtol=_route_params(params, "pymatting_cg_rtol", 1e-6),
            auto_route=auto_route,
        )
        response.strategy_name = "direct_pymatting_known_b"
        response.debug["backend"] = "direct-pymatting-known-b"
        execution_backend = "direct-pymatting-known-b"
    elif selected_backend == "direct-known-bg-glow":
        response = matte_known_bg_glow_direct(
            rgb,
            params=params,
            bg_color=fallback_bg_color,
            qa=False,
            auto_route=auto_route,
        )
        execution_backend = "direct-known-bg-glow"
    elif selected_backend == "comfy-corridorkey":
        ck_analysis = decision.analysis.get("corridorkey_analysis")
        if not isinstance(ck_analysis, dict):
            raise RuntimeError("CorridorKey direct route requires corridorkey_analysis metadata")
        response = matte_corridorkey_direct(
            rgb,
            corridorkey_analysis=ck_analysis,
            params=params,
            bg_color=fallback_bg_color,
            shadow_mode=shadow_mode,
            hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
            execution_profile=_route_params(params, "corridorkey_execution_profile", "auto"),
            corridorkey_client=(ck_factory or DirectCorridorKeyClientFactory()).get(),
            auto_route=auto_route,
        )
        response.debug["backend"] = "direct-corridorkey"
        execution_backend = "direct-corridorkey"
    else:
        raise RuntimeError(f"Unsupported direct backend: {selected_backend}")
    timings["backend_sec"] = time.perf_counter() - t
    metadata = {
        "selected_backend": selected_backend,
        "execution_backend": execution_backend,
        "route": auto_route.get("route"),
        "asset_kind": auto_route.get("asset_kind"),
        "parameter_profile": auto_route.get("parameter_profile"),
        "execution_profile": auto_route.get("execution_profile"),
        "shadow_mode": shadow_mode,
    }
    if execution_backend == "direct-corridorkey":
        hard_ui_hint = response.debug.get("hard_ui_hint") if isinstance(response.debug, dict) else None
        if isinstance(hard_ui_hint, dict) and isinstance(hard_ui_hint.get("execution_profile"), str):
            metadata["execution_profile"] = hard_ui_hint["execution_profile"]
    return DirectWorkerResult(response=response, timings=timings, metadata=metadata)


def direct_matte_auto(
    rgb: np.ndarray,
    *,
    source_alpha: np.ndarray | None = None,
    shadow_mode: str = "auto",
    corridorkey_screen_mode: str = "auto",
    corridorkey_preset: str = "auto",
    corridorkey_hard_ui_hint_mode: str = "bbox_2px",
    fallback_bg_color: tuple[int, int, int] = (0, 200, 0),
    ck_factory: DirectCorridorKeyClientFactory | None = None,
) -> DirectWorkerResult:
    """Run the isolated direct worker auto route on one RGB image."""
    t = time.perf_counter()
    decision = classify_route(
        rgb,
        source_alpha=source_alpha,
        screen_mode=corridorkey_screen_mode,  # type: ignore[arg-type]
        preset=corridorkey_preset,  # type: ignore[arg-type]
        fallback_background_color=fallback_bg_color,
    )
    route_sec = time.perf_counter() - t
    return direct_matte_from_decision(
        rgb,
        decision=decision,
        source_alpha=source_alpha,
        shadow_mode=shadow_mode,
        corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
        fallback_bg_color=fallback_bg_color,
        ck_factory=ck_factory,
        route_sec=route_sec,
    )


__all__ = [
    "DirectCorridorKeyClient",
    "DirectCorridorKeyClientFactory",
    "DirectWorkerResult",
    "direct_matte_from_decision",
    "direct_matte_auto",
    "matte_known_bg_glow_direct",
    "matte_corridorkey_direct",
]
