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

from .api import (
    MatteResponse,
    _matte_image_passthrough,
    _matte_image_pymatting_known_b,
    prepare_known_b_preprocessed_input,
)
from .corridorkey_hint import corridorkey_full_frame_prior_value
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


def _algorithm_from_backend(value: str) -> str:
    return "rgba_passthrough" if value == "passthrough" else value


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


def _corridorkey_default_hint(
    rgb: np.ndarray,
    *,
    selected_bg_color: tuple[int, int, int],
    execution_profile: str,
    screen_mode: str,
    source_prefix: str,
) -> tuple[np.ndarray, str]:
    prior_value, prior_kind = corridorkey_full_frame_prior_value(
        execution_profile=execution_profile,
        screen_mode=screen_mode,
    )
    return (
        np.full(rgb.shape[:2], prior_value, dtype=np.float32),
        f"{source_prefix}_full_frame_{prior_kind}_corridorkey_hint",
    )


def _corridorkey_hint_value_from_semantic_decision(semantic_decision: Any) -> float | None:
    if not isinstance(semantic_decision, dict) or "corridorkey_hint_value" not in semantic_decision:
        return None
    value = float(semantic_decision["corridorkey_hint_value"])
    if not np.isfinite(value):
        raise ValueError("corridorkey_hint_value must be finite")
    return float(np.clip(value, 0.0, 1.0))


def _corridorkey_disabled_postprocess_info(params: dict[str, Any]) -> dict[str, Any]:
    semantic_decision = params.get("semantic_decision")
    return {
        "semantic_decision": dict(semantic_decision) if isinstance(semantic_decision, dict) else {},
        "semantic_decision_applied": False,
        "semantic_decision_reason": "corridorkey_path_uses_hint_strength_only_no_post_alpha_constraints",
        "semantic_hint_value": _corridorkey_hint_value_from_semantic_decision(semantic_decision),
        "keep_floor_pixels": 0,
        "alpha_cap_pixels": 0,
        "remove_pixels": 0,
        "user_masks_applied": False,
    }


def _corridorkey_disabled_shadow_info(shadow_mode: str) -> dict[str, Any]:
    return {
        "mode": str(shadow_mode or "off"),
        "applied": False,
        "reason": "corridorkey_path_returns_raw_model_output_without_shadow_patch",
    }


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
    material_strength = float(np.clip(float(params.get("known_bg_glow_material_strength", 1.0)), 0.0, 2.0))
    result = matte_known_bg_glow(
        rgb,
        selected_bg_color,
        target_color,
        mode=mode,
        material_strength=material_strength,
    )
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
    execution_profile: str = "auto",
    corridorkey_client: Any | None = None,
) -> MatteResponse:
    """Execute CorridorKey directly from an already-computed route analysis.

    The production Comfy path analyzes the asset inside route selection and then
    analyzes it again inside the CorridorKey executor. This test path keeps the
    analysis single-pass and reuses the route metadata to remove that duplicate
    CPU work before the GPU refine call.
    """
    selected_bg_color = _ck_background_color(corridorkey_analysis, bg_color)
    screen_mode = str(corridorkey_analysis.get("screen_mode") or params.get("corridorkey_screen_mode") or "auto")
    parameter_profile = str(corridorkey_analysis.get("parameter_profile") or "")
    execution_profile = str(params.get("corridorkey_execution_profile") or params.get("execution_profile") or execution_profile)
    if execution_profile == "auto":
        if parameter_profile == "composite_character_corridor_only":
            execution_profile = "corridorkey-character"
        elif parameter_profile == "translucent_button":
            execution_profile = "corridorkey-transparent-button"
        elif parameter_profile == "screen_tinted_translucency":
            execution_profile = "corridorkey-effect-icon"
        else:
            execution_profile = "corridorkey-shaped-icon"
    preset = str(params.get("corridorkey_preset") or "auto")
    gamma_space = str(params.get("corridorkey_gamma_space", "sRGB"))
    despill_strength = float(params.get("corridorkey_despill_strength", 1.0))
    refiner_strength = float(params.get("corridorkey_refiner_strength", 1.0))
    auto_despeckle = str(params.get("corridorkey_auto_despeckle", "On"))
    despeckle_size = int(params.get("corridorkey_despeckle_size", 400))
    auto_mask = bool(params.get("corridorkey_auto_mask", False))

    # Route profiles own a complete CorridorKey recipe for auto runs. A manual
    # preset means the caller is steering the form values directly, so the
    # profile force-rewrite must yield to the supplied params (mirrors the Comfy
    # path in ``api.py`` which gates this same block on ``preset != "manual"``).
    forced_translucent_hint_settings = False
    if preset != "manual" and execution_profile in {
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
    if preset != "manual" and execution_profile == "corridorkey-character":
        gamma_space = "sRGB"
        despill_strength = 1.0
        refiner_strength = 1.0
        auto_despeckle = "Off"
        despeckle_size = 64

    client = corridorkey_client if corridorkey_client is not None else DirectCorridorKeyClient()
    hint_source = None
    hint_plan_metadata: dict[str, Any] | None = None
    semantic_decision = params.get("semantic_decision")
    semantic_hint_value = _corridorkey_hint_value_from_semantic_decision(semantic_decision)
    if hint_alpha is not None:
        hint_source = "provided_corridorkey_hint_mask"
    elif semantic_hint_value is not None:
        hint_alpha = np.full(rgb.shape[:2], semantic_hint_value, dtype=np.float32)
        hint_source = f"semantic_full_frame_constant_{semantic_hint_value:.2f}_corridorkey_hint"
        hint_plan_metadata = {
            "schema": "ermbg.corridorkey_constant_hint.v1",
            "source": "semantic_corridorkey_hint_value",
            "value": float(semantic_hint_value),
            "kind": "full_frame_constant",
        }
    elif not auto_mask:
        hint_alpha, hint_source = _corridorkey_default_hint(
            rgb,
            selected_bg_color=selected_bg_color,
            execution_profile=execution_profile,
            screen_mode=screen_mode,
            source_prefix="default",
        )
        if execution_profile == "corridorkey-character":
            hint_alpha, hint_source = _corridorkey_default_hint(
                rgb,
                selected_bg_color=selected_bg_color,
                execution_profile=execution_profile,
                screen_mode=screen_mode,
                source_prefix="character",
            )
        elif execution_profile == "corridorkey-transparent-button":
            hint_alpha, hint_source = _corridorkey_default_hint(
                rgb,
                selected_bg_color=selected_bg_color,
                execution_profile=execution_profile,
                screen_mode=screen_mode,
                source_prefix="glass",
            )
        elif execution_profile == "corridorkey-effect-icon":
            hint_alpha, hint_source = _corridorkey_default_hint(
                rgb,
                selected_bg_color=selected_bg_color,
                execution_profile=execution_profile,
                screen_mode=screen_mode,
                source_prefix="effect",
            )
    elif execution_profile == "corridorkey-transparent-button":
        hint_alpha, hint_source = _corridorkey_default_hint(
            rgb,
            selected_bg_color=selected_bg_color,
            execution_profile=execution_profile,
            screen_mode=screen_mode,
            source_prefix="glass",
        )
    elif execution_profile == "corridorkey-character":
        hint_alpha, hint_source = _corridorkey_default_hint(
            rgb,
            selected_bg_color=selected_bg_color,
            execution_profile=execution_profile,
            screen_mode=screen_mode,
            source_prefix="character",
        )
    elif execution_profile == "corridorkey-effect-icon":
        hint_alpha, hint_source = _corridorkey_default_hint(
            rgb,
            selected_bg_color=selected_bg_color,
            execution_profile=execution_profile,
            screen_mode=screen_mode,
            source_prefix="effect",
        )

    if hint_alpha is None:
        hint_alpha, hint_source = _corridorkey_default_hint(
            rgb,
            selected_bg_color=selected_bg_color,
            execution_profile=execution_profile,
            screen_mode=screen_mode,
            source_prefix="default",
        )

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
        execution_profile=execution_profile,
    )
    subject_alpha = remote.alpha.astype(np.float32)
    subject_foreground_srgb = remote.foreground_srgb.astype(np.uint8)
    subject_rgba = remote.rgba.astype(np.uint8)
    execution_decision_info = _corridorkey_disabled_postprocess_info(params)
    alpha = subject_alpha
    rgba_rgb_srgb = subject_foreground_srgb
    rgba = subject_rgba
    shadow_alpha = np.zeros(rgb.shape[:2], dtype=np.float32)
    shadow_alpha_physical = np.zeros(rgb.shape[:2], dtype=np.float32)
    shadow_info = _corridorkey_disabled_shadow_info(shadow_mode)
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
            "semantic_hint_plan": hint_plan_metadata,
        },
        "shadow": shadow_info,
        "semantic_prior": execution_decision_info,
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
        "execution_profile": execution_profile,
        "corridorkey_analysis": corridorkey_analysis,
        "soft_mask": alpha,
        "subject_alpha": subject_alpha,
        "corridorkey_subject_rgba": subject_rgba,
        "corridorkey_hint": remote.hint_alpha,
        "corridorkey_hint_plan": hint_plan_metadata,
        "corridorkey_raw_alpha": remote.raw_alpha,
        "shadow_alpha": shadow_alpha,
        "shadow_alpha_physical": shadow_alpha_physical,
        "shadow_layer_rgba": shadow_rgba,
        "shadow": shadow_info,
        "semantic_execution": execution_decision_info,
        "profile_settings": {
            "execution_profile": execution_profile,
            "forced_translucent_settings": forced_translucent_hint_settings,
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
    corridorkey_hint_mask: np.ndarray | None = None,
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
    algorithm = _algorithm_from_backend(decision.backend)
    params = dict(decision.params)
    shadow_mode = str(shadow_mode or "auto").strip().lower()
    if shadow_mode not in {"auto", "on", "off"}:
        raise ValueError("shadow_mode must be 'auto', 'on', or 'off'")

    t = time.perf_counter()
    if algorithm == "rgba_passthrough":
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
    elif algorithm in {"pymatting_known_b", "pymatting_fallback"}:
        known_b_bg_threshold = _route_params(params, "pymatting_bg_threshold", 3.5)
        known_b_fg_threshold = _route_params(params, "pymatting_fg_threshold", 24.0)
        known_b_bg_source = _route_params(params, "pymatting_bg_source", "custom")
        known_b_input_preprocessed = bool(_route_params(params, "pymatting_input_preprocessed", False))
        t_preprocess = time.perf_counter()
        if known_b_input_preprocessed:
            known_b_bg = _rgb_param(params, "pymatting_bg_color", fallback_bg_color)
            known_b_rgb = rgb
            known_b_bg_info = {
                "accepted": True,
                "source": "custom",
                "reason": "input_already_preprocessed",
                "background_color": [int(c) for c in known_b_bg],
            }
            preprocess_info = {
                "selected": [],
                "applied": [],
                "metadata": {
                    "known_background_normalization": {
                        "enabled": True,
                        "applied": False,
                        "reason": "input_already_preprocessed_by_caller",
                    }
                },
                "known_background_normalization": {
                    "enabled": True,
                    "applied": False,
                    "reason": "input_already_preprocessed_by_caller",
                },
            }
        else:
            known_b_rgb, known_b_bg, known_b_bg_info, preprocess_info = prepare_known_b_preprocessed_input(
                rgb,
                bg_source=known_b_bg_source,
                bg_color=_route_params(params, "pymatting_bg_color", fallback_bg_color),
                bg_threshold=known_b_bg_threshold,
                fg_threshold=known_b_fg_threshold,
                adaptive=False,
            )
        timings["known_b_preprocess_sec"] = time.perf_counter() - t_preprocess
        t_executor = time.perf_counter()
        response = _matte_image_pymatting_known_b(
            known_b_rgb,
            src_path=None,
            output_dir=None,
            qa=False,
            shadow_mode=shadow_mode,
            method=_route_params(params, "pymatting_method", "cf"),
            image_space=_route_params(params, "pymatting_image_space", "linear"),
            bg_source="custom",
            bg_color=known_b_bg,
            bg_info_override=known_b_bg_info,
            requested_bg_source=known_b_bg_source,
            preprocess_info=preprocess_info,
            bg_threshold=known_b_bg_threshold,
            fg_threshold=known_b_fg_threshold,
            boundary_band_px=_route_params(params, "pymatting_boundary_band_px", 2),
            adapt_bg_threshold=_route_params(params, "pymatting_adapt_bg_threshold", False),
            adapt_fg_threshold=_route_params(params, "pymatting_adapt_fg_threshold", True),
            adapt_boundary_band=_route_params(params, "pymatting_adapt_boundary_band", True),
            cg_maxiter=_route_params(params, "pymatting_cg_maxiter", 1000),
            cg_rtol=_route_params(params, "pymatting_cg_rtol", 1e-6),
            trimap_mode=_route_params(params, "pymatting_trimap_mode", "standard"),
            unknown_grow_px=_route_params(params, "pymatting_unknown_grow_px", 0),
            semantic_decision=_route_params(params, "semantic_decision", None),
            user_keep_mask=_route_params(params, "user_keep_mask", None),
            user_remove_mask=_route_params(params, "user_remove_mask", None),
            explicit_trimap=_route_params(params, "pymatting_explicit_trimap", None),
            auto_route=auto_route,
        )
        timings["known_b_executor_sec"] = time.perf_counter() - t_executor
        response_timings = response.debug.get("timings") if isinstance(response.debug, dict) else None
        if isinstance(response_timings, dict):
            for key, value in response_timings.items():
                if isinstance(value, (int, float)):
                    timings[f"known_b_{key}"] = float(value)
        response.strategy_name = "direct_pymatting_known_b"
        response.debug["backend"] = "direct-pymatting-known-b"
        execution_backend = "direct-pymatting-known-b"
    elif algorithm == "known_bg_glow":
        response = matte_known_bg_glow_direct(
            rgb,
            params=params,
            bg_color=fallback_bg_color,
            qa=False,
            auto_route=auto_route,
        )
        execution_backend = "direct-known-bg-glow"
    elif algorithm == "corridorkey":
        ck_analysis = decision.analysis.get("corridorkey_analysis")
        if not isinstance(ck_analysis, dict):
            raise RuntimeError("CorridorKey direct route requires corridorkey_analysis metadata")
        response = matte_corridorkey_direct(
            rgb,
            corridorkey_analysis=ck_analysis,
            params=params,
            bg_color=fallback_bg_color,
            shadow_mode=shadow_mode,
            hint_alpha=corridorkey_hint_mask,
            execution_profile=_route_params(params, "corridorkey_execution_profile", "auto"),
            corridorkey_client=(ck_factory or DirectCorridorKeyClientFactory()).get(),
            auto_route=auto_route,
        )
        response.debug["backend"] = "direct-corridorkey"
        execution_backend = "direct-corridorkey"
    else:
        raise RuntimeError(f"Unsupported direct algorithm: {algorithm}")
    timings["backend_sec"] = time.perf_counter() - t
    metadata = {
        "algorithm": algorithm,
        "execution_backend": execution_backend,
        "route": auto_route.get("route"),
        "asset_kind": auto_route.get("asset_kind"),
        "parameter_profile": auto_route.get("parameter_profile"),
        "execution_profile": auto_route.get("execution_profile"),
        "shadow_mode": shadow_mode,
    }
    if execution_backend == "direct-corridorkey":
        profile = response.debug.get("execution_profile") if isinstance(response.debug, dict) else None
        if isinstance(profile, str) and profile:
            metadata["execution_profile"] = profile
    return DirectWorkerResult(response=response, timings=timings, metadata=metadata)


def direct_matte_auto(
    rgb: np.ndarray,
    *,
    source_alpha: np.ndarray | None = None,
    shadow_mode: str = "auto",
    corridorkey_screen_mode: str = "auto",
    corridorkey_preset: str = "auto",
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
