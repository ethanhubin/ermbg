"""End-to-end matting pipeline (Phase 1.2 — green-screen + RMBG path).

Pipeline:
  image (sRGB uint8, optional source α)
    -> router.classify_strategy → Strategy(bg_type, image_type, keyer_mode, despill, ...)
    -> if Strategy.passthrough: return source α as-is (skip matting net)
    -> BiRefNet-matting -> alpha matte (continuous, hair/fur preserved)
    -> diagnose: B (measured background color), purity, edge contrast
    -> key α (chromatic | luminance | none) + merge missed components
    -> linear RGB
    -> despill: per Strategy
    -> sRGB RGBA
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from . import io
from .diagnose import BackgroundDiagnoser
from .keyer import (
    gate_alpha_by_keyer,
    key_alpha,
    merge_alpha_components,
    repair_hard_edge_alpha,
    repair_alpha_with_known_bg_key,
    repair_alpha_with_subject_support,
)
from .router import Strategy, classify_strategy
from .segmenter import build_segmenter
from .shadow import ShadowPrior, composite_subject_with_shadow, estimate_shadow_alpha
from .types import MattingResult, Trimap


def _trimap_from_alpha(alpha: np.ndarray, fg_th: float = 0.95, bg_th: float = 0.05) -> Trimap:
    sure_fg = alpha >= fg_th
    sure_bg = alpha <= bg_th
    unknown = ~sure_fg & ~sure_bg
    return Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown)


def matte(
    image_srgb: np.ndarray,
    source_alpha: np.ndarray | None = None,
    object_prompt: str | None = None,
    segmenter=None,
    diagnoser: BackgroundDiagnoser | None = None,
    strategy: Strategy | None = None,
    despill: str | None = None,
    use_keyer: bool | None = None,
    subject_support: np.ndarray | None = None,
    semantic_prior: Any | None = None,
    legacy_analytic_alpha: bool = False,
) -> MattingResult:
    """Run the matting pipeline on one sRGB uint8 image.

    Args:
        image_srgb: H×W×3 sRGB uint8.
        source_alpha: optional H×W float32 [0,1] alpha from the source file.
            If present and looks like a usable matte, the router will choose
            ``rgba_passthrough`` and the matting net is skipped.
        object_prompt: optional natural-language hint (currently unused).
        segmenter: pre-built segmenter (BiRefNetSegmenter by default).
        diagnoser: pre-built BackgroundDiagnoser; built fresh if None.
        strategy: explicit ``Strategy`` to use. If None (default), the
            router classifies the input and picks one.
        despill: optional override for ``strategy.despill``.
        use_keyer: optional override; ``True`` forces the strategy keyer on,
            ``False`` forces it off, ``None`` uses the strategy as-is.
        subject_support: optional H×W float32 [0,1] ownership mask from an
            independent segmenter. When provided, ERMBG may repair low-α
            regions inside this mask, but never uses the keyer as a direct
            whole-component alpha replacement.
        legacy_analytic_alpha: run the old projection+guided-filter path.
    """
    if image_srgb.dtype != np.uint8:
        raise ValueError("matte() expects sRGB uint8 input")
    if image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("matte() expects HxWx3 image")
    if subject_support is not None and subject_support.shape != image_srgb.shape[:2]:
        raise ValueError("subject_support must have shape HxW matching image_srgb")

    if strategy is None:
        strategy = classify_strategy(image_srgb, source_alpha=source_alpha)
    logger.info(f"router: strategy={strategy.name} ({strategy.notes})")

    # ------------------------------------------------------------------ Pass-through fast path
    if strategy.passthrough and source_alpha is not None and not legacy_analytic_alpha:
        return _passthrough_result(image_srgb, source_alpha, strategy)

    # ------------------------------------------------------------------ Matting net
    seg = segmenter if segmenter is not None else build_segmenter(backend="auto")
    soft = seg.segment(image_srgb, object_prompt=object_prompt)

    diag = diagnoser if diagnoser is not None else BackgroundDiagnoser()
    report = diag.diagnose(image_srgb, soft)
    B_srgb = np.array(report.background_color, dtype=np.uint8)
    semantic_subject_mask = getattr(semantic_prior, "subject_mask", None)
    if subject_support is not None and semantic_subject_mask is not None:
        shadow_subject_mask = np.maximum(subject_support, semantic_subject_mask)
        shadow_prior_source = "subject_support+semantic_prior"
    elif subject_support is not None:
        shadow_subject_mask = subject_support
        shadow_prior_source = "subject_support"
    elif semantic_subject_mask is not None:
        shadow_subject_mask = semantic_subject_mask
        shadow_prior_source = "semantic_prior"
    else:
        shadow_subject_mask = None
        shadow_prior_source = ""
    shadow_prior = (
        ShadowPrior(
            subject_mask=shadow_subject_mask,
            shadow_search_mask=getattr(semantic_prior, "shadow_search_mask", None),
            shadow_ownership_mask=getattr(semantic_prior, "shadow_ownership_mask", None),
            shadow_allowed=getattr(semantic_prior, "shadow_allowed", True),
            source=shadow_prior_source,
        )
        if shadow_subject_mask is not None
        or getattr(semantic_prior, "shadow_search_mask", None) is not None
        or getattr(semantic_prior, "shadow_ownership_mask", None) is not None
        or getattr(semantic_prior, "shadow_allowed", True) is False
        else None
    )
    raw_soft = soft.copy()
    pre_shadow_alpha, pre_shadow_info = estimate_shadow_alpha(
        image_srgb,
        raw_soft,
        B_srgb,
        prior=shadow_prior,
    )
    shadow_protect = pre_shadow_alpha > 0.0

    # ------------------------------------------------------------------ Keyer
    keyer_info: dict[str, Any] = {"used": False, "strategy_mode": strategy.keyer_mode}
    keyer_active = strategy.keyer_mode is not None and not legacy_analytic_alpha
    if use_keyer is True:
        keyer_active = strategy.keyer_mode is not None
    elif use_keyer is False:
        keyer_active = False

    if keyer_active and strategy.keyer_mode:
        key = key_alpha(image_srgb, B_srgb, mode=strategy.keyer_mode, thresholds=strategy.keyer_thresholds)
        if strategy.use_keyer_merge:
            soft, info = merge_alpha_components(soft, key)
            keyer_info.update({"used": True, "patched_components": info["patched_components"], "component_areas": info["component_areas"]})
            if info["patched_components"]:
                logger.info(f"keyer: patched {info['patched_components']} component(s) missed by matting net")
        else:
            keyer_info.update({"used": True, "patched_components": 0})
        if strategy.bg_type in {"white", "black"} and strategy.image_type == "graphic":
            full_color_key = key_alpha(image_srgb, B_srgb, mode="chromatic", thresholds=strategy.keyer_thresholds)
            soft, repair_info = repair_alpha_with_known_bg_key(soft, full_color_key)
            keyer_info["known_bg_repair"] = repair_info
            if repair_info["accepted_components"]:
                logger.info(
                    f"keyer: repaired {repair_info['accepted_components']} known-B hole(s) "
                    f"({repair_info['accepted_pixels']} px)"
                )
            soft, hard_edge_info = repair_hard_edge_alpha(image_srgb, soft, key, B_srgb)
            keyer_info["hard_edge_repair"] = hard_edge_info
            if hard_edge_info["accepted_components"]:
                logger.info(
                    f"keyer: repaired {hard_edge_info['accepted_components']} hard-edge component(s) "
                    f"({hard_edge_info['accepted_pixels']} px)"
                )
        if subject_support is not None:
            soft, repair_info = repair_alpha_with_subject_support(soft, key, subject_support)
            keyer_info["subject_repair"] = repair_info
            if repair_info["accepted_components"]:
                logger.info(
                    f"keyer: repaired {repair_info['accepted_components']} subject-owned hole(s) "
                    f"({repair_info['accepted_pixels']} px)"
                )
        if strategy.use_keyer_gate:
            soft, gate_info = gate_alpha_by_keyer(soft, key)
            keyer_info["pixels_gated"] = gate_info["pixels_gated"]
            keyer_info["mean_drop"] = gate_info["mean_drop"]
            if gate_info["pixels_gated"]:
                logger.info(
                    f"keyer: gated {gate_info['pixels_gated']} halo px "
                    f"(mean α drop {gate_info['mean_drop']:.3f})"
                )

    if shadow_protect.any():
        raised_by_keyer = shadow_protect & (soft > raw_soft)
        if raised_by_keyer.any():
            soft[raised_by_keyer] = raw_soft[raised_by_keyer]
        keyer_info["shadow_protected_pixels"] = int(shadow_protect.sum())
        keyer_info["shadow_keyer_raise_reverted_pixels"] = int(raised_by_keyer.sum())

    if report.verdict == "not-pure-bg":
        logger.warning(
            f"diagnose: verdict=not-pure-bg (purity_sigma={report.purity_sigma:.2f} > "
            f"{diag.t.purity_sigma_max}); results may be unreliable"
        )

    # ------------------------------------------------------------------ Despill
    despill_method = despill if despill is not None else strategy.despill
    C_lin = io.srgb_to_linear(image_srgb).astype(np.float32)
    B_lin = io.srgb_to_linear(B_srgb.reshape(1, 1, 3))[0, 0].astype(np.float32)

    if legacy_analytic_alpha:
        alpha, F_lin, trimap = _legacy_path(image_srgb, soft, C_lin, B_lin, B_srgb)
        despill_used = "legacy"
    else:
        alpha, F_lin, trimap = _new_path(
            soft,
            C_lin,
            B_lin,
            despill_method,
            protect_mask=getattr(semantic_prior, "subject_material_mask", None),
        )
        despill_used = despill_method

    subject_alpha = alpha.copy()
    shadow_alpha, shadow_info = estimate_shadow_alpha(
        image_srgb,
        subject_alpha,
        B_srgb,
        prior=shadow_prior,
    )
    if not shadow_info["detected"] and pre_shadow_info["detected"]:
        shadow_alpha = pre_shadow_alpha
        shadow_info = dict(pre_shadow_info)
        shadow_info["source"] = "pre_keyer"
    else:
        shadow_info["source"] = "post_despill"
    if shadow_info["detected"]:
        alpha, F_lin = composite_subject_with_shadow(F_lin, subject_alpha, shadow_alpha)
        trimap = _trimap_from_alpha(alpha)

    F_srgb = io.linear_to_srgb_u8(F_lin)
    alpha_u8 = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
    rgba = np.dstack([F_srgb, alpha_u8])

    from .trimap import trimap_to_uint8

    return MattingResult(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=F_srgb,
        foreground_linear=F_lin,
        trimap=trimap,
        background_color=report.background_color,
        diagnosis=report,
        debug={
            "soft_mask": soft,
            "subject_alpha": subject_alpha,
            "shadow_alpha": shadow_alpha,
            "shadow": shadow_info,
            "semantic_prior": semantic_prior.to_dict() if hasattr(semantic_prior, "to_dict") else {},
            "trimap_u8": trimap_to_uint8(trimap),
            "despill_method": despill_used,
            "keyer": keyer_info,
            "strategy": {
                "name": strategy.name,
                "bg_type": strategy.bg_type,
                "image_type": strategy.image_type,
                "keyer_mode": strategy.keyer_mode,
                "despill": strategy.despill,
                "passthrough": strategy.passthrough,
                "notes": strategy.notes,
                "extras": strategy.extras,
            },
        },
    )


def _passthrough_result(
    image_srgb: np.ndarray, source_alpha: np.ndarray, strategy: Strategy
) -> MattingResult:
    """Return the source as-is when it already carries a usable alpha."""
    h, w = image_srgb.shape[:2]
    alpha = source_alpha.astype(np.float32)
    alpha_u8 = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
    rgba = np.dstack([image_srgb, alpha_u8])
    trimap = _trimap_from_alpha(alpha)
    from .trimap import trimap_to_uint8

    logger.info("router: passthrough — using source α directly")
    return MattingResult(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=image_srgb.copy(),
        foreground_linear=io.srgb_to_linear(image_srgb).astype(np.float32),
        trimap=trimap,
        background_color=(0, 0, 0),
        diagnosis=None,
        debug={
            "soft_mask": alpha,
            "subject_alpha": alpha,
            "shadow_alpha": np.zeros((h, w), dtype=np.float32),
            "shadow": {
                "method": "none",
                "detected": False,
                "applied": False,
                "pixels": 0,
                "bbox_xyxy": [0, 0, 0, 0],
                "mean_alpha": 0.0,
                "p95_alpha": 0.0,
                "max_alpha": 0.0,
                "accepted_components": 0,
                "component_areas": [],
                "rejected_components": 0,
                "reason": "rgba passthrough",
            },
            "trimap_u8": trimap_to_uint8(trimap),
            "despill_method": "none",
            "keyer": {"used": False, "strategy_mode": None},
            "strategy": {
                "name": strategy.name,
                "bg_type": strategy.bg_type,
                "image_type": strategy.image_type,
                "keyer_mode": None,
                "despill": "none",
                "passthrough": True,
                "notes": strategy.notes,
                "extras": strategy.extras,
            },
        },
    )


def _new_path(
    soft: np.ndarray,
    C_lin: np.ndarray,
    B_lin: np.ndarray,
    despill_method: str,
    protect_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Trimap]:
    """BiRefNet-matting alpha + chosen despill."""
    from . import despill as despill_mod

    alpha = soft.astype(np.float32)
    alpha_out, F_lin = despill_mod.apply_despill(
        despill_method,
        C_lin,
        B_lin,
        alpha,
        protect_mask=protect_mask,
    )
    F_lin = np.clip(F_lin, 0.0, 1.0)
    trimap = _trimap_from_alpha(alpha_out)
    return alpha_out, F_lin, trimap


def _legacy_path(
    image_srgb: np.ndarray,
    soft: np.ndarray,
    C_lin: np.ndarray,
    B_lin: np.ndarray,
    B_srgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Trimap]:
    """Old trimap + projection + guided-filter + analytic decontaminate path,
    kept for regression comparison."""
    from . import alpha as alpha_mod
    from . import recover
    from .foreground import estimate_foreground_reference
    from .trimap import build_trimap

    trimap = build_trimap(image_srgb, soft, B_srgb)
    f_ref = estimate_foreground_reference(C_lin, trimap)
    alpha, _ = alpha_mod.estimate_alpha_full(C_lin, B_lin, f_ref, trimap)
    alpha = recover.fix_halo(C_lin, B_lin, alpha)
    F_lin = recover.recover_foreground(C_lin, B_lin, alpha, f_ref)
    F_lin = recover.decontaminate(F_lin, f_ref, B_lin, alpha)
    return alpha, np.clip(F_lin, 0.0, 1.0), trimap


__all__ = ["matte", "MattingResult"]
