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
    repair_opaque_interior_with_known_bg_key,
    resolve_hard_edge_alpha_with_known_bg_key,
    repair_alpha_with_known_bg_key,
    repair_alpha_with_subject_support,
)
from .router import Strategy, classify_strategy
from .segmenter import build_segmenter
from .shadow import (
    ShadowThresholds,
    ShadowPrior,
    composite_subject_with_shadow,
    estimate_shadow_alpha,
    exterior_scalar_darkening_mask,
    remove_small_display_shadow_components,
    shadow_alpha_to_display_alpha,
)
from .solid_graphic import SolidGraphicResult, analyze_solid_bg_graphic
from .types import MattingResult, Trimap


def _trimap_from_alpha(alpha: np.ndarray, fg_th: float = 0.95, bg_th: float = 0.05) -> Trimap:
    sure_fg = alpha >= fg_th
    sure_bg = alpha <= bg_th
    unknown = ~sure_fg & ~sure_bg
    return Trimap(sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown)


def _stabilize_foreground_for_export(
    foreground_linear: np.ndarray,
    subject_alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill weakly constrained straight-foreground RGB from sure foreground.

    A straight foreground layer stores color independent of alpha. Where the
    subject contribution is smaller than the background contribution
    (``subject_alpha < 0.5``), the inverse-composited RGB is underdetermined:
    tiny alpha/background errors can turn into black, green, or magenta RGB
    speckles even though those pixels are nearly invisible after compositing.
    For export, extend the nearest sure-foreground material color into that
    weak region. Alpha remains the ownership signal. The shadow-composited RGB
    used by ``rgba`` is kept separately; ``foreground_srgb`` is the clean
    subject-color layer for inspection and downstream color borrowing.
    """
    fg = foreground_linear.astype(np.float32, copy=True)
    a = np.clip(subject_alpha.astype(np.float32), 0.0, 1.0)
    sure_fg = a >= 0.95  # Same semantic threshold as the trimap sure-foreground gate.
    weak_foreground = a < 0.5  # Below this, known-background contribution dominates the color equation.

    info: dict[str, Any] = {
        "used": True,
        "sure_foreground_pixels": int(sure_fg.sum()),
        "filled_pixels": 0,
        "reason": "",
    }
    if not sure_fg.any():
        info["reason"] = "no sure foreground seed"
        return fg, info
    fill = weak_foreground & ~sure_fg
    if not fill.any():
        return fg, info

    from scipy import ndimage

    # distance_transform_edt returns the nearest zero-valued pixel. Passing the
    # inverse seed mask gives a deterministic nearest-material extension without
    # choosing a fixed pixel radius.
    _, nearest = ndimage.distance_transform_edt(~sure_fg, return_indices=True)
    fg[fill] = fg[nearest[0][fill], nearest[1][fill]]
    info["filled_pixels"] = int(fill.sum())
    return fg, info


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
    soft_mask: np.ndarray | None = None,
    shadow_mode: str = "on",
    legacy_analytic_alpha: bool = False,
    solid_graphic_prepass: bool | None = None,
    solid_graphic_alpha_refiner: str = "heuristic",
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
        soft_mask: optional precomputed H×W matting-net alpha. Used by batch
            and preview flows that already ran the segmenter for diagnosis.
        shadow_mode: ``"on"`` preserves the full two-pass shadow recovery,
            ``"off"`` skips shadow detection/compositing for faster previews,
            and ``"auto"`` currently maps to ``"on"`` for quality.
        legacy_analytic_alpha: run the old projection+guided-filter path.
        solid_graphic_prepass: try the analytic ownership-first path before
            building/running the matting net. ``None`` enables it only when no
            segmenter or precomputed soft mask was injected.
        solid_graphic_alpha_refiner: ``"heuristic"`` or an experimental
            ``"pymatting-*"`` method for the solid-background graphic prepass.
    """
    if image_srgb.dtype != np.uint8:
        raise ValueError("matte() expects sRGB uint8 input")
    if image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("matte() expects HxWx3 image")
    if subject_support is not None and subject_support.shape != image_srgb.shape[:2]:
        raise ValueError("subject_support must have shape HxW matching image_srgb")
    if soft_mask is not None and soft_mask.shape != image_srgb.shape[:2]:
        raise ValueError("soft_mask must have shape HxW matching image_srgb")
    if shadow_mode not in {"on", "off", "auto"}:
        raise ValueError("shadow_mode must be 'on', 'off', or 'auto'")

    if strategy is None:
        strategy = classify_strategy(image_srgb, source_alpha=source_alpha)
    logger.info(f"router: strategy={strategy.name} ({strategy.notes})")

    # ------------------------------------------------------------------ Pass-through fast path
    if strategy.passthrough and source_alpha is not None and not legacy_analytic_alpha:
        return _passthrough_result(image_srgb, source_alpha, strategy)

    if solid_graphic_prepass is None:
        solid_graphic_prepass = segmenter is None and soft_mask is None
    can_try_solid_graphic = (
        solid_graphic_prepass
        and not legacy_analytic_alpha
        and soft_mask is None
        and subject_support is None
        and semantic_prior is None
        and despill is None
        and use_keyer is None
        and shadow_mode != "off"
        and strategy.bg_type in {"saturated", "white", "black", "grey"}
    )
    if can_try_solid_graphic:
        solid = analyze_solid_bg_graphic(image_srgb, alpha_refiner=solid_graphic_alpha_refiner)
        if solid.accepted:
            logger.info(
                f"solid_graphic: accepted confidence={solid.confidence:.3f} "
                f"bg={solid.background_color}"
            )
            return solid_graphic_to_matting_result(solid, strategy, shadow_mode=shadow_mode)
        logger.info(f"solid_graphic: fallback ({solid.reason})")

    # ------------------------------------------------------------------ Matting net
    if soft_mask is None:
        seg = segmenter if segmenter is not None else build_segmenter(backend="auto")
        soft = seg.segment(image_srgb, object_prompt=object_prompt)
    else:
        # The matting net is by far the slowest stage. Batch/semantic-prior
        # flows may already have computed this exact alpha preview; accepting it
        # here avoids a second BiRefNet pass without changing downstream logic.
        soft = np.clip(soft_mask.astype(np.float32), 0.0, 1.0)

    diag = diagnoser if diagnoser is not None else BackgroundDiagnoser()
    report = diag.diagnose(image_srgb, soft)
    B_srgb = np.array(report.background_color, dtype=np.uint8)
    # The shadow, scalar-darkening, despill, and export stages all need linear
    # source RGB. Keep one per-request conversion so large Web mattes do not pay
    # the same full-image sRGB transform several times.
    C_lin = io.srgb_to_linear(image_srgb).astype(np.float32)
    B_lin = io.srgb_to_linear(B_srgb.reshape(1, 1, 3))[0, 0].astype(np.float32)
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
            subject_material_mask=getattr(semantic_prior, "subject_material_mask", None),
            shadow_search_mask=getattr(semantic_prior, "shadow_search_mask", None),
            shadow_ownership_mask=getattr(semantic_prior, "shadow_ownership_mask", None),
            shadow_allowed=getattr(semantic_prior, "shadow_allowed", True),
            source=shadow_prior_source,
        )
        if shadow_subject_mask is not None
        or getattr(semantic_prior, "subject_material_mask", None) is not None
        or getattr(semantic_prior, "shadow_search_mask", None) is not None
        or getattr(semantic_prior, "shadow_ownership_mask", None) is not None
        or getattr(semantic_prior, "shadow_allowed", True) is False
        else None
    )
    raw_soft = soft.copy()
    material_protect_mask = getattr(semantic_prior, "subject_material_mask", None)
    if material_protect_mask is not None:
        material_protect = np.asarray(material_protect_mask, dtype=np.float32) > 0.0
        if material_protect.shape != image_srgb.shape[:2]:
            raise ValueError("semantic_prior.subject_material_mask must have shape HxW matching image_srgb")
    else:
        material_protect = np.zeros(image_srgb.shape[:2], dtype=bool)
    shadow_enabled = shadow_mode != "off"
    if shadow_enabled:
        pre_shadow_alpha, pre_shadow_info = estimate_shadow_alpha(
            image_srgb,
            raw_soft,
            B_srgb,
            prior=shadow_prior,
            image_linear=C_lin,
        )
    else:
        pre_shadow_alpha, pre_shadow_info = _empty_shadow_result(
            image_srgb.shape[:2],
            reason="shadow_mode=off",
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
        if strategy.bg_type == "saturated" and shadow_enabled:
            # Saturated-screen assets can fail as one large key component whose
            # upper half has enough matting-net coverage that component-level
            # merge decides "already present" while a lower UI/body region is
            # still missing. This topology repair uses known-B color evidence
            # only to restore low-alpha pixels inside key-supported, anchored
            # subject regions; it avoids exterior fringe pixels so ordinary
            # green-screen antialiasing and shadows are not filled wholesale.
            # Keep it tied to shadow-enabled runs because dark cast shadows can
            # also be far from saturated B; the pre-shadow pass below provides
            # the measurable guard that distinguishes those from missed subject.
            soft, repair_info = repair_alpha_with_known_bg_key(
                soft,
                key,
                key_fg_threshold=0.65,
                matting_low_threshold=0.70,
                support_threshold=0.35,
                fg_anchor_threshold=0.85,
                exterior_margin_px=1,
                target_alpha_floor=0.92,
            )
            keyer_info["saturated_known_bg_repair"] = repair_info
            if repair_info["accepted_components"]:
                logger.info(
                    f"keyer: repaired {repair_info['accepted_components']} saturated known-B hole(s) "
                    f"({repair_info['accepted_pixels']} px)"
                )
            # Separate from hole filling: hard UI/product interiors on a known
            # saturated screen can remain α≈0.75–0.9 even though OKLab distance
            # says they are definitely not background. Snap only key-supported
            # interior pixels that are anchored to confident foreground, while
            # preserving outer antialiasing and measured shadow candidates.
            soft, opaque_info = repair_opaque_interior_with_known_bg_key(
                soft,
                key,
                shadow_protect_mask=shadow_protect,
                material_protect_mask=material_protect,
            )
            keyer_info["saturated_opaque_interior_repair"] = opaque_info
            if opaque_info["accepted_components"]:
                logger.info(
                    f"keyer: snapped {opaque_info['accepted_pixels']} saturated known-B interior px "
                    f"across {opaque_info['accepted_components']} component(s)"
                )
            # For a clean solid-screen hard edge, the only ambiguous pixels
            # should be the narrow exterior antialiasing band. Once the keyer
            # proves that topology (connected exterior bg + small transition
            # fraction), it can safely raise under-opaque edge/interior alpha
            # without eroding soft photographic details.
            soft, hard_edge_key_info = resolve_hard_edge_alpha_with_known_bg_key(
                soft,
                key,
                image_srgb=image_srgb,
                background_color=B_srgb,
                shadow_protect_mask=shadow_protect,
                material_protect_mask=material_protect,
            )
            keyer_info["saturated_hard_edge_key_resolve"] = hard_edge_key_info
            if hard_edge_key_info["raised_pixels"] or hard_edge_key_info["lowered_pixels"]:
                logger.info(
                    f"keyer: resolved saturated hard-edge alpha "
                    f"(+{hard_edge_key_info['raised_pixels']} / -{hard_edge_key_info['lowered_pixels']} px) "
                    "from clean known-B key"
                )
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

    if material_protect.any():
        # Subject-owned soft layers are deliberately restored to the segmenter
        # matte after keyer repair/gating. These regions are visually
        # underconstrained on white backgrounds; experience-driven repair gates
        # can otherwise turn glass/glow/smoke into opaque pale foreground.
        changed_by_keyer = material_protect & (np.abs(soft - raw_soft) > 1e-6)
        if changed_by_keyer.any():
            soft[changed_by_keyer] = raw_soft[changed_by_keyer]
        keyer_info["subject_material_protected_pixels"] = int(material_protect.sum())
        keyer_info["subject_material_keyer_reverted_pixels"] = int(changed_by_keyer.sum())

    if shadow_enabled and strategy.bg_type == "saturated" and "key" in locals():
        scalar_exterior, scalar_info = exterior_scalar_darkening_mask(
            image_srgb,
            B_srgb,
            known_background_mask=key <= 0.02,
            image_linear=C_lin,
        )
        scalar_exterior &= ~material_protect
        if scalar_exterior.any():
            # Chromatic distance alone mistakes ``C ~= scale * B`` for
            # foreground on saturated screens. For hard-edged assets this
            # creates black rims after unmix+chroma-cap. Reclassify only
            # scalar-darkened pixels connected to exterior known background;
            # interior subject material with a similar hue is not reachable
            # through the exterior flood and remains protected as subject.
            soft[scalar_exterior] = 0.0
        scalar_info["pixels"] = int(scalar_exterior.sum())
        keyer_info["exterior_scalar_darkening_reclassified"] = scalar_info

    if report.verdict == "not-pure-bg":
        logger.warning(
            f"diagnose: verdict=not-pure-bg (purity_sigma={report.purity_sigma:.2f} > "
            f"{diag.t.purity_sigma_max}); results may be unreliable"
        )

    # ------------------------------------------------------------------ Despill
    despill_method = despill if despill is not None else strategy.despill

    if legacy_analytic_alpha:
        alpha, F_lin, trimap = _legacy_path(image_srgb, soft, C_lin, B_lin, B_srgb)
        despill_used = "legacy"
    else:
        alpha, F_lin, trimap = _new_path(
            soft,
            C_lin,
            B_lin,
            despill_method,
            protect_mask=material_protect_mask,
        )
        despill_used = despill_method

    subject_alpha = alpha.copy()
    foreground_linear, foreground_export_info = _stabilize_foreground_for_export(F_lin, subject_alpha)
    if shadow_enabled:
        shadow_alpha_physical, shadow_info = estimate_shadow_alpha(
            image_srgb,
            subject_alpha,
            B_srgb,
            prior=shadow_prior,
            image_linear=C_lin,
        )
        if not shadow_info["detected"] and pre_shadow_info["detected"]:
            shadow_alpha_physical = pre_shadow_alpha
            shadow_info = dict(pre_shadow_info)
            shadow_info["source"] = "pre_keyer"
        else:
            shadow_info["source"] = "post_despill"
    else:
        shadow_alpha_physical, shadow_info = _empty_shadow_result(
            image_srgb.shape[:2],
            reason="shadow_mode=off",
        )
        shadow_info["source"] = "disabled"
    shadow_info["mode"] = shadow_mode
    # The detector returns linear known-background darkening strength. For the
    # exported RGBA, convert that to a black alpha that appears correct in
    # ordinary sRGB compositors; otherwise hard shadows look much too heavy.
    shadow_alpha = shadow_alpha_to_display_alpha(shadow_alpha_physical, B_srgb)
    display_shadow_filter_info: dict[str, Any] = {}
    if shadow_info["detected"]:
        default_shadow_thresholds = ShadowThresholds()
        min_display_shadow_area = float(
            max(8.0, float(default_shadow_thresholds.min_total_area_ratio) * float(alpha.size))
        )
        shadow_alpha, display_shadow_filter_info = remove_small_display_shadow_components(
            shadow_alpha,
            min_area=min_display_shadow_area,
        )
        physical_mask = shadow_alpha_physical > 0.0
        shadow_info["display_safe"] = {
            "enabled": True,
            "mean_alpha": float(shadow_alpha[physical_mask].mean()) if physical_mask.any() else 0.0,
            "p95_alpha": float(np.percentile(shadow_alpha[physical_mask], 95.0)) if physical_mask.any() else 0.0,
            "max_alpha": float(shadow_alpha[physical_mask].max()) if physical_mask.any() else 0.0,
            **display_shadow_filter_info,
        }
    rgba_rgb_linear = foreground_linear
    if shadow_info["detected"]:
        alpha, rgba_rgb_linear = composite_subject_with_shadow(foreground_linear, subject_alpha, shadow_alpha)
        trimap = _trimap_from_alpha(alpha)

    rgba_rgb_srgb = io.linear_to_srgb_u8(rgba_rgb_linear)
    foreground_srgb = io.linear_to_srgb_u8(foreground_linear)
    alpha_u8 = (np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
    rgba = np.dstack([rgba_rgb_srgb, alpha_u8])

    from .trimap import trimap_to_uint8

    return MattingResult(
        rgba=rgba,
        alpha=alpha,
        foreground_srgb=foreground_srgb,
        foreground_linear=foreground_linear,
        trimap=trimap,
        background_color=report.background_color,
        diagnosis=report,
        debug={
            "soft_mask": soft,
            "subject_alpha": subject_alpha,
            "shadow_alpha": shadow_alpha,
            "shadow_alpha_physical": shadow_alpha_physical,
            "shadow": shadow_info,
            "foreground_export": foreground_export_info,
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


def _empty_shadow_result(shape: tuple[int, int], *, reason: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the standard debug shape when shadow recovery is skipped."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    return out, {
        "method": "known_bg_scalar_darkening",
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
        "reason": reason,
    }


def solid_graphic_to_matting_result(
    solid: SolidGraphicResult,
    strategy: Strategy,
    *,
    shadow_mode: str,
) -> MattingResult:
    """Adapt the ownership-first solid-graphic engine to MattingResult."""
    from .trimap import trimap_to_uint8

    alpha = np.clip(solid.alpha.astype(np.float32), 0.0, 1.0)
    subject_alpha = np.clip(solid.subject_alpha.astype(np.float32), 0.0, 1.0)
    shadow_alpha = np.clip(alpha - subject_alpha, 0.0, 1.0)
    rgba_rgb_srgb = io.linear_to_srgb_u8(solid.rgba_rgb_linear)
    foreground_srgb = io.linear_to_srgb_u8(solid.foreground_linear)
    alpha_u8 = (alpha * 255.0 + 0.5).astype(np.uint8)
    trimap = _trimap_from_alpha(alpha)
    shadow_pixels = int((shadow_alpha > 0.0).sum())
    if shadow_pixels:
        yy, xx = np.nonzero(shadow_alpha > 0.0)
        bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    else:
        bbox = [0, 0, 0, 0]

    return MattingResult(
        rgba=np.dstack([rgba_rgb_srgb, alpha_u8]),
        alpha=alpha,
        foreground_srgb=foreground_srgb,
        foreground_linear=solid.foreground_linear,
        trimap=trimap,
        background_color=solid.background_color,
        diagnosis=None,
        debug={
            "soft_mask": subject_alpha,
            "subject_alpha": subject_alpha,
            "shadow_alpha": shadow_alpha,
            "shadow_alpha_physical": shadow_alpha,
            "shadow": {
                "method": "solid_bg_graphic_scalar_darkening",
                "detected": bool(shadow_pixels),
                "applied": bool(shadow_pixels),
                "pixels": shadow_pixels,
                "bbox_xyxy": bbox,
                "mean_alpha": float(shadow_alpha[shadow_alpha > 0.0].mean()) if shadow_pixels else 0.0,
                "p95_alpha": float(np.percentile(shadow_alpha[shadow_alpha > 0.0], 95.0)) if shadow_pixels else 0.0,
                "max_alpha": float(shadow_alpha.max()) if shadow_pixels else 0.0,
                "accepted_components": 1 if shadow_pixels else 0,
                "component_areas": [shadow_pixels] if shadow_pixels else [],
                "rejected_components": 0,
                "reason": "solid_bg_graphic ownership" if shadow_pixels else "no shadow layer",
                "mode": shadow_mode,
                "source": "solid_bg_graphic",
            },
            "solid_graphic": {
                "accepted": solid.accepted,
                "reason": solid.reason,
                "confidence": solid.confidence,
                "background_color": list(solid.background_color),
                "debug": solid.debug,
            },
            "ownership_masks": solid.ownership_masks,
            "trimap_u8": trimap_to_uint8(trimap),
            "despill_method": "solid_bg_graphic",
            "keyer": {"used": False, "strategy_mode": strategy.keyer_mode},
            "strategy": {
                "name": "solid_bg_graphic",
                "bg_type": strategy.bg_type,
                "image_type": strategy.image_type,
                "keyer_mode": None,
                "despill": "solid_bg_graphic",
                "passthrough": False,
                "notes": "Analytic ownership-first solid-background graphic path.",
                "extras": {
                    **strategy.extras,
                    "fallback_strategy": strategy.name,
                    "solid_graphic_confidence": solid.confidence,
                },
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


__all__ = ["matte", "MattingResult", "solid_graphic_to_matting_result"]
