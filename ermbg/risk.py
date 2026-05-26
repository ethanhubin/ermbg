"""Local evidence-region extraction for planner-driven matting.

``RiskRegion`` is the current implementation name for these evidence packages.
They are not semantic policy regions and they are not final masks. They only
summarize measurable local evidence that a rule planner or future VLM planner
can interpret before choosing tools.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .colorspace import oklab_distance, srgb_to_oklab
from .planner import RiskRegion


def _touches_border(component: np.ndarray) -> bool:
    return bool(
        component[0, :].any()
        or component[-1, :].any()
        or component[:, 0].any()
        or component[:, -1].any()
    )


def _component_regions(
    candidate: np.ndarray,
    *,
    region_prefix: str,
    kind,
    min_area: float,
    max_area: float | None = None,
    reject_border: bool = False,
    anchor_mask: np.ndarray | None = None,
    anchor_dilate_px: int = 2,
    base_evidence: dict[str, Any] | None = None,
) -> tuple[list[RiskRegion], dict[str, Any]]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    accepted: list[RiskRegion] = []
    accepted_areas: list[int] = []
    rejected = 0
    kernel = np.ones((3, 3), np.uint8)
    evidence = dict(base_evidence or {})

    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        comp = labels == label_idx
        if area < min_area or (max_area is not None and area > max_area):
            rejected += 1
            continue
        if reject_border and _touches_border(comp):
            rejected += 1
            continue
        if anchor_mask is not None:
            near_comp = cv2.dilate(
                comp.astype(np.uint8),
                kernel,
                iterations=max(1, int(anchor_dilate_px)),
            ).astype(bool)
            if not (near_comp & anchor_mask).any():
                rejected += 1
                continue

        region_evidence = {
            **evidence,
            "area": area,
            "component_index": len(accepted),
        }
        accepted.append(
            RiskRegion(
                id=f"{region_prefix}_{len(accepted)}",
                kind=kind,
                mask=comp,
                confidence=1.0,
                evidence=region_evidence,
            )
        )
        accepted_areas.append(area)

    return accepted, {
        "accepted_components": len(accepted),
        "accepted_pixels": int(sum(accepted_areas)),
        "component_areas": accepted_areas,
        "rejected_components": rejected,
    }


def extract_same_bg_enclosed_regions(
    image_srgb: np.ndarray,
    rgba: np.ndarray,
    background_color: tuple[int, int, int],
    *,
    bg_distance_max: float = 3.0,
    alpha_max: float = 0.20,
    fg_anchor_threshold: float = 0.85,
    anchor_dilate_px: int = 2,
    min_area_ratio: float = 0.0003,
    max_area_ratio: float = 0.20,
) -> tuple[list[RiskRegion], dict[str, Any]]:
    """Find enclosed low-alpha regions whose observed color equals known B."""
    if image_srgb.shape[:2] != rgba.shape[:2]:
        raise ValueError("image_srgb and rgba must share HxW")

    h, w = image_srgb.shape[:2]
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    close_to_bg = oklab_distance(lab, bg_lab) <= float(bg_distance_max)
    candidate = close_to_bg & (alpha <= float(alpha_max))
    confident_fg = alpha >= float(fg_anchor_threshold)
    min_area = max(1.0, min_area_ratio * float(h * w))
    max_area = max(min_area, max_area_ratio * float(h * w))

    regions, info = _component_regions(
        candidate,
        region_prefix="same_bg",
        kind="same_bg_enclosed_region",
        min_area=min_area,
        max_area=max_area,
        reject_border=True,
        anchor_mask=confident_fg,
        anchor_dilate_px=anchor_dilate_px,
        base_evidence={
            "bg_distance_max": bg_distance_max,
            "alpha_max": alpha_max,
            "fg_anchor_threshold": fg_anchor_threshold,
            "anchor_dilate_px": anchor_dilate_px,
        },
    )
    return regions, info


def extract_alpha_keyer_disagreement_regions(
    matting_alpha: np.ndarray,
    key_alpha: np.ndarray,
    *,
    key_fg_threshold: float = 0.75,
    matting_low_threshold: float = 0.65,
    fg_anchor_threshold: float = 0.85,
    anchor_dilate_px: int = 2,
    min_area_ratio: float = 0.00002,
) -> tuple[list[RiskRegion], dict[str, Any]]:
    """Find regions where keyer evidence says foreground but matting is low."""
    m = matting_alpha.astype(np.float32)
    k = key_alpha.astype(np.float32)
    if m.shape != k.shape:
        raise ValueError("matting_alpha and key_alpha must share HxW")

    h, w = m.shape
    candidate = (k >= key_fg_threshold) & (m <= matting_low_threshold)
    confident_fg = m >= fg_anchor_threshold
    return _component_regions(
        candidate,
        region_prefix="alpha_keyer",
        kind="alpha_keyer_disagreement",
        min_area=max(1.0, min_area_ratio * float(h * w)),
        anchor_mask=confident_fg,
        anchor_dilate_px=anchor_dilate_px,
        base_evidence={
            "key_fg_threshold": key_fg_threshold,
            "matting_low_threshold": matting_low_threshold,
            "fg_anchor_threshold": fg_anchor_threshold,
            "anchor_dilate_px": anchor_dilate_px,
        },
    )


def extract_hard_edge_candidate_regions(
    image_srgb: np.ndarray,
    matting_alpha: np.ndarray,
    key_alpha: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    key_fg_threshold: float = 0.90,
    matting_low_threshold: float = 0.85,
    lightness_contrast_min: float = 55.0,
    fg_anchor_threshold: float = 0.85,
    anchor_dilate_px: int = 2,
    min_area_ratio: float = 0.000005,
    max_area_ratio: float = 0.02,
) -> tuple[list[RiskRegion], dict[str, Any]]:
    """Find small high-contrast components that look like hard graphic edges."""
    m = matting_alpha.astype(np.float32)
    k = key_alpha.astype(np.float32)
    if image_srgb.shape[:2] != m.shape or k.shape != m.shape:
        raise ValueError("image_srgb, matting_alpha, and key_alpha must share HxW")

    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    lightness_contrast = np.abs(lab[..., 0] - bg_lab[0]).astype(np.float32) * 100.0
    candidate = (
        (k >= key_fg_threshold)
        & (m < matting_low_threshold)
        & (lightness_contrast >= lightness_contrast_min)
    )

    h, w = m.shape
    img_area = float(h * w)
    return _component_regions(
        candidate,
        region_prefix="hard_edge",
        kind="hard_edge_candidate",
        min_area=max(1.0, min_area_ratio * img_area),
        max_area=max(1.0, max_area_ratio * img_area),
        anchor_mask=m >= fg_anchor_threshold,
        anchor_dilate_px=anchor_dilate_px,
        base_evidence={
            "key_fg_threshold": key_fg_threshold,
            "matting_low_threshold": matting_low_threshold,
            "lightness_contrast_min": lightness_contrast_min,
            "fg_anchor_threshold": fg_anchor_threshold,
            "anchor_dilate_px": anchor_dilate_px,
        },
    )


def coalesce_risk_regions(
    regions: list[RiskRegion],
    *,
    kinds: tuple[str, ...] = ("hard_edge_candidate", "alpha_keyer_disagreement"),
    merge_distance_px: int = 3,
) -> list[RiskRegion]:
    """Merge nearby same-kind risk fragments into larger planner-friendly regions.

    The merged mask contains only original risk pixels; dilation is used only to
    decide which fragments belong together. This keeps pixel evidence honest
    while reducing planner/VLM noise.
    """
    if not regions:
        return []

    coalesce_kinds = set(kinds)
    passthrough = [region for region in regions if region.kind not in coalesce_kinds]
    grouped: list[RiskRegion] = []

    for kind in kinds:
        same_kind = [region for region in regions if region.kind == kind]
        if not same_kind:
            continue
        shape = same_kind[0].mask.shape
        if any(region.mask.shape != shape for region in same_kind):
            raise ValueError("all regions of a kind must share HxW")

        union = np.zeros(shape, dtype=bool)
        for region in same_kind:
            union |= region.mask
        if not union.any():
            continue

        if merge_distance_px > 0:
            ksize = 2 * int(merge_distance_px) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            support = cv2.dilate(union.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            support = union

        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
        group_idx = 0
        prefix = str(kind).replace("_candidate", "").replace("_disagreement", "")
        for label_idx in range(1, n_labels):
            support_comp = labels == label_idx
            members = [region for region in same_kind if (region.mask & support_comp).any()]
            if not members:
                continue
            mask = np.zeros(shape, dtype=bool)
            for region in members:
                mask |= region.mask
            source_ids = [region.id for region in members]
            grouped.append(
                RiskRegion(
                    id=f"{prefix}_group_{group_idx}",
                    kind=kind,  # type: ignore[arg-type]
                    mask=mask,
                    confidence=max(region.confidence for region in members),
                    evidence={
                        "coalesced": True,
                        "merge_distance_px": int(merge_distance_px),
                        "source_region_ids": source_ids,
                        "source_region_count": len(source_ids),
                        "area": int(mask.sum()),
                    },
                )
            )
            group_idx += 1

    return passthrough + grouped


__all__ = [
    "coalesce_risk_regions",
    "extract_alpha_keyer_disagreement_regions",
    "extract_hard_edge_candidate_regions",
    "extract_same_bg_enclosed_regions",
]
