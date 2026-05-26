"""Build visual planner requests for future VLM-backed candidate planning.

This module does not call a remote model. It packages the existing
``PlannerPromptBundle`` with small PNG attachments so a VLM can interpret local
EvidenceRegions and return the same constrained ``CandidatePlan`` JSON that the
rule planner already produces.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .planner import PlannerPromptBundle, RiskRegion, build_planner_prompt_bundle

DEFAULT_VLM_INSTRUCTIONS = [
    "Return CandidatePlan JSON only.",
    "Use only registered tools and existing region_id values.",
    "Do not output alpha, RGBA, masks, or image-processing code.",
    "For shadows, provide only semantic subject/search/ownership interpretation; local CV must verify known-background darkening and set opacity.",
    "Each candidate must be a complete whole-image interpretation, not one candidate per region.",
    "Combine compatible operations in the same candidate when they should be applied together.",
    "If two interpretations differ only in an ambiguous region, repeat shared repair operations in both candidates.",
    "Mark exactly one candidate as selected when any candidates are returned.",
    "Prefer 0 to 4 candidates. More than 4 candidates means the ambiguity is not structured enough.",
]

REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "same_bg_enclosed_region": (255, 0, 255),
    "alpha_keyer_disagreement": (0, 220, 255),
    "hard_edge_candidate": (255, 170, 0),
    "soft_edge_band": (80, 160, 255),
    "opaque_interior": (80, 220, 120),
    "translucent_candidate": (180, 120, 255),
    "subject_material_candidate": (40, 230, 180),
    "intentional_hole": (255, 80, 120),
    "subject_owned_region": (60, 220, 120),
    "owned_shadow_candidate": (90, 90, 255),
    "shadow_search_region": (120, 140, 255),
    "unknown": (255, 255, 0),
}


@dataclass(frozen=True)
class VLMImageAttachment:
    """One PNG image attached to a VLM planner request."""

    id: str
    purpose: str
    width: int
    height: int
    data_base64: str
    mime_type: str = "image/png"
    region_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "purpose": self.purpose,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "data_base64": self.data_base64,
            "metadata": dict(self.metadata),
        }
        if self.region_id is not None:
            payload["region_id"] = self.region_id
        return payload


@dataclass(frozen=True)
class VLMPlannerRequest:
    """Complete local payload for a VLM candidate planner."""

    planner_bundle: dict[str, Any]
    attachments: list[VLMImageAttachment]
    response_schema: dict[str, Any]
    system_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "planner_bundle": self.planner_bundle,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
            "response_schema": self.response_schema,
        }


def _assert_image_shapes(image_srgb: np.ndarray, base_rgba: np.ndarray) -> None:
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")
    if base_rgba.dtype != np.uint8 or base_rgba.ndim != 3 or base_rgba.shape[2] != 4:
        raise ValueError("base_rgba must be HxWx4 uint8")
    if image_srgb.shape[:2] != base_rgba.shape[:2]:
        raise ValueError("image_srgb and base_rgba must share HxW")


def _png_attachment(
    *,
    attachment_id: str,
    purpose: str,
    image: np.ndarray,
    region_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_side: int,
) -> VLMImageAttachment:
    if image.dtype != np.uint8:
        raise ValueError("attachment image must be uint8")
    pil = Image.fromarray(image, mode="RGBA" if image.shape[2] == 4 else "RGB")
    pil.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return VLMImageAttachment(
        id=attachment_id,
        purpose=purpose,
        region_id=region_id,
        width=int(pil.width),
        height=int(pil.height),
        data_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
        metadata=dict(metadata or {}),
    )


def _checker_composite(rgba: np.ndarray, cell: int = 16) -> np.ndarray:
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((xx // cell + yy // cell) % 2) * 70 + 185).astype(np.uint8)
    checker_rgb = np.dstack([checker, checker, checker])
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    comp = rgba[..., :3].astype(np.float32) * alpha + checker_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(comp, 0, 255).astype(np.uint8)


def _solid_composite(rgba: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    bg = np.empty(rgba.shape[:2] + (3,), dtype=np.uint8)
    bg[...] = np.asarray(color, dtype=np.uint8)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    comp = rgba[..., :3].astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(comp, 0, 255).astype(np.uint8)


def _overlay_regions(image_srgb: np.ndarray, regions: list[RiskRegion]) -> np.ndarray:
    overlay = image_srgb.copy()
    for region in regions:
        color = np.asarray(REGION_COLORS.get(region.kind, REGION_COLORS["unknown"]), dtype=np.float32)
        mask = region.mask.astype(bool)
        if not mask.any():
            continue
        overlay[mask] = (0.55 * overlay[mask].astype(np.float32) + 0.45 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in color), 2)
    return overlay


def _bbox_from_mask(mask: np.ndarray, padding_px: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape
    if ys.size == 0:
        return 0, 0, min(w, 1), min(h, 1)
    x0 = max(0, int(xs.min()) - padding_px)
    y0 = max(0, int(ys.min()) - padding_px)
    x1 = min(w, int(xs.max()) + 1 + padding_px)
    y1 = min(h, int(ys.max()) + 1 + padding_px)
    return x0, y0, x1, y1


def _region_crop(image_srgb: np.ndarray, region: RiskRegion, padding_px: int) -> tuple[np.ndarray, dict[str, Any]]:
    x0, y0, x1, y1 = _bbox_from_mask(region.mask, padding_px)
    crop = image_srgb[y0:y1, x0:x1].copy()
    local_mask = region.mask[y0:y1, x0:x1]
    if local_mask.any():
        color = np.asarray(REGION_COLORS.get(region.kind, REGION_COLORS["unknown"]), dtype=np.float32)
        crop[local_mask] = (0.45 * crop[local_mask].astype(np.float32) + 0.55 * color).astype(np.uint8)
        contours, _ = cv2.findContours(local_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, tuple(int(c) for c in color), 2)
    return crop, {"bbox_xyxy": [x0, y0, x1, y1], "padding_px": int(padding_px)}


def build_vlm_response_schema(bundle: PlannerPromptBundle) -> dict[str, Any]:
    """Return a strict-enough JSON schema for VLM CandidatePlan output."""
    payload = bundle.to_dict()
    tool_names = [str(tool["name"]) for tool in payload["tools"]]
    region_ids = [str(region["id"]) for region in payload["regions"]]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "label", "confidence", "selected", "operations", "reason"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "label": {"type": "string", "minLength": 1},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "selected": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "operations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["tool", "region_id", "parameters"],
                                "properties": {
                                    "tool": {"type": "string", "enum": tool_names},
                                    "region_id": {"type": "string", "enum": region_ids},
                                    "parameters": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {},
                                        "required": [],
                                    },
                                },
                            },
                        },
                    },
                },
            }
        },
    }


def build_vlm_planner_request(
    *,
    image_srgb: np.ndarray,
    base_rgba: np.ndarray,
    regions: list[RiskRegion],
    background_color: tuple[int, int, int] | None = None,
    strategy: dict[str, Any] | None = None,
    instructions: list[str] | None = None,
    bundle: PlannerPromptBundle | None = None,
    max_region_crops: int = 8,
    thumbnail_max_side: int = 768,
    crop_max_side: int = 384,
    crop_padding_px: int = 16,
) -> VLMPlannerRequest:
    """Package evidence JSON and visual context for a VLM planner.

    The returned object is intentionally JSON-serializable except for the
    dataclass wrappers. Attachments are small base64 PNGs: original thumbnail,
    base matte composites, an evidence overlay, and crops for the largest
    evidence regions.
    """
    _assert_image_shapes(image_srgb, base_rgba)
    for region in regions:
        if region.mask.shape != image_srgb.shape[:2]:
            raise ValueError(f"region {region.id} mask shape does not match image")

    planner_bundle = bundle or build_planner_prompt_bundle(
        image_shape=image_srgb.shape,
        regions=regions,
        background_color=background_color,
        strategy=strategy,
        instructions=instructions or DEFAULT_VLM_INSTRUCTIONS,
    )
    attachments = [
        _png_attachment(
            attachment_id="original_thumbnail",
            purpose="Original RGB input.",
            image=image_srgb,
            max_side=thumbnail_max_side,
        ),
        _png_attachment(
            attachment_id="base_on_checker",
            purpose="Current base RGBA matte composited on checkerboard.",
            image=_checker_composite(base_rgba),
            max_side=thumbnail_max_side,
        ),
        _png_attachment(
            attachment_id="base_on_black",
            purpose="Current base RGBA matte composited on black.",
            image=_solid_composite(base_rgba, (0, 0, 0)),
            max_side=thumbnail_max_side,
        ),
        _png_attachment(
            attachment_id="base_on_white",
            purpose="Current base RGBA matte composited on white.",
            image=_solid_composite(base_rgba, (255, 255, 255)),
            max_side=thumbnail_max_side,
        ),
        _png_attachment(
            attachment_id="evidence_overlay",
            purpose="Original image with local EvidenceRegions color-overlaid.",
            image=_overlay_regions(image_srgb, regions),
            max_side=thumbnail_max_side,
            metadata={"region_colors": REGION_COLORS},
        ),
    ]

    ranked_regions = sorted(regions, key=lambda region: int(region.mask.sum()), reverse=True)
    for index, region in enumerate(ranked_regions[: max(0, int(max_region_crops))]):
        crop, metadata = _region_crop(image_srgb, region, crop_padding_px)
        attachments.append(
            _png_attachment(
                attachment_id=f"region_crop_{index}",
                purpose=f"Crop around EvidenceRegion {region.id}.",
                image=crop,
                region_id=region.id,
                metadata={**metadata, "kind": region.kind, "evidence_kind": region.to_prompt_dict()["evidence_kind"]},
                max_side=crop_max_side,
            )
        )

    return VLMPlannerRequest(
        planner_bundle=planner_bundle.to_dict(),
        attachments=attachments,
        response_schema=build_vlm_response_schema(planner_bundle),
        system_prompt=(
            "You are ERMBG's visual planning layer. Interpret local evidence regions "
            "and return constrained CandidatePlan JSON for deterministic local matting tools."
        ),
    )


__all__ = [
    "DEFAULT_VLM_INSTRUCTIONS",
    "REGION_COLORS",
    "VLMImageAttachment",
    "VLMPlannerRequest",
    "build_vlm_planner_request",
    "build_vlm_response_schema",
]
