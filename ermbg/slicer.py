"""Rectangle slicing for solid-background sprite sheets or object sheets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import io
from .colorspace import oklab_distance, srgb_to_oklab


@dataclass(frozen=True)
class SliceBox:
    id: int
    bbox: tuple[int, int, int, int]  # x, y, width, height
    area: int

    def to_dict(self) -> dict[str, object]:
        x, y, w, h = self.bbox
        return {"id": self.id, "bbox": [x, y, w, h], "area": self.area}


@dataclass(frozen=True)
class SliceResult:
    background_color: tuple[int, int, int]
    foreground_mask: np.ndarray
    boxes: list[SliceBox]

    def to_dict(self) -> dict[str, object]:
        return {
            "background_color": list(self.background_color),
            "count": len(self.boxes),
            "boxes": [box.to_dict() for box in self.boxes],
        }


@dataclass(frozen=True)
class UIKindPrediction:
    kind: str
    confidence: float
    features: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "features": self.features,
        }


def estimate_background_color(image_srgb: np.ndarray, border_fraction: float = 0.08) -> tuple[int, int, int]:
    """Estimate the sheet background from border pixels.

    The slicer is meant for already-separated subjects on a solid or nearly
    solid color. Border sampling avoids needing a matting model and works for
    object sheets where subjects are separated away from the canvas edges.
    """
    h, w = image_srgb.shape[:2]
    border = max(1, int(round(min(h, w) * border_fraction)))
    samples = np.concatenate(
        [
            image_srgb[:border, :, :].reshape(-1, 3),
            image_srgb[-border:, :, :].reshape(-1, 3),
            image_srgb[:, :border, :].reshape(-1, 3),
            image_srgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(samples, axis=0)
    return tuple(int(c) for c in bg.astype(np.uint8))


def _border_samples(image_srgb: np.ndarray, border_fraction: float = 0.08) -> np.ndarray:
    h, w = image_srgb.shape[:2]
    border = max(1, int(round(min(h, w) * border_fraction)))
    return np.concatenate(
        [
            image_srgb[:border, :, :].reshape(-1, 3),
            image_srgb[-border:, :, :].reshape(-1, 3),
            image_srgb[:, :border, :].reshape(-1, 3),
            image_srgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )


def _checkerboard_background_mask(image_srgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]] | None:
    """Return a background mask when the sheet uses a baked checkerboard."""
    samples = _border_samples(image_srgb)
    sample_chroma = samples.max(axis=1).astype(np.int16) - samples.min(axis=1).astype(np.int16)
    neutral = sample_chroma <= 7
    if float(np.mean(neutral)) < 0.72:
        return None

    gray_samples = samples[neutral].astype(np.float32).mean(axis=1)
    lo = float(np.percentile(gray_samples, 25.0))
    hi = float(np.percentile(gray_samples, 75.0))
    if hi - lo < 8.0:
        return None

    image_i = image_srgb.astype(np.int16)
    chroma = image_i.max(axis=2) - image_i.min(axis=2)
    gray = image_srgb.astype(np.float32).mean(axis=2)
    tolerance = max(3.0, min(7.0, float(np.median(np.abs(gray_samples - np.median(gray_samples)))) * 0.35 + 3.0))
    bg_mask = (chroma <= 8) & (np.minimum(np.abs(gray - lo), np.abs(gray - hi)) <= tolerance)
    if float(np.mean(bg_mask)) < 0.20:
        return None
    bg = tuple(int(c) for c in np.median(samples[neutral], axis=0).astype(np.uint8))
    return bg_mask, bg


def foreground_mask_from_background(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | None = None,
    distance_threshold: float | None = None,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """Build a foreground mask by OKLab distance from the estimated background."""
    checker = None if background_color is not None else _checkerboard_background_mask(image_srgb)
    if checker is not None:
        bg_mask, bg = checker
        # Failure mechanism: checkerboard preview/export backgrounds are not a
        # single solid color. Treat only the two neutral border colors as
        # background; colored UI material and white/gray frame art remain
        # foreground instead of being compared to a misleading median gray.
        mask = ~bg_mask
        return mask, bg, 0.0

    bg = background_color or estimate_background_color(image_srgb)
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(bg, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    dist = oklab_distance(lab, bg_lab)

    if distance_threshold is None:
        h, w = image_srgb.shape[:2]
        border = max(1, int(round(min(h, w) * 0.08)))
        border_dist = np.concatenate(
            [
                dist[:border, :].reshape(-1),
                dist[-border:, :].reshape(-1),
                dist[:, :border].reshape(-1),
                dist[:, -border:].reshape(-1),
            ]
        )
        median = float(np.median(border_dist))
        mad = float(np.median(np.abs(border_dist - median)))
        # Empirical gate: solid backgrounds usually have OKLab noise below 2,
        # but JPEG/WebP compression and antialiased edges can raise border
        # samples. The max floor keeps subtle shadows off the background from
        # splitting into false objects; the MAD term follows noisy exports.
        distance_threshold = max(6.0, median + 6.0 * mad + 2.0)

    mask = dist > float(distance_threshold)
    mask = _include_attached_background_shadows(
        mask,
        lab,
        bg_lab,
        distance_threshold=float(distance_threshold),
    )
    return mask, bg, float(distance_threshold)


def _include_attached_background_shadows(
    foreground_mask: np.ndarray,
    lab: np.ndarray,
    bg_lab: np.ndarray,
    *,
    distance_threshold: float,
) -> np.ndarray:
    """Attach same-background darkening to nearby subject seeds.

    The slicer boxes object sheets, not physical alpha layers. Cast shadows are
    often a low-contrast scalar darkening of the known background: strong bands
    can become separate components, while the softer tail falls below the
    foreground distance threshold. This hysteresis mask accepts only pixels that
    get darker with very little OKLab hue/chroma drift, then keeps the connected
    shadow pieces that are spatially anchored to a high-confidence foreground
    seed. The numeric gates are empirical but tied to observable signals:
    OKLab L drop for shadow strength, ab distance for "same background family",
    and image-relative anchoring radius for soft cast-shadow reach.
    """
    if not foreground_mask.any():
        return foreground_mask

    h, w = foreground_mask.shape[:2]
    l_drop = (float(bg_lab[0]) - lab[..., 0]) * 100.0
    ab_dist = np.sqrt((lab[..., 1] - bg_lab[1]) ** 2 + (lab[..., 2] - bg_lab[2]) ** 2) * 100.0
    min_shadow_l_drop = max(1.6, min(3.2, distance_threshold * 0.32))
    max_shadow_ab_dist = max(3.5, min(5.2, distance_threshold * 0.72))
    shadow_like = (l_drop >= min_shadow_l_drop) & (ab_dist <= max_shadow_ab_dist)
    if not shadow_like.any():
        return foreground_mask

    radius = max(5, min(64, int(round(min(h, w) * 0.035))))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    anchored = cv2.dilate(foreground_mask.astype(np.uint8), kernel, iterations=1) > 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(shadow_like.astype(np.uint8), connectivity=8)
    attached_shadow = np.zeros_like(foreground_mask, dtype=bool)
    min_shadow_area = max(4, int(round(h * w * 0.000004)))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_shadow_area:
            continue
        component = labels == label
        if np.any(component & anchored):
            attached_shadow |= component

    if not attached_shadow.any():
        return foreground_mask
    return foreground_mask | attached_shadow


def find_slice_boxes(
    mask: np.ndarray,
    *,
    min_area: int = 64,
    padding: int = 2,
    close_iterations: int = 1,
) -> list[SliceBox]:
    """Find connected foreground rectangles, sorted top-to-bottom then left-to-right."""
    m = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    if close_iterations > 0:
        # A single close pass joins small antialias gaps within one object while
        # preserving clearly separated subjects on sprite/object sheets.
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    del labels
    h, w = mask.shape[:2]
    boxes: list[SliceBox] = []
    for label in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[label])
        if area < min_area:
            continue
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)
        boxes.append(SliceBox(id=0, bbox=(x0, y0, x1 - x0, y1 - y0), area=area))

    boxes = _merge_attached_shadow_boxes(boxes)
    boxes.sort(key=lambda box: (box.bbox[1], box.bbox[0]))
    return [SliceBox(id=i + 1, bbox=box.bbox, area=box.area) for i, box in enumerate(boxes)]


def _merge_attached_shadow_boxes(boxes: list[SliceBox]) -> list[SliceBox]:
    """Merge flat shadow components into the object box they visually belong to."""
    merged = [SliceBox(id=box.id, bbox=box.bbox, area=box.area) for box in boxes]
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        next_boxes: list[SliceBox] = []
        for i, base in enumerate(merged):
            if used[i]:
                continue
            current = base
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                other = merged[j]
                if _looks_like_attached_shadow_pair(current, other):
                    current = _union_slice_boxes(current, other)
                    used[j] = True
                    changed = True
            used[i] = True
            next_boxes.append(current)
        merged = next_boxes
    return merged


def _looks_like_attached_shadow_pair(a: SliceBox, b: SliceBox) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    a_area = max(1, a.area)
    b_area = max(1, b.area)
    lower, upper = (a, b) if ay >= by else (b, a)
    _, ly, lw, lh = lower.bbox
    _, uy, uw, uh = upper.bbox
    x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_ratio = x_overlap / float(max(1, min(aw, bw)))
    vertical_gap = max(0, max(ay, by) - min(ay + ah, by + bh))
    large_h = max(ah, bh)
    flat_lower = lh <= max(8, int(round(0.70 * max(1, uh))))
    shadow_width_support = lw >= max(4, int(round(0.70 * max(1, uw))))
    not_dominant_lower = lower.area <= max(a_area, b_area) * 1.20

    # Shadow strips are broad, low-height components below or beside a seeded
    # object. Merging requires strong horizontal overlap and a small vertical
    # gap, which keeps ordinary separated rows from being collapsed together.
    return (
        ly >= uy
        and overlap_ratio >= 0.35
        and vertical_gap <= max(12, int(round(0.22 * large_h)))
        and flat_lower
        and shadow_width_support
        and not_dominant_lower
    )


def _union_slice_boxes(a: SliceBox, b: SliceBox) -> SliceBox:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return SliceBox(id=0, bbox=(x0, y0, x1 - x0, y1 - y0), area=a.area + b.area)


def slice_image(
    image_srgb: np.ndarray,
    *,
    background_color: tuple[int, int, int] | None = None,
    distance_threshold: float | None = None,
    min_area: int = 64,
    padding: int = 2,
    close_iterations: int = 1,
    source_alpha: np.ndarray | None = None,
) -> SliceResult:
    if source_alpha is not None and source_alpha.shape == image_srgb.shape[:2] and np.any(source_alpha < 0.995):
        # Transparent PNG atlases already carry exact background ownership.
        # Color-distance slicing on their hidden RGB values is unstable because
        # transparent pixels often contain checkerboard previews or arbitrary
        # exporter residue.
        mask = source_alpha.astype(np.float32) > 0.015
        bg = (0, 0, 0)
    else:
        mask, bg, _ = foreground_mask_from_background(
            image_srgb,
            background_color=background_color,
            distance_threshold=distance_threshold,
        )
    boxes = find_slice_boxes(
        mask,
        min_area=min_area,
        padding=padding,
        close_iterations=close_iterations,
    )
    return SliceResult(background_color=bg, foreground_mask=mask.astype(np.float32), boxes=boxes)


def crop_slice(
    image_srgb: np.ndarray,
    mask: np.ndarray,
    box: SliceBox,
    *,
    transparent: bool = False,
) -> np.ndarray:
    x, y, w, h = box.bbox
    crop = image_srgb[y : y + h, x : x + w]
    if not transparent:
        return crop.copy()
    alpha = (mask[y : y + h, x : x + w] > 0.5).astype(np.uint8) * 255
    return np.dstack([crop, alpha]).astype(np.uint8)


def classify_ui_slice(
    crop_srgb: np.ndarray,
    box: SliceBox,
    sheet_shape: tuple[int, int],
    foreground_mask: np.ndarray | None = None,
) -> UIKindPrediction:
    """Classify common UI slices with cheap local CV heuristics.

    These are broad priors, not semantic truth: UI sprite sheets often contain
    predictable geometry. Empirical thresholds key on observable shape signals
    (aspect ratio, relative area, fill and edge density) so uncertain assets can
    fall back to ``asset`` rather than pretending to know the exact category.
    """
    x, y, w, h = box.bbox
    sheet_h, sheet_w = sheet_shape[:2]
    aspect = float(w) / float(max(1, h))
    rel_area = float(w * h) / float(max(1, sheet_w * sheet_h))

    if foreground_mask is not None:
        mask_crop = foreground_mask[y : y + h, x : x + w]
        fill_ratio = float(np.mean(mask_crop > 0.5)) if mask_crop.size else 0.0
    else:
        fill_ratio = 1.0

    gray = cv2.cvtColor(crop_srgb, cv2.COLOR_RGB2GRAY) if crop_srgb.size else np.zeros((1, 1), dtype=np.uint8)
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(np.mean(edges > 0)) if edges.size else 0.0

    features = {
        "aspect_ratio": aspect,
        "relative_area": rel_area,
        "fill_ratio": fill_ratio,
        "edge_density": edge_density,
    }

    # Panels are visually large containers; the relative-area gate prevents
    # ordinary square icons from being promoted just because they are boxy.
    if rel_area >= 0.22 and 0.65 <= aspect <= 2.8:
        return UIKindPrediction("panel", 0.82, features)

    # Buttons in game/UI sheets are usually long rounded rectangles. The fill
    # gate keeps thin divider lines and sparse decorations out of the button bin.
    if aspect >= 1.8 and fill_ratio >= 0.28:
        confidence = min(0.94, 0.62 + min(aspect, 4.0) * 0.07 + fill_ratio * 0.12)
        return UIKindPrediction("button", float(confidence), features)

    # Icons are compact, roughly square controls. Edge density helps distinguish
    # detailed icon art from a near-empty square crop.
    if 0.72 <= aspect <= 1.38 and rel_area <= 0.18:
        confidence = min(0.9, 0.62 + max(0.0, 0.18 - abs(1.0 - aspect)) + edge_density * 0.35)
        return UIKindPrediction("icon", float(confidence), features)

    # Badges/labels sit between icons and long buttons. This bucket is useful
    # for medium-width UI chips where pure geometry is suggestive but weaker.
    if 1.35 < aspect < 1.8 and fill_ratio >= 0.25:
        return UIKindPrediction("badge", 0.68, features)

    return UIKindPrediction("asset", 0.45, features)


def save_slices(
    image_srgb: np.ndarray,
    result: SliceResult,
    out_dir: Path,
    *,
    stem: str = "slice",
    transparent: bool = False,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for box in result.boxes:
        suffix = "rgba" if transparent else "rgb"
        path = out_dir / f"{stem}_{box.id:03d}_{suffix}.png"
        crop = crop_slice(image_srgb, result.foreground_mask, box, transparent=transparent)
        if transparent:
            io.save_rgba(path, crop)
        else:
            io.save_rgb(path, crop)
        paths.append(path)
    io.save_mask(out_dir / f"{stem}_mask.png", result.foreground_mask)
    (out_dir / f"{stem}.slices.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return paths


__all__ = [
    "SliceBox",
    "SliceResult",
    "UIKindPrediction",
    "classify_ui_slice",
    "crop_slice",
    "estimate_background_color",
    "find_slice_boxes",
    "foreground_mask_from_background",
    "save_slices",
    "slice_image",
]
