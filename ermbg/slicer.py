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
    padding: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "background_color": list(self.background_color),
            "count": len(self.boxes),
            "boxes": [box.to_dict() for box in self.boxes],
            "padding": self.padding,
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


def foreground_mask_from_background(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | None = None,
    distance_threshold: float | None = None,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    """Build a foreground mask by OKLab distance from the estimated background."""
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
    return mask, bg, float(distance_threshold)


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
        boxes.append(SliceBox(id=0, bbox=(x, y, bw, bh), area=area))

    boxes = merge_overlapping_slice_boxes(boxes)
    boxes.sort(key=lambda box: (box.bbox[1], box.bbox[0]))
    return [SliceBox(id=i + 1, bbox=box.bbox, area=box.area) for i, box in enumerate(boxes)]


def _boxes_overlap(a: SliceBox, b: SliceBox) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _boxes_are_near_aligned_parts(a: SliceBox, b: SliceBox) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    y_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
    x_gap = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    y_gap = max(0, max(ay, by) - min(ay + ah, by + bh))
    if x_gap > 0 or y_overlap > 0:
        return False
    overlap_ratio = x_overlap / float(max(1, min(aw, bw)))
    thin_ratio = min(ah, bh) / float(max(1, max(ah, bh)))
    # Sprite sheets often render a button/drop shadow as a separate thin
    # component whose bbox touches the main control. Merge only when the pieces
    # are vertically adjacent, strongly horizontally aligned, and one piece is
    # clearly a shallow appendage, so stacked independent controls stay split.
    return y_gap <= 2 and overlap_ratio >= 0.65 and thin_ratio <= 0.22


def _boxes_should_merge(a: SliceBox, b: SliceBox) -> bool:
    return _boxes_overlap(a, b) or _boxes_are_near_aligned_parts(a, b)


def _merge_pair(a: SliceBox, b: SliceBox) -> SliceBox:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return SliceBox(id=0, bbox=(x0, y0, x1 - x0, y1 - y0), area=a.area + b.area)


def merge_overlapping_slice_boxes(boxes: list[SliceBox]) -> list[SliceBox]:
    """Collapse any intersecting slice rectangles into their union rectangle.

    A fragmented translucent object can produce a main component plus interior
    highlight/bubble components. Once padded, those boxes overlap and should be
    exported as one larger slice, not as duplicate crops over the same pixels.
    """
    if not boxes:
        return []

    parents = list(range(len(boxes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    for i, box in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            if _boxes_should_merge(box, boxes[j]):
                union(i, j)

    groups: dict[int, list[SliceBox]] = {}
    for i, box in enumerate(boxes):
        groups.setdefault(find(i), []).append(box)

    merged: list[SliceBox] = []
    for group in groups.values():
        current = group[0]
        for box in group[1:]:
            current = _merge_pair(current, box)
        merged.append(current)

    merged.sort(key=lambda box: (box.bbox[1], box.bbox[0]))
    return [SliceBox(id=i + 1, bbox=box.bbox, area=box.area) for i, box in enumerate(merged)]


def slice_image(
    image_srgb: np.ndarray,
    *,
    background_color: tuple[int, int, int] | None = None,
    distance_threshold: float | None = None,
    min_area: int = 64,
    padding: int = 2,
    close_iterations: int = 1,
) -> SliceResult:
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
    return SliceResult(background_color=bg, foreground_mask=mask.astype(np.float32), boxes=boxes, padding=max(0, int(padding)))


def pad_slice_box(box: SliceBox, image_shape: tuple[int, int], padding: int) -> SliceBox:
    if padding <= 0:
        return box
    h, w = image_shape[:2]
    x, y, bw, bh = box.bbox
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w, x + bw + padding)
    y1 = min(h, y + bh + padding)
    return SliceBox(id=box.id, bbox=(x0, y0, x1 - x0, y1 - y0), area=box.area)


def crop_slice(
    image_srgb: np.ndarray,
    mask: np.ndarray,
    box: SliceBox,
    *,
    padding: int = 0,
    transparent: bool = False,
) -> np.ndarray:
    box = pad_slice_box(box, image_srgb.shape[:2], padding)
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
        crop = crop_slice(image_srgb, result.foreground_mask, box, padding=result.padding, transparent=transparent)
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
    "merge_overlapping_slice_boxes",
    "pad_slice_box",
    "save_slices",
    "slice_image",
]
