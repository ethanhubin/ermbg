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


def analyze_checkerboard_background(image_srgb: np.ndarray) -> dict[str, object]:
    """Detect a fake-transparent gray/white checkerboard sheet background."""
    if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
        raise ValueError("image_srgb must be HxWx3 uint8")

    h, w = image_srgb.shape[:2]
    if h < 32 or w < 32:
        return {"accepted": False, "reason": "image too small for checkerboard analysis"}

    stride = max(1, int(np.ceil(float(max(h, w)) / 768.0)))
    scale = 1.0 / float(stride)
    small = image_srgb[::stride, ::stride]
    sh, sw = small.shape[:2]
    band = max(4, min(24, int(round(float(min(sh, sw)) * 0.08))))
    yy, xx = np.indices((sh, sw))
    edge = np.zeros((sh, sw), dtype=bool)
    edge[:band, :] = True
    edge[-band:, :] = True
    edge[:, :band] = True
    edge[:, -band:] = True

    small_f = small.astype(np.float32)
    luma = (0.2126 * small_f[..., 0] + 0.7152 * small_f[..., 1] + 0.0722 * small_f[..., 2]).astype(np.float32)
    chroma_span = small_f.max(axis=2) - small_f.min(axis=2)
    neutral_bright = (small_f.mean(axis=2) >= 180.0) & (chroma_span <= 18.0)
    edge_sample_mask = edge & neutral_bright
    min_samples = max(64, int(round(float(edge.sum()) * 0.25)))
    if int(edge_sample_mask.sum()) < min_samples:
        return {
            "accepted": False,
            "reason": "insufficient bright neutral border samples",
            "bright_neutral_edge_pixels": int(edge_sample_mask.sum()),
            "min_samples": int(min_samples),
        }

    values = luma[edge_sample_mask]
    low_luma = float(np.percentile(values, 35.0))
    high_luma = float(np.percentile(values, 65.0))
    contrast = high_luma - low_luma
    # Fake transparency grids are visible but subtle. Very low contrast is just
    # drift; very high contrast is likely real artwork or a strong pattern.
    if contrast < 3.0 or contrast > 36.0:
        return {
            "accepted": False,
            "reason": "border neutral split is outside checkerboard contrast range",
            "luma_p35": low_luma,
            "luma_p65": high_luma,
            "contrast": contrast,
        }

    strip_masks = {
        "top": yy < band,
        "bottom": yy >= sh - band,
        "left": xx < band,
        "right": xx >= sw - band,
    }
    best: tuple[float, float, float, int, int, int, float, float, str, np.ndarray, float] | None = None
    max_tile = max(5, min(36, int(round(72.0 * scale)), int(round(float(min(sh, sw)) * 0.12))))
    for strip_name, strip_mask in strip_masks.items():
        sample_mask = strip_mask & neutral_bright
        if int(sample_mask.sum()) < max(64, int(round(float(strip_mask.sum()) * 0.25))):
            continue
        strip_values = luma[sample_mask]
        sample_y, sample_x = np.nonzero(sample_mask)
        # Period fitting does not need every edge pixel. A deterministic sample
        # keeps upload-time detection light while preserving all phases.
        if sample_y.size > 3000:
            take = np.linspace(0, sample_y.size - 1, 3000, dtype=np.int32)
            sample_y = sample_y[take]
            sample_x = sample_x[take]
            strip_values = strip_values[take]
        for tile in range(4, max_tile + 1):
            step = max(1, tile // 3)
            for ox in range(0, tile, step):
                for oy in range(0, tile, step):
                    p = (((sample_x + ox) // tile + (sample_y + oy) // tile) & 1).astype(bool)
                    if p.mean() < 0.25 or p.mean() > 0.75:
                        continue
                    v0 = float(strip_values[~p].mean())
                    v1 = float(strip_values[p].mean())
                    c = abs(v1 - v0)
                    if c < 3.0 or c > 36.0:
                        continue
                    pred = np.where(p, v1, v0)
                    mae = float(np.mean(np.abs(strip_values - pred)))
                    residual_p90 = float(np.percentile(np.abs(strip_values - pred), 90.0))
                    score = mae / max(c, 1e-6)
                    if best is None or score < best[0]:
                        best = (score, mae, c, tile, ox, oy, v0, v1, strip_name, sample_mask, residual_p90)

    if best is None:
        return {"accepted": False, "reason": "no periodic two-color checker model fit"}

    score, mae, model_contrast, tile, ox, oy, v0, v1, strip_name, sample_mask, residual_p90 = best
    if score > 1.35:
        return {
            "accepted": False,
            "reason": "checker model residual too high",
            "checker_score": score,
            "checker_mae": mae,
            "checker_contrast": model_contrast,
        }

    parity = (((xx + ox) // tile + (yy + oy) // tile) & 1).astype(bool)
    light_parity = parity if v1 >= v0 else ~parity
    dark_parity = ~light_parity
    light_pixels = small[sample_mask & light_parity]
    dark_pixels = small[sample_mask & dark_parity]
    if light_pixels.size == 0 or dark_pixels.size == 0:
        return {"accepted": False, "reason": "empty checker color cluster"}

    light_color = np.median(light_pixels, axis=0).astype(np.uint8)
    dark_color = np.median(dark_pixels, axis=0).astype(np.uint8)
    color_delta = float(np.linalg.norm(light_color.astype(np.float32) - dark_color.astype(np.float32)))
    if color_delta < 3.0 or color_delta > 64.0:
        return {
            "accepted": False,
            "reason": "checker color distance outside expected range",
            "color_delta": color_delta,
            "light_color": [int(c) for c in light_color],
            "dark_color": [int(c) for c in dark_color],
        }

    return {
        "accepted": True,
        "reason": "accepted",
        "source": "checkerboard_border_periodic_model",
        "background_color": [int(c) for c in light_color],
        "light_color": [int(c) for c in light_color],
        "dark_color": [int(c) for c in dark_color],
        "tile_px": float(tile / max(scale, 1e-6)),
        "phase_xy": [float(ox / max(scale, 1e-6)), float(oy / max(scale, 1e-6))],
        "scale": float(scale),
        "sample_stride": int(stride),
        "checker_score": score,
        "checker_mae": mae,
        "checker_residual_p90": residual_p90,
        "checker_contrast": model_contrast,
        "two_value_tendency": float(model_contrast / max(mae, 1e-6)),
        "color_delta": color_delta,
        "bright_neutral_edge_pixels": int(sample_mask.sum()),
        "bright_neutral_all_edge_pixels": int(edge_sample_mask.sum()),
        "edge_pixels": int(edge.sum()),
        "edge_sample_fraction": float(sample_mask.mean()),
        "fit_strip": strip_name,
    }


def _flood_from_border(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    work = mask.astype(np.uint8).copy()
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    seeds: list[tuple[int, int]] = []
    _, xs = np.nonzero(work[0:1, :])
    seeds.extend((int(x), 0) for x in xs)
    _, xs = np.nonzero(work[-1:, :])
    seeds.extend((int(x), h - 1) for x in xs)
    ys, _ = np.nonzero(work[:, 0:1])
    seeds.extend((0, int(y)) for y in ys)
    ys, _ = np.nonzero(work[:, -1:])
    seeds.extend((w - 1, int(y)) for y in ys)
    for x, y in seeds:
        if work[y, x]:
            cv2.floodFill(work, flood, (x, y), 2)
    return work == 2


def normalize_checkerboard_background_to_light_square(image_srgb: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Map exterior fake-transparent checkerboard pixels to the light square."""
    info = analyze_checkerboard_background(image_srgb)
    if not info.get("accepted", False):
        return image_srgb, {"enabled": True, "applied": False, **info}

    light = np.asarray(info["light_color"], dtype=np.uint8)
    dark = np.asarray(info["dark_color"], dtype=np.uint8)
    light_lab = srgb_to_oklab(light.reshape(1, 1, 3))[0, 0]
    dark_lab = srgb_to_oklab(dark.reshape(1, 1, 3))[0, 0]
    lab = srgb_to_oklab(image_srgb)
    distance_to_light = oklab_distance(lab, light_lab)
    distance_to_dark = oklab_distance(lab, dark_lab)
    nearest_distance = np.minimum(distance_to_light, distance_to_dark)
    # Checker squares can be blurred/compressed. Connectivity limits where this
    # loose color gate is allowed to rewrite pixels.
    close_to_checker = nearest_distance <= 7.5
    exterior_checker = _flood_from_border(close_to_checker)
    changed = exterior_checker & np.any(image_srgb != light.reshape(1, 1, 3), axis=2)
    if int(exterior_checker.sum()) < max(32, int(round(float(image_srgb.shape[0] * image_srgb.shape[1]) * 0.01))):
        return image_srgb, {
            "enabled": True,
            "applied": False,
            "reason": "insufficient exterior checkerboard support",
            **info,
            "exterior_checker_pixels": int(exterior_checker.sum()),
            "changed_pixels": 0,
        }

    normalized = image_srgb.copy()
    normalized[exterior_checker] = light.reshape(1, 3)
    return normalized, {
        "enabled": True,
        "applied": bool(changed.any()),
        "reason": "checkerboard background normalized to light square",
        **info,
        "background_color": [int(c) for c in light],
        "exterior_checker_pixels": int(exterior_checker.sum()),
        "changed_pixels": int(changed.sum()),
        "distance_threshold_oklab": 7.5,
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


def exterior_background_mask_from_background(
    image_srgb: np.ndarray,
    background_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Find background-color pixels connected to the image exterior."""
    bg = background_color or estimate_background_color(image_srgb)
    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(np.asarray(bg, dtype=np.uint8).reshape(1, 1, 3)).reshape(3)
    dist = oklab_distance(lab, bg_lab)

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
    # This is a tight background-ownership gate, separate from the foreground
    # detection threshold. It follows measured border noise, but stays strict so
    # low-contrast blue button material is not promoted to exterior background.
    background_threshold = max(1.5, median + 3.0 * mad + 1.0)
    close_to_bg = dist <= background_threshold

    count, labels = cv2.connectedComponents(close_to_bg.astype(np.uint8), connectivity=8)
    exterior_labels = np.zeros(count, dtype=bool)
    exterior_labels[np.unique(labels[0, :])] = True
    exterior_labels[np.unique(labels[-1, :])] = True
    exterior_labels[np.unique(labels[:, 0])] = True
    exterior_labels[np.unique(labels[:, -1])] = True
    exterior_labels[0] = False
    return exterior_labels[labels]


def find_slice_boxes(
    mask: np.ndarray,
    *,
    min_area: int = 64,
    padding: int = 2,
    close_iterations: int = 1,
    exterior_background_mask: np.ndarray | None = None,
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

    boxes = merge_overlapping_slice_boxes(boxes, exterior_background_mask=exterior_background_mask)
    boxes.sort(key=lambda box: (box.bbox[1], box.bbox[0]))
    return [SliceBox(id=i + 1, bbox=box.bbox, area=box.area) for i, box in enumerate(boxes)]


def _boxes_overlap(a: SliceBox, b: SliceBox) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _boxes_are_touching_shadow_strip(a: SliceBox, b: SliceBox) -> bool:
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
    # component whose bbox touches the main control. Without image background
    # evidence, stay conservative so stacked independent controls remain split.
    return y_gap <= 2 and overlap_ratio >= 0.65 and thin_ratio <= 0.22


def _boxes_are_vertically_related(a: SliceBox, b: SliceBox) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    y_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
    if x_overlap <= 0 or y_overlap > 0:
        return False
    y_gap = max(0, max(ay, by) - min(ay + ah, by + bh))
    gap_limit = min(96, max(2, int(round(max(ah, bh) * 0.35))))
    # With exterior-background evidence available, this is only a candidate
    # relation. The actual split/merge decision is made by the background
    # corridor test below.
    return y_gap <= gap_limit and x_overlap / float(max(1, min(aw, bw))) >= 0.45


def _has_exterior_background_barrier_between(a: SliceBox, b: SliceBox, exterior_background_mask: np.ndarray) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    top, bottom = (a, b) if ay <= by else (b, a)
    tx, ty, tw, th = top.bbox
    bx2, by2, bw2, _ = bottom.bbox
    x0 = max(tx, bx2)
    x1 = min(tx + tw, bx2 + bw2)
    y0 = ty + th
    y1 = by2
    if x1 <= x0 or y1 <= y0:
        return False

    gap = exterior_background_mask[y0:y1, x0:x1]
    if gap.size == 0:
        return False

    # A true split is an exterior-background corridor crossing the whole gap.
    # Connected left-to-right support expresses background-color obstruction
    # directly and avoids deciding from bbox size alone.
    count, labels = cv2.connectedComponents(gap.astype(np.uint8), connectivity=8)
    if count <= 1:
        return False
    left = set(int(v) for v in np.unique(labels[:, 0]) if int(v) != 0)
    right = set(int(v) for v in np.unique(labels[:, -1]) if int(v) != 0)
    return bool(left & right)


def _boxes_should_merge(a: SliceBox, b: SliceBox, exterior_background_mask: np.ndarray | None = None) -> bool:
    if _boxes_overlap(a, b):
        return True
    if exterior_background_mask is None:
        return _boxes_are_touching_shadow_strip(a, b)
    if not _boxes_are_vertically_related(a, b):
        return False
    if _has_exterior_background_barrier_between(a, b, exterior_background_mask):
        return False
    return True


def _merge_pair(a: SliceBox, b: SliceBox) -> SliceBox:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return SliceBox(id=0, bbox=(x0, y0, x1 - x0, y1 - y0), area=a.area + b.area)


def merge_overlapping_slice_boxes(
    boxes: list[SliceBox],
    *,
    exterior_background_mask: np.ndarray | None = None,
) -> list[SliceBox]:
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
            if _boxes_should_merge(box, boxes[j], exterior_background_mask=exterior_background_mask):
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
    exterior_background_mask = exterior_background_mask_from_background(image_srgb, background_color=bg)
    boxes = find_slice_boxes(
        mask,
        min_area=min_area,
        padding=padding,
        close_iterations=close_iterations,
        exterior_background_mask=exterior_background_mask,
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
    "exterior_background_mask_from_background",
    "find_slice_boxes",
    "foreground_mask_from_background",
    "merge_overlapping_slice_boxes",
    "pad_slice_box",
    "save_slices",
    "slice_image",
]
