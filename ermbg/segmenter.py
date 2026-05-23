"""Subject coarse segmentation.

Phase 1 uses BiRefNet (ZhengPeng7/BiRefNet) when torch is available. Without torch
we provide a deterministic placeholder that uses GrabCut so the pipeline can still
run end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import metrics


@dataclass
class CoarseMask:
    soft: np.ndarray  # float32 0..1, H x W
    bbox: tuple[int, int, int, int]  # x, y, w, h
    inner: np.ndarray  # bool, eroded mask
    outer: np.ndarray  # bool, dilated mask
    unknown_band: np.ndarray  # bool, outer XOR inner


def _band_radius(image_shape: tuple[int, int]) -> int:
    """Plan section 5.3: r = max(4, min(20, 0.008 * min(W, H)))."""
    h, w = image_shape[:2]
    return int(max(4, min(20, round(0.008 * min(w, h)))))


def make_bands(soft_mask: np.ndarray, radius: int | None = None) -> CoarseMask:
    """Build inner / outer / unknown band from a soft mask."""
    if soft_mask.dtype != np.float32:
        soft_mask = soft_mask.astype(np.float32)
        if soft_mask.max() > 1.5:
            soft_mask /= 255.0

    r = radius if radius is not None else _band_radius(soft_mask.shape)
    binary = (soft_mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    inner = cv2.erode(binary, kernel, iterations=r) > 0
    outer = cv2.dilate(binary, kernel, iterations=r) > 0
    unknown = outer & (~inner)

    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        bbox = (0, 0, soft_mask.shape[1], soft_mask.shape[0])
    else:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))

    return CoarseMask(soft=soft_mask, bbox=bbox, inner=inner, outer=outer, unknown_band=unknown)


# ---------------------------------------------------------------------------
# BiRefNet backend (requires torch + transformers)
# ---------------------------------------------------------------------------


class BiRefNetSegmenter:
    """Loads BiRefNet from HuggingFace once and reuses it.

    Defaults to the matting checkpoint (`ZhengPeng7/BiRefNet-matting`) which
    was trained on P3M-10k / AM-2k / AIM-500 / Distinctions-646 / HIM2K and
    directly emits a continuous alpha matte (not a saliency mask). Pass a
    different ``model_id`` for the saliency variant or BRIA RMBG.
    """

    def __init__(
        self,
        model_id: str = "ZhengPeng7/BiRefNet-matting",
        device: str | None = None,
        input_size: int = 1024,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except ImportError as e:
            raise ImportError(
                "BiRefNetSegmenter requires the 'torch' extra. "
                "Install with `pip install -e \".[torch]\"`."
            ) from e

        self.torch = torch
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.input_size = input_size

        self.model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(device).eval()
        # Match input dtype to whatever the model parameters were loaded with
        # (BiRefNet checkpoints on HF are often fp16).
        try:
            self.dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.dtype = torch.float32

    def segment(self, image: np.ndarray, object_prompt: str | None = None) -> np.ndarray:
        """Return float32 alpha matte in [0, 1] at the same H x W as ``image``.

        With the default ``BiRefNet-matting`` checkpoint this is a true alpha
        matte (continuous, hair/fur edge details preserved). With the older
        ``BiRefNet`` (saliency) checkpoint it is a soft binary mask.
        """
        del object_prompt  # BiRefNet is class-agnostic
        torch = self.torch
        h, w = image.shape[:2]
        size = self.input_size

        # Resize, normalize to ImageNet stats.
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device).to(self.dtype)

        with torch.no_grad():
            outputs = self.model(tensor)
            logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            if hasattr(logits, "sigmoid"):
                pred = logits.sigmoid()
            else:
                pred = torch.sigmoid(logits)
            pred = pred.float().squeeze().cpu().numpy()

        # Resize back to original.
        soft = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(soft, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Fallback: GrabCut-based segmenter (no torch needed)
# ---------------------------------------------------------------------------


class GrabCutSegmenter:
    """Crude fallback when BiRefNet/torch are unavailable.

    It is *not* meant to produce phase-2-quality masks; it exists so the pipeline
    is runnable / testable without GPU dependencies.
    """

    def __init__(self, iterations: int = 5, border_ratio: float = 0.05) -> None:
        self.iterations = iterations
        self.border_ratio = border_ratio

    def segment(self, image: np.ndarray, object_prompt: str | None = None) -> np.ndarray:
        del object_prompt
        h, w = image.shape[:2]
        bx = max(1, int(w * self.border_ratio))
        by = max(1, int(h * self.border_ratio))
        rect = (bx, by, w - 2 * bx, h - 2 * by)

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = np.zeros((h, w), dtype=np.uint8)
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(bgr, mask, rect, bgd, fgd, self.iterations, cv2.GC_INIT_WITH_RECT)
        out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
        # Light feathering so the soft mask isn't strictly binary.
        out = cv2.GaussianBlur(out, (5, 5), 0)
        return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_segmenter(
    backend: str = "auto",
    **kwargs,
):
    """Pick a segmenter based on what's installed.

    backend: 'auto' | 'birefnet' | 'grabcut'.
    """
    if backend in ("auto", "birefnet"):
        try:
            return BiRefNetSegmenter(**kwargs)
        except ImportError:
            if backend == "birefnet":
                raise
    return GrabCutSegmenter(**{k: v for k, v in kwargs.items() if k in ("iterations", "border_ratio")})


def segment_subject(
    image: np.ndarray, object_prompt: str | None = None, backend: str = "auto"
) -> CoarseMask:
    """Segment subject and return a CoarseMask.

    Convenience wrapper used by the CLI / smoke script.
    """
    seg = build_segmenter(backend=backend)
    soft = seg.segment(image, object_prompt=object_prompt)
    return make_bands(soft)


# Keep utility public.
__all__ = [
    "CoarseMask",
    "BiRefNetSegmenter",
    "GrabCutSegmenter",
    "build_segmenter",
    "make_bands",
    "segment_subject",
    "_band_radius",
]
