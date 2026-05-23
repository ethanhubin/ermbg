"""SDXL inpainting probe generator.

Strategy:
- Inpainting mask = NOT dilate(subject_mask). Only background pixels are repainted.
- Subject region is locked by the mask; pipeline still feeds the original image,
  so subject pixels are preserved by the inpainting model.
- Long edge capped at 1024 to fit Mac MPS memory.
"""

from __future__ import annotations

import numpy as np

from .generator import ProbeGenerator
from .prompts import build_prompts


def _resize_long_edge(image: np.ndarray, max_long: int) -> tuple[np.ndarray, tuple[int, int]]:
    import cv2

    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long:
        return image, (h, w)
    scale = max_long / long_edge
    new_h = int(round(h * scale / 8) * 8)
    new_w = int(round(w * scale / 8) * 8)
    new_h = max(64, new_h)
    new_w = max(64, new_w)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, (h, w)


class SDXLInpaintProbeGenerator(ProbeGenerator):
    name = "sdxl_inpaint"

    def __init__(
        self,
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device: str | None = None,
        dtype: str = "fp16",
        max_long_edge: int = 1024,
        steps: int = 30,
        guidance: float = 7.0,
        subject_lock_dilate: int = 6,
    ) -> None:
        try:
            import torch
            from diffusers import StableDiffusionXLInpaintPipeline
        except ImportError as e:
            raise ImportError(
                "SDXLInpaintProbeGenerator requires the 'torch' extra. "
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

        torch_dtype = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }[dtype]
        # MPS dislikes fp16 in some kernels; fall back to fp32 there.
        if device == "mps" and torch_dtype == torch.float16:
            torch_dtype = torch.float32

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, variant="fp16" if dtype == "fp16" else None
        )
        self.pipe.to(device)
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

        self.max_long_edge = max_long_edge
        self.steps = steps
        self.guidance = guidance
        self.subject_lock_dilate = subject_lock_dilate

    def generate(
        self,
        image: np.ndarray,
        subject_mask: np.ndarray,
        background_color: tuple[int, int, int],
        seed: int | None = None,
        object_prompt: str | None = None,
    ) -> np.ndarray:
        import cv2
        from PIL import Image

        torch = self.torch

        # 1. Resize so the long edge fits.
        small, original_hw = _resize_long_edge(image, self.max_long_edge)
        small_mask = cv2.resize(
            subject_mask.astype(np.float32),
            (small.shape[1], small.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2. Build inpaint mask: only paint outside the subject (1 = paint, 0 = keep).
        binary = (small_mask > 0.5).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        locked = cv2.dilate(binary, kernel, iterations=self.subject_lock_dilate)
        inpaint_mask = 255 - locked  # paint everything except the (dilated) subject

        # 3. Initialize the input image with the target background everywhere we want to repaint.
        bg = np.broadcast_to(np.array(background_color, dtype=np.uint8), small.shape).copy()
        init = small.copy()
        init[inpaint_mask > 0] = bg[inpaint_mask > 0]

        # 4. Build prompts.
        positive, negative = build_prompts(background_color, object_prompt=object_prompt)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        result = self.pipe(
            prompt=positive,
            negative_prompt=negative,
            image=Image.fromarray(init),
            mask_image=Image.fromarray(inpaint_mask),
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            strength=0.95,
            generator=generator,
        ).images[0]

        out_small = np.asarray(result.convert("RGB"))

        # 5. Hard-paste the original subject region back to guarantee inner pixels are unchanged.
        keep_mask = cv2.erode(binary, kernel, iterations=max(2, self.subject_lock_dilate - 2)) > 0
        out_small[keep_mask] = small[keep_mask]

        # 6. Resize back to original.
        h, w = original_hw
        if (h, w) != out_small.shape[:2]:
            out_small = cv2.resize(out_small, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return out_small
