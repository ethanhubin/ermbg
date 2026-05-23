"""Prompt templates for solid-color background generation.

Plan section 7: emphasize 'replace background only', preserve subject, no shadow,
no gradient, no halo, no relighting.

Phase 1.2: also exposes a 'green-screen' convention for upstream image
generators. Picked because:
  (1) green is far from skin / fur / wood / orange — chroma cap is clean;
  (2) high G channel — chroma cap math is well-conditioned;
  (3) decades of pro tooling assume green/blue screens.
"""

from __future__ import annotations


GREEN_SCREEN_RGB: tuple[int, int, int] = (0, 200, 0)
GREEN_SCREEN_PROMPT: str = (
    "subject placed on a perfectly flat saturated green studio screen background "
    "(approximately RGB 0/200/0), uniform color, no shadow, no gradient, "
    "no vignette, no reflection, no glow, no environment, no props"
)


_COLOR_NAMES: dict[tuple[int, int, int], str] = {
    (250, 250, 250): "pure clean white",
    (8, 8, 8): "deep solid black",
    (0, 200, 220): "vivid saturated cyan",
    (220, 30, 180): "vivid saturated magenta",
    (0, 200, 60): "vivid saturated green",
    GREEN_SCREEN_RGB: "saturated green-screen green",
}


def color_phrase(color: tuple[int, int, int]) -> str:
    """Closest named phrase for a probe color, or RGB triplet fallback."""
    key = tuple(int(c) for c in color)
    if key in _COLOR_NAMES:
        return _COLOR_NAMES[key]
    return f"plain solid color rgb({key[0]},{key[1]},{key[2]})"


POSITIVE_TEMPLATE = (
    "the same subject on a {color} studio background, "
    "background is a perfectly flat solid color, "
    "uniform lighting, photo, sharp, high resolution, "
    "subject completely unchanged, identical pose and silhouette, "
    "no shadow, no gradient, no vignette, no reflection, no glow"
)

NEGATIVE_TEMPLATE = (
    "redrawn subject, changed silhouette, modified hair, modified fur, "
    "shadow, drop shadow, gradient, vignette, reflection, halo, glow, "
    "ground, floor, environment, props, text, watermark, blurry, low-res"
)


def build_prompts(color: tuple[int, int, int], object_prompt: str | None = None) -> tuple[str, str]:
    """Returns (positive, negative)."""
    pos = POSITIVE_TEMPLATE.format(color=color_phrase(color))
    if object_prompt:
        pos = f"{object_prompt}, {pos}"
    return pos, NEGATIVE_TEMPLATE
