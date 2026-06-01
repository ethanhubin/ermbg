"""ERMBG — clean transparent matting via known-background routing.

Top-level API::

    from ermbg import matte_image, classify_image, classify_image_route
    r = matte_image("input.png", output_dir="out/", qa=True)
    r.rgba                      # H×W×4 numpy uint8
    r.strategy_name             # 'saturated_bg' | 'white_bg' | ... | 'rgba_passthrough'
    r.report['qa']['edge_halo_score_mean']

Or fast preview (no matting net):

    s = classify_image("input.png")
    print(s.bg_type, s.notes)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import MatteResponse, classify_image, classify_image_route, matte_image

__all__ = ["matte_image", "classify_image", "classify_image_route", "MatteResponse", "__version__"]
