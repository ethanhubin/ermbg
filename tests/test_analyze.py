from __future__ import annotations

import numpy as np

from ermbg.analyze import analyze_candidates
from ermbg.preprocess import NORMALIZE_KNOWN_BACKGROUND, REMOVE_CHECKERBOARD, apply_input_preprocess


def _ring_image() -> np.ndarray:
    h = w = 64
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    image[(r <= 22) & (r >= 9)] = (230, 0, 0)
    return image


def _solid_green_button() -> np.ndarray:
    image = np.full((64, 96, 3), (0, 200, 0), dtype=np.uint8)
    image[22:42, 28:68] = (230, 30, 20)
    return image


def test_analyze_enclosed_near_background_returns_semantic_candidates() -> None:
    result = analyze_candidates(_ring_image())

    assert result.status == "needs_decision"
    assert result.route["algorithm"] == "pymatting_known_b"
    assert result.ambiguity_regions[0].type == "enclosed_near_background"
    assert result.ambiguity_regions[0].evidence["touches_exterior_background"] is False
    assert [candidate.id for candidate in result.candidates] == [
        "auto_default",
        "protect_near_bg_subject",
        "cut_enclosed_holes",
    ]
    assert result.candidates[1].decision == {"enclosed_near_bg_policy": "subject"}
    assert result.candidates[2].decision == {"enclosed_near_bg_policy": "transparent_hole"}


def test_analyze_no_ambiguity_is_ready() -> None:
    result = analyze_candidates(_solid_green_button())

    assert result.status == "ready"
    assert result.default_candidate_id == "auto_default"
    assert [candidate.id for candidate in result.candidates] == ["auto_default"]
    assert result.ambiguity_regions == []


def test_analyze_consumes_preprocess_decision() -> None:
    image = np.full((96, 96, 3), 254, dtype=np.uint8)
    image[34:62, 28:68] = [120, 60, 210]
    preprocessed = apply_input_preprocess(image, selected=[REMOVE_CHECKERBOARD])

    result = analyze_candidates(preprocessed.image_srgb, preprocess=preprocessed.decision)

    assert result.preprocess is not None
    assert result.preprocess.selected == [REMOVE_CHECKERBOARD, NORMALIZE_KNOWN_BACKGROUND]
