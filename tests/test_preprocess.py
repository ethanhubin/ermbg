from __future__ import annotations

import numpy as np

from ermbg.preprocess import (
    NORMALIZE_KNOWN_BACKGROUND,
    REMOVE_CHECKERBOARD,
    analyze_input_preprocess,
    apply_input_preprocess,
    checkerboard_info_from_decision,
    normalize_known_background_preprocess,
)


def _checker_image(size: int = 96, tile: int = 12) -> np.ndarray:
    yy, xx = np.indices((size, size))
    parity = ((xx // tile + yy // tile) & 1).astype(bool)
    image = np.where(
        parity[..., None],
        np.array([254, 254, 254], dtype=np.uint8),
        np.array([243, 243, 243], dtype=np.uint8),
    )
    image[34:62, 28:68] = [120, 60, 210]
    return image.astype(np.uint8)


def test_preprocess_analysis_recommends_checkerboard_removal() -> None:
    image = _checker_image()

    analysis = analyze_input_preprocess(image)

    assert analysis.preprocess_id.startswith("pre_")
    assert [item.id for item in analysis.items] == [REMOVE_CHECKERBOARD]
    assert analysis.items[0].recommended is True
    assert analysis.background_model is not None
    assert analysis.background_model.color == (254, 254, 254)


def test_apply_preprocess_keeps_checkerboard_opt_in() -> None:
    image = _checker_image()

    skipped = apply_input_preprocess(image, selected=[])
    applied = apply_input_preprocess(image, selected=[REMOVE_CHECKERBOARD])

    skipped_info = checkerboard_info_from_decision(skipped.decision)
    applied_info = checkerboard_info_from_decision(applied.decision)
    assert skipped_info["requested"] is False
    assert skipped_info["applied"] is False
    assert applied_info["requested"] is True
    assert applied_info["applied"] is True
    assert skipped.image_srgb[0, 0].tolist() != skipped.image_srgb[0, 12].tolist()
    assert applied.image_srgb[0, 0].tolist() == [254, 254, 254]
    assert applied.image_srgb[0, 12].tolist() == [254, 254, 254]
    assert applied.decision.selected == [REMOVE_CHECKERBOARD]
    assert applied.decision.applied == [REMOVE_CHECKERBOARD]


def test_known_background_normalization_is_preprocess_decision() -> None:
    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((64, 64, 3), bg, dtype=np.uint8)
    image[0, 0] = (0, 199, 0)

    normalized, decision = normalize_known_background_preprocess(
        image,
        tuple(int(c) for c in bg),
        bg_threshold=3.5,
        fg_threshold=24.0,
        adaptive=True,
    )

    assert tuple(int(c) for c in normalized[0, 0]) == (0, 200, 0)
    assert decision.selected == [NORMALIZE_KNOWN_BACKGROUND]
    assert decision.applied == [NORMALIZE_KNOWN_BACKGROUND]
    assert decision.background_model is not None
    assert decision.background_model.color == (0, 200, 0)
    assert decision.metadata["known_background_normalization"]["applied"] is True
