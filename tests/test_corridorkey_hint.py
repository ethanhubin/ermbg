from __future__ import annotations

import cv2
import numpy as np

from ermbg.corridorkey_hint import (
    build_corridorkey_hint_plan,
    corridorkey_full_frame_prior_value,
    corridorkey_hint_diagnostic_variants,
    corridorkey_hint_variants,
    detect_corridorkey_hint_features,
)
from ermbg.corridorkey_runner import LocalCorridorKeyClient, _mask_to_numpy


def _feature_image() -> np.ndarray:
    bg = np.asarray([0, 37, 252], dtype=np.float32)
    hard = np.asarray([245, 80, 30], dtype=np.float32)
    glow = np.asarray([185, 80, 230], dtype=np.float32)
    image = np.broadcast_to(bg.astype(np.uint8), (128, 128, 3)).copy().astype(np.float32)
    cv2.circle(image, (38, 88), 22, hard.tolist(), -1, cv2.LINE_AA)
    yy, xx = np.mgrid[0:128, 0:128]
    dist = np.sqrt((yy - 42) ** 2 + (xx - 92) ** 2)
    alpha = np.clip((34.0 - dist) / 22.0, 0.0, 0.55).astype(np.float32)
    image = image * (1.0 - alpha[..., None]) + glow.reshape(1, 1, 3) * alpha[..., None]
    return np.clip(image + 0.5, 0, 255).astype(np.uint8)


def test_corridorkey_hint_features_are_position_agnostic() -> None:
    image = _feature_image()
    features = detect_corridorkey_hint_features(image, (0, 37, 252))

    assert features.metadata["pixels"]["subject_support"] > 0
    assert features.metadata["pixels"]["outline_mask"] >= features.metadata["pixels"]["subject_support"]
    assert features.metadata["pixels"]["outline_inner_mask"] > 0
    assert features.metadata["pixels"]["control_outline_mask"] > 0
    assert features.metadata["pixels"]["control_outline_inner_mask"] > 0
    assert features.metadata["pixels"]["control_outline_mask"] < features.metadata["pixels"]["outline_mask"]
    assert features.metadata["pixels"]["hard_subject"] > 0
    assert features.metadata["pixels"]["translucent_candidate"] > 0
    assert features.metadata["pixels"]["internal_transparency_candidate"] > 0
    assert features.metadata["pixels"]["soft_boundary_candidate"] > 0
    assert features.metadata["pixels"]["bbox_plus_2"] > features.metadata["pixels"]["subject_support"]
    # The translucent feature is intentionally away from the image center.
    ys, xs = np.nonzero(features.translucent_candidate)
    assert xs.mean() > 70
    assert ys.mean() < 70


def test_corridorkey_full_frame_prior_is_global_soft_prior() -> None:
    profiles = [
        "auto",
        "corridorkey-character",
        "corridorkey-transparent-button",
        "corridorkey-effect-icon",
        "corridorkey-shaped-icon",
    ]

    for profile in profiles:
        value, kind = corridorkey_full_frame_prior_value(
            execution_profile=profile,
            screen_mode="green",
        )
        assert value == 0.32
        assert kind == "soft_prior"


def test_corridorkey_hint_variants_have_ordered_translucent_support() -> None:
    image = _feature_image()
    plans = {
        variant: build_corridorkey_hint_plan(image, (0, 37, 252), variant=variant)
        for variant in corridorkey_hint_variants()
    }
    mask = plans["feature_balanced"].features.translucent_candidate
    internal = plans["feature_balanced"].features.internal_transparency_candidate
    diagnostic = build_corridorkey_hint_plan(image, (0, 37, 252), variant="full_frame_zero")

    assert plans["current_default_prior"].hint.min() == 0.32
    assert plans["current_default_prior"].hint.max() == 0.32
    assert "full_frame_zero" not in corridorkey_hint_variants()
    assert corridorkey_hint_diagnostic_variants() == ("full_frame_zero",)
    assert diagnostic.hint.min() == 0.0
    assert diagnostic.hint.max() == 0.0
    assert "bbox_plus_2_aggressive" not in corridorkey_hint_variants()
    assert plans["feature_conservative"].hint[mask].mean() > plans["feature_balanced"].hint[mask].mean()
    assert plans["feature_internal_opaque"].hint[internal].mean() > plans["feature_balanced"].hint[internal].mean()
    assert plans["feature_balanced"].hint[mask].mean() > plans["feature_translucent"].hint[mask].mean()
    semi_support = (plans["feature_internal_opaque"].hint > 0.05) & (plans["feature_internal_opaque"].hint < 0.95)
    assert int(semi_support.sum()) > int(internal.sum())
    assert plans["feature_balanced"].metadata["schema"] == "ermbg.corridorkey_hint_plan.v1"
    assert plans["feature_balanced"].hint.max() <= 1.0
    assert plans["feature_balanced"].hint.min() >= 0.0
    assert plans["feature_balanced"].metadata["policy"]["base"] == "soft_control_outline"
    background_values = plans["feature_balanced"].hint[plans["feature_balanced"].features.background]
    assert background_values.mean() < 0.05
    assert float((background_values > 0.25).mean()) < 0.08
    outside_control = ~plans["feature_balanced"].features.control_outline_mask
    assert float((plans["feature_internal_opaque"].hint[outside_control] > 0.25).mean()) < 0.05


def test_full_frame_zero_hint_passes_through_as_black_corridorkey_mask() -> None:
    hint = np.zeros((16, 16), dtype=np.float32)

    mask_tensor, debug = LocalCorridorKeyClient._corridorkey_mask_tensor_from_hint(
        hint,
        screen_color="blue",
        execution_profile="corridorkey-character",
        hint_source="provided_corridorkey_hint_mask",
    )

    mask = _mask_to_numpy(mask_tensor)
    assert mask is not None
    assert float(mask.min()) == 0.0
    assert float(mask.max()) == 0.0
    assert debug["convention"] == "corridorkey_full_frame_zero_hint"


def test_explicit_full_frame_white_hint_is_no_longer_inverted() -> None:
    hint = np.ones((16, 16), dtype=np.float32)

    mask_tensor, debug = LocalCorridorKeyClient._corridorkey_mask_tensor_from_hint(
        hint,
        screen_color="blue",
        execution_profile="corridorkey-character",
        hint_source="provided_corridorkey_hint_mask",
    )

    mask = _mask_to_numpy(mask_tensor)
    assert mask is not None
    assert float(mask.min()) == 1.0
    assert float(mask.max()) == 1.0
    assert debug["convention"] == "corridorkey_full_frame_foreground_hint"
