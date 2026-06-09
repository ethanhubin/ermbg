from __future__ import annotations

import numpy as np

from ermbg.corridorkey_hint import (
    corridorkey_full_frame_prior_value,
    corridorkey_hint_strengths,
)
from ermbg.corridorkey_runner import LocalCorridorKeyClient, _mask_to_numpy


def test_corridorkey_default_hint_is_global_soft_prior() -> None:
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

    value, kind = corridorkey_full_frame_prior_value(
        execution_profile="corridorkey-transparent-button",
        screen_mode="blue",
    )
    assert value == 0.32
    assert kind == "soft_prior"


def test_corridorkey_hint_strengths_are_constant_candidate_order() -> None:
    assert corridorkey_hint_strengths() == (0.0, 0.16, 0.32, 0.5, 0.7)


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


def test_explicit_full_frame_white_hint_is_not_inverted() -> None:
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


def test_blue_screen_without_screen_color_support_uses_channel_swap(monkeypatch) -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 2] = 200
    captured: dict[str, np.ndarray] = {}

    class FakeNode:
        def run(
            self,
            image,
            mask,
            gamma_space,
            despill_strength,
            refiner_strength,
            auto_despeckle,
            despeckle_size,
            unique_id=None,
        ):
            captured["image"] = image.detach().cpu().numpy()
            return image, mask, image, mask

    monkeypatch.setattr(LocalCorridorKeyClient, "_get_loaded_node", classmethod(lambda cls: FakeNode()))

    result = LocalCorridorKeyClient(backend_label="test", prompt_id="test").matte(
        rgb,
        background_color=(0, 0, 200),
        hint_alpha=np.zeros((2, 2), dtype=np.float32),
        screen_color="blue",
    )

    settings = result.debug["settings"]
    assert settings["requested_screen_color"] == "blue"
    assert settings["screen_color"] == "green"
    assert settings["screen_color_supported"] is False
    assert settings["blue_screen_adaptation"]["method"] == "channel_swap_gb"
    assert float(captured["image"][0, 0, 0, 1]) > 0.75
    assert float(captured["image"][0, 0, 0, 2]) == 0.0
