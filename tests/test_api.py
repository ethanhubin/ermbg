"""Tests for the high-level Python API (ermbg.api)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ermbg import MatteResponse, classify_image, matte_image
from ermbg.preprocess import repair_known_background_preprocess

pytestmark = pytest.mark.core


def _solid_green_with_red_subject(h=128, w=128):
    img = np.full((h, w, 3), [0, 200, 0], dtype=np.uint8)
    img[40:90, 40:90] = (220, 30, 30)
    return img


def test_classify_image_from_ndarray():
    img = _solid_green_with_red_subject()
    s = classify_image(img)
    assert s.bg_type == "saturated"


def test_classify_image_from_path(tmp_path):
    img = _solid_green_with_red_subject()
    p = tmp_path / "in.png"
    Image.fromarray(img).save(p)
    s = classify_image(p)
    assert s.bg_type == "saturated"


def test_classify_image_from_pil():
    img = _solid_green_with_red_subject()
    s = classify_image(Image.fromarray(img))
    assert s.bg_type == "saturated"


def test_matte_image_ndarray_returns_response():
    img = _solid_green_with_red_subject()
    r = matte_image(img, backend="pymatting-known-b")
    assert isinstance(r, MatteResponse)
    assert r.rgba.shape == (128, 128, 4)
    assert r.rgba.dtype == np.uint8
    assert r.alpha.shape == (128, 128)
    assert r.foreground_srgb.shape == (128, 128, 3)
    assert r.strategy_name == "pymatting_known_b"
    assert r.output_dir is None


def test_matte_image_rejects_removed_legacy_backend():
    img = _solid_green_with_red_subject()
    with pytest.raises(ValueError, match="removed"):
        matte_image(img, backend="grabcut")


def test_matte_image_pymatting_known_b_backend_skips_segmenter(monkeypatch):
    import ermbg.api as api

    def fail_build_segmenter(**kwargs):
        raise AssertionError("pymatting-known-b should not build a segmenter")

    monkeypatch.setattr(api, "build_segmenter", fail_build_segmenter)

    img = _solid_green_with_red_subject()
    r = matte_image(img, backend="pymatting-known-b")

    assert r.strategy_name == "pymatting_known_b"
    assert r.report["strategy"]["name"] == "pymatting_known_b"
    assert r.background_color == (0, 200, 0)
    assert r.debug["pymatting_known_b"]["pymatting"]["method"] == "cf"
    assert r.alpha[44:86, 44:86].mean() > 0.99


def test_matte_image_pymatting_known_b_accepts_parameters():
    img = _solid_green_with_red_subject()
    r = matte_image(
        img,
        backend="pymatting-known-b",
        pymatting_method="knn",
        pymatting_image_space="sRGB",
        pymatting_bg_source="custom",
        pymatting_bg_color=(0, 200, 0),
        pymatting_bg_threshold=4.5,
        pymatting_fg_threshold=28.0,
        pymatting_boundary_band_px=3,
        pymatting_adapt_bg_threshold=False,
        pymatting_adapt_fg_threshold=False,
        pymatting_adapt_boundary_band=False,
        pymatting_cg_maxiter=1500,
        pymatting_cg_rtol=1e-5,
    )

    params = r.debug["pymatting_known_b"]["parameters"]
    assert r.strategy_name == "pymatting_known_b"
    assert r.background_color == (0, 200, 0)
    assert params["method"] == "knn"
    assert params["image_space"] == "sRGB"
    assert params["bg_source"] == "custom"
    assert params["bg_threshold"] == 4.5
    assert params["fg_threshold"] == 28.0
    assert params["boundary_band_px"] == 3
    assert params["adapt_bg_threshold"] is False
    assert params["adapt_fg_threshold"] is False
    assert params["adapt_boundary_band"] is False
    assert params["cg_maxiter"] == 1500
    assert params["cg_rtol"] == 1e-5


def test_matte_image_pymatting_known_b_runs_preprocess_before_executor():
    img = _solid_green_with_red_subject()
    img[0, 0] = (0, 199, 0)

    r = matte_image(
        img,
        backend="pymatting-known-b",
        pymatting_bg_source="custom",
        pymatting_bg_color=(0, 200, 0),
    )

    preprocess = r.debug["input_preprocess"]["known_background_normalization"]
    assert preprocess["enabled"] is True
    assert "background_normalization" not in r.debug["pymatting_known_b"]


def test_matte_image_pymatting_known_b_consumes_semantic_decision():
    bg = np.array([255, 255, 255], dtype=np.uint8)
    img = np.full((96, 96, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[:96, :96]
    r = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
    img[(r <= 34) & (r >= 17)] = (230, 30, 30)

    subject = matte_image(
        img,
        backend="pymatting-known-b",
        pymatting_bg_source="custom",
        pymatting_bg_color=(255, 255, 255),
        shadow_mode="off",
        semantic_decision={"enclosed_near_bg_policy": "subject"},
    )
    hole = matte_image(
        img,
        backend="pymatting-known-b",
        pymatting_bg_source="custom",
        pymatting_bg_color=(255, 255, 255),
        shadow_mode="off",
        semantic_decision={"enclosed_near_bg_policy": "transparent_hole"},
    )

    inner = r <= 14
    assert subject.alpha[inner].mean() > 0.98
    assert hole.alpha[inner].mean() < 0.02
    assert subject.debug["pymatting_known_b"]["trimap"]["semantic_decision"]["enclosed_near_bg_policy"] == "subject"
    assert hole.debug["pymatting_known_b"]["trimap"]["semantic_decision"]["enclosed_near_bg_policy"] == "transparent_hole"


def test_matte_image_pymatting_known_b_consumes_user_keep_remove_masks():
    bg = np.array([255, 255, 255], dtype=np.uint8)
    img = np.full((96, 96, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[:96, :96]
    r = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
    img[(r <= 34) & (r >= 17)] = (230, 30, 30)
    inner = r <= 14

    kept = matte_image(
        img,
        backend="pymatting-known-b",
        pymatting_bg_source="custom",
        pymatting_bg_color=(255, 255, 255),
        shadow_mode="off",
        semantic_decision={"enclosed_near_bg_policy": "transparent_hole"},
        user_keep_mask=inner.astype(np.float32),
    )
    removed = matte_image(
        img,
        backend="pymatting-known-b",
        pymatting_bg_source="custom",
        pymatting_bg_color=(255, 255, 255),
        shadow_mode="off",
        semantic_decision={"enclosed_near_bg_policy": "subject"},
        user_remove_mask=inner.astype(np.float32),
    )

    assert kept.alpha[inner].mean() > 0.98
    assert removed.alpha[inner].mean() < 0.02
    assert kept.debug["pymatting_known_b"]["trimap"]["user_mask_decision"]["forced_subject_pixels"] == int(inner.sum())
    assert removed.debug["pymatting_known_b"]["trimap"]["user_mask_decision"]["forced_background_pixels"] == int(inner.sum())


def test_matte_image_pymatting_known_b_auto_background_falls_back_when_unstable():
    h = w = 64
    yy = np.linspace(0.0, 24.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 24.0, w, dtype=np.float32)[None, :]
    gray = 154.0 + (xx + yy) * 0.5
    img = np.dstack([gray, gray, gray + 2.0]).astype(np.uint8)
    img[20:46, 22:42] = (220, 40, 30)

    r = matte_image(img, backend="pymatting-known-b", shadow_mode="off")

    background = r.debug["pymatting_known_b"]["background"]
    params = r.debug["pymatting_known_b"]["parameters"]
    assert r.strategy_name == "pymatting_known_b"
    assert background["source"] == "auto_fallback_best_effort"
    assert background["auto_background"]["accepted"] is False
    assert background["auto_background"]["reason"] == "corner/background border is unstable"
    assert params["requested_bg_source"] == "auto"
    assert params["bg_source"] == "custom"
    assert params["adapt_bg_threshold"] is False


def test_matte_image_pymatting_known_b_recovers_neutral_ui_shadow():
    img = np.full((128, 128, 3), [0, 200, 0], dtype=np.uint8)
    img[72:98, 24:104] = [0, 120, 0]
    img[40:82, 28:100] = [240, 30, 30]

    r = matte_image(
        img,
        backend="pymatting-known-b",
        pymatting_bg_source="custom",
        pymatting_bg_color=(0, 200, 0),
        pymatting_fg_threshold=24.0,
        shadow_mode="on",
    )

    assert r.debug["shadow"]["source"] == "pymatting_known_b_shadow_patch"
    assert r.debug["shadow"]["applied"] is True
    assert r.debug["shadow"]["method"] == "unknown_domain_bidirectional_same_background_reconstruction"
    assert r.debug["trimap_u8"][90, 64] == 128
    assert r.debug["shadow"]["objective_shadow"]["mean_abs_error_after_u8"] < 1.0
    assert r.debug["shadow_alpha"][90, 64] > 0.20
    assert r.alpha[90, 64] > 0.20
    assert tuple(r.rgba[90, 64, :3]) == (0, 0, 0)


def test_pymatting_known_b_shadow_patch_reduces_overdark_raw_subject_alpha():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((48, 64, 3), bg, dtype=np.uint8)
    repair_domain = np.zeros((48, 64), dtype=bool)
    repair_domain[16:32, 20:44] = True
    image[repair_domain] = (0, 180, 0)

    subject_alpha = np.zeros((48, 64), dtype=np.float32)
    subject_alpha[repair_domain] = 0.50
    foreground = np.zeros((48, 64, 3), dtype=np.uint8)

    alpha, rgba_rgb, shadow_alpha, _, info = api._pymatting_known_b_unknown_domain_shadow_patch(
        image,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        repair_domain=repair_domain,
    )

    assert info["applied"] is True
    assert info["subject_alpha_reduced_pixels"] == int(repair_domain.sum())
    assert info["shadow_pixels"] == 0
    assert shadow_alpha[repair_domain].max() == 0.0
    assert np.allclose(alpha[repair_domain].mean(), 0.10, atol=0.01)


def test_pymatting_known_b_shadow_patch_prefers_source_shadow_over_screen_colored_foreground():
    import ermbg.api as api

    bg = np.array([3, 203, 6], dtype=np.uint8)
    image = np.full((48, 64, 3), bg, dtype=np.uint8)
    repair_domain = np.zeros((48, 64), dtype=bool)
    repair_domain[16:32, 20:44] = True
    image[repair_domain] = (3, 150, 2)

    subject_alpha = np.zeros((48, 64), dtype=np.float32)
    subject_alpha[repair_domain] = 0.55
    foreground = np.zeros((48, 64, 3), dtype=np.uint8)
    foreground[repair_domain] = (3, 61, 0)

    alpha, rgba_rgb, shadow_alpha, _, info = api._pymatting_known_b_unknown_domain_shadow_patch(
        image,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        repair_domain=repair_domain,
    )

    assert info["applied"] is True
    assert info["shadow_pixels"] == int(repair_domain.sum())
    assert info["subject_alpha_reduced_pixels"] == 0
    assert info["objective_shadow"]["source_shadow_written_pixels"] == int(repair_domain.sum())
    assert float(np.median(shadow_alpha[repair_domain])) > 0.20
    assert np.all(rgba_rgb[repair_domain] == 0)
    assert float(np.median(alpha[repair_domain])) > 0.20
    replay = (
        alpha[..., None] * rgba_rgb.astype(np.float32)
        + (1.0 - alpha[..., None]) * bg.astype(np.float32).reshape(1, 1, 3)
    )
    # A single display-space black alpha cannot fit all quantized sRGB
    # channels exactly, but it must be a close replay of the known-B darkening.
    assert np.abs(replay[repair_domain] - image[repair_domain].astype(np.float32)).mean() < 1.5


def test_pymatting_known_b_shadow_patch_extends_source_shadow_to_connected_screen_residue():
    import ermbg.api as api

    bg = np.array([3, 178, 10], dtype=np.uint8)
    image = np.full((48, 64, 3), bg, dtype=np.uint8)
    repair_domain = np.zeros((48, 64), dtype=bool)
    seed = np.zeros((48, 64), dtype=bool)
    residue = np.zeros((48, 64), dtype=bool)
    seed[16:32, 18:30] = True
    residue[16:32, 30:42] = True
    repair_domain |= seed | residue
    image[seed] = (3, 150, 2)
    image[residue] = (6, 145, 10)

    subject_alpha = np.zeros((48, 64), dtype=np.float32)
    subject_alpha[seed] = 0.55
    subject_alpha[residue] = 0.61
    foreground = np.zeros((48, 64, 3), dtype=np.uint8)
    foreground[seed] = (3, 61, 0)
    foreground[residue] = (8, 115, 10)

    alpha, rgba_rgb, shadow_alpha, _, info = api._pymatting_known_b_unknown_domain_shadow_patch(
        image,
        subject_alpha=subject_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        repair_domain=repair_domain,
    )

    assert info["applied"] is True
    assert info["subject_alpha_reduced_pixels"] == 0
    assert info["objective_shadow"]["source_shadow_seed_written_pixels"] == int(seed.sum())
    assert info["objective_shadow"]["source_shadow_connected_written_pixels"] == int(residue.sum())
    assert info["objective_shadow"]["source_shadow_written_pixels"] == int(repair_domain.sum())
    assert np.all(rgba_rgb[repair_domain] == 0)
    assert float(np.median(alpha[residue])) > 0.15
    assert float(np.median(shadow_alpha[residue])) > 0.15


def test_pymatting_known_b_shadow_patch_arbitrates_subject_edge_aa_before_shadow():
    path = (
        Path(__file__).resolve().parents[1]
        / "samples/corridorkey_semantic/button/button_blue_play_clipped_hard_shadow/blue.png"
    )
    result = matte_image(
        path,
        backend="pymatting-known-b",
        pymatting_bg_source="custom",
        pymatting_bg_color=(1, 94, 246),
        shadow_mode="on",
    )

    objective = result.debug["shadow"]["objective_shadow"]
    edge_aa = objective["subject_edge_aa"]
    assert edge_aa["candidate_pixels"] > 0
    assert edge_aa["written_pixels"] > 0
    assert objective["subject_edge_aa_written_pixels"] == edge_aa["written_pixels"]
    assert objective["mean_abs_error_after_u8"] < objective["mean_abs_error_before_u8"]
    assert objective["source_shadow_written_pixels"] < objective["candidate_pixels"]


def test_corridorkey_shadow_patch_gate_filters_preserved_subject_components():
    import ermbg.api as api

    subject = np.zeros((128, 128), dtype=np.float32)
    shadow_display = np.zeros((128, 128), dtype=np.float32)

    shadow_display[82:100, 32:96] = 0.30
    shadow_display[28:46, 44:92] = 0.36
    subject[28:46, 44:92] = 0.58

    filtered, gate = api._corridorkey_shadow_patch_gate(
        subject,
        shadow_display,
        {"detected": True, "accepted_components": 2},
    )

    assert gate["apply"] is True
    assert gate["missing_in_corridorkey"] is True
    assert gate["kept_components"] == 1
    assert filtered[86:96, 40:88].mean() > 0.20
    assert filtered[32:42, 50:86].max() == 0.0
    rejected = [item for item in gate["component_details"] if not item["apply"]]
    assert rejected
    assert rejected[0]["missing_in_corridorkey"] is False


def test_corridorkey_shadow_patch_gate_replaces_under_reconstructed_hard_shadow():
    import ermbg.api as api

    subject = np.zeros((128, 128), dtype=np.float32)
    shadow_display = np.zeros((128, 128), dtype=np.float32)
    shadow_display[78:96, 32:104] = 0.30
    subject[78:96, 32:104] = 0.12

    filtered, gate = api._corridorkey_shadow_patch_gate(
        subject,
        shadow_display,
        {"detected": True, "accepted_components": 1},
    )

    assert gate["apply"] is True
    assert gate["missing_in_corridorkey"] is True
    assert gate["component_details"][0]["under_reconstructed_shadow"] is True
    assert filtered[82:92, 40:96].mean() > 0.25

    preserved_subject = np.zeros((128, 128), dtype=np.float32)
    preserved_subject[78:96, 32:104] = 0.19
    preserved_filtered, preserved_gate = api._corridorkey_shadow_patch_gate(
        preserved_subject,
        shadow_display,
        {"detected": True, "accepted_components": 1},
    )

    assert preserved_gate["apply"] is False
    assert preserved_gate["missing_in_corridorkey"] is False
    assert preserved_filtered.max() == 0.0


def test_corridorkey_shadow_patch_removes_weak_ck_residue_in_patched_shadow():
    import ermbg.api as api
    from ermbg import io as ermbg_io

    bg = np.array([0, 200, 0], dtype=np.uint8)
    subject = np.zeros((128, 128), dtype=np.float32)
    subject[34:76, 44:92] = 1.0
    shadow = np.zeros((128, 128), dtype=np.float32)
    shadow[80:96, 32:112] = 0.42
    image = ermbg_io.linear_to_srgb_u8(
        (1.0 - shadow[..., None]) * ermbg_io.srgb_to_linear(np.broadcast_to(bg, (128, 128, 3)))
    )
    image[subject > 0] = (230, 40, 40)
    ck_alpha = np.maximum(subject, shadow * 0.20).astype(np.float32)
    protected_edge = np.zeros((128, 128), dtype=bool)
    protected_edge[80:96, 32:40] = True
    ck_alpha[protected_edge] = 0.62
    foreground = np.zeros((128, 128, 3), dtype=np.uint8)
    foreground[..., :] = (230, 40, 40)

    alpha, _, shadow_alpha, _, info = api._corridorkey_shadow_patch(
        image,
        subject_alpha=ck_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        shadow_mode="on",
    )

    patch_region = shadow > 0
    low_shadow_region = patch_region & ~protected_edge
    assert info["applied"] is True
    assert info["patch_gate"]["component_details"][0]["under_reconstructed_shadow"] is True
    assert info["patch_gate"]["corridorkey_shadow_residue_pixels_removed"] > 0
    assert shadow_alpha[protected_edge].max() == 0.0
    assert np.allclose(alpha[protected_edge], ck_alpha[protected_edge])
    assert shadow_alpha[low_shadow_region].mean() > ck_alpha[low_shadow_region].mean()
    assert np.allclose(alpha[low_shadow_region].mean(), shadow_alpha[low_shadow_region].mean(), atol=0.03)


def test_near_subject_shadow_bridge_rejects_outline_scale_expansion():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    image = np.full((96, 128, 3), bg, dtype=np.uint8)
    subject = np.zeros((96, 128), dtype=np.float32)
    foreground = np.zeros((96, 128, 3), dtype=np.uint8)
    shadow = np.zeros((96, 128), dtype=np.float32)

    core = np.zeros((96, 128), dtype=bool)
    core[28:60, 26:102] = True
    shadow_seed = np.zeros((96, 128), dtype=bool)
    shadow_seed[62:64, 26:102] = True
    gap = np.zeros((96, 128), dtype=bool)
    gap[60:62, 26:102] = True

    subject[core] = 1.0
    foreground[core] = (245, 180, 32)
    image[core] = (245, 180, 32)
    shadow[shadow_seed] = 0.30
    image[shadow_seed] = ((1.0 - shadow[shadow_seed, None]) * bg.reshape(1, 3) + 0.5).astype(np.uint8)

    refined, info = api._refine_near_subject_shadow_from_source_pixels(
        shadow,
        subject,
        image,
        tuple(int(c) for c in bg),
        foreground,
    )

    # Mechanism: contact-gap bridging is only a seam repair. If the would-be
    # bridge is larger than the accepted near-subject repair support, it would
    # expand the whole cast shadow along the UI outline, so it must be reported
    # and rejected instead of being written into the shadow alpha.
    assert info["contact_gap_bridge_rejected_as_expansion"] is True
    assert info["contact_gap_bridge_pixels"] == 0
    assert info["rejected_contact_gap_bridge_pixels"] >= int(gap.sum() * 0.8)
    assert refined[gap].max() == 0.0


def test_corridorkey_shadow_patch_uses_source_pixels_as_reprojection_target():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    subject = np.zeros((128, 128), dtype=np.float32)
    subject[34:74, 44:92] = 1.0
    target_display_shadow = np.zeros((128, 128), dtype=np.float32)
    horizontal = np.linspace(0.22, 0.32, 80, dtype=np.float32)
    target_display_shadow[80:96, 32:112] = horizontal[None, :]

    image = (
        (1.0 - target_display_shadow[..., None]) * bg.astype(np.float32).reshape(1, 1, 3)
        + 0.5
    ).astype(np.uint8)
    image[subject > 0] = (230, 40, 40)
    ck_alpha = np.maximum(subject, target_display_shadow * 0.28).astype(np.float32)
    foreground = np.zeros((128, 128, 3), dtype=np.uint8)
    foreground[..., :] = (230, 40, 40)

    _, _, shadow_alpha, _, info = api._corridorkey_shadow_patch(
        image,
        subject_alpha=ck_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        shadow_mode="on",
    )

    shadow_region = target_display_shadow > 0.0
    reprojection = info["patch_gate"]["source_reprojection"]
    assert info["applied"] is True
    assert reprojection["enabled"] is True
    assert reprojection["mean_abs_error_after_u8"] < reprojection["mean_abs_error_before_u8"]
    assert np.mean(np.abs(shadow_alpha[shadow_region] - target_display_shadow[shadow_region])) < 0.012


def test_corridorkey_shadow_patch_preserves_subject_antialiasing_at_contact_edge():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    subject = np.zeros((128, 128), dtype=np.float32)
    subject[34:78, 44:92] = 1.0
    subject[78, 44:92] = 0.20
    shadow = np.zeros((128, 128), dtype=np.float32)
    shadow[78:96, 32:112] = 0.30
    image = (
        (1.0 - shadow[..., None]) * bg.astype(np.float32).reshape(1, 1, 3)
        + 0.5
    ).astype(np.uint8)
    image[subject >= 1.0] = (230, 40, 40)
    image[subject == 0.20] = (184, 168, 0)
    ck_alpha = np.maximum(subject, shadow * 0.28).astype(np.float32)
    foreground = np.zeros((128, 128, 3), dtype=np.uint8)
    foreground[..., :] = (230, 40, 40)

    alpha, rgba_rgb, shadow_alpha, _, info = api._corridorkey_shadow_patch(
        image,
        subject_alpha=ck_alpha,
        subject_foreground_srgb=foreground,
        background_color=tuple(int(c) for c in bg),
        shadow_mode="on",
    )

    contact_edge = np.zeros((128, 128), dtype=bool)
    contact_edge[78, 44:92] = True
    exterior_shadow = np.zeros((128, 128), dtype=bool)
    exterior_shadow[88:94, 44:92] = True
    assert info["applied"] is True
    assert info["patch_gate"]["corridorkey_subject_edge_pixels_preserved"] >= int(contact_edge.sum())
    assert np.all(alpha[contact_edge] > shadow_alpha[contact_edge])
    assert rgba_rgb[contact_edge, 0].mean() > 40.0
    assert np.allclose(alpha[exterior_shadow].mean(), shadow_alpha[exterior_shadow].mean(), atol=0.03)


def test_corridorkey_shadow_patch_reprojects_near_subject_region():
    import ermbg.api as api

    bg = np.array([0, 200, 0], dtype=np.uint8)
    subject = np.zeros((64, 96), dtype=np.float32)
    subject[18:34, 24:72] = 1.0
    subject[34, 24:72] = 0.18
    shadow = np.zeros((64, 96), dtype=np.float32)
    shadow[36:44, 20:76] = 0.32
    source_shadow = np.zeros((64, 96), dtype=np.float32)
    source_shadow[34:44, 20:76] = 0.32
    foreground = np.zeros((64, 96, 3), dtype=np.uint8)
    foreground[..., :] = (230, 40, 40)
    image = (
        subject[..., None] * foreground.astype(np.float32)
        + (1.0 - subject[..., None]) * (1.0 - source_shadow[..., None]) * bg.astype(np.float32).reshape(1, 1, 3)
        + 0.5
    ).astype(np.uint8)

    repaired, info = api._refine_near_subject_shadow_from_source_pixels(
        shadow,
        subject,
        image,
        tuple(int(c) for c in bg),
        foreground,
    )

    gap = np.zeros_like(subject, dtype=bool)
    gap[35, 24:72] = True
    subject_edge = np.zeros_like(subject, dtype=bool)
    subject_edge[34, 24:72] = True
    assert info["repair_pixels"] > 0
    assert info["source_added_pixels"] >= int(gap.sum())
    assert info["source_reproject_pixels"] > 0
    assert info["mean_abs_error_after_u8"] < info["mean_abs_error_before_u8"]
    assert repaired[gap].mean() > 0.20
    assert repaired[subject_edge].mean() > 0.20


def test_corridorkey_shadow_patch_gate_rejects_broad_vertical_background_wash():
    import ermbg.api as api

    subject = np.zeros((128, 128), dtype=np.float32)
    shadow_display = np.zeros((128, 128), dtype=np.float32)
    shadow_display[8:120, 18:110] = 0.065

    filtered, gate = api._corridorkey_shadow_patch_gate(
        subject,
        shadow_display,
        {"detected": True, "accepted_components": 1},
    )

    assert gate["apply"] is False
    assert gate["missing_in_corridorkey"] is False
    assert gate["reason"] == "shadow candidates rejected as broad background wash or vertical subject residue"
    assert gate["rejected_missing_shape_components"] == 1
    assert gate["component_details"][0]["broad_low_contrast_wash"] is True
    assert gate["component_details"][0]["shadow_like_shape"] is False
    assert filtered.max() == 0.0


def test_matte_image_writes_files_when_output_dir_given(tmp_path):
    img = _solid_green_with_red_subject()
    p = tmp_path / "in.png"
    Image.fromarray(img).save(p)
    out = tmp_path / "out"
    r = matte_image(p, backend="pymatting-known-b", output_dir=out)
    assert r.output_dir == out
    assert (out / "in_rgba.png").exists()
    assert (out / "in_alpha.png").exists()
    assert (out / "in_shadow.png").exists()
    assert (out / "in_foreground.png").exists()
    assert (out / "in.report.json").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "ermbg.run.v1"
    assert manifest["outputs"]["rgba"] == "in_rgba.png"
    assert manifest["outputs"]["alpha"] == "in_alpha.png"
    assert manifest["request"]["backend"] == "pymatting-known-b"
    assert manifest["report"] == "in.report.json"


def test_matte_image_qa_adds_metrics_to_report(tmp_path):
    img = _solid_green_with_red_subject()
    out = tmp_path / "out"
    r = matte_image(img, backend="pymatting-known-b", output_dir=out, qa=True)
    assert "qa" in r.report
    assert "edge_halo_score_mean" in r.report["qa"]
    assert (out / "matte_qa").exists()


def test_matte_image_rejects_bad_dtype():
    bad = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        matte_image(bad)
