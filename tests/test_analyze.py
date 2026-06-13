from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import cv2
import pytest
from PIL import Image

from ermbg.analyze import analyze_candidates
from ermbg.preprocess import BACKGROUND_REPAIR, apply_input_preprocess
from ermbg.router import RouteCandidate, RouteDecision


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _same_key_blue_button() -> np.ndarray:
    image = np.full((128, 128, 3), (1, 95, 248), dtype=np.uint8)
    cv2.circle(image, (64, 64), 39, (112, 160, 248), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 42, (40, 88, 208), 2, cv2.LINE_AA)
    return image


def _translucent_badge_like_image() -> np.ndarray:
    h, w = 160, 280
    image = np.full((h, w, 3), 253, dtype=np.uint8)
    body = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(body, (140, 82), (116, 56), 0, 0, 360, 255, -1, cv2.LINE_AA)
    alpha = body.astype(np.float32) / 255.0 * 0.92
    bg = np.asarray([253, 253, 253], dtype=np.float32)
    green = np.asarray([170, 235, 15], dtype=np.float32)
    rgb = bg * (1.0 - alpha[..., None]) + green * alpha[..., None]

    band = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(band, (102, 48), (48, 12), -15, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(band, (178, 42), (13, 9), 0, 0, 360, 255, -1, cv2.LINE_AA)
    band_alpha = band.astype(np.float32) / 255.0
    near_white = np.asarray([252, 252, 238], dtype=np.float32)
    rgb = rgb * (1.0 - band_alpha[..., None]) + near_white * band_alpha[..., None]
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def _white_bg_colored_badge_with_shadow() -> np.ndarray:
    h = w = 160
    image = np.full((h, w, 3), 253, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = cy = 80
    dist = np.sqrt((xx - (cx + 6)) ** 2 + (yy - (cy + 8)) ** 2)
    strength = np.clip((58.0 - dist) / 18.0, 0.0, 1.0) * 0.55
    shadow = strength > 0.04
    image[shadow] = np.clip(
        image[shadow].astype(np.float32) * (1.0 - strength[shadow, None]),
        0,
        255,
    ).astype(np.uint8)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring = (r2 <= 45**2) & (r2 >= 38**2)
    body = r2 < 38**2
    image[ring] = (245, 112, 4)
    image[body] = (20, 155, 38)
    return image


def _outlined_cartoon_white_marking() -> np.ndarray:
    h = w = 128
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.circle(image, (64, 64), 42, (36, 24, 18), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 38, (245, 130, 20), -1, cv2.LINE_AA)
    cv2.ellipse(image, (55, 66), (18, 12), -20, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    return image


def _same_outline_cartoon_hole() -> np.ndarray:
    h = w = 128
    bg = (0, 200, 0)
    image = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.circle(image, (64, 64), 42, (25, 25, 20), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 38, (230, 180, 30), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 16, (25, 25, 20), -1, cv2.LINE_AA)
    cv2.circle(image, (64, 64), 12, bg, -1, cv2.LINE_AA)
    return image


def _decode_preview_luma(data_url: str) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)


def _flood_exterior(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    work = mask.astype(np.uint8).copy()
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for x in range(w):
        if work[0, x]:
            cv2.floodFill(work, flood, (x, 0), 2)
        if work[h - 1, x]:
            cv2.floodFill(work, flood, (x, h - 1), 2)
    for y in range(h):
        if work[y, 0]:
            cv2.floodFill(work, flood, (0, y), 2)
        if work[y, w - 1]:
            cv2.floodFill(work, flood, (w - 1, y), 2)
    return work == 2


def _non_boundary_unknown_component_count(unknown: np.ndarray, sure_bg: np.ndarray) -> int:
    exterior_bg = _flood_exterior(sure_bg)
    exterior_contact = cv2.dilate(
        exterior_bg.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(unknown.astype(np.uint8), 8)
    count = 0
    for label in range(1, n_labels):
        comp = labels == label
        if bool((comp & exterior_contact).any()):
            continue
        count += 1
    return count


class _FakeRouteDecision:
    def __init__(self, *, parameter_profile: str = "translucent_button") -> None:
        self.parameter_profile = parameter_profile

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": "pymatting_known_b",
            "route": "pymatting_known_b",
            "backend": "pymatting_known_b",
            "asset_kind": "known_bg_graphic",
            "parameter_profile": self.parameter_profile,
            "execution_profile": "pymatting-known-bg",
            "confidence": 0.8,
            "reasons": ["test_known_b_route"],
            "params": {
                "execution_profile": "pymatting-known-bg",
                "pymatting_bg_color": (253, 253, 253),
                "pymatting_bg_threshold": 3.5,
                "pymatting_fg_threshold": 24.0,
                "pymatting_adapt_bg_threshold": False,
                "pymatting_adapt_fg_threshold": True,
                "pymatting_adapt_boundary_band": True,
            },
            "analysis": {"corridorkey_analysis": {"parameter_profile": self.parameter_profile}},
        }

    def to_route_decision(self) -> RouteDecision:
        payload = self.to_dict()
        return RouteDecision(
            route=str(payload["route"]),
            asset_kind=str(payload["asset_kind"]),
            backend=str(payload["backend"]),
            params=dict(payload["params"]),  # type: ignore[arg-type]
            confidence=float(payload["confidence"]),
            reasons=list(payload["reasons"]),  # type: ignore[arg-type]
            analysis=dict(payload["analysis"]),  # type: ignore[arg-type]
        )


def test_analyze_enclosed_near_background_returns_semantic_candidates() -> None:
    result = analyze_candidates(_ring_image())
    repeated = analyze_candidates(_ring_image())

    assert result.status == "needs_decision"
    assert result.analysis_id is not None
    assert result.analysis_id == repeated.analysis_id
    assert result.route["algorithm"] == "pymatting_known_b"
    assert result.ambiguity_regions[0].type == "enclosed_near_background"
    assert result.ambiguity_regions[0].mask_ref == f"{result.analysis_id}:region_mask:ambiguous_enclosed_bg_0"
    assert result.ambiguity_regions[0].evidence["touches_exterior_background"] is False
    assert [candidate.id for candidate in result.candidates] == [
        "auto_recommended_holes",
        "protect_near_bg_subject",
        "cut_enclosed_holes",
    ]
    assert result.default_candidate_id == "auto_recommended_holes"
    assert result.candidates[0].decision["candidate_strategy"] == "auto_region_recommendation"
    assert result.candidates[0].decision["enclosed_near_bg_region_policies"] == {
        "ambiguous_enclosed_bg_0": "transparent_hole",
    }
    assert result.candidates[0].decision["enclosed_near_bg_policy"] == "transparent_hole"
    assert result.candidates[1].decision == {
        "enclosed_near_bg_policy": "subject",
        "enclosed_near_bg_region_policies": {"ambiguous_enclosed_bg_0": "subject"},
    }
    assert result.candidates[2].decision == {
        "enclosed_near_bg_policy": "transparent_hole",
        "enclosed_near_bg_region_policies": {"ambiguous_enclosed_bg_0": "transparent_hole"},
    }
    assert result.candidates[1].preview["assets"]["overlay"] == "candidate:protect_near_bg_subject:overlay"
    assert result.preview_assets["schema"] == "ermbg.analysis_preview_assets.v1"
    assert result.preview_assets["region_mask:ambiguous_enclosed_bg_0"]["data_url"].startswith("data:image/png;base64,")
    assert result.preview_assets["candidate:protect_near_bg_subject:overlay"]["data_url"].startswith("data:image/png;base64,")
    assert result.preview_assets["candidate:protect_near_bg_subject:trimap"]["data_url"].startswith("data:image/png;base64,")
    assert result.preview_assets["candidate:protect_near_bg_subject:trimap"]["execution_role"] == "pymatting_explicit_trimap"
    trimap_meta = result.preview_assets["candidate:protect_near_bg_subject:trimap"]["metadata"]
    assert trimap_meta["source"] == "known_b_bg_seed_outline_candidate_trimap"
    assert trimap_meta["candidate_assembly"]["bg_seed_outline"]["accepted"] is True
    assert trimap_meta["states"]["sure_bg"]["pixels"] > 0
    assert trimap_meta["states"]["unknown"]["pixels"] > 0
    assert trimap_meta["states"]["sure_fg"]["pixels"] > 0
    assert "hint" not in result.candidates[1].preview["assets"]
    assert "candidate:protect_near_bg_subject:hint" not in result.preview_assets


def test_analyze_outlined_cartoon_white_marking_defaults_to_subject() -> None:
    result = analyze_candidates(_outlined_cartoon_white_marking())

    assert result.status == "needs_decision"
    assert result.default_candidate_id == "auto_recommended_holes"
    assert {region.type for region in result.ambiguity_regions} == {"enclosed_near_background"}
    region = result.ambiguity_regions[0]
    assert region.evidence["subject_outline_confidence"] > 0.8
    assert region.evidence["hole_outline_confidence"] < 0.1
    assert result.candidates[0].decision["enclosed_near_bg_policy"] == "subject"
    assert result.candidates[0].confidence > result.candidates[2].confidence
    trimap_meta = result.preview_assets[result.candidates[0].preview["assets"]["trimap"]]["metadata"]
    assert trimap_meta["flat_opaque_internal_unknown"]["applied"] is True
    assert trimap_meta["flat_opaque_internal_unknown"]["released_pixels"] > 0
    assert trimap_meta["flat_opaque_internal_unknown"]["policy"] == "topological_subject_internal_unknown_to_sure_fg"
    assert trimap_meta["flat_opaque_internal_unknown"]["exterior_connected_unknown_pixels"] > 0
    assert trimap_meta["flat_opaque_internal_unknown"]["internal_unknown_pixels"] == trimap_meta["flat_opaque_internal_unknown"]["released_pixels"]


def test_analyze_same_outline_internal_hole_defaults_to_transparent() -> None:
    result = analyze_candidates(_same_outline_cartoon_hole())

    assert result.status == "needs_decision"
    assert result.default_candidate_id == "use_cut_hole_0"
    assert {region.type for region in result.ambiguity_regions} == {"enclosed_near_background"}
    region = result.ambiguity_regions[0]
    assert region.evidence["subject_outline_confidence"] > 0.8
    assert region.evidence["hole_outline_confidence"] > 0.65
    assert result.candidates[0].decision["enclosed_near_bg_policy"] == "transparent_hole"
    assert result.candidates[0].confidence > result.candidates[1].confidence
    trimap_meta = result.preview_assets[result.candidates[0].preview["assets"]["trimap"]]["metadata"]
    assert trimap_meta["flat_opaque_internal_unknown"]["applied"] is False


def test_analyze_known_b_connected_shadow_is_resolved_by_default_trimap() -> None:
    result = analyze_candidates(_white_bg_colored_badge_with_shadow())

    assert result.status == "ready"
    assert result.default_candidate_id == "auto_default"
    assert [candidate.id for candidate in result.candidates] == ["auto_default"]
    assert result.candidates[0].decision == {"policy": "auto_default"}
    assert all(region.type != "connected_known_b_shadow_ownership" for region in result.ambiguity_regions)

    default_meta = result.preview_assets[result.candidates[0].preview["assets"]["trimap"]]["metadata"]
    assembly = default_meta["candidate_assembly"]
    assert assembly["neutral_shadow_conflict_unknown_pixels"] > 0
    assert assembly["shadow_background"]["unknown_ownership_pixels"] == assembly["shadow_background"]["pixels"]
    assert assembly["bg_seed_outline"]["shadow_inward_unknown_pixels"] > 0


def test_analyze_dark_subject_on_white_is_not_connected_shadow_candidate() -> None:
    image = np.full((128, 192, 3), 253, dtype=np.uint8)
    cv2.putText(image, "A", (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (40, 40, 40), 8, cv2.LINE_AA)

    result = analyze_candidates(image)

    assert all(region.type != "connected_known_b_shadow_ownership" for region in result.ambiguity_regions)


def test_analyze_tight_background_match_allows_smaller_hole_components() -> None:
    image = np.full((512, 512, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (120, 120), (392, 392), (90, 130, 40), -1)
    cv2.rectangle(image, (180, 180), (188, 188), (255, 255, 255), -1)
    cv2.rectangle(image, (225, 170), (236, 179), (255, 255, 255), -1)
    cv2.rectangle(image, (310, 300), (322, 308), (255, 255, 255), -1)

    result = analyze_candidates(image)

    old_fixed_min_area = int(round(float(image.shape[0] * image.shape[1]) * 0.0005))
    small_regions = [
        region
        for region in result.ambiguity_regions
        if region.type == "enclosed_near_background" and region.area_px < old_fixed_min_area
    ]
    assert len(small_regions) >= 3
    assert result.status == "needs_decision"
    assert result.candidates[-1].decision["enclosed_near_bg_policy"] == "transparent_hole"
    assert set(result.candidates[-1].decision["enclosed_near_bg_region_policies"].values()) == {
        "transparent_hole",
    }
    for region in small_regions:
        evidence = region.evidence
        assert evidence["area_gate_source"] == "background_distance_confidence"
        assert evidence["bg_distance_p95"] <= 0.1
        assert evidence["min_area_px_effective"] < old_fixed_min_area


def test_analyze_hard_button_without_shadow_is_ready() -> None:
    result = analyze_candidates(_solid_green_button())

    assert result.status == "ready"
    assert result.analysis_id is not None
    assert result.default_candidate_id == "auto_default"
    assert result.ambiguity_regions == []
    assert [candidate.id for candidate in result.candidates] == ["auto_default"]
    assert result.candidates[0].preview["assets"]["overlay"] == "candidate:auto_default:overlay"
    assert result.preview_assets["candidate:auto_default:overlay"]["data_url"].startswith("data:image/png;base64,")


def test_analyze_same_key_button_returns_opaque_and_corridorkey_candidates() -> None:
    result = analyze_candidates(_same_key_blue_button())

    assert result.status == "needs_decision"
    assert result.ambiguity_regions == []
    assert result.default_route_candidate_id == "route_pymatting_known_b_same_key_opaque"
    assert [candidate.id for candidate in result.candidates] == [
        "route_pymatting_known_b_same_key_opaque__opaque_outline",
        "route_corridorkey_same_key_translucent__semi_transparent_corridorkey",
    ]
    assert result.candidates[0].decision["pymatting_trimap_mode"] == "same_key_opaque_body_outline"
    assert result.candidates[1].decision["policy"] == "same_key_semi_transparent_corridorkey"
    assert result.candidates[1].decision["corridorkey_hint_value"] == 0.32


def test_analyze_same_key_icon_closed_outline_keeps_internal_texture_out_of_unknown() -> None:
    image = np.asarray(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/icon/icon_icon_a03_hard_boundary_weak_contrast/green.png"
        ).convert("RGB"),
        dtype=np.uint8,
    )

    result = analyze_candidates(image)

    assert result.status == "needs_decision"
    assert result.default_route_candidate_id == "route_pymatting_known_b_same_key_opaque"
    assert result.candidates[0].decision["pymatting_trimap_mode"] == "same_key_opaque_body_outline"
    trimap_ref = result.candidates[0].preview["assets"]["trimap"]
    trimap = _decode_preview_luma(result.preview_assets[trimap_ref]["data_url"])
    unknown = (trimap >= 64) & (trimap <= 191)
    subject_domain = trimap > 32
    assert float(np.count_nonzero(unknown)) / float(np.count_nonzero(subject_domain)) < 0.16
    distance_to_exterior = cv2.distanceTransform(subject_domain.astype(np.uint8), cv2.DIST_L2, 3)
    assert np.count_nonzero(unknown & (distance_to_exterior >= 6.0)) == 0


def test_analyze_b001_no_shadow_button_has_stable_boundary_trimap() -> None:
    image = np.asarray(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_no_shadow/green.png"
        ).convert("RGB"),
        dtype=np.uint8,
    )

    result = analyze_candidates(image)

    assert result.status == "ready"
    assert result.default_candidate_id == "auto_default"
    assert {region.type for region in result.ambiguity_regions} == set()
    trimap = _decode_preview_luma(result.preview_assets["candidate:auto_default:trimap"]["data_url"])
    unknown = (trimap >= 64) & (trimap <= 191)
    sure_bg = trimap < 64
    assert _non_boundary_unknown_component_count(unknown, sure_bg) == 0


def test_analyze_corridorkey_screen_material_returns_semantic_candidates() -> None:
    image = np.asarray(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/character/character_char_a06_pale_hair_translucent_sleeves_white_glow_blue/blue.png"
        ).convert("RGB"),
        dtype=np.uint8,
    )

    result = analyze_candidates(image)

    assert result.status == "needs_decision"
    assert result.route["algorithm"] == "corridorkey"
    assert result.default_route_candidate_id == "route_corridorkey"
    assert {candidate["algorithm"] for candidate in result.route_candidates} == {
        "pymatting_known_b",
        "corridorkey",
    }
    # CorridorKey exposes only full-frame constant hint strengths.
    assert [
        region.type
        for region in result.ambiguity_regions
        if region.id.startswith("route_corridorkey__")
    ] == []
    corridorkey_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.route_candidate_id == "route_corridorkey"
    ]
    assert len(corridorkey_candidates) == len(result.candidates)
    assert [candidate.id for candidate in corridorkey_candidates] == [
        "route_corridorkey__corridorkey_hint_000",
        "route_corridorkey__corridorkey_hint_016",
        "route_corridorkey__auto_default",
        "route_corridorkey__corridorkey_hint_050",
        "route_corridorkey__corridorkey_hint_070",
    ]
    default = [candidate for candidate in corridorkey_candidates if candidate.default][0]
    assert default.id == "route_corridorkey__auto_default"
    assert default.decision["policy"] == "corridorkey_constant_hint"
    assert default.decision["corridorkey_hint_value"] == 0.32
    hint = _decode_preview_luma(result.preview_assets[default.preview["assets"]["hint"]]["data_url"])
    assert int(hint.min()) == int(hint.max())


def test_analyze_corridorkey_glass_portal_exposes_core_and_gradient_candidates() -> None:
    image = np.asarray(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/icon/icon_icon_d11_glass_portal_blue/blue.png"
        ).convert("RGB"),
        dtype=np.uint8,
    )

    result = analyze_candidates(image)

    assert result.status == "needs_decision"
    assert result.route["algorithm"] == "corridorkey"
    assert result.route["execution_profile"] == "corridorkey-character"
    assert result.default_candidate_id == "route_corridorkey__auto_default"
    # CorridorKey emits no feature-hint regions; candidates are pure constant
    # hint strengths.
    corridorkey_regions = [
        region
        for region in result.ambiguity_regions
        if region.id.startswith("route_corridorkey__")
    ]
    assert corridorkey_regions == []
    corridorkey_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.route_candidate_id == "route_corridorkey"
    ]
    assert len(corridorkey_candidates) == len(result.candidates)
    assert [candidate.id for candidate in corridorkey_candidates] == [
        "route_corridorkey__corridorkey_hint_000",
        "route_corridorkey__corridorkey_hint_016",
        "route_corridorkey__auto_default",
        "route_corridorkey__corridorkey_hint_050",
        "route_corridorkey__corridorkey_hint_070",
    ]
    default = [candidate for candidate in corridorkey_candidates if candidate.default][0]
    assert default.id == "route_corridorkey__auto_default"
    assert default.decision == {
        "policy": "corridorkey_constant_hint",
        "corridorkey_hint_value": 0.32,
        "review_region_types": [],
    }
    hint = _decode_preview_luma(result.preview_assets[default.preview["assets"]["hint"]]["data_url"])
    assert int(hint.min()) == int(hint.max())


def test_analyze_button_shadow_is_resolved_by_bg_seed_outline_without_candidates() -> None:
    image = np.asarray(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_hard_lite_shadow/green.png"
        ).convert("RGB"),
        dtype=np.uint8,
    )

    result = analyze_candidates(image)

    assert result.status == "ready"
    assert result.route["algorithm"] == "pymatting_known_b"
    assert result.ambiguity_regions == []
    assert result.default_candidate_id == "auto_default"
    assert [candidate.id for candidate in result.candidates] == ["auto_default"]
    assert result.candidates[0].default is True
    assert result.preview_assets["candidate:auto_default:trimap"]["execution_role"] == "pymatting_explicit_trimap"
    trimap_meta = result.preview_assets["candidate:auto_default:trimap"]["metadata"]
    assert trimap_meta["source"] == "known_b_bg_seed_outline_candidate_trimap"
    assert trimap_meta["candidate_assembly"]["bg_seed_outline"]["accepted"] is True
    assert trimap_meta["states"]["unknown"]["pixels"] > 0


def test_analyze_hard_heavy_button_shadow_stays_single_default_candidate() -> None:
    image = np.asarray(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/button/button_green_yellow_a_outlined_hard_heavy_shadow/green.png"
        ).convert("RGB"),
        dtype=np.uint8,
    )

    result = analyze_candidates(image)

    assert result.status == "ready"
    assert result.ambiguity_regions == []
    assert result.default_candidate_id == "auto_default"
    assert [candidate.id for candidate in result.candidates] == ["auto_default"]
    trimap_meta = result.preview_assets["candidate:auto_default:trimap"]["metadata"]
    assert trimap_meta["source"] == "known_b_bg_seed_outline_candidate_trimap"
    assert trimap_meta["states"]["unknown"]["pixels"] > 0


def test_analyze_b056_uses_hole_candidates_without_shadow_branching() -> None:
    image = np.asarray(
        Image.open(
            PROJECT_ROOT
            / "samples/corridorkey_semantic/button/button_hole_ornate_plate_blue/blue.png"
        ).convert("RGB"),
        dtype=np.uint8,
    )

    result = analyze_candidates(image)

    assert result.status == "needs_decision"
    assert {region.type for region in result.ambiguity_regions} == {"enclosed_near_background"}
    assert result.default_candidate_id == "use_cut_all_holes"
    assert [candidate.id for candidate in result.candidates] == [
        "use_cut_all_holes",
        "use_keep_all_holes",
    ]
    cut_meta = result.preview_assets["candidate:use_cut_all_holes:trimap"]["metadata"]
    keep_meta = result.preview_assets["candidate:use_keep_all_holes:trimap"]["metadata"]
    assert cut_meta["source"] == "known_b_bg_seed_outline_candidate_trimap"
    assert cut_meta["region_policy_application"] == "bg_seed_outline_region_overlay_applied"
    assert cut_meta["semantic_forced_bg_pixels"] > 0
    assert keep_meta["semantic_forced_fg_pixels"] > cut_meta["semantic_forced_bg_pixels"]
    assert cut_meta["semantic_hole_unknown_pixels"] > 0
    assert cut_meta["states"]["sure_bg"]["pixels"] > keep_meta["states"]["sure_bg"]["pixels"]
    assert cut_meta["states"]["sure_fg"]["pixels"] < keep_meta["states"]["sure_fg"]["pixels"]


def test_analyze_button_hole_near_shadow_returns_hole_only_candidates() -> None:
    image = np.full((96, 160, 3), (0, 200, 0), dtype=np.uint8)
    cv2.rectangle(image, (30, 25), (120, 65), (230, 30, 20), -1)
    cv2.rectangle(image, (65, 38), (85, 52), (0, 200, 0), -1)
    cv2.rectangle(image, (35, 68), (125, 76), (0, 170, 0), -1)

    result = analyze_candidates(image)

    assert result.status == "needs_decision"
    assert {region.type for region in result.ambiguity_regions} == {"enclosed_near_background"}
    assert result.default_candidate_id == "use_cut_hole_0"
    assert [candidate.id for candidate in result.candidates] == [
        "use_cut_hole_0",
        "use_keep_hole_0",
    ]
    assert result.candidates[0].default is True
    assert result.candidates[0].decision["enclosed_near_bg_region_policies"] == {
        "ambiguous_enclosed_bg_0": "transparent_hole",
    }
    assert result.candidates[0].decision["enclosed_near_bg_policy"] == "transparent_hole"
    assert result.candidates[0].decision["candidate_rank"] == 0
    assert result.candidates[1].decision["enclosed_near_bg_region_policies"] == {
        "ambiguous_enclosed_bg_0": "subject",
    }
    assert result.candidates[1].decision["enclosed_near_bg_policy"] == "subject"


def test_analyze_hole_policy_only_changes_hole_region_trimap() -> None:
    image = np.full((96, 160, 3), (0, 200, 0), dtype=np.uint8)
    cv2.rectangle(image, (30, 25), (120, 65), (230, 30, 20), -1)
    cv2.rectangle(image, (65, 38), (85, 52), (0, 200, 0), -1)
    cv2.rectangle(image, (35, 68), (125, 76), (0, 170, 0), -1)

    result = analyze_candidates(image)
    cut_hole = _decode_preview_luma(
        result.preview_assets["candidate:use_cut_hole_0:trimap"]["data_url"]
    )
    keep_hole = _decode_preview_luma(
        result.preview_assets["candidate:use_keep_hole_0:trimap"]["data_url"]
    )
    hole_mask = _decode_preview_luma(result.preview_assets["region_mask:ambiguous_enclosed_bg_0"]["data_url"]) > 0
    cut_meta = result.preview_assets["candidate:use_cut_hole_0:trimap"]["metadata"]

    assert np.count_nonzero(cut_hole[hole_mask] == 0) > 0
    assert np.count_nonzero(cut_hole[hole_mask] == 128) > 0
    assert np.all(keep_hole[hole_mask] == 255)
    hole_outer_edge = cv2.dilate(
        hole_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(bool) & ~hole_mask
    assert np.count_nonzero(cut_hole[hole_outer_edge] == 128) > 0
    assert cut_meta["semantic_hole_unknown_pixels"] > 0
    assert cut_meta["semantic_hole_unknown"]["components"][0]["release_px"] >= 1
    changed = cut_hole != keep_hole
    assert bool(changed.any())
    hole_boundary_domain = cv2.dilate(
        hole_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ).astype(bool)
    assert np.count_nonzero(changed & ~hole_boundary_domain) == 0


def test_analyze_translucent_known_b_uses_wide_near_background_preview_region(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_decision = _FakeRouteDecision().to_route_decision()
    monkeypatch.setattr(
        "ermbg.analyze.build_route_candidates",
        lambda *args, **kwargs: [
            RouteCandidate(
                id="route_pymatting_known_b",
                decision=fake_decision,
                default=True,
            )
        ],
    )

    result = analyze_candidates(_translucent_badge_like_image())

    assert result.status == "needs_decision"
    assert result.route["parameter_profile"] == "translucent_button"
    assert result.ambiguity_regions[0].evidence["evidence_mode"] == "translucent_known_b_material_band"
    assert result.ambiguity_regions[0].evidence["bg_distance_max"] == 20.0
    assert result.ambiguity_regions[0].area_px >= 300
    assert result.ambiguity_regions[0].bbox_xyxy[2] - result.ambiguity_regions[0].bbox_xyxy[0] >= 70
    assert result.candidates[1].preview["assets"] == {
        "overlay": "candidate:protect_near_bg_subject:overlay",
        "trimap": "candidate:protect_near_bg_subject:trimap",
    }
    trimap_meta = result.preview_assets["candidate:protect_near_bg_subject:trimap"]["metadata"]
    assert trimap_meta["source"] == "known_b_bg_seed_outline_candidate_trimap"
    assert trimap_meta["candidate_assembly"]["bg_seed_outline"]["accepted"] is True
    assert set(trimap_meta["states"]) == {"sure_bg", "unknown", "sure_fg"}
    assert all(state["pixels"] > 0 for state in trimap_meta["states"].values())
    trimap = _decode_preview_luma(result.preview_assets["candidate:protect_near_bg_subject:trimap"]["data_url"])
    unknown = (trimap >= 64) & (trimap <= 191)
    sure_bg = trimap < 64
    assert _non_boundary_unknown_component_count(unknown, sure_bg) <= 1
    assert trimap_meta["candidate_assembly"]["unknown_pixels"] > 0


def test_analyze_known_b_preview_trimap_uses_preprocessed_background_for_drifted_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _solid_green_button()
    decision = RouteDecision(
        route="pymatting_known_b",
        asset_kind="button",
        backend="pymatting_known_b",
        params={
            "execution_profile": "pymatting-hard-button",
            "pymatting_bg_source": "custom",
            "pymatting_bg_color": (0, 192, 0),
            "pymatting_bg_threshold": 3.5,
            "pymatting_fg_threshold": 24.0,
            "pymatting_boundary_band_px": 2,
            "pymatting_adapt_bg_threshold": False,
            "pymatting_adapt_fg_threshold": True,
            "pymatting_adapt_boundary_band": True,
            "pymatting_trimap_mode": "standard",
            "pymatting_unknown_grow_px": 0,
        },
        confidence=1.0,
        reasons=["test_quantized_screen_background"],
        analysis={
            "corridorkey_analysis": {
                "screen_mode": "green",
                "background_color": [0, 200, 0],
                "parameter_profile": "opaque_hard_ui_soft_shadow",
            }
        },
    )
    monkeypatch.setattr(
        "ermbg.analyze.build_route_candidates",
        lambda *args, **kwargs: [RouteCandidate(id="route_pymatting_known_b", decision=decision, default=True)],
    )

    result = analyze_candidates(image, fallback_background_color=(0, 200, 0))

    trimap_ref = result.candidates[0].preview["assets"]["trimap"]
    trimap_meta = result.preview_assets[trimap_ref]["metadata"]
    assert trimap_meta["states"]["sure_bg"]["pixels"] > 0
    assert trimap_meta["states"]["unknown"]["pixels"] > 0
    assert trimap_meta["states"]["sure_fg"]["pixels"] > 0
    assert trimap_meta["preprocess"]["known_background_normalization"]["applied"] is True
    trimap = _decode_preview_luma(result.preview_assets[trimap_ref]["data_url"])
    assert np.count_nonzero(trimap == 0) > 0


def test_analyze_consumes_preprocess_decision() -> None:
    image = np.full((96, 96, 3), 254, dtype=np.uint8)
    image[34:62, 28:68] = [120, 60, 210]
    preprocessed = apply_input_preprocess(image, selected=[BACKGROUND_REPAIR])

    result = analyze_candidates(preprocessed.image_srgb, preprocess=preprocessed.decision)

    assert result.preprocess is not None
    assert result.preprocess.selected == [BACKGROUND_REPAIR]
