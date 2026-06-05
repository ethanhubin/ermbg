from __future__ import annotations

import json

from ermbg.pipeline_contracts import (
    AmbiguityRegion,
    AnalyzeResult,
    BackgroundModel,
    ExecutionRequest,
    PreprocessAnalysis,
    PreprocessDecision,
    PreprocessItem,
    SemanticCandidate,
    SemanticDecision,
    UserMaskDecision,
    semantic_manifest_summary,
)


def _background_model() -> BackgroundModel:
    return BackgroundModel(
        color=(255, 255, 255),
        source="corner_probe",
        confidence=0.98,
        metadata={"drift_luma_max": 2.5},
    )


def _preprocess_decision() -> PreprocessDecision:
    return PreprocessDecision(
        selected=["background_repair"],
        applied=["background_repair"],
        metadata={"checkerboard": {"recommended": False}},
        background_model=_background_model(),
    )


def _analyze_result() -> AnalyzeResult:
    return AnalyzeResult(
        status="needs_decision",
        analysis_id="analysis_abc",
        preprocess=_preprocess_decision(),
        route={
            "algorithm": "pymatting_known_b",
            "asset_kind": "known_bg_graphic",
            "parameter_profile": "edge_cleanup",
            "execution_profile": "pymatting-known-bg",
        },
        default_candidate_id="protect_near_bg_subject",
        ambiguity_regions=[
            AmbiguityRegion(
                id="ambiguous_enclosed_bg_0",
                type="enclosed_near_background",
                bbox_xyxy=[39, 141, 250, 214],
                area_px=4971,
                mask_ref="mask_0",
                evidence={"touches_exterior_background": False},
                ambiguity={"reason": "single_image_semantic_ambiguity"},
            )
        ],
        candidates=[
            SemanticCandidate(
                id="protect_near_bg_subject",
                label="Keep internal light material",
                intent="Treat enclosed near-background pixels as subject-owned material.",
                default=True,
                confidence=0.72,
                risk_level="medium",
                decision={"enclosed_near_bg_policy": "subject"},
                regions=["ambiguous_enclosed_bg_0"],
                preview={"overlay_mask": "mask_0"},
                reasons=["region is enclosed by foreground support"],
            ),
            SemanticCandidate(
                id="cut_enclosed_holes",
                label="Transparent internal holes",
                default=False,
                confidence=0.48,
                risk_level="medium",
                decision={"enclosed_near_bg_policy": "transparent_hole"},
                regions=["ambiguous_enclosed_bg_0"],
            ),
        ],
        preview_assets={
            "candidate:protect_near_bg_subject:overlay": {
                "kind": "overlay",
                "candidate_id": "protect_near_bg_subject",
                "media_type": "image/png",
                "encoding": "data_url",
                "data_url": "data:image/png;base64,abc",
            }
        },
    )


def test_preprocess_analysis_json_roundtrip() -> None:
    analysis = PreprocessAnalysis(
        preprocess_id="pre_abc",
        items=[
            PreprocessItem(
                id="background_repair",
                label="Background repair",
                recommended=True,
                enabled_by_default=True,
                reason="detected_checkerboard_background",
                preview_assets={"overlay_png": "/api/preprocess-preview/pre_abc/background_repair.png"},
            )
        ],
        background_model=_background_model(),
        debug={"probe": "stage1"},
    )

    payload = json.loads(json.dumps(analysis.to_dict()))

    assert PreprocessAnalysis.from_dict(payload) == analysis


def test_analyze_result_json_roundtrip() -> None:
    analysis = _analyze_result()

    payload = json.loads(json.dumps(analysis.to_dict()))

    restored = AnalyzeResult.from_dict(payload)
    assert restored == analysis
    assert restored.status == "needs_decision"
    assert [candidate.id for candidate in restored.candidates] == [
        "protect_near_bg_subject",
        "cut_enclosed_holes",
    ]


def test_execution_request_json_roundtrip() -> None:
    request = ExecutionRequest(
        analysis_id="analysis_abc",
        preprocess=_preprocess_decision(),
        route=_analyze_result().route,
        selected_candidate_id="protect_near_bg_subject",
        semantic_decision=SemanticDecision(
            candidate_id="protect_near_bg_subject",
            decision={"enclosed_near_bg_policy": "subject"},
            source="web_user",
            confidence=0.72,
        ),
        user_mask=UserMaskDecision(
            keep_mask="mask_keep_png",
            remove_mask=None,
            source="web_user_brush",
            brush_version=1,
            summary={"keep_pixels": 64, "remove_pixels": 0},
        ),
        metadata={"request_id": "exec_abc"},
    )

    payload = json.loads(json.dumps(request.to_dict()))

    assert ExecutionRequest.from_dict(payload) == request


def test_semantic_manifest_summary_carries_stage1_fields() -> None:
    analysis = _analyze_result()
    semantic = SemanticDecision(
        candidate_id="protect_near_bg_subject",
        decision={"enclosed_near_bg_policy": "subject"},
        source="default",
    )
    mask = UserMaskDecision(keep_mask="keep.png", summary={"keep_pixels": 12})

    summary = semantic_manifest_summary(
        analyze=analysis,
        semantic_decision=semantic,
        user_mask=mask,
    )

    assert summary["preprocess"]["selected"] == ["background_repair"]
    assert summary["preprocess"]["applied"] == ["background_repair"]
    assert summary["semantic"]["analysis_id"] == "analysis_abc"
    assert summary["semantic"]["analysis_status"] == "needs_decision"
    assert summary["semantic"]["default_candidate_id"] == "protect_near_bg_subject"
    assert summary["semantic"]["selected_candidate_id"] == "protect_near_bg_subject"
    assert summary["semantic"]["ambiguity_types"] == ["enclosed_near_background"]
    assert summary["semantic"]["candidate_previews"]["protect_near_bg_subject"] == {"overlay_mask": "mask_0"}
    assert summary["semantic"]["preview_assets"]["candidate:protect_near_bg_subject:overlay"]["kind"] == "overlay"
    assert "data_url" not in summary["semantic"]["preview_assets"]["candidate:protect_near_bg_subject:overlay"]
    assert summary["semantic"]["user_mask_used"] is True

