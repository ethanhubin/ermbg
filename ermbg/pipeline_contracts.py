"""Data contracts for the staged ERMBG matting pipeline.

These structures describe the Preprocess -> Analyze -> Decide -> Execute
boundary. They are intentionally behavior-free in Stage 1 so existing Web and
Direct Worker execution keeps its current semantics while the new contract
becomes testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AnalysisStatus = Literal["ready", "needs_decision", "unsupported"]
RiskLevel = Literal["low", "medium", "high"]


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _str_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _int_bbox(value: Any) -> list[int]:
    items = _list(value)
    if len(items) != 4:
        return [0, 0, 0, 0]
    return [int(item) for item in items]


@dataclass(frozen=True)
class BackgroundModel:
    """Observable background evidence shared by Analyze and Execute."""

    color: tuple[int, int, int] | None = None
    source: str = "unknown"
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": list(self.color) if self.color is not None else None,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BackgroundModel:
        color_value = payload.get("color")
        color = tuple(int(c) for c in color_value) if isinstance(color_value, list) and len(color_value) == 3 else None
        confidence = payload.get("confidence")
        return cls(
            color=color,  # type: ignore[arg-type]
            source=str(payload.get("source") or "unknown"),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(frozen=True)
class PreprocessItem:
    id: str
    label: str
    recommended: bool = False
    enabled_by_default: bool = False
    reason: str | None = None
    preview_assets: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "recommended": self.recommended,
            "enabled_by_default": self.enabled_by_default,
            "reason": self.reason,
            "preview_assets": self.preview_assets,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreprocessItem:
        return cls(
            id=str(payload.get("id") or ""),
            label=str(payload.get("label") or ""),
            recommended=bool(payload.get("recommended", False)),
            enabled_by_default=bool(payload.get("enabled_by_default", False)),
            reason=str(payload["reason"]) if payload.get("reason") is not None else None,
            preview_assets=_dict(payload.get("preview_assets")),
            metadata=_dict(payload.get("metadata")),
        )


@dataclass(frozen=True)
class PreprocessAnalysis:
    preprocess_id: str
    items: list[PreprocessItem] = field(default_factory=list)
    background_model: BackgroundModel | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess_id": self.preprocess_id,
            "items": [item.to_dict() for item in self.items],
            "background_model": self.background_model.to_dict() if self.background_model else None,
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreprocessAnalysis:
        background_payload = payload.get("background_model")
        return cls(
            preprocess_id=str(payload.get("preprocess_id") or ""),
            items=[PreprocessItem.from_dict(item) for item in _list(payload.get("items")) if isinstance(item, dict)],
            background_model=BackgroundModel.from_dict(background_payload) if isinstance(background_payload, dict) else None,
            debug=_dict(payload.get("debug")),
        )


@dataclass(frozen=True)
class PreprocessDecision:
    selected: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    background_model: BackgroundModel | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "applied": self.applied,
            "metadata": self.metadata,
            "background_model": self.background_model.to_dict() if self.background_model else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PreprocessDecision:
        background_payload = payload.get("background_model")
        return cls(
            selected=_str_list(payload.get("selected")),
            applied=_str_list(payload.get("applied")),
            metadata=_dict(payload.get("metadata")),
            background_model=BackgroundModel.from_dict(background_payload) if isinstance(background_payload, dict) else None,
        )


@dataclass(frozen=True)
class AmbiguityRegion:
    id: str
    type: str
    bbox_xyxy: list[int]
    area_px: int
    mask_ref: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    ambiguity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "bbox_xyxy": self.bbox_xyxy,
            "area_px": self.area_px,
            "mask_ref": self.mask_ref,
            "evidence": self.evidence,
            "ambiguity": self.ambiguity,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AmbiguityRegion:
        return cls(
            id=str(payload.get("id") or ""),
            type=str(payload.get("type") or ""),
            bbox_xyxy=_int_bbox(payload.get("bbox_xyxy")),
            area_px=int(payload.get("area_px") or 0),
            mask_ref=str(payload["mask_ref"]) if payload.get("mask_ref") is not None else None,
            evidence=_dict(payload.get("evidence")),
            ambiguity=_dict(payload.get("ambiguity")),
        )


@dataclass(frozen=True)
class SemanticCandidate:
    id: str
    label: str
    decision: dict[str, Any]
    default: bool = False
    confidence: float | None = None
    intent: str | None = None
    risk_level: RiskLevel = "low"
    regions: list[str] = field(default_factory=list)
    preview: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "intent": self.intent,
            "default": self.default,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "regions": self.regions,
            "preview": self.preview,
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SemanticCandidate:
        confidence = payload.get("confidence")
        risk = str(payload.get("risk_level") or "low")
        if risk not in {"low", "medium", "high"}:
            risk = "low"
        return cls(
            id=str(payload.get("id") or ""),
            label=str(payload.get("label") or ""),
            intent=str(payload["intent"]) if payload.get("intent") is not None else None,
            default=bool(payload.get("default", False)),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            risk_level=risk,  # type: ignore[arg-type]
            decision=_dict(payload.get("decision")),
            regions=_str_list(payload.get("regions")),
            preview=_dict(payload.get("preview")),
            reasons=_str_list(payload.get("reasons")),
        )


@dataclass(frozen=True)
class AnalyzeResult:
    status: AnalysisStatus
    route: dict[str, Any]
    ambiguity_regions: list[AmbiguityRegion] = field(default_factory=list)
    candidates: list[SemanticCandidate] = field(default_factory=list)
    default_candidate_id: str | None = None
    analysis_id: str | None = None
    preprocess: PreprocessDecision | None = None
    preview_assets: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "analysis_id": self.analysis_id,
            "preprocess": self.preprocess.to_dict() if self.preprocess else None,
            "default_candidate_id": self.default_candidate_id,
            "route": self.route,
            "ambiguity_regions": [region.to_dict() for region in self.ambiguity_regions],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "preview_assets": self.preview_assets,
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnalyzeResult:
        status = str(payload.get("status") or "unsupported")
        if status not in {"ready", "needs_decision", "unsupported"}:
            status = "unsupported"
        preprocess_payload = payload.get("preprocess")
        return cls(
            status=status,  # type: ignore[arg-type]
            analysis_id=str(payload["analysis_id"]) if payload.get("analysis_id") is not None else None,
            preprocess=PreprocessDecision.from_dict(preprocess_payload) if isinstance(preprocess_payload, dict) else None,
            default_candidate_id=str(payload["default_candidate_id"]) if payload.get("default_candidate_id") is not None else None,
            route=_dict(payload.get("route")),
            ambiguity_regions=[
                AmbiguityRegion.from_dict(region)
                for region in _list(payload.get("ambiguity_regions"))
                if isinstance(region, dict)
            ],
            candidates=[
                SemanticCandidate.from_dict(candidate)
                for candidate in _list(payload.get("candidates"))
                if isinstance(candidate, dict)
            ],
            preview_assets=_dict(payload.get("preview_assets")),
            debug=_dict(payload.get("debug")),
        )


@dataclass(frozen=True)
class SemanticDecision:
    candidate_id: str
    decision: dict[str, Any]
    source: str = "auto_default"
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SemanticDecision:
        confidence = payload.get("confidence")
        return cls(
            candidate_id=str(payload.get("candidate_id") or ""),
            decision=_dict(payload.get("decision")),
            source=str(payload.get("source") or "auto_default"),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        )


@dataclass(frozen=True)
class UserMaskDecision:
    keep_mask: str | None = None
    remove_mask: str | None = None
    unknown_mask: str | None = None
    source: str = "none"
    brush_version: int = 1
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep_mask": self.keep_mask,
            "remove_mask": self.remove_mask,
            "unknown_mask": self.unknown_mask,
            "source": self.source,
            "brush_version": self.brush_version,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> UserMaskDecision:
        return cls(
            keep_mask=str(payload["keep_mask"]) if payload.get("keep_mask") is not None else None,
            remove_mask=str(payload["remove_mask"]) if payload.get("remove_mask") is not None else None,
            unknown_mask=str(payload["unknown_mask"]) if payload.get("unknown_mask") is not None else None,
            source=str(payload.get("source") or "none"),
            brush_version=int(payload.get("brush_version") or 1),
            summary=_dict(payload.get("summary")),
        )

    def is_used(self) -> bool:
        return bool(self.keep_mask or self.remove_mask or self.unknown_mask)


@dataclass(frozen=True)
class ExecutionRequest:
    preprocess: PreprocessDecision
    route: dict[str, Any]
    semantic_decision: SemanticDecision
    analysis_id: str | None = None
    selected_candidate_id: str | None = None
    user_mask: UserMaskDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "preprocess": self.preprocess.to_dict(),
            "route": self.route,
            "selected_candidate_id": self.selected_candidate_id,
            "semantic_decision": self.semantic_decision.to_dict(),
            "user_mask": self.user_mask.to_dict() if self.user_mask else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionRequest:
        preprocess_payload = payload.get("preprocess")
        semantic_payload = payload.get("semantic_decision")
        user_mask_payload = payload.get("user_mask")
        return cls(
            analysis_id=str(payload["analysis_id"]) if payload.get("analysis_id") is not None else None,
            preprocess=PreprocessDecision.from_dict(preprocess_payload if isinstance(preprocess_payload, dict) else {}),
            route=_dict(payload.get("route")),
            selected_candidate_id=str(payload["selected_candidate_id"]) if payload.get("selected_candidate_id") is not None else None,
            semantic_decision=SemanticDecision.from_dict(semantic_payload if isinstance(semantic_payload, dict) else {}),
            user_mask=UserMaskDecision.from_dict(user_mask_payload) if isinstance(user_mask_payload, dict) else None,
            metadata=_dict(payload.get("metadata")),
        )


def semantic_manifest_summary(
    *,
    preprocess: PreprocessDecision | None = None,
    analyze: AnalyzeResult | None = None,
    selected_candidate_id: str | None = None,
    semantic_decision: SemanticDecision | None = None,
    user_mask: UserMaskDecision | None = None,
) -> dict[str, Any]:
    """Return the manifest fragment planned for staged execution outputs."""

    preprocess_payload = preprocess or (analyze.preprocess if analyze else None)
    ambiguity_types = []
    if analyze is not None:
        ambiguity_types = sorted({region.type for region in analyze.ambiguity_regions if region.type})
    return {
        "preprocess": {
            "selected": preprocess_payload.selected if preprocess_payload else [],
            "applied": preprocess_payload.applied if preprocess_payload else [],
            "metadata": preprocess_payload.metadata if preprocess_payload else {},
        },
        "semantic": {
            "analysis_id": analyze.analysis_id if analyze else None,
            "analysis_status": analyze.status if analyze else None,
            "default_candidate_id": analyze.default_candidate_id if analyze else None,
            "selected_candidate_id": selected_candidate_id
            or (semantic_decision.candidate_id if semantic_decision else None),
            "semantic_decision": semantic_decision.to_dict() if semantic_decision else None,
            "ambiguity_types": ambiguity_types,
            "user_mask_used": bool(user_mask and user_mask.is_used()),
            "user_mask_summary": user_mask.summary if user_mask else {},
        },
    }


__all__ = [
    "AmbiguityRegion",
    "AnalyzeResult",
    "BackgroundModel",
    "ExecutionRequest",
    "PreprocessAnalysis",
    "PreprocessDecision",
    "PreprocessItem",
    "SemanticCandidate",
    "SemanticDecision",
    "UserMaskDecision",
    "semantic_manifest_summary",
]
