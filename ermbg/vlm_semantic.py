"""VLM semantic priors for known-background matting.

The VLM never writes alpha or foreground colors. It only classifies local CV
candidate regions as semantic constraints. Pixel strength and compositing stay
in deterministic code.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

from . import io
from .comfy import DEFAULT_COMFY_URL
from .colorspace import oklab_distance, srgb_to_oklab
from .despill import has_dominant_screen_channel
from .planner import RiskRegion
from .vlm_openai import (
    OpenAIVLMPlannerError,
    extract_openai_output_text,
    resolve_openai_api_key,
)
from .vlm_payload import REGION_COLORS

_RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass
class MattingSemanticPrior:
    """Semantic masks used as constraints before deterministic matting steps."""

    subject_material_mask: np.ndarray | None = None
    subject_mask: np.ndarray | None = None
    shadow_search_mask: np.ndarray | None = None
    shadow_ownership_mask: np.ndarray | None = None
    shadow_allowed: bool = True
    source: str = ""
    regions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "shadow_allowed": bool(self.shadow_allowed),
            "has_subject_material_mask": self.subject_material_mask is not None,
            "subject_material_pixels": int(self.subject_material_mask.sum()) if self.subject_material_mask is not None else 0,
            "has_subject_mask": self.subject_mask is not None,
            "subject_pixels": int(self.subject_mask.sum()) if self.subject_mask is not None else 0,
            "has_shadow_search_mask": self.shadow_search_mask is not None,
            "shadow_search_pixels": int(self.shadow_search_mask.sum()) if self.shadow_search_mask is not None else 0,
            "has_shadow_ownership_mask": self.shadow_ownership_mask is not None,
            "shadow_ownership_pixels": int(self.shadow_ownership_mask.sum()) if self.shadow_ownership_mask is not None else 0,
            "regions": list(self.regions),
        }


@dataclass(frozen=True)
class VLMSemanticAttachment:
    id: str
    purpose: str
    width: int
    height: int
    data_base64: str
    mime_type: str = "image/png"
    region_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLMSemanticRequest:
    image: dict[str, Any]
    regions: list[dict[str, Any]]
    attachments: list[VLMSemanticAttachment]
    response_schema: dict[str, Any]
    system_prompt: str


def extract_subject_material_candidate_regions(
    image_srgb: np.ndarray,
    subject_alpha: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    shadow_alpha: np.ndarray | None = None,
    alpha_min: float = 0.70,
    bg_distance_max: float = 45.0,
    min_area_ratio: float = 0.0007,
    max_regions: int = 12,
) -> list[RiskRegion]:
    """Find green-screen-like subject-color candidates for VLM interpretation.

    These are not accepted as foreground automatically. The VLM decides which
    candidate components are subject material rather than spill/background.
    """
    if image_srgb.dtype != np.uint8:
        raise ValueError("image_srgb must be uint8")
    if image_srgb.shape[:2] != subject_alpha.shape:
        raise ValueError("image_srgb and subject_alpha must share HxW")
    if shadow_alpha is not None and shadow_alpha.shape != subject_alpha.shape:
        raise ValueError("shadow_alpha must have shape HxW")

    h, w = subject_alpha.shape
    B_srgb = np.asarray(background_color, dtype=np.uint8).reshape(1, 1, 3)
    B_lin = io.srgb_to_linear(B_srgb)[0, 0]
    dominant = has_dominant_screen_channel(B_lin)
    if dominant is None:
        return []

    C_lin = io.srgb_to_linear(image_srgb)
    other = [idx for idx in range(3) if idx != dominant]
    dominant_like = C_lin[..., dominant] >= np.maximum(C_lin[..., other[0]], C_lin[..., other[1]])

    lab = srgb_to_oklab(image_srgb)
    bg_lab = srgb_to_oklab(B_srgb).reshape(3)
    bg_distance = oklab_distance(lab, bg_lab).astype(np.float32)

    candidate = (
        (subject_alpha.astype(np.float32) >= float(alpha_min))
        & dominant_like
        & (bg_distance <= float(bg_distance_max))
    )
    if shadow_alpha is not None:
        candidate &= shadow_alpha <= 0.02

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    min_area = max(32.0, float(min_area_ratio) * float(h * w))
    components: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label_idx
        components.append(
            (
                area,
                comp,
                {
                    "area": area,
                    "alpha_min": alpha_min,
                    "bg_distance_max": bg_distance_max,
                    "bg_distance_mean": float(bg_distance[comp].mean()),
                    "bg_distance_p95": float(np.percentile(bg_distance[comp], 95.0)),
                },
            )
        )

    components.sort(key=lambda item: item[0], reverse=True)
    regions: list[RiskRegion] = []
    for idx, (_, comp, evidence) in enumerate(components[:max_regions]):
        regions.append(
            RiskRegion(
                id=f"subject_material_{idx}",
                kind="subject_material_candidate",
                mask=comp,
                confidence=1.0,
                evidence=evidence,
            )
        )
    return regions


def extract_shadow_candidate_regions(
    image_srgb: np.ndarray,
    subject_alpha: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    *,
    shadow_alpha: np.ndarray | None = None,
    min_alpha: float = 0.015,
    min_area_ratio: float = 0.00045,
    max_regions: int = 8,
) -> list[RiskRegion]:
    """Convert measured scalar-darkening support into VLM-owned shadow candidates.

    These regions are still CV evidence, not accepted alpha. The VLM can say
    whether a candidate is semantically an owned contact/cast shadow; local code
    later re-measures its opacity from known-background darkening.
    """
    if image_srgb.dtype != np.uint8:
        raise ValueError("image_srgb must be uint8")
    if image_srgb.shape[:2] != subject_alpha.shape:
        raise ValueError("image_srgb and subject_alpha must share HxW")
    if shadow_alpha is None:
        from .shadow import estimate_shadow_alpha

        shadow_alpha, _ = estimate_shadow_alpha(image_srgb, subject_alpha, background_color)
    if shadow_alpha.shape != subject_alpha.shape:
        raise ValueError("shadow_alpha must have shape HxW")

    h, w = subject_alpha.shape
    support = (shadow_alpha.astype(np.float32) >= float(min_alpha)) & (subject_alpha.astype(np.float32) < 0.85)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
    min_area = max(16.0, float(min_area_ratio) * float(h * w))
    components: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label_idx
        values = shadow_alpha[comp].astype(np.float32)
        components.append(
            (
                area,
                comp,
                {
                    "area": area,
                    "shadow_alpha_mean": float(values.mean()),
                    "shadow_alpha_p95": float(np.percentile(values, 95.0)),
                    "shadow_alpha_max": float(values.max()),
                    "min_alpha": min_alpha,
                },
            )
        )

    components.sort(key=lambda item: item[0], reverse=True)
    regions: list[RiskRegion] = []
    for idx, (_, comp, evidence) in enumerate(components[:max_regions]):
        regions.append(
            RiskRegion(
                id=f"shadow_candidate_{idx}",
                kind="owned_shadow_candidate",
                mask=comp,
                confidence=1.0,
                evidence=evidence,
            )
        )
    return regions


def parse_semantic_prior_payload(
    payload: dict[str, Any],
    regions: list[RiskRegion],
    shape: tuple[int, int],
    *,
    confidence_min: float = 0.55,
    source: str = "vlm",
) -> MattingSemanticPrior:
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list):
        raise OpenAIVLMPlannerError("semantic prior payload must contain a regions list")

    by_id = {region.id: region for region in regions}
    subject_material = np.zeros(shape, dtype=np.float32)
    subject = np.zeros(shape, dtype=np.float32)
    shadow_search = np.zeros(shape, dtype=np.float32)
    shadow_ownership = np.zeros(shape, dtype=np.float32)
    seen_subject_material = seen_subject = seen_shadow_search = seen_shadow_ownership = False
    accepted: list[dict[str, Any]] = []
    saw_shadow_role = False
    has_shadow_candidate = any(
        region.kind in {"owned_shadow_candidate", "shadow_search_region"} for region in regions
    )

    for raw in raw_regions:
        if not isinstance(raw, dict):
            raise OpenAIVLMPlannerError("semantic prior region entries must be objects")
        region_id = raw.get("region_id")
        role = raw.get("role")
        confidence = float(raw.get("confidence", 0.0))
        if not isinstance(region_id, str) or not isinstance(role, str):
            raise OpenAIVLMPlannerError("semantic prior region requires region_id and role")
        region = by_id.get(region_id)
        if region is None:
            raise OpenAIVLMPlannerError(f"semantic prior referenced unknown region_id: {region_id}")
        entry = {
            "region_id": region_id,
            "role": role,
            "confidence": confidence,
            "reason": str(raw.get("reason", "")),
            "accepted": bool(confidence >= confidence_min),
        }
        accepted.append(entry)
        if confidence < confidence_min:
            continue
        mask = region.mask.astype(np.float32)
        if role == "subject_material":
            subject_material = np.maximum(subject_material, mask)
            subject = np.maximum(subject, mask)
            seen_subject_material = True
            seen_subject = True
        elif role == "subject":
            subject = np.maximum(subject, mask)
            seen_subject = True
        elif role == "shadow":
            shadow_ownership = np.maximum(shadow_ownership, mask)
            seen_shadow_ownership = True
            saw_shadow_role = True
        elif role == "shadow_search":
            shadow_search = np.maximum(shadow_search, mask)
            seen_shadow_search = True
            saw_shadow_role = True

    return MattingSemanticPrior(
        subject_material_mask=subject_material if seen_subject_material else None,
        subject_mask=subject if seen_subject else None,
        shadow_search_mask=shadow_search if seen_shadow_search else None,
        shadow_ownership_mask=shadow_ownership if seen_shadow_ownership else None,
        shadow_allowed=bool(payload.get("shadow_allowed", True)) if (saw_shadow_role or has_shadow_candidate) else True,
        source=source,
        regions=accepted,
    )


def build_vlm_semantic_request(
    *,
    image_srgb: np.ndarray,
    subject_alpha: np.ndarray,
    background_color: tuple[int, int, int] | np.ndarray,
    regions: list[RiskRegion],
    shadow_alpha: np.ndarray | None = None,
    thumbnail_max_side: int = 768,
    crop_max_side: int = 384,
    crop_padding_px: int = 24,
) -> VLMSemanticRequest:
    if image_srgb.shape[:2] != subject_alpha.shape:
        raise ValueError("image_srgb and subject_alpha must share HxW")
    h, w = subject_alpha.shape
    overlay = _overlay_regions(image_srgb, regions)
    attachments = [
        _png_attachment("original_thumbnail", "Original known-background image.", image_srgb, max_side=thumbnail_max_side),
        _png_attachment("subject_alpha", "Current BiRefNet subject alpha visualization.", _mask_rgb(subject_alpha), max_side=thumbnail_max_side),
        _png_attachment("evidence_overlay", "Candidate semantic-prior regions overlaid on original.", overlay, max_side=thumbnail_max_side),
    ]
    if shadow_alpha is not None:
        attachments.append(_png_attachment("shadow_alpha", "Current scalar-darkening shadow alpha visualization.", _mask_rgb(shadow_alpha), max_side=thumbnail_max_side))
    for idx, region in enumerate(regions):
        crop, metadata = _region_crop(image_srgb, region, crop_padding_px)
        attachments.append(
            _png_attachment(
                f"region_crop_{idx}",
                f"Crop around candidate region {region.id}.",
                crop,
                region_id=region.id,
                metadata=metadata | {"kind": region.kind, "evidence": region.evidence},
                max_side=crop_max_side,
            )
        )

    return VLMSemanticRequest(
        image={
            "width": int(w),
            "height": int(h),
            "background_color": [int(c) for c in np.asarray(background_color).reshape(3)],
        },
        regions=[region.to_prompt_dict() for region in regions],
        attachments=attachments,
        response_schema=_semantic_response_schema([region.id for region in regions]),
        system_prompt=(
            "You are ERMBG's semantic-prior layer. Classify only the provided CV "
            "candidate regions. Do not produce masks, alpha, colors, or code."
        ),
    )


class OpenAIVLMSemanticPriorClient:
    """Call OpenAI vision model to classify semantic prior regions."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        env_path: Path | None = None,
        timeout: float = 120.0,
        max_output_tokens: int = 1200,
        api_url: str = _RESPONSES_URL,
    ) -> None:
        self.model = model
        self.api_key = resolve_openai_api_key(api_key=api_key, env_path=env_path)
        self.timeout = timeout
        self.max_output_tokens = int(max_output_tokens)
        self.api_url = api_url
        self.last_request_payload: dict[str, Any] | None = None
        self.last_raw_response: dict[str, Any] | None = None

    def classify_request(
        self,
        request: VLMSemanticRequest,
        regions: list[RiskRegion],
        shape: tuple[int, int],
    ) -> MattingSemanticPrior:
        body = build_openai_semantic_payload(
            request,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
        )
        self.last_request_payload = body
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(self.api_url, headers=headers, json=body, timeout=self.timeout)
        if resp.status_code >= 400:
            raise OpenAIVLMPlannerError(
                f"OpenAI Responses API failed: {resp.status_code} {resp.text[:500]}"
            )
        try:
            raw = resp.json()
        except ValueError as e:
            raise OpenAIVLMPlannerError("OpenAI Responses API returned non-JSON response.") from e
        self.last_raw_response = raw
        text = extract_openai_output_text(raw)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise OpenAIVLMPlannerError("OpenAI semantic prior returned invalid JSON text.") from e
        return parse_semantic_prior_payload(payload, regions, shape, source=f"openai:{self.model}")


class ComfyQwenVLMSemanticPriorClient:
    """Call the remote ComfyUI Qwen3_VQA node to classify semantic regions."""

    def __init__(
        self,
        *,
        url: str = DEFAULT_COMFY_URL,
        model: str = "Qwen3-VL-4B-Instruct-FP8",
        quantization: str = "none",
        keep_model_loaded: bool = True,
        temperature: float = 0.1,
        max_new_tokens: int = 1200,
        min_pixels: int = 200704,
        max_pixels: int = 1003520,
        seed: int = 1,
        attention: str = "sdpa",
        timeout: float = 600.0,
        poll_interval: float = 1.0,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.model = model
        self.quantization = quantization
        self.keep_model_loaded = bool(keep_model_loaded)
        self.temperature = max(0.01, float(temperature))
        self.max_new_tokens = int(max_new_tokens)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.seed = int(seed)
        self.attention = attention
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self.client_id = uuid.uuid4().hex
        self.last_workflow: dict[str, Any] | None = None
        self.last_history: dict[str, Any] | None = None
        self.last_raw_text: str | None = None

    def classify_request(
        self,
        request: VLMSemanticRequest,
        regions: list[RiskRegion],
        shape: tuple[int, int],
    ) -> MattingSemanticPrior:
        image = _semantic_contact_sheet(request)
        server_name = self._upload_image(image, f"ermbg_qwen_semantic_{uuid.uuid4().hex[:8]}.png")
        prompt = _comfy_qwen_prompt(request)
        workflow = self._build_workflow(server_name, prompt)
        self.last_workflow = workflow
        prompt_id = self._queue(workflow)
        history = self._wait(prompt_id)
        self.last_history = history
        raw_text = _extract_comfy_preview_text(history)
        self.last_raw_text = raw_text
        payload = parse_qwen_json_text(raw_text)
        return parse_semantic_prior_payload(
            payload,
            regions,
            shape,
            source=f"comfy-qwen:{self.model}",
        )

    def _post(self, path: str, **kwargs):
        r = requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def _get(self, path: str, **kwargs):
        r = requests.get(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def _upload_image(self, image: np.ndarray, name: str) -> str:
        buf = BytesIO()
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(buf, format="PNG")
        buf.seek(0)
        resp = self._post(
            "/upload/image",
            files={"image": (name, buf, "image/png")},
            data={"overwrite": "true"},
        )
        return resp.json()["name"]

    def _queue(self, workflow: dict[str, Any]) -> str:
        result = self._post(
            "/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
        ).json()
        if "prompt_id" not in result:
            raise OpenAIVLMPlannerError(f"Comfy /prompt rejected semantic workflow: {result}")
        return result["prompt_id"]

    def _wait(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            data = self._get(f"/history/{prompt_id}").json()
            if prompt_id in data:
                entry = data[prompt_id]
                status = entry.get("status", {})
                if status.get("completed", False):
                    return entry
                if status.get("status_str") == "error":
                    raise OpenAIVLMPlannerError(f"Comfy Qwen workflow errored: {status}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"Comfy Qwen prompt {prompt_id} did not finish in {self.timeout}s")

    def _build_workflow(self, input_image: str, prompt: str) -> dict[str, Any]:
        return {
            "10": {"class_type": "LoadImage", "inputs": {"image": input_image}},
            "20": {
                "class_type": "Qwen3_VQA",
                "inputs": {
                    "text": prompt,
                    "model": self.model,
                    "quantization": self.quantization,
                    "keep_model_loaded": self.keep_model_loaded,
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_new_tokens,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    "seed": self.seed,
                    "attention": self.attention,
                    "image": ["10", 0],
                },
            },
            "30": {"class_type": "PreviewAny", "inputs": {"source": ["20", 0]}},
        }


def build_openai_semantic_payload(
    request: VLMSemanticRequest,
    *,
    model: str = "gpt-4o-mini",
    max_output_tokens: int = 1200,
) -> dict[str, Any]:
    user_text = {
        "image": request.image,
        "regions": request.regions,
        "task": _semantic_task_text(request.regions),
    }
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": json.dumps(user_text, ensure_ascii=False)}
    ]
    for attachment in request.attachments:
        content.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{attachment.mime_type};base64,{attachment.data_base64}",
            }
        )
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": request.system_prompt}]},
            {"role": "user", "content": content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ermbg_semantic_prior",
                "strict": True,
                "schema": request.response_schema,
            }
        },
        "max_output_tokens": int(max_output_tokens),
    }


def parse_qwen_json_text(text: str) -> dict[str, Any]:
    """Parse JSON returned by Qwen/PreviewAny, including wrapped string lists."""
    raw = text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
            raw = parsed[0].strip()
        elif isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise OpenAIVLMPlannerError("Qwen semantic prior did not return JSON")
        parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise OpenAIVLMPlannerError("Qwen semantic prior JSON must be an object")
    return parsed


def _extract_comfy_preview_text(history_entry: dict[str, Any]) -> str:
    for node_out in history_entry.get("outputs", {}).values():
        text_items = node_out.get("text")
        if isinstance(text_items, list) and text_items:
            return "\n".join(str(item) for item in text_items)
        if isinstance(text_items, str):
            return text_items
    raise OpenAIVLMPlannerError("Comfy Qwen history did not contain PreviewAny text")


def _comfy_qwen_prompt(request: VLMSemanticRequest) -> str:
    user_text = {
        "image": request.image,
        "regions": request.regions,
        "required_json_schema": {
            "shadow_allowed": "boolean",
            "regions": [
                {
                    "region_id": "one of the supplied ids",
                    "role": "subject_material|subject|shadow|shadow_search|background|uncertain",
                    "confidence": "0..1",
                    "reason": "short string",
                }
            ],
        },
    }
    return (
        "You are ERMBG's semantic-prior layer. The attached image is a labeled "
        "contact sheet containing the original image, alpha/shadow previews, an "
        "evidence overlay, and candidate region crops. Classify ONLY the supplied "
        "candidate regions. Do not create masks, alpha, colors, or code.\n\n"
        + _semantic_task_text(request.regions)
        + "\n\n"
        "Return raw JSON only. No markdown. No commentary.\n\n"
        + json.dumps(user_text, ensure_ascii=False)
    )


def _semantic_task_text(regions: list[dict[str, Any]]) -> str:
    kinds = {str(region.get("kind", "")) for region in regions}
    if kinds and kinds <= {"owned_shadow_candidate", "shadow_search_region"}:
        return (
            "Classify each candidate region for shadow ownership. Use role 'shadow' "
            "only when the candidate is an owned contact/cast shadow that should be "
            "kept with the subject. Use 'shadow_search' only for a broad area where "
            "CV may search for owned shadow evidence. Use 'subject' if the dark "
            "pixels are actually subject/body/material, and 'background' when they "
            "are plain background, unrelated marks, or artifacts. Set shadow_allowed "
            "false when none of the supplied candidates should be preserved as "
            "owned shadow. Local CV still verifies scalar background darkening and "
            "sets all shadow opacity."
        )
    if "owned_shadow_candidate" in kinds or "shadow_search_region" in kinds:
        return (
            "Classify each candidate region. Use 'shadow' only for owned contact/cast "
            "shadow and 'shadow_search' only for broad search areas. Use "
            "'subject_material' only when green/greenish pixels belong to the object "
            "material; use 'background' for green screen/spill or unrelated marks. "
            "Local CV will verify darkening and set alpha/opacity."
        )
    return (
        "Classify each candidate region. Use role 'subject_material' when the "
        "green/greenish pixels belong to the object material, such as glass, "
        "enamel, gem, panel fill, fabric, or intentional artwork. Use "
        "'background' when it is green screen/background spill, and 'shadow' "
        "only for owned contact/cast shadow. Local CV will use subject_material "
        "only to protect foreground colors from green despill; it will not set alpha."
    )


def _semantic_contact_sheet(request: VLMSemanticRequest) -> np.ndarray:
    tiles: list[tuple[str, Image.Image]] = []
    for attachment in request.attachments:
        raw = base64.b64decode(attachment.data_base64)
        image = Image.open(BytesIO(raw)).convert("RGB")
        label = attachment.id if attachment.region_id is None else f"{attachment.id} {attachment.region_id}"
        tiles.append((label, image))

    tile_w, tile_h = 384, 320
    label_h = 28
    cols = 2 if len(tiles) <= 4 else 3
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(tiles):
        col = idx % cols
        row = idx // cols
        x = col * tile_w
        y = row * (tile_h + label_h)
        draw.text((x + 8, y + 8), label[:55], fill=(0, 0, 0))
        im = image.copy()
        im.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(im, (x + (tile_w - im.width) // 2, y + label_h + (tile_h - im.height) // 2))
    return np.asarray(sheet, dtype=np.uint8)


def _semantic_response_schema(region_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["regions", "shadow_allowed"],
        "properties": {
            "shadow_allowed": {"type": "boolean"},
            "regions": {
                "type": "array",
                "maxItems": max(1, len(region_ids)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["region_id", "role", "confidence", "reason"],
                    "properties": {
                        "region_id": {"type": "string", "enum": region_ids},
                        "role": {
                            "type": "string",
                            "enum": ["subject_material", "subject", "shadow", "shadow_search", "background", "uncertain"],
                        },
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }


def _png_attachment(
    attachment_id: str,
    purpose: str,
    image: np.ndarray,
    *,
    region_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_side: int,
) -> VLMSemanticAttachment:
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pil.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return VLMSemanticAttachment(
        id=attachment_id,
        purpose=purpose,
        region_id=region_id,
        width=int(pil.width),
        height=int(pil.height),
        data_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
        metadata=dict(metadata or {}),
    )


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    m = (np.clip(mask.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.dstack([m, m, m])


def _overlay_regions(image_srgb: np.ndarray, regions: list[RiskRegion]) -> np.ndarray:
    overlay = image_srgb.copy()
    for region in regions:
        color = np.asarray(REGION_COLORS.get(region.kind, REGION_COLORS["unknown"]), dtype=np.float32)
        mask = region.mask.astype(bool)
        overlay[mask] = (0.50 * overlay[mask].astype(np.float32) + 0.50 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in color), 3)
    return overlay


def _region_crop(image_srgb: np.ndarray, region: RiskRegion, padding_px: int) -> tuple[np.ndarray, dict[str, Any]]:
    ys, xs = np.where(region.mask)
    h, w = region.mask.shape
    if ys.size:
        x0 = max(0, int(xs.min()) - padding_px)
        y0 = max(0, int(ys.min()) - padding_px)
        x1 = min(w, int(xs.max()) + 1 + padding_px)
        y1 = min(h, int(ys.max()) + 1 + padding_px)
    else:
        x0 = y0 = 0
        x1 = y1 = 1
    crop = image_srgb[y0:y1, x0:x1].copy()
    local = region.mask[y0:y1, x0:x1]
    color = np.asarray(REGION_COLORS.get(region.kind, REGION_COLORS["unknown"]), dtype=np.float32)
    crop[local] = (0.50 * crop[local].astype(np.float32) + 0.50 * color).astype(np.uint8)
    contours, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, tuple(int(c) for c in color), 2)
    return crop, {"bbox_xyxy": [x0, y0, x1, y1], "padding_px": int(padding_px)}


__all__ = [
    "MattingSemanticPrior",
    "ComfyQwenVLMSemanticPriorClient",
    "OpenAIVLMSemanticPriorClient",
    "VLMSemanticRequest",
    "build_openai_semantic_payload",
    "build_vlm_semantic_request",
    "extract_shadow_candidate_regions",
    "extract_subject_material_candidate_regions",
    "parse_semantic_prior_payload",
    "parse_qwen_json_text",
]
