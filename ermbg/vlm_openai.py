"""OpenAI-backed VLM planner client for ERMBG candidate plans.

This module is optional and network-backed. It never touches pixels directly:
OpenAI receives the visual planner request and must return constrained
CandidatePlan JSON, which is then parsed and validated by local code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from .planner import CandidatePlan
from .vlm_payload import VLMPlannerRequest
from .vlm_planner import parse_candidate_plans

_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIVLMPlannerError(RuntimeError):
    """Raised when the OpenAI VLM planner call fails or returns invalid output."""


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries from .env without adding a dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_openai_api_key(api_key: str | None = None, env_path: Path | None = None) -> str:
    """Resolve OPENAI_API_KEY from an explicit value, environment, or .env file."""
    if api_key:
        return api_key
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if env_path is not None:
        _load_env_file(env_path)
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key
    raise OpenAIVLMPlannerError("OpenAI VLM planner requires OPENAI_API_KEY in the environment or .env.")


def _schema_for_openai(request: VLMPlannerRequest) -> dict[str, Any]:
    """Return the response schema shape expected by Responses text.format."""
    return {
        "type": "json_schema",
        "name": "ermbg_candidate_plans",
        "strict": True,
        "schema": request.response_schema,
    }


def _attachment_prompt_lines(request: VLMPlannerRequest) -> list[str]:
    lines = []
    for attachment in request.attachments:
        suffix = f" region_id={attachment.region_id}" if attachment.region_id else ""
        lines.append(f"- {attachment.id}:{suffix} {attachment.purpose}")
    return lines


def build_openai_responses_payload(
    request: VLMPlannerRequest,
    *,
    model: str = "gpt-4o-mini",
    max_output_tokens: int = 2000,
) -> dict[str, Any]:
    """Build a Responses API payload from a local VLM planner request."""
    user_text = {
        "planner_bundle": request.planner_bundle,
        "attachment_manifest": [
            {
                "id": attachment.id,
                "purpose": attachment.purpose,
                "region_id": attachment.region_id,
                "width": attachment.width,
                "height": attachment.height,
                "metadata": attachment.metadata,
            }
            for attachment in request.attachments
        ],
        "task": (
            "Interpret the evidence regions and visual attachments. Return CandidatePlan JSON only. "
            "Do not invent region IDs or tools. Each candidate must describe a complete matte interpretation "
            "for the image, and may contain multiple operations. Do not return one candidate per region."
        ),
    }
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                json.dumps(user_text, ensure_ascii=False)
                + "\n\nAttachments, in order:\n"
                + "\n".join(_attachment_prompt_lines(request))
            ),
        }
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
            {
                "role": "system",
                "content": [{"type": "input_text", "text": request.system_prompt}],
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        "text": {"format": _schema_for_openai(request)},
        "max_output_tokens": int(max_output_tokens),
    }


def extract_openai_output_text(payload: dict[str, Any]) -> str:
    """Extract generated text from a Responses API response payload."""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    parts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            refusal = content.get("refusal")
            if isinstance(refusal, str) and refusal:
                raise OpenAIVLMPlannerError(f"OpenAI VLM planner refused: {refusal}")
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "".join(parts)
    raise OpenAIVLMPlannerError("OpenAI VLM planner response did not contain output text.")


class OpenAIVLMPlannerClient:
    """Call OpenAI Responses API and parse CandidatePlan JSON."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        env_path: Path | None = None,
        timeout: float = 120.0,
        max_output_tokens: int = 2000,
        api_url: str = _RESPONSES_URL,
    ) -> None:
        self.model = model
        self.api_key = resolve_openai_api_key(api_key=api_key, env_path=env_path)
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.api_url = api_url
        self.last_request_payload: dict[str, Any] | None = None
        self.last_raw_response: dict[str, Any] | None = None

    def plan_request(self, request: VLMPlannerRequest) -> list[CandidatePlan]:
        """Return parsed candidate plans for a visual planner request."""
        body = build_openai_responses_payload(
            request,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
        )
        self.last_request_payload = body
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
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
            candidate_payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise OpenAIVLMPlannerError("OpenAI VLM planner returned invalid JSON text.") from e
        return parse_candidate_plans(candidate_payload)


__all__ = [
    "OpenAIVLMPlannerClient",
    "OpenAIVLMPlannerError",
    "build_openai_responses_payload",
    "extract_openai_output_text",
    "resolve_openai_api_key",
]
