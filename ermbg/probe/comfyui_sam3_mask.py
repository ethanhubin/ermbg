"""Run SAM3 automatic mask generation on the remote ComfyUI server."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image

from ermbg.comfy import DEFAULT_COMFY_URL

_DEFAULT_WORKFLOW = Path(__file__).parent / "comfyui_sam3_mask.json"
_MASK_NODE = "40"
DEFAULT_SAM3_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"


@dataclass(frozen=True)
class ComfyUISAM3MaskResult:
    mask: np.ndarray
    debug: dict[str, Any]


class ComfyUISAM3MaskClient:
    """Submit a SAM3 mask workflow to a remote ComfyUI server."""

    def __init__(
        self,
        url: str = DEFAULT_COMFY_URL,
        workflow_path: Path | str | None = None,
        timeout: float = 600.0,
        poll_interval: float = 0.25,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.timeout = timeout
        self.poll_interval = poll_interval
        path = Path(workflow_path) if workflow_path else _DEFAULT_WORKFLOW
        self.workflow_template = json.loads(path.read_text(encoding="utf-8"))

    def _post(self, path: str, **kwargs):
        r = requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def _get(self, path: str, **kwargs):
        r = requests.get(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def _upload(self, image: np.ndarray, name: str) -> str:
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        buf.seek(0)
        files = {"image": (name, buf, "image/png")}
        data = {"overwrite": "true"}
        return self._post("/upload/image", files=files, data=data).json()["name"]

    def _queue(self, workflow: dict[str, Any]) -> str:
        body = {"prompt": workflow, "client_id": self.client_id}
        result = self._post("/prompt", json=body).json()
        if "prompt_id" not in result:
            raise RuntimeError(f"Comfy /prompt rejected: {result}")
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
                    raise RuntimeError(f"Comfy workflow errored: {entry.get('status')}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {self.timeout}s")

    def _download_node_image(self, history_entry: dict[str, Any], node_id: str, mode: str) -> np.ndarray:
        node_out = history_entry.get("outputs", {}).get(str(node_id), {})
        images = node_out.get("images", [])
        if not images:
            raise RuntimeError(f"No ComfyUI image output found for node {node_id}")
        img_meta = images[0]
        params = {
            "filename": img_meta["filename"],
            "subfolder": img_meta.get("subfolder", ""),
            "type": img_meta.get("type", "output"),
        }
        r = self._get("/view", params=params)
        im = Image.open(BytesIO(r.content)).convert(mode)
        return np.asarray(im, dtype=np.uint8)

    def _render_workflow(
        self,
        *,
        input_image: str,
        checkpoint: str,
        threshold: float,
        refine_iterations: int,
        filename_prefix: str,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> dict[str, Any]:
        rendered = Template(json.dumps(self.workflow_template)).safe_substitute(
            input_image=input_image,
            checkpoint=checkpoint,
            threshold=float(threshold),
            refine_iterations=int(refine_iterations),
            filename_prefix=json.dumps(filename_prefix)[1:-1],
        )
        workflow = json.loads(rendered)
        workflow.pop("_comment", None)
        inputs = workflow["20"]["inputs"]
        inputs["threshold"] = float(inputs["threshold"])
        inputs["refine_iterations"] = int(inputs["refine_iterations"])
        if image_width is not None and image_height is not None:
            # SAM3_Detect completes with an empty mask when it receives no
            # prompt. A full-image box gives the detector an automatic seed
            # while keeping the mask generation image-agnostic.
            inputs["bboxes"] = {
                "x": 0,
                "y": 0,
                "width": int(image_width),
                "height": int(image_height),
            }
        return workflow

    def mask(
        self,
        image_srgb: np.ndarray,
        *,
        checkpoint: str = DEFAULT_SAM3_CHECKPOINT,
        threshold: float = 0.5,
        refine_iterations: int = 2,
    ) -> ComfyUISAM3MaskResult:
        if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
            raise ValueError("mask() expects HxWx3 sRGB uint8")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0 <= int(refine_iterations) <= 5:
            raise ValueError("refine_iterations must be between 0 and 5")

        h, w = image_srgb.shape[:2]
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        prefix = f"ermbg_sam3_{uuid.uuid4().hex[:8]}"
        step_start = time.perf_counter()
        server_image = self._upload(image_srgb, f"{prefix}.png")
        timings["upload_sec"] = time.perf_counter() - step_start
        workflow = self._render_workflow(
            input_image=server_image,
            checkpoint=checkpoint,
            threshold=threshold,
            refine_iterations=refine_iterations,
            filename_prefix=prefix,
            image_width=w,
            image_height=h,
        )
        step_start = time.perf_counter()
        prompt_id = self._queue(workflow)
        timings["queue_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        history = self._wait(prompt_id)
        timings["wait_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        mask_u8 = self._download_node_image(history, _MASK_NODE, "L")
        timings["download_mask_sec"] = time.perf_counter() - step_start
        if mask_u8.shape != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask_u8.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)
        timings["total_sec"] = time.perf_counter() - total_start
        return ComfyUISAM3MaskResult(
            mask=mask,
            debug={
                "backend": "comfy-sam3",
                "prompt_id": prompt_id,
                "server_image": server_image,
                "mask_node": _MASK_NODE,
                "settings": {
                    "checkpoint": checkpoint,
                    "threshold": float(threshold),
                    "refine_iterations": int(refine_iterations),
                },
                "mask": {
                    "min": float(mask.min()),
                    "max": float(mask.max()),
                    "mean": float(mask.mean()),
                    "pixels_gt_50": int((mask > 0.5).sum()),
                },
                "timings": timings,
            },
        )


__all__ = [
    "DEFAULT_SAM3_CHECKPOINT",
    "ComfyUISAM3MaskClient",
    "ComfyUISAM3MaskResult",
]
