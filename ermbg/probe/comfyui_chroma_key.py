"""Run a plain ColorToMask chroma-key node on the remote ComfyUI server."""

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

_DEFAULT_WORKFLOW = Path(__file__).parent / "comfyui_chroma_key.json"
_ALPHA_NODE = "40"


@dataclass(frozen=True)
class ComfyChromaKeyResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground_srgb: np.ndarray
    debug: dict[str, Any]


class ComfyUIChromaKeyClient:
    """Submit a remote ComfyUI ColorToMask workflow and return an RGBA cutout."""

    def __init__(
        self,
        url: str = DEFAULT_COMFY_URL,
        workflow_path: Path | str | None = None,
        timeout: float = 120.0,
        poll_interval: float = 0.25,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.timeout = timeout
        self.poll_interval = poll_interval
        path = Path(workflow_path) if workflow_path else _DEFAULT_WORKFLOW
        self.workflow_template = json.loads(path.read_text())

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

    def _download_node_image(self, history_entry: dict[str, Any], node_id: str) -> np.ndarray:
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
        im = Image.open(BytesIO(r.content)).convert("L")
        return np.asarray(im, dtype=np.uint8)

    def _render_workflow(
        self,
        *,
        input_image: str,
        key_color: tuple[int, int, int],
        threshold: int,
        filename_prefix: str,
    ) -> dict[str, Any]:
        # The threshold is the ordinary chroma-key color range/tolerance. It is
        # intentionally kept explicit here so eval batches can compare whether
        # range tuning fixes green-screen leakage without involving a learned
        # matte model such as CorridorKey.
        rendered = Template(json.dumps(self.workflow_template)).safe_substitute(
            input_image=input_image,
            red=int(key_color[0]),
            green=int(key_color[1]),
            blue=int(key_color[2]),
            threshold=int(threshold),
            filename_prefix=json.dumps(filename_prefix)[1:-1],
        )
        workflow = json.loads(rendered)
        workflow.pop("_comment", None)
        inputs = workflow["20"]["inputs"]
        for key in ("red", "green", "blue", "threshold", "per_batch"):
            inputs[key] = int(inputs[key])
        return workflow

    def matte(
        self,
        image_srgb: np.ndarray,
        *,
        key_color: tuple[int, int, int] = (0, 200, 0),
        threshold: int = 35,
    ) -> ComfyChromaKeyResult:
        if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
            raise ValueError("matte() expects HxWx3 sRGB uint8")
        threshold = int(np.clip(threshold, 0, 255))

        h, w = image_srgb.shape[:2]
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        prefix = f"ermbg_chromakey_{uuid.uuid4().hex[:8]}"
        step_start = time.perf_counter()
        server_image = self._upload(image_srgb, f"{prefix}.png")
        timings["upload_sec"] = time.perf_counter() - step_start
        workflow = self._render_workflow(
            input_image=server_image,
            key_color=key_color,
            threshold=threshold,
            filename_prefix=prefix,
        )
        step_start = time.perf_counter()
        prompt_id = self._queue(workflow)
        timings["queue_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        history = self._wait(prompt_id)
        timings["wait_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        alpha_u8 = self._download_node_image(history, _ALPHA_NODE)
        timings["download_alpha_sec"] = time.perf_counter() - step_start

        if alpha_u8.shape != (h, w):
            alpha_u8 = cv2.resize(alpha_u8, (w, h), interpolation=cv2.INTER_LINEAR)

        alpha = alpha_u8.astype(np.float32) / 255.0
        rgba = np.dstack([image_srgb, alpha_u8]).astype(np.uint8)
        timings["total_sec"] = time.perf_counter() - total_start
        return ComfyChromaKeyResult(
            rgba=rgba,
            alpha=np.clip(alpha, 0.0, 1.0).astype(np.float32),
            foreground_srgb=image_srgb.astype(np.uint8),
            debug={
                "backend": "comfy-chromakey",
                "node": "ColorToMask",
                "prompt_id": prompt_id,
                "server_image": server_image,
                "alpha_node": _ALPHA_NODE,
                "key_color": list(key_color),
                "threshold": threshold,
                "timings": timings,
            },
        )


__all__ = ["ComfyChromaKeyResult", "ComfyUIChromaKeyClient"]
