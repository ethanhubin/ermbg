"""Run ERMBG PyMatting Known-B on the remote ComfyUI server."""

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

_DEFAULT_WORKFLOW = Path(__file__).parent / "comfyui_pymatting_known_b.json"
_FOREGROUND_NODE = "30"
_ALPHA_NODE = "50"
_TRIMAP_NODE = "60"


@dataclass(frozen=True)
class ComfyPyMattingKnownBResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground_srgb: np.ndarray
    trimap_u8: np.ndarray
    debug: dict[str, Any]


class ComfyUIPyMattingKnownBClient:
    """Submit the ERMBG PyMatting Known-B node to remote ComfyUI."""

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
        method: str,
        image_space: str,
        bg_source: str,
        bg_color: str,
        bg_threshold: float,
        fg_threshold: float,
        boundary_band_px: int,
        auto_adapt: bool,
        cg_maxiter: int,
        cg_rtol: float,
        filename_prefix: str,
    ) -> dict[str, Any]:
        rendered = Template(json.dumps(self.workflow_template)).safe_substitute(
            input_image=input_image,
            method=method,
            image_space=image_space,
            bg_source=bg_source,
            bg_color=bg_color,
            bg_threshold=float(bg_threshold),
            fg_threshold=float(fg_threshold),
            boundary_band_px=int(boundary_band_px),
            auto_adapt="true" if auto_adapt else "false",
            cg_maxiter=int(cg_maxiter),
            cg_rtol=float(cg_rtol),
            filename_prefix=json.dumps(filename_prefix)[1:-1],
        )
        workflow = json.loads(rendered)
        workflow.pop("_comment", None)
        inputs = workflow["20"]["inputs"]
        inputs["bg_threshold"] = float(inputs["bg_threshold"])
        inputs["fg_threshold"] = float(inputs["fg_threshold"])
        inputs["boundary_band_px"] = int(inputs["boundary_band_px"])
        inputs["auto_adapt"] = str(inputs["auto_adapt"]).strip().lower() in {"1", "true", "yes", "on"}
        inputs["cg_maxiter"] = int(inputs["cg_maxiter"])
        inputs["cg_rtol"] = float(inputs["cg_rtol"])
        return workflow

    def matte(
        self,
        image_srgb: np.ndarray,
        *,
        method: str = "cf",
        image_space: str = "linear",
        bg_source: str = "auto",
        bg_color: tuple[int, int, int] | None = None,
        bg_threshold: float = 3.5,
        fg_threshold: float = 30.0,
        boundary_band_px: int = 2,
        auto_adapt: bool = True,
        cg_maxiter: int = 1000,
        cg_rtol: float = 1e-6,
    ) -> ComfyPyMattingKnownBResult:
        if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
            raise ValueError("matte() expects HxWx3 sRGB uint8")

        h, w = image_srgb.shape[:2]
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        prefix = f"ermbg_pymatting_known_b_{uuid.uuid4().hex[:8]}"
        step_start = time.perf_counter()
        server_image = self._upload(image_srgb, f"{prefix}.png")
        timings["upload_sec"] = time.perf_counter() - step_start
        bg_color_text = ",".join(str(int(c)) for c in (bg_color or (0, 200, 0)))
        workflow = self._render_workflow(
            input_image=server_image,
            method=method,
            image_space=image_space,
            bg_source=bg_source,
            bg_color=bg_color_text,
            bg_threshold=bg_threshold,
            fg_threshold=fg_threshold,
            boundary_band_px=boundary_band_px,
            auto_adapt=auto_adapt,
            cg_maxiter=cg_maxiter,
            cg_rtol=cg_rtol,
            filename_prefix=prefix,
        )
        step_start = time.perf_counter()
        prompt_id = self._queue(workflow)
        timings["queue_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        history = self._wait(prompt_id)
        timings["wait_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        foreground = self._download_node_image(history, _FOREGROUND_NODE, "RGB")
        timings["download_foreground_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        alpha_u8 = self._download_node_image(history, _ALPHA_NODE, "L")
        timings["download_alpha_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        trimap_u8 = self._download_node_image(history, _TRIMAP_NODE, "L")
        timings["download_trimap_sec"] = time.perf_counter() - step_start

        if foreground.shape[:2] != (h, w):
            foreground = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if alpha_u8.shape != (h, w):
            alpha_u8 = cv2.resize(alpha_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        if trimap_u8.shape != (h, w):
            trimap_u8 = cv2.resize(trimap_u8, (w, h), interpolation=cv2.INTER_NEAREST)

        alpha = np.clip(alpha_u8.astype(np.float32) / 255.0, 0.0, 1.0)
        rgba = np.dstack([foreground, alpha_u8]).astype(np.uint8)
        timings["total_sec"] = time.perf_counter() - total_start
        return ComfyPyMattingKnownBResult(
            rgba=rgba,
            alpha=alpha.astype(np.float32),
            foreground_srgb=foreground.astype(np.uint8),
            trimap_u8=trimap_u8.astype(np.uint8),
            debug={
                "backend": "comfy-pymatting-known-b",
                "prompt_id": prompt_id,
                "server_image": server_image,
                "foreground_node": _FOREGROUND_NODE,
                "alpha_node": _ALPHA_NODE,
                "trimap_node": _TRIMAP_NODE,
                "settings": {
                    "method": method,
                    "image_space": image_space,
                    "bg_source": bg_source,
                    "bg_color": bg_color_text,
                    "bg_threshold": float(bg_threshold),
                    "fg_threshold": float(fg_threshold),
                    "boundary_band_px": int(boundary_band_px),
                    "auto_adapt": bool(auto_adapt),
                    "cg_maxiter": int(cg_maxiter),
                    "cg_rtol": float(cg_rtol),
                },
                "timings": timings,
            },
        )


__all__ = ["ComfyPyMattingKnownBResult", "ComfyUIPyMattingKnownBClient"]
