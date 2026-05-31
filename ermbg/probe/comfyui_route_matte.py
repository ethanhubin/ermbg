"""Run full ERMBG auto route/matte in one remote ComfyUI node."""

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

_DEFAULT_WORKFLOW = Path(__file__).parent / "comfyui_route_matte.json"
_ROUTE_NODE = "20"
_FOREGROUND_NODE = "30"
_ALPHA_NODE = "50"
_RGBA_RGB_NODE = "60"
_AUX_NODE = "70"


@dataclass(frozen=True)
class ComfyRouteMatteResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground_srgb: np.ndarray
    background_color: tuple[int, int, int]
    strategy_name: str
    report: dict[str, Any]
    debug: dict[str, Any]


class ComfyUIRouteMatteClient:
    """Submit the single-node ERMBG auto matte workflow to remote ComfyUI."""

    def __init__(
        self,
        url: str = DEFAULT_COMFY_URL,
        workflow_path: Path | str | None = None,
        timeout: float = 600.0,
        poll_interval: float = 0.05,
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

    def _upload(self, image: np.ndarray, alpha: np.ndarray | None, name: str) -> str:
        buf = BytesIO()
        if alpha is not None:
            alpha_u8 = np.clip(alpha.astype(np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            payload = np.dstack([image, alpha_u8])
            Image.fromarray(payload, mode="RGBA").save(buf, format="PNG")
        else:
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

    def _route_payload(self, history_entry: dict[str, Any]) -> dict[str, Any]:
        node_out = history_entry.get("outputs", {}).get(_ROUTE_NODE, {})
        texts = node_out.get("text") or node_out.get("string") or []
        if not texts:
            raise RuntimeError("ErmbgRouteMatte did not return route metadata text")
        return json.loads(str(texts[0]))

    def _render_workflow(
        self,
        *,
        input_image: str,
        shadow_mode: str,
        corridorkey_screen_mode: str,
        corridorkey_preset: str,
        corridorkey_hard_ui_hint_mode: str,
        fallback_bg_color: str,
        pymatting_method: str,
        pymatting_image_space: str,
        pymatting_bg_source: str,
        pymatting_bg_color: str,
        pymatting_bg_threshold: float,
        pymatting_fg_threshold: float,
        pymatting_boundary_band_px: int,
        pymatting_auto_adapt: bool,
        pymatting_cg_maxiter: int,
        pymatting_cg_rtol: float,
        filename_prefix: str,
    ) -> dict[str, Any]:
        rendered = Template(json.dumps(self.workflow_template)).safe_substitute(
            input_image=input_image,
            shadow_mode=shadow_mode,
            corridorkey_screen_mode=corridorkey_screen_mode,
            corridorkey_preset=corridorkey_preset,
            corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
            fallback_bg_color=fallback_bg_color,
            pymatting_method=pymatting_method,
            pymatting_image_space=pymatting_image_space,
            pymatting_bg_source=pymatting_bg_source,
            pymatting_bg_color=pymatting_bg_color,
            pymatting_bg_threshold=float(pymatting_bg_threshold),
            pymatting_fg_threshold=float(pymatting_fg_threshold),
            pymatting_boundary_band_px=int(pymatting_boundary_band_px),
            pymatting_auto_adapt="true" if pymatting_auto_adapt else "false",
            pymatting_cg_maxiter=int(pymatting_cg_maxiter),
            pymatting_cg_rtol=float(pymatting_cg_rtol),
            filename_prefix=json.dumps(filename_prefix)[1:-1],
        )
        workflow = json.loads(rendered)
        workflow.pop("_comment", None)
        inputs = workflow[_ROUTE_NODE]["inputs"]
        inputs["pymatting_bg_threshold"] = float(inputs["pymatting_bg_threshold"])
        inputs["pymatting_fg_threshold"] = float(inputs["pymatting_fg_threshold"])
        inputs["pymatting_boundary_band_px"] = int(inputs["pymatting_boundary_band_px"])
        inputs["pymatting_auto_adapt"] = str(inputs["pymatting_auto_adapt"]).lower() in {"1", "true", "yes", "on"}
        inputs["pymatting_cg_maxiter"] = int(inputs["pymatting_cg_maxiter"])
        inputs["pymatting_cg_rtol"] = float(inputs["pymatting_cg_rtol"])
        return workflow

    def matte(
        self,
        image_srgb: np.ndarray,
        *,
        source_alpha: np.ndarray | None = None,
        fallback_bg_color: tuple[int, int, int] = (0, 200, 0),
        shadow_mode: str = "on",
        corridorkey_screen_mode: str = "auto",
        corridorkey_preset: str = "auto",
        corridorkey_hard_ui_hint_mode: str = "bbox_2px",
        pymatting_method: str = "cf",
        pymatting_image_space: str = "linear",
        pymatting_bg_source: str = "auto",
        pymatting_bg_color: tuple[int, int, int] | None = None,
        pymatting_bg_threshold: float = 3.5,
        pymatting_fg_threshold: float = 30.0,
        pymatting_boundary_band_px: int = 2,
        pymatting_auto_adapt: bool = True,
        pymatting_cg_maxiter: int = 1000,
        pymatting_cg_rtol: float = 1e-6,
    ) -> ComfyRouteMatteResult:
        if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
            raise ValueError("matte() expects HxWx3 sRGB uint8")
        h, w = image_srgb.shape[:2]
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        prefix = f"ermbg_route_matte_{uuid.uuid4().hex[:8]}"

        step_start = time.perf_counter()
        server_image = self._upload(image_srgb, source_alpha, f"{prefix}.png")
        timings["upload_sec"] = time.perf_counter() - step_start
        bg_text = ",".join(str(int(c)) for c in fallback_bg_color)
        pymat_bg_text = ",".join(str(int(c)) for c in (pymatting_bg_color or fallback_bg_color))
        workflow = self._render_workflow(
            input_image=server_image,
            shadow_mode=shadow_mode,
            corridorkey_screen_mode=corridorkey_screen_mode,
            corridorkey_preset=corridorkey_preset,
            corridorkey_hard_ui_hint_mode=corridorkey_hard_ui_hint_mode,
            fallback_bg_color=bg_text,
            pymatting_method=pymatting_method,
            pymatting_image_space=pymatting_image_space,
            pymatting_bg_source=pymatting_bg_source,
            pymatting_bg_color=pymat_bg_text,
            pymatting_bg_threshold=pymatting_bg_threshold,
            pymatting_fg_threshold=pymatting_fg_threshold,
            pymatting_boundary_band_px=pymatting_boundary_band_px,
            pymatting_auto_adapt=pymatting_auto_adapt,
            pymatting_cg_maxiter=pymatting_cg_maxiter,
            pymatting_cg_rtol=pymatting_cg_rtol,
            filename_prefix=prefix,
        )
        step_start = time.perf_counter()
        prompt_id = self._queue(workflow)
        timings["queue_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        history = self._wait(prompt_id)
        timings["wait_sec"] = time.perf_counter() - step_start
        payload = self._route_payload(history)
        step_start = time.perf_counter()
        foreground = self._download_node_image(history, _FOREGROUND_NODE, "RGB")
        timings["download_foreground_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        alpha_u8 = self._download_node_image(history, _ALPHA_NODE, "L")
        timings["download_alpha_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        rgba_rgb = self._download_node_image(history, _RGBA_RGB_NODE, "RGB")
        timings["download_rgba_rgb_sec"] = time.perf_counter() - step_start
        timings["total_sec"] = time.perf_counter() - total_start

        if foreground.shape[:2] != (h, w):
            foreground = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if alpha_u8.shape != (h, w):
            alpha_u8 = cv2.resize(alpha_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        if rgba_rgb.shape[:2] != (h, w):
            rgba_rgb = cv2.resize(rgba_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        alpha = np.clip(alpha_u8.astype(np.float32) / 255.0, 0.0, 1.0)
        rgba = np.dstack([rgba_rgb.astype(np.uint8), alpha_u8.astype(np.uint8)])

        background = payload.get("background_color", [0, 0, 0])
        background_color = tuple(int(c) for c in background[:3])
        debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
        debug = dict(debug)
        node_timings = debug.get("timings") if isinstance(debug.get("timings"), dict) else {}
        debug["backend"] = payload.get("selected_backend", "auto")
        debug["prompt_id"] = prompt_id
        debug["server_image"] = server_image
        debug["remote_client_timings"] = timings
        debug["timings"] = {
            **node_timings,
            "remote_upload_sec": timings.get("upload_sec", 0.0),
            "remote_queue_sec": timings.get("queue_sec", 0.0),
            "remote_wait_sec": timings.get("wait_sec", 0.0),
            "remote_download_foreground_sec": timings.get("download_foreground_sec", 0.0),
            "remote_download_alpha_sec": timings.get("download_alpha_sec", 0.0),
            "remote_download_rgba_rgb_sec": timings.get("download_rgba_rgb_sec", 0.0),
            "remote_total_sec": timings.get("total_sec", 0.0),
        }
        report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
        return ComfyRouteMatteResult(
            rgba=rgba.astype(np.uint8),
            alpha=alpha.astype(np.float32),
            foreground_srgb=foreground.astype(np.uint8),
            background_color=background_color,
            strategy_name=str(payload.get("strategy_name") or report.get("strategy", {}).get("name") or "comfy_route_matte"),
            report=report,
            debug=debug,
        )


__all__ = ["ComfyRouteMatteResult", "ComfyUIRouteMatteClient"]
