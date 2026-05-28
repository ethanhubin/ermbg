"""Run the full ERMBG ComfyUI custom-node workflow remotely."""

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

_DEFAULT_WORKFLOW = Path(__file__).parent / "comfyui_ermbg_matte.json"
_FOREGROUND_NODE = "30"
_ALPHA_NODE = "50"
_RGBA_RGB_NODE = "60"


@dataclass(frozen=True)
class ComfyErmbgMatteResult:
    rgba: np.ndarray
    alpha: np.ndarray
    foreground_srgb: np.ndarray
    debug: dict[str, Any]


class ComfyUIErmbgMatteClient:
    """Submit the ERMBG AutoMatte workflow to a remote ComfyUI server."""

    def __init__(
        self,
        url: str = DEFAULT_COMFY_URL,
        workflow_path: Path | str | None = None,
        timeout: float = 600.0,
        poll_interval: float = 0.25,
        inspect_object_info: bool = False,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.timeout = timeout
        self.poll_interval = poll_interval
        path = Path(workflow_path) if workflow_path else _DEFAULT_WORKFLOW
        self.workflow_template = json.loads(path.read_text())
        # /object_info is very large on plugin-heavy ComfyUI installs and can
        # keep the server busy even after the client-side timeout fires. The
        # deployed ERMBG node is expected to be modern, so the hot path assumes
        # shadow_mode exists and only inspects the schema when explicitly asked.
        self._automatte_inputs: set[str] | None = None if inspect_object_info else {"shadow_mode"}

    def _post(self, path: str, **kwargs):
        return self._request_with_retry("post", path, **kwargs)

    def _get(self, path: str, **kwargs):
        return self._request_with_retry("get", path, **kwargs)

    def _request_with_retry(self, method: str, path: str, **kwargs):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request = requests.post if method == "post" else requests.get
                r = request(f"{self.base_url}{path}", timeout=60, **kwargs)
                r.raise_for_status()
                return r
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                # ComfyUI can briefly drop LAN connections while busy loading
                # nodes or receiving an upload. A short retry avoids surfacing
                # transient network hiccups as failed mattes.
                time.sleep(0.25 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _automatte_supports_input(self, name: str) -> bool:
        if self._automatte_inputs is None:
            try:
                r = requests.get(f"{self.base_url}/object_info", timeout=3)
                r.raise_for_status()
                info = r.json()
                required = info.get("ErmbgAutoMatte", {}).get("input", {}).get("required", {})
                optional = info.get("ErmbgAutoMatte", {}).get("input", {}).get("optional", {})
                self._automatte_inputs = set(required) | set(optional)
            except Exception:
                # object_info can be slow on plugin-heavy ComfyUI installs.
                # Prefer submitting the modern workflow over blocking every
                # remote matte; ComfyUI will return a specific validation error
                # if the deployed custom node is truly too old.
                self._automatte_inputs = {"shadow_mode"}
        return name in self._automatte_inputs

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
        matting_model: str,
        bg_color: tuple[int, int, int],
        despill: str | None,
        use_keyer: bool | None,
        shadow_mode: str,
        filename_prefix: str,
    ) -> dict[str, Any]:
        if use_keyer is True:
            keyer_value = "force_on"
        elif use_keyer is False:
            keyer_value = "force_off"
        else:
            keyer_value = "auto (router decides)"
        rendered = Template(json.dumps(self.workflow_template)).safe_substitute(
            input_image=input_image,
            matting_model=json.dumps(matting_model)[1:-1],
            bg_color=",".join(str(int(c)) for c in bg_color),
            despill=despill or "auto (router decides)",
            use_keyer=keyer_value,
            shadow_mode=shadow_mode,
            filename_prefix=json.dumps(filename_prefix)[1:-1],
        )
        workflow = json.loads(rendered)
        workflow.pop("_comment", None)
        return workflow

    def matte(
        self,
        image_srgb: np.ndarray,
        *,
        matting_model: str = "ZhengPeng7/BiRefNet-matting",
        bg_color: tuple[int, int, int] = (0, 200, 0),
        despill: str | None = None,
        use_keyer: bool | None = None,
        shadow_mode: str = "on",
    ) -> ComfyErmbgMatteResult:
        if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
            raise ValueError("matte() expects HxWx3 sRGB uint8")

        h, w = image_srgb.shape[:2]
        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        prefix = f"ermbg_full_{uuid.uuid4().hex[:8]}"
        step_start = time.perf_counter()
        server_name = self._upload(image_srgb, f"{prefix}.png")
        timings["upload_sec"] = time.perf_counter() - step_start
        workflow = self._render_workflow(
            input_image=server_name,
            matting_model=matting_model,
            bg_color=bg_color,
            despill=despill,
            use_keyer=use_keyer,
            shadow_mode=shadow_mode,
            filename_prefix=prefix,
        )
        if not self._automatte_supports_input("shadow_mode"):
            # Older deployed ERMBG Comfy nodes do not expose shadow_mode. Keep
            # the full remote execution path usable; the server will run its
            # default shadow behavior until the custom node is upgraded.
            workflow["20"]["inputs"].pop("shadow_mode", None)
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
        alpha_rgb = self._download_node_image(history, _ALPHA_NODE, "L")
        timings["download_alpha_sec"] = time.perf_counter() - step_start
        step_start = time.perf_counter()
        try:
            rgba_rgb = self._download_node_image(history, _RGBA_RGB_NODE, "RGB")
            timings["download_rgba_rgb_sec"] = time.perf_counter() - step_start
            rgba_rgb_source = _RGBA_RGB_NODE
        except RuntimeError:
            # Older deployed workflows saved only foreground+alpha. Keep the
            # client compatible, but modern ERMBG nodes separate clean
            # foreground RGB from the shadow-composited RGB used by final RGBA.
            rgba_rgb = foreground
            timings["download_rgba_rgb_sec"] = time.perf_counter() - step_start
            rgba_rgb_source = _FOREGROUND_NODE

        if foreground.shape[:2] != (h, w):
            foreground = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if rgba_rgb.shape[:2] != (h, w):
            rgba_rgb = cv2.resize(rgba_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if alpha_rgb.shape != (h, w):
            alpha_rgb = cv2.resize(alpha_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        alpha = alpha_rgb.astype(np.float32) / 255.0
        rgba = np.dstack([rgba_rgb, alpha_rgb])
        timings["total_sec"] = time.perf_counter() - total_start
        return ComfyErmbgMatteResult(
            rgba=rgba.astype(np.uint8),
            alpha=np.clip(alpha, 0.0, 1.0).astype(np.float32),
            foreground_srgb=foreground.astype(np.uint8),
            debug={
                "backend": "comfy-ermbg",
                "prompt_id": prompt_id,
                "server_image": server_name,
                "foreground_node": _FOREGROUND_NODE,
                "alpha_node": _ALPHA_NODE,
                "rgba_rgb_node": rgba_rgb_source,
                "timings": timings,
            },
        )


__all__ = ["ComfyErmbgMatteResult", "ComfyUIErmbgMatteClient"]
