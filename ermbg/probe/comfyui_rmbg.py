"""Run a ComfyUI rembg workflow as a baseline matter.

Treats the remote rembg node (isnet-general-use, equivalent to RMBG-1.4) as a
black-box matting baseline: it returns an RGBA PNG, no despill / no QA. Used
purely for evaluation comparisons against ours.
"""

from __future__ import annotations

import json
import time
import uuid
from io import BytesIO
from pathlib import Path
from string import Template

import numpy as np
import requests
from PIL import Image


_DEFAULT_WORKFLOW = Path(__file__).parent / "comfyui_rmbg.json"


class ComfyUIRembgBaseline:
    """Calls ComfyUI's `Image Rembg` node and returns the RGBA result.

    This is intentionally not a `ProbeGenerator` — it produces a finished RGBA,
    not a probe image. Lives next to the probe backends because it shares the
    HTTP client pattern.
    """

    def __init__(
        self,
        url: str = "http://192.168.0.8:8000",
        workflow_path: Path | str | None = None,
        timeout: float = 600.0,
        poll_interval: float = 1.0,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.timeout = timeout
        self.poll_interval = poll_interval
        path = Path(workflow_path) if workflow_path else _DEFAULT_WORKFLOW
        self.workflow_template = json.loads(path.read_text())

    # --- HTTP helpers -----------------------------------------------------

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

    def _queue(self, workflow: dict) -> str:
        body = {"prompt": workflow, "client_id": self.client_id}
        result = self._post("/prompt", json=body).json()
        if "prompt_id" not in result:
            raise RuntimeError(f"Comfy /prompt rejected: {result}")
        return result["prompt_id"]

    def _wait(self, prompt_id: str) -> dict:
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
        raise TimeoutError(f"prompt {prompt_id} did not finish in {self.timeout}s")

    def _download(self, history_entry: dict) -> np.ndarray:
        for node_out in history_entry.get("outputs", {}).values():
            for img_meta in node_out.get("images", []):
                params = {
                    "filename": img_meta["filename"],
                    "subfolder": img_meta.get("subfolder", ""),
                    "type": img_meta.get("type", "output"),
                }
                r = self._get("/view", params=params)
                im = Image.open(BytesIO(r.content)).convert("RGBA")
                return np.asarray(im, dtype=np.uint8)
        raise RuntimeError("No output image in ComfyUI history")

    # --- public API -------------------------------------------------------

    def matte(self, image_srgb: np.ndarray) -> np.ndarray:
        """Run rembg on the image, return H×W×4 sRGB+alpha uint8."""
        if image_srgb.dtype != np.uint8 or image_srgb.ndim != 3 or image_srgb.shape[2] != 3:
            raise ValueError("matte() expects HxWx3 sRGB uint8")

        upload_name = f"ermbg_rmbg_{uuid.uuid4().hex[:8]}.png"
        server_name = self._upload(image_srgb, upload_name)

        rendered = json.dumps(self.workflow_template)
        rendered = Template(rendered).safe_substitute(input_image=server_name)
        wf = json.loads(rendered)
        wf.pop("_comment", None)

        prompt_id = self._queue(wf)
        entry = self._wait(prompt_id)
        rgba = self._download(entry)

        # Comfy may resize for the rembg model; force back to original H×W.
        h, w = image_srgb.shape[:2]
        if rgba.shape[:2] != (h, w):
            import cv2
            rgba = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return rgba


__all__ = ["ComfyUIRembgBaseline"]
