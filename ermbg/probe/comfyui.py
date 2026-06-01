"""ComfyUI-backed probe generator.

Talks HTTP to a remote ComfyUI server: uploads the input image, submits a
parameterized workflow (Qwen-Image-Edit 2511 by default), polls /history, and
fetches the resulting image. Default workflow does instruction-based background
replacement, which is well-matched to the 'replace background but keep subject'
constraint without needing a mask, IP-Adapter, or ControlNet.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from string import Template

import numpy as np
import requests
from loguru import logger
from PIL import Image

from ermbg.comfy import DEFAULT_COMFY_URL

from .generator import ProbeGenerator
from .prompts import color_phrase


_DEFAULT_WORKFLOW = Path(__file__).parent / "comfyui_qwen_edit.json"


_COLOR_BG_PHRASE: dict[tuple[int, int, int], str] = {
    (250, 250, 250): "pure white",
    (8, 8, 8): "pure black",
    (0, 200, 220): "saturated cyan",
    (220, 30, 180): "saturated magenta",
    (0, 200, 60): "saturated green",
}


def _instruction(color: tuple[int, int, int], object_prompt: str | None = None) -> tuple[str, str]:
    """Build a Qwen-Image-Edit instruction.

    Strategy: short, declarative, leaves no room for interpretation. Tested
    longer prompts on 2026-05 and the model treats them as creative direction.
    """
    key = tuple(int(c) for c in color)
    bg = _COLOR_BG_PHRASE.get(key, f"solid color rgb{key}")
    pos = f"Change the background to a {bg} studio backdrop. Background must be one uniform color, no clouds, no fog, no gradient, no shadow, no texture."
    neg = "fog, clouds, mist, gradient, shadow, vignette, texture, pattern, props, blurry"
    return pos, neg


class ComfyUIProbeGenerator(ProbeGenerator):
    name = "comfyui"

    def __init__(
        self,
        url: str = DEFAULT_COMFY_URL,
        workflow_path: Path | str | None = None,
        timeout: float = 600.0,
        poll_interval: float = 1.0,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.timeout = timeout
        self.poll_interval = poll_interval
        path = Path(workflow_path) if workflow_path else _DEFAULT_WORKFLOW
        self.workflow_template = json.loads(path.read_text(encoding="utf-8"))

    # --- HTTP helpers ------------------------------------------------------

    def _post(self, path: str, **kwargs):
        r = requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def _get(self, path: str, **kwargs):
        r = requests.get(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def _upload_image(self, image: np.ndarray, name: str) -> str:
        """Upload via /upload/image. Returns the server-side filename."""
        from io import BytesIO
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        buf.seek(0)
        files = {"image": (name, buf, "image/png")}
        data = {"overwrite": "true"}
        resp = self._post("/upload/image", files=files, data=data)
        info = resp.json()
        return info["name"]

    def _queue(self, workflow: dict) -> str:
        body = {"prompt": workflow, "client_id": self.client_id}
        resp = self._post("/prompt", json=body)
        result = resp.json()
        if "prompt_id" not in result:
            raise RuntimeError(f"Comfy /prompt rejected workflow: {result}")
        return result["prompt_id"]

    def _wait_for_completion(self, prompt_id: str) -> dict:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            resp = self._get(f"/history/{prompt_id}")
            data = resp.json()
            if prompt_id in data:
                entry = data[prompt_id]
                status = entry.get("status", {})
                if status.get("completed", False):
                    return entry
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Comfy workflow errored: {entry.get('status')}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {self.timeout}s")

    def _download_first_image(self, history_entry: dict) -> np.ndarray:
        outputs = history_entry.get("outputs", {})
        for node_out in outputs.values():
            for img_meta in node_out.get("images", []):
                params = {
                    "filename": img_meta["filename"],
                    "subfolder": img_meta.get("subfolder", ""),
                    "type": img_meta.get("type", "output"),
                }
                r = self._get("/view", params=params)
                from io import BytesIO
                im = Image.open(BytesIO(r.content)).convert("RGB")
                return np.asarray(im, dtype=np.uint8)
        raise RuntimeError("No output images found in ComfyUI history")

    # --- Workflow filling --------------------------------------------------

    def _build_workflow(
        self,
        input_filename: str,
        width: int,
        height: int,
        positive: str,
        negative: str,
        seed: int,
    ) -> dict:
        rendered = json.dumps(self.workflow_template)
        rendered = Template(rendered).safe_substitute(
            input_image=input_filename,
            width=width,
            height=height,
            prompt=json.dumps(positive)[1:-1],          # escape for JSON
            negative_prompt=json.dumps(negative)[1:-1],
            seed=seed,
        )
        wf = json.loads(rendered)
        wf.pop("_comment", None)
        # Convert width/height/seed back to int (Template substitutes as strings).
        for node_id, node in wf.items():
            for k in ("width", "height", "seed"):
                if k in node.get("inputs", {}) and isinstance(node["inputs"][k], str):
                    node["inputs"][k] = int(node["inputs"][k])
        return wf

    # --- Public API --------------------------------------------------------

    def generate(
        self,
        image: np.ndarray,
        subject_mask: np.ndarray,
        background_color: tuple[int, int, int],
        seed: int | None = None,
        object_prompt: str | None = None,
    ) -> np.ndarray:
        del subject_mask  # Qwen-Image-Edit edits via instruction, not via mask
        if image.dtype != np.uint8:
            raise ValueError("ComfyUI generator expects uint8 sRGB input.")

        h, w = image.shape[:2]
        # Cap long edge to 1280 to keep VRAM happy and round to 16.
        max_long = 1280
        scale = min(1.0, max_long / max(h, w))
        new_w = int(round(w * scale / 16) * 16)
        new_h = int(round(h * scale / 16) * 16)
        if (new_w, new_h) != (w, h):
            import cv2
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        positive, negative = _instruction(background_color, object_prompt=object_prompt)

        upload_name = f"ermbg_{uuid.uuid4().hex[:8]}.png"
        server_name = self._upload_image(image, upload_name)

        wf = self._build_workflow(
            input_filename=server_name,
            width=new_w,
            height=new_h,
            positive=positive,
            negative=negative,
            seed=int(seed if seed is not None else 42),
        )

        logger.debug(f"Submitting ComfyUI workflow ({new_w}x{new_h}) prompt='{positive[:80]}...'")
        prompt_id = self._queue(wf)
        entry = self._wait_for_completion(prompt_id)
        out = self._download_first_image(entry)

        # Resize back to original input HxW so the rest of the pipeline lines up.
        if out.shape[:2] != (h, w):
            import cv2
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return out
