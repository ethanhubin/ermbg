"""ComfyUI prompt-aware subject-mask workflow renderer.

This module prepares the next-stage workflow without requiring the remote
ComfyUI server to be idle. The rendered workflow can be inspected, saved, and
later submitted through the same HTTP pattern used by the probe generators.
"""

from __future__ import annotations

import json
import time
import uuid
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import requests
from PIL import Image

from ermbg.comfy import DEFAULT_COMFY_URL

_DEFAULT_WORKFLOW = Path(__file__).parent / "comfyui_clipseg_ermbg.json"
_DEFAULT_OUTPUT_NAMES = {
    "40": "foreground",
    "60": "alpha",
    "80": "subject_mask",
}


def render_clipseg_ermbg_workflow(
    *,
    input_image: str,
    subject_prompt: str,
    filename_prefix: str = "ermbg_subject",
    clipseg_model: str = "CIDAS/clipseg-rd64-refined",
    matting_model: str = "ZhengPeng7/BiRefNet-matting",
    bg_color: str = "0,200,0",
    workflow_path: Path | str | None = None,
) -> dict[str, Any]:
    """Render the CLIPSeg -> ERMBG AutoMatte workflow template.

    ``input_image`` is the server-side filename returned by ComfyUI
    ``/upload/image``. For dry-run inspection, it can be any placeholder.
    """
    path = Path(workflow_path) if workflow_path else _DEFAULT_WORKFLOW
    template = json.loads(path.read_text())
    rendered = Template(json.dumps(template)).safe_substitute(
        input_image=input_image,
        subject_prompt=json.dumps(subject_prompt)[1:-1],
        filename_prefix=json.dumps(filename_prefix)[1:-1],
        clipseg_model=json.dumps(clipseg_model)[1:-1],
        matting_model=json.dumps(matting_model)[1:-1],
        bg_color=json.dumps(bg_color)[1:-1],
    )
    workflow = json.loads(rendered)
    workflow.pop("_comment", None)
    return workflow


class ComfyUISubjectMaskWorkflow:
    """Submit the CLIPSeg -> ERMBG workflow when the server is available."""

    def __init__(
        self,
        url: str = DEFAULT_COMFY_URL,
        timeout: float = 600.0,
        poll_interval: float = 1.0,
    ) -> None:
        self.base_url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _post(self, path: str, **kwargs):
        r = requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def _get(self, path: str, **kwargs):
        r = requests.get(f"{self.base_url}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r

    def upload_image(self, image: np.ndarray, name: str | None = None) -> str:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("upload_image expects HxWx3 uint8 sRGB")
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        buf.seek(0)
        upload_name = name or f"ermbg_subject_{uuid.uuid4().hex[:8]}.png"
        files = {"image": (upload_name, buf, "image/png")}
        data = {"overwrite": "true"}
        return self._post("/upload/image", files=files, data=data).json()["name"]

    def queue(self, workflow: dict[str, Any]) -> str:
        body = {"prompt": workflow, "client_id": self.client_id}
        result = self._post("/prompt", json=body).json()
        if "prompt_id" not in result:
            raise RuntimeError(f"Comfy /prompt rejected: {result}")
        return result["prompt_id"]

    def wait(self, prompt_id: str) -> dict[str, Any]:
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

    def history_images(self, history_entry: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten ComfyUI history image metadata and keep the producing node id."""
        images: list[dict[str, Any]] = []
        for node_id, node_out in history_entry.get("outputs", {}).items():
            for index, img_meta in enumerate(node_out.get("images", [])):
                images.append(
                    {
                        "node_id": str(node_id),
                        "index": index,
                        "filename": img_meta["filename"],
                        "subfolder": img_meta.get("subfolder", ""),
                        "type": img_meta.get("type", "output"),
                    }
                )
        return images

    def download_images(
        self,
        history_entry: dict[str, Any],
        out_dir: Path | str,
        output_names: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Download all image outputs from a completed subject-mask workflow.

        Files are named by the known SaveImage node roles when possible:
        ``foreground.png``, ``alpha.png``, and ``subject_mask.png``. Unknown
        outputs keep a stable ``node_<id>_<index>.png`` name.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        names = output_names or _DEFAULT_OUTPUT_NAMES

        records: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for meta in self.history_images(history_entry):
            params = {
                "filename": meta["filename"],
                "subfolder": meta.get("subfolder", ""),
                "type": meta.get("type", "output"),
            }
            r = self._get("/view", params=params)
            image = Image.open(BytesIO(r.content))

            base = names.get(meta["node_id"], f"node_{meta['node_id']}_{meta['index']}")
            seen[base] = seen.get(base, 0) + 1
            suffix = "" if seen[base] == 1 else f"_{seen[base]}"
            local_path = out_path / f"{base}{suffix}.png"
            image.save(local_path)

            records.append(
                {
                    **meta,
                    "role": names.get(meta["node_id"]),
                    "local_path": str(local_path),
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                }
            )
        if not records:
            raise RuntimeError("No output images found in ComfyUI history")
        return records

    def run(
        self,
        image: np.ndarray,
        *,
        subject_prompt: str,
        filename_prefix: str = "ermbg_subject",
        clipseg_model: str = "CIDAS/clipseg-rd64-refined",
        matting_model: str = "ZhengPeng7/BiRefNet-matting",
        bg_color: str = "0,200,0",
        upload_name: str | None = None,
        download_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Upload, queue, wait, and optionally download the full workflow."""
        server_image = self.upload_image(image, name=upload_name)
        workflow = render_clipseg_ermbg_workflow(
            input_image=server_image,
            subject_prompt=subject_prompt,
            filename_prefix=filename_prefix,
            clipseg_model=clipseg_model,
            matting_model=matting_model,
            bg_color=bg_color,
        )
        prompt_id = self.queue(workflow)
        history = self.wait(prompt_id)
        downloads = self.download_images(history, download_dir) if download_dir is not None else []
        return {
            "server_image": server_image,
            "prompt_id": prompt_id,
            "workflow": workflow,
            "history": history,
            "downloads": downloads,
        }


__all__ = ["ComfyUISubjectMaskWorkflow", "render_clipseg_ermbg_workflow"]
