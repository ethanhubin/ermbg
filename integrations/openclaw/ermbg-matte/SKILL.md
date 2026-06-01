---
name: ermbg-matte
description: Smart background matting for AI-generated game assets through ERMBG. Use for 智能抠图, AI生图抠图, ERMBG, smart matte, game asset matting, clean alpha, green/blue screen assets, white-background UI graphics, translucent buttons, effect icons, character cutouts, or dirty RGBA re-matting. This is an independent ERMBG tool, not RMBG/rembg.
---

# ERMBG Matte

Use this skill when the user wants ERMBG's route-aware matting instead of a
generic RMBG/rembg background remover.

```bash
python3 skills/ermbg-matte/scripts/ermbg_matte.py --image /path/to/input.png
```

The script submits the remote ComfyUI `ErmbgRouteMatte` workflow and writes the
run under:

```text
/Users/ethanhu/.openclaw/media/openclaw-production/images/ermbg/
```

Each run archives:

- `output.png`: final transparent PNG
- `foreground.png`: recovered foreground RGB
- `alpha.png`: final alpha as an image
- `rgba_rgb.png`: RGB layer used with alpha
- `aux.png`: auxiliary debug preview
- `metadata.json`: ERMBG route/result metadata
- `workflow.json`: submitted ComfyUI API workflow
- `history_outputs.json`: raw Comfy history outputs
- `manifest.json`: local run manifest

## Rules

- Return the printed `OUTPUT /absolute/path/output.png` path.
- Use this skill for ERMBG / smart matte / AI-generated game asset matting.
- Do not route ERMBG requests through `comfyui-rmbg`; ERMBG is an independent
  tool with its own route/profile contract.
- Do not install models or modify the ComfyUI server from this skill. Missing
  `ErmbgRouteMatte` nodes are server deployment problems.
- For advanced options, run:

```bash
python3 skills/ermbg-matte/scripts/ermbg_matte.py --help
```
