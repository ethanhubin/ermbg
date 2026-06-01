# ERMBG ComfyUI Nodes

This package provides optional ERMBG nodes for ComfyUI custom graphs. Web/API
default execution uses Direct Worker; these nodes are extension support.

## Nodes

### `ERMBG Route Strategy`

Analyzes image features and returns route metadata without executing matting.
Use it for debugging or custom graph branching.

Outputs include:

- `backend`
- `route`
- `asset_kind`
- `execution_profile`
- `json`

### `ERMBG Route Matte`

Runs ERMBG feature analysis and matting inside the ComfyUI Python process.
It consumes the shared `execution_profile` contract and calls the same ERMBG
PyMatting Known-B / CorridorKey implementation used by the Direct Worker path.

Outputs:

- `foreground`
- `alpha`
- `summary`
- `rgba_rgb`
- `aux`

### `ERMBG PyMatting Known-B`

Runs deterministic known-background PyMatting for hard UI and stable
known-background graphics.

### `ERMBG Classify`

Lightweight diagnostic classifier preview.

### `Convert Masks to Images`

Utility conversion from Comfy `MASK` to image preview.

## Typical Graph

```text
Image -> ERMBG Route Matte -> foreground / alpha / rgba_rgb
```

`ERMBG Route Strategy` is useful when a graph needs explicit branching.
