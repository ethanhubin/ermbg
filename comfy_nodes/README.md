# ERMBG ComfyUI Nodes

ERMBG Comfy 节点现在提供统一 auto 路由抠图节点、独立路由策略节点和
已知背景硬边 matting 能力。旧的 `ERMBG AutoMatte` / BiRefNet 全量抠图
节点已移除。

## 节点

### `ERMBG Route Strategy`

只做路由决策,不执行抠图。用于调试或手工 graph 分支。

输出:

- `backend`: `passthrough` / `comfy-pymatting-known-b` / `comfy-corridorkey`
- `route`: `rgba_passthrough` / `pymatting_known_b` / `pymatting_fallback` / `corridorkey`
- `asset_kind`: `button` / `icon` / `character` / `known_bg_graphic` / `unknown_fallback`
- `json`: 完整 `RouteDecision` metadata

### `ERMBG Route Matte`

生产 auto 路径。Mac/Web 只上传输入图并提交这个单节点 workflow;节点在
ComfyUI 进程内完成 route、PyMatting Known-B、CorridorKey、PyMatting fallback、
shadow patch 和 metadata 输出,避免 Mac 侧后处理和嵌套 Comfy prompt。

输出:

- `foreground`: 抠图前景 RGB
- `alpha`: 最终 alpha
- `summary`: route/report/debug JSON
- `rgba_rgb`: 与 `alpha` 组合成最终 RGBA 的 RGB 层
- `aux`: 辅助预览图

### `ERMBG PyMatting Known-B`

已知纯色背景 PyMatting 节点,用于硬边按钮和确定性 UI。它不会跑 BiRefNet,
也不会提交嵌套 Comfy prompt;只在当前 ComfyUI Python 进程里用 PyMatting
解 trimap 的未知边界带,再用已知背景色反解边缘前景色。

### `ERMBG Classify (preview)`

保留的轻量诊断节点,用于查看旧 `Strategy` 分类结果。

### `Convert Masks to Images`

MASK 到 IMAGE 的小工具节点。

## 推荐工作流

```text
KSampler -> VAEDecode -> ERMBG Route Matte -> foreground/alpha/rgba_rgb
```

`ERMBG Route Strategy` 仍可用于调试和自定义 graph 分支;生产 Web/API 的
`backend="auto"` 统一提交 `ERMBG Route Matte`。
