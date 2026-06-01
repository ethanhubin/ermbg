# ERMBG ComfyUI 节点

本包为 ComfyUI 自定义图提供可选的 ERMBG 节点。Web/API 默认执行使用 Direct
Worker;这些节点是扩展支持。

## 节点

### `ERMBG Route Strategy`

分析图片特征并返回 route 元数据,但不执行 matting。可用于调试或自定义图的分支。

输出包括:

- `backend`
- `route`
- `asset_kind`
- `execution_profile`
- `json`

### `ERMBG Route Matte`

在 ComfyUI 的 Python 进程内运行 ERMBG 特征分析和 matting。它消费共享的
`execution_profile` 契约,并调用与 Direct Worker 路径相同的 ERMBG
PyMatting Known-B / CorridorKey 实现。

输出:

- `foreground`
- `alpha`
- `summary`
- `rgba_rgb`
- `aux`

### `ERMBG PyMatting Known-B`

对硬边 UI 和稳定已知背景图形运行确定性的已知背景 PyMatting。

### `ERMBG Classify`

轻量的诊断分类器预览。

### `Convert Masks to Images`

从 Comfy `MASK` 到图像预览的工具转换。

## 典型图

```text
Image -> ERMBG Route Matte -> foreground / alpha / rgba_rgb
```

当图需要显式分支时,`ERMBG Route Strategy` 很有用。
