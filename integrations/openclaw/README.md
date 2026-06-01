# OpenClaw 集成

ERMBG 在 OpenClaw 里是独立 skill: `ermbg-matte`。

OpenClaw 不是 ERMBG 主线。当前主线是 Web/API/CLI 走 Direct Worker,远端 ComfyUI
`ErmbgRouteMatte` 是可选执行路径。这里保留的是一个可选外围适配器,用于需要
OpenClaw 调用时复用同一条 RouteMatte 合约。

它不是 `comfyui-rmbg` 的子模式,也不复用 RMBG/rembg 的意图入口。这样做的
目标是让 agent 明确区分:

- 普通 RMBG/rembg: 通用语义去背景。
- ERMBG: 面向 AI 生成游戏素材的 route-aware 智能抠图。

## 安装

```bash
scripts/install_openclaw_ermbg_skill.sh
```

默认安装到:

```text
~/.openclaw/workspace/skills/ermbg-matte/
```

## 调用

```bash
python3 ~/.openclaw/workspace/skills/ermbg-matte/scripts/ermbg_matte.py \
    --image /path/to/in.png
```

输出归档到:

```text
~/.openclaw/media/openclaw-production/images/ermbg/<日期>/<时间戳-输入名>/
```

每次运行包含:

- `output.png`: 最终透明 PNG
- `foreground.png`: 前景 RGB
- `alpha.png`: 最终 alpha 图
- `rgba_rgb.png`: 与 alpha 组合的 RGB 层
- `aux.png`: 辅助预览
- `metadata.json`: ERMBG route/result metadata
- `workflow.json`: 提交给 ComfyUI 的 API workflow
- `history_outputs.json`: Comfy history outputs
- `manifest.json`: 本地运行 manifest

## 服务器侧依赖

ComfyUI 服务器要先按 [DEPLOY.md](../../DEPLOY.md) 装好 ERMBG 节点。
`ermbg-matte` 提交的是 `ErmbgRouteMatte`;该节点在 Comfy 进程内完成 route、
参数选择、CorridorKey / PyMatting Known-B / PyMatting fallback / passthrough
和 ShadowPatch。

## 内部工作流

```text
LoadImage -> ERMBG Route Matte -> foreground / alpha / rgba_rgb / aux / metadata
```

OpenClaw skill 只负责上传输入、提交 workflow、下载输出和写归档。它不应该
复制 ERMBG router 的判断逻辑。

## 自定义 ComfyUI 服务器

```bash
COMFY_URL=http://10.0.0.5:8188 \
python3 ~/.openclaw/workspace/skills/ermbg-matte/scripts/ermbg_matte.py \
    --image /path/to/in.png
```
