# openclaw 集成

ERMBG 不是独立 skill,而是合并进 [openclaw](https://github.com/anthropics/openclaw) 已有的 `comfyui-rmbg` skill,作为 `--mode ermbg` 子模式存在。这样:

- 用户说"抠图 / 去背景 / remove background" → 走 `comfyui-rmbg` 的标准 RMBG 路径(快、够用)。
- 用户说"**智能抠图 / AI生图抠图 / smart matte / ERMBG**" → 走 `--mode ermbg`,触发 ERMBG RouteMatte 单节点管线。
- 用户说"magic wand / 只去同色边缘" → 走 `--mode edge-wand`(原有)。

一个 skill,多个触发词,意图精准命中。

## 调用

```bash
# AI 生图抠图(智能路由)
python3 ~/.openclaw/workspace/skills/comfyui-rmbg/scripts/comfyui_rmbg.py \
    --mode ermbg --image /path/to/in.png

# 等价的简写
python3 ~/.openclaw/workspace/skills/comfyui-rmbg/scripts/comfyui_rmbg.py \
    --smart --image /path/to/in.png
```

输出归档到 `~/.openclaw/media/openclaw-production/images/rmbg/<时间戳-uuid>/`,包含 `output.png` / `workflow.json` / `manifest.json` / `history_outputs.json`。`manifest.json` 里的 `mode: "ermbg"` 字段标识这次走的智能路径,`options` 里记录了路由参数。

## 服务器侧依赖

ComfyUI 服务器要先按 [DEPLOY.md](../../DEPLOY.md) 装好 ERMBG 节点。`--mode ermbg` 应提交 `ErmbgRouteMatte`;该节点在 Comfy 进程内完成 route、参数选择、CorridorKey / PyMatting Known-B / PyMatting fallback / passthrough 和 ShadowPatch。Auto 不再调用 RMBG fallback。

## 内部工作流

旧的 `ErmbgAutoMatte` 单节点工作流已移除。`--mode ermbg` 的工作流应是:

```
LoadImage → ERMBG Route Matte → foreground / alpha / rgba_rgb / metadata
```

## ermbg-matte 子模式选项

所有都是 `--mode ermbg` 才生效;默认全 `auto`,让 `ErmbgRouteMatte` 决策。

| 选项 | 默认 | 说明 |
|---|---|---|
| `--despill` | `auto` | `auto / unmix / chroma_cap / local_borrow / closed_form / none` |
| `--use-keyer` | `auto` | `auto / on / off` |
| `--bg-color` | `0,200,0` | R,G,B,重抠脏 RGBA 时的合成底色 |
| `--matting-model` | `ZhengPeng7/BiRefNet-matting` | HF 模型 ID |

## 自定义 ComfyUI 服务器

```bash
COMFY_URL=http://10.0.0.5:8188 python3 ... --mode ermbg --image ...
```

## 历史

最早是独立的 `ermbg-matte` skill。后来发现和 `comfyui-rmbg` 触发词冲突(都包含"抠图""去背景"),LLM 选谁不可控。合并后变成一个 skill 三个 mode,意图区分清楚。
