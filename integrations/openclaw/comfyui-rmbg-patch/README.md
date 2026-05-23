# comfyui-rmbg 合并补丁

这里是把 `--mode ermbg` 合并进 [openclaw](https://github.com/anthropics/openclaw) `comfyui-rmbg` skill 时的快照,用于复现合并(如果 openclaw 的 skill 仓库被重置/换机器)。

## 文件

- `SKILL.md.merged` — 合并后的 skill 描述,带"智能抠图 / AI生图抠图 / smart matte / ERMBG"触发词
- `comfyui_rmbg.py.merged` — 合并后的 runner 脚本

## 应用方式

直接覆盖目标 skill 目录:

```bash
cp SKILL.md.merged ~/.openclaw/workspace/skills/comfyui-rmbg/SKILL.md
cp comfyui_rmbg.py.merged ~/.openclaw/workspace/skills/comfyui-rmbg/scripts/comfyui_rmbg.py
chmod +x ~/.openclaw/workspace/skills/comfyui-rmbg/scripts/comfyui_rmbg.py
```

## 改了哪些(增量)

**新函数**:`ermbg_prompt(uploaded, prefix, despill, use_keyer, bg_color, matting_model)` — 构造 LoadImage → ErmbgAutoMatte → InvertMask → SaveImageWithAlpha 工作流。

**新 CLI 选项**:

```python
ap.add_argument("--mode", choices=["rmbg", "edge-wand", "ermbg"], default="rmbg", ...)
ap.add_argument("--smart", action="store_true", help="Shortcut for --mode ermbg")
ap.add_argument("--despill", default="auto", choices=[...])
ap.add_argument("--use-keyer", default="auto", choices=["auto", "on", "off"])
ap.add_argument("--bg-color", default="0,200,0")
ap.add_argument("--matting-model", default="ZhengPeng7/BiRefNet-matting")
```

**新 main() 分支**:`--mode ermbg` 走专用路径,先校验 `ErmbgAutoMatte` 节点存在,再 upload + 提交工作流 + 下载产物 + 写 manifest。

## 升级流程

如果 ERMBG 自身改动需要 skill 配合(比如改了节点签名),改 `comfyui_rmbg.py.merged`,提交,然后再 `cp` 到 openclaw 那边。
