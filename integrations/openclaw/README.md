# openclaw 集成

ERMBG 作为 [openclaw](https://github.com/anthropics/openclaw) bot 的 skill,通过局域网 ComfyUI 服务器提供"AI 出图 → 智能抠图"链路。

## 安装

```bash
cp -r integrations/openclaw/ermbg-matte ~/.openclaw/workspace/skills/
```

或软链(开发期推荐):

```bash
ln -s "$(pwd)/integrations/openclaw/ermbg-matte" ~/.openclaw/workspace/skills/ermbg-matte
```

## 使用

bot 会按照 skill 的 `SKILL.md` 描述匹配抠图请求("抠图" / "去背景" / "transparent PNG" 等)。也可以直接命令行调:

```bash
python3 ~/.openclaw/workspace/skills/ermbg-matte/scripts/ermbg_matte.py \
    --image /path/to/in.png
```

输出归档到 `~/.openclaw/media/openclaw-production/images/ermbg/<时间戳-uuid>/`,包含:

| 文件 | 内容 |
|---|---|
| `output.png` | 抠出的 RGBA |
| `workflow.json` | 提交给 ComfyUI 的 API workflow,可在 ComfyUI 里 Load 复现 |
| `manifest.json` | input/output 路径、prompt_id、router 选的策略、所有 CLI 选项 |
| `history.json` | 完整 ComfyUI history 输出 |

stdout 同时打印 `PROMPT_ID xxx` / `OUTPUT /abs/path` / `SUMMARY <策略名> | despill=... | <说明>`。

## 自定义 ComfyUI 服务器地址

```bash
COMFY_URL=http://10.0.0.5:8188 python3 ermbg_matte.py --image ...
```

## 参数

| 选项 | 默认 | 含义 |
|---|---|---|
| `--image` | required | 输入路径 |
| `--output-dir` | 自动归档 | 覆盖输出目录 |
| `--despill` | `auto` | `auto / unmix / chroma_cap / local_borrow / closed_form / none` |
| `--use-keyer` | `auto` | `auto / on / off` |
| `--bg-color` | `0,200,0` | 重抠脏 RGBA 时的合成底色,R,G,B |
| `--matting-model` | `ZhengPeng7/BiRefNet-matting` | HF 模型 ID |
| `--timeout` | 300 | 等待 ComfyUI 输出秒数 |

## 服务器侧依赖

ComfyUI 服务器要先按 [DEPLOY.md](../../DEPLOY.md) 装好 ERMBG 节点。skill 不会自己装,只调 HTTP API。

## skill 文件结构

```
ermbg-matte/
  SKILL.md                 — 给 LLM 看的 skill 说明 (frontmatter + 文档)
  agents/openai.yaml       — bot UI 显示信息
  scripts/
    ermbg_matte.py         — 主 runner
    workflow.template.json — ComfyUI API workflow 模板
```

模仿 `~/.openclaw/workspace/skills/comfyui-rmbg/` 的结构,bot 自动扫描该目录注册 skill。
