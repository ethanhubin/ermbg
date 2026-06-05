# OpenClaw 集成

ERMBG 在 OpenClaw 里是独立 skill: `ermbg-matte`。

OpenClaw 不是 ERMBG 主线。当前主线是 Web/API/CLI 走 Direct Worker。这里保留的
是一个可选外围适配器,用于需要 OpenClaw 调用时复用同一条 ERMBG route/matte
合约。

它不是通用 RMBG/rembg 的子模式,也不复用 RMBG/rembg 的意图入口。这样做的
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
- `manifest.json`: 本地运行 manifest

## 服务器侧依赖

OpenClaw skill 应调用维护中的 ERMBG Web/API 或 Direct Worker 服务。服务端完成
route、参数选择、CorridorKey / PyMatting Known-B / PyMatting fallback /
passthrough 和 ShadowPatch。

## 内部工作流

```text
input image -> ERMBG API -> foreground / alpha / rgba_rgb / aux / metadata
```

OpenClaw skill 只负责上传输入、提交请求、下载输出和写归档。它不应该复制
ERMBG router 的判断逻辑。

## 自定义 ERMBG 服务

```bash
ERMBG_DIRECT_URL=http://10.0.0.5:7871 \
python3 ~/.openclaw/workspace/skills/ermbg-matte/scripts/ermbg_matte.py \
    --image /path/to/in.png
```
