# ERMBG

AI 生图质量已足够高,但直接生成透明背景仍不可靠。ERMBG 的解法是主动制造有利的抠图条件:引导 AI 将资产生成在纯色背景(绿幕/蓝幕)上,再精确去除背景,输出像素级干净、可直接送入引擎 UI / 动效 / 角色链路的 RGBA PNG。

核心思路:将背景从"需要猜测的未知量"变为"已知且可解的约束"。通用 matting 模型走的是反方向——它们不假设背景已知,只靠语义先验猜边界,遇到游戏素材的精细结构时会抹平或吃掉关键细节。ERMBG 因此围绕三条理念:

- **确定性优先,模型兜底**:能从已知条件直接推导的,不交给模型猜;只有在条件不足时才回退到通用方法。
- **先定归属,再求解**:图像中不同区域的性质各异,统一规则无法兼顾;先判断每个区域属于什么,再用对应的方式处理,避免错误归类。
- **机制驱动,而非样本驱动**:规则建立在可观测的信号和明确的失败模式上,而不是针对具体样本打补丁,确保对没见过的素材也能泛化。

## 素材覆盖

| 素材类型 | 执行路径 |
| --- | --- |
| 硬边按钮 / UI 面板 | PyMatting Known-B |
| 玻璃 / 半透明按钮 | CorridorKey |
| 图标 / shaped icon | CorridorKey |
| 特效图标(glow、烟雾、软 alpha) | CorridorKey |
| 角色(发丝、毛发、半透明边缘) | CorridorKey |
| 已有 RGBA | passthrough |
| 未知 / 不稳定背景 | PyMatting fallback |

## Execution Profile

图片特征识别在执行前自动确定 profile。

| 素材 / 场景 | Execution profile | 执行路径 |
| --- | --- | --- |
| clean RGBA | `passthrough` | passthrough |
| 硬边 UI / 确定性按钮 | `pymatting-hard-button` | PyMatting Known-B |
| 已知背景 fallback | `pymatting-known-bg` | PyMatting Known-B |
| 未知 / 不稳定背景 | `pymatting-fallback` | PyMatting fallback |
| shaped icon | `corridorkey-shaped-icon` | CorridorKey |
| effect icon | `corridorkey-effect-icon` | CorridorKey |
| 半透明 / 玻璃按钮 | `corridorkey-transparent-button` | CorridorKey |
| 角色 | `corridorkey-character` | CorridorKey |

## 输出

| 字段 | 说明 |
| --- | --- |
| `rgba` | 可直接使用的 RGBA PNG |
| `alpha` | float32 `[0, 1]` soft mask |
| `foreground_srgb` | sRGB foreground companion |
| `strategy_name` | 实际执行策略 |
| `background_color` | 诊断到的背景色 |
| `debug.auto_route` | 识别结果、asset kind、profile 和 backend 选择 |
| `server_elapsed_sec` | 服务端耗时 |

## 安装

需要 Python 3.12。

```bash
uv venv .venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -e ".[web,dev,torch]"
```

`torch` extra 用于 Direct Worker 的 CorridorKey 路径。

## 使用

### Web UI

```powershell
.\scripts\start_local.ps1
```

Web 默认选 `Auto Route`。手动下拉只选 algorithm(`CorridorKey`、`PyMatting Known-B`、`Known-B Glow`、`Passthrough`)。

### CLI

```bash
.venv/bin/ermbg matte input.png --backend auto --out-dir out/result
```

`--backend` 可选值:`auto`、`pymatting-known-b`、`corridorkey`、`known_bg_glow`、`passthrough`。

### Python API

```python
from pathlib import Path
from PIL import Image
from ermbg.api import matte_image
from ermbg.io import save_rgba, save_mask

result = matte_image(
    Image.open("input.png").convert("RGBA"),
    backend="auto",
    output_dir=Path("out/result"),
)
save_rgba("out/result/output.png", result.rgba)
save_mask("out/result/alpha.png", result.alpha)
print(result.debug)
```

重要 metadata 字段:`execution_profile`、`parameter_profile`、`debug.auto_route.algorithm`、`server_elapsed_sec`。Direct Worker 还会返回 `debug.direct_worker.execution_backend`(如 `direct-corridorkey`、`direct-pymatting-known-b`)。

### Game Eval

批量回归验证,输出写入 `out/` 下的 batch 目录并生成 `summary.json`。

```bash
.venv/bin/python scripts/run_corridorkey_game_eval.py \
  --backend auto \
  --sample-id B001,I011,C001 \
  --out-dir out/smoke
```

去掉 `--sample-id` 跑完整 85 样本集。回归样本集:`samples/corridorkey_semantic/manifest.json`。

## 部署

配置写在 `ermbg.config.json`(共享默认值)和 gitignored `ermbg.local.json`(机器相关覆盖)。`services.direct_worker_urls` 是 Direct Worker URL 优先级列表,支持自动 fallback。

远端 Direct Worker:

```bash
scripts/sync_comfy_ssh.sh --smoke
scripts/restart_direct_worker_ssh.sh --restart
curl -sS "http://192.168.0.8:7871/health"
```

本机 Web + 远端 Worker:

```powershell
$env:ERMBG_DIRECT_URL = "<services.direct_worker_url>"
.\scripts\start_local.ps1 -SkipDirectWorker -DirectUrl $env:ERMBG_DIRECT_URL
```

详见 [DEPLOY.md](DEPLOY.md)。

## 验证

```bash
# 单元测试
.venv/bin/pytest -q

# Direct Worker HTTP smoke
.venv/bin/python scripts/smoke_direct_worker_http.py \
  --base-url <services.direct_worker_url> \
  --sample-id B001,I011

# Runtime capabilities
curl -sS "<web-url>/api/runtime-capabilities?include_comfy=false&include_object_info=false"
```

## 项目地图

| 路径 | 角色 |
| --- | --- |
| `ermbg/router.py` | route 决策、asset kind、execution profile |
| `ermbg/api.py` | 主 matting API 和 PyMatting Known-B 实现 |
| `ermbg/corridorkey_runner.py` | 进程内 CorridorKey runner |
| `ermbg/direct_worker.py` | direct 执行编排 |
| `ermbg/direct_worker_client.py` | Direct Worker HTTP client |
| `ermbg/direct_worker_server.py` | 远端 Direct Worker FastAPI 服务 |
| `ermbg/web.py` | Web UI、Web API、Game Eval |
| `scripts/run_corridorkey_game_eval.py` | 批量 eval |
| `samples/corridorkey_semantic/` | B/I/C 游戏素材样本集 |
| `out/` | eval batch、summary、debug 产物 |

## 扩展

- **ComfyUI 节点** — `comfy_nodes/`，提供 `ERMBG Route Matte`、`ERMBG Classify` 等节点，详见 `comfy_nodes/README.md`。
- **OpenClaw 适配器** — 可选独立 `ermbg-matte` skill，详见 `integrations/openclaw/README.md`。

## 文档

- [docs/architecture.md](docs/architecture.md) — 架构与服务边界
- [docs/ermbg-route-strategy.md](docs/ermbg-route-strategy.md) — route / profile / backend 契约
- [docs/install-startup.md](docs/install-startup.md) — 完整安装与启动流程
