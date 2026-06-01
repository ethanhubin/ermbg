# ERMBG

ERMBG 是面向 AI 生成游戏素材的智能抠图系统。
系统会分析图片特征、背景颜色、边缘结构、透明材质、阴影和素材语义,自动匹配合适的抠图算法与参数。
针对游戏 UI 素材,ERMBG 基于 PyMatting Known-B 进行像素级修复,利用已知背景色、边缘证据和重投影误差处理硬边、细边、孔洞、阴影和半透明区域。
针对复杂绿色/蓝色背景素材,ERMBG 借鉴影视行业多年绿幕抠图经验沉淀,使用 CorridorKey 处理角色、玻璃、发光、烟雾、毛发和软 alpha 边缘。

设计目标是在纯色背景上达到像素级完美的透明抠图,输出可直接进入游戏 UI、动效和角色素材生产链路的 RGBA PNG。

## Design

- **游戏 UI 像素级修复:** 基于 PyMatting Known-B,以已知背景色为强约束,结合颜色距离、边缘证据和重投影误差修复硬边、细边、孔洞和阴影。
- **图片特征识别:** 自动识别按钮、图标、特效、角色、玻璃/半透明材质、已有 alpha 和未知背景 fallback。
- **算法与参数匹配:** 根据识别结果选择 PyMatting Known-B、CorridorKey、passthrough 或 fallback,并生成对应执行参数。
- **影视级绿/蓝幕抠图:** 对复杂绿幕/蓝幕素材使用 CorridorKey,处理发丝、毛发、半透明、发光、烟雾和玻璃材质。
- **批量可验证:** Web run、CLI/API run 和 Game Eval 都写入 `out/` batch/artifact,保留 PNG 输出和机器可读 manifest。

## Game Asset Coverage

ERMBG 已完成面向游戏素材的类型适配:

| Asset type | Coverage |
| --- | --- |
| 硬边按钮 / UI 面板 | 纯色背景检测、硬边 alpha、孔洞归属、阴影保留 |
| 玻璃 / 半透明按钮 | 绿/蓝幕 CorridorKey、半透明主体保护、颜色保护 |
| 图标 / shaped icon | 形状 hint、key-color 材质保护、透明边缘 |
| 特效图标 | glow、烟雾、软 alpha、发光层 |
| 角色素材 | 1024 角色图、发丝、毛发、半透明边缘、glow |
| 已有 RGBA | alpha 质量检查和 passthrough |
| 未知 / 不稳定背景 | PyMatting fallback 和诊断 metadata |

## Technical Path

```text
Web/API backend=auto
  -> configured Direct Worker HTTP service
  -> image feature analysis
  -> matched algorithm + execution parameters
  -> direct PyMatting Known-B / direct CorridorKey / passthrough
  -> foreground + alpha + rgba_rgb + metadata
```

主要模块:

- `ermbg/router.py`: 图片特征识别、asset kind、`parameter_profile`、`execution_profile`
- `ermbg/api.py`: 高层 matting API 和 PyMatting Known-B 实现
- `ermbg/corridorkey_runner.py`: 共享 CorridorKey runner
- `ermbg/direct_worker_server.py`: Direct Worker HTTP 服务
- `ermbg/web.py`: Web UI、Web API、Game Eval 页面

## Output

ERMBG 输出透明 PNG,并保留独立 alpha、前景 RGB、识别/debug metadata 和 artifact manifest。典型输出字段包括:

- `rgba`: 可直接下载使用的 RGBA PNG
- `alpha`: float32 `[0, 1]` soft mask
- `foreground_srgb`: sRGB foreground companion
- `strategy_name`: 实际执行策略
- `background_color`: 诊断到的背景色
- `debug.auto_route`: 识别结果、asset kind、profile 和后端选择
- `server_elapsed_sec`: 服务端耗时

回归样本集位于 `samples/corridorkey_semantic/manifest.json`,覆盖 button、icon/effect 和 character 三类游戏素材。

## Execution Profiles

图片特征识别会在执行前确定最终 profile。`parameter_profile` 是分析元数据,用于解释系统匹配算法的依据；`execution_profile` 是执行契约,用于控制 hint mode、mask prior、color protection、refiner、despeckle 和 debug metadata。

| Asset / case                      | Execution profile                | Execution path                  |
| --------------------------------- | -------------------------------- | ------------------------------- |
| clean RGBA                        | passthrough                      | passthrough                     |
| hard UI / deterministic button    | `pymatting-hard-button`          | PyMatting Known-B               |
| known-background graphic fallback | `pymatting-known-bg`             | PyMatting Known-B               |
| unknown / unstable background     | `pymatting-fallback`             | PyMatting fallback              |
| shaped icon                       | `corridorkey-shaped-icon`        | CorridorKey                     |
| effect icon                       | `corridorkey-effect-icon`        | CorridorKey                     |
| translucent / glass button        | `corridorkey-transparent-button` | CorridorKey                     |
| character                         | `corridorkey-character`          | CorridorKey                     |

## Install

项目推荐使用 `.venv/` 和 Python 3.12。

```bash
cd <ermbg-root>
uv venv .venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -e ".[web,dev,torch]"
```

`torch` extra 用于 Direct Worker 的 CorridorKey 路径。只跑 PyMatting smoke 时
可以不装,但默认游戏素材主流程建议安装。

## Deploy

ERMBG Web 通过 `ermbg.config.json` 的 `services.direct_worker_url` 连接 Direct Worker。Direct Worker 可以安装在本机,也可以安装在远端服务器。环境变量 `ERMBG_DIRECT_URL` 可用于临时覆盖该配置。

本机单机部署:

```powershell
.\scripts\start_local.ps1
```

远端 Direct Worker:

```powershell
# worker server
.\.venv\Scripts\python.exe -m ermbg.direct_worker_server --host 0.0.0.0 --port 7871 --cpu-workers 4

# local web
.\scripts\start_local.ps1 -SkipDirectWorker -DirectUrl <services.direct_worker_url>
```

等价环境变量:

```powershell
$env:ERMBG_DIRECT_URL = "<services.direct_worker_url>"
$env:ERMBG_WEB_AUTO_BACKEND = "direct-worker"
.\.venv\Scripts\python.exe -m uvicorn ermbg.web:app --host 127.0.0.1 --port 7860
```

默认 Direct Worker 地址写在 `ermbg.config.json` 的 `services.direct_worker_url`。

## Use

### Web UI

```powershell
.\scripts\start_local.ps1
```

打开:

```text
<web-url>
```

Web 里默认选择 `Auto Direct Worker`。

### CLI Matte

```bash
.venv/bin/ermbg matte input.png --backend auto --out-dir out/manual_matte
```

CLI 支持的主要后端:

- `auto`
- `pymatting-known-b`

### Game Eval

Game Eval 是回归验证入口。所有输出写入 `out/` 下的 batch 目录,并生成 `summary.json`。

```bash
# Auto production path
.venv/bin/python scripts/run_corridorkey_game_eval.py \
  --backend auto \
  --sample-id B001,I011,C001 \
  --out-dir out/auto_smoke_20260531

# Direct Worker parity / speed path
.venv/bin/python scripts/run_corridorkey_game_eval.py \
  --backend direct-worker \
  --sample-id B001,I011,C001 \
  --out-dir out/direct_smoke_20260531
```

跑完整 85 样本时去掉 `--sample-id`。

## Python API

```python
from pathlib import Path

from ermbg.io import save_mask, save_rgba
from PIL import Image

from ermbg.api import matte_image

image = Image.open("input.png").convert("RGBA")
result = matte_image(
    image,
    backend="auto",
    output_dir=Path("out/api_matte"),
)

save_rgba("out/api_matte/output.png", result.rgba)
save_mask("out/api_matte/alpha.png", result.alpha)
print(result.debug)
```

重要 metadata:

- `requested_backend`
- `backend`
- `debug.auto_route.selected_backend`
- `debug.auto_route.route`
- `execution_profile`
- `parameter_profile`
- `server_elapsed_sec`

Direct Worker 还会返回 `debug.direct_worker.execution_backend`,例如 `direct-corridorkey` 或 `direct-pymatting-known-b`。

## Verification

常规本地测试:

```bash
.venv/bin/pytest -q
```

Direct Worker HTTP smoke:

```bash
.venv/bin/python scripts/smoke_direct_worker_http.py \
  --base-url <services.direct_worker_url> \
  --sample-id B001,I011
```

Runtime capability smoke through the local Web server:

```bash
curl -sS "<web-url>/api/runtime-capabilities?include_comfy=false&include_object_info=false"
```

Standard artifact manifest:

- Python API / CLI `output_dir` runs write `manifest.json` beside the existing
  `*_rgba.png`, `*_alpha.png`, `*_foreground.png`, and `*.report.json` files.
- Web `/api/matte-candidates` writes each run under
  `out/web_matte_runs_<YYYYMMDD>/.../` and returns `artifact_manifest`.
- Game Eval case directories write `manifest.json` and include
  `artifact_manifest` in each case `summary.json`, including the
  `direct-worker` path.
- Manifest schema is `ermbg.run.v1`.
- Artifact discovery API: `GET /api/artifacts` and
  `GET /api/artifacts/<artifact_id>`.

Auto 与 Direct 对齐时,重点看:

- 同一输入的 `parameter_profile` 是否一致;
- 同一输入的 `execution_profile` 是否一致;
- Direct 的实际执行后端是否符合 route 选择;
- 输出差异是否只在浮点/8-bit rounding 范围内;
- client time 和 server time 是否能解释差异。

Web/API 变更验证要求目标 Web 服务已运行,并用真实 HTTP 请求跑 `/api/matte-candidates` smoke。

## Project Map

| Path                                      | Role                                                  |
| ----------------------------------------- | ----------------------------------------------------- |
| `ermbg/router.py`                         | route decision, asset kind, execution profile         |
| `ermbg/api.py`                            | main matting API and PyMatting Known-B implementation |
| `ermbg/corridorkey_runner.py`             | shared in-process CorridorKey runner                  |
| `ermbg/direct_worker.py`                  | direct execution orchestration                        |
| `ermbg/direct_worker_client.py`           | local HTTP client for Direct Worker                   |
| `ermbg/direct_worker_server.py`           | remote Direct Worker FastAPI server                   |
| `ermbg/web.py`                            | Web UI, Web API, Game Eval launcher                   |
| `scripts/run_corridorkey_game_eval.py`    | manifest-backed batch eval                            |
| `scripts/benchmark_direct_worker_path.py` | direct path benchmark utilities                       |
| `scripts/smoke_direct_worker_http.py`     | remote worker smoke                                   |
| `samples/corridorkey_semantic/`           | B/I/C game asset sample set                           |
| `out/`                                    | eval batches, summaries, generated debug artifacts    |

## Docs

- `docs/install-startup.md` - default install/start flow.
- `docs/architecture.md` - core/adapters/runtimes architecture and service boundaries.
- `docs/ermbg-route-strategy.md` - route/profile/backend contract.
- `docs/corridorkey-semantic-paths.md` - semantic sample paths and B/I/C set.
- `docs/corridorkey-game-ui-plan.md` - game UI development plan.
- `docs/local-ownership.md` - diagnostic ownership scoring, not the main production path.
- `integrations/openclaw/README.md` - optional independent OpenClaw `ermbg-matte` skill integration.
- `DEPLOY.md` - deployment notes.

Documents under `docs/archive/` are reference-only material unless an active doc explicitly points back to them.

## Development Principles

- Algorithm changes should be mechanism-driven, not sample-id-driven.
- 图片识别和 profile 行为应在共享代码中修改,避免分别修改 Web、Direct Worker 或可选适配器。
- Generated eval artifacts should live in a self-contained batch under `out/`.
- Web production behavior should keep shadow handling on unless a preview-only speed mode is explicitly requested.

## Extension Support

### ComfyUI Nodes

`comfy_nodes/` provides ERMBG nodes for custom ComfyUI graphs:

- `ERMBG Route Matte`
- `ERMBG Route Strategy`
- `ERMBG PyMatting Known-B`
- `ERMBG Classify`

Install the node package into ComfyUI only when Comfy graphs need ERMBG nodes.
See `comfy_nodes/README.md` and `DEPLOY.md`.

### OpenClaw Adapter

OpenClaw support is an optional independent `ermbg-matte` adapter.

```bash
scripts/install_openclaw_ermbg_skill.sh

python3 ~/.openclaw/workspace/skills/ermbg-matte/scripts/ermbg_matte.py \
  --image /path/to/input.png
```

Adapters should call the maintained ERMBG service/API and avoid duplicating image feature matching logic.
