# ERMBG

ERMBG 是面向 AI 生成游戏素材的智能抠图系统。它的核心不是单一抠图模型,而是一个 route strategy layer: 先判断输入属于按钮、图标、特效、角色、透明材质、已带 RGBA 还是未知背景,再选择对应的执行 profile 和后端。

当前生产路径是:

```text
Web/API/CLI backend=auto
  -> remote ComfyUI ErmbgRouteMatte
  -> router classify_route()
  -> PyMatting Known-B / CorridorKey / passthrough
  -> foreground + alpha + rgba_rgb + metadata
```

本机主要负责上传、提交、轮询、下载和轻量诊断。重模型和主要执行逻辑跑在远端 Windows + GPU 机器上。

## Current Status

- **生产默认:** `backend=auto` 走远端 ComfyUI 单节点 `ErmbgRouteMatte`。
- **实验加速:** `backend=direct-worker` 走远端 Direct Worker,绕过 ComfyUI prompt 队列,用于验证吞吐和并发。
- **统一 profile:** Auto 和 Direct Worker 都必须使用同一套 router、`parameter_profile`、`execution_profile` 和共享执行代码。
- **主测试集:** `samples/corridorkey_semantic/manifest.json`,共 85 个样本: 56 button,20 icon/effect,9 character。
- **当前重点:** 游戏 UI 素材、绿/蓝幕已知背景、透明/半透明材质、特效层、角色边缘与硬按钮。
- **历史路径:** Local Ownership、旧 AutoMatte、BiRefNet/GrabCut 主链路等只作为诊断或归档背景,不是当前 Web/API 生产默认。

## Execution Profiles

路由必须在执行前确定最终 profile。执行层读取 profile,不再自行重新猜测“这是角色还是图标”。

| Asset / case                      | Backend selected by auto  | Execution profile                | Notes                                    |
| --------------------------------- | ------------------------- | -------------------------------- | ---------------------------------------- |
| clean RGBA                        | passthrough               | passthrough                      | 已有透明通道且质量足够时直接保留         |
| hard UI / deterministic button    | `comfy-pymatting-known-b` | `pymatting-hard-button`          | 硬边按钮、稳定已知背景、硬/软阴影按钮    |
| known-background graphic fallback | `comfy-pymatting-known-b` | `pymatting-known-bg`             | 非角色/非图标的稳定已知背景图形          |
| unknown / unstable background     | `comfy-pymatting-known-b` | `pymatting-fallback`             | Auto 不再默认调用 RMBG fallback          |
| shaped icon                       | `comfy-corridorkey`       | `corridorkey-shaped-icon`        | 有形状 hint 的图标,含 key-color 材质保护 |
| effect icon                       | `comfy-corridorkey`       | `corridorkey-effect-icon`        | 发光、烟雾、软 alpha 特效层              |
| translucent / glass button        | `comfy-corridorkey`       | `corridorkey-transparent-button` | 半透明按钮、玻璃材质、复杂按钮           |
| character                         | `comfy-corridorkey`       | `corridorkey-character`          | 1024 角色图,发丝、毛发、半透明和 glow    |

`parameter_profile` 是分析元数据,用于解释 router 为什么选择某条路径。`execution_profile` 是执行契约,用于控制 hint mode、mask prior、color protection、refiner、despeckle 和 debug metadata。

## Backends

### Auto RouteMatte

`backend=auto` 是 Web/API/CLI 的默认生产模式。它提交一张图片到远端 ComfyUI 的 `ErmbgRouteMatte` 节点,在 Comfy 进程内完成 route、参数选择和实际 matting。

远端 ComfyUI 地址来自 `COMFY_URL`,未设置时默认:

```bash
http://192.168.0.8:8000
```

本地代码变更后,按项目约定同步到远端:

```bash
scripts/sync_comfy_ssh.sh --smoke
```

如果改了 `comfy_nodes/`,还需要同步节点并重启 ComfyUI:

```bash
scripts/sync_comfy_ssh.sh --nodes
scripts/restart_comfy_ssh.sh --restart --dev-reload
```

### Direct Worker

`backend=direct-worker` 是新的测试后端。它仍然跑在远端服务器上,但不经过 ComfyUI prompt 队列,而是直接调用 `ermbg.direct_worker_server` 的 HTTP 接口。

Direct Worker 的目标不是维护另一套算法,而是验证:

- 绕过 ComfyUI 单队列后能提速多少;
- CPU 多核 PyMatting 是否能并行;
- CorridorKey GPU 路径在 4090 24G 上的真实吞吐;
- Auto 与 Direct 是否能做到同 profile、同参数、同执行逻辑。

共享代码边界:

- CorridorKey: `ermbg.corridorkey_runner.LocalCorridorKeyClient`
- PyMatting Known-B: `ermbg.api._matte_image_pymatting_known_b()`
- Direct HTTP client: `ermbg.direct_worker_client.matte_image_direct_worker()`
- Direct server: `ermbg.direct_worker_server`

启动远端 Direct Worker:

```bash
ssh ermbg-comfy 'cd /d C:\Users\darkv\ermbg_src && E:/ComfyUI/.venv/Scripts/python.exe -m ermbg.direct_worker_server --host 0.0.0.0 --port 7871 --cpu-workers 4'
```

默认 Direct Worker URL:

```bash
http://192.168.0.8:7871
```

最近一次 85 样本对比基线:

| Scope             | Auto client time | Direct client time | Speedup |
| ----------------- | ---------------: | -----------------: | ------: |
| all 85 cases      |            91.1s |              68.5s |   1.33x |
| PyMatting cases   |            13.5s |               5.6s |   2.40x |
| CorridorKey cases |            77.6s |              62.9s |   1.23x |

这组数字只作为当前机器和当前实现的基线。改动 route、profile 或 runner 后应重新跑同样样本集。

## Install

项目推荐使用 `.venv/` 和 Python 3.12。

```bash
cd /Users/ethanhu/Desktop/Git/ERMBG
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[torch,dev,web]"
```

最小本地依赖可以不装 `torch`,但 Web、Direct Worker server、测试和旧诊断工具会用到对应 extra。Mac 不应本地加载 SDXL、FLUX、Qwen 等重模型。

## Run

### Web UI

```bash
PYTHONPATH=. .venv/bin/python -m uvicorn ermbg.web:app --host 127.0.0.1 --port 7860
```

打开:

```text
http://127.0.0.1:7860
```

Web 里默认选择 `Auto RouteMatte`。`Direct Worker` 是测试后端,用于对比速度、并发和 profile 一致性。

### CLI Matte

```bash
.venv/bin/ermbg matte input.png --backend auto --out-dir out/manual_matte
```

CLI 支持的主要后端:

- `auto`
- `comfy-corridorkey`
- `comfy-pymatting-known-b`
- `pymatting-known-b`
- `comfy-rmbg`

Direct Worker 当前主要通过 Web/API 和 Game Eval 使用。

### Game Eval

Game Eval 是当前最重要的回归验证入口。所有输出写入 `out/` 下的 batch 目录,并生成 `summary.json`。

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
  --base-url http://192.168.0.8:7871 \
  --sample-id B001,I011
```

Auto 与 Direct 对齐时,重点看:

- 同一输入的 `parameter_profile` 是否一致;
- 同一输入的 `execution_profile` 是否一致;
- Direct 的实际执行后端是否只是从 `comfy-*` 映射到 `direct-*`;
- 输出差异是否只在浮点/8-bit rounding 范围内;
- client time 和 server time 是否能解释差异。

Web/API 改动后,必须确认 `127.0.0.1:7860` 正在运行更新后的 server,并用真实 HTTP 请求跑 `/api/matte-candidates` smoke。

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
| `comfy_nodes/ermbg_nodes.py`              | ComfyUI custom nodes                                  |
| `scripts/run_corridorkey_game_eval.py`    | manifest-backed batch eval                            |
| `scripts/benchmark_direct_worker_path.py` | direct path benchmark utilities                       |
| `scripts/smoke_direct_worker_http.py`     | remote worker smoke                                   |
| `samples/corridorkey_semantic/`           | current B/I/C game asset sample set                   |
| `out/`                                    | eval batches, summaries, generated debug artifacts    |

## Docs

- `docs/ermbg-route-strategy.md` - current route/profile/backend contract.
- `docs/corridorkey-semantic-paths.md` - semantic sample paths and B/I/C set.
- `docs/corridorkey-game-ui-plan.md` - game UI development plan.
- `docs/local-ownership.md` - diagnostic ownership scoring, not the main production path.
- `comfy_nodes/README.md` - ComfyUI node usage.
- `integrations/openclaw/README.md` - OpenClaw `comfyui-rmbg --mode ermbg` integration.
- `DEPLOY.md` - deployment notes.

Documents under `docs/archive/` are historical reference unless a current doc explicitly points back to them.

## OpenClaw

ERMBG is integrated into OpenClaw through the existing `comfyui-rmbg` skill as `--mode ermbg` / `--smart`.

```bash
python3 ~/.openclaw/workspace/skills/comfyui-rmbg/scripts/comfyui_rmbg.py \
  --mode ermbg \
  --image /path/to/input.png
```

This submits the remote `ErmbgRouteMatte` workflow and stores `output.png`, `workflow.json`, `manifest.json`, and Comfy history metadata in the OpenClaw media directory.

## Development Principles

- Algorithm changes should be mechanism-driven, not sample-id-driven.
- Route/profile behavior should be changed in shared code, not separately patched in Comfy and Direct Worker.
- Generated eval artifacts should live in a self-contained batch under `out/`.
- Web production behavior should keep shadow handling on unless a preview-only speed mode is explicitly requested.
- The Mac is for orchestration and lightweight CV. Heavy generation/inference belongs on the remote server.
