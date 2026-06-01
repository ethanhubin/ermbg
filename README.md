# ERMBG

ERMBG 是面向 AI 生成游戏素材的智能抠图系统。它的目标只有一个: 在纯色背景上做到
像素级完美的透明抠图,输出可以直接进入游戏 UI、动效和角色素材生产链路的 RGBA PNG。

通用抠图模型(rembg/RMBG 等)是为照片设计的,追求"语义上像前景"。但游戏素材的痛点
不在语义,而在边缘: 1px 的硬边、细描边、内部孔洞、接触阴影、玻璃和 glow 的半透明
过渡——这些地方差几个像素就会在新背景上露馅。ERMBG 因此换了一个出发点。

## 核心理念

**确定性优先,模型兜底。** 游戏素材通常生成在已知的纯色/绿幕/蓝幕上,背景色、
边缘拓扑和阴影都是可测量的强证据。ERMBG 优先用这些证据直接求解,而不是先跑通用
模型再修它的错;只有证据不足时(照片、发丝、混合背景)才回退到通用 matting。

**先定归属,再算 alpha。** 一块区域是背景、孔洞、半透明主体还是阴影,是不同性质
的问题,不该用一条阈值一刀切。ERMBG 先判定区域归属,再按归属分别求解,所以玻璃
不会被当成孔洞、阴影不会被当成主体。

**机制驱动,而非样本驱动。** 不围绕样本 ID、文件名、坐标或固定颜色打补丁。每条
规则都对应一个可观测信号和一个要保护的失败模式,这样才能泛化到没见过的素材。

## 设计

- **游戏 UI 像素级修复:** 基于 PyMatting Known-B,以已知背景色为强约束,结合颜色距离、边缘证据和重投影误差修复硬边、细边、孔洞和阴影。
- **图片特征识别:** 自动识别按钮、图标、特效、角色、玻璃/半透明材质、已有 alpha 和未知背景 fallback。
- **算法与参数匹配:** 根据识别结果选择 PyMatting Known-B、CorridorKey、passthrough 或 fallback,并生成对应执行参数。
- **影视级绿/蓝幕抠图:** 对复杂绿幕/蓝幕素材使用 CorridorKey,处理发丝、毛发、半透明、发光、烟雾和玻璃材质。
- **批量可验证:** Web run、CLI/API run 和 Game Eval 都写入 `out/` batch/artifact,保留 PNG 输出和机器可读 manifest。

## 游戏素材覆盖范围

ERMBG 已完成面向游戏素材的类型适配:

| 素材类型 | 覆盖能力 |
| --- | --- |
| 硬边按钮 / UI 面板 | 纯色背景检测、硬边 alpha、孔洞归属、阴影保留 |
| 玻璃 / 半透明按钮 | 绿/蓝幕 CorridorKey、半透明主体保护、颜色保护 |
| 图标 / shaped icon | 形状 hint、key-color 材质保护、透明边缘 |
| 特效图标 | glow、烟雾、软 alpha、发光层 |
| 角色素材 | 1024 角色图、发丝、毛发、半透明边缘、glow |
| 已有 RGBA | alpha 质量检查和 passthrough |
| 未知 / 不稳定背景 | PyMatting fallback 和诊断 metadata |

## 技术路径

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

## 输出

ERMBG 输出透明 PNG,并保留独立 alpha、前景 RGB、识别/debug metadata 和 artifact manifest。典型输出字段包括:

- `rgba`: 可直接下载使用的 RGBA PNG
- `alpha`: float32 `[0, 1]` soft mask
- `foreground_srgb`: sRGB foreground companion
- `strategy_name`: 实际执行策略
- `background_color`: 诊断到的背景色
- `debug.auto_route`: 识别结果、asset kind、profile 和后端选择
- `server_elapsed_sec`: 服务端耗时

回归样本集位于 `samples/corridorkey_semantic/manifest.json`,覆盖 button、icon/effect 和 character 三类游戏素材。

## 执行 Profile

图片特征识别会在执行前确定最终 profile。`parameter_profile` 是分析元数据,用于解释系统匹配算法的依据；`execution_profile` 是执行契约,用于控制 hint mode、mask prior、color protection、refiner、despeckle 和 debug metadata。

| 素材 / 场景                        | Execution profile                | 执行路径                        |
| --------------------------------- | -------------------------------- | ------------------------------- |
| clean RGBA                        | passthrough                      | passthrough                     |
| 硬边 UI / 确定性按钮              | `pymatting-hard-button`          | PyMatting Known-B               |
| 已知背景图形 fallback             | `pymatting-known-bg`             | PyMatting Known-B               |
| 未知 / 不稳定背景                 | `pymatting-fallback`             | PyMatting fallback              |
| shaped icon                       | `corridorkey-shaped-icon`        | CorridorKey                     |
| effect icon                       | `corridorkey-effect-icon`        | CorridorKey                     |
| 半透明 / 玻璃按钮                 | `corridorkey-transparent-button` | CorridorKey                     |
| 角色                              | `corridorkey-character`          | CorridorKey                     |

## 安装

项目推荐使用 `.venv/` 和 Python 3.12。

```bash
cd <ermbg-root>
uv venv .venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -e ".[web,dev,torch]"
```

`torch` extra 用于 Direct Worker 的 CorridorKey 路径。只跑 PyMatting smoke 时
可以不装,但默认游戏素材主流程建议安装。

## 部署

ERMBG Web 通过配置中的 `services.direct_worker_url` 连接 Direct Worker。共享默认值写在 `ermbg.config.json`; 机器相关覆盖写在 gitignored `ermbg.local.json`。环境变量 `ERMBG_DIRECT_URL` 可用于临时覆盖该配置。

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

默认 Direct Worker 地址写在 `ermbg.config.json` 的 `services.direct_worker_url`。每台机器自己的 Comfy/Direct Worker 地址、`web.auto_backend` 和 `web.enable_comfy` 应写入 `ermbg.local.json`。
ComfyUI 不是默认运行路径,`COMFY_URL` 没有代码级 fallback; 需要 Comfy 的机器必须显式配置 `services.comfy_url` 或 `COMFY_URL`。

## 使用

### Web UI

```powershell
.\scripts\start_local.ps1
```

打开:

```text
<web-url>
```

Web 里默认选择 `Auto`; 实际 auto 路径由本机配置的 `web.auto_backend` 决定。

### CLI 抠图

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

## 验证

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

通过本地 Web 服务跑运行时能力 smoke:

```bash
curl -sS "<web-url>/api/runtime-capabilities?include_comfy=false&include_object_info=false"
```

标准 artifact manifest:

- Python API / CLI 的 `output_dir` run 会在已有的 `*_rgba.png`、
  `*_alpha.png`、`*_foreground.png` 和 `*.report.json` 旁边写入 `manifest.json`。
- Web 的 `/api/matte-candidates` 把每次 run 写到
  `out/web_matte_runs_<YYYYMMDD>/.../` 下,并返回 `artifact_manifest`。
- Game Eval 的每个 case 目录写入 `manifest.json`,并在各 case 的
  `summary.json` 中包含 `artifact_manifest`,`direct-worker` 路径也一样。
- Manifest schema 为 `ermbg.run.v1`。
- Artifact 发现 API: `GET /api/artifacts` 和
  `GET /api/artifacts/<artifact_id>`。

对齐 Auto 与 Direct 时,重点看:

- 同一输入的 `parameter_profile` 是否一致;
- 同一输入的 `execution_profile` 是否一致;
- Direct 的实际执行后端是否符合 route 选择;
- 输出差异是否只在浮点/8-bit rounding 范围内;
- client time 和 server time 是否能解释差异。

Web/API 变更验证要求目标 Web 服务已运行,并用真实 HTTP 请求跑 `/api/matte-candidates` smoke。

## 项目地图

| 路径                                      | 角色                                                  |
| ----------------------------------------- | ----------------------------------------------------- |
| `ermbg/router.py`                         | route 决策、asset kind、execution profile             |
| `ermbg/api.py`                            | 主 matting API 和 PyMatting Known-B 实现              |
| `ermbg/corridorkey_runner.py`             | 共享的进程内 CorridorKey runner                       |
| `ermbg/direct_worker.py`                  | direct 执行编排                                       |
| `ermbg/direct_worker_client.py`           | Direct Worker 的本地 HTTP client                      |
| `ermbg/direct_worker_server.py`           | 远端 Direct Worker FastAPI 服务                       |
| `ermbg/web.py`                            | Web UI、Web API、Game Eval 启动器                     |
| `scripts/run_corridorkey_game_eval.py`    | 基于 manifest 的批量 eval                             |
| `scripts/benchmark_direct_worker_path.py` | direct 路径基准测试工具                               |
| `scripts/smoke_direct_worker_http.py`     | 远端 worker smoke                                     |
| `samples/corridorkey_semantic/`           | B/I/C 游戏素材样本集                                  |
| `out/`                                    | eval batch、summary、生成的 debug 产物                |

## 文档

- `docs/install-startup.md` - 默认安装/启动流程。
- `docs/architecture.md` - core/adapters/runtimes 架构与服务边界。
- `docs/ermbg-route-strategy.md` - route/profile/backend 契约。
- `docs/corridorkey-semantic-paths.md` - 语义样本路径与 B/I/C 集。
- `docs/corridorkey-game-ui-plan.md` - 游戏 UI 开发计划。
- `docs/local-ownership.md` - 诊断用归属打分,不是主生产路径。
- `integrations/openclaw/README.md` - 可选的独立 OpenClaw `ermbg-matte` skill 集成。
- `DEPLOY.md` - 部署说明。

`docs/archive/` 下的文档仅供参考,除非某个活跃文档明确指回它们。

## 开发原则

- 算法改动应由机制驱动,而非由样本 ID 驱动。
- 图片识别和 profile 行为应在共享代码中修改,避免分别修改 Web、Direct Worker 或可选适配器。
- 生成的 eval 产物应放在 `out/` 下一个自包含的 batch 中。
- Web 生产行为应保持阴影处理开启,除非显式要求只用于预览的提速模式。

## 扩展支持

### ComfyUI 节点

`comfy_nodes/` 为自定义 ComfyUI 图提供 ERMBG 节点:

- `ERMBG Route Matte`
- `ERMBG Route Strategy`
- `ERMBG PyMatting Known-B`
- `ERMBG Classify`

仅当 Comfy 图需要 ERMBG 节点时,才把节点包安装进 ComfyUI。
参见 `comfy_nodes/README.md` 和 `DEPLOY.md`。

### OpenClaw 适配器

OpenClaw 支持是一个可选的独立 `ermbg-matte` 适配器。

```bash
scripts/install_openclaw_ermbg_skill.sh

python3 ~/.openclaw/workspace/skills/ermbg-matte/scripts/ermbg_matte.py \
  --image /path/to/input.png
```

适配器应调用维护中的 ERMBG 服务/API,避免重复实现图片特征匹配逻辑。
