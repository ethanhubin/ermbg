# ERMBG —— 专为游戏资产链路打造的像素级抠图工具

[![Python Version](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-orange?style=flat-square)](https://github.com/ethanhubin/ermbg)

`ERMBG` 是一款专为游戏 UI、图标、特效、角色等资产量身定制的高精度、工业级自动抠图解决方案。

## 解决的行业痛点

当前 AI 生成图像技术已广泛应用，但直接生成带透明通道（Alpha）的资产仍不可靠。最常见的问题是: 要求透明背景时,AI会生成黑白格的"假透明"。而目前市面上的抠图工具,又会遇到边缘残留白边/杂色，半透明玻璃、特效资产的通道破损。无法直接满足游戏引擎（Unity / Unreal）的渲染规范。

**ERMBG 的解法是：借鉴影视行业绿幕经验，主动制造纯色背景约束。**
在资产生成阶段，引导 AI 将目标物体生成在纯色（绿幕/蓝幕）背景上。随后由 ERMBG 进行精准的背景扣除与边缘修复，输出像素级干净、可直接投入游戏 UI、动效及角色链路的透明 RGBA PNG。

---

## 设计理念

传统的通用抠图模型（如普通的 Rembg、SAM 等）核心专注于解决**复杂非结构化背景**下的物体提取问题。

但在面对游戏美术资产时，通用模型由于缺乏对游戏渲染特征的感知，在处理**边缘抗锯齿（Anti-aliasing）、各种形态的发光（Glow）、细腻软阴影以及粒子贴图（Particle Textures）**时精度不足，极易导致边缘不平滑、抠图漏色、产生脏点或阴影丢失等视觉瑕疵。

针对上述工业管线痛点，ERMBG 围绕以下三条核心理念进行构建：

* **精细化识别与通道修复**：专为游戏资产特征优化，精准识别抗锯齿边缘、软阴影与发光粒子，进行针对性的通道修复与去溢色处理，确保输出质量达到像素级生产标准。
* **资产分类与智能路由**：游戏资产形态多样（如硬边 UI、毛发角色、雾化特效等）。系统通过特征识别自动判定资产类别并匹配最优 Profile 路由，调用专用算法进行差异化处理。
* **机制驱动的泛化能力**：算法规则建立在可观测的图像信号与明确的数学约束上，而非针对特定样本进行打补丁式微调，确保对未知的新美术素材具备稳健的泛化表现。

---

## 素材覆盖与智能路由

系统在接收图像后会触发特征识别，自动判断素材类型并分流至对应的处理管线（Pipeline）：

| 素材类型 | 实际游戏场景 | ERMBG 处理优势 | 执行路径 (技术细节) |
| :--- | :--- | :--- | :--- |
| **硬边按钮 / UI 面板** | 扁平化 UI、游戏九宫格框体等 | BG-seed outline trimap 保持硬边、抗锯齿和孔洞 | PyMatting Known-B |
| **玻璃 / 半透明按钮** | 带有透明度梯度与折射的 UI 资产 | 完美保留透明度渐变，杜绝杂色与黑边 | CorridorKey |
| **图标 / Shaped Icon** | 装备图标、技能图标等剪影资产 | 保持图形本身的轮廓结构不畸变 | CorridorKey |
| **特效图标** | 包含 Glow（发光）、烟雾、软 Alpha 边缘 | 完整保留光晕与雾化半透明效果，避免被误判为杂色 | CorridorKey |
| **角色资产** | 2D 角色立绘、带发丝/毛发的怪物资产 | 精准提取发丝级细节与半透明过渡边缘 | CorridorKey |
| **已有 RGBA** | 已包含透明通道的原始图片 | 自动识别并直接放行，避免二次处理损耗 | passthrough |
| **未知 / 不稳定背景** | 未能完美生成在纯色背景上的复杂图像 | 自动切换至兜底策略，最大程度还原边缘 | PyMatting fallback |

---

## 真实执行主线

```text
input
  -> Preprocess
  -> Analyze
  -> Decide
  -> Execute
  -> Output
```

- Preprocess 处理棋盘格/背景场归一化等输入问题。
- Analyze 生成 route candidates、semantic candidates 和轻量 preview。
- Decide 选择默认或用户指定候选。
- Execute 只运行一次最终 request。

Known-B 当前主线由 Analyze 生成 explicit trimap：从强置信 BG seed 往内搜索到真实
outline，填充 outline 内部作为 FG core，边缘/过渡/shadow-facing 区域作为 unknown。
孔洞作为候选 overlay 到 trimap；shadow 不再作为独立语义候选。

## 自动化路由机制 (Execution Profile)

图片特征识别在执行前自动确定配置（Profile）：


```

[ 输入图像 ] ──> ( 特征自动识别 ) ──> 决定 [ Execution Profile ] ──> 路由至最强 [ 执行路径 ]

```

| 素材 / 场景 | Execution Profile | 执行路径 |
| :--- | :--- | :--- |
| clean RGBA | `passthrough` | passthrough |
| 硬边 UI / 确定性按钮 | `pymatting-hard-button` | PyMatting Known-B |
| 已知背景 fallback | `pymatting-known-bg` | PyMatting Known-B |
| 未知 / 不稳定背景 | `pymatting-fallback` | PyMatting fallback |
| shaped icon | `corridorkey-shaped-icon` | CorridorKey |
| effect icon | `corridorkey-effect-icon` | CorridorKey |
| 半透明 / 玻璃按钮 | `corridorkey-transparent-button` | CorridorKey |
| 角色 | `corridorkey-character` | CorridorKey |

---

## 📤 输出字段说明

每次抠图任务均会输出丰富的数据结构，方便下游工具链无缝承接：

| 字段 | 说明 |
| :--- | :--- |
| `rgba` | 可直接使用的 RGBA PNG |
| `alpha` | float32 `[0, 1]` soft mask |
| `foreground_srgb` | sRGB foreground companion |
| `strategy_name` | 实际执行策略 |
| `background_color` | 诊断到的背景色 |
| `debug.auto_route` | 识别结果、asset kind、profile 和 backend 选择 |
| `server_elapsed_sec` | 服务端耗时 |

---

## 📦 安装指南

需要 Python 3.12。 建议使用 `uv` 进行虚拟环境管理与依赖安装：

```bash
uv venv .venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -e ".[web,dev,torch]"

```

> 💡 `torch` extra 用于 Direct Worker 的 CorridorKey 路径。

---

## 🚀 使用方法

### 1. Web UI 界面

```powershell
.\scripts\start_local.ps1

```

Web 默认选 `Auto Route`。手动下拉只选 algorithm（`CorridorKey`、`PyMatting Known-B`、`Known-B Glow`、`Passthrough`）。

### 2. CLI 命令行

```bash
.venv/bin/ermbg matte input.png --backend auto --out-dir out/result

```

`--backend` 可选值：`auto`、`pymatting-known-b`、`corridorkey`、`known_bg_glow`、`passthrough`。

### 3. Python API 接入

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

> 💡 重要 metadata 字段：`execution_profile`、`parameter_profile`、`debug.auto_route.algorithm`、`server_elapsed_sec`。Direct Worker 还会返回 `debug.direct_worker.execution_backend`（如 `direct-corridorkey`、`direct-pymatting-known-b`）。

### 4. Game Eval (批量回归验证)

批量回归验证，输出写入 `out/` 下的 batch 目录并生成 `summary.json`。

```bash
.venv/bin/python scripts/run_corridorkey_game_eval.py \
  --backend auto \
  --sample-id B001,I011,C001 \
  --out-dir out/smoke

```

去掉 `--sample-id` 跑完整 86 样本集。回归样本集：`samples/corridorkey_semantic/manifest.json`。

---

## 🌐 部署与分布式架构

配置写在 `ermbg.config.json`（共享默认值）和 gitignored `ermbg.local.json`（机器相关覆盖）。`services.direct_worker_urls` 是 Direct Worker URL 优先级列表，支持自动 fallback。

### 远端 Direct Worker 部署：

```bash
# 先同步当前源码快照，再重启远端 Direct Worker
scripts/sync_comfy_ssh.sh --clean --smoke
scripts/restart_direct_worker_ssh.sh --restart
curl -sS "http://192.168.0.8:7871/health"

```

### 本机前端 Web + 远端 Worker 联动：

```powershell
$env:ERMBG_DIRECT_URL = "<services.direct_worker_url>"
.\scripts\start_local.ps1 -SkipDirectWorker -DirectUrl $env:ERMBG_DIRECT_URL

```

详见 [docs/modules/operations.md](docs/modules/operations.md)。

---

## 🧪 自动化测试与验证

```bash
# 单元测试
.venv/bin/pytest -q

# Direct Worker HTTP smoke
.venv/bin/python scripts/smoke_direct_worker_http.py \
  --base-url <services.direct_worker_url> \
  --sample-id B001,I011

# Runtime capabilities
curl -sS "<web-url>/api/runtime-capabilities"

```

---

## 🗺️ 项目地图 (Project Map)

| 路径 | 角色 |
| --- | --- |
| `ermbg/router.py` | route 决策、asset kind、execution profile |
| `ermbg/api.py` | 主 matting API 和 PyMatting Known-B 实现 |
| `ermbg/analyze.py` | Analyze、route/semantic candidates、Known-B explicit trimap preview |
| `ermbg/pymatting_refine.py` | Known-B BG-seed outline trimap builder |
| `ermbg/corridorkey_runner.py` | 进程内 CorridorKey runner |
| `ermbg/direct_worker.py` | direct 执行编排 |
| `ermbg/direct_worker_client.py` | Direct Worker HTTP client |
| `ermbg/direct_worker_server.py` | 远端 Direct Worker FastAPI 服务 |
| `ermbg/web.py` | Web UI、Web API、Game Eval |
| `scripts/run_corridorkey_game_eval.py` | 批量 eval |
| `samples/corridorkey_semantic/` | B/I/C 游戏素材样本集 |
| `out/` | eval batch、summary、debug 产物 |

---

## 📄 参考文档

* [docs/README.md](docs/README.md) — 文档入口和阅读顺序
* [docs/architecture.md](docs/architecture.md) — 主线架构与服务边界
* [docs/modules/route-profiles.md](docs/modules/route-profiles.md) — route / profile / backend 契约
* [docs/modules/operations.md](docs/modules/operations.md) — 完整安装与启动流程
