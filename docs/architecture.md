# ERMBG 架构

ERMBG 是一个共享的抠图 core,搭配多个适配器和运行时后端。生产环境的默认形态
是本地 Web/API 编排,调用 ERMBG Direct Worker 服务。ComfyUI 为 Comfy 图提供
可选的自定义节点。

## 整体形态

```text
入口
  Web UI / Web API
  CLI / Python API
  ComfyUI 自定义节点
        |
        v
ERMBG core 契约
  输入归一化
  route strategy
  parameter_profile
  execution_profile
  共享的 matting / shadow / metadata 代码
        |
        v
运行时后端
  Direct Worker HTTP 服务
  本地轻量 Python/CV 路径
  可选的 ComfyUI 自定义节点
```

入口负责表达用户意图并传入图片数据。素材分类归属于共享的 route/profile 契约。

## 分层

### Core

Core 是 `ermbg/` 下定义生产行为的代码:

- `router.py`: route 决策、asset kind、`parameter_profile` 和
  `execution_profile`。
- `api.py`: 高层 `matte_image()` 契约和维护中的本地执行辅助函数,
  包括 PyMatting Known-B。
- `corridorkey_runner.py`: 共享的进程内 CorridorKey 适配器,被 Comfy 节点
  包装层和 Direct Worker 共同使用。
- `shadow.py`、`ownership.py`、`known_bg_hard_ui.py`、`pymatting_refine.py`
  及相关模块: 可复用的 matting 机制。

Core 拥有输出语义: 前景 RGB、alpha、RGBA RGB 层、route 元数据、debug 元数据
和耗时元数据。

### 适配器

适配器把外部调用方翻译成 core 契约:

- `ermbg.web`: 本地 FastAPI Web UI、Web API 和 Game Eval 启动器。
- CLI/Python API: 供脚本和测试使用的本地直连集成。
- `comfy_nodes/ermbg_nodes.py`: 可选的 ComfyUI 自定义节点包装层。
- `integrations/openclaw`: 可选的独立 OpenClaw `ermbg-matte` skill 集成。

适配器保持轻薄。它们暴露 UI 控件、选择一个请求后端,并把请求传入共享的
route 逻辑。

### 运行时

运行时后端决定执行发生在哪里:

- **Direct Worker 运行时**: `backend=auto` 时 Web/API 的默认运行时。它是
  围绕共享 router/profile 和执行代码的一个 HTTP worker。
- **本地运行时**: 轻量的确定性工作,例如 PyMatting、OpenCV/numpy 工具、
  route 调试、诊断和测试。
- **Comfy 运行时**: 覆盖在共享 route/profile 和执行代码之上的可选图/节点
  适配器。

## 生产契约

Web `backend=auto` 意味着:

```text
local Web/API/CLI
  -> ermbg.direct_worker_server
  -> classify_route()
  -> passthrough / PyMatting Known-B / CorridorKey / PyMatting fallback
  -> foreground + alpha + rgba_rgb + metadata
```

`backend=direct-worker` 意味着显式请求同一个服务:

```text
local Web/API/eval client
  -> ermbg.direct_worker_server
  -> 共享 route/profile 契约
  -> direct-corridorkey 或 direct-pymatting-known-b 执行
```

Direct Worker 消费共享的 route 元数据和 execution profile。

## ComfyUI 节点契约

维护中的 Comfy 节点表面是:

- `ERMBG Route Matte`: 可选的 Comfy 图 auto route 和抠图节点。
- `ERMBG Route Strategy`: 仅做 route 的调试/分支节点。
- `ERMBG PyMatting Known-B`: 用于硬边 UI 和稳定已知背景图形的确定性
  已知背景节点。
- `ERMBG Classify (preview)`: 轻量诊断节点。
- `Convert Masks to Images`: 工具节点。

自定义 Comfy 图可以用 `ERMBG Route Strategy` 做分支。Web/API 生产环境对
`backend=auto` 使用 Direct Worker;显式 Comfy 路径属于调试/扩展路径。

## 可选的 OpenClaw 适配器

OpenClaw 集成是一个可选的独立 `ermbg-matte` skill:

```bash
python3 ~/.openclaw/workspace/skills/ermbg-matte/scripts/ermbg_matte.py \
  --image /path/to/input.png
```

该路径应调用维护中的 ERMBG 服务/API,并归档 `output.png`、`manifest.json`
和运行时元数据。要把 ERMBG 的 route 逻辑保留在共享 core 中。

OpenClaw 专属功能应保持为覆盖在同一 route/matte 契约之上的轻薄适配器。

## 运行规则

1. `router.py` 是 asset family、`parameter_profile` 和
   `execution_profile` 的唯一真相来源。
2. 适配器保持轻薄。Web、CLI、Direct Worker、Comfy 节点和可选集成都把数据
   传入共享契约。
3. Direct Worker 是 Web/API 的服务边界。
4. ComfyUI 是用于自定义 Comfy workflow 的可选图宿主。
5. 每个适配器都应写出可浏览的产物,包含输出 PNG、route 元数据、耗时元数据,
   以及适用情况下的 `ermbg.run.v1` manifest。

## 反模式

- 在 Web JavaScript、可选适配器代码、Comfy 包装代码或 Direct Worker 胶水
  代码里重复实现 route 启发式。
- 新增一个会引入新 profile 契约的后端。
- 通过对样本 ID、文件名、一次性尺寸或固定坐标分支来修一个样本。
- 在本地 Web 进程内运行重型生成模型或 VLM 模型。
- 因为 ComfyUI 不可用就让正常的 Web 启动失败。
- 在相关 Direct Worker 或可选 Comfy 适配器尚未重启并 smoke 验证前,就把本地
  源码改动当作已部署。
