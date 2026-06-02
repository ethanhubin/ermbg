# ERMBG · Agent 契约

本文件是引导性契约。只有当任务涉及某一领域时，才去加载该领域的深入文档。

## 运行时默认值

- Web/API 的 `backend=auto` 使用 Direct Worker。
- 服务端点与 Web 默认值都写在 `ermbg.config.json` 中。
- 允许用环境变量在单个 shell 会话内临时覆盖：
  `ERMBG_DIRECT_URL`、`COMFY_URL`、`ERMBG_WEB_AUTO_BACKEND`、
  `ERMBG_WEB_AUTO_FALLBACK_BACKEND`、`ERMBG_ENABLE_COMFY`。
- 正常的 Web 启动和运行时能力检查都走 Direct Worker。
- ComfyUI 是可选的扩展支持，仅用于自定义 Comfy 图，或显式
  `comfy-*` 后端调试。

## 去哪里查

- 产品概览与安装/启动：`README.md`、`docs/install-startup.md`。
- 架构与 profile 契约：`docs/architecture.md`、
  `docs/ermbg-route-strategy.md`。
- 本地归属细节：`docs/local-ownership.md`。
- 可选的 Comfy 节点工作：`comfy_nodes/README.md`、`DEPLOY.md`。
- 历史材料：`docs/archive/`；不要把已归档的计划当作活跃内容。

## 开发基础

- Python 虚拟环境为 `.venv/`，Python 版本 3.12。
- 在 Windows 上使用 `.venv\Scripts\python.exe` 和 `.venv\Scripts\pytest.exe`。
- 保持测试通过。涉及 Web/运行时的改动，需要包含 `tests/test_web.py` 和
  `tests/test_runtime_capabilities.py`。
- 生成的 eval/debug 产物必须放在 `out/` 下一个自包含的 batch 目录中，
  并附带一份机器可读的 summary。
- 做某一算法路径的批量样本测试时，必须显式固定 execution backend，
  例如 PyMatting Known-B 使用
  `--fixed-execution-backend direct-pymatting-known-b`。不要只用
  “这些样本当前会被 auto route 到该路径”来代替固定路线；若有样本路由到
  其它 backend，应记录为 error，而不是静默混跑。
- 批量测试产物必须写标准 `ermbg.run.v1` manifest：batch 根目录要有
  `manifest.json` 和 `summary.json`，每个 case 目录也要有 `manifest.json`
  和 `summary.json`。summary 中要记录固定 backend、实际 execution backend、
  case manifest 路径和输出文件路径，保证 Web 后台列表稳定发现。最终
  `rgba` 是通用输出；`trimap`、`alpha`、`shadow` 等诊断图只在该后端实际
  产出时写入并列入 manifest，不作为所有 backend 的强制要求。

## 算法规则

- 设计目标：在纯色背景上达到像素级完美的透明抠图。
- 游戏 UI 素材使用 PyMatting Known-B，再基于可测量的已知背景证据
  做像素级修复。
- 复杂的绿幕/蓝幕素材使用 CorridorKey。
- 图片特征分类在执行前决定算法和参数。执行阶段消费
  `execution_profile`；它不得重新推断素材类别。
- 算法修复必须由机制驱动。不要围绕样本 ID、文件名、坐标、图标尺寸或单一
  观测到的颜色去调参，除非 Ethan 要求做一次性的特例处理。
- 启发式阈值、置信度门限、过渡带宽度、面积比和重映射常数，都需要在附近写
  注释，说明意图、可观测信号以及要保护的失败模式。
- Web 抠图必须保留阴影处理，除非 Ethan 明确要求只用于预览的提速模式。

## Web 验证

在改动 `ermbg/web.py`、Web UI/API 行为或某个 Web 侧后端之后：

1. 用 `scripts\start_local.ps1` 或等价方式重启本地服务。
2. 确认端口 `7860` 由预期的 `uvicorn ermbg.web:app` 进程持有。
3. 确认首页包含改动后的标记，例如 `Auto Direct Worker`
   或相关 UI 文案。
4. 用 `backend=auto` 对 `/api/matte-candidates` 跑一次真实 HTTP smoke，
   确认 HTTP 200、route/profile 元数据以及 `server_elapsed_sec`。
5. 如果 Web 报告 Direct Worker 连接错误，先检查
   `<services.direct_worker_url>/health`，再去改算法代码。

最终的 Web 状态报告应说明 `7860` 的 PID、Web 是如何启动的、Direct Worker
的健康状态，以及真实 HTTP smoke 的结果。

## 可选的 Comfy 工作

- 仅在需要自定义图支持或显式 `comfy-*` 后端工作时才使用 Comfy。
- 如果 `comfy_nodes/` 有改动，需重新安装/同步自定义节点并重启 ComfyUI。
- Comfy 包装层要保持轻薄，覆盖在共享的 ERMBG API 和 CorridorKey runner 之上。
