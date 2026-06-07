# ERMBG · Agent 契约

本文件是引导性契约。只有当任务涉及某一领域时，才去加载该领域的深入文档。

## 运行时默认值

- Web/API 的 `backend=auto` 使用 Direct Worker。
- 服务端点与 Web 默认值都写在 `ermbg.config.json` 中。
- `services.direct_worker_url` 是主 Direct Worker 地址；
  `services.direct_worker_urls` 是 server URL 优先级列表。若同一个 worker
  可通过本机和远端 IP 访问，必须显式配置 `name`、`url`、`priority`，
  由 Web/API 按优先级尝试并 fallback。
- 允许用环境变量在单个 shell 会话内临时覆盖：
  `ERMBG_DIRECT_URL`、`ERMBG_WEB_AUTO_BACKEND`、
  `ERMBG_WEB_AUTO_FALLBACK_BACKEND`。
- 正常的 Web 启动和运行时能力检查都走 Direct Worker。

## 去哪里查

- 文档入口：`docs/README.md`。
- 产品概览：`README.md`。
- 主线架构：`docs/architecture.md`。
- 模块细节：`docs/modules/`。
- 安装/启动：`docs/modules/operations.md`。
- profile 契约：`docs/modules/route-profiles.md`。
- Known-B 当前算法：`docs/modules/known-b.md`。
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
- 当前 game eval manifest 全量为 87 个样本；标准流程不要继续按旧 85/86 样本口径汇报。

## 算法相关任务

只有在任务涉及路由、profile、matting executor、像素归属、算法参数或质量回归时，
才加载对应算法文档：

- 路由/profile 契约：`docs/modules/route-profiles.md`。
- PyMatting Known-B、BG-seed outline trimap 和孔洞候选：`docs/modules/known-b.md`。
- CorridorKey、绿幕/蓝幕和复杂 UI 路径：`docs/modules/corridorkey.md`。

顶层只保留跨模块硬边界：

- 设计目标：在已知或可测背景上生成高质量透明抠图，纯色背景图形优先追求像素级准确。
- Analyze/Decide 在 Execute 前决定 algorithm、profile、参数和语义约束；
  Execute 消费显式 request，不得重新推断素材类别。
- Known-B 当前主线由 Analyze 生成 explicit trimap：强置信 BG seed 向内找 outline，
  outline 内填充 FG core，边缘/transition/shadow-facing 区域为 unknown；enclosed
  near-B holes 是语义候选，shadow 不再作为独立候选。
- Route 决策只描述 algorithm/profile/params，不描述具体 server URL。
- 算法修复必须由机制驱动。不要围绕样本 ID、文件名、坐标、图标尺寸或单一观测颜色
  做特例调参，除非 Ethan 明确要求一次性处理。

## Web 验证

在改动 `ermbg/web.py`、Web UI/API 行为或某个 Web 侧后端之后：

1. 先确认目标 Direct Worker 是本机还是远端。远端算法改动必须先跑
   `scripts/sync_comfy_ssh.sh --clean --smoke`,再跑
   `scripts/restart_direct_worker_ssh.sh --restart`。
2. 用 `scripts\start_local.ps1` 或等价方式重启本地 Web；如果走远端,
   Web 必须连接 `ermbg.local.json` 或 `ERMBG_DIRECT_URL` 中的远端 URL。
3. 确认端口 `7860` 由预期的 `uvicorn ermbg.web:app` 进程持有。
4. 确认首页包含改动后的标记，例如 `Auto Route`、`CorridorKey`
   或相关 UI 文案。
5. 用 `/api/runtime-capabilities` 确认 Direct Worker URL、health
   `git_sha`/同步标记和 GPU/CPU 能力。
6. 对 `/api/preprocess-analysis` 和 `/api/analyze-candidates` 跑真实 HTTP smoke，
   确认 Preprocess/Analyze 元数据完整,且 Analyze 不执行完整 matte。
7. 用 Analyze payload 对 `/api/execute-candidate` 跑真实 HTTP smoke，确认
   HTTP 200、`algorithm`、route/profile 元数据、`execution_backend`、
   `execution_server_url`、`server_elapsed_sec`；涉及 CorridorKey hint 候选时，
   还要确认 `semantic_decision.corridorkey_hint_variant` 与 Direct Worker
   `algorithm_debug.corridorkey_hint_plan.variant` 一致。
8. `/api/matte-candidates` 只作为旧调用方兼容层，不作为新候选路径质量验证入口。
9. 如果 Web 报告 Direct Worker 连接错误，先检查
   `<services.direct_worker_url>/health`、远端 `7871` 监听和本机 `.venv`
   import `cv2`，再去改算法代码。

最终的 Web 状态报告应说明 `7860` 的 PID、Web 是如何启动的、Direct Worker
的健康状态，以及真实 HTTP smoke 的结果。
