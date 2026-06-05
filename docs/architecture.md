# ERMBG 主架构

本文是 ERMBG 当前主线的架构入口。模块细节在 `docs/modules/` 下维护；历史计划在 `docs/archive/` 下保留。

## 目标

ERMBG 的主目标是在已知或可测背景上生成高质量透明抠图，尤其覆盖游戏 UI、纯色背景图形、绿幕/蓝幕素材、硬边按钮、软边图标、glow、孔洞和轻量阴影。

当前生产主线：

```text
input
  -> Preprocess
  -> Analyze
  -> Decide
  -> Execute
  -> Output
```

## 运行边界

Web/API 的 `backend=auto` 默认使用 Direct Worker。Web 负责上传、前端交互、候选选择和请求编排；Direct Worker 是 Web/API 主执行边界。

服务地址不属于 route 决策。Web 根据 `ermbg.config.json`、`ermbg.local.json` 或环境变量中的 Direct Worker URL 列表选择 server 并 fallback。Route 决策只描述 algorithm/profile/params。

## 阶段

### Preprocess

Preprocess 发生在语义判断之前，处理输入素材本身的可观测问题。

当前实现：

- `ermbg.preprocess.analyze_input_preprocess()`
- `ermbg.preprocess.apply_input_preprocess()`
- `background_repair`
- `/api/preprocess-analysis`

`background_repair` 是主线唯一预处理开关。启用后允许棋盘格背景修复和 Known-B 背景场归一化；关闭后两者都不应用。Preprocess 不能裁决主体、孔洞、阴影或半透明材质归属。

### Analyze

Analyze 在 matting 执行前完成 route/profile 和语义争议分析。

当前实现：

- `ermbg.analyze.analyze_candidates()`
- 共享 `router.classify_route()`
- `AnalyzeResult.route`
- `ambiguity_regions`
- `SemanticCandidate[]`
- `preview_assets`
- `/api/analyze-candidates`

Analyze 可以输出默认候选和争议候选。候选是语义决策候选，不是已执行完成的 RGBA 结果。候选预览只能是 overlay、trimap/hint 参考或其他轻量图。PyMatting Known-B 候选的三态 trimap 可以作为 Execute 输入复用，但仍不是已执行 matte。

Analyze 必须产出稳定 `analysis_id`，并把服务端生成或可发现的轻量 preview assets 写入 payload，供 Decide UI 和后续 manifest 审计引用。

### Decide

Decide 由调用方或用户选择最终语义决策。

当前 Web 行为：

- 上传后自动 Analyze 并生成候选。
- 候选列表位于左侧抠图按钮上方。
- 选中候选后，抠图按钮直接执行当前候选。
- PyMatting Known-B 默认显示 trimap 预览；CorridorKey 默认显示 hint。
- 粗 keep/remove mask 可作为语义约束进入 Execute。

无争议样本可以默认选中 `auto_default`，但主线仍保留 Analyze/Decide/Execute request 边界。

### Execute

Execute 只消费最终决策并执行一次。

当前实现：

- `/api/execute-candidate`
- `ExecutionRequest`
- Web 将 Analyze 的 `route/profile/params` 显式转换为 Direct Worker `route_decision`
- Direct Worker `/matte` 收到 `route_decision` 后跳过 `classify_route()`
- PyMatting Known-B、CorridorKey、Known-B Glow、passthrough 或 fallback 只消费 request

执行阶段不得重新推断 asset kind，不得私有运行另一套背景归一化，不得绕过用户候选或粗 mask 裁决。

## 输出

输出应包含：

- RGBA PNG
- alpha/trimap 等后端实际产出的诊断图
- `execution_backend`
- `execution_server_url`
- route/profile metadata
- preprocess/semantic/execution request summary
- `ermbg.run.v1` manifest

## API 形态

主线：

```text
POST /api/preprocess-analysis
POST /api/analyze-candidates
POST /api/execute-candidate
```

旧兼容层不应出现在 Web 主页面命名中。

## 反模式

- 在候选阶段预跑多个完整 matte。
- 在 Web JS、Direct Worker 或可选集成里重新实现一套路由规则。
- Execute 阶段重新 classify asset kind。
- Execute 阶段私有运行和 Preprocess 不一致的背景归一化。
- 把粗 mask 当作最终 alpha。
- 用样本 ID、文件名、固定坐标或一次性阈值特例修复算法问题。
- 把归档计划当作当前主线。
