# ERMBG 主架构

本文是 ERMBG 当前生产主线的唯一架构入口。模块细节在
`docs/modules/` 下维护；`docs/archive/` 只解释历史来龙去脉，不能作为当前契约。

## 目标

ERMBG 面向游戏 UI、图标、特效和角色资产，在已知或可测背景上生成干净透明
RGBA。系统优先利用生成阶段主动制造的纯色背景约束，而不是把所有素材都交给通用
复杂背景抠图模型。

当前主线:

```text
input
  -> Preprocess
  -> Analyze
  -> Decide
  -> Execute
  -> Output
```

核心边界:

- Web/API 的 `backend=auto` 默认走 Direct Worker。
- Route 只描述 algorithm/profile/params，不描述 server URL。
- Analyze/Decide 在 Execute 前确定 route、候选和语义约束。
- Execute 只消费显式 request，不重新推断 asset kind。
- `/api/matte-candidates` 仅用于旧调用方兼容；Web 主线是 Analyze/Execute 分离，
  新候选质量验证必须走 `/api/analyze-candidates` -> `/api/execute-candidate`。

## 运行边界

Web 负责上传、前置分析、候选展示、用户选择、mask 交互和请求编排。Direct Worker
是主执行边界，负责 PyMatting Known-B、CorridorKey、Known-B Glow、passthrough
和 fallback 的实际 matting。

服务地址来自 `ermbg.config.json`、gitignored `ermbg.local.json` 或环境变量
`ERMBG_DIRECT_URL`。同一个 worker 可配置本机/远端多个 URL，由 Web/API 按优先级
尝试和 fallback。路由算法不得把 server URL 当作决策输入。

## Preprocess

Preprocess 只处理输入素材本身的可观测问题，不裁决主体、孔洞、阴影或半透明材质
归属。

当前唯一主线开关是 `background_repair`:

- 假透明棋盘格修复；
- Known-B 背景场归一化；
- 机制 metadata 写入 preprocess contract。

用户关闭 `background_repair` 时，Analyze 和 Execute 必须消费未修复的同一输入。
Known-B 背景归一化一旦进入 contract，执行端不得私自重跑一套不同的归一化逻辑。

## Analyze

Analyze 是轻量阶段，不预跑多个完整 matte。它负责:

- 调用 `router.build_route_candidates()` 生成 route/model candidates；
- 用 `select_default_route_candidate()` 选择默认 route；
- 为每个 route candidate 生成模型内语义候选；
- 生成稳定 `analysis_id`；
- 生成服务端 preview assets。

`AnalyzeResult.route` 是默认 route 的兼容字段；完整模型候选在
`AnalyzeResult.route_candidates[]`。多 route candidate 或语义争议会使状态变为
`needs_decision`。

### Known-B Analyze

PyMatting Known-B 的 Analyze 主线会生成 explicit trimap preview，并在 Execute 时复用。
当前机制是:

1. 用已知背景色归一化输入背景场。
2. 从强置信外部 BG seed 出发，沿 BG 证据向内搜索。
3. 在明显色彩/归属断层处形成 subject outline。
4. 填充 outline 内部作为 FG core。
5. 其余边缘、过渡和 shadow-facing 区域作为 unknown。
6. 只有 enclosed near-B islands 进入语义候选；候选会把对应区域 overlay 为
   `sure_bg` 或 `sure_fg`。

shadow 不是当前语义候选维度。shadow-like evidence 只在 trimap builder 内作为
边界 unknown 扩展证据，并且扩展必须与 outline / shadow 区域连通，不能在主体内部
散开。

### CorridorKey Analyze

CorridorKey route 可输出 screen material / translucent ownership 风险候选。预览是 hint
或 overlay，不是最终 RGBA。

## Decide

Decide 是用户或调用方选择最终候选的阶段。

Web 当前行为:

- 上传后运行 Analyze；
- 候选列表展示在执行按钮附近；
- 点击候选只切换 preview，不执行 matte；
- 点击执行后调用 `/api/execute-candidate`；
- 粗 mask 是 keep/remove 语义约束，remove 覆盖 keep。

无争议样本默认选中 `auto_default`，但仍通过同一个 Execute request 边界执行。

## Execute

Execute 只运行一次最终决策:

- `/api/execute-candidate` 构造 `ExecutionRequest` 摘要；
- Web 将选中的 `route_candidate`、`semantic_candidate`、preprocess 和 user mask 合成
  Direct Worker 表单；
- Direct Worker `/matte` 收到 `route_decision` 后跳过 `classify_route()`；
- Known-B 若收到 `pymatting_explicit_trimap`，直接消费该 trimap。

执行阶段不得:

- 重新 classify asset kind；
- 私自改变 preprocess 结果；
- 绕过用户候选或 mask；
- 在 Web JS 或 Direct Worker server 层重写 router 规则。

## 输出与 Manifest

输出至少包括:

- `rgba`；
- 后端实际产出的 `alpha`、`foreground`、`trimap` 等诊断图；
- `execution_backend`；
- `execution_server_url`；
- route/profile metadata；
- preprocess/semantic/execution request summary；
- 标准 `ermbg.run.v1` manifest。

批量 eval 的 batch 根目录必须有 `summary.json` 和 `manifest.json`；每个 case 目录也必须
有 `summary.json` 和 `manifest.json`。

## 反模式

- 用样本 ID、文件名、坐标或单一颜色特例修算法。
- 在候选阶段预跑多个完整 matte。
- 把 Known-B 问题简化为“多标 sure-FG 或 sure-BG”。
- 让 shadow 重新变成独立语义候选，导致内部线条或暗装饰跑进 unknown。
- Execute 阶段重新推断素材类别。
- 把归档计划当作当前主线。
