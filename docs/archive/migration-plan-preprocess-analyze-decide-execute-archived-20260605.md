# Preprocess / Analyze / Decide / Execute 迁移计划

本文是重构开发计划。它只说明如何把现有 ERMBG 迁移到
`Preprocess -> Analyze -> Decide -> Execute` 主线,不重新定义架构原则。

主线说明见:

- `docs/architecture.md`
- `docs/semantic-candidate-workflow.md`
- `docs/ermbg-route-strategy.md`

## 目标状态

完成后,Web/API/CLI 的主路径应满足:

```text
input
  -> Preprocess
     去网格、Known-B 背景场归一化等原始素材前置加工
  -> Analyze
     route/profile、争议区域、语义候选、trimap/hint 参考图
  -> Decide
     无争议自动通过;有争议由用户选择候选或粗 Mask 修正
  -> Execute
     Direct Worker 只执行最终决策一次
  -> Output
     rgba / alpha / foreground / metadata / manifest
```

核心行为:

- 无争议样本仍然一键自动输出。
- 有争议样本先展示候选,不预跑多个完整 matte。
- 用户候选和粗 mask 是执行约束,不是后处理擦 alpha。
- 背景归一化统一前置,Analyze 和 Execute 使用同一 `preprocessed_rgb` /
  `background_model`。
- Direct Worker 是 Web/API 主线执行边界。

## 非目标

本迁移不做:

- 不接入 VLM 作为默认语义裁决。
- 不把候选阶段变成三次完整 matting 预览。
- 不重写 PyMatting、CorridorKey 或 ShadowPatch 的核心算法。
- 不新增主线执行后端。
- 不用样本 ID、文件名、固定坐标做特例。
- 不把用户粗 mask 当最终 alpha。

## 阶段 0: 冻结基线

目的: 在重构前知道当前系统怎么坏、怎么好。

任务:

- 固定一组小 smoke 样本:
  - clean RGBA passthrough;
  - 绿/蓝幕 CorridorKey;
  - PyMatting Known-B 硬按钮;
  - 白底角色内部近白争议样本;
  - 假透明网格背景样本;
  - 未知/不稳定背景 fallback。
- 跑现有 `tests/test_web.py`、`tests/test_runtime_capabilities.py`、相关 API 测试。
- 保存一个 `out/` batch,包含 route metadata、执行后端、耗时和主要输出。

验收:

- 有一份可复查的 baseline summary。
- 已记录当前已知失败,尤其 enclosed near-B 主体/孔洞争议。

回滚:

- 此阶段不改行为。

## 阶段 1: 定义数据结构

目的: 先立契约,再搬实现。

新增或明确以下结构:

- `PreprocessAnalysis`
- `PreprocessDecision`
- `BackgroundModel`
- `AnalyzeResult`
- `AmbiguityRegion`
- `SemanticCandidate`
- `SemanticDecision`
- `UserMaskDecision`
- `ExecutionRequest`

最小字段:

```text
PreprocessDecision
  selected preprocess item ids
  applied item ids
  preprocess metadata

AnalyzeResult
  status: ready | needs_decision | unsupported
  route decision
  ambiguity regions
  candidates
  preview asset refs

SemanticCandidate
  id
  label
  default
  confidence
  decision payload
  affected region ids

ExecutionRequest
  preprocess decision
  route decision
  selected candidate id / semantic decision
  optional user mask decision
```

验收:

- 数据结构有单元测试覆盖 JSON roundtrip。
- manifest 草案字段能表达默认候选、最终候选、preprocess 和 mask 摘要。

回滚:

- 结构可以先作为 debug-only 字段挂在现有响应里,不改变执行行为。

## 阶段 2: 抽出 Preprocess

目的: 把原始素材前置加工统一到语义判断之前。

任务:

- 将现有去网格检测纳入 `PreprocessAnalysis`:
  - 上传后检测;
  - 推荐默认勾选;
  - 用户可取消;
  - 记录 applied/selected/debug。
- 将 Known-B 背景场归一化从 executor 私有逻辑迁到 Preprocess:
  - 输入: 原图和背景模型;
  - 输出: `preprocessed_rgb`、`background_model`、归一化 debug;
  - 不裁决主体/孔洞/阴影语义。
- 保证 Analyze 和 Execute 都消费同一份 preprocess 结果。

验收:

- 去网格用户取消后,Analyze 使用未去网格输入。
- Known-B 背景归一化不再在多个 executor 中各跑一套。
- 背景归一化不得消除 enclosed near-B 争议区域。
- manifest/debug 记录 preprocess selected/applied。

测试:

- checkerboard endpoint / Web 表单测试;
- Known-B 背景轻微漂移合成测试;
- 白底角色近白争议保护测试。

回滚:

- 保留旧执行路径开关,允许 executor 临时使用旧归一化,但 debug 必须标明 legacy。

## 阶段 3: 实现 Analyze

目的: 在不执行重型 matting 的情况下输出 route、争议区和候选。

任务:

- 新增轻量 `/api/analyze-candidates` API。
- Analyze 消费 `preprocessed_rgb` / `background_model`。
- 复用现有 `classify_route()` 生成默认 route/profile。
- 首批实现 enclosed near-background 争议检测:
  - `auto_default`;
  - `protect_near_bg_subject`;
  - `cut_enclosed_holes`。
- 生成候选 overlay、trimap 参考图、必要时的 CorridorKey hint 参考图。

验收:

- 无争议样本返回 `status=ready`。
- 白底角色内部近白样本返回 `status=needs_decision`。
- 候选阶段不调用 PyMatting/CorridorKey 重型执行。
- 候选 preview 是 overlay/trimap/hint 参考,不是最终 RGBA。

测试:

- mock executor,断言 Analyze 不触发执行。
- enclosed near-B 合成样本:
  - 透明孔洞;
  - 主体白色材质;
  - 两者都可解释的争议样本。

回滚:

- Web 可继续走旧 `backend=auto` 直接执行,Analyze 只作为 debug 面板。

## 阶段 4: Web/API 前置裁决

目的: UI 从“跑完结果再选”改成“先选语义再跑”。

任务:

- 上传后先跑 Preprocess 检测。
- 用户勾选/取消前置加工后重新 Analyze。
- `ready` 时自动执行默认候选。
- `needs_decision` 时显示:
  - 原图/前置加工后图;
  - 争议区域 overlay;
  - 候选卡片;
  - trimap/hint 参考图。
- 候选按钮触发 Execute。

验收:

- 有争议样本不会在用户选择前跑完整 matte。
- 候选卡片不显示 solver 参数或 debug JSON。
- 用户改变 preprocess 勾选会刷新候选。
- UI 显示默认候选但不隐藏其他候选。

测试:

- `tests/test_web.py` 覆盖:
  - Analyze 响应;
  - needs_decision 页面状态;
  - candidate selection 请求;
  - preprocess 勾选流。

回滚:

- 保留一个隐藏/配置开关允许 Web 走旧直接执行。

## 阶段 5: Execute 消费最终决策

目的: 执行阶段只消费最终 contract,不再重判语义。

任务:

- Execute 接收:
  - `PreprocessDecision`;
  - route decision;
  - selected candidate / `SemanticDecision`;
  - optional `UserMaskDecision`。
- Known-B trimap 构造消费 semantic decision:
  - `protect_near_bg_subject`;
  - `cut_enclosed_holes`。
- 保证 ShadowPatch repair domain 和 semantic decision 对齐。
- 输出 manifest 记录:
  - `preprocess`;
  - `analysis_status`;
  - `default_candidate_id`;
  - `selected_candidate_id`;
  - `semantic_decision`;
  - `user_mask_used`;
  - `execution_backend`;
  - `execution_server_url`。

验收:

- 同一 `analysis_id` + candidate 执行结果可复现。
- executor 不重新估一套背景归一化。
- 被选择为主体保护的区域不进入 sure-bg。
- 被选择为透明孔洞的区域核心进入 forced background/transparent。

测试:

- Known-B 单元测试覆盖两个候选的 trimap 差异。
- manifest/schema 测试覆盖语义字段。
- Web HTTP smoke 确认 metadata 完整。

回滚:

- candidate 缺失时走 `auto_default`。
- legacy 请求不带 candidate 时保持旧行为,但 debug 标记 `semantic_decision=legacy_auto`。

## 阶段 6: 粗 Mask 前置

目的: 候选仍不准时提供低成本人工兜底。

任务:

- Web mask 编辑切换为 `keep/remove` 语义笔刷。
- 生成 `UserMaskDecision`:
  - `keep_mask`;
  - `remove_mask`;
  - optional `unknown_mask`。
- mask 在 Execute 阶段转为:
  - forced subject/protected subject;
  - forced background/transparent;
  - trimap unknown。
- 冲突规则: 后画覆盖先画。

验收:

- `keep_mask` 能覆盖默认透明孔洞。
- `remove_mask` 能覆盖默认主体保护。
- 空 mask 不改变候选。
- 满图 mask 需要 UI 警告或 debug 高风险标记。

测试:

- mask shape/empty/full/conflict 单元测试。
- Web canvas 请求测试。
- 真实样本手工 smoke。

回滚:

- 保留现有 mask debug 模式,但主 UI 只暴露粗裁决语义。

## 阶段 7: 收敛旧 `/api/matte-candidates`

目的: 消除“候选=已跑多个完整结果”的旧语义。

任务:

- 将旧 `/api/matte-candidates` 标记为兼容层。
- 新增或稳定:
  - `/api/preprocess-analysis`;
  - `/api/analyze-candidates`;
  - `/api/execute-candidate`。
- Web 后台列表显示:
  - preprocess 状态;
  - selected candidate;
  - mask 使用;
  - route/profile;
  - execution backend。
- 更新 eval manifest writer。

验收:

- 新流程 endpoint 覆盖 Web 主路径。
- 旧 endpoint 仍能服务脚本,但 metadata 指向新语义字段。
- 后台列表可以按语义候选状态筛选/浏览。

测试:

- `tests/test_web.py`;
- artifact/manifest tests;
- batch eval summary tests。

回滚:

- 旧 endpoint 保留一段时间,但不再作为 UI 主入口。

## 阶段 8: 文档与清理

任务:

- 更新:
  - `docs/architecture.md`;
  - `docs/semantic-candidate-workflow.md`;
  - `docs/ermbg-route-strategy.md`;
  - `docs/install-startup.md`;
  - `docs/README.md`。
- 删除或标记过时 debug 文案。
- 清理重复背景归一化代码。
- 清理 Web 中“候选结果”旧命名。

验收:

- 主线文档只讲一个流程。
- 专题文档只补充细节。
- 搜索 `candidate` 不再把执行结果候选和语义候选混在一起。

## 验证矩阵

每个阶段至少保留以下验证:

| 类别 | 样本 | 预期 |
|---|---|---|
| clean RGBA | 已有透明 PNG | passthrough |
| 去网格 | 假透明棋盘格 | Preprocess 推荐去网格 |
| Known-B 硬边 | 绿/白/灰纯背景 UI | ready 或正确候选 |
| enclosed near-B | 白底角色内部白色 | needs_decision |
| 透明孔洞 | UI 镂空/文字洞 | 可选 cut holes |
| CorridorKey | 绿/蓝幕角色/图标/玻璃 | route 到对应 profile |
| fallback | 不稳定背景 | pymatting fallback |
| mask | keep/remove 粗笔刷 | 覆盖候选决策 |

## 完成定义

迁移完成需要同时满足:

- Web 主路径使用 `Preprocess -> Analyze -> Decide -> Execute`。
- 有争议样本在执行前展示语义候选。
- 候选阶段不跑多个完整 matte。
- Preprocess 统一承载去网格和 Known-B 背景场归一化。
- Execute 只运行一次最终决策。
- manifest 记录 preprocess、candidate、mask、route 和 execution backend。
- 主线文档和测试都更新。
