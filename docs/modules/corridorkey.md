# CorridorKey 模块

本文对齐当前 CorridorKey 路径、游戏素材样本和验证方式。

## 文件

- `ermbg/corridorkey.py`
- `ermbg/corridorkey_runner.py`
- `ermbg/direct_worker.py`
- `ermbg/direct_worker_client.py`
- `ermbg/direct_worker_server.py`
- `ermbg/router.py`
- `ermbg/web.py`
- `samples/corridorkey_semantic/manifest.json`
- `tests/test_analyze.py`
- `tests/test_direct_worker.py`
- `tests/test_direct_worker_server.py`
- `tests/test_router.py`

## 职责

CorridorKey 用于复杂绿幕/蓝幕素材，尤其是:

- 软边和 feather；
- glow / particle / mist；
- 透明或半透明 UI；
- 角色发丝、毛发、透明布料；
- 同幕布色材质风险较高的素材。

主线中 CorridorKey 由 route/profile 选择，并由 Direct Worker 执行。Web 不持有
CorridorKey 私有 route 逻辑。

当前执行 profile:

- `corridorkey-shaped-icon`: shaped icon / soft boundary detail；
- `corridorkey-effect-icon`: screen-tinted translucency / effect icon；
- `corridorkey-transparent-button`: 半透明或玻璃按钮；
- `corridorkey-character`: 角色、发丝、毛发、透明布料和复杂细节复合素材。

硬边不透明 UI、同幕色实体 UI 和稳定已知背景优先回到 PyMatting Known-B；当 Known-B
和 CorridorKey 都有证据时，Analyze 暴露多个 route candidates，由默认 route 或用户选择
决定执行模型。

## 输入

CorridorKey direct 路径需要:

- `corridorkey_analysis`;
- `params`;
- 可选 `corridorkey_hint_mask`;
- 可选 `semantic_decision`;
- 可选 user masks。

当 Web 通过 `route_decision` 调用 Direct Worker 时，`corridorkey_analysis` 必须随
Analyze route 一起传递。

Direct Worker 的 `route_decision` 是执行事实来源。未显式传入的
`corridorkey_screen_mode`、`corridorkey_preset`、`corridorkey_hard_ui_hint_mode`
不得覆盖 route params；只有无 `route_decision` 的兼容调用或手动参数执行才使用表单默认值。

CorridorKey 语义候选当前只作为分析/预览信号，不对最终 alpha 做自动
floor/cap/remove 约束。单图启发式硬阈值不具备足够普适性，也容易造成 alpha
断裂；若要改变 CorridorKey 输出，应走连续 hint/profile 实验并单独验证。

user keep/remove masks 也在 CorridorKey 分支执行，remove 覆盖 keep。

## Analyze 候选

CorridorKey 当前可输出 `screen_material_or_translucency`、`glass_core_transparency`
和 `soft_alpha_gradient` 风险区域。Analyze 不再只给 review-only 候选，而是把
CorridorKey 的争议解法表达成可执行的 hint variant:

- `feature_balanced`: 默认适中解；
- `feature_translucent`: 更透明的候选；
- `feature_conservative`: 更保留前景的候选；
- `feature_internal_opaque`: 内部硬边争议区更接近不透明的候选。

候选 preview 中的 hint 与 Execute 阶段使用同一个 `ermbg/corridorkey_hint.py`
生成逻辑。Execute 通过 `semantic_decision.corridorkey_hint_variant` 生成
`corridorkey_hint_mask` 并传给 Direct Worker；候选不通过输出后的硬 alpha
floor/cap/remove 约束生效。

## Hint 生成规范

CorridorKey 路径的半透明控制应通过 hint image / profile 进入求解过程，而不是
在模型输出后用硬 alpha 阈值改写结果。它与 Known-B 路径用 trimap 控制 unknown
区域是同一类原则：先给求解器一个结构化先验，再比较不同先验下的 alpha。

当前实验模块:

```text
ermbg/corridorkey_hint.py
scripts/probe_corridorkey_hint_influence.py
tests/test_corridorkey_hint.py
```

hint 特征检测必须保持位置无关，不能假设透明区域在中间。候选区域来自:

- `key_alpha` 中间态：既不是纯背景，也不是强前景；
- `subject_support` 连通主体支持域；
- `hard_subject` 高置信前景支持；
- `translucent_candidate` 半透明/同幕色材料候选；
- `soft_boundary_candidate` 与外部背景相邻的软边界候选。

当前用于测试 CorridorKey 响应的 hint variants:

- `current_default_prior`: 复现当前 `corridorkey-character` 全帧 soft prior；
- `feature_balanced`: 适中默认候选，从 `chromatic_key_alpha` 生成软边 unknown ring；
- `feature_conservative`: 略微扩大 soft support，并提高半透明/反光区域前景支持；
- `feature_internal_opaque`: 外边缘仍交给 CorridorKey，但把主体内部硬边透明争议区
  提高到接近不透明支持；
- `feature_translucent`: 缩窄 soft support，并降低半透明/反光区域前景支持。

`full_frame_zero` 是字面全帧黑/零值 CorridorKey hint，仅作为诊断项，必须用探针脚本的
`--include-full-frame-zero-diagnostic` 显式打开，不作为推荐候选。
CorridorKey runner 不再对 hint 做来源/白色特殊反转；源头生成的 hint 数值就是送入
CorridorKey mask 的语义。路由 profile 需要全帧 soft prior 时，源头直接生成常量
`0.32`；需要 zero prior 时，源头直接生成常量 `0.0`。

`bbox+2` 曾作为大图速度实验项验证。实测没有带来稳定加速，且作为全矩形
aggressive hint 会改变解的形态，因此不再作为候选输出。若后续要做性能优化，
应在 executor 层做真正的 ROI crop/uncrop，而不是只把 hint 外围置零。

纯常量 hint 强度实验显示：低到中等强度的全帧常量输入 hint 对 CorridorKey 输出
影响很弱；全帧零值 hint 只作为诊断，不表达候选解释。
候选应优先来自 feature-driven hint，而不是全帧常量强度。

单图确定性纯色背景的 hint 生成规范:

- `outline_mask` 只描述主体支持拓扑，不直接等同于可控制域；
- 内部透明控制必须先生成更保守的 `control_outline_mask`：它来自离外部背景足够远的
  近幕色/半透明证据，闭合后裁剪在主体内部；在 I021-B 这类样本上，应落在金属边框内部；
- 没有稳定 `control_outline_mask` 时不得启用内部透明控制；
- 常规 feature hint 主干是 `control_outline_mask` 的 soft matte，不直接把 `key_alpha`
  的细纹理写进 hint；
- `control_outline_mask` 边缘必须软过渡，外边缘细节交给 CorridorKey 自己求解；
- `control_outline_mask` 外必须为 0，内部透明候选不得影响其外部；
- 半透明、反光、同幕色污染区域保留为灰度 unknown，而不是直接挖空；
- 若近幕色/透明区域位于主体支持域内部、且离外轮廓较远，它是候选争议区；
  默认可适中解释，另给 `feature_internal_opaque` 让用户快速判断“内部透明是否应保留为不透明材质”；
- 主体 outline 外只做小半径外扩和软化，允许包住发丝/边缘混色，但不能扩成 bbox
  矩形或全帧前景；
- 候选差异来自外扩半径、软化宽度、半透明 floor 和 alpha 曲线，而不是样本位置或
  单个区域硬阈值。

这些 variant 是 Analyze -> Execute 主路径的候选契约。修改默认前，仍需先跑
`scripts/probe_corridorkey_hint_influence.py --run-remote` 做离线机制验证，再跑
`/api/analyze-candidates` 和 `/api/execute-candidate` 的真实 HTTP 主路径，确认
Web 候选与 probe 使用同一套 hint、差异足够大且没有明显 alpha 断裂或背景残留扩散。

## 样本验证

规范样本集:

```text
samples/corridorkey_semantic/manifest.json
samples/corridorkey_semantic/sheets/full_samples_v1_sheet.jpg
```

当前 manifest 全量为 87 个样本：button 57、icon 21、character 9；green 58、blue 29。
蓝幕 route/profile 已接入路由与 Direct Worker，但 `corridorkey.py` 仍将蓝幕标记为
analysis-ready / 需要继续验证的质量风险。

批量测试某一算法路径时必须固定 execution backend，例如:

```text
--fixed-execution-backend direct-pymatting-known-b
```

不要用“当前 auto route 会到这个 backend”代替固定路线。
