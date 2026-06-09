# Route / Profile 模块

本文对齐当前 `ermbg.router`、Analyze、Web 和 Direct Worker 的 route/profile 契约。

## 文件

- `ermbg/router.py`
- `ermbg/analyze.py`
- `ermbg/direct_worker.py`
- `ermbg/direct_worker_server.py`
- `tests/test_router.py`
- `tests/test_analyze.py`

## RouteDecision

Route 只描述算法、素材类型、参数和证据，不描述 server URL。

核心字段：

- `route`
- `asset_kind`
- `backend`
- `params`
- `confidence`
- `reasons`
- `analysis`

`RouteDecision.to_dict()` 派生：

- `algorithm`
- `parameter_profile`
- `execution_profile`
- 可选 `corridorkey_analysis`

## RouteCandidate

`build_route_candidates()` 输出完整 route/model candidates。每个 candidate 包含一个可执行
`RouteDecision`：

- `id`
- `algorithm` / `route` / `backend`
- `asset_kind`
- `execution_profile`
- `parameter_profile`
- `params`
- `confidence`
- `evidence`
- `risks`
- `default`

`select_default_route_candidate()` 选择默认候选。`classify_route()` 是兼容 wrapper，
等价于返回默认 route candidate 的 `RouteDecision`。

## 当前 Algorithm

- `pymatting_known_b`: 已知背景硬边图形/UI 主路径。
- `corridorkey`: 绿幕/蓝幕复杂软边、透明材质、角色和细节边界路径。
- `known_bg_glow`: 已知背景 glow 专用路径。
- `rgba_passthrough`: 输入已有 alpha 且可复用。
- fallback: 本地兼容 PyMatting 路径。

## 决策顺序

profile 是 route 的结果标签，不是模型选择输入。

顺序：

1. clean alpha -> `rgba_passthrough`。
2. 判断背景是否 known/measurable/stable；不可信则 fallback。
3. 成像模型选择：
   - continuous emissive/falloff glow -> `known_bg_glow`
   - hard opaque + AA -> `pymatting_known_b`
   - complex linear mix、透明材质、fine-detail boundary -> `corridorkey`
   - unknown / unstable B -> fallback
4. 模型内参数 recipe：
   - Known-B: BG-seed outline trimap、same-key body、hole decisions、unknown 分布
   - CorridorKey: execution profile、despill/refiner/despeckle、常量 hint 强度
   - Glow: single-target-line、adaptive-ray、chromatic-swap-ray 参数
5. Analyze 的局部语义候选只能生成 semantic decision 或参数 override，不能反向重选模型。

宽高比、padding、bbox width/height fraction 只能用于阈值归一化、最小面积或异常防守，
不能作为模型语义判断的核心证据。

## Route 争议

当 known screen + stable B 同时支持 Known-B hard opaque solver 和 CorridorKey
composite/detail solver 时，Analyze 必须输出多个 route candidates，而不是在 Execute
阶段重新猜。

典型约束：

- B056 ornate hole plate 应保留 Known-B hole candidate，不因内部暗纹误触 CorridorKey
  或 shadow 语义候选。
- B057 same-key hard shadow 由 opaque Known-B guard 压回 Known-B。
- C 类角色或透明细节素材可由 CorridorKey 成为默认 route。

## 执行约束

- route 在 matting 前完成。
- Direct Worker URL 由配置选择，不属于 route。
- profile 专属调参必须落在共享 router/执行代码中，不要在 Web JS 中重写。
- Execute 消费显式 `route_decision`，不得重新推断 asset kind。
- CorridorKey 当前只通过 `corridorkey_hint_value` 或显式 `corridorkey_hint_mask`
  改变 hint，不使用输出后 alpha 修补来改变结果。
