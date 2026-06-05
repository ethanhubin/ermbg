# Route / Profile 模块

本文对齐当前 `ermbg.router`、Web 和 Direct Worker 的 route/profile 契约。

## 文件

- `ermbg/router.py`
- `ermbg/analyze.py`
- `ermbg/direct_worker.py`
- `ermbg/direct_worker_server.py`
- `tests/test_router.py`
- `tests/test_analyze.py`

## RouteDecision

route 决策描述算法和参数,不描述 server URL。

核心字段:

- `route`;
- `asset_kind`;
- `backend`;
- `params`;
- `confidence`;
- `reasons`;
- `analysis`。

`RouteDecision.to_dict()` 会派生:

- `algorithm`;
- `parameter_profile`;
- `execution_profile`;
- 可选 `corridorkey_analysis`。

## 当前 algorithm / backend

- `pymatting_known_b`: 已知背景图形和硬 UI 主路径。
- `corridorkey`: 绿幕/蓝幕复杂 UI、软边、glow、透明材质等路径。
- `known_bg_glow`: 已知背景 glow 专用路径。
- `rgba_passthrough`: 输入已有 alpha 且可直接复用。
- fallback: 本地 PyMatting 等兼容路径。

## 约束

- route 在 matting 执行前完成。
- Direct Worker server URL 由配置选择,不是 route 决策的一部分。
- profile 专属调参必须落在共享 router/执行代码中,不要在 Web JS 中重写。
- Execute 阶段消费 `route_decision`,不得重新推断 asset kind。

## 当前缺口

- 个别样本族仍存在 route 分类回归风险。最近全量 Direct Worker server 测试中,
  `test_direct_worker_manual_known_bg_glow_preserves_chromatic_swap_ray_mode` 显示一个
  glow 样本当前被 route 到 `pymatting_known_b`,而测试期望 `known_bg_glow`。
  这应作为 route/profile 回归单独处理。
