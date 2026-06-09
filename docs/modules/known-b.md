# PyMatting Known-B 模块

本文对齐当前 Known-B 主线实现。

## 文件

- `ermbg/pymatting_refine.py`
- `ermbg/api.py`
- `ermbg/analyze.py`
- `ermbg/preprocess.py`
- `tests/test_pymatting_refine.py`
- `tests/test_analyze.py`

## 职责

Known-B 用于背景色已知或可稳定估计的图形/UI 素材。它的核心不是直接给每个像素做
FG/BG 分类，而是构造能让 PyMatting 正确求解 alpha 的 trimap。

主线职责分工:

- Preprocess 可做背景场归一化；
- Analyze 生成 Known-B explicit trimap preview；
- Decide 只裁决真实语义争议；
- Execute 消费 explicit trimap、semantic decision 和 user mask；
- executor 不重新推断 hole、shadow 或素材类别。

## 当前 Trimap 算法

当前 Known-B trimap 主线是 BG-seed outline:

1. 以已知背景色和归一化后的输入建立强置信 BG。
2. 只从外部 BG seed 出发向主体方向搜索。
3. 当 BG 证据遇到明显色彩/归属断层时，认为那里是 subject outline。
4. outline 内部连通填充为 FG core。
5. outline 边缘、抗锯齿、transition 和 shadow-facing 区域进入 unknown。
6. 剩余明确外部背景为 sure-BG。

这个算法的约束是“从 BG 往内找边界”，不是从暗线、阴影或主体内部纹理出发找 FG。
因此内部金属纹、黑线、暗装饰不会因为颜色像 shadow 就扩散成主体内部 unknown。

## Unknown 第一性原则

PyMatting 真正求解的是 unknown 区:

- `sure_bg` 是背景边界条件；
- `sure_fg` 是主体边界条件；
- `unknown` 承载抗锯齿、半透明、阴影和边缘求解。

任何 Known-B 调参都必须回答:

- 这次改动如何改变 unknown 的形状和连通性？
- unknown 是否同时接触可靠 BG 和主体色证据？
- shadow-facing unknown 是否与 outline / shadow 区域绑定？
- 是否避免了内部暗线、孔洞边缘和远离边界的纹理被误放进 unknown？

如果解释只是“把更多像素标成 FG/BG”，通常说明机制没有落到 PyMatting 求解路径上。

## Hole 候选

enclosed near-B islands 是当前 Known-B 主要语义候选。

Analyze 会检测不连通外部背景、但颜色接近已知 B 的封闭区域:

- 单个干净孔洞：`use_cut_hole_0` / `use_keep_hole_0`；
- 多个重复孔洞：`use_cut_all_holes` / `use_keep_all_holes`；
- 非 button/兼容图形可保留 `protect_near_bg_subject` /
  `cut_enclosed_holes` 形式。

候选 decision:

```json
{
  "enclosed_near_bg_region_policies": {
    "ambiguous_enclosed_bg_0": "transparent_hole"
  },
  "enclosed_near_bg_policy": "transparent_hole"
}
```

或:

```json
{
  "enclosed_near_bg_region_policies": {
    "ambiguous_enclosed_bg_0": "subject"
  },
  "enclosed_near_bg_policy": "subject"
}
```

候选只 overlay 对应 hole region:

- `transparent_hole` -> 该 region 强制 `sure_bg`；
- `subject` -> 该 region 强制 `sure_fg`；
- 周边边缘仍由 BG-seed outline trimap 决定。

## Shadow 规则

shadow 当前不是语义候选。旧的 `button_shadow_ownership`、
`shadow_ownership_policy`、`solve_shadow`、`keep_dark` 已从主线清理。

shadow-like evidence 只在 trimap builder 内部使用:

- 在靠近 outline 的 shadow 区域外扩/内扩 unknown；
- 扩展必须与 outline / shadow evidence 连通；
- 扩展方向必须绑定该 shadow 区域；
- 不允许在主体内部沿暗线乱跑。

如果普通按钮阴影结果不好，优先检查:

- BG seed 是否稳定；
- outline 是否闭合并贴真实主体边界；
- shadow-facing unknown 是否足够靠近 outline；
- inward release 是否只发生在与 shadow/outline 联通的边界上。

不要重新把 shadow 做成用户候选，除非重新设计了能避免内部暗纹误伤的机制。

## Same-Key Body 特例

普通硬边按钮使用 standard body。same-key opaque body 是特例:

- 主体颜色与 known-B 背景同色系或非常接近；
- standard trimap 难以识别主体 core；
- route/profile 有 same-key opaque plateau 证据；
- shadow anchor 只作为 body tracing 的方向提示，不生成候选。

候选可通过:

```json
{
  "button_body_policy": "opaque_subject",
  "pymatting_trimap_mode": "same_key_opaque_body_outline",
  "pymatting_unknown_grow_px": 2
}
```

普通阴影按钮不应通过 opaque body 修复。

## Execute 输入

Known-B Execute 可消费:

- `pymatting_explicit_trimap`;
- `semantic_decision`;
- `user_keep_mask`;
- `user_remove_mask`;
- Known-B thresholds 和 solver params。

背景场归一化只属于 Preprocess。Execute 收到的图像应已经是 preprocess 后的 RGB；
executor 不接收归一化开关，也不读取归一化 metadata。

当 `pymatting_explicit_trimap` 存在时，executor 直接使用该三态 trimap，并在其上应用
user mask。remove mask 覆盖 keep mask。

## 调参规则

新增或修改 Known-B 机制时，代码注释必须说明思路:

- 使用了什么可观测证据；
- 该证据如何约束 BG、outline、FG core 或 unknown；
- 保护的失败模式是什么；
- 哪些情况刻意不处理。

参数必须说明含义，例如面积比、距离、连通域、过渡带宽、shadow inward 像素等。
禁止围绕样本 ID、文件名、坐标或单一颜色做特例修复。
