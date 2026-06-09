# Analyze / Decide 模块

本文对齐当前 `ermbg.analyze` 和 Web 候选 UI。

## 文件

- `ermbg/analyze.py`
- `ermbg/web.py`
- `tests/test_analyze.py`
- `tests/test_web.py`

## Analyze 入口

```python
analyze_candidates(
    image_srgb,
    preprocess=None,
    screen_mode="auto",
    preset="auto",
    fallback_background_color=(0, 200, 0),
)
```

Analyze 是轻量阶段，不调用完整 matting backend。

当前步骤：

1. 调用 `router.build_route_candidates()` 生成 route/model candidates。
2. 用 `router.select_default_route_candidate()` 选择默认 route。
3. 将默认 route 写入 `AnalyzeResult.route` 兼容字段。
4. 将所有 route candidates 写入 `AnalyzeResult.route_candidates[]`。
5. 对每个 route candidate 生成绑定该 route 的 semantic candidates。
6. 生成稳定 `analysis_id`。
7. 生成服务端 preview assets。

`ready` 表示可直接执行默认候选；`needs_decision` 表示存在 route 或语义候选需要选择。

## Known-B Analyze

Known-B Analyze 会生成 explicit trimap preview。该 preview 不是最终 RGBA，
但可以作为 Execute 输入复用。

当前 Known-B preview 思路：

```text
known B normalization
  -> exterior BG seed
  -> inward search until color/ownership break
  -> subject outline
  -> fill outline as FG core
  -> boundary / transition / shadow-facing unknown
  -> apply selected hole policy overlay
```

候选来源：

- `enclosed_near_background`: 封闭近背景色区域，可能是透明孔洞，也可能是主体材质；
- `button_body_subject_ownership`: same-key opaque body 特例；
- CorridorKey route 的 screen material / translucency 风险。

shadow 不再生成 semantic candidate。shadow-like evidence 只作为 Known-B trimap
builder 内部的边界 unknown 证据。

## 当前语义候选

无高影响争议：

- `auto_default`

内部近背景色争议：

- `protect_near_bg_subject`
- `cut_enclosed_holes`
- button 单孔洞：`use_cut_hole_0` / `use_keep_hole_0`
- button 多孔洞：`use_cut_all_holes` / `use_keep_all_holes`

same-key opaque body：

- `use_opaque_body`
- `use_standard_body` 或组合候选中的 standard body 备选

CorridorKey：

- `corridorkey_hint_000`
- `corridorkey_hint_016`
- `auto_default`，默认 `0.32`
- `corridorkey_hint_050`
- `corridorkey_hint_070`

## Candidate Decision

Known-B hole decision:

```json
{
  "enclosed_near_bg_region_policies": {
    "ambiguous_enclosed_bg_0": "transparent_hole"
  },
  "enclosed_near_bg_policy": "transparent_hole"
}
```

same-key body decision:

```json
{
  "button_body_policy": "opaque_subject",
  "pymatting_trimap_mode": "same_key_opaque_body_outline",
  "pymatting_unknown_grow_px": 2
}
```

CorridorKey constant-hint decision:

```json
{
  "policy": "corridorkey_constant_hint",
  "corridorkey_hint_value": 0.32
}
```

CorridorKey 候选通过 Execute 阶段生成全帧常量 hint mask 并送入模型，
不通过输出后的硬 alpha 约束生效。

## Preview Assets

Preview assets 是服务端生成的轻量图：

- Known-B: `trimap`，三态 `0/128/255`，带 `execution_role=pymatting_explicit_trimap`；
- Known-B hole candidates: trimap 上叠加 region policy；
- CorridorKey: `hint`，带 `execution_role=corridorkey_hint_mask`，metadata 记录
  `corridorkey_hint_value` 和 hint policy；
- 通用: `overlay`、`region_mask:*`。

Preview 用于 Decide 和 manifest 审计，不代表最终 matte。Web 点击候选只切换
preview，不会执行。

## Decide UI

Web 主线：

1. 上传图片。
2. 调用 `/api/preprocess-analysis`。
3. 调用 `/api/analyze-candidates`。
4. 展示 route/semantic candidates。
5. 用户选择候选和可选 keep/remove mask。
6. 调用 `/api/execute-candidate`。

Execute request 会根据 `selected_candidate_id` 找到 semantic candidate 绑定的
`route_candidate_id`，再从 `route_candidates[]` 取完整 route。Direct Worker
消费这个显式 route，不重新推断素材类别。

## 用户 Mask

Web mask 是语义约束：

- keep mask 强制主体；
- remove mask 强制背景；
- remove 覆盖 keep；
- 空 mask 不改变候选。

mask 不是最终 alpha。
