# Analyze / Decide 模块

本文对齐当前 `ermbg.analyze` 和 Web 候选 UI 实现。

## 文件

- `ermbg/analyze.py`
- `ermbg/web.py`
- `tests/test_analyze.py`
- `tests/test_web.py`

## Analyze

入口:

```python
analyze_candidates(
    image_srgb,
    preprocess=None,
    screen_mode="auto",
    preset="auto",
    fallback_background_color=(0, 200, 0),
)
```

当前行为:

1. 调用共享 `router.classify_route()` 生成 route/profile。
2. 将 `RouteDecision` 转成 `AnalyzeResult.route`,包含:
   - `algorithm`;
   - `route`;
   - `backend`;
   - `asset_kind`;
   - `parameter_profile`;
   - `execution_profile`;
   - `confidence`;
   - `reasons`;
   - `params`;
   - `analysis`;
   - `corridorkey_analysis`。
3. 对 PyMatting Known-B 路径,基于已知背景色运行 Known-B 背景归一化 preprocess helper。
4. 检测 `enclosed_near_background` 争议区域。
5. 对 CorridorKey 的同幕布色/半透明材质风险,输出
   `screen_material_or_translucency` 争议区域。
6. 生成稳定 `analysis_id`。
7. 为候选生成轻量服务端 preview assets。`overlay` 是通用预览;
   PyMatting Known-B 候选输出 `trimap`;CorridorKey 候选输出 `hint`。
   Known-B `trimap` 是三态 `0/128/255` PNG,标记
   `execution_role=pymatting_explicit_trimap`,选中候选后可由 Execute 直接消费。
8. 输出 `ready` 或 `needs_decision`。

## 当前语义候选

无高影响争议:

- `auto_default`

内部近背景色争议:

- `auto_default`;
- `protect_near_bg_subject`;
- `cut_enclosed_holes`。

同幕布色/半透明材质争议:

- `auto_default`;
- `preserve_screen_material`;
- `remove_screen_tint`。

候选 `decision` 当前用于 Known-B trimap 语义约束:

- `{"policy": "auto_default"}`;
- `{"enclosed_near_bg_policy": "subject"}`;
- `{"enclosed_near_bg_policy": "transparent_hole"}`。

CorridorKey 风险候选当前表达为:

- `{"screen_material_policy": "preserve"}`;
- `{"screen_material_policy": "background"}`。

## Decide UI

Web 主线:

1. 上传图片。
2. 调用 `/api/analyze-candidates`。
3. 渲染语义候选。
4. 用户点击候选时只切换候选预览,不会执行 matte。
5. 预览优先使用 Analyze payload 中的服务端 `preview_assets`,缺失时前端可退回
   bbox 级绘制。UI 不再提供 `Overlay / Trimap / Hint` 手动切换;PyMatting
   Known-B 候选默认显示原图加 trimap unknown 红色半透明蒙层,只有透明洞候选
   会再叠加 overlay;CorridorKey 候选默认显示 hint。
6. 用户点击“确定抠图”后调用 `/api/execute-candidate`。

候选预览必须便宜,不能调用 PyMatting、CorridorKey 或远端重型模型。

## 粗 Mask

Web mask 是语义约束:

- keep mask 强制主体;
- remove mask 强制背景;
- remove 覆盖 keep 冲突;
- 空 mask 不改变候选。

mask 不代表最终 alpha。

## 当前缺口

- shadow 归属候选仍待继续扩展。
- `ready` 样本目前也进入候选确认式 UI,后续可按产品取舍恢复一键自动执行默认候选。
