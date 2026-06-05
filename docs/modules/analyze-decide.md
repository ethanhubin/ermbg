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
5. 输出 `ready` 或 `needs_decision`。

## 当前语义候选

无高影响争议:

- `auto_default`

内部近背景色争议:

- `auto_default`;
- `protect_near_bg_subject`;
- `cut_enclosed_holes`。

候选 `decision` 当前用于 Known-B trimap 语义约束:

- `{"policy": "auto_default"}`;
- `{"enclosed_near_bg_policy": "subject"}`;
- `{"enclosed_near_bg_policy": "transparent_hole"}`。

## Decide UI

Web 主线:

1. 上传图片。
2. 调用 `/api/analyze-candidates`。
3. 渲染语义候选。
4. 用户点击候选时只切换候选预览。
5. 预览支持 `Overlay / Trimap / Hint`。
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

- Analyze 目前只实现了 `enclosed_near_background` 高争议类型。
- `preview_assets` 仍为空,Web 现在用 bbox 在前端临时绘制 overlay/trimap/hint。
- `analysis_id` 未稳定生成。
- `ready` 样本目前也进入候选确认式 UI,后续可按产品取舍恢复一键自动执行默认候选。
