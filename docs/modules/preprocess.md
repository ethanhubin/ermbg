# Preprocess 模块

本文对齐当前 `ermbg.preprocess` 实现。

## 文件

- `ermbg/preprocess.py`
- `tests/test_preprocess.py`
- Web API: `/api/preprocess-analysis`

## 职责

Preprocess 是语义判断之前的输入清理阶段。它只处理输入素材本身的可观测问题，不裁决主体、孔洞、阴影或半透明材质归属。

当前外部决策项只有：

- `background_repair`: 背景修复。启用后统一允许假透明棋盘格背景归一化，以及 Known-B 路径的已知背景场归一化；关闭后两者都不应用。

棋盘格检测、棋盘格归一化、Known-B 背景场归一化是 `background_repair` 下的机制细节，可以写入 metadata/debug，但不再作为独立用户候选或独立 preprocess item 暴露。

## API

`analyze_input_preprocess(image_srgb)` 返回 `PreprocessAnalysis`:

- `preprocess_id`
- `items`
- `background_model`
- `debug.checkerboard`

`apply_input_preprocess(image_srgb, selected=[...])` 返回 `PreprocessResult`:

- `image_srgb`
- `analysis`
- `decision`

`repair_known_background_preprocess(image_srgb, background_color, ...)` 返回:

- normalized RGB
- `PreprocessDecision`，其 `selected/applied` 使用 `background_repair`

## 主线约束

- Preprocess 必须先于 route/semantic Analyze。
- 用户关闭 `background_repair` 时，Analyze 和 Execute 都必须消费未应用背景修复的输入。
- Known-B 背景场归一化结果要作为 contract 传入 Execute，executor 不再私有重跑另一套背景归一化。
- Preprocess 输出可以影响可观测背景模型，但不能把内部近背景色区域直接判为主体或孔洞。

## 当前状态

- `/api/preprocess-analysis` 暴露 `background_repair` 决策和机制 debug。
- Analyze 会在 Known-B route 且 `background_repair` 被选中时执行 Known-B 背景场归一化，并把结果合并进 preprocess contract。
