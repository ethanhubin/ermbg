# Preprocess 模块

本文对齐当前 `ermbg.preprocess` 实现。

## 文件

- `ermbg/preprocess.py`
- `tests/test_preprocess.py`
- Web API: `/api/preprocess-analysis`

## 职责

Preprocess 是语义判断之前的输入清理阶段。它只处理输入素材本身的可观测问题，不裁决
主体、孔洞、阴影或半透明材质归属。

当前唯一外部决策项:

- `background_repair`

启用后允许:

- 假透明棋盘格背景修复；
- Known-B 背景场归一化；
- 相关机制 debug/metadata 写入 contract。

关闭后 Analyze 和 Execute 都必须消费未应用修复的同一输入。

## API

`analyze_input_preprocess(image_srgb)` 返回:

- `preprocess_id`;
- `items`;
- `background_model`;
- `debug.checkerboard`。

`apply_input_preprocess(image_srgb, selected=[...])` 返回:

- `image_srgb`;
- `analysis`;
- `decision`。

`repair_known_background_preprocess(image_srgb, background_color, ...)` 返回:

- normalized RGB；
- `PreprocessDecision`，其 `selected/applied` 使用 `background_repair`。

## 主线约束

- Preprocess 先于 route/semantic Analyze。
- Known-B 背景归一化结果要传入 Execute。
- executor 不再私有重跑另一套背景归一化。
- Preprocess 不能把内部近背景色区域判为主体或孔洞。

## 当前状态

- `/api/preprocess-analysis` 暴露 `background_repair` 推荐和 debug。
- Analyze 在 Known-B route 且 `background_repair` 被选中时执行 Known-B 背景场归一化，
  并把结果合并进 preprocess contract。
