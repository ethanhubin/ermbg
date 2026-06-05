# Preprocess 模块

本文对齐当前 `ermbg.preprocess` 实现。

## 文件

- `ermbg/preprocess.py`
- `tests/test_preprocess.py`
- Web API: `/api/preprocess-analysis`

## 职责

Preprocess 是语义判断之前的输入清理阶段。它只处理输入观测问题,不裁决主体、孔洞、
阴影或半透明材质归属。

当前支持:

- `remove_checkerboard`: 检测并可选归一化假透明棋盘格背景。
- `normalize_known_background`: Known-B 背景场归一化 helper,由 Analyze 选择并传给 Execute。

## API

`analyze_input_preprocess(image_srgb)` 返回 `PreprocessAnalysis`:

- `preprocess_id`;
- `items`;
- `background_model`;
- `debug.checkerboard`。

`apply_input_preprocess(image_srgb, selected=[...])` 返回 `PreprocessResult`:

- `image_srgb`;
- `analysis`;
- `decision`。

`normalize_known_background_preprocess(image_srgb, background_color, ...)` 返回:

- normalized RGB;
- `PreprocessDecision`。

## 主线约束

- Preprocess 必须先于 route/semantic Analyze。
- 用户关闭某个前置加工时,Analyze 和 Execute 都必须消费同一份未应用该加工的输入。
- Known-B 背景归一化结果要作为 contract 传入 Execute,executor 不再私有重跑另一套归一化。
- Preprocess 输出可以影响可观测背景模型,但不能把内部近背景色区域直接判为主体或孔洞。

## 当前缺口

- `/api/preprocess-analysis` 目前主要暴露去网格建议;Known-B 背景归一化仍主要由 Analyze 内部基于 route 选择。
- `analysis_id` 仍未稳定生成,后续可用于跨请求缓存和审计。
