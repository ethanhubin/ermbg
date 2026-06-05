# Pipeline Contracts

本文对齐当前 `ermbg.pipeline_contracts` 实现。

## 文件

- `ermbg/pipeline_contracts.py`
- `tests/test_pipeline_contracts.py`

## 核心对象

### Preprocess

- `BackgroundModel`: 可观测背景证据，包含 `color`、`source`、`confidence`、`metadata`。
- `PreprocessItem`: 可推荐的前置加工项，例如 `background_repair`。
- `PreprocessAnalysis`: 前置分析结果，包含 `preprocess_id`、`items`、`background_model`、`debug`。
- `PreprocessDecision`: 最终应用的前置加工决策，包含 `selected`、`applied`、`metadata`、`background_model`。

`background_repair` 是当前唯一主线 preprocess 决策项。checkerboard 与 Known-B background normalization 是该项下的机制 metadata。

### Analyze

- `AmbiguityRegion`: 争议区域，包含 `id`、`type`、`bbox_xyxy`、`area_px`、`mask_ref`、`evidence`、`ambiguity`。
- `SemanticCandidate`: 语义候选，包含 `id`、`label`、`intent`、`decision`、`regions`、`preview`、`reasons`。
- `AnalyzeResult`: Analyze 输出，包含稳定 `analysis_id`、`status`、`route`、`ambiguity_regions`、`candidates`、`default_candidate_id`、`preprocess`、`preview_assets`。

`AnalyzeResult.status` 当前取值：

- `ready`
- `needs_decision`
- `unsupported`

Preview assets 是轻量预览，不是已执行 matte。PyMatting Known-B 候选可以携带 explicit trimap 作为 Execute 输入复用；CorridorKey 候选可以携带 hint。

### Decide / Execute

- `SemanticDecision`: 最终候选裁决。
- `UserMaskDecision`: 粗 keep/remove mask 的引用和摘要。
- `ExecutionRequest`: Execute 阶段唯一输入 contract。

`ExecutionRequest` 必须携带：

- `preprocess`
- `route`
- `semantic_decision`
- 可选 `analysis_id`
- 可选 `selected_candidate_id`
- 可选 `user_mask`
- `metadata`

## Manifest 摘要

`semantic_manifest_summary()` 生成 manifest 中的 pipeline fragment:

- `preprocess.selected`
- `preprocess.applied`
- `semantic.analysis_status`
- `semantic.selected_candidate_id`
- `semantic.ambiguity_types`
- `semantic.candidate_previews`
- `semantic.preview_assets` 摘要
