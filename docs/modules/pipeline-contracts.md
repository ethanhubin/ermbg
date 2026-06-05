# Pipeline Contracts

本文对齐当前 `ermbg.pipeline_contracts` 实现。

## 文件

- `ermbg/pipeline_contracts.py`
- `tests/test_pipeline_contracts.py`

## 核心对象

### Preprocess

- `BackgroundModel`: 可观测背景证据,包含 `color`、`source`、`confidence`、`metadata`。
- `PreprocessItem`: 可推荐的前置加工项,例如 `remove_checkerboard`。
- `PreprocessAnalysis`: 前置分析结果,包含 `preprocess_id`、`items`、`background_model`、`debug`。
- `PreprocessDecision`: 最终应用的前置加工决策,包含 `selected`、`applied`、`metadata`、`background_model`。

### Analyze

- `AmbiguityRegion`: 争议区域,包含 `id`、`type`、`bbox_xyxy`、`area_px`、`mask_ref`、`evidence`、`ambiguity`。
- `SemanticCandidate`: 语义候选,包含 `id`、`label`、`intent`、`decision`、`regions`、`preview`、`reasons`。
- `AnalyzeResult`: Analyze 输出,包含 `status`、`route`、`ambiguity_regions`、`candidates`、`default_candidate_id`、`preprocess`。

`AnalyzeResult.status` 当前取值:

- `ready`;
- `needs_decision`;
- `unsupported`。

### Decide / Execute

- `SemanticDecision`: 最终候选裁决。
- `UserMaskDecision`: 粗 keep/remove/unknown mask 的引用和摘要。
- `ExecutionRequest`: Execute 阶段唯一输入 contract。

`ExecutionRequest` 必须携带:

- `preprocess`;
- `route`;
- `semantic_decision`;
- 可选 `analysis_id`;
- 可选 `selected_candidate_id`;
- 可选 `user_mask`;
- `metadata`。

## Manifest 摘要

`semantic_manifest_summary()` 生成 manifest 中的 pipeline fragment:

- `preprocess.selected`;
- `preprocess.applied`;
- `semantic.analysis_status`;
- `semantic.default_candidate_id`;
- `semantic.selected_candidate_id`;
- `semantic.semantic_decision`;
- `semantic.user_mask_used`;
- `semantic.user_mask_summary`。

## 约束

- contract 层只负责序列化和边界表达,不承载算法行为。
- 新字段必须可 JSON 序列化。
- Web/API/Direct Worker 之间传递 contract 时,不要靠文件名或样本 ID 推断语义。
