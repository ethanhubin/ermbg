# Pipeline Contracts

本文对齐当前 `ermbg.pipeline_contracts` 和 Web/Direct Worker 主线。

## Preprocess

核心对象:

- `BackgroundModel`: 可观测背景证据，包含 `color`、`source`、`confidence`、`metadata`。
- `PreprocessItem`: 可推荐的前置加工项。
- `PreprocessAnalysis`: `/api/preprocess-analysis` 输出。
- `PreprocessDecision`: 最终应用的前置加工决策。

当前唯一主线 item:

- `background_repair`

棋盘格修复和 Known-B 背景场归一化都是 `background_repair` 下的机制 metadata，不再作为
独立用户候选。

## Analyze

核心对象:

- `AmbiguityRegion`: 争议区域，包含 `id`、`type`、`bbox_xyxy`、`area_px`、
  `mask_ref`、`evidence`、`ambiguity`。
- `SemanticCandidate`: 语义候选，包含 `id`、`label`、`intent`、`decision`、
  `regions`、`preview`、`reasons`、可选 `route_candidate_id`。
- `AnalyzeResult`: 包含 `analysis_id`、`status`、`route_candidates`、
  `default_route_candidate_id`、兼容字段 `route`、`ambiguity_regions`、`candidates`、
  `default_candidate_id`、`preprocess`、`preview_assets`。

`AnalyzeResult.status`:

- `ready`;
- `needs_decision`;
- `unsupported`。

Preview assets 是轻量预览。Known-B candidate trimap 可以携带
`execution_role=pymatting_explicit_trimap`，Execute 可复用；CorridorKey candidate 可携带
hint。

当前 semantic region 类型:

- `enclosed_near_background`;
- `button_body_subject_ownership`;
- `screen_material_or_translucency`。

shadow ownership 不是当前 semantic region 类型。

## Decide / Execute

核心对象:

- `SemanticDecision`: 最终候选裁决。
- `UserMaskDecision`: keep/remove 粗 mask 摘要。
- `ExecutionRequest`: Execute 阶段唯一输入 contract。

`ExecutionRequest` 必须携带:

- `preprocess`;
- `route`;
- `semantic_decision`;
- 可选 `analysis_id`;
- 可选 `selected_candidate_id`;
- 可选 `user_mask`;
- `metadata`。

Execute request 应能复现:

- 选中的 route candidate；
- 选中的 semantic candidate；
- user mask；
- Known-B explicit trimap 或 CorridorKey hint；
- 实际 execution backend。

## Manifest 摘要

`semantic_manifest_summary()` 写入:

- `preprocess.selected`;
- `preprocess.applied`;
- `semantic.analysis_status`;
- `semantic.selected_candidate_id`;
- `semantic.ambiguity_types`;
- `semantic.candidate_previews`;
- `semantic.preview_assets` 摘要。
