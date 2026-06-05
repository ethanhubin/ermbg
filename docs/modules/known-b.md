# PyMatting Known-B 模块

本文对齐当前 Known-B 路径和本地归属实现。

## 文件

- `ermbg/pymatting_refine.py`
- `ermbg/known_bg_hard_ui.py`
- `ermbg/local_ownership.py`
- `ermbg/ownership.py`
- `ermbg/analyze.py`
- `ermbg/preprocess.py`
- `tests/test_pymatting_refine.py`
- `tests/test_analyze.py`

## 职责

Known-B 路径用于已知或可稳定估计背景色的图形/UI 素材。它基于可测背景证据构造
trimap,再用 PyMatting 求解 unknown 区域。

主线规则:

- 背景场归一化属于 Preprocess/Analyze contract,不是 executor 私有步骤。
- 语义争议在 Analyze/Decide 阶段表达为候选。
- Execute 只消费最终语义约束和 user mask。

## 当前语义约束

Known-B trimap 消费:

- `semantic_decision.enclosed_near_bg_policy`;
- `user_keep_mask`;
- `user_remove_mask`;
- `pymatting_input_preprocessed_known_b`;
- `pymatting_background_normalization`。

`enclosed_near_bg_policy` 当前值:

- `subject`;
- `transparent_hole`;
- 默认不强制。

## 本地归属

本地归属仍是 Known-B 执行中的确定性证据模型,用于区分:

- 外部背景;
- 透明孔洞;
- 硬主体支持;
- 软主体层;
- shadow-like layer;
- conservative unknown。

它必须使用可测信号,例如颜色距离、拓扑、连通性、背景一致性、阴影拟合和 alpha 分布。

## 当前缺口

- Known-B 争议候选目前覆盖内部近背景色孔洞/主体二义性。
- `translucent_button` profile 使用更宽的近背景材质证据带来生成 preview region,
  以覆盖白色高光、半透明材质和近背景内部区域;普通硬图形仍使用精确背景匹配,
  避免真实孔洞区域膨胀。
- shadow 归属候选仍可继续扩展。
- Analyze 已输出服务端 preview assets 和像素级 region mask refs;后续可继续把执行端
  shadow/local ownership 的证据前移到语义候选。
