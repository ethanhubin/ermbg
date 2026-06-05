# 语义候选工作流

本文档定义 ERMBG 下一阶段的产品和架构方向: 把原始素材前置加工、route 策略、
语义争议判断和用户裁决都前置到 matting 执行之前。目标不是继续用更多阈值把
所有样本自动判对,而是在单张图像无法可靠判定语义归属时,把争议显式呈现给用户,
最后只执行一次最终选择的抠图方案。

## 背景

近期 Known-B 和 CorridorKey 的失败越来越集中在语义归属争议上:

- 封闭的近背景色区域可能是真透明孔洞,也可能是主体白毛、白衣、眼白、高光或浅色材质。
- 深色区域可能是主体描边、内部纹理,也可能是已知背景上的阴影。
- 近幕色区域可能是背景残留,也可能是主体拥有的同色材质。
- 玻璃、glow、抗锯齿、半透明按钮和角色软边常同时具备两种合理解释。

这些问题不是简单的参数问题。继续调 `bg_threshold`、`fg_threshold`、面积比、
局部带宽等参数,会让一个样本变好、另一个样本变坏。阈值规则只能移动错误边界,
不能消除单图语义二义性。VLM 可以作为可选排序器或解释器,但也不应成为唯一真相。

因此,复杂的自动 route/参数/特征分析只保留为默认建议器。生产主线应把高争议
语义显式化,让用户在执行前裁决。

## 新主线

```text
输入图
  -> 原始素材前置加工 Preprocess
  -> 轻量语义分析 Analyze
  -> 无争议?
       yes -> 使用默认决策 Execute -> 输出 RGBA
       no  -> 展示语义候选 Decide
               -> 用户选择候选
               -> 候选仍不准? 进入粗 Mask 编辑
               -> Execute -> 输出 RGBA
```

旧流程偏向:

```text
输入图 -> auto route -> 跑完整 matte -> 展示结果候选
```

新流程改为:

```text
输入图 -> preprocess -> auto route + ambiguity analyze -> 展示尚未执行的语义候选 -> 跑一次最终 matte
```

候选是**执行计划候选**,不是已经消耗算力生成的最终抠图候选。只有用户选择后,
Direct Worker 才执行对应的 matting。

## 分层职责

### Preprocess

Preprocess 是语义判断之前的原始素材前置加工层。它处理输入图本身的可观测缺陷,
为后续 route、候选分析、trimap/hint 预览和最终执行提供同一份更稳定的素材。

这一层已经有现成产品形态: Web 上传后自动检测假透明棋盘格背景,推荐勾选
`去网格`,用户可以接受或取消。它不是最终抠图结果,但它确实是一个前置候选:
系统检测到一种原始素材问题,给出推荐加工,并让用户裁决是否应用。

Preprocess 负责:

- 检测并可选应用假透明棋盘格/网格背景归一化;
- 检测并可选应用已知背景场归一化,把生成噪声、压缩噪声、轻微漂移等背景观测误差
  统一到稳定背景模型;
- 生成 `preprocessed_rgb`、`background_model`、`preprocess_decisions` 和 debug;
- 把所有前置加工的默认推荐暴露给 UI/API,允许用户勾选或取消;
- 保证后续 Analyze、候选预览和 Execute 消费同一份前置加工契约。

Preprocess 不负责:

- 判断封闭近背景区域到底是主体材质还是透明孔洞;
- 判断暗色区域到底是主体纹理还是阴影;
- 通过归一化把语义争议区域提前改成背景或主体;
- 调用 PyMatting、CorridorKey、VLM 或其他重型模型。

前置加工候选与语义候选的区别:

- 前置加工候选回答: `要不要先修正原始素材的背景观测问题?`
- 语义候选回答: `这些争议区域应归属主体、背景、孔洞、阴影还是 unknown?`

前置加工发生得更早。语义候选必须基于已经确定的 preprocess 决策生成。

### Analyze

Analyze 是轻量阶段,读取 `preprocessed_rgb` 和 `background_model`,不调用重型
matting 模型。它负责:

- 测量背景稳定性和背景色;
- 运行现有 route/profile 分析,得到默认 `algorithm`、`asset_kind`、
  `parameter_profile`、`execution_profile`;
- 检测高争议区域,例如 enclosed near-B、主体同背景色材质、疑似孔洞、
  疑似阴影/主体暗纹冲突、玻璃/半透明冲突;
- 输出语义候选列表、默认候选、争议区域 mask/overlay、解释文案和执行 payload。

Analyze 可以使用现有复杂启发式,但它的结论只是默认建议。它不得在高争议区域
强行把单一解释当作最终真相。

### 背景归一化位置

背景归一化应统一放在 Preprocess,也就是进入语义判断之前。它属于原始素材加工,
不是某个候选或 executor 私有步骤。

归一化原则:

- 只稳定已证明属于外部背景或背景观测误差的像素;
- 不用归一化结果直接解决主体/孔洞/阴影语义;
- 输出统一 `background_model` 和 `preprocessed_rgb`,后续 route、候选、trimap
  预览、CorridorKey hint 预览和最终执行都消费它;
- 如果用户关闭某个前置加工,Analyze 和 Execute 都必须看到未应用该加工的同一输入;
- executor 不应再偷偷运行另一套背景估计/归一化并改变语义边界。

现有两个背景归一化路径应收敛到这一层:

- 假透明棋盘格/网格背景归一化: 上传后检测,推荐勾选 `去网格`,用户可裁决。
- Known-B 背景场归一化: 对稳定已知背景上的漂移/压缩/低频不均匀做前置加工,
  生成后续 Known-B 和 CorridorKey 共享的背景模型。

### Decide

Decide 是用户或调用方选择语义策略的阶段。Web UI 是主要实现面:

- 无争议时不打断用户,直接执行默认候选。
- 有争议时展示 2-3 个候选卡片,让用户先做语义选择。
- 候选卡片只展示轻量预览: 原图、争议区域 overlay、候选影响范围、名称和理由。
- 用户选择一个候选后才执行完整 matte。
- 如果候选仍不准确,用户可以进入粗 Mask 编辑,用少量笔刷补充裁决。

API 调用方也可以跳过 UI,直接传入 `candidate_id` 或 `semantic_decision` 执行。

### Execute

Execute 消费最终决策,运行一次生产 matting:

- `PreprocessDecision`;
- `RouteDecision`;
- `SemanticDecision`;
- 可选 `UserMaskDecision`;
- Direct Worker server URL 选择;
- 执行参数和输出 manifest。

执行阶段不得重新推断素材语义,也不得私有化前置加工;只消费 Preprocess/
Analyze/Decide 给出的契约。

## 高争议候选

候选数量应少而稳定。第一版只需要覆盖最常见、最有破坏性的争议。

### 候选通用字段

每个候选都应是可序列化对象:

```json
{
  "id": "protect_near_bg_subject",
  "label": "保留内部浅色",
  "intent": "把被主体包围的近背景色区域视为主体材质。",
  "default": true,
  "confidence": 0.72,
  "risk_level": "medium",
  "decision": {
    "enclosed_near_bg_policy": "subject"
  },
  "regions": ["ambiguous_enclosed_bg_0", "ambiguous_enclosed_bg_1"],
  "preview": {
    "overlay_mask": "mask_id",
    "bbox_xyxy": [39, 141, 250, 214]
  },
  "reasons": [
    "近背景色区域不连通外部背景",
    "区域被强主体颜色和描边包围",
    "单图证据无法证明它一定是透明孔洞"
  ]
}
```

字段说明:

- `id`: 稳定机器名,用于 API 和 manifest。
- `label`: UI 上的短名称。
- `intent`: 一句话说明这个候选会保护或移除什么。
- `default`: 现有复杂分析推荐的默认项。
- `confidence`: 默认排序用,不代表语义绝对正确。
- `risk_level`: `low`、`medium`、`high`,用于决定是否自动执行。
- `decision`: executor 消费的语义策略。
- `regions`: 关联到 Analyze 输出的争议区域 ID。
- `preview`: 轻量 overlay 信息,不是最终 matte。
- `reasons`: 给用户和 debug 用的可解释理由。

### `auto_default`

保留现有 route/参数/阈值分析的默认解释。

用途:

- 无争议时直接执行。
- 有争议时作为第一个候选展示。
- 作为 CLI/API 非交互调用的默认策略。

约束:

- `auto_default` 不能隐藏争议。如果检测到高争议区域,即使默认置信度最高,
  也必须把其他候选暴露给 UI/API。

### `protect_near_bg_subject`

把封闭的近背景色区域视为主体材质或主体拥有的软层。

适用:

- 白底角色里的白毛、白脸、尾巴白尖、眼白、白色高光;
- 灰/白/黑背景上的浅色主体材质;
- 被主体强色、深色描边或主体拓扑包围的近背景色组件。

执行含义:

- 争议区域不得成为 `sure_bg`;
- 区域核心可作为 `protected_subject` 或 `sure_fg`;
- 区域边界可保持 `unknown`,让 Known-B 求解抗锯齿;
- foreground RGB 默认保留源图颜色。

UI 文案建议:

- 名称: `保留内部浅色`
- 副文案: `适合白毛、白衣、眼白、高光等主体材质`

### `cut_enclosed_holes`

把封闭的近背景色区域视为透明孔洞。

适用:

- UI 面板镂空;
- 字母或图标内部孔洞;
- 确实应透出背景的装饰开口;
- 被主体包围,但视觉上像空洞而非材质的区域。

执行含义:

- 争议区域核心成为 `sure_bg` 或 `forced_transparent`;
- 孔洞边缘保持 `unknown`,让 Known-B 求解抗锯齿;
- 孔洞内满足背景族标量变暗的区域可作为孔洞侧阴影处理。

UI 文案建议:

- 名称: `透明内部孔洞`
- 副文案: `适合镂空 UI、文字洞、图标开口`

### `preserve_shadow`

当争议是主体暗纹和背景阴影冲突时,把相关暗区域作为阴影层保留。

适用:

- 已知背景上的外部投影;
- 按钮或图标贴地阴影;
- 与主体接触、满足同背景标量变暗模型的软层。

执行含义:

- 区域不进入主体 alpha;
- 作为 `shadow_layer` 写入 `rgba_rgb` companion;
- foreground 保持干净主体色。

UI 文案建议:

- 名称: `保留阴影`
- 副文案: `适合投影或接触阴影`

### `remove_shadow_like_subject`

当暗区更可能是主体描边、纹理或内部材质时,禁止 ShadowPatch 把它当阴影。

适用:

- 角色五官、衣褶、深色描边;
- 深色按钮纹理;
- 与外部背景变暗模型相似但被主体包围的暗区。

执行含义:

- 区域作为主体或 unknown 进入主体求解;
- 不参与 shadow repair。

UI 文案建议:

- 名称: `保留暗色主体`
- 副文案: `适合描边、五官、衣褶和深色纹理`

第一阶段可以只实现 `auto_default`、`protect_near_bg_subject`、
`cut_enclosed_holes`。阴影候选可复用同一契约后续加入。

## 争议区域

Analyze 应把像素级证据压缩成区域级对象。UI 不应展示几十个零散像素点,而应展示
合并后的争议区域。

争议区域字段:

```json
{
  "id": "ambiguous_enclosed_bg_0",
  "type": "enclosed_near_background",
  "bbox_xyxy": [39, 141, 250, 214],
  "area_px": 4971,
  "mask_ref": "mask_id",
  "evidence": {
    "background_color": [254, 253, 254],
    "touches_exterior_background": false,
    "enclosed_by_subject_support": true,
    "near_background_fraction": 0.94
  },
  "ambiguity": {
    "transparent_hole_score": 0.48,
    "subject_material_score": 0.72,
    "reason": "single_image_semantic_ambiguity"
  }
}
```

区域合并规则:

- 同一语义争议下相邻或同属一个主体部件的组件应合并显示。
- 极小且视觉影响低的组件可折叠进 `minor_regions` 统计,不单独打扰用户。
- 高影响区域按面积、预期 alpha 差异和位置排序。
- overlay 应能显示全部影响范围,但候选卡片默认只展示排名最高的区域 bbox。

## Web 呈现

候选呈现的目标是让用户快速回答一个语义问题,而不是阅读算法参数。

### 上传后前置加工

上传图片后,Web 先运行 Preprocess 检测,再进入语义 Analyze。前置加工以可勾选
控件呈现:

- `去网格`: 已有模式。检测到假透明棋盘格时自动勾选,用户可取消。
- `背景归一化`: 规划模式。检测到稳定已知背景存在轻微漂移/噪声时自动推荐,
  用户可取消。

前置加工控件应显示:

- 简短名称;
- 推荐状态,例如 `推荐`;
- 影响范围或原因,例如 `检测到 16px 棋盘格`、`背景轻微漂移`;
- 轻量预览,例如原图/加工后差异或影响区域 overlay。

前置加工控件不应显示:

- PyMatting 结果;
- CorridorKey 结果;
- solver 参数;
- 语义候选卡片。

用户改变前置加工勾选状态后,必须重新运行 Analyze,因为语义候选依赖
`preprocessed_rgb` 和 `background_model`。

### 无争议

Web 行为:

1. 上传图片。
2. 检测并显示前置加工推荐。
3. 基于当前前置加工决策运行 Analyze。
4. 显示 route 摘要。
5. 自动执行默认候选。
6. 展示最终 RGBA、alpha、foreground 和 debug。

页面不应出现候选拦截。

### 有争议

Web 行为:

1. 上传图片后先检测前置加工并给出推荐勾选。
2. 基于当前前置加工决策调用 Analyze。
3. 页面进入 `需要确认` 状态,不立即跑完整 matte。
4. 主画布显示前置加工后的分析图或原图对照,叠加争议区域 overlay。
5. 右侧或底部展示候选卡片。
6. 用户点击候选后,按钮文案为 `按此方案抠图`。
7. 执行完成后展示最终结果。

候选卡片内容:

- 名称: 例如 `保留内部浅色`。
- 小预览: 原图局部 + 半透明 overlay,不是最终 matte。
- 影响范围: `12,947 px · 5 个区域`。
- 适用提示: `适合白毛、白衣、眼白、高光`。
- 默认标记: `推荐`。
- 风险提示: 例如 `可能保留真实孔洞`。

卡片不展示:

- 原始阈值;
- solver 参数;
- 大段 debug JSON;
- 多个已经执行完的 RGBA 结果。

### 候选预览

候选预览应便宜:

- 原图上加彩色 overlay;
- 透明候选用蓝色或棋盘纹理标记将被移除的区域;
- 主体保护候选用暖色或描边标记将被保留的区域;
- 鼠标悬停时高亮对应 bbox;
- 可切换 `原图` / `候选影响` / `差异区域`。

不要在候选阶段调用 CorridorKey、PyMatting 或远端重型模型。

### 文案原则

用户看到的是语义选择,不是算法选择:

- 用 `保留内部浅色`,不用 `enclosed_bg_policy=subject`。
- 用 `透明内部孔洞`,不用 `sure_bg core`。
- 用 `保留阴影`,不用 `shadow_alpha physical replay`。
- 用 `继续编辑`,不用 `provide trimap constraint`。

## 粗 Mask 兜底

Mask 是候选不够准确时的轻量裁决输入,不是让用户画精确 alpha。

第一版只需要两个笔刷:

- `保留`: 用户涂到的区域必须归属主体。
- `移除`: 用户涂到的区域必须归属背景或透明。

可选第三个笔刷:

- `未知/边缘`: 用户标记为交给 matting 求解的软边区域。

Mask 契约:

```json
{
  "keep_mask": "mask_id_or_png",
  "remove_mask": "mask_id_or_png",
  "unknown_mask": "mask_id_or_png",
  "source": "web_user_brush",
  "brush_version": 1
}
```

执行语义:

- `keep_mask` 转为 `forced_subject` 或 `protected_subject`;
- `remove_mask` 转为 `forced_background` 或 `forced_transparent`;
- `unknown_mask` 转为 trimap unknown;
- 用户 mask 优先级高于自动候选,但仍需形状校验和冲突处理。

冲突处理:

- 同一像素同时被 `keep` 和 `remove` 覆盖时,以后画的笔刷为准。
- 空 mask 不改变候选。
- 满图 keep/remove 需要 UI 二次确认或 API 报告高风险。

Mask UI 应只服务于粗裁决:

- 支持画笔大小、撤销、清空、缩放/平移;
- 显示当前候选 overlay;
- 不要求用户描精确边缘;
- 不直接把 brush mask 当最终 alpha 导出。

## API 契约草案

### Preprocess

建议新增或扩展轻量接口:

```http
POST /api/preprocess-analysis
```

返回:

```json
{
  "preprocess_id": "pre_abc",
  "items": [
    {
      "id": "remove_checkerboard",
      "label": "去网格",
      "recommended": true,
      "enabled_by_default": true,
      "reason": "detected_checkerboard_background",
      "preview_assets": {
        "overlay_png": "/api/preprocess-preview/pre_abc/remove_checkerboard.png"
      }
    },
    {
      "id": "normalize_known_background",
      "label": "背景归一化",
      "recommended": true,
      "enabled_by_default": true,
      "reason": "stable_known_background_with_mild_drift"
    }
  ]
}
```

调用方随后把选中的前置加工传给 Analyze。

### Analyze

建议新增轻量接口:

```http
POST /api/analyze-candidates
```

返回:

```json
{
  "status": "needs_decision",
  "preprocess": {
    "preprocess_id": "pre_abc",
    "selected": ["remove_checkerboard", "normalize_known_background"]
  },
  "default_candidate_id": "protect_near_bg_subject",
  "route": {
    "algorithm": "pymatting_known_b",
    "asset_kind": "known_bg_graphic",
    "parameter_profile": "edge_cleanup",
    "execution_profile": "pymatting-known-bg"
  },
  "ambiguities": [
    {
      "id": "ambiguous_enclosed_bg_0",
      "type": "enclosed_near_background",
      "bbox_xyxy": [39, 141, 250, 214],
      "area_px": 4971
    }
  ],
  "candidates": [
    {
      "id": "protect_near_bg_subject",
      "label": "保留内部浅色",
      "default": true,
      "decision": {"enclosed_near_bg_policy": "subject"}
    },
    {
      "id": "cut_enclosed_holes",
      "label": "透明内部孔洞",
      "default": false,
      "decision": {"enclosed_near_bg_policy": "transparent_hole"}
    }
  ],
  "preview_assets": {
    "overlay_png": "/api/candidate-preview/<id>/overlay.png"
  }
}
```

`status`:

- `ready`: 无争议,调用方可直接执行默认候选。
- `needs_decision`: 有争议,建议用户选择。
- `unsupported`: Analyze 无法构建有效候选,应走旧式 auto 或 fallback。

### Execute

最终执行接口接受候选和可选 mask:

```http
POST /api/execute-candidate
```

输入:

```json
{
  "analysis_id": "analysis_abc",
  "preprocess_id": "pre_abc",
  "candidate_id": "protect_near_bg_subject",
  "user_mask": {
    "keep_mask": "mask_id",
    "remove_mask": "mask_id"
  }
}
```

输出沿用现有 matte 执行响应中的 RGBA、alpha、foreground、debug、route
metadata 和 manifest 语义。

`/api/matte-candidates` 保留为兼容层,用于脚本和旧调用方。Web 主线应走
`/api/preprocess-analysis`、`/api/analyze-candidates`、`/api/execute-candidate`
的分离流程。兼容层必须在 metadata 中标明自身不是新的候选主入口。

## Manifest 和记录

最终输出必须记录:

- `preprocess_id`;
- `preprocess_decisions`,包括检测结果、推荐值、用户选择和是否应用;
- `analysis_id`;
- `default_candidate_id`;
- `selected_candidate_id`;
- `semantic_decision`;
- `ambiguity_regions`;
- `user_mask` 摘要,包括是否使用、像素数、冲突数;
- 实际 `execution_backend`、`execution_profile`、`execution_server_url`;
- 是否跳过候选阶段,以及原因。

`ermbg.run.v1` 可以扩展:

```json
{
  "route": {
    "algorithm": "pymatting_known_b",
    "execution_profile": "pymatting-known-bg"
  },
  "preprocess": {
    "selected": ["remove_checkerboard", "normalize_known_background"],
    "applied": ["normalize_known_background"]
  },
  "semantic": {
    "analysis_status": "needs_decision",
    "default_candidate_id": "protect_near_bg_subject",
    "selected_candidate_id": "protect_near_bg_subject",
    "ambiguity_types": ["enclosed_near_background"],
    "user_mask_used": false
  }
}
```

## 默认自动路径

现有复杂参数和特征分析的定位:

- 排序候选;
- 决定无争议时是否直接执行;
- 选择默认 `execution_profile`;
- 生成 debug 解释;
- 为 CLI/API 非交互场景提供保守默认。

它不再承担:

- 在语义二义性区域强制给唯一答案;
- 靠继续细化阈值消灭所有用户可见争议;
- 在执行阶段重新改写 route 语义。

无争议的定义应该保守。只要争议区域足够大、足够影响 alpha、且存在两个可信解释,
就进入候选阶段。

## 验证要求

新增测试应覆盖流程,而不只是某个样本的最终 alpha:

- 去网格检测在上传后返回推荐,用户取消后 Analyze 使用未去网格输入。
- Known-B 背景场归一化在语义判断前完成,Analyze 和 Execute 使用同一
  `preprocessed_rgb`/`background_model`。
- 背景归一化不得把封闭近背景主体/孔洞争议提前消掉。
- 无争议样本直接返回 `ready`,且只执行一次。
- 白底角色内部近白区域返回 `needs_decision`,包含 `protect_near_bg_subject`
  和 `cut_enclosed_holes`。
- UI 镂空样本默认可偏向 `cut_enclosed_holes`,但仍能选择 `protect_near_bg_subject`。
- 候选阶段不调用 PyMatting/CorridorKey 重型执行。
- 用户选择候选后,Execute 消费同一个 `analysis_id` 和 `candidate_id`。
- `keep_mask` 能覆盖默认透明孔洞决策。
- `remove_mask` 能覆盖默认主体保护决策。
- manifest 记录默认候选、最终候选和 mask 使用情况。

真实样本回归应按争议类型记录,不要按文件名写特例。

## 迁移计划

1. 定义 `PreprocessAnalysis`、`PreprocessDecision`、`AnalyzeResult`、
   `SemanticCandidate`、`AmbiguityRegion`、`SemanticDecision`、
   `UserMaskDecision` 数据结构。
2. 将现有去网格检测纳入 Preprocess 契约,保留上传后推荐勾选的产品形态。
3. 将 Known-B 背景场归一化迁到 Preprocess,并让 Analyze/Execute 消费同一
   `preprocessed_rgb` 和 `background_model`。
4. 为 Known-B enclosed near-background 实现轻量 Analyze,先不改变执行输出。
5. Web 增加候选拦截页面: 有争议时展示 overlay 和候选卡片。
6. Execute 接受 `candidate_id`,把 selected semantic decision 传入 Known-B trimap
   构造。
7. 把现有 Mask 功能前置为 `keep/remove` 裁决输入。
8. 将 `/api/matte-candidates` 的旧“跑多个结果”用途收敛为兼容层,新流程以
   Preprocess/Analyze/Execute 为主。
9. 扩展 manifest、summary 和 Web 后台列表,显示前置加工和语义候选状态。
10. 再考虑可选 VLM: 只用于候选命名、排序或解释,不得绕过本地 validator 和
   用户裁决。

## 反模式

- 为单个样本继续增加互相打架的阈值分支。
- 在候选阶段跑三次完整 matting 后让用户选图。
- 在 Analyze 或 Execute 内部各跑一套不一致的背景归一化。
- 用背景归一化提前解决主体/孔洞/阴影语义争议。
- 把用户粗 mask 当最终 alpha。
- 让 Web 前端重新实现候选检测逻辑。
- VLM 直接输出最终 mask 或跳过本地证据验证。
- 在执行阶段重新推断 asset kind 或争议语义。
