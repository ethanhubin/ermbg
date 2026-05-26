# ERMBG · 工程实现现状

## 0. 系统设计目标

基于 AI 生图大模型已经具备的"指定特定背景色出图"的能力,在**已知/可控的纯色背景**这个前提下做抠图优化,产出能直接复用的 RGBA 资产,解决现有 AI 生图资产在二次合成场景下的边缘脏、白边/黑边、半透明区域错乱等问题。

整个系统因此分为两段,各自的职责是:

**第一段:出图(可控背景)**
- 通过 AI 生图模型(Banana,GPT-Image,ComfyUI 本地 Qwen / Flux 等)在生成阶段就把背景指定为目标颜色(默认绿幕 RGB(0, 200, 0));
- 出图时让背景尽量是常量纯色,边缘不要"AI 加光晕/阴影/反光";
- 出图本身不在 ERMBG 主路径里,但 [ermbg/probe/](ermbg/probe/) 提供 ComfyUI 客户端和工作流模板,以及背景色常量在 [ermbg/probe/prompts.py](ermbg/probe/prompts.py)。

**第二段:抠图(本仓库主路径)**
- 在已知背景色的前提下,**自动**判断输入类型(透明 PNG / 绿底 / 白底 / 黑底 / 灰底 / 噪声底),为每类挑一个最优策略,产出 RGBA;
- 已知 B 是这套管线的核心约束 — 它让 despill 从"猜"变成"投影",让 QA 从"主观看"变成"重合成误差有 ground truth";
- 用户**不需要**指定参数,router 看图选策略。

---

## 1. 核心管线(第二段:抠图)

```
image (sRGB uint8 + 可选 source α)
  │
  ├── router.classify_strategy → Strategy
  │      bg_type ∈ {rgba_passthrough, saturated, white, black, grey, noisy}
  │      image_type ∈ {graphic, photo}
  │      keyer_mode, despill, use_keyer_merge, passthrough, ...
  │
  ├── if Strategy.passthrough:
  │      直接复用 source α,跳过下面整条管线
  │
  ├── BiRefNetSegmenter("ZhengPeng7/BiRefNet-matting")  → soft α
  │      [回退] GrabCutSegmenter,无 torch 时
  │
  ├── BackgroundDiagnoser
  │      → DiagnosisReport(B, σ, q10, verdict)
  │      verdict=not-pure-bg 时 logger.warning 提示
  │
  ├── key α (chromatic | luminance | none) + merge_alpha_components
  │      把 BiRefNet 漏掉的小连通分量从 key α 补回主 α,不覆盖主主体
  │      white/black graphic 额外执行 known-B / hard-edge 局部修复
  │
  ├── sRGB → linear RGB
  │
  ├── apply_despill(method, C_lin, B_lin, α) → (α_out, F_lin)
  │      ├── auto         (饱和 B): unmix + chroma_cap
  │      ├── unmix        F = (C-(1-α)B)/α,低 α 走 KDTree 兜底
  │      ├── chroma_cap   Vlahos 通道压制 + local_borrow
  │      ├── local_borrow KDTree 在 sure_fg 中借色
  │      ├── closed_form  pymatting 联合反演 α + F
  │      └── none         baseline
  │
  ├── F_lin → sRGB,α → uint8,dstack 出 RGBA
  │
  ├── generate_matte_candidates
  │      对同色内区等真实歧义生成 transparent_hole / same_color_marking 候选
  │
  └── run_qa: 合到 6 张背景 + 每张配 lightwrap → 多指标
```

入口:[ermbg/matting.py:38](ermbg/matting.py#L38) 的 `matte()`,CLI 是 [ermbg/cli.py:107](ermbg/cli.py#L107) 的 `ermbg matte`。

---

## 2. Router 策略表

[ermbg/router.py](ermbg/router.py) 的 `classify_strategy(image, source_alpha)` 返回一个 `Strategy`:

| bg_type | 触发条件 | keyer_mode | despill | 备注 |
|---|---|---|---|---|
| `rgba_passthrough` | 源图 α 至少 5% 像素半透明 | — | none | 直接复用源 α(待加智能脏度检测,见 §10) |
| `saturated` | OKLab chroma ≥ 8 | chromatic | auto | 绿幕/品红/青底,最佳工况 |
| `white` | L ≥ 85,chroma 低 | luminance | unmix | 白底卡通图 |
| `black` | L ≤ 15,chroma 低 | luminance | unmix | 黑底 |
| `grey` | 中亮度,chroma 低 | luminance(放宽阈值) | local_borrow | 信号弱,以网络 α 为主 |
| `noisy` | corner σ > 18 | none | local_borrow | 不是纯色底,降级到只用网络 α |

`image_type` 影响 keyer 阈值:
- `graphic`(向量 / 卡通 / logo,色调集中):紧阈值 `bg_max=4 / fg_min=14`
- `photo`(自然图):宽阈值 `bg_max=6 / fg_min=22`

判定基于 quantize 后 top-8 颜色覆盖比例。

---

## 3. 模块清单

| 文件 | 职责 |
|---|---|
| [ermbg/router.py](ermbg/router.py) | **新**:`classify_strategy` + `Strategy` dataclass,看图选策略 |
| [ermbg/keyer.py](ermbg/keyer.py) | **新**:`chromatic_key_alpha` / `luminance_key_alpha` / `key_alpha`(分发器)+ `merge_alpha_components` |
| [ermbg/segmenter.py](ermbg/segmenter.py) | `BiRefNetSegmenter` / `GrabCutSegmenter`(回退)/ `make_bands` |
| [ermbg/diagnose.py](ermbg/diagnose.py) | `BackgroundDiagnoser` 单图诊断 + risk_map |
| [ermbg/metrics.py](ermbg/metrics.py) | `measure_background_color` / `background_purity_sigma` / `edge_contrast_q10` 等;σ 测量已修(避免 AA 边污染)|
| [ermbg/despill.py](ermbg/despill.py) | `chroma_cap` / `local_foreground_borrow` / `unmix_foreground`(新)/ `closed_form_matting` / `apply_despill`(新增 `auto` 和 `unmix`) |
| [ermbg/candidates.py](ermbg/candidates.py) | `generate_matte_candidates`:同色内区等信息论歧义的候选生成 |
| [ermbg/lightwrap.py](ermbg/lightwrap.py) | Brinkmann light wrap 边缘晕修 |
| [ermbg/qa.py](ermbg/qa.py) | 6 背景合成 + halo / noise / 细结构指标 |
| [ermbg/matting.py](ermbg/matting.py) | 端到端 `matte()` 主入口,接 router |
| [ermbg/web.py](ermbg/web.py) | FastAPI 后台 UI/API,结果区支持候选缩略图切换 |
| [ermbg/cli.py](ermbg/cli.py) | `segment` / `diagnose` / `matte` / `phase1` / `probe` |
| [ermbg/colorspace.py](ermbg/colorspace.py) | sRGB ↔ OKLab |
| [ermbg/io.py](ermbg/io.py) | `load_image_with_alpha`(新,返回 (rgb, source_α))/ `load_rgb` 自动合到指定背景 / sRGB ↔ linear |
| [ermbg/types.py](ermbg/types.py) | `Trimap` / `MattingResult` dataclass |

---

## 4. 默认决策

- **抠图模型**:`ZhengPeng7/BiRefNet-matting`(MIT,matting-trained,Mac MPS ≈1 GB)
- **Despill 默认**:`auto`(由 router 选择;CLI 也可手动覆盖)
- **Keyer**:由 router 选择 chromatic / luminance / 无;merge 只在 saturated/white/black 上启用,避免覆盖网络的边缘羽化
- **诊断阈值**([diagnose.py:26](ermbg/diagnose.py#L26)):
  - `purity_sigma_max = 5.0`(uint8 RGB std)
  - `edge_contrast_q10_min = 8.0`(OKLab ΔE)
- **Band radius**:`max(4, min(20, round(0.008 * min(W, H))))`
- **QA 背景**:black / white / grey / cyan / magenta / checker,每张配 lightwrap

---

## 5. 颜色空间约定

- 内部计算:**linear RGB**(`io.srgb_to_linear` / `io.linear_to_srgb_u8`)
- 颜色距离:**OKLab**(`colorspace.oklab_distance`)
- α / soft mask:**float32 [0,1]**,HxW
- RGBA 输出:**uint8 sRGB**,α 在 last channel

---

## 6. 成像模型

```
C = α · F + (1 − α) · B          (in linear RGB)
```

- `C`:观测像素颜色(sRGB → linear)
- `B`:`measure_background_color` 给的常量背景
- `α`:BiRefNet-matting 直出 + keyer 补漏
- `F`:由 despill 给出:
  - `unmix`:`F = (C-(1-α)B)/α`,低 α 走 KDTree 兜底(已知 B 时的正解)
  - `chroma_cap`:`F.d = min(F.d, max(F[other2]))`(d = B 的主导通道)
  - `local_borrow`:`F = KDTree-weighted mean of sure_fg neighbors`
  - `closed_form`:pymatting 联合反演 α 和 F

`auto` = `unmix` + 在饱和 B 上叠 `chroma_cap` 兜底色边。

---

## 7. CLI

```bash
# 端到端(主路径,router 自动选策略)
.venv/bin/ermbg matte samples/inputs/3.png

# 手动覆盖 despill
.venv/bin/ermbg matte samples/inputs/3.png --despill unmix
.venv/bin/ermbg matte samples/inputs/3.png --despill chroma_cap

# 关掉 keyer
.venv/bin/ermbg matte samples/inputs/3.png --no-keyer

# 只看诊断
.venv/bin/ermbg diagnose samples/inputs/3.png

# 批量
.venv/bin/ermbg phase1 --input-dir samples/inputs --out-dir samples/outputs/phase1

# 粗分割(只出 mask + rough trimap)
.venv/bin/ermbg segment samples/inputs/3.png
```

输出:`*_rgba.png` / `*_alpha.png` / `*_foreground.png` / `*_trimap.png` / `*.report.json` / `*_qa/on_*.png`。

`*.report.json` 现在包含:
- `diagnosis`(B / σ / verdict)
- `strategy`(router 决策:bg_type / image_type / keyer_mode / despill / extras)
- `keyer`(used / patched_components / component_areas)
- `qa`(recomp_err / halo_per_bg / α_noise / thin_keep)

---

## 8. QA 指标

[ermbg/qa.py](ermbg/qa.py) 输出:

| 指标 | 含义 | 越小越好 |
|---|---|:---:|
| `recomposition_error_on_observed_bg` | linear RMS 误差,`α·F+(1-α)·B` vs 原图 | ✓ |
| `edge_halo_score_per_bg[name]` | α∈(0, 0.15] 像素合到 bg 后与 bg 的 OKLab ΔE | ✓ |
| `edge_halo_score_mean` | 5 张非 checker bg 的均值 | ✓ |
| `alpha_noise_p95` | α∈(0.05, 0.95) 区域 \|∇α\| 的 P95 | ✓ |
| `thin_structure_preservation` | 输入 mask 中 <0.5% 面积的小连通分量在输出中存活的比例 | ✗(越大越好)|

---

## 9. 测试

```
tests/test_colorspace.py   OKLab round-trip
tests/test_despill.py      chroma_cap / local_borrow / unmix / dispatcher
tests/test_diagnose.py     purity / contrast / verdict
tests/test_io.py           sRGB↔linear,save/load
tests/test_keyer.py        chromatic + luminance keyer + merge
tests/test_candidates.py   同色内区候选生成
tests/test_lightwrap.py    halo 抑制
tests/test_matting.py      end-to-end smoke
tests/test_router.py       策略分类(saturated/white/black/grey/passthrough/graphic/photo)
tests/test_api.py          高阶 API / subject_mask 输入
tests/test_cli.py          CLI subject_mask smoke
tests/test_web.py          后台 UI / 候选 API
tests/test_comfy_nodes.py  ComfyUI 节点输入输出
tests/test_comfy_subject_workflow.py  CLIPSeg→ERMBG workflow 渲染/下载
```

日常重点回归跑 `.venv/bin/pytest -q -m core`,覆盖 router / keyer / API / CLI / subject-mask workflow。提交前跑全量 `.venv/bin/pytest -q`;当前 102 项全部通过。新增模块要带 smoke 测试。

---

## 10. 已知问题 / TODO

### 10.1 RGBA 输入的智能脏度检测(已实现)

`rgba_passthrough` 现在不再无条件触发。只要源图 α 至少 5% 像素非全不透明,router 会先跑 `assess_source_alpha(rgb, alpha)` 做无监督卫生检查:

- **fringe_dE**:在 α∈(0.05, 0.6] 的软边带,按 straight / sRGB-premul / linear-premul 三种包装恢复 implied F,取最贴近 opaque interior 的 OKLab 距离。
- **low_alpha_residual**:检查 α≈0 区域 RGB 是否接近 premultiplied black,或是否泄露 interior / 原始背景色。
- **bimodal_fraction**:识别硬二值 α,避免缺失 AA 的粗抠图被直接复用。

任一指标失败 → strategy 从 `rgba_passthrough` 落回正常背景分类路径,由主流程重抠。测试覆盖 clean passthrough、halo reject、low-alpha leak、binarized matte、无 opaque interior 等边界。

### 10.2 9/10 白底 graphic 闭环(已落地)

9/10 号样本把两个问题分开了:

1. **硬边细描边**:10 号红圈外的黑色细描边是 hard edge,不应被 BiRefNet 羽化成灰白半透明。当前 white/black graphic 路径已加入 `hard_edge` repair:只对 keyer 高置信、与背景亮度差极大、当前 alpha 偏低、且贴近可信前景的小型组件抬 alpha,保住 1px 描边。
2. **同色内区歧义**:红圈内部白色区域与白底 B 同色,单张图像无法证明它到底是透明洞还是主体上的白色标记。当前不再强行猜唯一答案,而是在 [ermbg/candidates.py](ermbg/candidates.py) 生成 `transparent_hole` 和 `same_color_marking` 候选;后台 `/api/matte-candidates` 返回候选 JSON,UI 结果区用缩略图 tab 点击切换。

这两个修复对应不同边界:hard-edge 是局部策略识别后可以确定执行;同色内区是信息论歧义,必须暴露候选或等待用户意图。

### 10.3 verdict=not-pure-bg 时仍然继续

虽然加了 logger.warning,但管线照样跑。下一步可以让 router 在 `not-pure-bg` 时直接走 `noisy_bg` 策略。

### 10.4 当前主线:Known-B Candidate Matting

12 号样本相关分析已归档到 [docs/archive/sample-12-white-bg-panel-analysis.md](docs/archive/sample-12-white-bg-panel-analysis.md)。那份文档记录的是临时测试迭代和多次反转,不再作为当前设计依据。

当前主线重新收敛到项目原始 contract:

```text
Known background color B -> robust alpha + foreground recovery
```

优先级:

1. 饱和 / 白 / 黑 / 灰背景都最大化利用已知 B 的颜色证据。
2. keyer 只提供证据和约束,不直接替换最终 alpha。
3. 内部低 α 修复必须有 topology guard,外轮廓软边不能被抬高。
4. 多背景 QA 作为验收,特别看 black / checker / saturated 背景。
5. 只有当 foreground 与 B 同色、多对象选择、真实洞语义不明确,或局部材质策略冲突时,才升级到候选、轻交互或视觉语义提示。

Phase 2 名称:

```text
Known-B Candidate Matting
```

包含两部分:

1. **Robust Alpha Fusion**:像素颜色明显不是 B,但 matting alpha 偏低时,用 known-B/keyer 证据和 topology guard 保守抬 alpha。
2. **Ambiguity Candidate Generation**:像素颜色等于 B,语义上可能是洞也可能是主体图案时,自动生成少量候选让用户选择。

### 10.5 Region Policy Map

10 号样本暴露的不是主体归属歧义,而是**同一张图里不同区域需要不同抠图策略**。黑色描边是硬边,应该追求轮廓忠实;毛发/绒毛是软边,应该保留连续 alpha;透明或反光材质还需要单独的混合恢复策略。

因此下一层中间表示应从全图 `image_type ∈ {graphic, photo}` 升级为区域级 policy map:

```text
Image / local evidence / optional vision annotation
  -> RegionPolicyMap
  -> finite local MattingPlan
  -> deterministic executor
  -> QA / debug overlays
```

建议第一版标签:

| Policy | 典型区域 | 执行策略 |
|---|---|---|
| `hard_edge` | logo 描边、图标轮廓、文字边缘 | 用 known-B keyer / edge snapping 保护轮廓,允许抬高 alpha |
| `soft_hair` | 毛发、绒毛、烟雾状边缘 | 保护 BiRefNet soft alpha,禁止硬 keyer 拉满/拉空 |
| `opaque_interior` | 主体内部实色面 | known-B repair 可修低 alpha 缺口 |
| `translucent` | 玻璃、薄纱、半透明材质 | 保留低 alpha,用 unmix/foreground recovery 处理颜色 |
| `intentional_hole` | 花环、相框、镂空 logo、窗口 | hole 约束优先,不做 subject repair |
| `shadow_or_contact` | 投影、接触阴影 | 默认作为背景去除,后续可输出 shadow matte |
| `unknown` | 证据冲突区域 | 生成候选或请求轻量意图输入 |

视觉模型的角色是提供区域证据和策略先验,例如"这段是硬描边"、"这里是毛发"、"这个内圈是透明洞"。它不直接输出最终 alpha;最终像素操作仍由本地 keyer / matting / unmix / topology / QA 执行。

当前已落地的第一条本地规则是 `hard_edge`:在 white/black graphic 路径里,只对 keyer 高置信、与背景亮度差极大、当前 alpha 偏低、且贴近可信前景的小型组件抬 alpha。它解决 10 号样本的黑色细描边被软化问题,同时避免把所有外轮廓抗锯齿直接二值化。

### 10.6 候选与轻交互边界

当 local evidence 无法证明唯一答案时,不要先要求用户画 mask。系统应先生成有限数量的合理候选:

```text
base known-B matte
  -> detect ambiguous regions
  -> propose finite interpretations
  -> execute local plans
  -> render candidate composites
  -> user selects one
```

典型候选:

| Ambiguity | Candidate A | Candidate B |
|---|---|---|
| 内部区域颜色等于 B | transparent_hole | same_color_marking |
| 多对象 | selected_object | all_objects |
| 硬边/软边冲突 | hard_edge_snap | preserve_soft_alpha |

候选数量不固定。默认本地规则可以给 0-2 个候选;当升级到视觉/语言模型时,由模型根据风险区域和工具目录推理需要几个候选。约束是:

1. 候选必须由已注册本地工具组成,不能让模型直接输出 alpha / RGBA。
2. 候选优先作用于本地 `RiskRegion` 提供的区域;模型可以拆分、命名、排序和解释,但不能凭空绕过本地风险证据。
3. 超过 4 个候选通常说明歧义没有被结构化好,UI 应折叠为"推荐候选 + 需要用户意图"。

只有候选都不对时,才进入 keep/remove/hole 粗笔触或文字提示纠偏。纠偏输入表达意图,不是最终 alpha:

```text
instruction / keep_mask / remove_mask / hole_mask / subject_mask
  -> OwnershipMap / BackgroundMap / HoleMap / RegionPolicyMap / RiskMap
  -> finite MattingPlan
  -> deterministic executor
```

Phase 边界:

- **Phase 2**:Known-B robust alpha fusion + ambiguity candidate generation。
- **Phase 3**:RegionPolicyMap 的本地规则版,覆盖 hard_edge / soft_hair / intentional_hole。
- **Phase 4**:router / QA 发现风险后,可选择调用视觉模型提供区域策略先验。
- **Phase 5**:VLM/LLM 只输出有限 plan schema,仍不碰像素算法。

### 10.7 VLM Tool Planner

更准确的架构不是"大模型抠图",而是:

```text
Local Analyzer
  -> RiskRegion[] / local evidence / ToolCatalog
VLM/LLM Planner
  -> CandidatePlan[] with variable length
Local Executor
  -> deterministic ERMBG tools
Validator / QA
  -> reject illegal plans, score candidates, produce debug overlays
```

大模型负责推理:

- 这个风险区域在语义上更像主体、背景、透明洞、硬边、软边还是半透明材质。
- 每个区域应该调用哪些本地工具。
- 需要生成几个候选,以及候选的排序和解释。

大模型不负责:

- 输出最终 alpha。
- 输出最终 RGBA。
- 编写或选择任意未注册图像处理代码。
- 绕过 topology guard / QA / 参数范围限制。

#### ToolCatalog v0

第一版只暴露已经存在或很容易包装的确定性工具:

| Tool | 作用 | 底层实现 |
|---|---|---|
| `preserve_hole` | 保留透明内洞 | 保持指定区域低 alpha |
| `fill_same_color_region` | 把同背景色内区解释为主体图案 | 候选 executor 局部抬 alpha 并保留输入 RGB |
| `repair_opaque_interior` | 修复主体内部低 alpha 缺口 | `repair_alpha_with_known_bg_key` / `repair_alpha_with_subject_support` |
| `snap_hard_edge` | 修复 logo / 文字 / 描边硬边 | `repair_hard_edge_alpha` |
| `preserve_soft_alpha` | 保护毛发 / 绒毛 / 烟雾软边 | 禁止 hard gate / hard fill 作用于该区域 |
| `mark_translucent` | 标记玻璃 / 薄纱等半透明区域 | 保留 partial alpha,交给 unmix / foreground recovery |

每个工具必须有机器可读 contract:

```text
name / purpose / input parameters / parameter ranges
allowed_when / rejects_when / risks
```

#### Plan Schema v0

Planner 输出受限 JSON,候选数量可变:

```json
{
  "candidates": [
    {
      "id": "transparent_center",
      "label": "中心透明",
      "confidence": 0.78,
      "operations": [
        {"tool": "preserve_hole", "region_id": "r3"},
        {"tool": "snap_hard_edge", "region_id": "r1", "alpha_floor": 0.95}
      ],
      "reason": "主体像环形徽章,中心白色区域更可能是镂空"
    }
  ]
}
```

执行前必须校验:

- `tool` 必须在 `ToolCatalog` 中。
- `region_id` 必须来自本地 `RiskRegion` 或用户 intent map。
- 参数必须在工具 contract 允许范围内。
- 工具的 `allowed_when` 必须能由本地证据或用户意图支持。

第一步实现目标是本地闭环,不接真实 VLM:

```text
RiskRegion[] + ToolCatalog -> rule/mock CandidatePlan[] -> PlanExecutor -> MatteCandidate[] -> QA/report/UI
```

等 schema、validator、executor 稳定后,再把 VLM 接到 `CandidatePlan[]` 生成位置。

---

## 11. 基础设施

- **本地**:Mac,Python 3.12,uv 管理 `.venv/`。BiRefNet-matting 跑 MPS,~1 GB。SDXL 在 16 GB MPS OOM。
- **ComfyUI 远端**:`http://192.168.0.8:8000`(Win + RTX 4090,24 GB VRAM)。重生成 / 重 inference 都走它。客户端在 [ermbg/probe/comfyui.py](ermbg/probe/comfyui.py),工作流模板 `ermbg/probe/comfyui_*.json`。
- **OpenAI**:`OPENAI_API_KEY` 在 `.env`,只在显式需要 `gpt-image-1` 时用。

### 11.1 ComfyUI 节点已部署

`ErmbgAutoMatte` / `ErmbgClassify` 已经装到上述远端 ComfyUI 服务器。安装方式见 [DEPLOY.md](DEPLOY.md):

- `ermbg` 包通过 pip install 进 `E:\ComfyUI\.venv`
- 节点目录改名为 `E:\ComfyUI\custom_nodes\ermbg-comfy\`(避免和 ermbg 包同名)
- 重启 ComfyUI 后 `/object_info` 暴露两个节点

### 11.2 openclaw skill

ERMBG 合并进 openclaw 现有的 `comfyui-rmbg` skill,作为 `--mode ermbg` 子模式存在 — 一个 skill,三个 mode(`rmbg` / `ermbg` / `edge-wand`),触发词区分:普通"抠图 / 去背景"走标准 RMBG,**"智能抠图 / AI生图抠图 / smart matte / ERMBG"** 触发 ermbg 路径。产物归档到 `~/.openclaw/media/openclaw-production/images/rmbg/`。详细参数与合并补丁见 [integrations/openclaw/README.md](integrations/openclaw/README.md) 和 [integrations/openclaw/comfyui-rmbg-patch/](integrations/openclaw/comfyui-rmbg-patch/)。

---

## 12. 关键判断

```
干净的半透明抠图 = 准确 α + 干净 foreground RGB + 去背景污染。
```

实现拆分:
- **router** 看图选路:passthrough / chromatic / luminance / noisy
- **α** 来自 BiRefNet-matting,关键工况由 keyer 补漏
- **F** 由 despill 给出,已知 B 时 unmix 是闭式正解
- **去污染** 集成在 despill 内

设计目标本质:用户给图就行,系统自己判断怎么抠。
