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
| [ermbg/lightwrap.py](ermbg/lightwrap.py) | Brinkmann light wrap 边缘晕修 |
| [ermbg/qa.py](ermbg/qa.py) | 6 背景合成 + halo / noise / 细结构指标 |
| [ermbg/matting.py](ermbg/matting.py) | 端到端 `matte()` 主入口,接 router |
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
tests/test_lightwrap.py    halo 抑制
tests/test_matting.py      end-to-end smoke
tests/test_router.py       策略分类(saturated/white/black/grey/passthrough/graphic/photo)
tests/test_api.py          高阶 API / subject_mask 输入
tests/test_cli.py          CLI subject_mask smoke
tests/test_comfy_nodes.py  ComfyUI 节点输入输出
tests/test_comfy_subject_workflow.py  CLIPSeg→ERMBG workflow 渲染/下载
```

日常重点回归跑 `.venv/bin/pytest -q -m core`,覆盖 router / keyer / API / CLI / subject-mask workflow。提交前跑全量 `.venv/bin/pytest -q`;当前 91 项全部通过。新增模块要带 smoke 测试。

---

## 10. 已知问题 / TODO

### 10.1 RGBA 输入的智能脏度检测(已实现)

`rgba_passthrough` 现在不再无条件触发。只要源图 α 至少 5% 像素非全不透明,router 会先跑 `assess_source_alpha(rgb, alpha)` 做无监督卫生检查:

- **fringe_dE**:在 α∈(0.05, 0.6] 的软边带,按 straight / sRGB-premul / linear-premul 三种包装恢复 implied F,取最贴近 opaque interior 的 OKLab 距离。
- **low_alpha_residual**:检查 α≈0 区域 RGB 是否接近 premultiplied black,或是否泄露 interior / 原始背景色。
- **bimodal_fraction**:识别硬二值 α,避免缺失 AA 的粗抠图被直接复用。

任一指标失败 → strategy 从 `rgba_passthrough` 落回正常背景分类路径,由主流程重抠。测试覆盖 clean passthrough、halo reject、low-alpha leak、binarized matte、无 opaque interior 等边界。

### 10.2 白底 keyer merge 在小连通分量上失效

10 号图(白底蓝星 + 红环 + 小红点),luminance keyer 算 α 没问题,但 `merge_alpha_components` 没把小红点补回来 — 因为白底场景下 BiRefNet 给的小红点 α 不是严格 0,触发了"matting 已经看见"的跳过分支。需要在 white_bg / black_bg 路径用更严格的"matting present"阈值,或改用主体大小过滤。

### 10.3 verdict=not-pure-bg 时仍然继续

虽然加了 logger.warning,但管线照样跑。下一步可以让 router 在 `not-pure-bg` 时直接走 `noisy_bg` 策略。

### 10.4 下一阶段:Semantic Matting Planner

12 号样本暴露的问题不是单纯"洞要不要填",而是"这块区域是否语义上属于主体"。下一阶段不继续调白底 luminance keyer 阈值,而是把主体归属作为独立信号接入:

```text
Vision / prompt-aware segmentation -> OwnershipMap
Language planner                  -> 有限操作计划
ERMBG local algorithms             -> α 修复 / unmix / despill / QA
```

近期落地顺序:

1. **接通 prompt-aware `subject_mask` 工作流**:ComfyUI 中用 CLIPSeg / Florence / SAM 生成完整主体 ownership mask,传给 `ERMBG AutoMatte(subject_mask=...)`。当前已新增 `ermbg/probe/comfyui_clipseg_ermbg.json` 和 `scripts/05_comfy_subject_mask_workflow.py`,支持离线渲染 JSON,空闲后 `--submit` 等待完成并下载 foreground / alpha / subject_mask / summary。
2. **CLI/API 回归入口**:CLI `matte` 支持 `--subject-mask`,方便用 `samples/outputs/clipseg_12/*.png` 这类离线 mask 跑 12 号回归;`scripts/04_subject_mask_regression.py` 固化 nomask vs subject-mask 的 QA 对比。
3. **12 号验收基准**:右上角浅绿色面板缺口必须修复;黑底 / checker / cyan / magenta QA 不允许新增白边;report 中必须记录 `keyer.subject_repair.accepted_pixels`。当前 CLIPSeg prompt-aware mask 已把 recomp 从 0.0308 降到约 0.012,black halo 从约 10.0 降到约 6.0。
4. **抽象中间表示**:在 `subject_mask` 稳定后再升级为 `OwnershipMap / HoleMap / RiskMap / Plan`,记录来源、置信度、故意透明洞、matting/keyer/semantic 分歧。
5. **触发策略保守化**:只在低对比、matting/keyer 大面积冲突、疑似内部洞、QA 失败或用户明确指定对象时调用视觉语义分支;简单绿幕/干净 RGBA 仍走本地快速路径。

原则:视觉模型负责理解和提供区域证据,LLM 只在有限操作集合内规划,最终像素级 α / foreground RGB / QA 仍由本地确定性代码执行。

### 10.5 产品重构:Interactive Intent Matting

2026-05-26 进一步把第二阶段目标从"语义 planner"收束成更贴近用户的 **Interactive Intent Matting**:

```text
默认智能完成。
失败时给用户一个极轻的纠偏入口。
纠偏输入表达意图,不是让用户精修 alpha。
```

这不是推翻 `Semantic Matting Planner`,而是把它的产品边界说清楚:planner 不应该要求用户画完美 mask,也不应该让 VLM/LLM 直接改像素。用户只需要补充系统缺失的"意图信息",例如:

- 文字提示:`这是一个花环,洞要抠干净` / `保留整个浅绿色面板` / `只保留左边这个徽章`;
- 粗笔触 keep:`这块要保留`;
- 粗笔触 remove:`这块要去掉`;
- 粗笔触 hole:`这里是洞,要透明`;
- prompt-aware subject mask:由 CLIPSeg / Florence / SAM 根据一句提示生成粗 ownership support。

这些输入统一进入中间表示,而不是直接替换最终 alpha:

```text
instruction / keep_mask / remove_mask / hole_mask / subject_mask
  -> OwnershipMap / BackgroundMap / HoleMap / RiskMap
  -> 有限 MattingPlan
  -> ERMBG deterministic executor
  -> QA / debug overlays
```

建议第二阶段 API 表面逐步扩展:

```bash
ermbg matte input.png \
  --keep-mask rough_keep.png \
  --remove-mask rough_remove.png \
  --hole-mask rough_hole.png \
  --instruction "this is a wreath; keep leaves, remove the center hole"
```

ComfyUI 节点对应增加可选 MASK/STRING 输入:

```text
keep_mask
remove_mask
hole_mask
instruction
```

第一版实现原则:

1. **自动优先**:绿幕/黑白底清晰图/干净 RGBA/小漏检/脏边仍然无参数完成。
2. **轻交互只补意图**:笔触和文字只产生 ownership / background / hole 约束,不直接成为最终 matte。
3. **本地代码细化**:边缘、despill、unmix、gate、QA 仍由本地确定性算法执行。
4. **hole 约束优先级最高**:花环、相框、窗户、镂空 logo 的洞不能被自动 subject repair 填掉。
5. **报告可解释**:`report.json` 要记录哪些用户意图被采纳、哪些候选被拒绝、repair mask / rejected mask / risk overlay 在哪里。

从用户体验上,系统应该主动暴露极少量纠偏动作:

```text
检测到浅色主体可能被白底吃掉。可以粗涂要保留的区域。
检测到内部洞不确定。可以标记"洞要透明"。
检测到多个对象。可以粗涂目标对象或用一句话指定。
```

阶段边界:

- **Phase 2**:消费已有粗 mask / instruction,生成 Comfy workflow,不默认自动远端调用;先把中间表示和 report 打通。
- **Phase 3**:router / QA 发现风险后,可选择自动请求 prompt-aware segmentation 分支。
- **Phase 4**:VLM/LLM 只输出有限 plan schema,仍不碰像素算法。

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
