# ERMBG

> 给 AI 出图配套的智能抠图工具:**指定背景色出图 → 自动选策略抠干净 → 输出可直接复用的 RGBA**。

---

## 设计理念

抠图难,本质上是因为单张复杂背景图的边缘像素已经把前景、背景、模糊、压缩噪声全混在一起,信息论上不可逆。但**当下游用 AI 生图大模型出图时,你完全可以让背景在生成阶段就是已知纯色**(绿幕 / 白底 / 黑底)。这就把"反演混合"问题改写成了"已知 B 解 α、F"的可解问题。

ERMBG 把这套链路拆成两段:

```
[ 第一段 ]  AI 生图模型   →  指定背景色出图 (默认绿幕 RGB(0,200,0))
[ 第二段 ]  ERMBG 抠图  →  自动识别背景类型,选最优策略,输出 RGBA
```

第一段只是约定生图参数,第二段是这个仓库的工程实现。**用户不需要填任何参数**,系统自己判断每张图怎么抠。

---

## 关键特点

### 1. 前端 router:看图选策略

每张图先经过 [`classify_strategy`](ermbg/router.py),输出一份 `Strategy`,里面定好 keyer 模式 / despill 方法 / 是否补漏 / 是否压晕。决策表:

| 情形 | 触发条件(自动检测) | keyer | despill | 备注 |
|---|---|---|---|---|
| 透明 PNG 且干净 | RGBA 通过 hygiene 检查 | — | none | 直接 pass-through |
| 透明 PNG 但脏 | hygiene 检测到边缘 halo / 旧背景泄漏 / α 二值化 | 重抠 | 按背景再选 | **不让脏资产蒙混过关** |
| 饱和底 (绿/品红/青...) | OKLab chroma ≥ 8 | chromatic | auto | 最佳工况 |
| 白底 | L ≥ 85,chroma 低 | luminance | unmix | |
| 黑底 | L ≤ 15,chroma 低 | luminance | unmix | |
| 灰底 | 中亮度,chroma 低 | luminance(放宽阈值) | local_borrow | 信号弱,以网络 α 为主 |
| 噪声底 | 角落 σ > 18 | none | local_borrow | 不是纯色底,降级 |

`image_type`(graphic vs photo)进一步影响 keyer 阈值与 gate 是否启用:卡通 / logo 收紧并启用 gate(压去 BiRefNet 的羽化白晕),自然图放宽并保护软边(头发不被切硬)。

### 2. 已知 B 时用闭式 unmix,不靠经验

经典绿幕脚本用 `chroma_cap`(把 G 通道压到 max(R, B))这种启发式;但只要 B 已知,正解就是

```
F = (C − (1 − α) · B) / α
```

ERMBG 直接闭式解,低 α 区域用 KDTree 借色兜底。在饱和 B 上额外叠 chroma_cap 抹掉残余色边,综合最优。

### 3. 智能脏度检测(无监督 / 无参数)

如果你给的是已经抠过的 RGBA,系统不会盲目复用源 α,而是跑 [`assess_source_alpha`](ermbg/router.py) 三个无监督指标:

- **fringe_dE**:在软边带把 RGB 按 straight/sRGB-premul/linear-premul 三种 α 包装解出 F,取最贴近 interior 的那个的 OKLab 距离 — 健壮地穿过所有 α 包装惯例,识别白边/黑边/绿边。
- **low_alpha_residual**:α≈0 的像素 RGB 是否泄露 interior 色,或保留了原始底色。
- **bimodal_fraction**:α 是否被硬阈值二值化,缺失 AA。

任何一项不通过 → 自动重抠,把脏 RGBA 当原图处理。

### 4. Keyer ↔ Matting 双向校准

- **merge**:keyer 看到的小连通分量(BiRefNet 漏掉的小红点),按 coverage<30% 判定漏检,补回主 α(并加 1px feather 防止亮斑)。
- **known-B repair**:白底 / 黑底 graphic 额外用完整 OKLab 距离判断"不是背景 B"的内部低 α 缺口,只修远离外轮廓且连接可信前景的区域。
- **hard-edge repair**:白底 / 黑底 graphic 里,对高对比细描边这类硬边组件局部抬 α,保住 1px 墨线轮廓。
- **candidate generation**:同色内区这类信息论歧义不强行猜唯一答案,自动给出 `transparent_hole` / `same_color_marking` 等候选,后台 UI 可点缩略图切换。
- **gate**(只对 graphic):keyer 高度自信"是背景"(ΔE 极小)的像素,把 matting 的羽化 α 压到 keyer α — 修掉 BiRefNet 在硬边图形上典型的白色光晕。
- **fg protect**:matting α ≥ 0.85 的像素永不被 gate 拉低,保 photo 类的软边。

### 5. 多背景 QA 自带 ground truth

把 RGBA 合成到 black / white / grey / cyan / magenta / checker 6 张背景,每张配 lightwrap 变体,自动算 recomposition_error / edge_halo_score / alpha_noise / thin_structure_preservation。**不主观看图,用数字判通过**。

---

## 安装

需要 Python 3.11 / 3.12(PyTorch wheel 不全)。Mac 推荐 uv:

```bash
git clone <this-repo>
cd ERMBG
uv venv && source .venv/bin/activate
uv pip install -e ".[torch,dev]"
```

首次运行会从 HuggingFace 下载 BiRefNet-matting 权重(≈1 GB)到 `~/.cache/huggingface`。

---

## 使用

### CLI

```bash
# 端到端,自动选策略
ermbg matte input.png

# 仅看 router 决策(不跑抠图模型,秒回)
ermbg diagnose input.png

# 批量
ermbg phase1 --input-dir samples/legacy/inputs --out-dir out/phase1

# 可选:传入独立主体归属 mask,只作为 ownership 约束,不直接替换 alpha
ermbg matte input.png --subject-mask ownership_mask.png

# ComfyUI 忙时可先离线渲染 prompt-aware subject_mask 工作流 JSON
.venv/bin/python scripts/05_comfy_subject_mask_workflow.py \
  --input input.png \
  --prompt "the complete object to keep" \
  --out out/comfy_workflows/subject_mask_ermbg.json \
  --filename-prefix subject_mask_ermbg

# ComfyUI 空闲时加 --submit:会等待完成并下载 foreground / alpha / subject_mask
.venv/bin/python scripts/05_comfy_subject_mask_workflow.py \
  --input input.png \
  --prompt "the complete object to keep" \
  --out out/comfy_workflows/subject_mask_ermbg.json \
  --filename-prefix subject_mask_ermbg \
  --submit
```

输出:`*_rgba.png` / `*_alpha.png` / `*_foreground.png` / `*_trimap.png` / `*.report.json` / `*_qa/on_*.png`。

### Python API

```python
from ermbg import matte_image, classify_image

# 一行抠图
r = matte_image("input.png", output_dir="out/", qa=True)
r.rgba              # H×W×4 numpy uint8
r.strategy_name     # 'saturated_bg' / 'white_bg' / 'rgba_passthrough' / ...
r.report['qa']['edge_halo_score_mean']

# 可选 subject_mask 只作为主体归属约束,不会直接替换最终 alpha
r = matte_image("input.png", subject_mask="ownership_mask.png", output_dir="out/", qa=True)

# 秒回预览(不跑 BiRefNet)
s = classify_image("input.png")
print(s.bg_type, s.image_type, s.notes)
```

支持 path / numpy uint8 (HxWx3 或 HxWx4) / PIL Image。完整示例见 [examples/quickstart.py](examples/quickstart.py)。

### ComfyUI 节点

`comfy_nodes/` 提供两个自定义节点:

- **ERMBG AutoMatte**:接 IMAGE(+ 可选 source_mask / subject_mask),自动路由,出 foreground / alpha / 调试 summary。
- **ERMBG Classify (preview)**:只跑 router,秒回 bg_type / image_type / 完整策略 JSON,做工作流分支用。

最简工作流:

```
KSampler → VAEDecode → ERMBG AutoMatte → SaveImage(RGBA)
```

详见 [comfy_nodes/README.md](comfy_nodes/README.md)。把节点装到局域网 ComfyUI 服务器的步骤见 [DEPLOY.md](DEPLOY.md)。

### openclaw bot skill

ERMBG 合并进了 openclaw 已有的 `comfyui-rmbg` skill,作为 `--mode ermbg` 子模式。专属触发词 **"智能抠图 / AI生图抠图 / smart matte / ERMBG"** 走 ERMBG 路径,普通"抠图 / 去背景"仍走原来的 RMBG。一个 skill 三个 mode,意图区分清楚。详见 [integrations/openclaw/README.md](integrations/openclaw/README.md)。

---

## 项目结构

```
ermbg/
  router.py        前端策略选择 + RGBA 卫生检测
  keyer.py         chromatic / luminance keyer + merge / gate
  segmenter.py     BiRefNet-matting / GrabCut 回退
  diagnose.py      背景诊断 (B / σ / verdict)
  despill.py       unmix / chroma_cap / local_borrow / closed_form
  risk.py          EvidenceRegion/RiskRegion 本地证据提取
  planner.py       ToolCatalog / CandidatePlan / rule planner
  vlm_planner.py   PlannerClient 协议 / rule client / JSON parser
  vlm_payload.py   PlannerPromptBundle + 缩略图/overlay/crop -> VLM 请求
  executor.py      执行 CandidatePlan 的本地工具调度器
  candidates.py    执行 planner 候选,输出可选 RGBA
  matting.py       端到端 matte() 主入口
  api.py           matte_image / classify_image 高阶 API
  web.py           后台上传 UI + 候选缩略图切换
  qa.py            6 背景 QA + 多指标
  cli.py           segment / diagnose / matte / phase1 / probe
  lightwrap.py     Brinkmann 边缘光晕修正
  metrics.py       OKLab 距离 / 背景采样 / IoU / Hausdorff
  io.py / colorspace.py / types.py

comfy_nodes/       ComfyUI 自定义节点
examples/          quickstart 等示例
tests/             123 项 pytest
samples/vlm_eval/ AI 生成的 VLM planner 评估集,每个 case 含 input.png / case.json
samples/legacy/inputs/    12 张测试图,涵盖各类背景
```

---

## 实测效果与 RMBG 对比

下面四组图来自 `samples/legacy/inputs/6.png`、`7.png`、`8.png`、`11.png` 的实测输出。每组从左到右依次是原图、ERMBG 合成到 checker 背景、RMBG baseline 合成到 checker 背景、白底 ERMBG / RMBG 对比、黑底 ERMBG / RMBG 对比、ERMBG / RMBG alpha 对比。

![Sample 6 ERMBG vs RMBG](docs/assets/readme/sample_6_ermbg_vs_rmbg.png)

![Sample 7 ERMBG vs RMBG](docs/assets/readme/sample_7_ermbg_vs_rmbg.png)

![Sample 8 ERMBG vs RMBG](docs/assets/readme/sample_8_ermbg_vs_rmbg.png)

![Sample 11 ERMBG vs RMBG](docs/assets/readme/sample_11_ermbg_vs_rmbg.png)

| 输入 | ERMBG halo mean | RMBG halo mean | ERMBG recomp err | RMBG recomp err |
|---|---:|---:|---:|---:|
| 6 | 1.06 | 7.15 | 0.0216 | 0.0206 |
| 7 | 2.34 | 4.75 | 0.0071 | 0.0063 |
| 8 | 1.25 | 3.80 | 0.0097 | 0.0577 |
| 11 | 3.03 | 5.64 | 0.0202 | 0.2375 |

指标来自各自的 `report.json`:ERMBG 使用本仓库自动路由和 despill,RMBG baseline 使用 `comfyui-rembg-isnet-general-use` 后再跑同一套 QA。

---

## 目前的工况

| 输入 | 路由 | 评价 |
|---|---|---|
| 绿底 / 品红 / 青底 (graphic) | saturated_bg + chromatic + auto + gate | 最佳,recomp_err ≈ 0.02 |
| 白底 / 黑底 (graphic) | white_bg / black_bg + luminance + known-B repair + unmix + gate | 浅色面板等低对比内部区域可自动修复 |
| 灰底 photo | grey_bg + luminance(宽阈值)+ local_borrow | 软边保留 |
| 自然底 photo | noisy_bg + local_borrow | 降级到只用网络 α |
| 干净 RGBA | rgba_passthrough | 直接复用 |
| 脏 RGBA | hygiene 拒绝 → 重抠 | 不让脏资产蒙混 |

下一阶段方向是 **Known-B Candidate Matting + Evidence-to-Policy Planning**:优先把已知背景色下的 keyer / matting 融合做扎实;遇到同色内洞、多对象等信息论歧义时自动生成候选让用户选择;再把本地 CV 提取的证据区域交给视觉/语义模型解释成区域策略,例如 hard_edge / soft_hair / translucent / intentional_hole,由本地确定性算法分别执行。当前已落地 `hard_edge` 局部修复,以及同色内区的 `transparent_hole` / `same_color_marking` 候选和后台缩略图切换。

VLM/LLM 的定位是 **工具调度器 + 区域策略规划器**,不是直接抠图模型。ERMBG 会把本地 `RiskRegion`(实现名;语义上是 EvidenceRegion)、证据摘要和 `ToolCatalog` 告诉模型,由模型推理这些证据区域在语义上是什么、哪些区域调用哪些本地工具、需要生成几个候选;候选数量不固定,但每个候选必须由已注册工具组成,最终 alpha / foreground / despill / QA 仍由本地确定性代码执行。传给 planner/VLM 的区域 JSON 同时包含内部兼容 `kind` 和更证据化的 `evidence_kind`,工具也同时暴露 `allowed_region_kinds` / `allowed_evidence_kinds`。

详细工程现状见 [clean_transparent_matting_engineering_plan.md](clean_transparent_matting_engineering_plan.md)。

---

## 测试

```bash
# 日常重点回归:router/keyer/API/CLI/subject-mask workflow
.venv/bin/pytest -q -m core

# 提交前全量
.venv/bin/pytest -q
```

覆盖 router 决策表 / keyer / despill / 诊断 / hygiene / planner / VLM planner adapter / executor / risk / API / Web 候选接口 / ComfyUI 节点 / subject-mask workflow / 端到端 smoke,共 123 项。

---

## 开发反馈

提 issue 时附上:输入图、`*.report.json`、对应 `*_qa/on_black.png`(白晕一眼可见)。
