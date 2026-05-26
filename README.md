# ERMBG

> 给 AI 出图配套的智能抠图工具:
> **指定背景色出图 -> 自动选策略抠干净 -> 输出可直接复用的 RGBA**。

ERMBG 面向 AI 生成资产。只要生成阶段能指定已知、尽量恒定的背景色,
抠图就不再完全靠猜:系统可以利用观测颜色 `C`、已知背景 `B`、alpha
和 foreground recovery 模型做可验证的反演。

```text
AI image model  ->  known background image  ->  ERMBG  ->  RGBA asset
                      default: RGB(0,200,0)
```

目标很简单:用户给图,系统自己判断怎么抠。

## 关键特点

- **自动路由**:干净 RGBA、绿/品红/青等饱和底、白/黑/灰底、噪声底自动分流。
- **Known-B foreground recovery**:已知背景时用 linear-RGB unmix,不只靠经验 chroma 脚本。
- **RGBA hygiene check**:脏透明 PNG 的白边/黑边/旧背景泄漏/硬二值 alpha 会被识别并重抠。
- **Keyer + matting 融合**:补小漏检、守住 topology、修 hard edge、对同色歧义生成候选。
- **Owned shadow 保留**:对符合 known-B scalar darkening 的源图阴影生成 shadow matte;
  VLM 只做语义约束,强度仍由本地 CV 测量。
- **多背景 QA**:black / white / grey / cyan / magenta / checker,并带 lightwrap 变体。
- **多入口**:CLI、Python API、ComfyUI 节点、Web/API、OpenClaw skill。

## 安装

推荐 Python 3.12。Mac 上推荐 `uv`:

```bash
git clone <this-repo>
cd ERMBG
uv venv
source .venv/bin/activate
uv pip install -e ".[torch,dev]"
```

首次 BiRefNet 运行会下载约 1 GB 权重到 HuggingFace cache。

## 快速使用

```bash
# 端到端抠图,router 自动选策略
.venv/bin/ermbg matte input.png

# 只看诊断/router 决策
.venv/bin/ermbg diagnose input.png

# 批量
.venv/bin/ermbg phase1 --input-dir samples/legacy/inputs --out-dir out/phase1

# 可选 ownership mask,只作为约束,不直接替换最终 alpha
.venv/bin/ermbg matte input.png --subject-mask ownership_mask.png
```

输出:

```text
*_rgba.png
*_alpha.png
*_shadow.png
*_foreground.png
*_trimap.png
*.report.json
*_qa/on_*.png
```

## VLM Semantic Prior

VLM 是语义约束层,不是 alpha 生成器。

```text
VLM: subject / owned shadow / material-region plausibility
CV: pixel membership, alpha, foreground RGB, shadow opacity
```

`--vlm-prior` 默认是 shadow-only:

```bash
.venv/bin/ermbg matte samples/vlm_eval_game/ui_hard_button_soft_shadow/green.png \
  --vlm-prior \
  --vlm-provider comfy-qwen \
  --vlm-model Qwen3-VL-4B-Instruct-FP8
```

同色主体材质保护可以显式打开,但不要和 shadow 验收混在一起:

```bash
.venv/bin/ermbg matte input.png --vlm-prior --vlm-prior-mode material
```

Provider:

- `openai`
- `comfy-qwen`:使用远端 ComfyUI `http://192.168.0.8:8000`

## Python API

```python
from ermbg import classify_image, matte_image

r = matte_image("input.png", output_dir="out/", qa=True)

r.rgba
r.strategy_name
r.report["qa"]["edge_halo_score_mean"]

s = classify_image("input.png")
print(s.bg_type, s.image_type, s.notes)
```

输入支持 path、`numpy uint8` (`HxWx3` / `HxWx4`) 或 PIL Image。
完整示例见 [examples/quickstart.py](examples/quickstart.py)。

## ComfyUI

`comfy_nodes/` 提供:

- **ERMBG AutoMatte**:IMAGE 加可选 source/subject mask -> foreground、alpha、debug summary。
- **ERMBG Classify (preview)**:只跑 router,用于工作流分支。

最简工作流:

```text
KSampler -> VAEDecode -> ERMBG AutoMatte -> SaveImage(RGBA)
```

详见 [comfy_nodes/README.md](comfy_nodes/README.md) 和 [DEPLOY.md](DEPLOY.md)。

## OpenClaw

ERMBG 已合入 OpenClaw 的 `comfyui-rmbg` skill,作为 `--mode ermbg` 子模式。
触发词如 **智能抠图 / AI生图抠图 / smart matte / ERMBG** 走 ERMBG 路径;
普通“抠图 / 去背景”仍可走标准 RMBG。

详见 [integrations/openclaw/README.md](integrations/openclaw/README.md)。

## 示例效果

下面四组图来自 legacy samples,对比 ERMBG 和 RMBG baseline。每组包含原图、
checker 合成、白/黑底对比和 alpha 对比。

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

指标来自各自 `report.json`;ERMBG 和 RMBG baseline 使用同一套 QA 评分。

## 测试

```bash
# 日常重点回归
.venv/bin/pytest -q -m core

# 提交/接力前全量
.venv/bin/pytest -q

# Shadow / semantic-prior 专项
.venv/bin/pytest tests/test_shadow.py tests/test_vlm_semantic.py -q
```

全量测试最近一次通过为 154 项。

## 文档索引

- [clean_transparent_matting_engineering_plan.md](clean_transparent_matting_engineering_plan.md):
  当前工程接力入口,pipeline、defaults、CLI、report 字段、focus。
- [docs/g02-soft-shadow-analysis.md](docs/g02-soft-shadow-analysis.md):
  G02 soft shadow、Qwen provider、scope correction、artifact 和下一步。
- [docs/known-b-candidate-matting.md](docs/known-b-candidate-matting.md):
  EvidenceRegion -> CandidatePlan 设计和 VLM planner 边界。
- [DEPLOY.md](DEPLOY.md):ComfyUI 部署。
- [comfy_nodes/README.md](comfy_nodes/README.md):ComfyUI 节点用法。

## 开发反馈

提 issue 时请附上:

- 输入图;
- `*.report.json`;
- 能看清 artifact 的 `*_qa/on_black.png` 或其他合成图;
- 预期语义,尤其是同色区域应当是主体材质、透明洞,还是 owned shadow。
