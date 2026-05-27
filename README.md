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
- **Local ownership 归属判断**:对 hole / soft subject / shadow-like layer 做本地多假设评分,
  默认走本地确定性证据。
- **Owned shadow 保留**:对符合 known-B scalar darkening 的源图阴影生成 shadow matte,
  强度由本地 CV 测量。
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

## Local Ownership

当前方向是 Local Ownership:默认用本地证据做 region ownership 判断。旧的模型规划路线已经归档,
不再作为工程主路径。

```text
known background image
  -> local matte
  -> local evidence regions
  -> local multi-hypothesis ownership scoring
  -> execution-mask arbitration
  -> protected matte only when soft subject material needs protection
```

当前角色:

- `hole`:透明洞/背景区域,保持低 alpha。
- `opaque_subject`:硬主体漏检,允许受保护的 alpha repair。
- `subject_soft_layer`:玻璃、辉光、烟雾、柔边或半透明主体层,保护 soft alpha。
- `shadow_like_layer`:已知背景的 scalar darkening,走 shadow matte。
- `conservative_unknown`:证据不足,保留当前 alpha。

当前 G02/G04/G06 green+white 小批次:

```bash
.venv/bin/python scripts/10_local_ownership_batch.py \
  --out-dir out/local_ownership_resolved2_g02_g04_g06_20260527 \
  --sample-id G02,G04,G06 \
  --variants green,white
```

结果:

- `ok=6/6`
- `expected_role_hit=6/6`
- G02 使用 base matte,避免 material-protected rerun 误伤软阴影。
- G04/G06 使用 protected matte,避免半透明/辉光被 keyer/repair 泛白或变硬。

详情见 [docs/local-ownership.md](docs/local-ownership.md)。

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

正式 Web 抠图路径是 **`comfy-ermbg`**:Mac 侧负责上传、HTTP 编排和轻量诊断,
远端 ComfyUI `ErmbgAutoMatte` 节点运行完整 ERMBG pipeline。后续算法更新
不能只以本地 Python 跑通为准,必须同步验证远端节点和 Web API。

`comfy_nodes/` 提供:

- **ERMBG AutoMatte**:IMAGE 加可选 source/subject mask -> foreground、alpha、debug summary。
- **ERMBG Classify (preview)**:只跑 router,用于工作流分支。

最简工作流:

```text
KSampler -> VAEDecode -> ERMBG AutoMatte -> SaveImage(RGBA)
```

开发/迭代验证流程见 [docs/comfy-ermbg-development.md](docs/comfy-ermbg-development.md)。
节点部署见 [comfy_nodes/README.md](comfy_nodes/README.md) 和 [DEPLOY.md](DEPLOY.md)。

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

# Local ownership / shadow 专项
.venv/bin/pytest tests/test_ownership.py tests/test_shadow.py tests/test_risk.py -q

# Comfy/Web 正式路径相关变更
.venv/bin/pytest tests/test_api.py tests/test_comfy_ermbg_matte.py tests/test_comfy_nodes.py tests/test_web.py -q
```

全量测试数量会随当前分支变化;接力前以本地 `.venv/bin/pytest -q` 为准。

真实回归样本放在 `samples/regression/`。例如
`samples/regression/small_ui_icon_green/` 覆盖小尺寸 UI 图标、主体贴边、
角落绿幕稳定但整圈边缘被主体污染的场景。

## 文档索引

- [docs/local-ownership.md](docs/local-ownership.md):
  当前工程接力入口:local ownership、执行层仲裁、G02/G04/G06 现状和复现命令。
- [docs/comfy-ermbg-development.md](docs/comfy-ermbg-development.md):
  正式 `comfy-ermbg` 路径的开发、同步、远端 smoke 和 Web 验证流程。
- [docs/archive/](docs/archive/):
  旧模型规划、candidate-planner、G02 单样本路线归档,只作历史参考。
- [DEPLOY.md](DEPLOY.md):ComfyUI 部署。
- [comfy_nodes/README.md](comfy_nodes/README.md):ComfyUI 节点用法。

## 开发反馈

提 issue 时请附上:

- 输入图;
- `*.report.json`;
- 能看清 artifact 的 `*_qa/on_black.png` 或其他合成图;
- 预期语义,尤其是同色区域应当是主体材质、透明洞,还是 owned shadow。
