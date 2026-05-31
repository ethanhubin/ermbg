# ERMBG ComfyUI Nodes

把 ERMBG 的智能抠图(自动选 saturated / white / black / passthrough 策略)接进 ComfyUI,贴合"AI 出图 → 自动抠干净"的工作流。

## 节点

### `ERMBG AutoMatte`
端到端,接 KSampler/VAEDecode 的输出,出干净 RGBA。

输入:
- `image` (IMAGE,必填) — 要抠的图
- `source_mask` (MASK,可选) — 已有的 α(比如来自其他分割节点)。给了之后,router 会自动评估它干不干净:干净就直接 pass,脏就重抠。
- `subject_mask` (MASK,可选) — 独立主体归属 mask,只用于修复主体内部低 α 缺口,不会直接替换最终 α。
- `despill` — 默认 `auto (router decides)`,需要时可手动覆盖
- `use_keyer` — 默认 `auto`,可强开/强关
- `bg_color` — 当源 RGBA 太脏被重抠时用作合成底色,默认绿幕 `0,200,0`
- `matting_model` — HF 模型 id,默认 BiRefNet-matting

输出:
- `foreground` (IMAGE) — 去污染后的前景 RGB(直 α,非预乘)
- `alpha` (MASK) — 最终 α
- `summary` (STRING) — 一行调试信息,包含策略名 / despill / 说明

### `ERMBG Classify (preview)`
不跑抠图模型,只跑 router。秒回"我会用什么策略",用来在 ComfyUI 里做条件分支或 debug。

输出 `bg_type` / `image_type` / 完整 JSON。

### `ERMBG PyMatting Known-B`
实验性的已知纯色背景 PyMatting 节点,用于硬边界/抗锯齿图形的 A/B 测试。它不会跑 BiRefNet,也不会提交嵌套 Comfy prompt;只在当前 ComfyUI Python 进程里用 PyMatting 解 trimap 的未知边界带,再用已知背景色反解边缘前景色。

输入:
- `image` (IMAGE,必填) — 已经渲染在纯色背景上的图
- `method` — `cf` / `knn` / `lbdm` / `lkm` / `rw` / `sm`,默认 `cf`
- `image_space` — 默认 `linear`,也可切到 `sRGB` 做对比
- `bg_source` — `auto` / `green` / `blue` / `custom`
- `bg_color` — `custom` 时使用的 `R,G,B`
- `bg_threshold` / `fg_threshold` — 生成 known-B trimap 的背景/前景距离阈值
- `boundary_band_px` — trimap 未知边界带宽度
- `cg_maxiter` / `cg_rtol` — 迭代求解器参数

输出:
- `foreground` (IMAGE) — 已按 known-B 反污染的前景 RGB
- `alpha` (MASK) — PyMatting α
- `summary` (STRING) — 方法、背景色、未知像素数
- `rgba_rgb` (IMAGE) — 与 `alpha` 配套的 RGB
- `trimap` (IMAGE) — trimap 预览

## 安装

依赖要先装到 ComfyUI 的 Python 环境里:

```bash
# 在 ComfyUI 的 venv 中
pip install ermbg

# 或本地 dev 安装
cd /path/to/ERMBG
pip install -e ".[torch]"
```

然后把这个目录链到 ComfyUI:

```bash
# 软链(开发期推荐,改代码立即生效)
ln -s /path/to/ERMBG/comfy_nodes ~/ComfyUI/custom_nodes/ermbg

# 或拷贝
cp -r comfy_nodes ~/ComfyUI/custom_nodes/ermbg
```

重启 ComfyUI,在节点面板 → ERMBG 分类下应该能看到两个节点。

## 工作流示例

最简单的"出图 → 抠图"链:

```
KSampler → VAEDecode → ERMBG AutoMatte → Save Image (RGBA)
                                       ↘ MASK output → Mask preview
```

带预览(看 router 选了啥):

```
LoadImage → ERMBG Classify → ShowText (json)
          ↘ ERMBG AutoMatte → ...
```

如果你的工作流前面已经有别的分割节点(比如 SAM),把 MASK 接到 AutoMatte 的 `source_mask`。AutoMatte 会自动评估这个 mask 的卫生度:边缘干净就直接用,有 halo / 二值化 / 旧背景泄漏就重抠。

如果本地证据无法稳定判断主体归属,可以把 CLIPSeg / Florence / SAM 生成的粗 ownership mask 接到 `subject_mask`。这条输入只回答"哪些区域属于主体",ERMBG 仍会用 keyer、外轮廓保护和 QA 来决定实际修复范围。

服务器忙时可以先只渲染工作流 JSON,不提交队列:

```bash
.venv/bin/python scripts/05_comfy_subject_mask_workflow.py \
  --input input.png \
  --prompt "the complete object to keep" \
  --out out/comfy_workflows/subject_mask_ermbg.json \
  --filename-prefix subject_mask_ermbg
```

等 ComfyUI 空闲后加 `--submit` 即可上传、排队、等待完成并下载 foreground / alpha / subject mask 三个调试输出;如果只想排队不等待,再加 `--no-wait`。
