# ERMBG

基于纯色背景探针图的半透明抠图工程。设计思路见 [clean_transparent_matting_engineering_plan.md](clean_transparent_matting_engineering_plan.md)。

当前进度：**Phase 0 + Phase 1 (探针生成可行性验证)**。Phase 2+ 的解析式 matting 等 Phase 1 报告出来后再排。

## 安装

需要 Python 3.11 或 3.12（PyTorch / diffusers 在 3.13/3.14 上 wheel 不全）。Mac 推荐用 pyenv：

```bash
pyenv install 3.12.5
pyenv local 3.12.5
python -m venv .venv && source .venv/bin/activate

# 基础依赖（不含 torch，可先跑 synthetic 探针 + validator）
pip install -e ".[dev]"

# 真要跑 SDXL inpainting / BiRefNet 才装：
pip install -e ".[torch,dev]"
```

模型权重首次会从 HuggingFace 下载，缓存到 `~/.cache/huggingface`。

## CLI

```bash
ermbg --help
ermbg segment <input>              # 跑粗分割，输出 mask + 粗 trimap
ermbg probe <input> --color white  # 生成单张探针图
ermbg validate --input samples/inputs --out samples/outputs/phase1
```

## 目录结构

```
ermbg/        # 主包
  probe/      # 探针生成（synthetic / SDXL inpaint）
  ...
tests/        # 单元测试
scripts/      # 评测脚本
samples/      # 输入与产物
```

## Phase 1 目标

跑出 5 类物体 × 多种背景色 × 多种生成器的探针图，用 ProbeValidator 度量主体保真度，给出后续 Phase 2 的决策依据。详见 [clean_transparent_matting_engineering_plan.md](clean_transparent_matting_engineering_plan.md) 第 8 节。
