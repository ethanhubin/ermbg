---
name: ermbg-matte
description: Smart matting for AI-generated assets. Routes per image (saturated bg → chromatic key, white/black → luminance key, clean RGBA → passthrough, dirty RGBA → re-matte) and outputs a clean transparent PNG. Use when the user asks to 抠图/去背景/transparent PNG/RGBA/扣图 on an image that came out of an AI generator (Flux, SDXL, Qwen-Edit, GPT-Image, etc.) — especially when the original was generated on green screen / pure-color background. Smarter than basic RMBG: handles small subjects matting nets miss, kills BiRefNet halo on hard-edged graphics, and refuses to pass through dirty RGBA without re-matting.
---

# ERMBG AutoMatte

抠图工具,专为 "AI 生图 → 去背景" 链路设计。系统自己识别背景类型(绿幕 / 白底 / 黑底 / 已透明 RGBA)挑最优策略,无需用户填参数。

## 调用方式

```bash
python3 skills/ermbg-matte/scripts/ermbg_matte.py --image /path/to/input.png
```

会写出 `output.png`(RGBA)、`workflow.json`(可复现的 ComfyUI workflow)、`manifest.json`(策略 / 路由 / 指标),并打印 `OUTPUT /absolute/path/output.png`。

## 选项

```bash
# 强制 despill(默认 auto,router 自己选)
--despill {auto,unmix,chroma_cap,local_borrow,closed_form,none}

# 强开/强关 keyer(默认 auto)
--use-keyer {auto,on,off}

# 重抠时合成底色,默认绿幕
--bg-color "0,200,0"

# 自定义输出目录
--output-dir /path/to/dir
```

## 与 comfyui-rmbg 的区别

| | comfyui-rmbg | ermbg-matte |
|---|---|---|
| 单主体 / 简单 | ✓ | ✓ |
| 多个独立物体(如卡通图三件套)| 漏小物体 | 自动补漏 |
| 白底硬边 logo | BiRefNet 留白晕 | keyer gate 压去 |
| 已抠过的脏 RGBA | 直接 pass | 自动检测并重抠 |
| 路由透明度 | 单一节点 | 输出策略 + 指标 |

ermbg-matte 是首选,只有 ComfyUI 自带 RMBG 已经够用且不需要分析报告时才用 comfyui-rmbg。

## 规则

- 输出归档到 `/Users/ethanhu/.openclaw/media/openclaw-production/images/ermbg/`,每次新建子目录(uuid),不要覆盖。
- 把 `OUTPUT` 路径回报给用户,需要时直接 `<image_url>file://...</image_url>` 让 UI 渲染。
- 不要在这个 skill 里安装模型 / 改 ComfyUI 服务器配置;依赖问题报具体的脚本错误。
- `python3 skills/ermbg-matte/scripts/ermbg_matte.py --help` 看完整选项。
