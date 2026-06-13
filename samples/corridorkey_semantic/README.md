# CorridorKey 完整测试样本 v1

本目录是 phase 1 样本构建之后,CorridorKey / ERMBG 的规范完整测试样本集。

Phase 1 状态: 已完成。当前样本集包含 92 个样本:

- Button: 60 个 case,包含有/无描边/半透明阴影矩阵、白描边按钮、真实玻璃按钮,
  以及 known-B 孔洞回归。
- Icon/effect: 23 个 case,包含硬边界、软边界、半透明图标、粒子、平滑 glow
  和手工加入的 known-B 游戏图标回归。
- Character: 9 个 1024x1024 的 case,每个都刻意组合发丝/毛发细节、硬不透明边、
  半透明材质和/或 glow。

幕布约定:

- 绿幕: RGB(0, 200, 0)
- 蓝幕: RGB(0, 0, 200)

把 `manifest.json` 作为机器可读的入口。每个 case 通过 `green` 或 `blue` key 暴露
其被批准的那个幕布。测试运行器应只统计 case 中存在的幕布,而不是把每个 case 乘以
所有可能的背景。

下一阶段: 在这个确认集上跑完整的识别/matting 评估,按样本族检查失败,然后调优
CorridorKey 的 route 选择和按 route 的参数。
