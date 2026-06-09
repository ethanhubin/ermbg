# CorridorKey 模块

本文只描述当前主线实现。历史实验策略不再属于 CorridorKey 当前契约。

## 当前职责

CorridorKey 用于复杂绿幕/蓝幕素材，尤其是角色、发丝、毛发、透明布料、
玻璃/半透明按钮、screen-tinted effect icon 等 Known Screen 上的复杂边界。

当前主线由 Analyze/Decide 选择 route/profile，由 Direct Worker 执行：

- Web/API 的 `backend=auto` 默认走 Direct Worker。
- Route 只决定 algorithm/profile/params，不描述 server URL。
- Execute 消费显式 request，不重新推断素材类别。
- CorridorKey Direct Worker 返回模型原始 alpha/foreground/rgba，不做输出后硬修补。

## 当前文件

- `ermbg/corridorkey.py`: 绿幕/蓝幕资产分析和推荐 profile。
- `ermbg/router.py`: 决定是否使用 CorridorKey，以及 execution profile。
- `ermbg/analyze.py`: 暴露 CorridorKey 常量 hint 强度候选。
- `ermbg/corridorkey_hint.py`: CorridorKey 常量 hint 契约。
- `ermbg/corridorkey_runner.py`: 本地加载 CorridorKey node/processor 并执行。
- `ermbg/direct_worker.py`: Direct Worker CorridorKey 执行入口。
- `scripts/probe_corridorkey_hint_strengths.py`: 常量 hint 强度探针。
- `tests/test_corridorkey_hint.py`: 常量 hint 和 runner 行为测试。

## Hint 契约

CorridorKey 当前只使用全帧常量 hint。默认值是 `0.32`。

Analyze 为同一个 CorridorKey route 暴露这些候选：

- `0.00`: 诊断/极低 prior；
- `0.16`: 低强度 prior；
- `0.32`: 默认候选；
- `0.50`: 中高强度 prior；
- `0.70`: 当前保守上限候选。

候选 decision 使用：

```json
{
  "policy": "corridorkey_constant_hint",
  "corridorkey_hint_value": 0.32
}
```

Execute 读取 `semantic_decision.corridorkey_hint_value`，生成同尺寸全帧灰度
`corridorkey_hint_mask` 传给 Direct Worker。若没有语义值也没有显式上传 hint，
Direct Worker 使用 `corridorkey_full_frame_prior_value()` 的默认 `0.32`。

Web 左侧候选预览是红色半透明蒙层，用来表达候选强度；实际传给模型的是全帧
灰度常量图，不是 feature mask。

## 可调入口

需要改变 CorridorKey 结果时，只通过 Analyze 候选的 `corridorkey_hint_value`、
route/profile 参数或显式上传的 `corridorkey_hint_mask` 进入模型求解过程。
CorridorKey 当前不使用输出后 alpha 修补来改变结果。

## 蓝幕处理

蓝幕 route/profile 已接入 Direct Worker。若当前加载的 CorridorKey node/settings
支持 `screen_color`，runner 直接传入 `blue`。若不支持，runner 对输入做 G/B
通道互换，把蓝幕适配成模型可处理的绿幕，并在输出 foreground 时换回。
debug 中通过 `settings.blue_screen_adaptation` 记录是否发生了该适配。

## 诊断输出

Direct Worker CorridorKey 的标准输出是：

- `rgba`: 最终透明 PNG；
- `alpha`: 模型 alpha；
- `foreground`: 模型 foreground；
- `hint`: 实际送入 CorridorKey 的 hint；
- `raw_alpha`: runner 原始 alpha。

`trimap` 不是 CorridorKey 的强制输出。batch/eval manifest 只登记实际产出的
诊断图，不把其它 backend 的诊断项强行套到 CorridorKey 上。

## 验证方式

常量 hint 强度机制验证使用：

```powershell
.venv\Scripts\python.exe scripts\probe_corridorkey_hint_strengths.py --run-remote
```

game eval 必须固定 execution backend，例如：

```powershell
.venv\Scripts\python.exe scripts\run_corridorkey_game_eval.py --backend direct-worker --use-analyze-candidates
```

若要做某个固定后端的批量性能/质量验证，必须显式使用固定 execution backend，
不允许只依赖 auto route 当前会选到该后端。
