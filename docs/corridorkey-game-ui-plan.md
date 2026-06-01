# 游戏 UI 工作流计划

这是游戏 UI 素材的当前开发计划。主线已转向由 ERMBG 拥有 routing、并按 profile
做执行: ERMBG 决定最终路径和参数,然后由执行后端运行 PyMatting Known-B、
CorridorKey、passthrough 或 PyMatting fallback。生产环境的默认执行后端是
Direct Worker;远端 ComfyUI `ErmbgRouteMatte` 节点是可选的执行路径。

## 概要

- 游戏 UI 素材先经过 ERMBG strategy 路由,并在 matting 前产出最终的
  `execution_profile`。
- 从活跃 route 中移除旧的 `comfy-ermbg`/AutoMatte 全程 matting 后端。
- 未知背景路由到 PyMatting fallback;auto 不再调用 RMBG。
- route 分析、参数选择、CorridorKey/PyMatting 执行、ShadowPatch 和元数据生成,
  都在执行后端内完成（默认 Direct Worker,可选远端 `ErmbgRouteMatte` 节点）。
- 对绿/蓝已知幕布的图标、角色、玻璃和半透明按钮 profile 使用 CorridorKey。
- 对确定性硬边按钮以及未知/不稳定 fallback 使用 PyMatting Known-B。

## 主线架构

```text
上传图片
  -> ERMBG route（Direct Worker，默认；或远端 ErmbgRouteMatte，可选）
      -> 幕布/颜色分析
      -> 最终 execution_profile + 参数
      -> PyMatting Known-B、CorridorKey、passthrough 或 PyMatting fallback
      -> ShadowPatch / QA / route 元数据
      -> RGBA 游戏 UI 素材
```

调用端（Web/API/CLI）只负责上传图片、提交请求、轮询并下载结果图和元数据。
对生产 auto 路径,它不应重复 route 分析,也不应做 matting 后的修复。

## 幕布与颜色分析

route 分析器返回:

- `screen_mode`: `green`、`blue` 或 `unknown`。
- `background_color`: 测得的 sRGB key 色。
- `background_confidence`: 该图片是已知幕布素材的置信度。
- `purity_sigma`: 背景在可信区域上的稳定程度。
- `subject_key_color_risk`: 主体材质是否接近 key 色。
- `execution_profile`: 最终 execution profile,例如
  `corridorkey-transparent-button`、`corridorkey-character` 或
  `pymatting-hard-button`。
- `recommended_settings`: CorridorKey/PyMatting 的 gamma、阈值、despill、
  refiner、despeckle、hint、颜色保护和 ShadowPatch 设置。

分析应使用可观测信号,而非样本专属规则:

- 用可信角和边界带作为背景候选;
- 用 OKLab 距离比较绿幕和蓝幕假设;
- 用背景方差/纯度决定 auto 模式是否安全;
- 用主体/key 色重叠风险来避免抹掉绿色或蓝色主体细节;
- 用连通分量尺度来避免 despeckle 移除小的 UI 装饰。

## 执行 Profile

execution profile 是生产契约。CorridorKey 语义 profile 可以参与 routing,
但在 router 做出决策之后,执行不得重新推断 asset family。

| Execution profile | 路径 | 说明 |
|---|---|---|
| `corridorkey-character` | CorridorKey | 全幅角色控制,颜色保护关闭 |
| `corridorkey-transparent-button` | CorridorKey | 全幅玻璃控制,颜色保护关闭 |
| `corridorkey-effect-icon` | CorridorKey | 针对加法或软 alpha 图标的全幅特效控制 |
| `corridorkey-shaped-icon` | CorridorKey | 针对仍需保护的图标材质的 shaped hint |
| `pymatting-hard-button` | PyMatting Known-B | 确定性硬边 UI 和按钮族 |
| `pymatting-known-bg` | PyMatting Known-B | 稳定已知背景图形 fallback |
| `pymatting-fallback` | PyMatting Known-B | 未知或不稳定背景 fallback |

## 参数自适应

推荐默认值:

- 纯绿幕、主体 key 色风险低: 标准 CorridorKey 设置。
- 主体含绿/蓝类材质: 降低激进的 despill/refiner 行为,更多依赖颜色保护。
- 小图标、卡槽、闪光或细 UI 装饰: 降低 `despeckle_size` 或关闭自动 despeckle。
- 玻璃、glow、透明或软渐变: 保留软 hint,避免硬归属掩码。

报告应同时记录选中的值和选择原因,以便 Web 结果和批量 summary 可调试。

## ShadowPatch 层

`ShadowPatch` 是 CorridorKey 游戏 UI 路径所采用的阴影策略。它不是 ERMBG
fallback,也不编辑 CorridorKey 主体层。它只在已经路由到 CorridorKey 的
绿/蓝已知幕布素材上运行;未知背景跳过该路径,转入 PyMatting fallback。
层叠结构为:

```text
shadow layer     由已知背景标量变暗测得
subject layer    CorridorKey 的 RGBA/alpha,保持为硬边的所有者
```

导出的 `rgba.png` 是把 shadow layer 合成到 CorridorKey 主体层之下的扁平化
结果。Debug 输出保持各层分离:

- `corridorkey_subject_rgba.png`
- `corridorkey_subject_alpha.png`
- `shadow_layer.png`
- `shadow.png`
- `shadow_physical.png`

触发条件刻意保守:

- 先从 `C_linear ~= scale * B_linear` 检测连贯的已知背景阴影候选;
- 要求高置信度阴影证据: 足够的可见支持、被接受的连通分量,以及非平凡的
  测得显示不透明度;
- 要求 CorridorKey 尚未把同一阴影区域保留为 alpha。如果 CorridorKey alpha
  已经与测得的阴影支持相当,则跳过该 patch,以避免双重变暗。

一旦触发通过,提取范围会刻意比通用阴影路径更广。目的是覆盖整条软尾和接触
区域;与主体的任何重叠都是无害的,因为最终合成会把未改动的 CorridorKey 主体
放在阴影层之上。

不要用颜色保护或 hint 掩码来恢复阴影。颜色保护用于保护接近 key 色的主体
材质,很容易把阴影误分类,而 hint 掩码会引入主体-阴影接触伪影。ShadowPatch
应保持为一个测量驱动的已知背景后处理,并在 `report["shadow"]["patch_gate"]`
中带显式的 debug 指标。

## 蓝幕支持

不要把蓝幕支持仅仅当作修改 `bg_color`。当前的 `comfy-corridorkey` 包装层现在
会通过 Comfy workflow 传入显式的 `screen_mode`,蓝幕样本也是活跃完整 eval 的
一部分,但蓝幕语义应限定在绿幕无法覆盖的问题上。

B016-B030 蓝色按钮块在 2026-05-31 从黄色按钮改为蓝幕上的绿色按钮。理由:
黄色/橙色 UI 可以在绿幕上评估和修复;蓝幕应补充绿色主体材质的覆盖,而这正是
绿幕无法干净分离的样本族。

黄底蓝幕的研究仍可作为对 CorridorKey 局限的诊断: 在蓝幕上的无描边黄色按钮上,
CorridorKey 可能把蓝色背景变暗分解为脏黄前景加部分 alpha。这是模型分解的弱点,
而非 ShadowPatch 的核心失败,这些样本不再是活跃的 B016-B030 目标。

历史的 direct CorridorKey 蓝/绿基线:

```text
out/corridorkey_full_blue_green_baseline_20260531/summary.json
```

结果: 83/83 全部成功完成。这早于最终的 85 样本 RouteMatte auto 契约,但作为
direct CorridorKey 参考仍然有用。B016-B030 全部作为蓝幕绿色按钮样本成功运行。

最新的完整 RouteMatte 基线:

```text
out/auto_routematte_routefix_20260531/summary.json
out/auto_routematte_routefix_20260531/timing_report.md
```

结果: 85/85 全部通过 Web/API `backend=auto` 成功完成,该基线运行提交远端
`ErmbgRouteMatte` 节点。活跃集为 56 个按钮、20 个 icon/effect 样本和 9 个角色
样本。route 分布为 37 个 PyMatting Known-B case 和 48 个 CorridorKey case。

针对性的 execution-profile 验证:

```text
out/verify_route_profiles_character_glass_icon_20260531/summary.json
```

它验证 B046-B049 为 `corridorkey-transparent-button`、I011-I012 为
`corridorkey-effect-icon`、I019-I020 为 `corridorkey-shaped-icon`、
C001-C009 为 `corridorkey-character`。

## Web UI 与 Debug 控件

Web UI 应呈现 route 决策,而不是强迫用户去推理原始后端:

- 游戏 UI 工作的默认后端: `auto`。
- 显示 `requested_backend`、选中后端、route、asset kind、`execution_profile`、
  parameter profile、测得背景、置信度、server elapsed time 和 route 原因。
- 手动/debug 控件仍可指向 `comfy-corridorkey` 或 `comfy-pymatting-known-b`,
  但生产质量审计应从 `backend=auto` 开始。
- 掩码编辑仍是 debug/操作员辅助。它应提供粗略的 hint 或保护信号,而不是直接
  替换最终的细节 alpha。

## 测试与验证

离线测试:

- 绿、蓝和未知背景分类;
- 主体 key 色风险会改变推荐参数;
- 默认 despeckle 设置不会移除小 UI 组件;
- 蓝幕元数据绝不报告仅绿幕的策略名;
- mask 输入校验 shape、空 mask、满 mask 和编辑过的 mask。

批量测试:

- 已有的游戏 UI 绿幕样本;
- 当前 manifest 中已批准的绿/蓝幕样本;
- 接近绿/蓝的主体材质;
- 玻璃、glow、透明渐变、细描边和小装饰。
- 对全部 game-eval 样本做 ShadowPatch 命中扫描;检查 `shadowpatch_hits.json`
  和每个应用 case 的最终 Web 结果。

远端/Web 验证:

- 通过 ComfyUI 跑 direct `ErmbgRouteMatte` smoke;
- 对 `direct-worker`,在使用其耗时数据前,先在同一 manifest 子集上验证它与
  `backend=auto` 的 parity。`selected_backend`、`parameter_profile`、
  `execution_profile` 和 hint source 应一致;大幅的 alpha/RGBA 差异意味着执行层
  发生了分叉,必须在共享代码中修复,而不是作为独立后端去调参。
- 当某个 profile 专属后端被改动时,跑针对性的 direct `comfy-corridorkey` 和
  `comfy-pymatting-known-b` smoke;
- 通过 `127.0.0.1:7860` 跑真实 HTTP `/api/matte-candidates` smoke;
- 把批量 summary 保存到 `out/` 下,含选中后端、route、`execution_profile`、
  设置、耗时和质量指标。

## 与既有 ERMBG 工作的关系

之前的纯背景/本地归属工作作为 fallback 和 QA 基础设施仍然有用,但它不再是
游戏 UI 素材的主要细节 matting 路线图。ERMBG 应聚焦于围绕 CorridorKey 和
PyMatting 的编排层: 输入分析、profile 选择、参数自适应、mask hint、ShadowPatch、
诊断、批量评估和 Web 控件。未知背景 fallback 是带配置 fallback 背景的
PyMatting Known-B,而不是 RMBG。

Direct Worker 路径是这同一路线图的生产执行后端（绕过队列）。它可能把
`direct-corridorkey` 或 `direct-pymatting-known-b` 报告为执行后端,但它绝不能
分叉 CorridorKey 行为。所有进程内 CorridorKey 执行都属于
`ermbg.corridorkey_runner.LocalCorridorKeyClient`,它被 Comfy 自定义节点包装层
和 Direct Worker 共同使用。
