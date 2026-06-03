# CorridorKey 语义路径

## 当前状态

Phase 1 样本构建已完成。规范的完整测试集为:

```text
samples/corridorkey_semantic/manifest.json
samples/corridorkey_semantic/sheets/full_samples_v1_sheet.jpg
```

已批准的样本集包含 85 个样本:

| 类别 | 数量 | 重点 |
|---|---:|---|
| Button | 56 | 有/无描边/半透明按钮边界、硬/软阴影强度、白描边按钮、真实玻璃按钮、known-B 孔洞回归 |
| Icon / effect | 20 | 硬边界、软边界、半透明图标、粒子特效、平滑 glow |
| Character | 9 | 1024x1024 复合样本,组合发丝/毛发、硬不透明边、半透明材质和 glow |

幕布约定:

- green: RGB(0, 200, 0)
- blue: RGB(0, 0, 200)

每个 case 有一个已批准的幕布输入。评估代码应从 manifest 中统计选中的样本数,
而不是把每个 case 乘以所有可能的背景。

## 基线: 2026-05-31

当前的 B016-B030 蓝幕按钮块现在使用蓝幕上的绿色主体按钮,而非蓝幕上的黄色
按钮。这让样本集与预期的幕布角色划分保持一致: 黄色/橙色按钮在绿幕上已经可解,
而蓝幕覆盖应聚焦于绿幕无法干净分离的绿色主体材质。

变更的样本块:

| 样本 | 幕布 | 主体 | 重点 |
|---|---|---|---|
| B016-B020 | 蓝 | 绿色描边按钮 | 不透明硬边 UI,含描边/无阴影/硬阴影/软阴影 |
| B021-B025 | 蓝 | 绿色无描边按钮 | 不透明硬边 UI,无描边/无阴影/硬阴影/软阴影 |
| B026-B030 | 蓝 | 半透明绿色按钮 | 蓝幕上的半透明 UI 材质 |

B016-B030 仅启用绿色主体蓝幕块。除非代表一个新的当前失败类别,否则不要往
manifest 里添加其他蓝幕配色研究。

最新的完整 RouteMatte 基线:

```text
out/auto_routematte_routefix_20260531/summary.json
out/auto_routematte_routefix_20260531/timing_report.md
```

结果: 85/85 全部成功完成,使用 Web/API `backend=auto`,该基线运行提交远端
`ErmbgRouteMatte` 节点（先于当前 Direct Worker 默认契约,作为历史记录保留）。
Auto 不再调用 RMBG fallback;未知或不稳定背景作为 `pymatting_fallback` 路由到
PyMatting Known-B。当前主线使用 Direct Worker,该段属于历史 RouteMatte 基线记录。

最新完整 B/I/C run 的 route 分布:

| Route | Algorithm | Execution backend | 数量 |
|---|---|---|---:|
| `pymatting_known_b` | `pymatting_known_b` | `direct-pymatting-known-b` | 37 |
| `corridorkey` | `corridorkey` | `direct-corridorkey` | 48 |

类别/后端拆分:

| 类别 | `pymatting_known_b` | `corridorkey` |
|---|---:|---:|
| Button | 37 | 19 |
| Icon / effect | 0 | 20 |
| Character | 0 | 9 |

最新的 execution-profile 验证:

```text
out/verify_route_profiles_character_glass_icon_20260531/summary.json
```

这次针对性的运行验证 router 现在会在执行前选定最终 `execution_profile`,
因此 CorridorKey 的参数选择不会在玻璃按钮、角色和图标之间相互渗透。

| 样本族 | Execution profile | 后端 |
|---|---|---|
| B046-B049 真实玻璃按钮 | `corridorkey-transparent-button` | `direct-corridorkey` |
| I011-I012 软/特效图标 | `corridorkey-effect-icon` | `direct-corridorkey` |
| I019-I020 shaped glow 图标 | `corridorkey-shaped-icon` | `direct-corridorkey` |
| C001-C009 角色 | `corridorkey-character` | `direct-corridorkey` |

Profile 规则:

- `corridorkey-transparent-button`: 全幅玻璃控制,强制关闭颜色保护,
  profile 专属 mask prior。
- `corridorkey-character`: 全幅角色控制,强制关闭颜色保护,角色专属 mask prior。
- `corridorkey-effect-icon`: 针对加法/软 alpha 图标材质的全幅特效控制。
- `corridorkey-shaped-icon`: shaped 前景 hint,包含可能仍需启用保护的
  key-color 材质图标。
- `pymatting-hard-button`: 确定性 known-B 硬边 UI/按钮 route。

2026-05-31 完整 run 的耗时:

| 范围 | 数量 | 均值 | 中位 | P95 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| 整体 client elapsed | 85 | 1.073s | 0.936s | 3.961s | 0.190s | 5.172s |
| PyMatting Known-B client elapsed | 37 | 0.355s | 0.266s | 0.846s | 0.190s | 1.085s |
| CorridorKey client elapsed | 48 | 1.626s | 1.025s | 4.569s | 0.915s | 5.172s |
| Button client elapsed | 56 | 0.585s | 0.433s | 1.196s | 0.190s | 1.355s |
| Icon client elapsed | 20 | 1.077s | 1.021s | 1.405s | 0.989s | 1.432s |
| Character client elapsed | 9 | 4.097s | 3.985s | 4.949s | 3.173s | 5.172s |

这次 route-fix run 把硬软阴影按钮 B020/B025、白描边接触阴影按钮 B053/B054 和
known-B 孔洞按钮 B055/B056 移回 PyMatting。最慢的仍是角色 CorridorKey run,
以 C005 领先,client 5.172s / node 3.048s。`alpha > 128` 覆盖率最低的是 I010,
其次是 B056、I020、I019 和若干重阴影/软阴影按钮 case。B056 现在已被策略正确
路由,但仍需在 PyMatting 路径上做质量调优。该覆盖率指标对半透明/glow 样本仍
只是分诊信号,而非 ground truth。

## 阶段计划

### Phase 1: 样本覆盖

状态: 已完成。

目标是在调优 route 识别之前,先构建一个真实的游戏素材样本集。最终样本集刻意
通过看起来真实的素材来覆盖边界和 alpha 机制,而非抽象占位图。

接受范围:

- 按钮是受控的 UI 几何: 有/无边框、阴影强度、半透明材质,以及模型生成的
  真实玻璃。
- 图标和特效归为一类,因为它们的关键抠图压力是复杂边界加内部颜色/软 alpha。
- 角色是复合样本,因为发丝、半透明材质和硬边天然会同时出现。

### Phase 2: 识别与 Route 审计

状态: route profile 契约已建立;继续审计各样本族,排查识别器盲点和路径专属
质量问题。

下一步是把完整的确认样本集跑一遍 Web/Game Eval 路径,并按样本族检查失败。
这一阶段应回答: 每个样本该使用哪个 route 或候选集?

输入:

- `samples/corridorkey_semantic/manifest.json`
- Web eval 页面: `<web-url>/eval/game`
- CLI eval:

```bash
.venv/bin/python scripts/run_corridorkey_game_eval.py \
  --backend auto \
  --fixed-execution-backend direct-corridorkey
```

审计输出应写到 `out/` 下并包含 summary JSON。不要用参数调优来掩盖一个错误的
route 决策。缺失或低置信度的 route 应记录为识别器盲点。

### Phase 3: 路径专属参数调优

状态: 推迟到 Phase 2 可用之后。

只有在 route 选择足够准确之后,才应按 execution profile 调优 CorridorKey 参数:
颜色保护、despill、refiner 强度、despeckle、前景恢复,以及阴影/软层处理。不要
以会让 `corridorkey-character`、`corridorkey-transparent-button`、
`corridorkey-effect-icon` 或 `corridorkey-shaped-icon` 相互影响的方式去调
通用 CorridorKey 设置。

## 待审计的 Route 压力

| 压力 | 代表样本族 |
|---|---|
| 硬不透明 UI 边界 | button A/B 无阴影和硬阴影行 |
| 自有接触阴影 | button 硬/软阴影行 |
| 半透明 UI 玻璃 | button C 行和真实玻璃按钮 |
| 硬图标边界 | icon A 组 |
| 软/碎片化图标边界 | icon B 组 |
| 半透明图标材质 | icon C 组 |
| 加法或软 alpha 特效 | icon D 组 |
| 混合角色边界 | character 复合组 |

## 活跃集边界

历史生成的样本集不是活跃输入。`docs/archive/` 中的历史分析可能仍提到已退役的
样本 ID,但活跃开发和 Web/Game Eval 应使用
`samples/corridorkey_semantic/manifest.json` 中的 B/I/C 样本 ID。
