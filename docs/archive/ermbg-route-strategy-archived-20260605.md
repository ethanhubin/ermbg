# ERMBG Route 策略

本文档定义 Web、API、Direct Worker 和 Game Eval 共用的
route/profile/algorithm/execution 契约。

Web `backend="auto"` 把图片提交给配置的 Direct Worker 服务。worker 运行
`ermbg.router.classify_route()`,只选择 algorithm、execution profile 和参数,
再由服务配置决定实际 server URL。

route 选择必须在 matting 开始前完成。执行代码直接消费
`RouteDecision.params.execution_profile` 及相关参数。profile 专属的调参应放在
共享的 route/执行代码中,以保持 Web 和 Direct Worker 对齐。

route 之前还有原始素材前置加工阶段。去网格、Known-B 背景场归一化等处理属于
preprocess,应在语义判断之前完成,并把结果作为 route/candidate analyze 的输入。
它们不是 executor 私有调参,也不是用来提前解决主体/孔洞/阴影归属的语义规则。

下一阶段 route 策略还必须在执行前输出语义候选。复杂自动分析只负责推荐默认路径;
当单图证据无法可靠判断主体材质、透明孔洞、阴影或同背景色区域归属时,router/
candidate analyzer 应返回候选和争议区域,由用户或调用方裁决后再执行 matting。
完整契约见 `docs/semantic-candidate-workflow.md`。

## 执行 Profile

`execution_profile` 是图片分析与 matting 之间的公开契约。
`parameter_profile` 是分析元数据,用于解释为何选择该 route。

| Execution profile | 执行路径 | Asset kind | 意图 |
|---|---|---|---|
| `corridorkey-character` | CorridorKey | `character` | 带发丝、毛发、glow、半透明材质和硬边的角色素材。 |
| `corridorkey-transparent-button` | CorridorKey | `button` | 绿/蓝幕上的玻璃或半透明按钮。 |
| `corridorkey-effect-icon` | CorridorKey | `icon` | 作为单一特效层求解的加法或软 alpha 特效图标。 |
| `corridorkey-shaped-icon` | CorridorKey | `icon` | 带 shaped hint 和 key-color 材质保护的图标。 |
| `pymatting-hard-button` | PyMatting Known-B | `button` | 有稳定纯色背景证据的硬边 UI/按钮。 |
| `pymatting-known-bg` | PyMatting Known-B | `known_bg_graphic` | button/icon/character 类之外的稳定已知背景图形。 |
| `pymatting-fallback` | PyMatting fallback | `unknown_fallback` | 未知或不稳定背景的 fallback。 |

当 alpha 质量已经可用时,clean RGBA 输入可以走 passthrough。

## Route 职责

router 负责识别:

- 已有 alpha 的质量以及是否符合 passthrough 条件;
- 背景颜色与稳定性;
- 纯绿/蓝幕证据;
- 硬边 UI、图标、特效、角色、玻璃和半透明材质信号;
- 最终的 execution profile 和后端参数;
- 不稳定或未知背景的 fallback 背景色。
- 是否存在应前置裁决的高争议语义区域,例如 enclosed near-B 主体/孔洞争议、
  暗色主体/阴影争议、同幕布色主体材质/背景残留争议。

执行代码必须保留这些决策。在 router 已经选定 execution profile 之后,它不应
再根据本地语义 hint 去重新分类 asset kind。

当存在高争议区域时,执行代码也不应擅自选择唯一语义。它应消费
`SemanticDecision` 和可选 `UserMaskDecision`,把它们转成 sure foreground、
sure background、protected subject、forced transparent 或 trimap unknown。

## Known-B 路径

PyMatting Known-B 路径面向游戏 UI 和稳定的纯背景图形。它的职责是在已知背景
证据之上做像素级修复:

- 消费 preprocess 产出的 `background_model` 和必要的背景场归一化结果;
- 用局部材质 core 动态确定 sure foreground,避免固定 10px inset;
- sure foreground 保持在主体内部,描边、抗锯齿、孔洞边缘和所有疑似阴影都进入 unknown;
- trimap 自身要检查 PyMatting 证据是否足够。unknown 不是越窄越好:
  对高强度硬阴影按钮,如果 known-B 标量阴影贴着主体,而当前 unknown
  主要由阴影、描边或屏幕色污染组成,缺少足够主体侧颜色证据,trimap
  阶段应局部释放邻近 sure foreground 到 unknown。这个 pass 的目标是
  构造可解且证据平衡的 PyMatting 输入,不是按样本调参,也不是让
  ShadowPatch 在输出阶段弥补上游吃掉的信息。
- 对解出的 alpha 做 unmix 物理一致性修复(见下);
- 以 trimap unknown 作为唯一 ShadowPatch 修复域,对已知背景上经源图证明的标量变暗做同背景重投影重建;
- 为导出稳定前景 RGB。

归属决策使用可测量证据,例如颜色距离、连通分量、局部支持度和同背景重投影
误差。执行参数默认 `pymatting_bg_threshold=3.5`,
`pymatting_fg_threshold=24`, `pymatting_boundary_band_px=2`,
`pymatting_auto_adapt=true`。详细证据模型见 `docs/local-ownership.md`。

Known-B executor 不应再私有运行另一套背景归一化并改变候选阶段看到的证据。
如果执行仍需要背景归一化图像,必须来自 preprocess 阶段,并记录到 manifest/debug。

Known-B 中最典型的前置候选是封闭近背景区域:

- `auto_default`: 使用现有复杂分析推荐的默认解释;
- `protect_near_bg_subject`: 把被主体包围的近背景色区域视为主体材质,适合白毛、
  白衣、眼白、高光等;
- `cut_enclosed_holes`: 把被主体包围的近背景色区域视为透明孔洞,适合镂空 UI、
  文字洞和图标开口。

这些候选应在 trimap 构造前确定。候选阶段只生成 overlay 和执行计划,不调用
PyMatting/CorridorKey 跑多个完整结果。

### Unmix 物理一致性修复

单已知背景的 unmix 是欠定的: 一个像素只有 3 个观测(RGB),却要同时解
前景色 `F`(3 维)和 alpha `a`(1 维)。求解器(PyMatting)在局部平滑约束下
偶尔会给出物理上不可能的解 —— 典型是把**近乎不透明的暗色描边**判成半透明
(例如棕红描边 `(78,58,12)` 被解成 `a≈0.78`)。

此时 `F = (C - (1-a)·B) / a` 在背景主通道上会被解成**负值**: unmix 以为那
22% 的"透明部分"是纯绿幕,从棕色里多扣了绿色分量。负值被 clip 到 0 后,棕红
描边就变成了洋红脏边 —— 这种瑕疵在绿底/棋盘格上几乎看不出,但**放到互补色
(品红)背景上立刻暴露**。

修复判据是物理的,不是分类:

- **脏像素 = clip 前的 `F_raw` 任一通道落在 `[0,1]` 之外**。这等价于
  "该 `(F, a)` 对无法一致地重新合成到任意背景上" —— 而这正是透明抠图必须
  成立的不变量。健康像素天然满足,不受影响。
- **修复**: 借用最近健康邻居的 `F`,把源色 `C` 投影到 `(B, F_neighbor)`
  线段上,反解出自洽的 alpha,再用该 alpha 重新求 `F`。
- **门控**: 只接受会**抬高** alpha 的修复。压低 alpha 会侵蚀已知良好的主体
  种子,而脏像素信号并不证明这一点。这条门控同时保证真半透明像素不被误改
  (它们的真 alpha 必然满足下界,不触发修复)。

验证用双背景判据: 放回原背景色与原图对比看重建误差,再放到互补色背景看
边缘是否干净。注意**重建对比必须在 sRGB 合成域**进行 —— ShadowPatch 的
`shadow_alpha_to_display_alpha` 已为 sRGB viewer 校准过阴影 alpha,在 linear
域对比会得到失配的误差。

代码: `ermbg/api.py` 的 `_repair_known_b_unmix_consistency`,在 unmix clip
之前调用; 统计写入 `report.strategy.extras.unmix_consistency_repair`
(`dirty_pixels` / `repaired_pixels` / `alpha_lift_mean` / `alpha_lift_max`)。

## CorridorKey 路径

CorridorKey 处理复杂的绿幕/蓝幕素材,这些素材受益于影视风格的抠像实践:

- 带发丝、毛发、glow 和软 alpha 的角色;
- 半透明/玻璃按钮;
- 带 key-color 材质保护的 shaped 图标;
- 带加法或烟雾状软边的特效图标。

`ermbg.corridorkey_runner.LocalCorridorKeyClient` 是共享的进程内适配器。
Direct Worker 调用同一个 runner,以保持 hint 转换、颜色保护、模型调用和
debug 元数据对齐。

## Direct Worker 后端

`backend="direct-worker"` 是 Web/API 和 Game Eval 的服务后端。服务 URL 来自
配置中的 `services.direct_worker_url`: 共享默认值在 `ermbg.config.json`,本机
覆盖在 gitignored `ermbg.local.json`,并可用 `ERMBG_DIRECT_URL` 作为环境覆盖。
当同一个 worker 可通过多个 IP 访问时,把 server URL 列表写入
`services.direct_worker_urls`,例如 `loopback` 和 `lan-gpu`,并用 `priority`
控制尝试顺序。Web 下拉框只选择 algorithm,不再渲染 local/remote 或
`direct-worker:<name>` 服务选项。

Direct Worker 报告两层元数据:

- `algorithm`: router 选择的逻辑算法,例如 `corridorkey`、`pymatting_known_b`;
- `debug.direct_worker.execution_backend`: 具体的 direct 执行路径,例如
  `direct-corridorkey` 或 `direct-pymatting-known-b`。
- `execution_server_url` / `server_fallback_chain`: Web 选择的实际 Direct Worker
  server 和 fallback 记录。

对同一输入,Web auto 和 Direct Worker run 应保持 `parameter_profile` 和
`execution_profile` 稳定。适配器之间预期的输出差异应停留在浮点或 8-bit
rounding 级别。

## 验证

使用覆盖各个 profile 的针对性样本:

- hard button -> `pymatting-hard-button`
- blue/green glass button -> `corridorkey-transparent-button`
- effect icon -> `corridorkey-effect-icon`
- shaped icon -> `corridorkey-shaped-icon`
- character -> `corridorkey-character`
- random/unknown background -> `pymatting-fallback`

常用命令:

```bash
.venv/bin/python scripts/run_corridorkey_game_eval.py --backend auto --sample-id I003,I019,I008,B010 --out-dir out/auto_parity_<date>
.venv/bin/python scripts/run_corridorkey_game_eval.py --backend direct-worker --sample-id I003,I019,I008,B010 --out-dir out/direct_parity_<date>
.venv/bin/python scripts/smoke_direct_worker_http.py --base-url <services.direct_worker_url> --sample-id B001,I011
```

在 Web 侧 route 改动之后:

1. 重启本地 Web 服务。
2. 验证首页包含 `Auto Route`、`CorridorKey`、`PyMatting Known-B`。
3. 用真实样本调用 `/api/analyze-candidates`,确认 Analyze 返回 route/profile、
   候选和争议 metadata,且不执行完整 matte。
4. 对无需用户裁决或已选择默认候选的样本调用 `/api/execute-candidate`,确认
   Direct Worker 只消费最终决策并执行一次。
5. 用 `backend=auto` 向兼容层 `/api/matte-candidates` 提交真实样本。
6. 确认 `requested_backend`、`backend`、`debug.auto_route.algorithm`、
   `debug.auto_route.route`、`execution_backend`、`execution_server_url`、
   `execution_profile` 和 `server_elapsed_sec`。
7. 对 Direct Worker 改动,还要用 `backend=direct-worker` 提交一个 CorridorKey
   样本,并确认 `debug.direct_worker.execution_backend`。

算法改动应由机制驱动。测试应尽量用合成覆盖捕捉失败类别,并对用户可见的回归
补充真实样本批量覆盖。
