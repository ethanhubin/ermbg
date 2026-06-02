# ERMBG Route 策略

本文档定义 Web、API、Direct Worker、Game Eval 和可选适配器共用的
route/profile/backend 契约。

Web `backend="auto"` 把图片提交给配置的 Direct Worker 服务。worker 运行
`ermbg.router.classify_route()`,选择一个 execution profile,并分发到维护中的
matting 路径。可选的 ComfyUI 节点对自定义图使用同一套 route 契约。

route 选择必须在 matting 开始前完成。执行代码直接消费
`RouteDecision.params.execution_profile` 及相关参数。profile 专属的调参应放在
共享的 route/执行代码中,以保持 Web、Direct Worker 和可选适配器对齐。

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

执行代码必须保留这些决策。在 router 已经选定 execution profile 之后,它不应
再根据本地语义 hint 去重新分类 asset kind。

## Known-B 路径

PyMatting Known-B 路径面向游戏 UI 和稳定的纯背景图形。它的职责是在已知背景
证据之上做像素级修复:

- 检测到背景不均匀时,先把已知背景和低 alpha 背景尾部归一化到测得背景色;
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
Direct Worker 和可选 Comfy 节点都调用同一个 runner,以保持 hint 转换、
颜色保护、模型调用和 debug 元数据对齐。

## Direct Worker 后端

`backend="direct-worker"` 是 Web/API 和 Game Eval 的服务后端。服务 URL 来自
配置中的 `services.direct_worker_url`: 共享默认值在 `ermbg.config.json`,本机
覆盖在 gitignored `ermbg.local.json`,并可用 `ERMBG_DIRECT_URL` 作为环境覆盖。
当本机和远端 worker 同时存在时,把命名地址写入
`services.direct_worker_urls`,例如 `local` 和 `remote`。Web 会把它们渲染成
`direct-worker:local`、`direct-worker:remote`、`direct-corridorkey:local`
等显式选项,选项文本必须包含实际 URL。

Direct Worker 报告两层后端元数据:

- `selected_backend`: router 选择的逻辑 route 后端;
- `debug.direct_worker.execution_backend`: 具体的 direct 执行路径,例如
  `direct-corridorkey` 或 `direct-pymatting-known-b`。

对同一输入,Web auto 和 Direct Worker run 应保持 `parameter_profile` 和
`execution_profile` 稳定。适配器之间预期的输出差异应停留在浮点或 8-bit
rounding 级别。

## 可选的 Comfy 节点

ComfyUI 支持位于 `comfy_nodes/`,用于自定义 Comfy 图。它使用同一套
route/profile 契约,并通过 `services.comfy_url` 或 `COMFY_URL` 配置。Web 默认
配置使用 Direct Worker。Comfy 不是默认运行路径,没有代码级 URL fallback; 使用
Comfy 后端前必须在本机配置或环境变量中显式设置地址。

节点细节和安装步骤见 `comfy_nodes/README.md` 和 `DEPLOY.md`。

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
2. 验证首页包含 `Auto` 和 `direct-worker`。
3. 用 `backend=auto` 向 `/api/matte-candidates` 提交真实样本。
4. 确认 `requested_backend`、`backend`、`debug.auto_route.selected_backend`、
   `debug.auto_route.route`、`execution_profile` 和 `server_elapsed_sec`。
5. 对 Direct Worker 改动,还要用 `backend=direct-worker` 提交一个 CorridorKey
   样本,并确认 `debug.direct_worker.execution_backend`。

算法改动应由机制驱动。测试应尽量用合成覆盖捕捉失败类别,并对用户可见的回归
补充真实样本批量覆盖。
