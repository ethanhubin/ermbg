# CorridorKey 模块

本文对齐当前 CorridorKey 路径、游戏素材样本和验证方式。

## 文件

- `ermbg/corridorkey_runner.py`
- `ermbg/direct_worker.py`
- `ermbg/router.py`
- `samples/corridorkey_semantic/manifest.json`
- `tests/test_direct_worker.py`
- `tests/test_direct_worker_server.py`

## 职责

CorridorKey 用于复杂绿幕/蓝幕素材，尤其是:

- 软边和 feather；
- glow / particle / mist；
- 透明或半透明 UI；
- 角色发丝、毛发、透明布料；
- 同幕布色材质风险较高的素材。

主线中 CorridorKey 由 route/profile 选择，并由 Direct Worker 执行。Web 不持有
CorridorKey 私有 route 逻辑。

## 输入

CorridorKey direct 路径需要:

- `corridorkey_analysis`;
- `params`;
- 可选 `corridorkey_hint_mask`;
- 可选 `semantic_decision`;
- 可选 user masks。

当 Web 通过 `route_decision` 调用 Direct Worker 时，`corridorkey_analysis` 必须随
Analyze route 一起传递。

## Analyze 候选

CorridorKey 当前可输出 `screen_material_or_translucency` 风险候选:

- `preserve_screen_material`;
- `remove_screen_tint`。

候选 preview 是 hint/overlay，不是最终 RGBA。

## 样本验证

规范样本集:

```text
samples/corridorkey_semantic/manifest.json
samples/corridorkey_semantic/sheets/full_samples_v1_sheet.jpg
```

批量测试某一算法路径时必须固定 execution backend，例如:

```text
--fixed-execution-backend direct-pymatting-known-b
```

不要用“当前 auto route 会到这个 backend”代替固定路线。
