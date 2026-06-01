# 本地归属（Local Ownership）

本地归属是已知背景抠图所用的确定性证据层。它在生成执行掩码之前,先决定某个
区域归哪种操作所有。

```text
已知背景图片
  -> 本地 matte
  -> 证据区域
  -> 归属打分
  -> 执行掩码仲裁
```

## 角色

| 角色 | 含义 | 执行意图 |
|---|---|---|
| `hole` | 背景 / 透明开口 | 保持 alpha 低 |
| `opaque_subject` | 漏掉的硬主体支持 | 允许受控的 alpha 修复 |
| `subject_soft_layer` | 玻璃、glow、烟雾、抗锯齿、半透明材质 | 保留软 alpha |
| `shadow_like_layer` | 主体支持附近对已知背景的标量变暗 | 保留测得的阴影 |
| `conservative_unknown` | 局部证据不明确 | 保持基础 alpha |

## 信号

归属使用可测量的局部信号:

- alpha 分布;
- 相对已知背景的 OKLab 距离;
- 局部饱和度和色度偏移;
- 标量变暗拟合,`C_linear ~= scale * B_linear`;
- 拓扑: 主体支持、边界接触、外部占比;
- 已有的 risk/debug 区域证据。

规则必须保持基于特征。不要把样本 ID、文件名或一次性坐标写进去。

## 仲裁

区域打分是宽松的;执行掩码更严格:

- 丢弃微小的 soft-layer 斑点;
- 连贯的软材质会压制小的 shadow-like 碎片;
- 仅含阴影的区域保持基础 matte 路径;
- 仅当存在连贯的 `subject_soft_layer` 时,才使用受保护的 soft-layer 重渲染。

这把半透明主体材质与同一已知背景上测得的阴影分离开来。

## 代码

- `ermbg/ownership.py`: 信号测量、角色排序、执行掩码仲裁。
- `ermbg/matting.py`: `subject_material_mask` 执行约束。
- `ermbg/web.py`: 浏览本地归属和 Game Eval batch。

## 测试

```powershell
.\.venv\Scripts\pytest.exe tests\test_ownership.py tests\test_shadow.py tests\test_risk.py
```

产生产物的归属 probe 应写到 `out/` 下一个 batch 目录,并附带 summary JSON。
