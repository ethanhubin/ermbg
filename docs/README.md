# ERMBG 文档入口

本文是 `docs/` 的阅读入口。当前文档结构收敛为:

```text
docs/architecture.md       主架构,唯一主线定义
docs/modules/*.md          模块细节,对齐当前实现
docs/archive/*.md          历史计划、旧专题和一次性分析
```

## 先读

1. [`architecture.md`](architecture.md): 当前生产主线和边界。
2. [`modules/pipeline-contracts.md`](modules/pipeline-contracts.md): Preprocess / Analyze / Decide / Execute 数据契约。
3. [`modules/preprocess.md`](modules/preprocess.md): 输入前置加工。
4. [`modules/analyze-decide.md`](modules/analyze-decide.md): route 分析、语义候选、候选预览和用户裁决。
5. [`modules/execute-direct-worker.md`](modules/execute-direct-worker.md): Execute request、Direct Worker 和 Web/API 主执行边界。

## 模块文档

- [`modules/route-profiles.md`](modules/route-profiles.md): `RouteDecision`、algorithm、profile 和执行参数。
- [`modules/known-b.md`](modules/known-b.md): PyMatting Known-B、本地归属、背景归一化和语义约束。
- [`modules/corridorkey.md`](modules/corridorkey.md): CorridorKey 路径、游戏 UI 样本和回归验证。
- [`modules/operations.md`](modules/operations.md): 安装、启动、Direct Worker 配置和 Web smoke。

## 当前主线

```text
input
  -> Preprocess
  -> Analyze
  -> Decide
  -> Execute
  -> Output
```

关键约束:

- Web/API 的 `backend=auto` 默认走 Direct Worker。
- 候选阶段只生成语义候选和轻量预览,不预跑多个完整 matte。
- Execute 消费 Analyze/Decide 生成的显式 request,不重新推断素材类别。
- 粗 mask 是 keep/remove 语义约束,不是最终 alpha。
- 旧 `/api/matte-candidates` 是兼容层,新 Web 主线使用 Analyze/Execute 分离流程。

## 归档规则

`docs/archive/` 只保留历史材料。归档文档可以解释来龙去脉,但不能作为当前主线契约。
新主线变更优先更新 `architecture.md`,再更新相关 `docs/modules/*.md`。
