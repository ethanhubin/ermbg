# ERMBG 文档入口

本文是 `docs/` 的阅读入口。当前活跃文档结构:

```text
docs/architecture.md       当前主架构，唯一主线定义
docs/modules/*.md          当前实现的模块细节
docs/archive/*.md          历史计划、旧专题和一次性分析
```

## 推荐阅读顺序

1. [`architecture.md`](architecture.md): 当前生产主线和边界。
2. [`modules/pipeline-contracts.md`](modules/pipeline-contracts.md): Preprocess / Analyze / Decide / Execute 数据契约。
3. [`modules/known-b.md`](modules/known-b.md): PyMatting Known-B、BG-seed outline trimap、hole candidate。
4. [`modules/analyze-decide.md`](modules/analyze-decide.md): route candidates、semantic candidates、preview assets 和 UI 裁决。
5. [`modules/execute-direct-worker.md`](modules/execute-direct-worker.md): Execute request、explicit trimap、Direct Worker 边界。

## 模块文档

- [`modules/preprocess.md`](modules/preprocess.md): 输入前置加工和背景修复。
- [`modules/route-profiles.md`](modules/route-profiles.md): `RouteDecision`、route candidates、algorithm/profile。
- [`modules/known-b.md`](modules/known-b.md): Known-B 当前算法与调参规则。
- [`modules/corridorkey.md`](modules/corridorkey.md): CorridorKey 路径与验证。
- [`modules/operations.md`](modules/operations.md): 安装、启动、远端同步和 Web smoke。

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
- 候选阶段只生成语义候选和轻量预览，不预跑多个完整 matte。
- Execute 消费 Analyze/Decide 生成的显式 request，不重新推断素材类别。
- Known-B Analyze 生成 explicit trimap；Execute 可直接消费该 trimap。
- Known-B 当前语义候选主要是 enclosed near-B holes；shadow 不是候选。
- 粗 mask 是 keep/remove 语义约束，不是最终 alpha。
- 旧 `/api/matte-candidates` 仅用于旧调用方兼容，不作为当前候选质量验证入口。

## 归档规则

`docs/archive/` 只保留历史材料。归档文档可以解释来龙去脉，但不能作为当前主线契约。
新主线变更优先更新 `architecture.md`，再更新相关 `docs/modules/*.md` 和 `AGENTS.md`。
