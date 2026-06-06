# Execute / Direct Worker 模块

本文对齐当前 Web Execute 和 Direct Worker 实现。

## 文件

- `ermbg/web.py`
- `ermbg/direct_worker_client.py`
- `ermbg/direct_worker_server.py`
- `ermbg/api.py`
- `tests/test_web.py`
- `tests/test_direct_worker_server.py`
- `tests/test_runtime_capabilities.py`

## Web Execute

主入口:

```text
POST /api/execute-candidate
```

Web 执行前构造 `ExecutionRequest` 摘要:

- `preprocess`;
- `route`;
- `selected_candidate_id`;
- `semantic_decision`;
- `user_mask`;
- `metadata.schema = ermbg.execution_request.summary.v1`。

然后 Web 将选中的 route candidate 转成 Direct Worker 表单字段 `route_decision`。

## Direct Worker

主入口:

```text
POST /matte
```

模式:

1. 无 `route_decision`: 兼容旧调用，读取图片后运行 `classify_route()`。
2. 有 `route_decision`: 直接构造 `RouteDecision`，跳过 `classify_route()`。

Web/API 主线必须使用第二种模式。

## Known-B Explicit Trimap

Analyze 可为 PyMatting Known-B candidate 输出 `execution_role=pymatting_explicit_trimap`
的三态 PNG:

- `0`: sure-BG；
- `128`: unknown；
- `255`: sure-FG。

Web Execute 解码该 PNG，并作为 `pymatting_explicit_trimap` 发送给 Direct Worker。
Known-B executor 收到后直接消费，不重建 base trimap。user keep/remove mask 仍在执行端
应用，remove 覆盖 keep。

如果 explicit trimap 缺失或 route 不是 Known-B，执行端才回退到对应路径自己的 builder。

## Known-B Preprocess 传递

Web/Analyze 已应用背景场归一化时，会传入:

- `pymatting_input_preprocessed_known_b=true`;
- `pymatting_background_normalization={...}`;
- `pymatting_bg_source=custom`;
- `pymatting_bg_color`;
- Known-B thresholds 和 solver params。

Direct Worker executor 必须跳过私有背景归一化。

## 输出

Web payload 和 manifest 应记录:

- `pipeline_mode=execute_candidate`;
- `execution_backend`;
- `execution_server_url`;
- `execution_request`;
- `debug.semantic_execution`;
- `debug.direct_worker`;
- `debug.input_preprocess`;
- artifact manifest path。

显式 Execute request 的真实 HTTP smoke 中，Direct Worker `timings.route_sec` 应为 `0.0`
或接近 0，说明没有重新 classify。
