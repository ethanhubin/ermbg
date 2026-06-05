# Execute / Direct Worker 模块

本文对齐当前 Web Execute 和 Direct Worker 实现。

## 文件

- `ermbg/web.py`
- `ermbg/direct_worker_client.py`
- `ermbg/direct_worker_server.py`
- `tests/test_web.py`
- `tests/test_direct_worker_server.py`
- `tests/test_runtime_capabilities.py`

## Web Execute

主入口:

```text
POST /api/execute-candidate
```

Web 在执行前构造 `ExecutionRequest` 摘要,并从 Analyze contract 中提取 route/profile:

- `preprocess`;
- `route`;
- `selected_candidate_id`;
- `semantic_decision`;
- `user_mask`;
- `metadata.schema = ermbg.execution_request.summary.v1`。

随后 Web 将 Analyze 的 route 显式转成 Direct Worker 表单字段 `route_decision`。

## Direct Worker

主入口:

```text
POST /matte
```

Direct Worker 支持两种模式:

1. 没有 `route_decision`: 兼容旧路径,读取图片后运行 `classify_route()`。
2. 有 `route_decision`: 直接构造 `RouteDecision`,跳过 `classify_route()`。

Web/API 主线应使用第二种模式。

## route_decision

`route_decision` 是 JSON object,当前包含:

- `route`;
- `algorithm`;
- `backend`;
- `asset_kind`;
- `parameter_profile`;
- `execution_profile`;
- `confidence`;
- `reasons`;
- `params`;
- `analysis`;
- 可选 `corridorkey_analysis`。

Direct Worker 将它转换为 `RouteDecision`,再交给 `direct_matte_from_decision()`。

## Known-B Preprocess 传递

Web 根据 Analyze/ExecutionRequest contract 对 Known-B 输入做前置归一化,并传入:

- `pymatting_input_preprocessed_known_b=true`;
- `pymatting_background_normalization={...}`;
- `pymatting_bg_source=custom`;
- `pymatting_bg_color`;
- Known-B thresholds 和 solver params。

Direct Worker executor 收到这些字段后应跳过私有背景归一化。

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

## 验证信号

显式 Execute request 的真实 HTTP smoke 中,Direct Worker `timings.route_sec` 应为 `0.0`
或接近 0,说明它没有重新 classify。
