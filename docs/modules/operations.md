# 运行与验证模块

本文对齐当前安装、启动、Direct Worker 配置和 Web smoke 流程。

## 安装

```powershell
cd <ermbg-root>
uv venv .venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -e ".[web,dev,torch]"
```

Windows 上使用:

```powershell
.venv\Scripts\python.exe
.venv\Scripts\pytest.exe
```

## 配置

默认配置:

- `ermbg.config.json`

本机覆盖:

- `ermbg.local.json`

关键字段:

- `services.direct_worker_url`;
- `services.direct_worker_urls`;
- `web.auto_backend`;
- `web.auto_fallback_backend`。

临时覆盖:

- `ERMBG_DIRECT_URL`;
- `ERMBG_WEB_AUTO_BACKEND`;
- `ERMBG_WEB_AUTO_FALLBACK_BACKEND`。

## 本地启动

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_local.ps1 -DirectUrl http://127.0.0.1:7871
```

期望进程:

- `python -m ermbg.direct_worker_server --host 127.0.0.1 --port 7871`;
- `python -m uvicorn ermbg.web:app --host 127.0.0.1 --port 7860`。

## 必跑测试

涉及 Web 或运行时改动时至少运行:

```powershell
.venv\Scripts\pytest.exe tests\test_web.py tests\test_runtime_capabilities.py
```

涉及 Direct Worker request 或 executor 边界时补充:

```powershell
.venv\Scripts\pytest.exe tests\test_direct_worker_server.py
```

## Web smoke

1. 确认 `7860` 由 `uvicorn ermbg.web:app` 持有。
2. 确认首页包含改动标记,例如 `Auto Route`、`Overlay`、`Trimap`、`Hint`。
3. 请求 `/api/runtime-capabilities?include_comfy=false&include_object_info=false`。
4. 用真实图片调用 `/api/analyze-candidates`。
5. 用 Analyze payload 调用 `/api/execute-candidate`。
6. 检查 `execution_backend`、`execution_server_url`、`server_elapsed_sec`。
7. 对显式 Execute request,检查 Direct Worker `timings.route_sec` 为 `0.0` 或接近 0。

## 产物规则

生成的 eval/debug 产物放在 `out/` 下自包含目录中,并写机器可读 `summary.json`。
批量测试产物使用标准 `ermbg.run.v1` manifest。
