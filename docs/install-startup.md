# ERMBG 安装与启动

默认服务流程使用 Direct Worker。Direct Worker 既可以和 Web 跑在同一台机器上,
也可以跑在远端服务器上;Web 从配置中读取它的 HTTP URL。

## 运行时形态

```text
浏览器
  -> 本地 ERMBG Web UI / API :7860
  -> 配置的 ERMBG Direct Worker URL
  -> 共享 router + execution profiles
  -> PyMatting Known-B / CorridorKey runner / passthrough
```

ComfyUI 是可选适配器。当某个 Comfy 图需要 ERMBG 节点时,再安装 `comfy_nodes/`。

## 安装

```powershell
cd <ermbg-root>
uv venv .venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -e ".[web,dev,torch]"
```

`torch` extra 支持 Direct Worker 的 CorridorKey 路径。只跑 PyMatting 的测试
可以不装它。游戏素材服务安装应包含 `torch`。

## 配置

共享默认配置位于 `ermbg.config.json`。机器相关覆盖写入 gitignored
`ermbg.local.json`; 两者结构相同,本机文件只需要写要覆盖的字段:

```json
{
  "services": {
    "direct_worker_url": "...",
    "direct_worker_urls": {
      "local": "http://127.0.0.1:7871",
      "remote": "http://192.168.0.8:7871"
    },
    "comfy_url": "..."
  },
  "web": {
    "auto_backend": "direct-worker",
    "auto_fallback_backend": "pymatting-known-b",
    "enable_comfy": false
  }
}
```

`services.direct_worker_url` 是 `backend=auto` 和旧的 `direct-worker` 选项使用的
主地址。`services.direct_worker_urls` 是命名地址表;Web 后端下拉框会显示
`direct-worker:<name>` / `direct-corridorkey:<name>` 以及对应 URL。这样本地
和远端 worker 同时配置时,选择项会显式写出地址,不要再靠隐藏覆盖猜当前走哪台。

环境变量 `ERMBG_DIRECT_URL`、`COMFY_URL`、`ERMBG_WEB_AUTO_BACKEND`、
`ERMBG_WEB_AUTO_FALLBACK_BACKEND` 和 `ERMBG_ENABLE_COMFY` 可在单个 shell
会话内覆盖配置。

配置优先级: 环境变量 / `.env` > `ermbg.local.json` > `ermbg.config.json` >
代码默认值。切换机器或工作环境时优先改 `ermbg.local.json`,不要改共享默认配置。
ComfyUI 不是默认运行路径; `services.comfy_url` 没有代码级 fallback。需要 Comfy
路径的机器必须在 `ermbg.local.json`、`.env` 或环境变量里显式配置 `COMFY_URL`。

## 用本地 Direct Worker 启动 Web

同时启动两个服务:

```powershell
.\scripts\start_local.ps1
```

手动命令:

```powershell
Start-Process -WindowStyle Hidden -WorkingDirectory . `
  -FilePath .\.venv\Scripts\python.exe `
  -ArgumentList "-m ermbg.direct_worker_server --host 127.0.0.1 --port 7871"

Start-Process -WindowStyle Hidden -WorkingDirectory . `
  -FilePath .\.venv\Scripts\python.exe `
  -ArgumentList "-m uvicorn ermbg.web:app --host 127.0.0.1 --port 7860"
```

打开:

```text
<web-url>
```

## 用远端 Direct Worker 启动 Web

固定流程:

1. 同步本地源码到远端源码树。
2. 用远端 Direct Worker 专用脚本重启 `7871`。
3. 用本地 Web 连接远端 URL。

```bash
scripts/sync_comfy_ssh.sh --smoke
scripts/restart_direct_worker_ssh.sh --restart
curl -sS "http://192.168.0.8:7871/health"
```

脚本默认远端路径为 `C:/Users/darkv/ermbg_src`,Python 为
`E:/ComfyUI/.venv/Scripts/python.exe`,并通过 Windows 任务计划启动 worker,
避免进程随 SSH 会话结束而退出。

手动启动远端 worker 仅用于排查:

```powershell
cd C:\path\to\ermbg
.\.venv\Scripts\python.exe -m ermbg.direct_worker_server --host 0.0.0.0 --port 7871 --cpu-workers 4
```

启动本地 Web 服务,并指向远端 worker:

```powershell
.\scripts\start_local.ps1 -SkipDirectWorker -DirectUrl <services.direct_worker_url>
```

环境变量形式:

```powershell
$env:ERMBG_DIRECT_URL = "<services.direct_worker_url>"
$env:ERMBG_WEB_AUTO_BACKEND = "direct-worker"
$env:ERMBG_ENABLE_COMFY = "0"
.\.venv\Scripts\python.exe -m uvicorn ermbg.web:app --host 127.0.0.1 --port 7860
```

## 验证

```powershell
curl.exe -sS "<services.direct_worker_url>/health"
curl.exe -sS "<web-url>/api/runtime-capabilities?include_comfy=false&include_object_info=false"
```

能力响应:

- `local.status = ok`
- `direct_worker.status = ok`
- `web.auto_backend = direct-worker`
- `web.enable_comfy = false`
- `comfy.status = disabled`

Web 行为改动或服务重启后,还要做一次真实上传 smoke:

```bash
curl -fsS -X POST http://127.0.0.1:7860/api/matte-candidates \
  -F "file=@samples/corridorkey_semantic/button/button_green_yellow_a_outlined_hard_heavy_shadow/green.png" \
  -F "backend=auto" \
  -F "shadow_enabled=true" \
  -o /tmp/ermbg_web_smoke.json
jq '{backend,strategy,route,execution_backend,server_elapsed_sec}' /tmp/ermbg_web_smoke.json
```

如果本机 `.venv` 出现 `ModuleNotFoundError: No module named 'cv2'`,先修环境,
不要改算法:

```bash
uv pip install --reinstall opencv-python-headless
.venv/bin/python - <<'PY'
import cv2
print(cv2.__version__)
PY
```

## 可选的 Comfy 适配器

Comfy 图支持:

1. 把 ERMBG 安装到 ComfyUI 的 Python 环境。
2. 把 `comfy_nodes/` 复制到 Comfy 的 `custom_nodes/ermbg-comfy`。
3. 重启 Comfy,因为自定义节点是在进程启动时扫描的。
4. 验证 `/object_info` 中包含 `ErmbgRouteMatte`、`ErmbgRouteStrategy`、
   `ErmbgPyMattingKnownB` 和 `ErmbgClassify`。

Web 侧调试显式 Comfy 后端:

```text
ERMBG_ENABLE_COMFY=1
COMFY_URL=<services.comfy_url>
```

正常的 Web/API 使用保持 `ERMBG_ENABLE_COMFY=0`。
