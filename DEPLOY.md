# 部署到 ComfyUI 服务器

把 ERMBG 装到一台跑 ComfyUI 的机器上,让 `ErmbgAutoMatte` / `ErmbgClassify` 出现在节点面板。本文以局域网 Windows 服务器(`darkv@192.168.0.8`,RTX 4090,ComfyUI 0.21.1 desktop 版,uv 管理的 venv)为例,其他平台同理。

## 前提

- ComfyUI 已能跑起来,HTTP API 可访问(默认 `http://<host>:8000`)
- ComfyUI 用的 Python venv 里已经有 torch / transformers / numpy / opencv / scipy / pymatting / loguru / huggingface_hub。**ERMBG 的依赖大多和 ComfyUI 一致,通常无需额外装包**
- 知道 ComfyUI 的:
  - `main.py` 启动命令(查 `Get-CimInstance Win32_Process -Filter ProcessId=<PID>`)
  - `custom_nodes/` 路径(常见:`<base-directory>\custom_nodes`)
  - venv 的 `python.exe` 绝对路径

## 步骤

### 1. 打包仓库(本地)

```bash
cd /path/to/ERMBG
tar --exclude='.venv' --exclude='__pycache__' --exclude='samples/outputs' \
    --exclude='out' --exclude='.git' --exclude='*.egg-info' \
    -czf /tmp/ermbg_deploy.tgz \
    ermbg comfy_nodes pyproject.toml README.md
```

### 2. 上传到服务器

```bash
scp /tmp/ermbg_deploy.tgz darkv@192.168.0.8:/c:/Users/darkv/
ssh darkv@192.168.0.8 'cmd /c "mkdir C:\Users\darkv\ermbg_src && tar -xzf C:\Users\darkv\ermbg_deploy.tgz -C C:\Users\darkv\ermbg_src"'
```

### 3. 装 ermbg 包到 ComfyUI 的 venv

```bash
ssh darkv@192.168.0.8 'cmd /c "E:\ComfyUI\.venv\Scripts\python.exe -m pip install C:\Users\darkv\ermbg_src"'
```

ermbg 包就装在 site-packages 了,任何 Python 程序(包括 custom_nodes 里的节点代码)都能 `from ermbg import matte_image`。

### 4. 把节点目录放进 custom_nodes

**重要**:目录名**不能叫 `ermbg`** — 会跟刚装的 `ermbg` 包同名冲突。改名为 `ermbg-comfy`(或别的):

```bash
ssh darkv@192.168.0.8 'cmd /c "xcopy /E /I /Y C:\Users\darkv\ermbg_src\comfy_nodes E:\ComfyUI\custom_nodes\ermbg-comfy"'
```

### 5. 重启 ComfyUI

ComfyUI 启动时只扫描一次 custom_nodes,需要重启进程才能识别新节点。

如果 desktop 版会自己 respawn backend,直接 kill 主 Python 进程:

```bash
ssh darkv@192.168.0.8 'powershell -Command "Get-CimInstance Win32_Process -Filter ProcessId=<PID> | ForEach-Object { $_.CommandLine }"'  # 确认 PID
ssh darkv@192.168.0.8 'powershell -Command "Stop-Process -Id <PID> -Force"'
```

服务器版可能要手动重启 systemd / PM2 服务。

### 6. 验证节点上线

```bash
curl -s http://192.168.0.8:8000/object_info | \
    python3 -c "import json,sys; d=json.load(sys.stdin); \
        print('ermbg:', [k for k in d if k.lower().startswith('ermbg')])"
# 期望:['ErmbgAutoMatte', 'ErmbgClassify']
```

## 已知坑

- **`/manager/reboot` 不一定起作用**。ComfyUI Manager 桌面版的 reboot 端点在我这台上是 405,实际重启走 `Stop-Process` + 自动 respawn 路径才管用。
- **`ermbg` 名字冲突**。装好包后 site-packages 下有 `ermbg/`,custom_nodes 不能也叫 `ermbg`,否则 ComfyUI 加载器会用相对导入,把 `ermbg` 优先解析到 custom_nodes 那个目录,丢失主包内容。
- **首次跑会下 BiRefNet 权重**(≈1 GB)到 HF 缓存,延迟集中在第一次推理。
- **PIL / numpy 版本** 安装时如果报"无法卸载现有包",是 ComfyUI 还在跑、文件被占用 — 先 kill 再装。

## 升级

代码改动后,重复步骤 1–5。pip install 会覆盖 site-packages 里旧的 ermbg。custom_nodes 里只放 `comfy_nodes/` 的内容,改这里也只需要重启。
