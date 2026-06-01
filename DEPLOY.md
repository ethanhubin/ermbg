# 可选的 ComfyUI 节点部署

ERMBG Web/API 默认使用 Direct Worker。本文档介绍如何把可选的 ERMBG 节点
安装到 ComfyUI 环境中。

共享服务端点在 `ermbg.config.json` 中配置; 每台机器自己的覆盖写在
gitignored `ermbg.local.json`:

```json
{
  "services": {
    "comfy_url": "..."
  }
}
```

环境变量 `COMFY_URL` 可在单个 shell 会话内覆盖 `services.comfy_url`。优先级为
环境变量 / `.env` > `ermbg.local.json` > `ermbg.config.json` > 代码默认值。
ComfyUI 不是默认运行路径,因此 `services.comfy_url` 没有代码级 fallback; 使用
Comfy 节点或 Comfy 后端前必须显式配置地址。

## 节点

可选的 Comfy 包提供:

- `ERMBG Route Matte`
- `ERMBG Route Strategy`
- `ERMBG PyMatting Known-B`
- `ERMBG Classify`

## 安装

1. 把 ERMBG 安装到 ComfyUI 使用的 Python 环境:

   ```powershell
   <comfy-python> -m pip install -e <ermbg-root>
   ```

2. 把节点包复制或链接到 ComfyUI 的 `custom_nodes`:

   ```powershell
   Copy-Item -Recurse -Force <ermbg-root>\comfy_nodes <comfy-custom-nodes>\ermbg-comfy
   ```

   自定义节点目录不能命名为 `ermbg`,因为这会与 Python 包导入冲突。

3. 重启 ComfyUI。自定义节点是在进程启动时加载的。

4. 从配置的 `services.comfy_url` 验证节点注册情况:

   ```bash
   curl -s "<services.comfy_url>/object_info" | python -c "import json,sys; d=json.load(sys.stdin); print([k for k in d if k.startswith('Ermbg')])"
   ```

   预期的 key 包括 `ErmbgRouteMatte`、`ErmbgRouteStrategy`、
   `ErmbgPyMattingKnownB` 和 `ErmbgClassify`。

## 更新

- `ermbg/` 下的改动需要让 Comfy 的 Python 环境导入更新后的 ERMBG 包。
  使用 editable 安装可以简化这一步。
- `comfy_nodes/` 下的改动需要重新复制/链接节点包并重启 ComfyUI。
- Comfy 包装层要保持轻薄。route/profile 决策和 CorridorKey 执行应放在共享的
  ERMBG 代码中,而不是放在 Comfy 专属分支里。
