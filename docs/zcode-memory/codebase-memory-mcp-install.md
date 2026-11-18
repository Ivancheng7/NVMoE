---
name: codebase-memory-mcp-install
description: codebase-memory-mcp 装在哪、为什么必须设 CBM_RUNTIME_DIR 和 CBM_CACHE_DIR 两个环境变量
metadata:
  node_type: memory
  type: project
  originSessionId: sess_f140571a-500a-4942-88a7-6d80b4a8780b
---

codebase-memory-mcp 0.11.0 装在 `C:\cbm\bin\codebase-memory-mcp.exe`，索引缓存 `C:\cbm\cache`，守护进程会合目录 `C:\cbm\runtime`。已接入 ZCode 用户配置 `~/.zcode/cli/config.json` 的 `mcp.servers.codebase-memory-mcp`。

**Why:** 该二进制启动时会校验路径上每一级目录的权限，拒绝任何"别的账号可修改"的目录。这台机器的 `C:\Users\27236` 上有 Codex Windows 沙箱留下的 `orca\CodexSandboxUsers` 组（Modify 权限）和一条旧系统残留 SID，导致默认位置（用户目录下的 `%LOCALAPPDATA%\Programs` 和 `~/.cache`）全部启动失败。`C:\cbm` 已用 `SetAccessRuleProtection($true,$false)` 收紧为仅当前用户可访问。

**How to apply:** 任何调用它的地方（MCP 配置、CLI、脚本）都必须带上这两个环境变量，否则会报 "secure daemon endpoint could not be created" 或 "cache-private" 错误。不要改回用户目录下的默认位置，除非先处理掉 Codex 沙箱的 ACL——那会影响 Codex 的沙箱写入能力，属于另一个工具的配置，不要擅自动。相关：[[wsl-run-environment]]
