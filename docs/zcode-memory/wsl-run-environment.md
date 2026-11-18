---
name: wsl-run-environment
description: WSL2 上 localMoE 真实推理的运行环境：源码克隆、构建必须带 RUSTC_WRAPPER、模型与 cuBLAS 路径
metadata:
  node_type: memory
  type: project
  originSessionId: sess_73dcebc4-2d00-4ea2-9a82-2b27c4336e16
---

localMoE 的真实推理只在 WSL2 内运行，Windows 侧的 `C:\Users\27236\Desktop\localmoe-clean` 是编辑目录。

- 真源码是 WSL 里的一个 **git clone**：`/srv/localmoe/build/localmoe-clean-git`（ext4，2026-09-16 建立），`origin` 指向主仓库 `/mnt/c/Users/27236/Desktop/localmoe`。同步 = Windows 侧提交 → WSL 侧 `git pull --ff-only`；**WSL 侧看不到未提交的改动**。
- `origin` 只能指向主仓库 `Desktop\localmoe`。`Desktop\localmoe-clean` 的 `.git` 是写着 Windows 绝对路径的**文件**，Linux 的 git 解析不了，任何命令都报 `fatal: not a git repository`。`localmoe-clean` 本身就是主仓库的一个 worktree（主仓库还挂着 `~/.codex/worktrees/560f/localmoe` 等）。
- **把实验分支送进 WSL clone（2026-09-16 实测的路子）**：worktree 同样不能被 WSL 的 git 读（`git fetch /mnt/c/.../localmoe-slot-reuse` 报 `not a git repository: .../localmoe-clean/C:/Users/...`），要在**主仓库**里能看到的引用才行。所以：Windows 侧 `git bundle create .zcode/x.bundle ^<父提交> <分支>` → WSL 侧 `git fetch -f /mnt/c/.../.zcode/x.bundle <分支>:refs/heads/<分支>`（该分支正被 checkout 时会被拒，先 `git checkout --detach` 再取）。测完 `git checkout -f codex/business-restart` + 删掉 clone 里的临时分支 + 重建，别把 clone 留在实验分支上。
- **`git reset --hard FETCH_HEAD` 挪的是当前分支，不是"临时区"**：2026-09-16 实测踩过——那时 HEAD 正在 `codex/business-restart` 上，结果把克隆里的**交付分支**挪到了探针提交（三次测量的构建源码都对，坏的只是标签，但会让人以为已切回干净状态）。绝不要在交付分支上敲它；只在 `--detach` 的 HEAD 上敲。修法：`git checkout -f -B codex/business-restart origin/codex/business-restart`。
- 编译门槛与 `RUSTC_WRAPPER` 的完整细节见 [[cargo-check-ice-wrapper]]：不带 wrapper 会 ICE；2026-09-16 起 `run-identified.sh` 已自动设好，wrapper 的执行位也提交进 git 了。
- 为什么不直接在 `/mnt/c` 上构建（2026-09-16 实测）：`/mnt/c` 是 9p，每个文件操作约 7–9 ms，而 ext4 是微秒级——遍历+哈希整棵 rust 源码 0.64 s vs 0.016 s、读完所有源码 0.63 s vs 0.010 s、建 500 个空文件 5.0 s vs 1.5 s。因为 `CARGO_TARGET_DIR` 本来就指到 ext4，直接在 `/mnt/c` 构建只多约 1.3 s，**所以选 clone 不是为了速度**，是为了 git 能用（`/mnt/c` 上那个 `.git` 是 Windows 路径指针文件）、快照隔离（不跟别的 agent 正在写的文件抢），以及 ext4 上权限和 inotify 是真实的。
- 模型：`/srv/localmoe/models/deepseek-v4-flash-0731-9e165c30e2704aec5d9d593cce3eebd58bbef1cb`（ext4 原生路径，挂载源 `/dev/sdd`）。
- Flash-0731 的 exact 路径固定加载 CUDA 12.8 的 `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.5.5`（SHA-256 `18a5df54…`，`cublasGetVersion=120805`）。
- 每次正式运行用独立 `CARGO_TARGET_DIR=/srv/localmoe/targets/<name>-target`，产物在 `release/colibri-cuda`；结果写到 `/srv/localmoe/targets/<name>.jsonl`。
- 想拿 CUDA API 级耗时**别用 `nsys`**：2024.6.2 在这台 WSL2 上抓不到 CUDA 数据（导出时报 `does not contain CUDA trace data`）。要定位热点走插桩计时，做法见 [[flash-0731-expert-slot-cost]]。
- 旧拷贝目录 `/srv/localmoe/build/localmoe-clean-cuda128`（手动 rsync 时代）已被 clone 取代；确认别人不再用之前不要删。2026-09-16 也把这个旧目录从坏掉的 worktree 指针改成了真 git 仓库（remote 名 `win`，指向主仓库），同步用 `git fetch win && git reset --mixed win/codex/business-restart`。注意它里面设了 `core.autocrlf=true`：该目录工作区是 CRLF 而仓库存 LF，**改成 false 会让约 265 个文件全显示为已修改**，把真改动淹掉。这个目录里常带别的会话未提交的在建工作，**永远不要对它 `git checkout -f` 或 `reset --hard`**。
- 2026-09-16 实测：clone → 构建 30 秒成功 → 单步跑出 token `55222`、43 层、`status: PASS`，与记录基线一致。
- WSL 内 `bash -lc` 里带 `$var` 的 for 循环会被吞掉，逐条写命令更可靠；从 Windows 侧敲命令用 `wsl.exe -e bash -c '单行命令'`（外层单引号、内层双引号，嵌套变量写 `\$`，构建类命令用 `-lc` 才有 cargo）。输出开头那行 localhost 代理警告是噪音。

## 官方 oracle 的 python 环境

`/srv/localmoe/toolchains/flash-reference-py312` —— **用这个**。Python 3.12.13 + torch 2.10.0+cu128 + tilelang 0.1.8 + triton 3.6.0，`torch.cuda.get_device_capability()` = `(12, 0)`，匹配 `/srv/localmoe/sources/deepseek-v4-flash-git/inference/requirements.txt`。

旁边的 `flash-reference-py314` 是坏的，别用：tilelang 在 Python 3.14 上 import 就崩，`tvm_ffi/registry.py` 报 `AttributeError: attribute '__dict__' of 'type' objects is not writable`。

## 从 Git Bash 调 wsl.exe 的三个坑（2026-09-16 都实际踩过）

1. **`bash -lc '...'` 里的 `$VAR` 和 `$?` 不可靠。** `$?` 恒为 0（连 `python3 -c "sys.exit(3)"` 都报 0），`$E` 这类变量会被吞成空串——路径悄悄变成 `/lib/...` 或 `/bin/python`，于是得出"没有 torch""那是仓库副本不是环境"这类**假结论**。写死字面路径，不要用变量。真要判退出码，用 wsl.exe 自己的：`wsl.exe -- bash -lc 'exit 7'; echo $?` 在 Git Bash 里正确返回 7。
2. **`wsl.exe` 的输出重定向到 Windows 文件是 UTF-16**，Read 工具报 "Unsupported or binary text encoding"；前台管道接 `tr -d '\000'` 可读。ZCode 后台任务的 stdout 日志文件同理，也要先 `tr -d '\000'` 再看。另外中文经 Git Bash 显示成 GBK 乱码（`纇Km0R` 那串）——只是显示层噪音，不影响命令执行，别去"修"。
2b. **`pgrep -fa colibri-cuda` 会匹配到自己那条 bash 命令行**，看起来像"还有进程在跑"。判断真进程用 `pgrep -c`，或拿 pid 后 `ps -o pid,etime,time,pcpu,rss,stat -p <pid>`、再看 `/proc/<pid>/io`。
2c. **要把 Windows 侧写的分析脚本喂给 WSL 里的结果文件**：脚本写进 `.zcode/`（未跟踪，不进库），从 WSL 里 `python3 /mnt/c/Users/27236/Desktop/localmoe-clean/.zcode/foo.py /srv/localmoe/targets/x.jsonl` 直接跑，不用拷进 ext4、也不污染仓库。
2d. **`sed -n "1,12p;/x/,$p"` 里的 `$` 会被外层双引号吃掉**（报 `unexpected ','`）；按段的显示改用 `grep -A N`，或先落到 `/tmp/xx.txt` 再 `head`/`grep` 分开看。
2e. **在 `wsl.exe -e bash -c '...'` 里给 shell 变量赋 `/mnt/c/...` 路径会被 MSYS 改写**（2026-09-16 实测：`P=/mnt/c/...; python3 $P` 报 `Files/Git/mnt/c/...: No such file or directory`），也别在单引号里塞多行 Python（引号/反斜杠被吞，报 `unexpected EOF`）。做法：脚本写进 `.zcode/*.py` 或 `.zcode/*.sh`，调用时**写死字面绝对路径**，不做变量赋值。
3. **外层单引号意味着内层 Python 不能有单引号**；heredoc 用 `<<"PY"`，Python 里只用双引号，否则 f-string 的反斜杠转义会炸。
4. **WSL 会自己重启**（2026-09-16 实测遇到一次：`uptime` 只有 8 分钟、`/tmp` 全空）。重启会杀掉后台在跑的长任务（那次 `run-identified.sh test` 就这样没了，日志也不见了）。所以长任务的日志别只写 `/tmp`——写到 `/srv/localmoe/targets/` 或 clone 目录里；「后台任务已启动」不等于「会跑完」，回来先看产物在不在。

相关：[[flash-0731-official-oracle-unstable]]、[[flash-0731-baseline-run-recipe]]
