---
name: jargon-vocab-writeback
description: jargon-decoder 讲过的英文词会写回雅思项目的单词本，落点和复习命令在哪里
metadata:
  node_type: memory
  type: project
  originSessionId: sess_6fa585d2-a043-4085-962a-57e6cdc0504e
---

`jargon-decoder` skill（`C:\Users\27236\.agents\skills\jargon-decoder`）每次讲完术语，会把英文原词写回雅思项目 `C:\Users\27236\Desktop\IELTS`，2026-09-16 接通。

- 写入工具：`tools/save_vocab_record.py`；记录在 `.ielts/vocab/<日期>-<单词>-<短哈希>.json`，聚合在 `.ielts/progress.json` 的 `vocabulary` 段（`writing` 段由写作工具重建，两者互不覆盖）。
- **payload 里每条记录必须自带 `schemaVersion: 1`**，否则脚本立刻报 `error: schemaVersion is required` 退出——skill 文档的字段映射表没写这一条，2026-09-17 第一次调用就这么失败了一次。实操：整批词写成一个 `{"words": [...]}` 临时 JSON 文件，把文件路径当参数传进去，一次调用写完全部；`word` 只放英文原词（脚本用正则拒中文），中文黑话放 `jargonZh`。
- 同名单词只留一条记录，重复写只加 `seenCount`；`status`（new / learning / known）是复习进度，不传参就保持原值。
- 复习：`python tools/save_vocab_record.py --list 20 --project-root "C:/Users/27236/Desktop/IELTS"`。本机 Python 3.10.11 + pytest 9.0.3，测试在 `tests/`。
- IELTS 项目里**没有**独立的单词 skill，只有 `ielts`、`ielts-reading`、`ielts-writing`、`ielts-speaking` 四个；面板 `dashboard/` 只读 `writing`，单词本还没接进面板。

相关：[[plain-language-reporting]]
