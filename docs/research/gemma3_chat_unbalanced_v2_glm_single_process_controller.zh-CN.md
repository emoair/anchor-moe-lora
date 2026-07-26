# GLM 单进程实时控制器

状态：这是 additive 控制器契约。它从 consumer 提交
`60e88aa7a6c2dcdd8a3cd9f496f84d3a1ea76bb2` 的四个精确 `100644` Git blob
认证 sharded-v1 binding。真实执行仍须等待本控制器独立审计、提交并推送，
并在同一进程内加载 credential 与 runtime-HMAC 内存槽。它不修改冻结的
FINAL、source overlay、四个 batch profile、batch 实现或 teacher 实现。

Controller config 静态锁定本模块的 canonical path 与物理 SHA-256。Loader
在 consumer/Git 检查、凭据读取、provider 构造或网络操作之前认证单次 bytes
snapshot，并在 config 加载末端重验相同 bytes。

## 目标与边界

`gemma3_chat_unbalanced_v2_live_controller.py` 在完整
`exact1 -> bounded15 -> bulk` 生命周期内只持有一个 `RuntimeSecretSlots`。
凭据只从匿名 OS pipe 或进程内 receiver 读取一次；同一进程的一把随机
HMAC key 同时认证 receipts 与控制器 WAL。退出时只声明 best-effort 清理，
不声称 Python/HTTP client 的 forensic zeroization。

控制器启动时拒绝非空 output root，因此不提供、也不声称跨进程断点续跑。
旧 batch 的 `--resume-execute` 行为不变，本控制器不调用它。

## 认证 WAL 与路径身份

控制器创建 `O_EXCL` genesis、同 PID/run-ID process lock 和零填充序号的
不可变 WAL entry。entry 同时形成全局物理 SHA 链和逐 job 链；dispatch
event 必须先 HMAC 认证并 fsync，之后才能调用 `teacher.complete`。

终端提交固定为 signed `GROUP_PREPARE`、冻结 batch 的物理 group commit、
单次 bytes snapshot 复核 group/receipt/event，再签名 `GROUP_TERMINAL`。
`events.jsonl` 与 phase receipts 只是 projection，每次 replay 前必须逐项
等于认证序列。

Genesis 还签名 output root、alignment、automation、groups/WAL 目录、追加
日志和 dataset marker 的物理身份。每次运行时读写前后重验目录身份与
reparse 状态；symlink/junction/reparse 替换必须在外部写入前拒绝，动态
WAL/group/status/kill-switch 路径也同样检查。

## 阶梯与回退

控制器锁定 `smoke_exact1`、`bounded_small_c1`、`bulk_c30`、`bulk_c16`
四个物理 SHA-256。Exact1 全绿后才运行 15，15 全绿后才运行 c30。

c16 仅在以下条件同时成立时启用：存在非空、按 event/entry digest 确定
排序的 signed c30 `provider_rate_limit` inventory；uncertain=0；签名
cooldown lease 已到期。Lease 绑定完整事件 inventory 的 hash/count、源
WAL chain tip、fallback profile、run ID，以及 `RateLimitError` 与 profile
给出的等待时间。`status.json` 仅用于观测，绝不授权回退。等待最多按
60 秒切片，并在每片重验 consumer、secret slot、source、kill switch 与
WAL。本控制器不接受 `provider_instability_reconciled`。

## 零模型复现

```powershell
$env:PYTHONPATH='src'
python -m anchor_mvp.data.gemma3_chat_unbalanced_v2_live_controller --dry-run
python -m anchor_mvp.data.gemma3_chat_unbalanced_v2_live_controller --validate-only
python -m pytest -q tests/test_gemma3_chat_unbalanced_v2_live_controller.py
```

真实执行必须显式开启，且禁止通过 argv、环境变量、TTY 或普通文件传递凭据：

```powershell
<anonymous-pipe-producer> | python -m anchor_mvp.data.gemma3_chat_unbalanced_v2_live_controller --execute --credential-stdin
```

公开状态只含 hashes、counts、固定 reason code 和阶段状态，不含 credential、
runtime HMAC、prompt、answer 或 source/heldout 正文。所有结论仍为 diagnostic；
formal training 与质量授权均为 false。
