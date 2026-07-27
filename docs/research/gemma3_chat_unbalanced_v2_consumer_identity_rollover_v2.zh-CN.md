# Gemma 3 unbalanced-v2 Consumer 身份滚动

该 additive rollover 在不修改任何仍引用旧 `60e88aa7...` Consumer 的冻结
controller、finalizer、manifest 或 binding 文件的前提下，认证新 Consumer
提交 `6240ae111182104f22f08e1a569deae866c6e210`。

model-free controller 会认证 commit、parent、tree、branch、upstream、精确
17 文件差异、每个 Git blob 以及对应物理文件字节。release implementation
binding 与 preflight receipt 分别使用各自的 Draft 2020-12 schema 校验，
mandatory sidecar 也必须匹配；在终端边界再次重验 Git 与物理字节。Consumer
工作区中无关的脏文件不被信任，也不会被读取。

Git 发现不依赖 `PATH`。controller 以 lexical path、字节数和 SHA-256 固定
`C:/Program Files/Git/mingw64/bin/git.exe`，拒绝 symlink/reparse 路径组件，
并在读取前后交叉核对 `lstat` 与打开句柄的身份；后续只调用该认证后的绝对
路径。环境变量不能替换 Git 可执行文件。

版本化 Teacher 链新增 finalizer config、final-manifest schema、external
binding schema、integrated config 和 release implementation。三个携带
Consumer 身份的层级都绑定同一个 `6240ae111...`。旧 v1/v2 链保持逐字节
不变，并明确不可用于本流程。

model-free 验证命令：

```powershell
$env:PYTHONPATH='src'
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe -m anchor_mvp.data.gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2 --validate-only
```

唯一允许的 blocker 是 `controller_credential_slot_unloaded` 与
`runtime_hmac_slot_unloaded`。凭据仍只能从匿名进程通道进入；runtime HMAC
在同一 controller 进程内生成并持有。命令行、环境变量、文件或 receipt
均不能传入密钥。

本契约的 provider、network、model、GPU 请求均为 0；它不授权训练、
formal release 或 live execution。

全部验证命令固定使用同一个本地 Python 环境，不假设存在独立
`ruff.exe`：

```powershell
$python = 'D:/LLM/envs/gemma3-keras-torch/Scripts/python.exe'
& $python -m pytest -q tests/test_gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py
& $python -m ruff check scripts/data/build_gemma3_chat_unbalanced_v2_consumer_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v2.py tests/test_gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py
& $python -m ruff format --check scripts/data/build_gemma3_chat_unbalanced_v2_consumer_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v2.py tests/test_gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py
& $python -m py_compile scripts/data/build_gemma3_chat_unbalanced_v2_consumer_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v2.py tests/test_gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py
```

本次验证固定观测的工具身份为 `ruff 0.15.21`。
