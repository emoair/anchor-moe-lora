# Gemma 3 unbalanced-v2 consumer identity rollover

This additive rollover authenticates consumer commit
`6240ae111182104f22f08e1a569deae866c6e210` without changing the frozen
controller, finalizer, manifest, or binding files that reference the superseded
`60e88aa7...` consumer.

The model-free controller authenticates the commit, parent, tree, branch,
upstream, exact 17-file diff, every Git blob, and the corresponding physical
file bytes. It separately validates the release-implementation binding and
preflight receipt with their own Draft 2020-12 schemas, verifies the mandatory
receipt sidecar, and repeats the Git and physical checks at the terminal
boundary. Unrelated dirty files in the consumer worktree are not trusted or
read.

Git discovery does not depend on `PATH`. The controller pins
`C:/Program Files/Git/mingw64/bin/git.exe` by lexical path, byte length, and
SHA-256; rejects symlink/reparse components; compares `lstat` and open-handle
identities before and after reading; then invokes only that authenticated
absolute executable. Environment substitution is not supported.

The versioned Teacher chain contains a new finalizer config, final-manifest
schema, external binding schema, integrated config, and release implementation.
All three consumer-bearing layers bind the same `6240ae111...` identity. The
old v1/v2 chain remains byte-identical and is explicitly non-consumable for this
flow.

Model-free validation:

```powershell
$env:PYTHONPATH='src'
D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe -m anchor_mvp.data.gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2 --validate-only
```

The only permitted blockers are
`controller_credential_slot_unloaded` and `runtime_hmac_slot_unloaded`.
Credentials still enter only through the anonymous process channel, and the
runtime HMAC is generated and retained in the same controller process. No key
is accepted through arguments, environment variables, files, or receipts.

This contract performs zero provider, network, model, and GPU requests. It does
not authorize training, formal release, or live execution.

All verification commands are anchored to the same authenticated local Python
environment. They do not assume a standalone `ruff.exe`:

```powershell
$python = 'D:/LLM/envs/gemma3-keras-torch/Scripts/python.exe'
& $python -m pytest -q tests/test_gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py
& $python -m ruff check scripts/data/build_gemma3_chat_unbalanced_v2_consumer_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v2.py tests/test_gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py
& $python -m ruff format --check scripts/data/build_gemma3_chat_unbalanced_v2_consumer_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v2.py tests/test_gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py
& $python -m py_compile scripts/data/build_gemma3_chat_unbalanced_v2_consumer_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v2.py tests/test_gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.py
```

The frozen tool identity observed for this verification is `ruff 0.15.21`.
