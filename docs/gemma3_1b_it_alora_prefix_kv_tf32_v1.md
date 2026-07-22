# Gemma 3 1B IT aLoRA Prefix-KV TF32 diagnostic

This diagnostic-only mechanical probe is isolated from the Qwen v1/v2
profiles. It loads the local KerasHub-to-Transformers export into float32
parameter storage, enables TF32 with eager attention, and attaches a
deterministic rank-4 PEFT aLoRA adapter only to all 26 `q_proj` layers.

The probe executes base and aLoRA full/split routes, adapter-off/active prefix
routes, and a missing-trigger control. Its request uses the versioned
`anchor.gemma3-plaintext-r2.v1` serialization, tokenized exactly once with the
physical SentencePiece model and an explicit BOS. It does not depend on the
unavailable Windows `tensorflow-text` package or an unbound chat template.

The exported HF config reports BOS/EOS `1/2`, while the physical SentencePiece
model reports `2/1`. The diagnostic fails unless physical SentencePiece BOS
`2` is used, and records both identities instead of silently conflating them.

```powershell
$env:PYTHONPATH = "src"
anchor-gemma3-alora-prefix-kv --config configs/research/gemma3_1b_it_alora_prefix_kv_tf32_v1.yaml --preflight
anchor-gemma3-alora-prefix-kv --config configs/research/gemma3_1b_it_alora_prefix_kv_tf32_v1.yaml --execute
```

The checked-in RTX 3080 Ti receipt passed all diagnostic gates: all 26 prefix
KV layers are bit-equal, base and aLoRA full/split paired differences are zero,
the deterministic adapter effect is non-zero, and the missing-trigger effect
is zero. Peak allocated VRAM was about 3.83 GiB.

This is a local proxy signal only. It does not claim numeric equivalence,
quality improvement, training readiness, multi-stream execution, zero-copy KV,
or full-generation KV sharing.
