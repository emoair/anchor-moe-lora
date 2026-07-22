# Qwen request-local trigger receipt v2

This receipt closes the metadata gap around the original 44-token Qwen
request-2 probe. It authenticates one complete chat-templated serialization,
one tokenizer call, and the trigger's covering token span without publishing
raw token IDs or a global token index.

## Scope

- Tokenizer-only and offline: no model weights, GGUF, GPU, provider, network,
  Gold, heldout, or scaffold JSONL bodies are read.
- Bound consumer dependency baseline:
  `b0441e6beaa07b180d7fc69e462b4d2babf21792`. It must be an ancestor of, or
  equal to, the current checkout; it is not a permanent `HEAD == baseline`
  lock. The executed materializer is bound separately by physical
  config/schema/implementation SHA-256 identities.
- Bound producer baseline:
  `744e23f975b13923903f5fabe04c32e74ea25dc4`.
- Qwen tokenizer revision:
  `3c3787b7c81927cc64ad45dc32ff1c9ce2a5de34`.
- Tokenizer binding SHA-256:
  `a76b0f60e5c1e2d92b8a8d9131f9afe9edfda3fcbf0221c4234359f70e806425`.

This is a diagnostic companion artifact. It does **not** authorize training,
formal evaluation, numeric-equivalence claims, KV-sharing claims, or formal
thresholds. The referenced TF32 run remains `proxy_signal_passed` only.

## Frozen probe facts

| Field | Value |
| --- | --- |
| Complete request-2 tokens | 44 |
| Trigger covering span | `[25, 33)` |
| Index semantics | zero-based, end-exclusive |
| Covering span width | 8 tokens |
| Leading overhang | 0 UTF-8 bytes / 0 codepoints |
| Trailing overhang | 1 UTF-8 byte / 1 codepoint |
| Exact request-2 UTF-8 SHA-256 | `ed6adfcbd0052fdda52a5ab8c52ed04d6e55c7f62493f0d326d4e1b29d55c9f3` |
| Ordered token-ID digest | `d989d46116cd50f30d5bba1be48a366e2a04efb8c156550d0f11a532f19121e6` |
| Trigger token-ID digest | `1d6889128be1b4b84ae22999ffe267a1cc862209b7c38ef3f932a5e69851a412` |

The ordered-ID digest algorithm is
`sha256_concat_signed_int64_big_endian_v1`: encode each ordered token ID as one
signed 64-bit big-endian integer, concatenate those bytes, then SHA-256 the
result. No JSON or delimiter is part of this canonical digest preimage.

## Reproduce

Use the pinned `anchor-mvp` environment and the local authenticated tokenizer
directory. The default path is `D:\LLM\models\qwen2.5-1.5b-instruct-hf`.

```powershell
$env:PYTHONPATH = "src"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
conda run -n anchor-mvp python `
  scripts\research\materialize_qwen_request_local_trigger_receipt_v2.py `
  --output runs\qwen_request_local_trigger_receipt_v2_rebuild\receipt.json
```

The command refuses to overwrite an existing output directory. To verify the
checked-in artifact without replacing it:

```powershell
$env:PYTHONPATH = "src"
conda run -n anchor-mvp python -m pytest -q `
  tests\test_qwen_request_local_trigger_receipt_v2.py
```

The focused suite performs an offline byte-identical rebuild in a temporary
directory and validates the mandatory sidecar format:
`<64 lowercase hex>  receipt.json\n`.

## Artifacts

- Config: `configs/research/qwen_request_local_trigger_receipt_v2.yaml`
- Config schema:
  `configs/research/qwen_request_local_trigger_receipt_v2_config.schema.json`
- Receipt schema:
  `configs/research/qwen_request_local_trigger_receipt_v2.schema.json`
- Materializer:
  `src/anchor_mvp/research/qwen_request_local_trigger_receipt_v2.py`
- Receipt:
  `fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json`
- Mandatory sidecar:
  `fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json.sha256`

The producer must still publish and bind its v2 companion schema/manifest
before any downstream prerequisite state can be advanced. Formal-v3 release
locks and the remaining protected source inventories are separate gates.
