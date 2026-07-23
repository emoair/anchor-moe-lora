# Gemma 3 1B IT tokenizer binding (diagnostic v1)

This receipt makes the local Gemma 3 1B IT export usable without guessing its
chat format. It authenticates all five export files, applies an explicit
runtime correction for the exported BOS/EOS mismatch, runs the repository's
real five-role causal materializer, and checks all 1,000 records without
loading the model or requesting a GPU.

It is a diagnostic prerequisite only. It does **not** authorize training or a
formal result.

## One command

From the repository root:

```powershell
& "$env:USERPROFILE\.conda\envs\anchor-mvp\python.exe" scripts\research\build_gemma3_tokenizer_binding_v1.py --publish
```

The default local export is:

```text
D:\LLM\models\google-gemma-3-1b-it-keras-v3\hf-export-keras-hub-0.29.1-bf16
```

To use the same authenticated files at another location for this launch only:

```powershell
$env:ANCHOR_GEMMA3_1B_IT_HF_PATH = "D:\path\to\hf-export"
& "$env:USERPROFILE\.conda\envs\anchor-mvp\python.exe" scripts\research\build_gemma3_tokenizer_binding_v1.py --publish
Remove-Item Env:\ANCHOR_GEMMA3_1B_IT_HF_PATH
```

Publishing is atomic and no-replace. If the output already exists, the command
stops instead of overwriting evidence. Audit an existing output with:

```powershell
& "$env:USERPROFILE\.conda\envs\anchor-mvp\python.exe" scripts\research\audit_gemma3_tokenizer_binding_v1.py
```

## Exact serialization

The exported `config.json` declares BOS=1/EOS=2, while the authenticated
SentencePiece model and tokenizer metadata bind BOS=2/EOS=1. Canonical files
are never edited. The runtime overlay is:

```text
[BOS=2]
SP("<start_of_turn>user\n" + prompt + "<end_of_turn>")
[EOS=1]
SP("\n<start_of_turn>model\n" + target + "<end_of_turn>")
[EOS=1]
SP("\n")
```

Literal `<bos>` and `<eos>` are never passed to SentencePiece. Prompt and
assistant-prefix labels are `-100`; the target, `<end_of_turn>`, and EOS are
trainable labels; the terminal separator newline is masked. The structured
sequence must match the local HF tokenizer's visible-template encoding with
`fix_mistral_regex=false` for every record.

The text structure is bound to Google's Gemma documentation:

- [Gemma prompt structure](https://ai.google.dev/gemma/docs/core/prompt-structure)
- [Gemma PyTorch guide](https://ai.google.dev/gemma/docs/core/pytorch_gemma)

Gemma IT has no separate system role in this contract. System instructions
belong inside the first user prompt.

## Why the sequence length is 768

The authenticated corpus is not safely representable at 512 tokens:
514 of 1,000 records exceed 512, with the security role reaching 665 tokens.
The Gemma diagnostic profile therefore freezes 768 and forbids truncation.
This choice was made before any GPU run and is part of the receipt, not a
runtime adjustment.

The manifest publishes only aggregate per-role/per-split token statistics and
ordered sequence digests. It never publishes prompts, targets, record IDs, or
raw token ID arrays.

## Bound files

- Config: `configs/research/gemma3_1b_it_tokenizer_binding_v1.yaml`
- Chat policy: `configs/research/gemma3_1b_it_chat_template_policy_v1.json`
- Implementation: `src/anchor_mvp/training/gemma3_tokenizer_binding_v1.py`
- Output: `fixtures/research/gemma3_1b_it_tokenizer_binding_v1`

The build reads no Gold, heldout, provider, or network source. Hashing the
weight file authenticates bytes; it is not a model load.
