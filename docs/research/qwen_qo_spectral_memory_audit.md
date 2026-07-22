# Q/O LoRA spectral memory-risk audit

This diagnostic asks a narrow question: did the equal-budget Q+O adapter learn a
small number of strong residual-stream writeback directions that align with
frequent training-target token embeddings? It does **not** claim to prove that an
adapter stores a readable answer, source file, exploit, or arbitrary byte string.

## Scope and isolation

- CPU only; no model forward pass, GPU, provider, or network request.
- Reads the two 80-record synthetic **training** partitions only. It never opens
  eval-proxy, held-out, Gold, or protected bodies.
- Authenticates the Qwen base checkpoint, tokenizer, comparison manifest, and the
  Q+O and wide adapter artifacts before analysis.
- Loads only selected rows of `model.embed_tokens.weight`; it never materializes
  the 1.5B model.
- Emits only metrics and SHA-256 inventories. No sample text or token ID is
  written to the receipt.

## Mathematics

For one LoRA projection, `ΔW = (α/r) B A`. Instead of materializing the dense
1536×1536 matrix, the runner computes thin QR factorizations

`B = Q_B R_B`, `Aᵀ = Q_A R_A`

and runs `torch.linalg.svdvals((α/r) R_B R_Aᵀ)`. Those are exactly the non-zero
singular values of `ΔW`. The spectral norm is `svdvals.max()`; the known-bad
`torch.linalg.matrix_norm(..., ord=2)` path is never used.

Each layer reports Frobenius and spectral norms, stable rank, entropy-based energy
effective rank, top-1/2/4 singular-energy fractions, and cross-layer energy
concentration. The audit also compares the O-projection B-column subspace with:

1. the 128 most frequent target tokens;
2. 128 prompt-only control tokens;
3. 128 low-frequency target tokens; and
4. 128 deterministic unseen random-vocabulary controls.

The selection is hashed but never serialized as token IDs. Projection energy is
`||Q_Bᵀ e_token||²` for normalized embedding direction `e_token`.

## Run

```powershell
conda run -n anchor-mvp python scripts/research/audit_qwen_qo_spectral_memory.py `
  --config configs/research/qwen_qo_spectral_memory_audit_v1.yaml
```

The command atomically creates:

`artifacts/diagnostics/qwen_qo_spectral_memory_audit_v1/receipt.json`

and a mandatory `receipt.json.sha256`. The destination must not already exist.

## Interpretation boundary

A high top-1 energy fraction plus target-frequency alignment is evidence of a
low-dimensional template/writeback shortcut risk. It is not causal proof of
verbatim memorization. This synthetic fixture contains routing JSON rather than
real exploit bodies, so exploit-code memorization is explicitly untested. A
future causal audit should use harmless unique canaries, paraphrased extraction
prompts, template-family-held-out evaluation, and multiple seeds.

## Reproduced result

For the Q+O adapter, O-projection top-1 singular energy is `82.3186%`, its
energy-effective rank is `2.1161 / 8`, and total O delta energy is `1.76715x`
Q delta energy. The O-column subspace aligns `1.95911x` more strongly with
frequent target tokens than deterministic random-vocabulary controls; the
prompt-only control ratio is `1.00688x`. The same pattern appears in the
equal-budget Wide arm (`86.6574%` top-1 energy and `2.04342x` target/control
alignment).

O energy is distributed across layers rather than exploding in a few layers:
the four largest layers contain only `24.34%` of O energy, corresponding to
`25.71 / 28` effective layers. This is evidence of a broad, low-dimensional
writeback shortcut. It is not proof of verbatim answer or exploit-code storage.
The authenticated result is
[`results/qwen_qo_spectral_memory_audit_v1_receipt.json`](results/qwen_qo_spectral_memory_audit_v1_receipt.json).
