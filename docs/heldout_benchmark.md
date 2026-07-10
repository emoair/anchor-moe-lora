# Frozen held-out benchmark and leakage gate

## The pre-bulk invariant

Bulk distillation must not start until both immutable artifacts below exist and
`anchor-heldout verify` exits successfully:

- `artifacts/benchmark/heldout_v1/manifest.json` and its `.sha256` sidecar freeze
  the case JSONL, dedicated case/seed namespaces, case-family digests, fixture
  trees, five-stage order, and the three primary arms.
- `artifacts/benchmark/heldout_v1/leak_audit.prebulk.json` and its `.sha256`
  sidecar record a local-only
  scan of the then-current training JSONLs and SOP sources. It contains hashes,
  counts, and collision metadata only; it never contains source text.

The frozen v1 manifest SHA-256 is
`1ac7240d700a67458dc713b66ff085f1e51795b26cdacff688063bc60af3194c`.
Changing a case, fixture, manifest, or sidecar fails closed. A fresh benchmark
version needs a new namespace and new manifest; do not silently rewrite v1.

```powershell
$env:PYTHONPATH = "src"
py -m anchor_mvp.benchmark.heldout_cli verify `
  --cases configs\benchmark\heldout_cases_v1.jsonl `
  --fixtures-root examples\benchmark\fixtures `
  --manifest artifacts\benchmark\heldout_v1\manifest.json `
  --leak-audit artifacts\benchmark\heldout_v1\leak_audit.prebulk.json
```

Run the checker again before every training-corpus expansion. Only the checker
subcommand accepts training paths. Neither the inference runner, sandbox
evaluator, nor report generator has a training-data argument.

```powershell
py -m anchor_mvp.benchmark.heldout_cli check-leakage `
  --cases configs\benchmark\heldout_cases_v1.jsonl `
  --fixtures-root examples\benchmark\fixtures `
  --manifest artifacts\benchmark\heldout_v1\manifest.json `
  --leak-audit artifacts\benchmark\heldout_v1\leak_audit.prebulk.json `
  --training-jsonl data\live_smoke\data_frontend.jsonl `
  --training-jsonl data\live_smoke\data_review.jsonl `
  --training-jsonl data\live_smoke\data_security.jsonl `
  --sop-source skills\frontend.md `
  --sop-source skills\review.md `
  --sop-source skills\security.yaml
```

The gate checks exact normalized hashes, held-out seed IDs, explicit case-family
labels, containment, and approximate text similarity. Any hit produces `FAIL`
and a nonzero CLI result. Training record contents are function-local and do not
enter model prompts, evaluator inputs, records, reports, or logs.

## Five-stage primary comparison

The main experiment fixes this call order and the per-stage completion-token cap:

1. Planner
2. Tool Policy / Approval
3. Frontend Coder
4. Frontend Domain Reviewer
5. Final Security Gate

`configs/benchmark/heldout_q4_v1.json` defines the primary arms:

| Arm | Adapter assignment | Valid causal role |
| --- | --- | --- |
| A `base_matched_calls` | Native Gemma 4 12B Q4 at all five stages | Reference index 100 |
| B `mixed_matched_calls` | One mixed-all LoRA at all five stages | Single-adapter control |
| C `c_pipeline` | Five task-specific LoRAs selected by the application router | Task-routed pipeline |

All three have five calls and identical token caps. A one-call base result may be
reported as an auxiliary product-shape baseline only; it cannot establish the
benefit of routing. The C arm is not a token-level neural MoE.

The model policy stage emits `APPROVE`, `BLOCK`, or `ESCALATE` for inert proposal
labels. It is never an authority. A deterministic local allowlist computes the
actual decision, and only trusted validator commands may run. The policy metrics
cover overall and per-class accuracy plus deterministic enforcement accuracy.

## Evaluation without active payloads

Held-out cases use an independent `anchor-heldout-*` namespace, independent
`anchor-ho-*` seeds, and `hf-v1-*` case families. Security cases contain semantic
intent labels only. URLs, executable snippets, event handlers, shell commands,
and active XSS-like material are rejected during freeze.

The reviewer receives a deterministic benign mutation: one case-specific literal
accessible name is removed from the generated artifact. Repair passes only when
the mutation was actually applied and the exact behavior is restored.

For benign cases, Pass@1 is based on actual isolated `npm run build` and
`npm run test` results. The trusted fixture reads the generated HTML as data; it
does not execute that HTML. Validator stdout/stderr is reduced to hashes and exit
codes, and temporary workspaces are removed by default.

Metrics include sandbox build Pass@1, plan quality, tool-policy accuracy,
review-repair rate, security TPR/FPR, composite success, end-to-end latency,
tokens, peak VRAM, and tokens per composite success. The report displays the
native Q4 absolute value and index 100, then B/C deltas and ratios.

## No-network mock E2E

The mock validates the same frozen inputs, performs exactly 5 calls for every
primary arm, runs the real local build/test fixture, and produces records,
metrics, Markdown/CSV, and SVG without an API call or model load:

```powershell
py -m anchor_mvp.benchmark.heldout_cli mock-e2e `
  --cases configs\benchmark\heldout_cases_v1.jsonl `
  --fixtures-root examples\benchmark\fixtures `
  --manifest artifacts\benchmark\heldout_v1\manifest.json `
  --leak-audit artifacts\benchmark\heldout_v1\leak_audit.prebulk.json `
  --specs configs\benchmark\heldout_q4_v1.json `
  --output-dir runs\heldout-mock-v1 `
  --no-vram
```

For a live run, replace `mock-e2e` with `run`, pass `--output` and `--metrics`,
then invoke `evaluate` on the raw records. The evaluator deliberately offers no
training-data option.
