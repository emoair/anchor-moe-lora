# Five-role Q-only consumer v2

This additive consumer authenticates the complete 1,000-record synthetic
diagnostic fixture before selecting one role. The dataset is 200 task bundles
times five roles: `planner`, `tool_policy`, `frontend_gen`, `frontend_review`,
and `security_gate`. Every role has 160 `train` and 40 `eval_proxy` records.
The five planned adapters are independent rank-4 `q_proj` LoRAs. `O-only` and
`Q+O` are diagnostic overlay labels only; they receive no duplicate rows and
cannot enter this primary runner.

## Quick start

Run from `D:\LLM\anchor-moe-lora-neural-swarm`. Dataset-only mode loads no
tokenizer, model, CUDA runtime, provider, or trainer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/research/run_synthetic_five_role_qonly_v2_preflight.ps1
```

Python discovery order is: explicit `-PythonExecutable`, `ANCHOR_PYTHON`,
active `CONDA_PREFIX`, repository `.venv`, `$HOME\.conda\envs\anchor-mvp`,
then a real `python` command. WindowsApps aliases that cannot answer
`--version` are rejected. Diagnose Python only with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/research/run_synthetic_five_role_qonly_v2_preflight.ps1 `
  -CheckPythonOnly
```

Expected final line:

```text
[five-role preflight] PASS: all five roles authenticated serially; no model, CUDA, provider, or training request was made.
```

Run one role directly:

```powershell
python scripts/research/prepare_synthetic_five_role_qonly_v2.py --role planner
```

The expected JSON status is
`passed_dataset_only_dry_run_training_blocked`. To run all five roles without
the helper:

```powershell
$roles = "planner","tool_policy","frontend_gen","frontend_review","security_gate"
foreach ($role in $roles) {
  python scripts/research/prepare_synthetic_five_role_qonly_v2.py --role $role
  if ($LASTEXITCODE -ne 0) { throw "preflight failed: $role" }
}
```

Optional tokenizer-only length checks remain offline and still load no model:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts/research/run_synthetic_five_role_qonly_v2_preflight.ps1 `
  -TokenizerOnly
```

Add `-PublishPreflight` only with `-TokenizerOnly`. Receipts are written to
role-isolated, no-replace directories under
`artifacts/diagnostics/qwen2_5_1_5b_synthetic_five_role_qonly_v2/<role>/preflight`.

## What the preflight proves

Before applying `--role`, it authenticates the mandatory manifest sidecar,
record and manifest schemas, both partition byte snapshots, all 1,000 records,
200 five-role bundles, the canonical stage map, bundle split, per-cell quotas,
and the 200 namespace-neutral semantic identities. It checks each bundle's
five board inventories item by item and binds each current target back to its
segment hash. Current/future segment IDs, summaries, target JSON and serialized
answers must not occur in the prompt. Filtering happens before tokenization.

The declared training numerics are BF16 compute, TF32 matrix multiplication,
micro-batch 1, sequence length 512, `use_cache=false`, and five serial Q-only
rank-4 jobs. This module emits a plan and preflight only; training execution is
intentionally absent.

## Private-tail KV boundary

The identical frozen prefix is adapter-off and read-only. Once one Q-only
expert is activated, its post-activation prompt and generated tokens belong to
that expert's append-only private tail KV. A private tail is never reused by a
different expert. After an expert commits text, that text is re-encoded into
the next shared context. This lets experts behave as independent agents without
claiming full-generation KV sharing. Ordinary in-stack Q-LoRA also cannot claim
exact KV sharing. Therefore `runtime_private_tail_materialized=false` and
`execution_authorized=false` remain hard gates.

## Common blocked results

- `five_role_fixture_identity_pending`: final dataset hashes were not locked.
- An `identity_mismatch` or SHA error: the fixture/config is stale or changed;
  rebuild and audit it, then update the complete hash set together.
- A tokenizer error: verify the configured local Qwen tokenizer directory and
  the pinned `tokenizer.json` and `tokenizer_config.json` hashes.
- Missing model weight files do not affect dataset-only or tokenizer-only
  checks. There is deliberately no training command in this consumer; do not
  treat a preflight plan as model execution.
- No usable Python: run `conda activate anchor-mvp`, set `ANCHOR_PYTHON`, or
  pass `-PythonExecutable C:\path\to\python.exe`.
- `five_role_preflight_output_exists`: published receipts never overwrite old
  evidence; use a fresh namespace or intentionally archive the old one.
- A `forbidden_*` error: causal filtering rejected the dataset before
  tokenization. Rebuild it; the CLI never prints record bodies.

This proxy is not held-out, does not satisfy the two independent 600-record
confirmation tracks, does not authorize training, and cannot support formal,
quality, physical-KV, or multi-stream claims.
