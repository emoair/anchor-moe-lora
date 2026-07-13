# Project status and remaining work

Last evidence refresh: 2026-07-13. This file separates completed evidence from
planned work; it is not a production-readiness claim.

## Verified locally

- The checkout was migrated intact to `D:\LLM\anchor-moe-lora`: 1,102 files,
  32,511,123,597 bytes, valid Git history, and a clean worktree before formal-v2.
- A persistent Transformers/bitsandbytes NF4 checkpoint of the frozen Gemma 4 12B
  base loads in seconds. Its model footprint is 7,503,924,322 bytes.
- A real rank-16 QLoRA gate completed forward, backward, paged AdamW 8-bit update,
  adapter save, and exact reload on one RTX 3080 Ti 12GB over OCuLink PCIe 4.0 x4.
- A 128-step frontend candidate completed. It is feasibility evidence only because
  it repeatedly exposed a small 18-record corpus and showed overfitting.
- Formal-v1 completed B `mixed_all-r16` for 75 steps and C `planner-r16` for 15
  steps. C `tool_policy-r16` was terminated at step 3 after 95 minutes without a
  new optimizer step; no adapter/checkpoint was promoted from that interrupted job.
- The formal-v1 freeze contains 75 records, 15 per expert, with snapshot SHA-256
  `a6cb9a4443c1f73775a1aa815e575a2ae1026b78cc6c4df9a3fb440670c19320`.
- Formal-v2 implements the same frozen snapshot and sample exposure through a
  manual active-label loop with a hard 9GiB peak-VRAM gate. All 12 dry runs and
  offline tests pass, and its two-step real 12B CUDA probe completed within the
  steady-state gate; long sequential training is not yet promoted.
- Formal-v3 now binds B/C/D/E/F, smoke, and probe only to a future immutable
  `anchor.training-snapshot.v2` under `artifacts/formal_v3/dataset`. The preflight
  independently verifies that snapshot and the actual reloadable bitsandbytes NF4
  directory; it rejects growing automation output, stale/partial adapters, and
  unbound adaptive-allocation manifests before GPU work.
- Held-out leakage scanning currently reports zero collisions. Model binaries,
  teacher JSONL, adapters, checkpoints, runtime logs, and credentials are ignored
  and are not part of the public repository.

## Distillation audit

The current authoritative candidate is `automated_v3`, not the older 338-row v2
corpus. Its first live fast-10 quota window made 502 teacher requests and recorded
938,855 output tokens before the provider reported quota exhaustion. Offline
repartitioning now yields 260 strict-gold rows and no retained negative or reject
rows:

- plan: 121 / 128 minimum;
- tool policy: 121 / 128 minimum, with APPROVE 42, ESCALATE 40, and BLOCK 39;
- frontend: 18 / 128 minimum;
- review: 0 / 128 minimum;
- security: 0 / 128 minimum.

The 40 historical Tool Policy teacher disagreements are retained as explicit audit
evidence but use deterministic-oracle-normalized supervision; they are not presented
as teacher agreement and do not retain contrary reasoning. The corrected TSX gate
accepts all 18 existing frontend artifacts. Held-out leakage remains PASS with zero
collisions.

The v2 partition contract separates a 192-row raw collection target from a 128-row
minimum gold floor for each expert. It collects first and partitions afterward:
isolated unsafe, secret-bearing, or malformed model rows are content-free quarantined,
while gold integrity, held-out leakage, coverage, and label balance remain fail-closed
corpus gates. The current partition is still `training_ready=false`; no formal-v3
snapshot or training JSONL has been published.

One live OpenCode collect sample has a public completed outcome with build, test, and
lint PASS. It remains outside strict execution gold because the stricter tool-result
and final-diff acceptance contract did not pass. Collect data and strict execution
data therefore remain separate evidence classes.

## Prioritized work

### P0: unblock trustworthy data

1. Resume `automation.full_v3.fast.yaml` in a fresh quota epoch. Filling the raw
   target from the current cursor requires at most 764 additional calls under the
   present five-stage plan; the actual stop condition remains gold coverage and label
   quotas, not raw perfection.
2. Repartition offline and require all five experts to reach 128 strict-gold records,
   Tool Policy and Security label quotas to pass, and held-out/gold-integrity gates to
   remain clean.
3. Publish the immutable formal-v3 snapshot only through
   `prepare_full_v3_snapshot.py`; exit code 3 means keep collecting and forbids
   training.
4. Produce additional independent OpenCode execution fixtures before using execution
   trajectories as strict training gold. Do not promote collect-only negatives into
   that class.

### P0: finish controlled training

1. Preserve the completed formal-v2 2-step GPU smoke evidence: peak allocated/
   reserved VRAM within the steady-state gate, finite loss, checkpoint save, and
   exact reload.
2. After the formal-v3 snapshot exists, run one-step smoke and two-step probe before
   starting the five independent rank-16 C adapters. The launcher is serial and
   resumes only at verified expert-job boundaries; safety checkpoints are explicitly
   warm-start-only, not exact optimizer/RNG resume.
3. Train B, C, and D from the same snapshot and 640-sample exposure contract. Keep A
   as the untrained identical Q4 base. Train E/F only after their shared adaptive
   mechanism and allocations are frozen on calibration data; F must exactly match
   B's total adapter parameter budget.
4. Bind every benchmark arm to actual checkpoint metadata, dataset/schedule SHA,
   tokenizer hash, optimizer, TF32/BF16/NF4 settings, steps, and supervised tokens.

### P1: serving and benchmark

1. Complete the five specialist adapters before claiming the routed workflow runs.
2. Run the frozen held-out benchmark for A/B/C/D and report A as index 100 with
   Pass@1, security TPR/FPR, tool-policy accuracy, latency, tokens, VRAM, and errors.
3. For llama.cpp, add Gemma4 Unified conversion support, produce a same-source 12B
   GGUF, convert PEFT adapters to GGUF LoRA, add request-level LoRA mapping, and build
   a clean CUDA `llama-server`. The existing Vulkan server contains a local Gemma4
   softcap modification and is not a controlled baseline.
4. Compare llama.cpp and vLLM only after both load the same model/adapter contract.

## Publication boundary

The public repository demonstrates a tested training and orchestration framework.
It does not ship model weights, distilled teacher data, API credentials, benchmark
answers, or claims that formal-v2 quality/performance evaluation is complete.
