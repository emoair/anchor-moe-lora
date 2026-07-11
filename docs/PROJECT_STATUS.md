# Project status and remaining work

Last evidence refresh: 2026-07-11. This file separates completed evidence from
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
  offline tests pass; the real 12B GPU smoke is still pending.
- Held-out leakage scanning currently reports zero collisions. Model binaries,
  teacher JSONL, adapters, checkpoints, runtime logs, and credentials are ignored
  and are not part of the public repository.

## Distillation audit

The current API corpus has 338 rows: plan 128, tool policy 128, frontend 51,
review 16, and security 15. All retained rows identify `kimi-for-coding`, carry SOP
provenance, contain public decision artifacts rather than retained hidden reasoning,
and preserve same-seed DAG dependencies.

The corpus is a candidate, not a finished training set:

- the final 640-row gate is only 52.81% complete;
- the persisted failure budget is exhausted (`276/200`) and cannot be resumed just
  because the provider quota window reset;
- all 15 security labels are PASS, so the corpus cannot teach BLOCK behavior;
- OpenCode plus external-Skill execution has one failed live artifact and no valid
  public outcome; current API rows use local SOPs rather than executed third-party
  Skills;
- executable frontend checks, review-repair verification, deterministic policy
  labels, semantic deduplication, and balanced-label gates are not yet enforced.

Offline P0 hardening now separates quota epochs from the durable attempt ledger,
deduplicates and quarantines repeated failures, applies deterministic Tool Policy and
Security gold labels, rejects nested hidden-reasoning fields, and adds OpenCode batch,
Skill provenance, append-only gold, and execution-heldout gates. All offline tests
pass. This does not yet prove a successful live OpenCode execution sample.

Spend the reset quota only on a single confirmed execution smoke first. API bulk
distillation remains blocked until executable frontend/review gates and the live
OpenCode promotion sequence pass.

## Prioritized work

### P0: unblock trustworthy data

1. Run one OpenCode plus audited external-Skill live execution sample and require a
   validated public outcome, isolated workspace, tests, and append-only gold record.
2. Promote only through the checked-in `1 -> 2 -> 4 -> 8` ramp after each stage meets
   its success-rate and leakage gates.
3. Add TSX parse/typecheck/build checks, review mutation-repair verification, trace
   grounding, semantic deduplication, and category/label balance gates.
4. Produce 3-6 successful independent OpenCode fixtures before treating execution
   data as trainable, even if the first single-sample smoke passes.

### P0: finish controlled training

1. Run the formal-v2 2-step GPU smoke and require peak allocated/reserved VRAM at
   or below 9GiB, finite loss, checkpoint save, and exact reload.
2. If it passes, train B, C, and D from the same frozen snapshot under the recorded
   optimizer-step and sample-exposure contract.
3. Keep A as the untrained identical Q4 base. Evaluate E only after its
   complexity-adaptive rank allocation is frozen on calibration data.
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
