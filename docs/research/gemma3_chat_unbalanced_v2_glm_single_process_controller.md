# GLM single-process live controller

Status: additive controller contract. The sharded-v1 consumer binding is
authenticated from four exact `100644` Git blobs at consumer commit
`60e88aa7a6c2dcdd8a3cd9f496f84d3a1ea76bb2`. Live execution remains blocked
until this controller passes independent review and is committed/pushed, and
until credential and runtime-HMAC slots are loaded in memory. It does not alter
the frozen FINAL, source overlay, four batch profiles, batch implementation, or
teacher implementation.

The controller config statically pins this controller module's canonical path
and physical SHA-256. The loader authenticates one bytes snapshot before any
consumer/Git check, credential access, provider construction, or network
operation, then rechecks the same bytes at the end of config loading.

## Purpose and boundaries

`gemma3_chat_unbalanced_v2_live_controller.py` owns one `RuntimeSecretSlots`
object for the complete `exact1 -> bounded15 -> bulk` lifecycle. The credential
is received once from an anonymous OS pipe or in-process receiver. One
runtime-random HMAC key authenticates every receipt and controller-WAL entry for
that process. Slot clearing is best effort; forensic zeroization is not claimed.

The controller rejects a non-empty output root at process start. It does not
offer or claim cross-process resume. The existing batch `--resume-execute`
surface remains unchanged and is not used here.

## Authenticated WAL and physical paths

The controller creates an immutable `O_EXCL` genesis, a same-PID/run-ID process
lock, and zero-padded immutable WAL entries. Entries form a global physical-SHA
chain and per-job chains. A dispatch event is HMAC-authenticated and fsynced
before `teacher.complete` can run.

Terminal commits are signed `GROUP_PREPARE`, frozen physical group commit,
single-snapshot group/receipt/event verification, then signed `GROUP_TERMINAL`.
`events.jsonl` and phase receipts are projections and must equal the
authenticated sequence before every replay.

Genesis signs the physical identities of the output root, alignment,
automation, group/WAL directories, append-only logs, and dataset marker. Every
runtime access rechecks the layout. Symlink, junction, or reparse substitution
must be rejected before an external write; dynamic WAL, group, status, and
kill-switch paths are checked too. Terminal events must cross-bind their HMAC
receipt, physical group, and output/rejection.

## Ramp and fallback

The physical SHA-256 identities of `smoke_exact1`, `bounded_small_c1`,
`bulk_c30`, and `bulk_c16` are locked. Exact1 must be fully green before 15;
15 must be fully green before c30.

c16 is allowed only after a non-empty, deterministically sorted inventory of
signed c30 `provider_rate_limit` events, zero uncertain jobs, and an expired
signed cooldown lease. The lease binds the event inventory digest/count, source
WAL chain tip, fallback profile, run ID, and the delay from `RateLimitError` and
the profile. `status.json` is observational and never authorizes fallback.
Waiting uses at most 60-second slices with consumer, slot, source, kill-switch,
and WAL rechecks. `provider_instability_reconciled` is not accepted.

## Model-free commands

```powershell
$env:PYTHONPATH='src'
python -m anchor_mvp.data.gemma3_chat_unbalanced_v2_live_controller --dry-run
python -m anchor_mvp.data.gemma3_chat_unbalanced_v2_live_controller --validate-only
python -m pytest -q tests/test_gemma3_chat_unbalanced_v2_live_controller.py
```

Live execution accepts no credential in argv, environment, a TTY, or a regular
file:

```powershell
<anonymous-pipe-producer> | python -m anchor_mvp.data.gemma3_chat_unbalanced_v2_live_controller --execute --credential-stdin
```

Public output contains only hashes, counts, fixed reason codes, and stage
states. It contains no credentials, runtime HMAC values, prompts, answers, or
source/heldout bodies. Claims remain diagnostic; formal training and quality
authorization remain false.
