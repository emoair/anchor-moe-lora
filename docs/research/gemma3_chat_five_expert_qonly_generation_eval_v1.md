# Gemma 3 chat five-expert generation evaluation v1

This is the post-training evaluator for
`gemma3_chat_five_expert_qonly_v1`. It compares three generation arms on the
authenticated `eval_proxy` split:

1. frozen Q8 base with every adapter disabled;
2. the adapter whose role matches the record;
3. a fixed, pre-registered wrong-route adapter.

The five roles are `humor`, `serious`, `angry_style`, `tool_call`, and
`review_audit`. The wrong-route mapping is a derangement, so no role can map
back to itself.

The checked-in evaluator config binds the FINAL consumer implementation,
consumer config, manifest, and both partitions. It deliberately leaves the
training run ID and run-receipt SHA-256 as `pending`, so a clean checkout is
fail-closed until `-Bind` creates an immutable run-local config.

## Completed diagnostic result

The first bound run completed:

- training run: `gemma3-chat-q1024-20260725-052403`
- evaluation run: `eval-chat-q1024-20260725-063309`
- evaluation receipt SHA-256:
  `ea3ca5e12d7af9bd3e40a0a81bd32dc74060cb051e319377f9e55b3b8c86956b`
- receipt sidecar physical SHA-256:
  `75a9d663aba21a9f77760d2937db3d854d2316ba508360b3eb8f7d9398879c0c`

The evaluator generated 75 aggregate-only observations: 25 distinct
eval-proxy records, five roles, and three arms per record. Correct-adapter
format/proxy pass rates were `0.8`, `1.0`, and `1.0` for `humor`, `serious`,
and `angry_style`. Both `tool_call` and `review_audit` remained `0.0` in all
three arms. The macro correct-minus-base delta was `+0.44`; the macro
correct-minus-wrong-route delta was `+0.08`.

These numbers show a useful controlled style-routing signal, but they also
show that the 1,000-row diagnostic did not teach reliable tool calls or audit
objects. They are proxy results, not held-out quality or formal validation.

## One-click use

Set the project Python once if it is not already on `PATH`:

```powershell
$env:ANCHOR_TRAINING_PYTHON = "C:\path\to\python.exe"
```

Inspect the current gate without reading any dataset body:

```powershell
.\scripts\research\evaluate_gemma3_chat_five_expert_qonly_v1.ps1 -Status
```

After training finishes, create a fully bound, immutable evaluation config.
This authenticates the dataset, receipts, and adapters but does not load the
model or request the GPU:

```powershell
.\scripts\research\evaluate_gemma3_chat_five_expert_qonly_v1.ps1 `
  -Bind -TrainingRunId <training-run-id>
```

The command prints the new `bound_config.json` path. Use that path for the
model-free preflight and the fixed low-cost generation comparison:

```powershell
$Bound = "runs\gemma3_chat_five_expert_qonly_generation_eval_v1\bindings\<training-run-id>\bound_config.json"
.\scripts\research\evaluate_gemma3_chat_five_expert_qonly_v1.ps1 -Preflight -Config $Bound
.\scripts\research\evaluate_gemma3_chat_five_expert_qonly_v1.ps1 -Execute -Config $Bound
```

Both the checked-in template and every generated bound config have a mandatory
`sha256sum`-format sidecar.

The result is atomically published under
`artifacts/diagnostics/gemma3_chat_five_expert_qonly_generation_eval_v1/<run-id>/`.
Existing outputs are never replaced.

## What is authenticated

The evaluator requires all identities conjunctively:

- the final consumer config and implementation;
- the producer manifest and both physical partitions;
- the completed training run receipt and mandatory sidecar;
- all ten smoke/full phase receipts and sidecars;
- all five PEFT adapters and both adapter files for each role;
- the frozen local Gemma 3 model-file contract.

Any missing, pending, changed, or cross-bound identity stops before model
loading. The base Q8 SCB inventory and local model files are checked before and
after generation.

## Aggregate metrics

The receipt contains no prompt, target, generated answer, raw token IDs, or
per-record metrics. It reports only aggregate values:

- natural-language output-form validity and reference trigram similarity
  proxies for humor, serious, and angry style;
- tool-call JSON/closed-schema validity and evidence-field consistency;
- review-audit JSON/closed-schema validity and dependency-field consistency;
- the stable Air persona identity consistency and provider-misattribution
  absence on identity probes;
- EOS termination, repeated 4-grams, generated-token throughput, and peak
  Torch memory;
- correct-adapter minus wrong-route and correct-adapter minus base deltas.

The style metric is deliberately named a reference proxy. It is not an
independent semantic judge. Likewise, a positive wrong-route delta is a
controlled diagnostic signal, not proof of routing causality, held-out
generalization, or formal quality.

## Runtime boundary

The base is loaded with bitsandbytes INT8, skipped frozen tensors retain the
training config dtype, adapters are inference-only, decoding is greedy, and
generation uses the model's dynamic cache. This evaluator does **not** claim a
Q8 KV cache, exact full-generation KV sharing, formal authorization, or
held-out evaluation.
