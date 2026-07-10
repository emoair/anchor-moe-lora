# Gemma 4 12B resource route

This note records the resource-safe route for the frozen 12B base. It is based
only on Google, Hugging Face, and vLLM first-party documentation and model
repositories as checked on 2026-07-10. It deliberately separates training
formats from deployment formats.

## Decision

There is currently **no Google-published, pre-quantized Q4/NF4/QAT checkpoint
for the non-instruction-tuned `google/gemma-4-12B` base**. Google's official
[Gemma 4 QAT collection](https://huggingface.co/collections/google/gemma-4-qat-q4-0)
contains only `-it` 12B variants. Therefore the experiment must not substitute
an official `-it-qat-*` artifact for the frozen base: it would change both the
starting weights and the experiment being measured.

The recommended route is:

1. Keep `google/gemma-4-12B` as the canonical base.
2. For QLoRA, load that BF16 checkpoint with bitsandbytes NF4 plus double
   quantization and train only LoRA parameters.
3. Once a one-step load/train/save/reload gate succeeds on a machine with
   enough host RAM, save a local bitsandbytes 4-bit Transformers checkpoint.
   Reuse that exact locally derived checkpoint for later QLoRA loads and vLLM
   serving. Record its source revision, quantization config, and hashes.
4. Do not repeat BF16-to-NF4 in-flight conversion on the 22.8 GiB host while
   other memory-heavy jobs are running. If the one-time conversion cannot stay
   below a measured RAM limit, perform it on a machine with more RAM; this does
   not change the canonical source weights.
5. Treat bulk distillation, training, and vLLM serving as mutually exclusive
   resource phases on the 12 GiB GPU host.

The Hugging Face
[bitsandbytes guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
states that 4-bit training supports extra parameters, recommends NF4 for
training 4-bit base models, and documents serializing and reloading 4-bit
Transformers models. The
[PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
documents attaching LoRA to a bitsandbytes 4-bit base. These are the relevant
training formats; GGUF and compressed-tensors W4A16 are deployment formats.

## Official checkpoint matrix

| Model ID | Alignment status | Stored form | QLoRA training source | vLLM route | Use here |
| --- | --- | --- | --- | --- | --- |
| `google/gemma-4-12B` | Pre-trained, not `-it`; however, Google documents safety/content filtering and safety evaluation, so it must not be described as guaranteed “uncensored” | BF16 Transformers, local weight file about 23.9 GB | **Yes.** Load with bitsandbytes NF4 and train LoRA only | **Yes.** In-flight bitsandbytes, or preferably a locally saved pre-quantized bitsandbytes derivative | Canonical base |
| `google/gemma-4-12B-it` | Instruction-tuned | BF16 Transformers | Technically PEFT-compatible, but wrong starting weights for this experiment | Yes | Processor/chat-template reference only; not base weights |
| `google/gemma-4-12B-it-qat-q4_0-unquantized` | Instruction-tuned | Half-precision QAT source; despite the name, not Q4-compressed storage (official repository is about 24 GB) | It would need another training-compatible quantization step; no load-memory advantage and wrong alignment | Custom conversion/research source | Do not use |
| `google/gemma-4-12B-it-qat-q4_0-gguf` | Instruction-tuned | Q4_0 GGUF | **No** for the Transformers/PEFT QLoRA path | Local inference format; not the selected vLLM training/serving bridge | Do not use |
| `google/gemma-4-12B-it-qat-w4a16-ct` | Instruction-tuned | Compressed-tensors W4A16 | **No** as the QLoRA training source | **Yes**, optimized vLLM inference | Do not use for A/B/C base comparison |

Google's [Gemma 4 overview](https://ai.google.dev/gemma/docs/core) explicitly
maps `-qat-q4_0-gguf` to local deployment, `-qat-w4a16-ct` to vLLM/SGLang
serving, and `-qat-q4_0-unquantized` to conversion/research. The same page
reports approximately 26.7 GB for 12B BF16 inference versus 6.7 GB for Q4_0,
including 20% static-loading overhead but excluding KV cache. That is roughly a
75% static-weight-memory reduction, not a promise that a complete training or
vLLM process fits in 6.7 GB.

## Safety wording

`google/gemma-4-12B` is the correct **non-instruction-tuned** checkpoint. It is
not accurate to call it “without safety work” or guaranteed “uncensored”. The
official [12B model card](https://huggingface.co/google/gemma-4-12B) says that
the pre-training data underwent CSAM, sensitive-data, quality, and safety
filtering, and that the family underwent safety evaluations. What the base
avoids is the `-it` instruction-tuning stage; it does not erase pre-training
data policy or learned safety behavior.

This distinction matters for the security-adapter experiment: report the base
as `pretrained_non_it`, not `unaligned` or `uncensored`.

## Resource expectations on RTX 3080 Ti 12 GiB

The following are planning bounds, not measured guarantees:

- BF16 static load: Google estimates about 26.7 GB, which cannot fit in 12 GiB
  VRAM and is also dangerously close to/exceeds this host's usable RAM during a
  conversion spike.
- NF4/Q4 static weights: approximately one quarter of BF16. Google estimates
  6.7 GB for Q4_0 inference. Hugging Face documents a 4x model-memory reduction
  for bitsandbytes 4-bit loading.
- Double quantization: Hugging Face documents an additional saving of about
  0.4 bits per parameter. It helps, but activations, LoRA weights, gradients,
  CUDA kernels, temporary buffers, and allocator fragmentation remain.
- QLoRA smoke: sequence length 128-256, batch 1, rank 8 or 16, `q_proj` and
  `v_proj` only, gradient checkpointing on, cache off, and one optimizer step.
  Measure peak host RAM and VRAM from process start, including model load.
- Serving smoke: context 512-1024, one sequence, one active LoRA, rank at most
  16, eager execution, no speculative decoder, and no concurrent training.

The Google estimate explicitly excludes KV-cache growth and says fine-tuning
overhead depends on framework, sequence length, batch size, and PEFT method.
Therefore a successful 6.7 GB weight load is not itself a passing 12 GiB
training or serving gate.

### Measured Windows NF4 smoke on 2026-07-11

The real `frontend_gen`, rank-16, `q_proj`/`v_proj`, sequence-128, one-sample,
one-step smoke passed every preflight gate and loaded successfully. It ran from
02:09:56 to 03:32:54 Asia/Shanghai before being stopped by the operator because
no `global_step=1`, finite loss, or adapter directory had been produced after
about 83 minutes. This is an incomplete run, not a successful step and not an
OOM result.

During sustained compute the RTX 3080 Ti stayed at 99-100% utilization and about
74 C. Observed GPU allocation was roughly 11.95-12.07 GiB, leaving only about
21-140 MiB free at the tightest points. Host available memory recovered after
load and remained roughly 12.5-13.4 GiB. Termination released GPU use to below
1 GiB without destabilizing the system.

Consequences: the native-Windows online bitsandbytes NF4 route fails the
throughput/promotion gate even though it can load. Do not multiply this profile
to eight steps or six adapters. The next real training attempt must first use a
faster WSL2/Linux kernel path or a verified, reloadable pre-quantized
Transformers/PEFT checkpoint, then repeat the same one-step gate and record loss,
peak memory, adapter save, and adapter reload evidence.

## vLLM compatibility

Use vLLM **0.23.0 or newer** for this 12B Unified architecture and adapter route.
The [v0.23.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.23.0)
record the first stable encoder-free Gemma 4 Unified support; current
[supported-model documentation](https://docs.vllm.ai/en/latest/models/supported_models/)
lists `Gemma4UnifiedForConditionalGeneration` and states that its language
model and LoRA support are inherited from the Gemma 4 implementation.

The official vLLM
[bitsandbytes guide](https://docs.vllm.ai/en/latest/features/quantization/bnb/)
supports both in-flight 4-bit quantization and pre-quantized bitsandbytes
checkpoints. The official
[LoRA quantization example](https://docs.vllm.ai/en/latest/examples/features/lora/)
shows LoRA with `quantization="bitsandbytes"`. On this host, the locally saved
base-derived bitsandbytes artifact is preferable because it avoids repeating a
high-RAM conversion each time the server starts.

Do not mix a LoRA trained against `google/gemma-4-12B` with an official
`google/gemma-4-12B-it-qat-w4a16-ct` server. Matching tensor shapes do not make
different frozen base weights a valid or fair adapter base.

All benchmark arms must use the exact same locally serialized Q4/NF4 base,
quantization config, revision, and hashes: A runs that base at every stage, B
runs one mixed adapter on it at every stage, and C routes specialist adapters on
the same base. Changing the Q4 serialization between arms invalidates the result.

## Promotion gates

Promote from one phase to the next only when all checks pass:

1. **Load gate:** peak host RAM and VRAM remain below configured limits; no
   paging storm or WSL/system instability.
2. **Training gate:** one finite-loss step at short sequence length; adapter
   save/unload/reload succeeds; the frozen-base hash and quantization config are
   recorded.
3. **Serving gate:** vLLM loads the same base-derived quantization plus one
   adapter and completes one 512-1024-token-context request without OOM.
4. **DAG gate:** five sequential adapter stages complete with only one active
   adapter and one request in flight.
5. **Scale gate:** only after the above may data concurrency increase. Training
   and serving stay stopped during high-concurrency distillation.

If the first gate remains unstable, use `google/gemma-4-E4B` only as a pipeline
smoke substitute and label the run non-comparable. It is not a replacement for
the final 12B benchmark.
