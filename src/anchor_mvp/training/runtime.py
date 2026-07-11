"""Heavyweight Gemma 4 QLoRA runtime, imported only for an explicit execution."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import checkpoint_metadata, write_json
from .progress import TrainingProgress


class _JsonlMapDataset:
    """Minimal random-access JSONL dataset without Arrow-backed copies.

    Only byte offsets are retained in memory. Records are decoded on demand,
    which keeps distilled code payloads out of a second in-memory table and
    makes the smoke gate stop indexing as soon as its single sample is found.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        max_records: int | None = None,
    ) -> None:
        if max_records is not None and max_records < 1:
            raise ValueError("max_records must be positive when provided")
        self._index: list[tuple[Path, int, int]] = []
        for path in paths:
            with path.open("rb") as handle:
                line_number = 0
                while True:
                    offset = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    line_number += 1
                    if not raw.strip():
                        continue
                    self._index.append((path, offset, line_number))
                    if max_records is not None and len(self._index) >= max_records:
                        return

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self._index)
        if index < 0 or index >= len(self._index):
            raise IndexError(index)
        path, offset, line_number = self._index[index]
        with path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.readline()
        try:
            record = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid JSONL record at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"JSONL record at {path}:{line_number} must be an object"
            )
        return record


def _make_text_collator(processor: Any, max_length: int):
    """Build an assistant-preserving completion-only text collator.

    Gemma 4's generation prompt contains thought-channel tokens that are not
    present in an ordinary assistant turn, so prompt token counts are not a
    reliable assistant boundary. Locate the final assistant content with fast
    tokenizer offsets, then keep recent prompt context plus the beginning of
    the completion when an example is longer than the training window.
    """

    def collate(examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        rows: list[tuple[list[int], list[int]]] = []
        for example in examples:
            messages = example["messages"]
            if not messages or messages[-1].get("role") != "assistant":
                raise RuntimeError(
                    "completion-only training requires a final assistant message"
                )
            assistant_content = messages[-1].get("content")
            if not isinstance(assistant_content, str) or not assistant_content.strip():
                raise RuntimeError(
                    "completion-only training requires non-empty assistant content"
                )
            full_text = processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            ).strip()
            target_text = assistant_content.strip()
            content_start = full_text.rfind(target_text)
            if content_start < 0:
                raise RuntimeError(
                    "assistant content was not found in the rendered chat template"
                )
            content_end = content_start + len(target_text)
            encoded = processor.tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            input_ids = [int(value) for value in encoded["input_ids"]]
            offsets = [tuple(value) for value in encoded["offset_mapping"]]
            target_indices = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > content_start and start < content_end
            ]
            if not target_indices:
                raise RuntimeError("assistant content produced no target tokens")
            target_start = target_indices[0]
            completion = input_ids[target_start:]
            completion_limit = max(1, (max_length * 3) // 4)
            completion = completion[:completion_limit]
            prompt_budget = max_length - len(completion)
            prompt = input_ids[max(0, target_start - prompt_budget) : target_start]
            selected = prompt + completion
            labels = [-100] * len(prompt) + completion
            if not completion or all(value == -100 for value in labels):
                raise RuntimeError(
                    "assistant-preserving truncation produced no target tokens"
                )
            rows.append((selected, labels))

        import torch

        sequence_length = max(len(input_ids) for input_ids, _ in rows)
        padding_side = getattr(processor.tokenizer, "padding_side", "right")
        if padding_side not in ("left", "right"):
            raise RuntimeError(f"unsupported tokenizer padding_side={padding_side!r}")
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError(
                "completion-only collator requires a tokenizer pad token"
            )
        padded_ids: list[list[int]] = []
        padded_labels: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for input_ids, labels in rows:
            padding = sequence_length - len(input_ids)
            if padding_side == "left":
                padded_ids.append([pad_id] * padding + input_ids)
                padded_labels.append([-100] * padding + labels)
                attention_masks.append([0] * padding + [1] * len(input_ids))
            else:
                padded_ids.append(input_ids + [pad_id] * padding)
                padded_labels.append(labels + [-100] * padding)
                attention_masks.append([1] * len(input_ids) + [0] * padding)
        input_tensor = torch.tensor(padded_ids, dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "mm_token_type_ids": torch.zeros_like(input_tensor),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    return collate


def _cuda_memory_detail(torch: Any, device: Any) -> dict[str, int]:
    if getattr(device, "type", "cpu") != "cuda" or not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_mib": int(free_bytes // (1024 * 1024)),
        "total_mib": int(total_bytes // (1024 * 1024)),
        "allocated_mib": int(torch.cuda.memory_allocated(device) // (1024 * 1024)),
        "reserved_mib": int(torch.cuda.memory_reserved(device) // (1024 * 1024)),
        "peak_allocated_mib": int(
            torch.cuda.max_memory_allocated(device) // (1024 * 1024)
        ),
    }


def _assert_trainable_scope(model: Any, torch: Any) -> int:
    trainable = 0
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable += parameter.numel()
        if "lora_" not in name:
            unexpected.append(name)
        # The requested experiment keeps the adapter itself in BF16.
        if parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(torch.bfloat16)
    if not trainable:
        raise RuntimeError("LoRA attachment produced zero trainable parameters")
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise RuntimeError(
            f"base freeze invariant failed; unexpected trainable parameters: {preview}"
        )
    return trainable


def _prepare_gemma4_for_kbit_training(model: Any, torch: Any) -> dict[str, int]:
    """Freeze the base without PEFT's blanket BF16-to-FP32 cast.

    PEFT's generic helper promotes every non-Params4bit tensor to FP32. Gemma 4's
    large frozen embedding then consumes roughly two extra GiB. For this BF16
    experiment, only normalization parameters need the FP32 stability promotion;
    frozen embeddings and the tied LM head remain in their original BF16 dtype.
    """

    frozen_parameters = 0
    norm_parameters_fp32 = 0
    preserved_bf16_parameters = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        frozen_parameters += parameter.numel()
        if (
            "norm" in name.casefold()
            and parameter.dtype in {torch.float16, torch.bfloat16}
            and parameter.__class__.__name__ != "Params4bit"
        ):
            parameter.data = parameter.data.to(torch.float32)
            norm_parameters_fp32 += parameter.numel()
        elif parameter.dtype == torch.bfloat16:
            preserved_bf16_parameters += parameter.numel()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return {
        "frozen_parameters": int(frozen_parameters),
        "norm_parameters_fp32": int(norm_parameters_fp32),
        "preserved_bf16_parameters": int(preserved_bf16_parameters),
    }


def _capture_probes(
    model: Any,
    processor: Any,
    cases: Sequence[Mapping[str, Any]],
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Capture one-forward, logits-only evidence for held-out cases.

    Generation is intentionally excluded: a smoke gate only needs evidence
    that the adapter changes the next-token distribution and survives reload.
    Avoiding KV-cache growth here materially lowers peak VRAM and host pressure.
    """

    public: list[dict[str, Any]] = []
    logits_by_id: dict[str, Any] = {}
    was_training = model.training
    old_use_cache = getattr(model.config, "use_cache", False)
    model.eval()
    model.config.use_cache = False
    try:
        for case in cases:
            text = processor.apply_chat_template(
                [{"role": "user", "content": case["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            encoded = processor(text=[text], return_tensors="pt")
            device = model.device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            with torch.inference_mode():
                # Gemma 4 supports logits_to_keep, avoiding allocation of a
                # prompt_length x vocabulary logits tensor.
                output = model(
                    **inputs,
                    use_cache=False,
                    logits_to_keep=1,
                )
                next_logits = output.logits[0, -1].detach().float().cpu()
            del output
            probabilities = torch.softmax(next_logits, dim=-1)
            top_k = min(5, int(probabilities.numel()))
            top_probabilities, top_token_ids = torch.topk(probabilities, k=top_k)
            logits_by_id[str(case["id"])] = next_logits
            public.append(
                {
                    "id": case["id"],
                    "expert": case["expert"],
                    "probe": "next_token_logits",
                    "top_token_ids": [int(value) for value in top_token_ids.tolist()],
                    "top_token_probabilities": [
                        round(float(value), 8) for value in top_probabilities.tolist()
                    ],
                    "next_logits_sha256": hashlib.sha256(
                        next_logits.numpy().tobytes()
                    ).hexdigest(),
                }
            )
    finally:
        model.config.use_cache = old_use_cache
        if was_training:
            model.train()
    return public, logits_by_id


def _compare_probes(
    before_public: Sequence[Mapping[str, Any]],
    before_logits: Mapping[str, Any],
    after_public: Sequence[Mapping[str, Any]],
    after_logits: Mapping[str, Any],
    torch: Any,
) -> list[dict[str, Any]]:
    before_by_id = {str(item["id"]): item for item in before_public}
    comparisons: list[dict[str, Any]] = []
    for after in after_public:
        identifier = str(after["id"])
        before = before_by_id[identifier]
        delta = torch.max(
            torch.abs(after_logits[identifier] - before_logits[identifier])
        ).item()
        comparisons.append(
            {
                "id": identifier,
                "top_token_changed": before["top_token_ids"][0]
                != after["top_token_ids"][0],
                "max_abs_next_logit_delta": float(delta),
                "before_top_token_ids": before["top_token_ids"],
                "after_top_token_ids": after["top_token_ids"],
                "before_logits_sha256": before["next_logits_sha256"],
                "after_logits_sha256": after["next_logits_sha256"],
            }
        )
    return comparisons


def _run_one_step_smoke(
    model: Any,
    processor: Any,
    record: Mapping[str, Any],
    training: Mapping[str, Any],
    torch: Any,
    reporter: TrainingProgress | None = None,
) -> tuple[int, float]:
    """Run the smoke update without importing Trainer, TRL, Datasets, or Arrow."""

    if int(training["max_steps"]) != 1:
        raise RuntimeError("smoke-gate runtime requires training.max_steps=1")
    if int(training["gradient_accumulation_steps"]) != 1:
        raise RuntimeError("smoke-gate runtime requires gradient_accumulation_steps=1")

    dataset = [record]
    result = _run_low_memory_training(
        model,
        processor,
        dataset,
        training,
        torch,
        reporter,
    )
    return int(result["global_step"]), float(result["train_loss"])


def _sample_schedule(
    dataset_size: int,
    *,
    max_steps: int,
    gradient_accumulation_steps: int,
    seed: int,
) -> list[int]:
    """Return a deterministic, epoch-shuffled schedule with exact exposure.

    The schedule is deliberately independent of PyTorch/Trainer versions. For
    the frozen formal datasets, steps * accumulation equals four complete
    epochs, so every record is exposed exactly four times while epoch order is
    reproducibly shuffled.
    """

    if dataset_size < 1:
        raise RuntimeError("low-memory training dataset has no records")
    if max_steps < 1 or gradient_accumulation_steps < 1:
        raise RuntimeError("training steps and gradient accumulation must be positive")
    required = max_steps * gradient_accumulation_steps
    rng = random.Random(seed)
    schedule: list[int] = []
    while len(schedule) < required:
        epoch = list(range(dataset_size))
        rng.shuffle(epoch)
        schedule.extend(epoch)
    return schedule[:required]


def _schedule_sha256(schedule: Sequence[int]) -> str:
    encoded = json.dumps(list(schedule), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _assert_training_peak_within_budget(
    torch: Any,
    device: Any,
    *,
    maximum_peak_vram_gib: float,
) -> dict[str, int]:
    detail = _cuda_memory_detail(torch, device)
    if not detail:
        return detail
    allocated_peak_bytes = int(torch.cuda.max_memory_allocated(device))
    reserved_peak_bytes = int(torch.cuda.max_memory_reserved(device))
    peak_bytes = max(allocated_peak_bytes, reserved_peak_bytes)
    limit_bytes = int(maximum_peak_vram_gib * 1024**3)
    if peak_bytes > limit_bytes:
        raise RuntimeError(
            "formal-v2 hard VRAM gate exceeded: "
            f"peak_allocated_gib={allocated_peak_bytes / 1024**3:.3f} "
            f"peak_reserved_gib={reserved_peak_bytes / 1024**3:.3f} "
            f"limit_gib={maximum_peak_vram_gib:.3f}"
        )
    return detail


def _save_adapter_safety_checkpoint(
    model: Any,
    checkpoint_root: Path,
    *,
    step: int,
    micro_step: int,
    run_id: str,
    seed: int,
    sample_schedule_sha256: str,
) -> Path:
    """Atomically publish adapter-only crash salvage for a manual run.

    Optimizer, scheduler, scaler, and RNG state are intentionally not included.
    Loading this directory is a warm start from LoRA weights, not an exact
    continuation of the interrupted optimizer trajectory.
    """

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    name = f"checkpoint-step-{step:06d}-{run_id[:12]}"
    destination = checkpoint_root / name
    temporary = checkpoint_root / f".{name}.tmp"
    if destination.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite safety checkpoint: {destination}")
    temporary.mkdir()
    model.save_pretrained(str(temporary), safe_serialization=True)
    write_json(
        temporary / "safety_checkpoint.json",
        {
            "schema_version": "anchor.adapter-safety-checkpoint.v1",
            "global_step": step,
            "micro_step": micro_step,
            "seed": seed,
            "sample_schedule_sha256": sample_schedule_sha256,
            "resume_capability": "adapter_weights_warm_start_only",
            "optimizer_state_saved": False,
            "scheduler_state_saved": False,
            "rng_state_saved": False,
            "continuation_warning": (
                "Do not claim exact optimizer resume; load the adapter weights and "
                "restart optimizer, scheduler, and sample scheduling explicitly."
            ),
        },
    )
    temporary.replace(destination)
    return destination


def _run_low_memory_training(
    model: Any,
    processor: Any,
    dataset: Any,
    training: Mapping[str, Any],
    torch: Any,
    reporter: TrainingProgress | None = None,
    safety_checkpoint_root: Path | None = None,
) -> dict[str, Any]:
    """Train with the proven active-label manual path and bounded VRAM.

    This avoids Trainer/TRL/Arrow allocations, keeps one micro-batch resident,
    and implements optimizer-step gradient accumulation explicitly.
    """

    max_steps = int(training["max_steps"])
    accumulation = int(training["gradient_accumulation_steps"])
    seed = int(training.get("seed", 0))
    schedule = _sample_schedule(
        len(dataset),
        max_steps=max_steps,
        gradient_accumulation_steps=accumulation,
        seed=seed,
    )
    schedule_sha256 = _schedule_sha256(schedule)
    maximum_peak_vram_gib = float(training.get("maximum_training_peak_vram_gib", 9.0))
    if maximum_peak_vram_gib <= 0:
        raise RuntimeError("maximum_training_peak_vram_gib must be positive")

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer_name = str(training.get("optim", "adamw_torch"))
    if optimizer_name == "paged_adamw_8bit":
        from bitsandbytes.optim import PagedAdamW8bit  # type: ignore[import-not-found]

        optimizer = PagedAdamW8bit(trainable, lr=float(training["learning_rate"]))
    else:
        optimizer = torch.optim.AdamW(trainable, lr=float(training["learning_rate"]))
    if reporter:
        reporter.emit("optimizer", "initialized", detail={"name": optimizer_name})
    collator = _make_text_collator(processor, int(training["max_seq_length"]))
    device = model.device
    model.train()
    model.config.use_cache = False
    device_type = getattr(device, "type", "cuda")
    warmup_steps = max(
        0, math.ceil(max_steps * float(training.get("warmup_ratio", 0.0)))
    )
    scheduler_name = str(training.get("lr_scheduler_type", "constant_with_warmup"))
    if scheduler_name != "constant_with_warmup":
        raise RuntimeError("manual low-memory runtime requires constant_with_warmup")

    def lr_scale(step_index: int) -> float:
        if warmup_steps and step_index < warmup_steps:
            return float(step_index + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    if reporter:
        reporter.emit(
            "manual_trainer",
            "started",
            step=0,
            detail={
                "engine": "manual_active_labels_v2",
                "optimizer": optimizer_name,
                "gradient_accumulation_steps": accumulation,
                "dataset_records": len(dataset),
                "sample_exposures": len(schedule),
                "sample_schedule_sha256": schedule_sha256,
                "maximum_training_peak_vram_gib": maximum_peak_vram_gib,
                "checkpoint_resume_capability": "adapter_weights_warm_start_only",
            },
        )
    _assert_training_peak_within_budget(
        torch, device, maximum_peak_vram_gib=maximum_peak_vram_gib
    )
    step_losses: list[float] = []
    micro_step = 0
    for global_step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for accumulation_index in range(1, accumulation + 1):
            record = dataset[schedule[micro_step]]
            micro_step += 1
            batch = collator([record])
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            model_inputs = dict(batch)
            labels = model_inputs["labels"]
            shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
            active_positions = torch.nonzero(
                (shifted != -100).any(dim=0), as_tuple=False
            ).flatten()
            if active_positions.numel() == 0:
                raise RuntimeError("low-memory sample has no supervised tokens")
            model_inputs["logits_to_keep"] = active_positions
            model_inputs["shift_labels"] = shifted.index_select(-1, active_positions)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                output = model(**model_inputs, use_cache=False)
                raw_loss = output.loss
                scaled_loss = raw_loss / accumulation
            loss_value = float(raw_loss.detach().float().cpu().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"manual low-memory runtime produced non-finite loss: {loss_value}"
                )
            accumulated_loss += loss_value
            scaled_loss.backward()
            memory_detail = _assert_training_peak_within_budget(
                torch, device, maximum_peak_vram_gib=maximum_peak_vram_gib
            )
            minimum_free_mib = int(training.get("minimum_backward_free_vram_mib", 0))
            if memory_detail and memory_detail["free_mib"] < minimum_free_mib:
                raise RuntimeError(
                    "insufficient lossless VRAM headroom after micro-backward: "
                    f"free_mib={memory_detail['free_mib']} required_mib={minimum_free_mib}"
                )
            if reporter:
                reporter.emit(
                    "micro_step",
                    "completed",
                    step=global_step,
                    loss=loss_value,
                    detail={"accumulation_index": accumulation_index, **memory_detail},
                )
            del batch, model_inputs, output, raw_loss, scaled_loss
        averaged_loss = accumulated_loss / accumulation
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        memory_detail = _assert_training_peak_within_budget(
            torch, device, maximum_peak_vram_gib=maximum_peak_vram_gib
        )
        step_losses.append(averaged_loss)
        if reporter:
            reporter.emit(
                "optimizer_step",
                "completed",
                step=global_step,
                loss=averaged_loss,
                detail={"learning_rate": scheduler.get_last_lr()[0], **memory_detail},
            )
        save_steps = int(training.get("save_steps", 0))
        if (
            safety_checkpoint_root is not None
            and save_steps > 0
            and global_step % save_steps == 0
        ):
            run_id = reporter.run_id if reporter is not None else "unreported"
            if reporter:
                reporter.emit(
                    "adapter_safety_checkpoint",
                    "started",
                    step=global_step,
                    loss=averaged_loss,
                    detail={"resume_capability": "adapter_weights_warm_start_only"},
                )
            checkpoint_path = _save_adapter_safety_checkpoint(
                model,
                safety_checkpoint_root,
                step=global_step,
                micro_step=micro_step,
                run_id=run_id,
                seed=seed,
                sample_schedule_sha256=schedule_sha256,
            )
            if reporter:
                reporter.emit(
                    "adapter_safety_checkpoint",
                    "completed",
                    step=global_step,
                    loss=averaged_loss,
                    detail={
                        "path": str(checkpoint_path),
                        "resume_capability": "adapter_weights_warm_start_only",
                        "optimizer_state_saved": False,
                        "scheduler_state_saved": False,
                    },
                )
    train_loss = sum(step_losses) / len(step_losses)
    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0
    if getattr(device, "type", "cpu") == "cuda" and torch.cuda.is_available():
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
    result = {
        "global_step": max_steps,
        "train_loss": train_loss,
        "micro_steps": micro_step,
        "dataset_records": len(dataset),
        "sample_exposures": len(schedule),
        "sample_schedule_sha256": schedule_sha256,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "maximum_training_peak_vram_gib": maximum_peak_vram_gib,
    }
    if reporter:
        reporter.emit(
            "manual_trainer",
            "completed",
            step=max_steps,
            loss=train_loss,
            detail=result,
        )
    del scheduler, optimizer
    return result


def _train_adapter_impl(
    config: Mapping[str, Any],
    *,
    dataset_paths: Sequence[Path],
    output_dir: Path,
    allow_model_download: bool,
    manifest: Mapping[str, Any],
    smoke_heldout_cases: Sequence[Mapping[str, Any]] | None = None,
    reporter: TrainingProgress,
) -> dict[str, Any]:
    """Run one adapter SFT job. This is the only function that imports ML stacks."""

    reporter.emit("runtime_imports", "started")
    import torch
    from peft import (  # type: ignore[import-not-found]
        LoraConfig,
        PeftModel,
        get_peft_model,
    )
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForMultimodalLM,
        AutoProcessor,
        BitsAndBytesConfig,
        TrainerCallback,
    )

    reporter.emit("runtime_imports", "completed")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to train a 12B QLoRA job on CPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected profile requires CUDA BF16 support")
    allow_tf32 = bool(config["training"].get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    reporter.emit(
        "cuda_math",
        "configured",
        detail={"allow_tf32": allow_tf32, "primary_compute_dtype": "bfloat16"},
    )

    minimum_gib = float(config["guardrails"].get("minimum_gpu_memory_gib", 11.0))
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gib < minimum_gib:
        raise RuntimeError(
            f"GPU has {total_gib:.2f} GiB, below configured minimum {minimum_gib:.2f} GiB"
        )
    torch.cuda.reset_peak_memory_stats(0)

    quant = config["quantization"]
    load_strategy = config["model"]["load_strategy"]
    model_id = config["model"]["id"]
    revision = config["model"].get("revision")
    config_path = Path(str(config["_config_path"]))
    project_root = (
        config_path.parent / config.get("paths", {}).get("project_root", "../..")
    ).resolve()
    local_model_path = (project_root / config["model"]["local_path"]).resolve()
    model_source = str(local_model_path) if local_model_path.is_dir() else model_id
    common_load_args = {
        "revision": revision if model_source == model_id else None,
        "local_files_only": not allow_model_download,
        "trust_remote_code": bool(config["model"].get("trust_remote_code", False)),
    }
    common_load_args = {
        key: value for key, value in common_load_args.items() if value is not None
    }
    processor_args = dict(common_load_args)
    processor_args["revision"] = config["model"].get("processor_revision")
    processor_args = {
        key: value for key, value in processor_args.items() if value is not None
    }
    reporter.emit("processor_load", "started")
    processor = AutoProcessor.from_pretrained(
        config["model"]["processor_id"], **processor_args
    )
    reporter.emit("processor_load", "completed")
    model_load_args: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": config["model"].get("attention_implementation", "sdpa"),
        # Keep placement honest: do not silently make the 12 GB run appear to
        # fit through CPU/disk offload and then compare its latency as a GPU run.
        "device_map": {"": 0},
        **common_load_args,
    }
    if load_strategy == "bnb_nf4_online":
        model_load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quant["double_quant"],
            bnb_4bit_quant_type=quant["quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage={
                "uint8": torch.uint8,
                "bfloat16": torch.bfloat16,
            }[str(quant["quant_storage_dtype"])],
        )
    reporter.emit(
        "model_load",
        "started",
        detail={
            "strategy": str(load_strategy),
            "source": "local" if local_model_path.is_dir() else "hub",
        },
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_source,
        **model_load_args,
    )
    reporter.emit("model_load", "completed")
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError(
            "loaded checkpoint is not a training-compatible bitsandbytes 4-bit model; "
            "use bnb_nf4_online for a BF16/FP16 Transformers checkpoint or provide "
            "a PEFT-compatible prequantized 4-bit checkpoint"
        )
    model.config.use_cache = False
    reporter.emit("kbit_prepare", "started")
    preparation = _prepare_gemma4_for_kbit_training(model, torch)
    reporter.emit(
        "kbit_prepare",
        "completed",
        detail={**preparation, **_cuda_memory_detail(torch, model.device)},
    )
    gc.collect()
    torch.cuda.empty_cache()
    reporter.emit(
        "post_load_cleanup",
        "completed",
        detail=_cuda_memory_detail(torch, model.device),
    )

    lora = config["lora"]
    lora_kwargs: dict[str, Any] = {
        "r": lora["rank"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "bias": lora["bias"],
        "task_type": lora["task_type"],
    }
    # PEFT has an architecture-specific Gemma 4 default. An explicit list is
    # still supported for reproducible target-module ablations.
    if lora.get("target_modules"):
        lora_kwargs["target_modules"] = lora["target_modules"]
    reporter.emit("lora_attach", "started")
    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    trainable_parameters = _assert_trainable_scope(model, torch)
    reporter.emit(
        "lora_attach",
        "completed",
        detail={"trainable_parameters": int(trainable_parameters)},
    )

    pre_public: list[dict[str, Any]] = []
    pre_logits: dict[str, Any] = {}
    if smoke_heldout_cases is not None:
        if not smoke_heldout_cases:
            raise RuntimeError(
                "smoke-gate needs at least one held-out case for the selected expert"
            )
        reporter.emit("pre_probe", "started")
        pre_public, pre_logits = _capture_probes(
            model, processor, smoke_heldout_cases, torch
        )
        reporter.emit("pre_probe", "completed")
        if training := config.get("training"):
            if (
                training.get("empty_cache_after_probe") is True
                and torch.cuda.is_available()
            ):
                gc.collect()
                torch.cuda.empty_cache()
                reporter.emit(
                    "memory_cleanup",
                    "completed",
                    detail=_cuda_memory_detail(torch, model.device),
                )

    training = config["training"]
    # Model loading and WDDM allocator cleanup may temporarily reserve more
    # memory than the steady-state training step. Promotion evidence must use
    # the train/save/reload window, not stale loader high-water marks.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
        reporter.emit(
            "training_peak_reset",
            "completed",
            detail=_cuda_memory_detail(torch, model.device),
        )
    trainer: Any | None = None
    manual_training: dict[str, Any] | None = None
    if smoke_heldout_cases is not None:
        # This path must stay independent of the Arrow/TRL stack: it is the
        # resource-safety gate that runs before any full training job.
        dataset = _JsonlMapDataset(dataset_paths, max_records=1)
        if len(dataset) < 1:
            raise RuntimeError("smoke-gate dataset has no records")
        global_step, train_loss = _run_one_step_smoke(
            model, processor, dataset[0], training, torch, reporter
        )
    elif training.get("runtime_engine") == "manual_active_labels_v2":
        dataset = _JsonlMapDataset(dataset_paths)
        manual_training = _run_low_memory_training(
            model,
            processor,
            dataset,
            training,
            torch,
            reporter,
            safety_checkpoint_root=output_dir / "safety-checkpoints",
        )
        global_step = int(manual_training["global_step"])
        train_loss = float(manual_training["train_loss"])
    else:
        # The full training route retains TRL for now. It is never imported by
        # smoke-gate, so a low-memory machine is screened before Arrow tables
        # or the Trainer stack can be initialized.
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

        # Domain records intentionally carry different structured input/output
        # payloads. Arrow attempts to unify those nested schemas for mixed_all
        # even though training consumes only the common messages field. The
        # lightweight map dataset preserves each canonical record and lets the
        # collator project exactly the fields used by SFT.
        dataset = _JsonlMapDataset(dataset_paths)
        output_dir.mkdir(parents=True, exist_ok=True)
        args = SFTConfig(
            output_dir=str(output_dir),
            run_name=config["run_name"],
            max_length=training["max_seq_length"],
            max_steps=training["max_steps"],
            per_device_train_batch_size=training["per_device_train_batch_size"],
            gradient_accumulation_steps=training["gradient_accumulation_steps"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=training["learning_rate"],
            warmup_ratio=training["warmup_ratio"],
            lr_scheduler_type=training["lr_scheduler_type"],
            optim=training["optim"],
            logging_steps=training["logging_steps"],
            save_steps=training["save_steps"],
            save_total_limit=training["save_total_limit"],
            bf16=True,
            fp16=False,
            report_to=training.get("report_to", "none"),
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            seed=training["seed"],
        )

        class _ProgressCallback(TrainerCallback):
            def on_log(
                self,
                args: Any,
                state: Any,
                control: Any,
                logs: Any = None,
                **kwargs: Any,
            ) -> None:
                values = logs if isinstance(logs, Mapping) else {}
                raw_loss = values.get("loss")
                loss = float(raw_loss) if isinstance(raw_loss, (int, float)) else None
                reporter.emit(
                    "trainer_log", "running", step=int(state.global_step), loss=loss
                )

            def on_step_end(
                self, args: Any, state: Any, control: Any, **kwargs: Any
            ) -> None:
                reporter.emit("trainer_step", "completed", step=int(state.global_step))

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=_make_text_collator(processor, training["max_seq_length"]),
            processing_class=processor,
            callbacks=[_ProgressCallback()],
        )
        reporter.emit("trainer", "started", step=0)
        result = trainer.train()
        global_step = int(trainer.state.global_step)
        train_loss = result.metrics.get("train_loss")
        reporter.emit(
            "trainer",
            "completed",
            step=global_step,
            loss=float(train_loss) if isinstance(train_loss, (int, float)) else None,
        )
    post_public: list[dict[str, Any]] = []
    post_logits: dict[str, Any] = {}
    if smoke_heldout_cases is not None:
        reporter.emit("post_probe", "started", step=global_step, loss=train_loss)
        post_public, post_logits = _capture_probes(
            model, processor, smoke_heldout_cases, torch
        )
        reporter.emit("post_probe", "completed", step=global_step, loss=train_loss)
    peak_allocated_gib = torch.cuda.max_memory_allocated(0) / 1024**3
    peak_reserved_gib = torch.cuda.max_memory_reserved(0) / 1024**3
    reporter.emit("adapter_save", "started", step=global_step, loss=train_loss)
    output_dir.mkdir(parents=True, exist_ok=True)
    if trainer is None:
        model.save_pretrained(str(output_dir))
    else:
        trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir / "processor"))
    reporter.emit("adapter_save", "completed", step=global_step, loss=train_loss)

    smoke_evidence: dict[str, Any] | None = None
    if smoke_heldout_cases is not None:
        pre_post = _compare_probes(
            pre_public, pre_logits, post_public, post_logits, torch
        )
        adapter_files = [
            path.name
            for path in output_dir.iterdir()
            if path.name
            in {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}
        ]
        reload_error: str | None = None
        reloaded_public: list[dict[str, Any]] = []
        reload_comparison: list[dict[str, Any]] = []
        try:
            reporter.emit(
                "adapter_reload", "started", step=global_step, loss=train_loss
            )
            if trainer is not None:
                del trainer
            base_model = model.unload()
            reloaded_model = PeftModel.from_pretrained(
                base_model,
                str(output_dir),
                is_trainable=False,
                autocast_adapter_dtype=False,
            )
            reloaded_public, reloaded_logits = _capture_probes(
                reloaded_model, processor, smoke_heldout_cases, torch
            )
            reload_comparison = _compare_probes(
                post_public, post_logits, reloaded_public, reloaded_logits, torch
            )
            reporter.emit(
                "adapter_reload", "completed", step=global_step, loss=train_loss
            )
        except Exception as exc:  # evidence must survive a reload failure
            reload_error = f"{type(exc).__name__}: {exc}"
            reporter.emit(
                "adapter_reload",
                "failed",
                step=global_step,
                loss=train_loss,
                detail={"error_type": type(exc).__name__},
            )

        checks = {
            "one_sample_one_step": global_step == 1 and len(dataset) == 1,
            "loss_finite": isinstance(train_loss, (int, float))
            and math.isfinite(train_loss),
            "peak_vram_within_device": peak_reserved_gib < total_gib,
            "adapter_saved": "adapter_config.json" in adapter_files
            and any(name.startswith("adapter_model.") for name in adapter_files),
            "adapter_reloaded": reload_error is None
            and bool(reload_comparison)
            and all(
                item["max_abs_next_logit_delta"] <= 1e-4 for item in reload_comparison
            ),
            # Greedy text may legitimately remain unchanged after one step. The
            # required held-out output difference is measured on the model's
            # next-token distribution, with text change retained as evidence.
            "heldout_output_distribution_changed": bool(pre_post)
            and all(item["max_abs_next_logit_delta"] > 0 for item in pre_post),
        }
        smoke_evidence = {
            "executed": True,
            "passed": all(checks.values()),
            "checks": checks,
            "global_step": global_step,
            "train_loss": train_loss,
            "peak_vram": {
                "allocated_gib": round(peak_allocated_gib, 3),
                "reserved_gib": round(peak_reserved_gib, 3),
                "device_total_gib": round(total_gib, 3),
            },
            "adapter_files": sorted(adapter_files),
            "pre_post": pre_post,
            "post_reload": reload_comparison,
            "reload_error": reload_error,
            "pre_outputs": pre_public,
            "post_outputs": post_public,
            "reloaded_outputs": reloaded_public,
        }
    metadata = checkpoint_metadata(
        manifest,
        global_step=global_step,
        trainable_parameters=trainable_parameters,
    )
    if smoke_evidence is not None:
        metadata["smoke_gate"] = smoke_evidence
    metadata_path = write_json(output_dir / "checkpoint_metadata.json", metadata)
    response = {
        "global_step": global_step,
        "train_loss": train_loss,
        "trainable_parameters": trainable_parameters,
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
    }
    if manual_training is not None:
        response["manual_training"] = manual_training
    if smoke_evidence is not None:
        response["smoke_gate"] = smoke_evidence
    return response


def train_adapter(
    config: Mapping[str, Any],
    *,
    dataset_paths: Sequence[Path],
    output_dir: Path,
    allow_model_download: bool,
    manifest: Mapping[str, Any],
    smoke_heldout_cases: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one adapter job with phase-level progress that survives failures."""

    reporter = TrainingProgress(output_dir)
    reporter.emit("runtime", "started")
    try:
        result = _train_adapter_impl(
            config,
            dataset_paths=dataset_paths,
            output_dir=output_dir,
            allow_model_download=allow_model_download,
            manifest=manifest,
            smoke_heldout_cases=smoke_heldout_cases,
            reporter=reporter,
        )
    except BaseException as exc:
        reporter.emit("runtime", "failed", detail={"error_type": type(exc).__name__})
        raise
    reporter.emit(
        "runtime",
        "completed",
        step=int(result.get("global_step", 0)),
        loss=(
            float(result["train_loss"])
            if isinstance(result.get("train_loss"), (int, float))
            else None
        ),
    )
    return result
