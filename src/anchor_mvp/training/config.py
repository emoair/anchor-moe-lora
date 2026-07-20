"""Configuration loading and invariant checks for Gemma 4 QLoRA runs."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .formal_v3_schedule import (
    CANDIDATE_TASKS_PER_STAGE,
    CANDIDATE_WORK_ORDERS,
    SPLIT_SCHEMA,
    FormalV3ScheduleError,
    validate_exposure_control,
)


ALLOWED_ADAPTERS = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
    "mixed_all",
)
SPECIALIST_ADAPTERS = ALLOWED_ADAPTERS[:-1]
ALLOWED_RANKS = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
PER_ADAPTER_TRAINING_OVERRIDE_FIELDS = frozenset({"max_steps", "save_steps"})
_INFERENCE_ONLY_SERIALIZATION_MARKERS = ("-gguf", ".gguf", "-w4a16-ct")
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
FULL_SNAPSHOT_SCHEMA = "anchor.training-snapshot.v2"
PARTIAL_SNAPSHOT_SCHEMA = "anchor.per-expert-partial-training-snapshot.v1"
PARTIAL_TRAINING_MODE = "per_expert_partial_gold"
PARTIAL_RECORDS_PER_EXPERT = 128


class ConfigError(ValueError):
    """Raised when a training config violates a safety or experiment invariant."""


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable: "
                f"{json_error}"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ConfigError(f"top-level config must be a mapping: {path}")
    return value


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        resolved = os.getenv(name, fallback)
        if resolved is None:
            raise ConfigError(f"environment variable {name!r} is required")
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_with_inheritance(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    chain = set() if seen is None else set(seen)
    if path in chain:
        raise ConfigError(f"config inheritance cycle detected at {path}")
    chain.add(path)
    raw = _read_mapping(path)
    parent_name = raw.pop("extends", None)
    overrides = raw.pop("overrides", {})
    if not isinstance(overrides, Mapping):
        raise ConfigError("config overrides must be a mapping")
    if parent_name is None:
        return _deep_merge(raw, overrides)
    if not isinstance(parent_name, str) or not parent_name.strip():
        raise ConfigError("config extends must be a non-empty path")
    parent_path = (path.parent / parent_name).resolve()
    parent = _load_with_inheritance(parent_path, chain)
    return _deep_merge(_deep_merge(parent, raw), overrides)


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML, expand ``${NAME:-fallback}``, and validate it."""

    config_path = Path(path).expanduser().resolve()
    config = _expand_env(_load_with_inheritance(config_path))
    config["_config_path"] = str(config_path)
    validate_training_config(config)
    return config


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Validate the parts that define the controlled A/B/C experiment."""

    if config.get("version") != 1:
        raise ConfigError("version must be 1")

    model = _mapping(config, "model")
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigError("model.id must be a non-empty string")
    normalized_id = model_id.lower()
    if any(marker in normalized_id for marker in _INFERENCE_ONLY_SERIALIZATION_MARKERS):
        raise ConfigError(
            "model.id identifies an inference-only serialization unsupported by this "
            "PEFT training entrypoint (GGUF Q4_0 or vLLM compressed-tensors W4A16). "
            "4-bit quantization itself is supported: use a standard BF16/FP16 "
            "Transformers checkpoint quantized online to bitsandbytes NF4, or a "
            "Transformers/PEFT-compatible 4-bit checkpoint."
        )
    if model.get("architecture") != "multimodal_lm":
        raise ConfigError("Gemma 4 Unified requires model.architecture=multimodal_lm")
    if model.get("training_format") != "transformers_or_peft_4bit":
        raise ConfigError(
            "model.training_format must be 'transformers_or_peft_4bit' for this trainer"
        )
    if model.get("load_strategy") not in ("bnb_nf4_online", "prequantized_peft_4bit"):
        raise ConfigError(
            "model.load_strategy must be 'bnb_nf4_online' or 'prequantized_peft_4bit'"
        )
    if model.get("attention_implementation") != "sdpa":
        raise ConfigError("model.attention_implementation must remain 'sdpa'")
    processor_id = model.get("processor_id")
    if not isinstance(processor_id, str) or not processor_id.strip():
        raise ConfigError("model.processor_id must be a non-empty string")
    if not isinstance(model.get("local_path"), str) or not model["local_path"].strip():
        raise ConfigError("model.local_path must be a non-empty project-relative path")

    quant = _mapping(config, "quantization")
    expected_quant = {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
        "quant_storage_dtype": "bfloat16",
        "freeze_base_model": True,
    }
    for key, expected in expected_quant.items():
        if quant.get(key) != expected:
            raise ConfigError(f"quantization.{key} must be {expected!r}")

    lora = _mapping(config, "lora")
    rank = _positive_int(lora.get("rank"), "lora.rank")
    if rank not in ALLOWED_RANKS:
        raise ConfigError(f"lora.rank must be one of {ALLOWED_RANKS}")
    _positive_int(lora.get("alpha"), "lora.alpha")
    if lora.get("dtype") != "bfloat16":
        raise ConfigError("lora.dtype must be 'bfloat16'")
    if lora.get("bias") != "none":
        raise ConfigError("lora.bias must be 'none' to keep the base frozen")
    dropout = lora.get("dropout")
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ConfigError("lora.dropout must be in [0, 1)")

    training = _mapping(config, "training")
    max_seq_length = _positive_int(
        training.get("max_seq_length"), "training.max_seq_length"
    )
    sequence_contract = training.get("sequence_contract")
    _positive_int(
        training.get("per_device_train_batch_size"),
        "training.per_device_train_batch_size",
    )
    _positive_int(
        training.get("gradient_accumulation_steps"),
        "training.gradient_accumulation_steps",
    )
    _positive_int(training.get("max_steps"), "training.max_steps")
    if training.get("gradient_checkpointing") is not True:
        raise ConfigError("training.gradient_checkpointing must remain enabled")
    if training.get("allow_tf32") is not True:
        raise ConfigError("training.allow_tf32 must remain enabled on Ampere")
    if training.get("per_device_train_batch_size") != 1:
        raise ConfigError("the checked-in 12 GB safety profile requires batch size 1")
    maximum_sequence_length = (
        4096 if sequence_contract == "compact_v2_no_truncation" else 512
    )
    if max_seq_length > maximum_sequence_length:
        raise ConfigError(
            f"training profile caps max_seq_length at {maximum_sequence_length}"
        )
    if training.get("loss_logits") != "active_labels_only":
        raise ConfigError("training.loss_logits must remain 'active_labels_only'")
    runtime_engine = training.get("runtime_engine", "trainer")
    if runtime_engine not in {"trainer", "manual_active_labels_v2"}:
        raise ConfigError(
            "training.runtime_engine must be 'trainer' or 'manual_active_labels_v2'"
        )
    activation_offload = training.get("activation_offload_to_cpu", False)
    if not isinstance(activation_offload, bool):
        raise ConfigError("training.activation_offload_to_cpu must be boolean")
    activation_pin = training.get("activation_offload_pin_memory", True)
    if not isinstance(activation_pin, bool):
        raise ConfigError("training.activation_offload_pin_memory must be boolean")
    expandable_segments = training.get(
        "cuda_allocator_expandable_segments", False
    )
    if not isinstance(expandable_segments, bool):
        raise ConfigError(
            "training.cuda_allocator_expandable_segments must be boolean"
        )
    if activation_offload and runtime_engine != "manual_active_labels_v2":
        raise ConfigError(
            "activation CPU offload requires manual_active_labels_v2"
        )
    if training.get("empty_cache_after_probe") is not True:
        raise ConfigError("training.empty_cache_after_probe must remain enabled")
    selector = training.get("probe_sample_selector", "first")
    if selector not in {"first", "max_rendered_tokens"}:
        raise ConfigError(
            "training.probe_sample_selector must be 'first' or "
            "'max_rendered_tokens'"
        )
    if selector == "max_rendered_tokens":
        if sequence_contract != "compact_v2_no_truncation":
            raise ConfigError(
                "max_rendered_tokens smoke selection requires compact-v2"
            )
        if training.get("probe_pad_to_max_length") is not False:
            raise ConfigError(
                "max_rendered_tokens smoke selection forbids synthetic max padding"
            )
    minimum_backward_free = _positive_int(
        training.get("minimum_backward_free_vram_mib"),
        "training.minimum_backward_free_vram_mib",
    )
    if minimum_backward_free < 256:
        raise ConfigError(
            "training.minimum_backward_free_vram_mib must be at least 256"
        )
    if runtime_engine == "manual_active_labels_v2":
        if sequence_contract == "compact_v2_no_truncation":
            if max_seq_length not in {512, 1024, 2048, 4096}:
                raise ConfigError(
                    "compact-v2 manual runtime requires max_seq_length in "
                    "{512, 1024, 2048, 4096}"
                )
            coverage_manifest = training.get("coverage_manifest")
            if not isinstance(coverage_manifest, str) or not coverage_manifest.strip():
                raise ConfigError(
                    "compact-v2 manual runtime requires training.coverage_manifest"
                )
            if max_seq_length > 1024:
                expected_tier = "2k" if max_seq_length == 2048 else "4k"
                if (
                    training.get("experimental_sequence_profile") is not True
                    or training.get("sequence_tier") != expected_tier
                ):
                    raise ConfigError(
                        f"compact-v2 {expected_tier} requires explicit "
                        "experimental_sequence_profile=true and matching "
                        "training.sequence_tier"
                    )
            if training.get("probe_pad_to_max_length") is True and (
                training.get("is_sequence_feasibility_probe") is not True
                or training.get("per_device_train_batch_size") != 1
            ):
                raise ConfigError(
                    "probe_pad_to_max_length requires an explicit batch-1 "
                    "sequence feasibility probe"
                )
        elif max_seq_length != 64:
            raise ConfigError("formal-v2 manual runtime requires max_seq_length=64")
        if training.get("optim") != "paged_adamw_8bit":
            raise ConfigError("formal-v2 manual runtime requires paged_adamw_8bit")
        if training.get("lr_scheduler_type") != "constant_with_warmup":
            raise ConfigError("formal-v2 manual runtime requires constant_with_warmup")
        if training.get("sample_order") not in {
            "deterministic_epoch_shuffle_v1",
            "deterministic_stage_stratified_epoch_v1",
        }:
            raise ConfigError(
                "manual runtime requires a supported deterministic sample order"
            )
        maximum_peak = training.get("maximum_training_peak_vram_gib")
        maximum_allowed_peak = (
            11.25
            if sequence_contract == "compact_v2_no_truncation"
            else 9.0
        )
        if (
            not isinstance(maximum_peak, (int, float))
            or not 0 < maximum_peak <= maximum_allowed_peak
        ):
            raise ConfigError(
                "manual runtime maximum_training_peak_vram_gib must be in "
                f"(0, {maximum_allowed_peak}]"
            )
        peak_metric = training.get(
            "peak_vram_gate_metric", "allocated_or_reserved"
        )
        if peak_metric not in {"allocated", "allocated_or_reserved"}:
            raise ConfigError(
                "training.peak_vram_gate_metric must be 'allocated' or "
                "'allocated_or_reserved'"
            )
        if peak_metric == "allocated" and sequence_contract != "compact_v2_no_truncation":
            raise ConfigError(
                "allocated-only peak gating requires compact_v2_no_truncation "
                "and its post-cleanup free-memory gate"
            )
        _positive_int(training.get("save_steps"), "training.save_steps")

    adapters = _mapping(config, "adapters")
    missing = set(ALLOWED_ADAPTERS) - set(adapters)
    extra = set(adapters) - set(ALLOWED_ADAPTERS)
    if missing or extra:
        raise ConfigError(
            f"adapters must be exactly {ALLOWED_ADAPTERS}; missing={missing}, extra={extra}"
        )
    for name in ALLOWED_ADAPTERS:
        entry = adapters[name]
        if not isinstance(entry, Mapping):
            raise ConfigError(f"adapters.{name} must be a mapping")
        datasets = entry.get("datasets")
        if (
            not isinstance(datasets, list)
            or not datasets
            or not all(isinstance(p, str) and p for p in datasets)
        ):
            raise ConfigError(
                f"adapters.{name}.datasets must be a non-empty list of paths"
            )
        training_overrides = entry.get("training_overrides", {})
        if not isinstance(training_overrides, Mapping):
            raise ConfigError(
                f"adapters.{name}.training_overrides must be a mapping"
            )
        unsupported = set(training_overrides) - PER_ADAPTER_TRAINING_OVERRIDE_FIELDS
        if unsupported:
            raise ConfigError(
                f"adapters.{name}.training_overrides contains unsupported fields: "
                f"{sorted(unsupported)}"
            )
        for field, value in training_overrides.items():
            _positive_int(value, f"adapters.{name}.training_overrides.{field}")
    mixed = adapters["mixed_all"]["datasets"]
    if (
        len(mixed) < len(SPECIALIST_ADAPTERS)
        and sequence_contract != "compact_v2_no_truncation"
    ):
        raise ConfigError("mixed_all must combine all five specialist datasets")
    if sequence_contract == "compact_v2_no_truncation":
        expected = adapters["mixed_all"].get("expected_experts")
        if not isinstance(expected, list) or set(expected) != set(SPECIALIST_ADAPTERS):
            raise ConfigError(
                "compact-v2 mixed_all must declare all five expected_experts"
            )

    guardrails = _mapping(config, "guardrails")
    if guardrails.get("reject_deployment_artifacts_for_training") is not True:
        raise ConfigError("deployment-artifact rejection guardrail must remain enabled")

    scale_gate = _mapping(config, "scale_gate")
    required_datasets = _mapping(scale_gate, "required_datasets")
    if set(required_datasets) != set(SPECIALIST_ADAPTERS):
        raise ConfigError("scale_gate.required_datasets must name all five specialists")
    for expert, path in required_datasets.items():
        if not isinstance(path, str) or not path.strip():
            raise ConfigError(f"scale_gate.required_datasets.{expert} must be a path")
    base_artifact = _mapping(scale_gate, "base_artifact")
    for field in (
        "repo_id",
        "revision",
        "local_path",
        "download_manifest",
        "weight_file",
        "sha256",
    ):
        if (
            not isinstance(base_artifact.get(field), str)
            or not base_artifact[field].strip()
        ):
            raise ConfigError(
                f"scale_gate.base_artifact.{field} must be non-empty text"
            )
    _positive_int(base_artifact.get("bytes"), "scale_gate.base_artifact.bytes")
    training_artifact = _mapping(scale_gate, "training_artifact")
    if training_artifact.get("format") != "transformers-bitsandbytes-nf4":
        raise ConfigError(
            "scale_gate.training_artifact.format must be "
            "'transformers-bitsandbytes-nf4'"
        )
    for field in ("local_path", "manifest"):
        value = training_artifact.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"scale_gate.training_artifact.{field} must be a non-empty path"
            )
    if training_artifact["local_path"] != model["local_path"]:
        raise ConfigError(
            "scale_gate.training_artifact.local_path must match model.local_path"
        )
    _positive_int(
        training_artifact.get("model_footprint_bytes"),
        "scale_gate.training_artifact.model_footprint_bytes",
    )
    minimum_free = scale_gate.get("minimum_free_vram_gib")
    if not isinstance(minimum_free, (int, float)) or minimum_free <= 0:
        raise ConfigError("scale_gate.minimum_free_vram_gib must be positive")
    minimum_free_host = scale_gate.get("minimum_free_host_memory_gib")
    if not isinstance(minimum_free_host, (int, float)) or minimum_free_host <= 0:
        raise ConfigError("scale_gate.minimum_free_host_memory_gib must be positive")
    for field in ("heldout_cases", "required_smoke_gate_manifest"):
        if not isinstance(scale_gate.get(field), str) or not scale_gate[field].strip():
            raise ConfigError(f"scale_gate.{field} must be a non-empty path")

    snapshot = scale_gate.get("dataset_snapshot")
    experiment = str(config.get("experiment", ""))
    formal_v3 = experiment.startswith("anchor-moe-lora-formal-v3")
    formal_partial = experiment.startswith("anchor-moe-lora-formal-partial-v1")
    if (formal_v3 or formal_partial) and not isinstance(snapshot, Mapping):
        label = "formal-v3" if formal_v3 else "formal-partial-v1"
        raise ConfigError(
            f"{label} requires scale_gate.dataset_snapshot; growing automation "
            "outputs are not valid training inputs"
        )
    if snapshot is not None:
        if not isinstance(snapshot, Mapping):
            raise ConfigError("scale_gate.dataset_snapshot must be a mapping")
        schema_version = snapshot.get("schema_version")
        if schema_version not in {FULL_SNAPSHOT_SCHEMA, PARTIAL_SNAPSHOT_SCHEMA}:
            raise ConfigError(
                "scale_gate.dataset_snapshot.schema_version is unsupported"
            )
        for field in ("manifest", "sidecar"):
            value = snapshot.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(
                    f"scale_gate.dataset_snapshot.{field} must be a non-empty path"
                )
        if snapshot.get("immutable") is not True:
            raise ConfigError("scale_gate.dataset_snapshot.immutable must be true")
        minimum = _positive_int(
            snapshot.get("minimum_records_per_expert"),
            "scale_gate.dataset_snapshot.minimum_records_per_expert",
        )
        if schema_version == FULL_SNAPSHOT_SCHEMA and minimum < 256:
            raise ConfigError(
                "formal-v3 immutable snapshots require at least 256 records per expert "
                "as a quality floor, not a scale limit"
            )
        if formal_v3 and schema_version == FULL_SNAPSHOT_SCHEMA:
            fullscale = scale_gate.get("formal_v3_fullscale")
            if not isinstance(fullscale, Mapping):
                raise ConfigError(
                    "formal-v3 requires scale_gate.formal_v3_fullscale"
                )
            if (
                fullscale.get("maximum_candidate_records_per_expert")
                != CANDIDATE_TASKS_PER_STAGE
                or fullscale.get("candidate_work_orders")
                != CANDIDATE_WORK_ORDERS
                or fullscale.get("selection") != "all_gold_accepted_then_split"
            ):
                raise ConfigError(
                    "formal-v3 snapshot must bind the full 19008-task/95040-work-order "
                    "candidate population and freeze all accepted Gold before splitting"
                )
            split = fullscale.get("split_contract")
            if not isinstance(split, Mapping):
                raise ConfigError(
                    "formal-v3 snapshot requires a train/calibration/heldout split contract"
                )
            expected_split = {
                "schema_version": SPLIT_SCHEMA,
                "train_role": "training_only",
                "calibration_role": "rank_allocation_only",
                "heldout_role": "evaluation_only_hash_metadata",
                "require_pairwise_disjoint": True,
                "require_heldout_content_read_false": True,
            }
            for field, expected in expected_split.items():
                if split.get(field) != expected:
                    raise ConfigError(
                        f"formal-v3 split_contract.{field} must be {expected!r}"
                    )
        if schema_version == PARTIAL_SNAPSHOT_SCHEMA:
            if formal_v3:
                raise ConfigError("formal-v3 cannot use a partial Gold snapshot")
            if not formal_partial:
                raise ConfigError(
                    "partial Gold snapshots require a formal-partial-v1 experiment"
                )
            if (
                snapshot.get("training_mode") != PARTIAL_TRAINING_MODE
                or snapshot.get("not_for_end_to_end_claim") is not True
                or snapshot.get("balanced_records_per_expert")
                != PARTIAL_RECORDS_PER_EXPERT
                or minimum != PARTIAL_RECORDS_PER_EXPERT
            ):
                raise ConfigError(
                    "partial Gold snapshots require explicit per-expert mode, "
                    "not_for_end_to_end_claim=true, and balanced 128/expert"
                )
        elif formal_partial:
            raise ConfigError(
                "formal-partial-v1 requires the partial Gold snapshot schema"
            )

    if formal_v3:
        if "lowmem" in experiment:
            if (
                sequence_contract != "formal_v3_lowmem_truncated_v1"
                or max_seq_length != 64
                or training.get("truncation_policy")
                != "assistant_preserving_prompt_tail_completion_prefix_v1"
                or training.get("full_trajectory_training") is not False
            ):
                raise ConfigError(
                    "formal-v3 lowmem is a 64-token truncated control only; it "
                    "must not claim full-trajectory training"
                )
        if (
            training.get("sample_order")
            != "deterministic_stage_stratified_epoch_v1"
        ):
            raise ConfigError(
                "formal-v3 requires deterministic stage-stratified scheduling"
            )
        try:
            validate_exposure_control(training.get("exposure_control"))
        except FormalV3ScheduleError as exc:
            raise ConfigError(str(exc)) from exc
        formal_arm = experiment.rsplit("-", 1)[-1]
        if formal_arm in {"B", "C", "D", "E", "F"} and config.get(
            "active_adapter"
        ) is not None:
            resolved = training.get("resolved_exposure")
            if not isinstance(resolved, Mapping):
                raise ConfigError(
                    "formal-v3 B-F adapter runs require a snapshot-materialized "
                    "training.resolved_exposure plan"
                )
            if (
                resolved.get("schema_version")
                != "anchor.formal-v3-exposure-plan.v1"
                or resolved.get("arm") != formal_arm
                or resolved.get("control_invariant")
                != "equal_total_and_per_stage_sample_exposure_B_through_F"
                or resolved.get("heldout_content_read") is not False
                or resolved.get("max_steps_per_adapter_job")
                != training.get("max_steps")
                or not isinstance(resolved.get("dataset_snapshot_sha256"), str)
                or not re.fullmatch(
                    r"[0-9a-f]{64}", resolved["dataset_snapshot_sha256"]
                )
            ):
                raise ConfigError(
                    "formal-v3 materialized exposure plan is invalid or unbound"
                )


def select_adapter(
    config: Mapping[str, Any], adapter_name: str, rank: int | None = None
) -> dict[str, Any]:
    """Return an isolated run config for one adapter and optional rank ablation."""

    if adapter_name not in ALLOWED_ADAPTERS:
        raise ConfigError(
            f"unknown adapter {adapter_name!r}; choose from {ALLOWED_ADAPTERS}"
        )
    selected_rank = config["lora"]["rank"] if rank is None else rank
    if selected_rank not in ALLOWED_RANKS:
        raise ConfigError(f"rank must be one of {ALLOWED_RANKS}")

    run = copy.deepcopy(dict(config))
    run["adapter_name"] = adapter_name
    active_adapter = copy.deepcopy(config["adapters"][adapter_name])
    training_overrides = active_adapter.pop("training_overrides", {})
    run["active_adapter"] = active_adapter
    run["training"] = _deep_merge(run["training"], training_overrides)
    run["lora"]["rank"] = selected_rank
    # Keep a conventional 2*r scaling during the rank ablation.
    run["lora"]["alpha"] = 2 * selected_rank
    run["run_name"] = f"{adapter_name}-r{selected_rank}"
    validate_training_config(run)
    return run
