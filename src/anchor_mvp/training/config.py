"""Configuration loading and invariant checks for Gemma 4 QLoRA runs."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


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
_INFERENCE_ONLY_SERIALIZATION_MARKERS = ("-gguf", ".gguf", "-w4a16-ct")
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


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
    if max_seq_length > 512:
        raise ConfigError("12 GB safety profile caps max_seq_length at 512")
    if training.get("loss_logits") != "active_labels_only":
        raise ConfigError("training.loss_logits must remain 'active_labels_only'")
    if training.get("empty_cache_after_probe") is not True:
        raise ConfigError("training.empty_cache_after_probe must remain enabled")
    minimum_backward_free = _positive_int(
        training.get("minimum_backward_free_vram_mib"),
        "training.minimum_backward_free_vram_mib",
    )
    if minimum_backward_free < 256:
        raise ConfigError(
            "training.minimum_backward_free_vram_mib must be at least 256"
        )
    runtime_engine = training.get("runtime_engine", "trainer")
    if runtime_engine not in {"trainer", "manual_active_labels_v2"}:
        raise ConfigError(
            "training.runtime_engine must be 'trainer' or 'manual_active_labels_v2'"
        )
    if runtime_engine == "manual_active_labels_v2":
        if max_seq_length != 64:
            raise ConfigError("formal-v2 manual runtime requires max_seq_length=64")
        if training.get("optim") != "paged_adamw_8bit":
            raise ConfigError("formal-v2 manual runtime requires paged_adamw_8bit")
        if training.get("lr_scheduler_type") != "constant_with_warmup":
            raise ConfigError("formal-v2 manual runtime requires constant_with_warmup")
        if training.get("sample_order") != "deterministic_epoch_shuffle_v1":
            raise ConfigError(
                "formal-v2 manual runtime requires deterministic_epoch_shuffle_v1"
            )
        maximum_peak = training.get("maximum_training_peak_vram_gib")
        if not isinstance(maximum_peak, (int, float)) or not 0 < maximum_peak <= 9.0:
            raise ConfigError(
                "formal-v2 maximum_training_peak_vram_gib must be in (0, 9.0]"
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
    mixed = adapters["mixed_all"]["datasets"]
    if len(mixed) < len(SPECIALIST_ADAPTERS):
        raise ConfigError("mixed_all must combine all five specialist datasets")

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
    formal_v3 = str(config.get("experiment", "")).startswith(
        "anchor-moe-lora-formal-v3"
    )
    if formal_v3 and not isinstance(snapshot, Mapping):
        raise ConfigError(
            "formal-v3 requires scale_gate.dataset_snapshot; growing automation "
            "outputs are not valid training inputs"
        )
    if snapshot is not None:
        if not isinstance(snapshot, Mapping):
            raise ConfigError("scale_gate.dataset_snapshot must be a mapping")
        if snapshot.get("schema_version") != "anchor.training-snapshot.v2":
            raise ConfigError(
                "scale_gate.dataset_snapshot.schema_version must be "
                "'anchor.training-snapshot.v2'"
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
        if minimum < 256:
            raise ConfigError(
                "formal-v3 immutable snapshots require at least 256 records per expert"
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
    run["active_adapter"] = copy.deepcopy(config["adapters"][adapter_name])
    run["lora"]["rank"] = selected_rank
    # Keep a conventional 2*r scaling during the rank ablation.
    run["lora"]["alpha"] = 2 * selected_rank
    run["run_name"] = f"{adapter_name}-r{selected_rank}"
    validate_training_config(run)
    return run
