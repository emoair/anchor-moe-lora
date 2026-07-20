"""Reproducibility manifests and checkpoint metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config: Mapping[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    encoded = json.dumps(
        public, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_manifest(
    config: Mapping[str, Any],
    *,
    dependency_report: Mapping[str, Any],
    datasets: Sequence[Mapping[str, Any]],
    mode: str,
) -> dict[str, Any]:
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "run_name": config.get("run_name"),
        "adapter_name": config.get("adapter_name"),
        "base_model": config["model"]["id"],
        "base_model_revision": config["model"].get("revision"),
        "base_model_local_path": config["model"].get("local_path"),
        "processor": config["model"].get("processor_id"),
        "processor_revision": config["model"].get("processor_revision"),
        "training_precision": {
            "base_weights": (
                "4-bit NF4 (quantized online and frozen)"
                if config["model"]["load_strategy"] == "bnb_nf4_online"
                else "training-compatible prequantized 4-bit checkpoint (frozen)"
            ),
            "load_strategy": config["model"]["load_strategy"],
            "double_quant": config["quantization"]["double_quant"],
            "compute": config["quantization"]["compute_dtype"],
            "adapter": config["lora"]["dtype"],
        },
        "lora": {
            "rank": config["lora"]["rank"],
            "alpha": config["lora"]["alpha"],
            "dropout": config["lora"]["dropout"],
        },
        "training_profile": {
            "runtime_engine": config["training"].get("runtime_engine", "trainer"),
            "max_seq_length": config["training"]["max_seq_length"],
            "sequence_contract": config["training"].get("sequence_contract"),
            "truncation_policy": config["training"].get("truncation_policy"),
            "full_trajectory_training": config["training"].get(
                "full_trajectory_training"
            ),
            "runtime_sequence_statistics": (
                "written_after_execution_to_runtime_observations.sequence_statistics"
            ),
            "max_steps": config["training"]["max_steps"],
            "per_device_train_batch_size": config["training"][
                "per_device_train_batch_size"
            ],
            "gradient_accumulation_steps": config["training"][
                "gradient_accumulation_steps"
            ],
            "gradient_checkpointing": config["training"]["gradient_checkpointing"],
            "seed": config["training"]["seed"],
            "sample_order": config["training"].get("sample_order"),
            "maximum_training_peak_vram_gib": config["training"].get(
                "maximum_training_peak_vram_gib"
            ),
            "activation_offload_to_cpu": config["training"].get(
                "activation_offload_to_cpu", False
            ),
            "activation_offload_pin_memory": config["training"].get(
                "activation_offload_pin_memory", True
            ),
            "cuda_allocator_expandable_segments": config["training"].get(
                "cuda_allocator_expandable_segments", False
            ),
            "probe_sample_selector": config["training"].get(
                "probe_sample_selector", "first"
            ),
        },
        "config_sha256": config_fingerprint(config),
        "config_path": config.get("_config_path"),
        "datasets": list(datasets),
        "environment": dict(dependency_report),
    }
    if config["training"].get("runtime_engine") == "manual_active_labels_v2":
        records = sum(
            int(item.get("valid_records", 0))
            for item in datasets
            if item.get("exists") is True
        )
        exposures = int(config["training"]["max_steps"]) * int(
            config["training"]["gradient_accumulation_steps"]
        )
        manifest["sample_exposure_plan"] = {
            "order": config["training"]["sample_order"],
            "dataset_records": records,
            "sample_exposures": exposures,
            "complete_epochs": exposures // records if records else None,
            "balanced_complete_epochs": bool(records and exposures % records == 0),
        }
        resolved = config["training"].get("resolved_exposure")
        if isinstance(resolved, Mapping):
            manifest["sample_exposure_plan"].update(
                {
                    "derivation": resolved.get("mode"),
                    "target_epochs": resolved.get("epochs"),
                    "records_per_stage": resolved.get("records_per_stage"),
                    "optimizer_steps_per_stage": resolved.get(
                        "optimizer_steps_per_stage"
                    ),
                    "padding_exposures_per_stage": resolved.get(
                        "padding_exposures_per_stage"
                    ),
                    "planned_exposures_by_stage": resolved.get(
                        "planned_exposures_by_stage"
                    ),
                    "padding_exposures_by_stage": resolved.get(
                        "padding_exposures_by_stage"
                    ),
                    "arm_total_sample_exposures": resolved.get(
                        "arm_total_sample_exposures"
                    ),
                    "control_invariant": resolved.get("control_invariant"),
                    "dataset_snapshot_sha256": resolved.get(
                        "dataset_snapshot_sha256"
                    ),
                    "snapshot_manifest_sha256": resolved.get(
                        "snapshot_manifest_sha256"
                    ),
                    "heldout_content_read": resolved.get(
                        "heldout_content_read"
                    ),
                    "evaluation_contract": resolved.get(
                        "evaluation_contract"
                    ),
                }
            )
        manifest["safety_checkpoints"] = {
            "save_steps": int(config["training"]["save_steps"]),
            "resume_capability": "adapter_weights_warm_start_only",
            "optimizer_state_saved": False,
            "scheduler_state_saved": False,
            "rng_state_saved": False,
        }
    return manifest


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def checkpoint_metadata(
    manifest: Mapping[str, Any], *, global_step: int, trainable_parameters: int
) -> dict[str, Any]:
    return {
        "metadata_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": manifest.get("run_name"),
        "adapter_name": manifest.get("adapter_name"),
        "base_model": manifest.get("base_model"),
        "config_sha256": manifest.get("config_sha256"),
        "global_step": global_step,
        "trainable_parameters": trainable_parameters,
        "artifact_type": "peft_adapter",
        "merge_status": "unmerged",
    }
