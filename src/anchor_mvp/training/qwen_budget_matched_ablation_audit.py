"""Offline auditor and comparison manifest for the budget-matched ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from safetensors.numpy import load_file as load_safetensors

from . import qwen_budget_matched_ablation as runner
from . import qwen_lora_diagnostic as qdiag
from . import qwen_synthetic_scaffold_diagnostic as synth


SCHEMA_VERSION = "anchor.qwen25-1.5b-budget-matched-ablation-comparison.v1"
SCHEMA_PATH = "configs/research/qwen_budget_matched_ablation_comparison.schema.json"
IMPLEMENTATION_PATH = "src/anchor_mvp/training/qwen_budget_matched_ablation_audit.py"
EXACT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "diagnostic_receipt.json",
    "diagnostic_receipt.json.sha256",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_binding(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(runner._root()).as_posix(),
        "sha256": _sha256(path.read_bytes()),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _authenticate_json(path: Path) -> tuple[Mapping[str, Any], str]:
    qdiag._assert_physical_path(path, require_file=True, label=str(path))
    sidecar = path.with_name(f"{path.name}.sha256")
    qdiag._assert_physical_path(sidecar, require_file=True, label=str(sidecar))
    data = path.read_bytes()
    digest = _sha256(data)
    if sidecar.read_bytes() != f"{digest}  {path.name}\n".encode("ascii"):
        raise RuntimeError(f"SHA-256 sidecar mismatch: {path}")
    return _read_json(path), digest


def _validate_saved_tensors(
    artifact_dir: Path,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    observed_files = {item.name for item in artifact_dir.iterdir()}
    if observed_files != EXACT_FILES:
        raise RuntimeError(
            f"artifact exact-file inventory drifted: {artifact_dir}: "
            f"{sorted(observed_files)}"
        )
    for item in artifact_dir.iterdir():
        qdiag._assert_physical_path(item, require_file=True, label=str(item))
    ranks = runner._ranks(profile)
    alphas = runner._alphas(profile)
    adapter_config = _read_json(artifact_dir / "adapter_config.json")
    if (
        adapter_config.get("base_model_name_or_path") != qdiag.EXPECTED_MODEL_ID
        or adapter_config.get("bias") != "none"
        or float(adapter_config.get("lora_dropout", -1)) != 0.0
        or adapter_config.get("use_rslora") is not False
        or adapter_config.get("use_dora") is not False
        or set(adapter_config.get("target_modules", [])) != set(ranks)
    ):
        raise RuntimeError("saved adapter_config escaped the comparison contract")
    state = load_safetensors(artifact_dir / "adapter_model.safetensors")
    output_sizes = {
        "q_proj": runner.HIDDEN_SIZE,
        "o_proj": runner.HIDDEN_SIZE,
        "k_proj": runner.KV_SIZE,
        "v_proj": runner.KV_SIZE,
    }
    observed: set[tuple[int, str, str]] = set()
    parameters = 0
    for name, value in state.items():
        match = runner._LORA_RE.search(name)
        if match is None:
            raise RuntimeError(f"unexpected saved LoRA tensor: {name}")
        layer, module, side = int(match.group(1)), match.group(2), match.group(3)
        if module not in ranks:
            raise RuntimeError(f"saved tensor escaped target modules: {name}")
        key = (layer, module, side)
        if key in observed:
            raise RuntimeError(f"duplicate saved tensor: {name}")
        observed.add(key)
        rank = ranks[module]
        expected_shape = (
            (rank, runner.HIDDEN_SIZE) if side == "A" else (output_sizes[module], rank)
        )
        if tuple(int(item) for item in value.shape) != expected_shape:
            raise RuntimeError(f"saved tensor shape mismatch: {name}: {value.shape}")
        parameters += int(value.size)
    required = {
        (layer, module, side)
        for layer in range(runner.EXPECTED_LAYERS)
        for module in ranks
        for side in ("A", "B")
    }
    expected_tensors = int(profile["expected_trainable_tensors"])
    if (
        observed != required
        or len(state) != expected_tensors
        or parameters != runner.EXPECTED_PARAMETERS
    ):
        raise RuntimeError("saved tensor coverage or parameter budget drifted")
    return {
        "ranks": ranks,
        "alphas": alphas,
        "tensor_count": len(state),
        "parameters": parameters,
        "all_shapes_valid": True,
        "unexpected_tensors": 0,
    }


def _eval_view_digest(
    config: Mapping[str, Any], dataset: synth.ScaffoldDataset
) -> dict[str, Any]:
    model_path = Path(str(config["model"]["local_path"])).resolve()
    qdiag._assert_physical_path(
        model_path, require_directory=True, label="local tokenizer directory"
    )
    tokenizer = synth._load_tokenizer_from_path(model_path)
    rows: list[dict[str, str]] = []
    ordered = sorted(dataset.eval_proxy, key=lambda item: item.record_id)
    for example in ordered:
        encoded = synth.tokenize_example(
            tokenizer,
            example,
            sequence_length=int(config["training"]["sequence_length"]),
        )
        rows.append(
            {
                "record_id_sha256": _sha256(example.record_id.encode("utf-8")),
                "input_ids_sha256": synth._signed_int64_sequence_sha256(
                    encoded["input_ids"]
                ),
                "labels_sha256": synth._signed_int64_sequence_sha256(encoded["labels"]),
            }
        )
    return {
        "algorithm": "record_id_ascending_sha256_signed_int64_token_views_v1",
        "records": len(rows),
        "ordered_views_sha256": synth._compact_json_sha256(rows),
        "raw_record_ids_emitted": False,
        "raw_token_ids_emitted": False,
    }


def _preflight(
    path: Path, expected_sha256: str, expected_profile: str
) -> Mapping[str, Any]:
    value, digest = _authenticate_json(path)
    if (
        digest != expected_sha256
        or value.get("profile") != expected_profile
        or value.get("ready") is not True
        or value.get("claims", {}).get("formal") is not False
    ):
        raise RuntimeError(f"preflight identity mismatch: {path}")
    return value


def build_comparison(
    config: Mapping[str, Any], preflight_paths: Mapping[str, Path]
) -> dict[str, Any]:
    dataset = synth.load_dataset(config)
    common_values: dict[str, list[Any]] = {}
    arms: list[dict[str, Any]] = []
    for profile_name in runner.PROFILES:
        profile = runner._profile(config, profile_name)
        artifact_dir = Path(
            str(config["output"]["adapter_dir_template"]).format(profile=profile_name)
        )
        if not artifact_dir.is_absolute():
            artifact_dir = runner._root() / artifact_dir
        qdiag._assert_physical_path(
            artifact_dir, require_directory=True, label=f"{profile_name} artifact"
        )
        receipt, receipt_sha = _authenticate_json(
            artifact_dir / "diagnostic_receipt.json"
        )
        if (
            receipt.get("schema_version") != runner.SCHEMA_VERSION
            or receipt.get("profile") != profile_name
            or receipt.get("status") != "passed_controlled_proxy_only"
        ):
            raise RuntimeError(f"diagnostic receipt drifted: {profile_name}")
        preflight = _preflight(
            preflight_paths[profile_name],
            str(receipt["preflight_sha256"]),
            profile_name,
        )
        if receipt.get("config_sha256") != preflight.get("config_sha256"):
            raise RuntimeError(f"config binding drifted: {profile_name}")
        saved = _validate_saved_tensors(artifact_dir, profile)
        physical_hashes = {
            name: _sha256((artifact_dir / name).read_bytes())
            for name in ("adapter_config.json", "adapter_model.safetensors")
        }
        if physical_hashes != receipt.get("adapter_artifact_sha256"):
            raise RuntimeError(f"adapter file hashes drifted: {profile_name}")
        if (
            receipt["base_hash_before"] != receipt["base_hash_after"]
            or receipt["base_hash_before"] != receipt["reloaded_base_hash"]
            or float(receipt["max_abs_reload_logit_delta"]) != 0.0
            or receipt["train_order"]["records"] != 80
            or receipt["train_order"]["unique_records"] != 80
            or receipt["train_order"]["duplicates"] != 0
            or receipt["train_order"]["missing"] != 0
            or receipt["optimizer_steps"] != 80
        ):
            raise RuntimeError(f"runtime integrity gate failed: {profile_name}")
        common_fields = {
            "config_sha256": receipt["config_sha256"],
            "dataset_manifest_sha256": receipt["dataset_manifest_sha256"],
            "partition_sha256": receipt["partition_sha256"],
            "ordered_record_ids_sha256": receipt["train_order"][
                "ordered_record_ids_sha256"
            ],
            "ordered_tokenized_examples_sha256": receipt["train_order"][
                "ordered_tokenized_examples_sha256"
            ],
            "base_hash": receipt["base_hash_before"],
            "eval_macro_loss_before": receipt["eval_proxy_before"]["macro_loss"],
            "eval_micro_nll_before": receipt["eval_proxy_before"][
                "micro_target_token_nll"
            ],
            "eval_target_tokens": receipt["eval_proxy_before"]["target_tokens"],
            "train_full_tokens": receipt["performance"]["train_full_tokens"],
        }
        for key, value in common_fields.items():
            common_values.setdefault(key, []).append(value)
        arms.append(
            {
                "profile": profile_name,
                "receipt": {
                    "path": (artifact_dir / "diagnostic_receipt.json")
                    .relative_to(runner._root())
                    .as_posix(),
                    "sha256": receipt_sha,
                },
                "preflight": _file_binding(preflight_paths[profile_name]),
                "adapter_artifact_sha256": physical_hashes,
                "saved_tensor_audit": saved,
                "eval_macro_loss_after": receipt["eval_proxy_after"]["macro_loss"],
                "eval_micro_nll_after": receipt["eval_proxy_after"][
                    "micro_target_token_nll"
                ],
                "eval_ppl_after": receipt["eval_proxy_after"]["micro_target_token_ppl"],
                "eval_macro_loss_delta_percent": receipt[
                    "eval_proxy_macro_loss_delta_percent"
                ],
                "bundle_improved_count": receipt["bundle_improved_count"],
                "train_tokens_per_second": receipt["performance"][
                    "train_full_tokens_per_second"
                ],
                "train_wall_seconds": receipt["performance"]["train_wall_seconds"],
                "peak_allocated_vram_bytes": receipt["performance"][
                    "peak_allocated_vram_bytes"
                ],
                "integrity": {
                    "base_unchanged": True,
                    "save_reload_logit_delta": 0.0,
                    "all_lora_tensors_observed_nonzero_gradient": receipt["lora"][
                        "all_tensors_nonzero_gradient_observed"
                    ],
                },
            }
        )
    common: dict[str, Any] = {}
    for key, values in common_values.items():
        encoded = [json.dumps(item, sort_keys=True) for item in values]
        if len(set(encoded)) != 1:
            raise RuntimeError(f"cross-arm fairness field drifted: {key}")
        common[key] = values[0]
    ranking = [
        item["profile"]
        for item in sorted(arms, key=lambda item: item["eval_macro_loss_after"])
    ]
    config_path = Path(config["_config_path"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_controlled_proxy_comparison_only",
        "config": _file_binding(config_path),
        "runner": _file_binding(runner._root() / runner.IMPLEMENTATION_PATH),
        "auditor": _file_binding(runner._root() / IMPLEMENTATION_PATH),
        "dataset": {
            "manifest_sha256": dataset.manifest_sha256,
            "partition_sha256": dict(dataset.partition_sha256),
            "train_records": len(dataset.train),
            "eval_proxy_records": len(dataset.eval_proxy),
            "eval_ordered_token_views": _eval_view_digest(config, dataset),
        },
        "common": common,
        "arms": arms,
        "ranking": ranking,
        "claims": {
            "diagnostic_only": True,
            "controlled_proxy_only": True,
            "formal": False,
            "training_authorized": False,
            "eval_proxy_is_heldout": False,
            "statistical_significance_claimed": False,
            "deterministic_algorithms_enabled": False,
            "multi_seed_validated": False,
        },
        "audit": {
            "protected_body_reads": 0,
            "heldout_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "tokenizer_loads": 1,
            "saved_tensors_fully_validated": True,
            "final_directories_reopened": True,
        },
    }
    schema = _read_json(runner._root() / SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda item: list(item.path),
    )
    if errors:
        raise RuntimeError("comparison schema validation failed: " + errors[0].message)
    return report


def publish(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, str]:
    if not output_dir.is_absolute():
        output_dir = runner._root() / output_dir
    diagnostics = (runner._root() / "artifacts" / "diagnostics").resolve()
    output_dir = output_dir.resolve()
    if diagnostics not in output_dir.parents or os.path.lexists(output_dir):
        raise RuntimeError("comparison output must be new under artifacts/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "comparison.json"
    data = runner._canonical_json_bytes(report)
    digest = _sha256(data)
    path.write_bytes(data)
    (output_dir / "comparison.json.sha256").write_bytes(
        f"{digest}  comparison.json\n".encode("ascii")
    )
    authenticated, observed = _authenticate_json(path)
    if authenticated != report or observed != digest:
        raise RuntimeError("published comparison failed reopen authentication")
    return path, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit three budget-matched LoRA arms")
    parser.add_argument("--config", required=True)
    parser.add_argument("--q-only-preflight", required=True)
    parser.add_argument("--q-plus-o-preflight", required=True)
    parser.add_argument("--wide-preflight", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = runner.load_config(args.config)
    paths = {
        "q_only": Path(args.q_only_preflight),
        "q_plus_o": Path(args.q_plus_o_preflight),
        "wide_budget_matched": Path(args.wide_preflight),
    }
    paths = {
        key: value if value.is_absolute() else runner._root() / value
        for key, value in paths.items()
    }
    report = build_comparison(config, paths)
    path, digest = publish(report, Path(args.output))
    print(
        json.dumps(
            {"comparison": str(path), "sha256": digest, "ranking": report["ranking"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
