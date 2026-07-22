"""Read-only Q/O contribution audit for the trained Q+O synthetic adapter.

The audit never mutates adapter weights.  It evaluates four in-memory views by
temporarily changing PEFT LoRA scaling: full, Q only, O only, and adapter off.
All published results are aggregate teacher-forced metrics; prompt and target
bodies are deliberately absent from the receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from anchor_mvp.training import qwen_budget_matched_ablation as trained
from anchor_mvp.training import qwen_lora_diagnostic as qdiag
from anchor_mvp.training import qwen_synthetic_scaffold_diagnostic as synth
from anchor_mvp.training.config import ConfigError, _expand_env
from anchor_mvp.training.manifest import config_fingerprint


CONFIG_VERSION = "anchor.qwen25-1.5b-qo-memory-ablation-audit-config.v1"
RECEIPT_VERSION = "anchor.qwen25-1.5b-qo-memory-ablation-audit-receipt.v1"
CONFIG_PATH = "configs/research/qwen_qo_memory_ablation_audit_v1.yaml"
IMPLEMENTATION_PATH = "src/anchor_mvp/research/qwen_qo_memory_ablation_audit.py"
MODES = ("full", "q_only_contribution", "o_only_contribution", "adapter_off")
ROLES = ("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate")
_ADAPTER_FILES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.safetensors",
        "diagnostic_receipt.json",
        "diagnostic_receipt.json.sha256",
    }
)
_OUTPUT_FILES = frozenset({"audit_receipt.json", "audit_receipt.json.sha256"})
_MAX_METADATA_BYTES = 2_000_000
_MAX_ADAPTER_BYTES = 20_000_000


def _root() -> Path:
    return qdiag._project_root_from_module()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _strict_path(path: str | Path, expected: str) -> Path:
    canonical = _root() / Path(expected)
    requested = Path(path)
    resolved = requested if requested.is_absolute() else _root() / requested
    if os.path.normcase(str(resolved.resolve())) != os.path.normcase(
        str(canonical.resolve())
    ):
        raise ConfigError(f"path must remain exactly {expected}")
    return qdiag._assert_physical_path(canonical, require_file=True, label=expected)


def load_config(path: str | Path) -> dict[str, Any]:
    canonical = _strict_path(path, CONFIG_PATH)
    snapshot = synth._read_snapshot(canonical, max_bytes=_MAX_METADATA_BYTES)
    if b"\r" in snapshot.data:
        raise ConfigError("Q/O audit config must be LF-only")
    try:
        value = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError("Q/O audit config is not UTF-8 YAML") from exc
    if not isinstance(value, dict):
        raise ConfigError("Q/O audit config must be a mapping")
    result = _expand_env(value)
    result["_config_path"] = str(canonical)
    validate_config(result)
    snapshot.assert_unchanged()
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "claim_scope",
        "model",
        "dataset",
        "adapter",
        "audit",
        "claims",
        "_config_path",
    }:
        raise ConfigError("Q/O audit config fields drifted")
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope") != "synthetic_teacher_forced_diagnostic_only"
    ):
        raise ConfigError("Q/O audit config identity drifted")
    model = synth._mapping(config.get("model"), "model")
    if (
        model.get("id") != qdiag.EXPECTED_MODEL_ID
        or model.get("local_files_only") is not True
        or model.get("allow_network") is not False
        or model.get("trust_remote_code") is not False
        or model.get("expected_source_revision") != qdiag.EXPECTED_SOURCE_REVISION
        or model.get("expected_source_repo") != qdiag.EXPECTED_SOURCE_REPO
    ):
        raise ConfigError("Q/O audit model/network contract drifted")
    dataset = synth._mapping(config.get("dataset"), "dataset")
    if (
        dataset.get("kind") != "synthetic_nl_scaffold_diagnostic_v1"
        or dataset.get("expected_manifest_sha256")
        != "64b1ce813477deef48de16dbdc0d2561bbeaa0ef5d6248862e9f2bedc8acc0dd"
        or dataset.get("train_records") != 80
        or dataset.get("eval_proxy_records") != 20
        or dataset.get("formal_inputs_allowed") is not False
        or dataset.get("heldout_allowed") is not False
        or dataset.get("protected_source_paths_allowed") is not False
    ):
        raise ConfigError("Q/O audit dataset contract drifted")
    adapter = synth._mapping(config.get("adapter"), "adapter")
    if adapter != {
        "path": "artifacts/diagnostics/qwen2_5_1_5b_synthetic_scaffold_budget_matched_q_plus_o_step80",
        "source_profile": "q_plus_o",
        "expected_receipt_sha256": "dc94204df696db795f3e657c679d67918e3aa2723b2d0bdf7dee899ef4490f6e",
        "expected_adapter_config_sha256": "17af30108f7163bc30773d5d53e7847fd51f6e3faaf779f4f79f84e368883222",
        "expected_adapter_model_sha256": "fab58c506a103f09b45e22b1aa10c0d7073e6c5c032dc49f2e89d7219427d6ba",
        "expected_trainable_parameters": 1_376_256,
        "expected_q_rank": 8,
        "expected_o_rank": 8,
    }:
        raise ConfigError("Q/O adapter binding drifted")
    audit = synth._mapping(config.get("audit"), "audit")
    if audit != {
        "modes": list(MODES),
        "sequence_length": 512,
        "batch_size": 1,
        "compute_dtype": "bfloat16",
        "tf32": True,
        "ood_generator_seed": 20260723,
        "ood_bundles": 4,
        "ood_roles_per_bundle": 5,
        "expected_ood_records": 20,
        "output_dir": "artifacts/diagnostics/qwen2_5_1_5b_qo_memory_ablation_audit_v1",
    }:
        raise ConfigError("Q/O audit execution contract drifted")
    if config.get("claims") != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
        "ood_proxy_is_heldout": False,
        "statistical_significance_claimed": False,
    }:
        raise ConfigError("Q/O audit claims drifted")


def _exact_files(path: Path, expected: frozenset[str], label: str) -> None:
    qdiag._assert_physical_path(path, require_directory=True, label=label)
    observed = {item.name for item in path.iterdir()}
    if observed != set(expected):
        raise ConfigError(f"{label} exact file inventory drifted: {sorted(observed)!r}")
    for name in expected:
        qdiag._assert_physical_path(
            path / name, require_file=True, label=f"{label}/{name}"
        )


def authenticate_adapter(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = synth._mapping(config.get("adapter"), "adapter")
    adapter_dir = _root() / Path(str(binding["path"]))
    _exact_files(adapter_dir, _ADAPTER_FILES, "Q+O adapter")
    snapshots = {
        "adapter_config.json": synth._read_snapshot(
            adapter_dir / "adapter_config.json", max_bytes=_MAX_METADATA_BYTES
        ),
        "adapter_model.safetensors": synth._read_snapshot(
            adapter_dir / "adapter_model.safetensors", max_bytes=_MAX_ADAPTER_BYTES
        ),
        "diagnostic_receipt.json": synth._read_snapshot(
            adapter_dir / "diagnostic_receipt.json", max_bytes=_MAX_METADATA_BYTES
        ),
        "diagnostic_receipt.json.sha256": synth._read_snapshot(
            adapter_dir / "diagnostic_receipt.json.sha256", max_bytes=1024
        ),
    }
    expected = {
        "adapter_config.json": binding["expected_adapter_config_sha256"],
        "adapter_model.safetensors": binding["expected_adapter_model_sha256"],
        "diagnostic_receipt.json": binding["expected_receipt_sha256"],
    }
    for name, digest in expected.items():
        if snapshots[name].sha256 != digest:
            raise ConfigError(f"authenticated Q+O {name} SHA-256 mismatch")
    if snapshots["diagnostic_receipt.json.sha256"].data != (
        f"{binding['expected_receipt_sha256']}  diagnostic_receipt.json\n".encode(
            "ascii"
        )
    ):
        raise ConfigError("Q+O diagnostic receipt sidecar is malformed")
    receipt = synth._strict_json(
        snapshots["diagnostic_receipt.json"].data, "Q+O diagnostic receipt"
    )
    lora = synth._mapping(receipt.get("lora"), "receipt.lora")
    claims = synth._mapping(receipt.get("claims"), "receipt.claims")
    if (
        receipt.get("schema_version") != trained.SCHEMA_VERSION
        or receipt.get("status") != "passed_controlled_proxy_only"
        or receipt.get("profile") != "q_plus_o"
        or receipt.get("dataset_manifest_sha256")
        != config["dataset"]["expected_manifest_sha256"]
        or lora.get("ranks") != {"o_proj": 8, "q_proj": 8}
        or lora.get("trainable_parameters") != 1_376_256
        or claims.get("formal") is not False
        or claims.get("training_authorized") is not False
    ):
        raise ConfigError("Q+O diagnostic receipt semantic binding drifted")
    adapter_config = synth._strict_json(
        snapshots["adapter_config.json"].data, "adapter_config.json"
    )
    if (
        adapter_config.get("base_model_name_or_path") != qdiag.EXPECTED_MODEL_ID
        or set(adapter_config.get("target_modules", [])) != {"q_proj", "o_proj"}
        or adapter_config.get("r") != 8
        or adapter_config.get("lora_alpha") != 16
        or adapter_config.get("use_dora") is not False
        or adapter_config.get("use_rslora") is not False
    ):
        raise ConfigError("Q+O adapter_config escaped the expected LoRA scope")
    return {
        "path": adapter_dir,
        "receipt": receipt,
        "snapshots": snapshots,
        "file_sha256": {name: item.sha256 for name, item in snapshots.items()},
    }


def assert_adapter_unchanged(authenticated: Mapping[str, Any]) -> None:
    _exact_files(authenticated["path"], _ADAPTER_FILES, "Q+O adapter")
    for snapshot in authenticated["snapshots"].values():
        snapshot.assert_unchanged()


def _ood_task_specs() -> tuple[tuple[str, str, str], ...]:
    return (
        (
            "en",
            "Design a local command-line unit converter with explicit dimensional checks and deterministic decimal rounding.",
            "unit-converter-cli",
        ),
        (
            "en",
            "Review a private notes search endpoint that supports tags, pagination, and a strict no-network execution policy.",
            "notes-search-endpoint",
        ),
        (
            "zh-CN",
            "设计一个离线 CSV 架构校验器，要求列类型明确、错误定位稳定且不修改输入文件。",
            "csv-schema-validator",
        ),
        (
            "zh-CN",
            "实现一个无障碍配色选择器，提供对比度提示、键盘操作和确定性的导出格式。",
            "accessible-color-picker",
        ),
    )


def _role_target(role: str, slug: str, language: str) -> str:
    action = {
        "planner": "decompose_and_assign",
        "tool_policy": "approve_local_minimal_tools",
        "frontend_gen": "implement_bounded_component",
        "frontend_review": "review_against_contract",
        "security_gate": "allow_if_local_and_validated",
    }[role]
    value = {
        "action": action,
        "contract": slug,
        "language": language,
        "role": role,
        "status": "ready",
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_ood_examples(seed: int) -> tuple[synth.ScaffoldExample, ...]:
    examples: list[synth.ScaffoldExample] = []
    for task_index, (language, request, slug) in enumerate(_ood_task_specs()):
        source = _sha(f"qo-ood-v1\0{seed}\0{task_index}\0{slug}".encode())
        bundle = f"qo-ood-bundle-v1:{source}"
        for role in ROLES:
            prompt = (
                "This is a new deterministic OOD diagnostic task. "
                f"Act only as {role}; return one compact JSON object. "
                f"Language={language}. Request={request}"
            )
            target = _role_target(role, slug, language)
            record_hash = _sha(f"{bundle}\0{role}".encode())
            examples.append(
                synth.ScaffoldExample(
                    record_id=f"qo-ood-record-v1:{record_hash}",
                    split="ood_proxy",
                    variant="json_only",
                    source_bundle_id=bundle,
                    prompt=prompt,
                    target=target,
                )
            )
    if len(examples) != 20 or len({item.record_id for item in examples}) != 20:
        raise RuntimeError("OOD generator did not produce 20 unique records")
    return tuple(examples)


def _body_digest_inventory(
    dataset: synth.ScaffoldDataset, ood: Sequence[synth.ScaffoldExample]
) -> dict[str, Any]:
    source = (*dataset.train, *dataset.eval_proxy)
    source_bundles = {item.source_bundle_id for item in source}
    ood_bundles = {item.source_bundle_id for item in ood}
    if source_bundles & ood_bundles:
        raise RuntimeError("OOD source bundle overlaps train/eval source bundle")
    source_bodies = {
        _sha(value.encode("utf-8"))
        for item in source
        for value in (item.prompt, item.target)
    }
    ood_bodies = {
        _sha(value.encode("utf-8"))
        for item in ood
        for value in (item.prompt, item.target)
    }
    if source_bodies & ood_bodies:
        raise RuntimeError("OOD prompt/target body exactly overlaps source fixture")
    rows = [
        {
            "record_id_sha256": _sha(item.record_id.encode("utf-8")),
            "bundle_id_sha256": _sha(item.source_bundle_id.encode("utf-8")),
            "prompt_sha256": _sha(item.prompt.encode("utf-8")),
            "target_sha256": _sha(item.target.encode("utf-8")),
        }
        for item in sorted(ood, key=lambda value: value.record_id)
    ]
    return {
        "algorithm": "sha256_utf8_exact_body_and_compact_sorted_rows_v1",
        "records": len(rows),
        "source_bundles": len(ood_bundles),
        "rows_sha256": synth._compact_json_sha256(rows),
        "exact_body_overlap_count": 0,
        "source_bundle_overlap_count": 0,
        "raw_bodies_emitted": False,
    }


def build_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    config_snapshot = synth._read_snapshot(
        _strict_path(str(config["_config_path"]), CONFIG_PATH),
        max_bytes=_MAX_METADATA_BYTES,
    )
    implementation = synth._read_snapshot(
        _strict_path(IMPLEMENTATION_PATH, IMPLEMENTATION_PATH),
        max_bytes=_MAX_METADATA_BYTES,
    )
    adapter = authenticate_adapter(config)
    dataset = synth.load_dataset(config)
    ood = build_ood_examples(int(config["audit"]["ood_generator_seed"]))
    inventory = _body_digest_inventory(dataset, ood)
    output = _output_path(config)
    report = {
        "schema_version": RECEIPT_VERSION,
        "status": "ready_read_only_teacher_forced_proxy_audit",
        "ready": True,
        "config": {
            "path": CONFIG_PATH,
            "logical_sha256": config_fingerprint(config),
            "physical_sha256": config_snapshot.sha256,
        },
        "implementation": {
            "path": IMPLEMENTATION_PATH,
            "sha256": implementation.sha256,
        },
        "adapter": {
            "path": str(config["adapter"]["path"]),
            "file_sha256": adapter["file_sha256"],
            "source_profile": "q_plus_o",
        },
        "dataset": {
            "manifest_sha256": dataset.manifest_sha256,
            "partition_sha256": dict(dataset.partition_sha256),
            "train_records": len(dataset.train),
            "eval_proxy_records": len(dataset.eval_proxy),
        },
        "ood_proxy": inventory,
        "modes": list(MODES),
        "output_dir": output.relative_to(_root()).as_posix(),
        "claims": dict(config["claims"]),
        "audit": {
            "model_loads": 0,
            "gpu_requests": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "heldout_reads": 0,
            "protected_body_reads": 0,
        },
    }
    config_snapshot.assert_unchanged()
    implementation.assert_unchanged()
    assert_adapter_unchanged(adapter)
    return report


def _output_path(config: Mapping[str, Any]) -> Path:
    relative = Path(str(config["audit"]["output_dir"]))
    output = (_root() / relative).resolve()
    diagnostics = (_root() / "artifacts" / "diagnostics").resolve()
    if diagnostics not in output.parents or output == diagnostics:
        raise ConfigError("Q/O audit output escaped artifacts/diagnostics")
    if os.path.lexists(output):
        raise ConfigError(f"Q/O audit output already exists: {output}")
    return output


def _base_config(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    return {
        "schema_version": qdiag.SCHEMA_VERSION,
        "paths": {"project_root": "../.."},
        "model": dict(config["model"]),
        "lora": {
            "rank": 4,
            "alpha": 8,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": ["q_proj"],
        },
        "training": {
            "max_steps": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "sequence_length": 128,
            "learning_rate": 0.0001,
            "seed": 1337,
        },
        "dataset": {
            "kind": "inline_toy_plumbing_v1",
            "formal_inputs_allowed": False,
            "heldout_allowed": False,
        },
        "output": {"adapter_dir": output.relative_to(_root()).as_posix()},
        "_config_path": config["_config_path"],
    }


def _lora_state_sha256(model: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        count += 1
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(qdiag._tensor_sha256(parameter, torch).encode("ascii"))
        digest.update(b"\n")
    if count != 112:
        raise RuntimeError(f"Q+O adapter exposes {count} tensors, expected 112")
    return digest.hexdigest()


@contextlib.contextmanager
def contribution_view(model: Any, mode: str) -> Iterator[None]:
    if mode not in MODES:
        raise ValueError(f"unknown Q/O contribution mode: {mode}")
    if mode == "adapter_off":
        if not hasattr(model, "disable_adapter"):
            raise RuntimeError("PEFT model has no disable_adapter context")
        with model.disable_adapter():
            yield
        return
    disabled = {
        "full": set(),
        "q_only_contribution": {"o_proj"},
        "o_only_contribution": {"q_proj"},
    }[mode]
    restored: list[tuple[Any, str, float]] = []
    observed = {"q_proj": 0, "o_proj": 0}
    try:
        for name, module in model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in observed or not hasattr(module, "scaling"):
                continue
            scaling = module.scaling
            if not isinstance(scaling, dict) or set(scaling) != {"default"}:
                raise RuntimeError(f"unexpected PEFT scaling map at {name}")
            observed[leaf] += 1
            if leaf in disabled:
                old = float(scaling["default"])
                restored.append((module, "default", old))
                scaling["default"] = 0.0
        if observed != {"q_proj": 28, "o_proj": 28}:
            raise RuntimeError(f"Q/O module coverage drifted: {observed}")
        yield
    finally:
        for module, adapter_name, value in restored:
            module.scaling[adapter_name] = value


def _metrics(
    model: Any,
    tokenizer: Any,
    examples: Sequence[synth.ScaffoldExample],
    sequence_length: int,
    torch: Any,
) -> dict[str, Any]:
    model.eval()
    losses: list[tuple[float, int]] = []
    with torch.no_grad():
        for example in examples:
            batch = synth._cuda_batch(tokenizer, example, sequence_length, torch)
            loss = model(**batch, use_cache=False).loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("teacher-forced audit produced non-finite loss")
            target_tokens = int(
                torch.count_nonzero(batch["labels"][:, 1:] != -100).item()
            )
            losses.append((float(loss.detach().cpu()), target_tokens))
    tokens = sum(tokens for _, tokens in losses)
    macro = sum(loss for loss, _ in losses) / len(losses)
    micro = sum(loss * tokens for loss, tokens in losses) / tokens
    return {
        "records": len(losses),
        "target_tokens": tokens,
        "macro_loss": macro,
        "micro_target_token_nll": micro,
        "micro_target_token_ppl": math.exp(micro),
    }


def execute(config: Mapping[str, Any]) -> dict[str, Any]:
    preflight = build_preflight(config)
    authenticated = authenticate_adapter(config)
    dataset = synth.load_dataset(config)
    ood = build_ood_examples(int(config["audit"]["ood_generator_seed"]))
    _body_digest_inventory(dataset, ood)
    output = _output_path(config)
    base_config = _base_config(config, output)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Q/O memory audit requires BF16-capable CUDA")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started = time.perf_counter()
    with qdiag._authenticated_model_snapshot(base_config, output.parent) as (
        model_path,
        snapshot_identity,
    ):
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to("cuda")
        captured, base_hash_before = qdiag._capture_base_parameters(base, torch)
        model = PeftModel.from_pretrained(
            base,
            authenticated["path"],
            is_trainable=False,
            local_files_only=True,
        )
        tokenizer = synth._load_tokenizer_from_path(model_path)
        sequence_length = int(config["audit"]["sequence_length"])
        for example in (*dataset.train, *dataset.eval_proxy, *ood):
            synth.tokenize_example(
                tokenizer, example, sequence_length=sequence_length, torch=None
            )
        adapter_hash_before = _lora_state_sha256(model, torch)
        torch.cuda.reset_peak_memory_stats()
        results: dict[str, Any] = {}
        for mode in MODES:
            with contribution_view(model, mode):
                train_metrics = _metrics(
                    model, tokenizer, dataset.train, sequence_length, torch
                )
                eval_metrics = _metrics(
                    model, tokenizer, dataset.eval_proxy, sequence_length, torch
                )
                ood_metrics = _metrics(model, tokenizer, ood, sequence_length, torch)
            results[mode] = {
                "train": train_metrics,
                "eval_proxy": eval_metrics,
                "ood_proxy": ood_metrics,
                "generalization_gap_macro": eval_metrics["macro_loss"]
                - train_metrics["macro_loss"],
                "generalization_gap_micro": eval_metrics["micro_target_token_nll"]
                - train_metrics["micro_target_token_nll"],
            }
        torch.cuda.synchronize()
        adapter_hash_after = _lora_state_sha256(model, torch)
        if adapter_hash_after != adapter_hash_before:
            raise RuntimeError(
                "in-memory Q/O contribution audit mutated adapter weights"
            )
        base_hash_after = qdiag._assert_base_parameters_unchanged(captured, torch)
        if base_hash_after != base_hash_before:
            raise RuntimeError("Q/O contribution audit mutated base weights")
        assert_adapter_unchanged(authenticated)
        receipt = {
            **preflight,
            "status": "passed_read_only_teacher_forced_proxy_audit",
            "ready": True,
            "results": results,
            "immutability": {
                "base_hash_before": base_hash_before,
                "base_hash_after": base_hash_after,
                "adapter_state_sha256_before": adapter_hash_before,
                "adapter_state_sha256_after": adapter_hash_after,
            },
            "model_snapshot": dict(snapshot_identity),
            "performance": {
                "wall_seconds": time.perf_counter() - started,
                "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
            },
            "audit": {
                "model_loads": 1,
                "gpu_requests": 1,
                "provider_requests": 0,
                "network_requests": 0,
                "heldout_reads": 0,
                "protected_body_reads": 0,
            },
        }
        _publish_receipt(output, receipt)
        return receipt


def _publish_receipt(output: Path, receipt: Mapping[str, Any]) -> None:
    data = _canonical_json_bytes(receipt)
    digest = _sha(data)
    with qdiag._adapter_staging_directory(output) as staging:
        (staging / "audit_receipt.json").write_bytes(data)
        (staging / "audit_receipt.json.sha256").write_bytes(
            f"{digest}  audit_receipt.json\n".encode("ascii")
        )
        _validate_output(staging, data, digest)
        qdiag._rename_directory_noreplace(staging, output)
    try:
        _validate_output(output, data, digest)
    except Exception:
        synth._remove_failed_publish(output)
        raise


def _validate_output(path: Path, data: bytes, digest: str) -> None:
    _exact_files(path, _OUTPUT_FILES, "Q/O audit output")
    receipt = synth._read_snapshot(
        path / "audit_receipt.json", max_bytes=_MAX_METADATA_BYTES
    )
    sidecar = synth._read_snapshot(path / "audit_receipt.json.sha256", max_bytes=1024)
    if (
        receipt.data != data
        or receipt.sha256 != digest
        or sidecar.data != f"{digest}  audit_receipt.json\n".encode("ascii")
    ):
        raise RuntimeError("Q/O audit output authentication failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Q/O contribution audit")
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    result = execute(config) if args.execute else build_preflight(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
