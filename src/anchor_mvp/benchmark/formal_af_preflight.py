"""Offline readiness gate for the registered formal A--F benchmarks.

The gate deliberately does not open the held-out case file.  It validates only
frozen metadata, run registries, adapter hashes, allocation manifests and the
fairness contract.  A live evaluator remains responsible for the held-out
integrity check immediately before execution.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ..training.experiment_registry import (
    ExperimentRegistryError,
    verify_registry,
)
from .heldout import PRIMARY_STAGES, HeldoutGateError, validate_primary_specs
from .models import load_specs
from .segment_protocol import (
    ARTIFACT_PROTOCOL,
    PROMPT_BUNDLE_SHA256,
    PROMPT_BUNDLE_VERSION,
    SEGMENT_CONTRACT_VERSION,
    SEGMENTED_REVIEW_PROTOCOL,
    SegmentContract,
    prompt_bundle_payload,
)


SCHEMA = "anchor.formal-af-benchmark.v1"
LEGACY_RUN_ID = "formal-partial-v1-forced-20260715-v1"
# Kept as a public compatibility alias.  Only the legacy repair-code contract
# is pinned to this value; segmented formal-v2 run IDs are registry-bound.
RUN_ID = LEGACY_RUN_ID
GROUPS = ("A", "B", "C", "D", "E", "F")
BASELINE_TO_GROUP = {
    "base_matched_calls": "A",
    "mixed_matched_calls": "B",
    "c_pipeline": "C",
    "d_budget_matched_pipeline": "D",
    "e_adaptive_pareto_pipeline": "E",
    "f_adaptive_budget_matched_pipeline": "F",
}
EXPECTED_RANKS = {
    "A": {},
    "B": {"mixed_all": 16},
    "C": {
        "planner": 16,
        "tool_policy": 16,
        "frontend_gen": 16,
        "frontend_review": 16,
        "security_gate": 16,
    },
    "D": {
        "planner": 3,
        "tool_policy": 3,
        "frontend_gen": 4,
        "frontend_review": 3,
        "security_gate": 3,
    },
    "E": {
        "planner": 8,
        "tool_policy": 4,
        "frontend_gen": 16,
        "frontend_review": 12,
        "security_gate": 4,
    },
    "F": {
        "planner": 4,
        "tool_policy": 1,
        "frontend_gen": 6,
        "frontend_review": 4,
        "security_gate": 1,
    },
}
EXPECTED_PARAMETERS = {
    "A": 0,
    "B": 10_387_456,
    "C": 51_937_280,
    "D": 10_387_456,
    "E": 28_565_504,
    "F": 10_387_456,
}
STAGE_TO_REGISTRY_NAME = {
    "planner": "planner",
    "tool_policy": "tool_policy",
    "frontend": "frontend_gen",
    "review": "frontend_review",
    "security": "security_gate",
}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}")

_SEGMENTED_COUNTS = {
    "frontend_segment_count": 10,
    "review_segment_count": 10,
    "expected_physical_calls": 23,
    "max_tokens_per_call": 512,
}


class FormalAFPreflightError(ValueError):
    """A fail-closed A--F readiness check failed."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _inside(root: Path, raw: str, label: str) -> Path:
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FormalAFPreflightError(
            "unsafe_path", f"{label} escapes the project root"
        ) from exc
    return candidate


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalAFPreflightError("invalid_contract", f"{label} must be an object")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalAFPreflightError(
            "missing_or_invalid_metadata", f"{label} is missing or invalid"
        ) from exc
    if not isinstance(value, dict):
        raise FormalAFPreflightError(
            "invalid_contract", f"{label} must contain one JSON object"
        )
    return value


def _validate_hash(value: Any, label: str) -> str:
    text = str(value)
    if not _HEX64.fullmatch(text):
        raise FormalAFPreflightError(
            "invalid_contract", f"{label} must be a lowercase SHA-256"
        )
    return text


def _run_id(config: Mapping[str, Any], *, segmented: bool) -> str:
    run_id = str(config.get("run_id", ""))
    if not _RUN_ID.fullmatch(run_id):
        raise FormalAFPreflightError(
            "run_id_mismatch", "formal benchmark run_id is not a safe registry ID"
        )
    if not segmented and run_id != LEGACY_RUN_ID:
        raise FormalAFPreflightError(
            "run_id_mismatch",
            f"legacy formal benchmark must bind run_id {LEGACY_RUN_ID}",
        )
    return run_id


def _validate_segmented_binding_shapes(config: Mapping[str, Any]) -> None:
    for key, path_key in (
        ("segment_contract_binding", "path"),
        ("prompt_bundle_binding", "path"),
    ):
        binding = _mapping(config.get(key), key)
        if not str(binding.get(path_key, "")).strip():
            raise FormalAFPreflightError(
                "invalid_contract", f"{key}.{path_key} must be non-empty"
            )
        _validate_hash(binding.get("sha256"), f"{key} digest")

    processor = _mapping(config.get("processor_binding"), "processor_binding")
    if not str(processor.get("manifest_path", "")).strip():
        raise FormalAFPreflightError(
            "invalid_contract", "processor_binding.manifest_path must be non-empty"
        )
    _validate_hash(
        processor.get("manifest_sha256"), "processor manifest digest"
    )
    _validate_hash(processor.get("tree_sha256"), "processor tree digest")


def _expected_registry_ranks(arm: Mapping[str, Any], group: str) -> dict[str, int]:
    stage_ranks = {
        str(key): int(value)
        for key, value in _mapping(
            arm.get("stage_adapter_ranks", {}),
            f"group {group} stage_adapter_ranks",
        ).items()
    }
    if group == "A":
        return {}
    if group == "B":
        if set(stage_ranks.values()) != {16}:
            raise FormalAFPreflightError(
                "rank_contract_mismatch", "B must reuse one rank-16 mixed adapter"
            )
        return {"mixed_all": 16}
    try:
        return {
            STAGE_TO_REGISTRY_NAME[stage]: stage_ranks[stage]
            for stage in PRIMARY_STAGES
        }
    except KeyError as exc:
        raise FormalAFPreflightError(
            "rank_contract_mismatch", f"group {group} has incomplete stage ranks"
        ) from exc


def validate_contract(config_path: str | Path) -> dict[str, Any]:
    """Validate static A--F controls without opening registries or held-out cases."""

    path = Path(config_path).resolve()
    config = _load_json(path, "formal A--F benchmark config")
    if config.get("schema_version") != SCHEMA:
        raise FormalAFPreflightError("invalid_contract", "unsupported A--F schema")

    base = _mapping(config.get("base_binding"), "base_binding")
    source_sha = _validate_hash(base.get("base_source_sha256"), "base source digest")
    q4_sha = _validate_hash(base.get("q4_artifact_sha256"), "Q4 artifact digest")
    _validate_hash(base.get("weight_set_sha256"), "base weight-set digest")
    if base.get("format") != "transformers-bitsandbytes-nf4":
        raise FormalAFPreflightError(
            "base_format_mismatch", "formal run requires the frozen bitsandbytes NF4 base"
        )

    token = _mapping(config.get("token_contract"), "token_contract")
    if tuple(token.get("stages", ())) != PRIMARY_STAGES:
        raise FormalAFPreflightError(
            "token_contract_mismatch", "all arms require the same five ordered stages"
        )
    token_cap = int(token.get("max_tokens_per_call", 0))
    review_protocol = str(token.get("review_protocol", ""))
    segmented = review_protocol == SEGMENTED_REVIEW_PROTOCOL
    _run_id(config, segmented=segmented)
    if segmented:
        expected = {
            "frontend_segment_count": int(
                token.get("frontend_segment_count", 0)
            ),
            "review_segment_count": int(token.get("review_segment_count", 0)),
            "expected_physical_calls": int(
                token.get("expected_physical_calls", 0)
            ),
            "max_tokens_per_call": token_cap,
        }
        if expected != _SEGMENTED_COUNTS:
            raise FormalAFPreflightError(
                "token_contract_mismatch",
                "formal-v2 requires exactly 10/10 segments, 23 calls and cap 512",
            )
        if (
            token.get("artifact_protocol") != ARTIFACT_PROTOCOL
            or token.get("segment_contract_version")
            != SEGMENT_CONTRACT_VERSION
        ):
            raise FormalAFPreflightError(
                "token_contract_mismatch",
                "formal-v2 changed the segmented artifact contract",
            )
        sampling = _mapping(token.get("sampling"), "token_contract.sampling")
        if sampling.get("temperature") != 0.0 or sampling.get("top_p") != 1.0:
            raise FormalAFPreflightError(
                "token_contract_mismatch",
                "formal-v2 requires deterministic temperature=0 and top_p=1",
            )
        _validate_segmented_binding_shapes(config)
    elif (
        review_protocol != "repair_code_v1"
        or int(token.get("repair_code_review_calls", 0)) != 1
        or token_cap <= 0
        or int(token.get("max_review_cycles", -1)) != 2
    ):
        raise FormalAFPreflightError(
            "token_contract_mismatch",
            "formal partial-v1 requires one repair_code_v1 reviewer call",
        )

    specs = load_specs(path)
    try:
        validate_primary_specs(specs, require_verified_q4=True)
    except HeldoutGateError as exc:
        raise FormalAFPreflightError("fairness_contract_mismatch", str(exc)) from exc
    raw_arms = config.get("baselines")
    if not isinstance(raw_arms, list):
        raise FormalAFPreflightError("invalid_contract", "baselines must be a list")
    by_name = {str(item.get("name")): item for item in raw_arms if isinstance(item, Mapping)}
    if set(by_name) != set(BASELINE_TO_GROUP):
        raise FormalAFPreflightError(
            "arm_set_mismatch", "formal benchmark requires exactly A through F"
        )

    for name, group in BASELINE_TO_GROUP.items():
        arm = _mapping(by_name[name], f"group {group}")
        if arm.get("registry_group") != group:
            raise FormalAFPreflightError(
                "registry_binding_mismatch", f"baseline {name} must bind registry group {group}"
            )
        if arm.get("base_source_sha256") != source_sha or arm.get(
            "q4_artifact_sha256"
        ) != q4_sha:
            raise FormalAFPreflightError(
                "base_binding_mismatch", f"group {group} changed the frozen base"
            )
        if int(arm.get("max_tokens_per_call", 0)) != token_cap:
            raise FormalAFPreflightError(
                "token_contract_mismatch", f"group {group} changed the token cap"
            )
        if arm.get("review_protocol") != review_protocol:
            raise FormalAFPreflightError(
                "token_contract_mismatch",
                f"group {group} changed the frozen review protocol",
            )
        if segmented and {
            "artifact_protocol": arm.get("artifact_protocol"),
            "segment_contract_version": arm.get("segment_contract_version"),
            "frontend_segment_count": int(arm.get("frontend_segment_count", 0)),
            "review_segment_count": int(arm.get("review_segment_count", 0)),
        } != {
            "artifact_protocol": token["artifact_protocol"],
            "segment_contract_version": token["segment_contract_version"],
            "frontend_segment_count": token["frontend_segment_count"],
            "review_segment_count": token["review_segment_count"],
        }:
            raise FormalAFPreflightError(
                "token_contract_mismatch",
                f"group {group} changed the frozen segment contract",
            )
        observed_ranks = _expected_registry_ranks(arm, group)
        if observed_ranks != EXPECTED_RANKS[group]:
            raise FormalAFPreflightError(
                "rank_contract_mismatch", f"group {group} rank allocation changed"
            )
        parameters = int(arm.get("adapter_trainable_parameters", 0))
        if parameters != EXPECTED_PARAMETERS[group]:
            raise FormalAFPreflightError(
                "parameter_contract_mismatch",
                f"group {group} parameter budget changed",
            )
        artifacts = _mapping(
            arm.get("stage_adapter_artifacts", {}),
            f"group {group} stage_adapter_artifacts",
        )
        if group == "A" and artifacts:
            raise FormalAFPreflightError(
                "adapter_contract_mismatch", "A must not bind an adapter"
            )
        if group != "A" and set(artifacts) != set(PRIMARY_STAGES):
            raise FormalAFPreflightError(
                "adapter_contract_mismatch", f"group {group} must bind all five stages"
            )

    for group in ("E", "F"):
        name = next(name for name, value in BASELINE_TO_GROUP.items() if value == group)
        arm = by_name[name]
        if arm.get("calibration_status") != "calibration_pending":
            raise FormalAFPreflightError(
                "calibration_label_mismatch",
                f"group {group} must retain calibration_pending disclosure",
            )
        if not arm.get("allocation_frozen") or arm.get("status") != "ready":
            raise FormalAFPreflightError(
                "allocation_not_frozen", f"group {group} allocation is not frozen"
            )

    metrics = _mapping(config.get("metrics_plan"), "metrics_plan")
    if metrics.get("index_baseline") != "A" or metrics.get("index_value") != 100:
        raise FormalAFPreflightError("metrics_contract_mismatch", "A must be indexed at 100")
    if tuple(metrics.get("equal_budget_comparison", ())) != ("B", "D", "F"):
        raise FormalAFPreflightError(
            "metrics_contract_mismatch", "equal-budget comparison must be B/D/F"
        )
    if tuple(metrics.get("capacity_comparison", ())) != ("C", "E"):
        raise FormalAFPreflightError(
            "metrics_contract_mismatch", "capacity comparison must be C/E"
        )
    backend_audit = _mapping(config.get("backend_audit"), "backend_audit")
    serial = _mapping(
        backend_audit.get("vllm_serial_runtime_lora"),
        "backend_audit.vllm_serial_runtime_lora",
    )
    if not (
        serial.get("eligible") is True
        and serial.get("catalog_gate") == "base_model_only_before_heldout"
        and serial.get("maximum_active_loras") == 1
        and serial.get("maximum_cpu_loras") == 1
        and serial.get("allow_static_lora_modules") is False
        and serial.get("require_localhost_admin") is True
        and serial.get("server_project_root_transport")
        == "explicit_absolute_posix"
    ):
        raise FormalAFPreflightError(
            "backend_contract_mismatch",
            "formal vLLM must keep one local runtime LoRA and no static adapter set",
        )
    llama = _mapping(
        backend_audit.get("llama_cpp"),
        "backend_audit.llama_cpp",
    )
    if llama.get("eligible_for_this_frozen_contract") is not False:
        raise FormalAFPreflightError(
            "backend_contract_mismatch",
            "llama.cpp must remain blocked for the NF4/PEFT contract",
        )

    return config


def _verify_frozen_metadata(root: Path, config: Mapping[str, Any]) -> None:
    heldout = _mapping(config.get("heldout_binding"), "heldout_binding")
    if heldout.get("preflight_reads_case_content") is not False:
        raise FormalAFPreflightError(
            "heldout_access_contract_mismatch",
            "offline preflight must not read held-out case content",
        )
    for key in ("manifest", "leak_audit"):
        path = _inside(root, str(heldout.get(f"{key}_path", "")), f"held-out {key}")
        expected = _validate_hash(heldout.get(f"{key}_sha256"), f"held-out {key} digest")
        if sha256_file(path) != expected:
            raise FormalAFPreflightError(
                "heldout_metadata_changed", f"held-out {key} metadata digest changed"
            )


def _verify_segmented_execution_bindings(
    root: Path, config: Mapping[str, Any]
) -> dict[str, str]:
    """Verify public formal-v2 protocol artifacts without opening case content."""

    token = _mapping(config.get("token_contract"), "token_contract")
    segment_binding = _mapping(
        config.get("segment_contract_binding"), "segment_contract_binding"
    )
    segment_path = _inside(
        root, str(segment_binding.get("path", "")), "segment contract"
    )
    segment_sha = _validate_hash(
        segment_binding.get("sha256"), "segment contract digest"
    )
    if sha256_file(segment_path) != segment_sha:
        raise FormalAFPreflightError(
            "segment_contract_changed", "segmented evaluation contract changed"
        )
    segment = _load_json(segment_path, "segment contract")
    observed_segment_contract = {
        "artifact_protocol": segment.get("artifact_protocol"),
        "segment_contract_version": segment.get("segment_contract_version"),
        "review_protocol": segment.get("review_protocol"),
        "frontend_segment_count": segment.get("frontend_segment_count"),
        "review_segment_count": segment.get("review_segment_count"),
        "expected_physical_calls": segment.get("expected_physical_calls"),
        "max_tokens_per_call": segment.get(
            "max_completion_tokens_per_physical_call"
        ),
    }
    expected_segment_contract = {
        "artifact_protocol": token.get("artifact_protocol"),
        "segment_contract_version": token.get("segment_contract_version"),
        "review_protocol": token.get("review_protocol"),
        "frontend_segment_count": token.get("frontend_segment_count"),
        "review_segment_count": token.get("review_segment_count"),
        "expected_physical_calls": token.get("expected_physical_calls"),
        "max_tokens_per_call": token.get("max_tokens_per_call"),
    }
    if observed_segment_contract != expected_segment_contract:
        raise FormalAFPreflightError(
            "segment_contract_mismatch",
            "segment contract file differs from the formal-v2 token contract",
        )

    prompt_binding = _mapping(
        config.get("prompt_bundle_binding"), "prompt_bundle_binding"
    )
    prompt_path = _inside(
        root, str(prompt_binding.get("path", "")), "prompt bundle"
    )
    prompt_sha = _validate_hash(
        prompt_binding.get("sha256"), "prompt bundle digest"
    )
    if sha256_file(prompt_path) != prompt_sha:
        raise FormalAFPreflightError(
            "prompt_bundle_changed", "formal-v2 public prompt bundle changed"
        )
    prompt_bundle = _load_json(prompt_path, "prompt bundle")
    if (
        prompt_bundle.get("schema_version")
        != "anchor.formal-v2-prompt-bundle.v1"
        or prompt_bundle.get("prompt_bundle_version") != PROMPT_BUNDLE_VERSION
        or prompt_bundle.get("canonical_prompt_bundle_sha256")
        != PROMPT_BUNDLE_SHA256
        or prompt_bundle.get("payload") != prompt_bundle_payload()
    ):
        raise FormalAFPreflightError(
            "prompt_bundle_contract_mismatch",
            "prompt bundle differs from the executable public prompt contract",
        )
    implementation = _mapping(
        prompt_bundle.get("implementation"), "prompt bundle implementation"
    )
    implementation_path = _inside(
        root,
        str(implementation.get("path", "")),
        "prompt bundle implementation",
    )
    implementation_sha = _validate_hash(
        implementation.get("sha256"), "prompt implementation digest"
    )
    if sha256_file(implementation_path) != implementation_sha:
        raise FormalAFPreflightError(
            "prompt_implementation_changed",
            "prompt implementation differs from the frozen prompt bundle",
        )

    processor_binding = _mapping(
        config.get("processor_binding"), "processor_binding"
    )
    processor_manifest_path = _inside(
        root,
        str(processor_binding.get("manifest_path", "")),
        "processor manifest",
    )
    processor_manifest_sha = _validate_hash(
        processor_binding.get("manifest_sha256"), "processor manifest digest"
    )
    if sha256_file(processor_manifest_path) != processor_manifest_sha:
        raise FormalAFPreflightError(
            "processor_manifest_changed", "formal-v2 processor manifest changed"
        )
    processor_manifest = _load_json(
        processor_manifest_path, "processor manifest"
    )
    processor_tree_sha = _validate_hash(
        processor_binding.get("tree_sha256"), "processor tree digest"
    )
    if (
        processor_manifest.get("schema_version")
        != "anchor.formal-af-processor.v1"
        or processor_manifest.get("tree_sha256") != processor_tree_sha
    ):
        raise FormalAFPreflightError(
            "processor_binding_mismatch",
            "processor manifest tree differs from the formal-v2 binding",
        )
    return {
        "segment_contract_manifest_sha256": segment_sha,
        "segment_contract_sha256": SegmentContract(
            artifact_protocol=str(token["artifact_protocol"]),
            contract_version=str(token["segment_contract_version"]),
            frontend_segments=int(token["frontend_segment_count"]),
            review_segments=int(token["review_segment_count"]),
        ).binding_sha256(
            max_completion_tokens_per_physical_call=int(
                token["max_tokens_per_call"]
            )
        ),
        "prompt_bundle_manifest_sha256": prompt_sha,
        "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        "processor_manifest_sha256": processor_manifest_sha,
        "processor_tree_sha256": processor_tree_sha,
    }


def _verify_allocation_manifests(root: Path, config: Mapping[str, Any]) -> None:
    by_group = {
        str(item["registry_group"]): item
        for item in config["baselines"]
        if isinstance(item, Mapping)
    }
    for group in ("E", "F"):
        arm = by_group[group]
        path = _inside(
            root,
            str(arm.get("allocation_manifest_path", "")),
            f"group {group} allocation manifest",
        )
        expected = _validate_hash(
            arm.get("allocation_manifest_sha256"),
            f"group {group} allocation digest",
        )
        if sha256_file(path) != expected:
            raise FormalAFPreflightError(
                "allocation_manifest_changed",
                f"group {group} allocation manifest changed",
            )


def preflight(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Verify that every A--F artifact is complete and immutable.

    No model, GPU backend, network client, held-out case file, or fixture is opened.
    """

    root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = validate_contract(config_path)
    token_contract = _mapping(config.get("token_contract"), "token_contract")
    segmented = token_contract.get("review_protocol") == SEGMENTED_REVIEW_PROTOCOL
    run_id = _run_id(config, segmented=segmented)
    run_path = _inside(
        root, str(config.get("run_manifest_path", "")), "run manifest"
    )
    run = _load_json(run_path, "run manifest")
    if run.get("schema_version") != "anchor.af-run-manifest.v1":
        raise FormalAFPreflightError("run_manifest_invalid", "unsupported run manifest")
    if run.get("run_id") != run_id:
        raise FormalAFPreflightError("run_id_mismatch", "run manifest changed run_id")

    base = _mapping(config.get("base_binding"), "base_binding")
    registered_base = _mapping(run.get("base_artifact"), "run base_artifact")
    expected_base = {
        "format": base["format"],
        "manifest_sha256": base["q4_artifact_sha256"],
        "source_weight_sha256": base["base_source_sha256"],
        "weight_set_sha256": base["weight_set_sha256"],
    }
    for key, expected in expected_base.items():
        if registered_base.get(key) != expected:
            raise FormalAFPreflightError(
                "base_binding_mismatch", f"run manifest changed base {key}"
            )
    dataset = _mapping(run.get("dataset_snapshot"), "run dataset_snapshot")
    expected_dataset = _mapping(config.get("dataset_binding"), "dataset_binding")
    if dataset.get("snapshot_sha256") != expected_dataset.get("snapshot_sha256"):
        raise FormalAFPreflightError(
            "dataset_binding_mismatch", "run manifest changed dataset snapshot"
        )
    if dataset.get("not_for_end_to_end_claim") is not True:
        raise FormalAFPreflightError(
            "dataset_claim_mismatch", "partial dataset limitation is missing"
        )

    gate = _mapping(config.get("registry_gate"), "registry_gate")
    if tuple(gate.get("required_groups", ())) != GROUPS:
        raise FormalAFPreflightError(
            "registry_gate_mismatch", "registry gate must require exactly A through F"
        )
    run_groups = _mapping(run.get("groups"), "run groups")
    registry_paths: dict[str, Path] = {}
    missing: list[str] = []
    for group in GROUPS:
        entry = _mapping(run_groups.get(group), f"run group {group}")
        path = _inside(root, str(entry.get("registry_path", "")), f"group {group} registry")
        registry_paths[group] = path
        if not path.is_file():
            missing.append(group)
    if missing:
        raise FormalAFPreflightError(
            "training_incomplete",
            "A--F registries are not complete",
            details={"missing_registry_groups": missing},
        )

    execution_bindings = (
        _verify_segmented_execution_bindings(root, config) if segmented else {}
    )
    _verify_frozen_metadata(root, config)
    _verify_allocation_manifests(root, config)
    by_group = {
        str(item["registry_group"]): item
        for item in config["baselines"]
        if isinstance(item, Mapping)
    }
    required_status = _mapping(gate.get("required_status"), "required_status")
    registry_locks: dict[str, Any] = {}
    runtime_bindings: dict[str, Any] = {}
    for group in GROUPS:
        try:
            verified = verify_registry(root, run_path.parent, group=group)
        except ExperimentRegistryError as exc:
            raise FormalAFPreflightError(
                "registry_verification_failed",
                f"group {group} registry verification failed: {exc}",
            ) from exc
        registry = _load_json(registry_paths[group], f"group {group} registry")
        if registry.get("status") != required_status.get(group):
            raise FormalAFPreflightError(
                "training_incomplete", f"group {group} is not in its terminal status"
            )
        summary = _mapping(registry.get("adapter_summary"), f"group {group} summary")
        observed_ranks = {
            str(key): int(value)
            for key, value in _mapping(
                summary.get("ranks", {}), f"group {group} ranks"
            ).items()
        }
        if observed_ranks != EXPECTED_RANKS[group]:
            raise FormalAFPreflightError(
                "registry_rank_mismatch", f"group {group} indexed unexpected ranks"
            )
        if int(summary.get("trainable_parameter_total", 0)) != EXPECTED_PARAMETERS[group]:
            raise FormalAFPreflightError(
                "registry_parameter_mismatch",
                f"group {group} indexed an unexpected parameter total",
            )
        records = registry.get("adapters", [])
        if not isinstance(records, list):
            raise FormalAFPreflightError(
                "registry_adapter_mismatch", f"group {group} adapter index is invalid"
            )
        by_artifact = {str(item.get("artifact_name")): item for item in records}
        expected_artifacts = {
            str(value)
            for value in _mapping(
                by_group[group].get("stage_adapter_artifacts", {}),
                f"group {group} stage artifacts",
            ).values()
        }
        if set(by_artifact) != expected_artifacts:
            raise FormalAFPreflightError(
                "registry_adapter_mismatch",
                f"group {group} adapter artifacts differ from the frozen routing map",
            )
        stage_models = _mapping(
            by_group[group].get("stage_models", {}), f"group {group} stage models"
        )
        stage_artifacts = _mapping(
            by_group[group].get("stage_adapter_artifacts", {}),
            f"group {group} stage artifacts",
        )
        group_runtime: dict[str, Any] = {}
        for stage in PRIMARY_STAGES:
            model_id = str(stage_models.get(stage, ""))
            if not model_id:
                raise FormalAFPreflightError(
                    "runtime_binding_mismatch",
                    f"group {group} has no model id for stage {stage}",
                )
            if group == "A":
                group_runtime[stage] = {
                    "model_id": model_id,
                    "adapter_artifact": None,
                    "adapter_dir": None,
                    "adapter_sha256": None,
                }
                continue
            artifact = str(stage_artifacts[stage])
            record = _mapping(by_artifact[artifact], f"group {group} adapter {artifact}")
            final_files = _mapping(
                record.get("final_files"), f"group {group} adapter final_files"
            )
            adapter_model = _mapping(
                final_files.get("adapter_model"),
                f"group {group} adapter model file",
            )
            adapter_model_path = _inside(
                root,
                str(adapter_model.get("path", "")),
                f"group {group} adapter model path",
            )
            group_runtime[stage] = {
                "model_id": model_id,
                "adapter_artifact": artifact,
                "adapter_dir": adapter_model_path.parent.relative_to(root).as_posix(),
                "adapter_sha256": str(record.get("adapter_sha256")),
            }
        runtime_bindings[group] = group_runtime
        registry_locks[group] = {
            "registry_sha256": verified["registry_sha256"],
            "status": registry["status"],
            "ranks": observed_ranks,
            "adapter_sha256": {
                artifact: str(record.get("adapter_sha256"))
                for artifact, record in sorted(by_artifact.items())
            },
        }

    serial = _mapping(
        _mapping(config.get("backend_audit"), "backend_audit").get(
            "vllm_serial_runtime_lora"
        ),
        "backend_audit.vllm_serial_runtime_lora",
    )
    result = {
        "schema_version": "anchor.formal-af-preflight.v1",
        "status": "ready",
        "execution_authorized": True,
        "offline_only": True,
        "heldout_case_content_read": False,
        "run_id": run_id,
        "config_sha256": sha256_file(config_path),
        "run_manifest_sha256": sha256_file(run_path),
        "base_q4_artifact_sha256": base["q4_artifact_sha256"],
        "per_stage_token_cap": token_contract["max_tokens_per_call"],
        "review_protocol": token_contract["review_protocol"],
        "registry_locks": registry_locks,
        "runtime_bindings": runtime_bindings,
        "comparison_plan": {
            "A_index": 100,
            "equal_budget": ["B", "D", "F"],
            "capacity": ["C", "E"],
            "E_calibration_status": "calibration_pending",
        },
        "serial_runtime_contract": {
            "base_model_id": runtime_bindings["A"]["planner"]["model_id"],
            "maximum_active_loras": serial["maximum_active_loras"],
            "maximum_cpu_loras": serial["maximum_cpu_loras"],
            "catalog_gate": serial["catalog_gate"],
            "allow_static_lora_modules": serial["allow_static_lora_modules"],
            "require_localhost_admin": serial["require_localhost_admin"],
            "server_project_root_transport": serial[
                "server_project_root_transport"
            ],
        },
        "llama_cpp_eligible": False,
    }
    if not segmented:
        result["repair_code_review_calls"] = 1
        return result

    runtime_bindings_sha = _canonical_sha256(runtime_bindings)
    sampling = _mapping(token_contract.get("sampling"), "token_contract.sampling")
    execution_contract = {
        "run_id": run_id,
        **execution_bindings,
        "artifact_protocol": token_contract["artifact_protocol"],
        "segment_contract_version": token_contract["segment_contract_version"],
        "review_protocol": token_contract["review_protocol"],
        "frontend_segment_count": token_contract["frontend_segment_count"],
        "review_segment_count": token_contract["review_segment_count"],
        "expected_physical_calls": token_contract["expected_physical_calls"],
        "max_tokens_per_call": token_contract["max_tokens_per_call"],
        "sampling": {
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
        },
        "runtime_bindings_sha256": runtime_bindings_sha,
    }
    result.update(
        {
            **execution_bindings,
            "artifact_protocol": token_contract["artifact_protocol"],
            "segment_contract_version": token_contract["segment_contract_version"],
            "frontend_segment_count": token_contract["frontend_segment_count"],
            "review_segment_count": token_contract["review_segment_count"],
            "expected_physical_calls": token_contract["expected_physical_calls"],
            "runtime_bindings_sha256": runtime_bindings_sha,
            "execution_contract": execution_contract,
            "execution_contract_sha256": _canonical_sha256(execution_contract),
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline formal A--F benchmark preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="validate static controls without inspecting training registries",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.contract_only:
            config = validate_contract(args.config)
            result: dict[str, Any] = {
                "schema_version": "anchor.formal-af-preflight.v1",
                "status": "contract_valid",
                "execution_authorized": False,
                "offline_only": True,
                "heldout_case_content_read": False,
                "run_id": config["run_id"],
                "config_sha256": sha256_file(args.config),
            }
        else:
            result = preflight(args.config, args.project_root)
    except FormalAFPreflightError as exc:
        result = {
            "schema_version": "anchor.formal-af-preflight.v1",
            "status": "blocked",
            "execution_authorized": False,
            "offline_only": True,
            "heldout_case_content_read": False,
            "error_code": exc.code,
            "error": str(exc),
            **exc.details,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
