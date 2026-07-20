"""Read-only, heldout-body-blind preflight for formal-v3 A--F evaluation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .formal_v3_registry import (
    ARMS,
    BENCHMARK_SCHEMA,
    BUNDLE_SCHEMA,
    REGISTRY_CONTRACT_KEYS,
    REGISTRY_SCHEMA,
    STAGES,
    TRAINING_NAMES,
    _allocation_binding,
    _canonical_bytes,
    _forbid_formal_v2,
    _inside,
    _load_json,
    _mapping,
    _require_sha,
    _verify_sidecar,
)
from .heldout import HeldoutGateError, validate_primary_specs
from .models import load_specs
from ..training.manifest import sha256_file


PREFLIGHT_SCHEMA = "anchor.formal-v3-af-preflight.v1"


class FormalV3PreflightError(ValueError):
    """A finalized formal-v3 bundle is not safe to evaluate."""

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


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _translate_error(exc: Exception, code: str = "registry_invalid") -> None:
    if isinstance(exc, FormalV3PreflightError):
        raise exc
    raise FormalV3PreflightError(code, str(exc)) from exc


def _verify_file(root: Path, entry: Mapping[str, Any], label: str) -> Path:
    try:
        path = _inside(root, entry.get("path"), label)
        expected = _require_sha(entry.get("sha256"), f"{label} digest")
    except Exception as exc:  # translated to the public preflight error type
        _translate_error(exc)
    if not path.is_file() or sha256_file(path) != expected:
        raise FormalV3PreflightError(
            "artifact_changed", f"{label} is missing or changed"
        )
    expected_bytes = entry.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise FormalV3PreflightError(
            "artifact_changed", f"{label} byte size changed"
        )
    return path


def _verify_registry_sources(root: Path, registry: Mapping[str, Any]) -> None:
    snapshot = _mapping(registry.get("snapshot"), "registry snapshot")
    base = _mapping(registry.get("base"), "registry base")
    heldout = _mapping(registry.get("heldout"), "registry heldout")
    allocations = _mapping(registry.get("allocations"), "registry allocations")
    arms = _mapping(registry.get("arms"), "registry arms")

    for value, label in (
        (snapshot, "snapshot"),
        (base, "base manifest"),
        (heldout, "heldout manifest metadata"),
    ):
        path_key = "manifest_path"
        sha_key = "manifest_sha256"
        path = _inside(root, value.get(path_key), label)
        expected = _require_sha(value.get(sha_key), f"{label} digest")
        if not path.is_file() or sha256_file(path) != expected:
            raise FormalV3PreflightError(
                "artifact_changed", f"{label} changed after finalization"
            )
    snapshot_sidecar = _mapping(snapshot.get("sidecar"), "snapshot sidecar")
    snapshot_sidecar_path = _verify_file(
        root, snapshot_sidecar, "snapshot sidecar"
    )
    snapshot_path = _inside(root, snapshot.get("manifest_path"), "snapshot")
    try:
        declared_snapshot_sha = _verify_sidecar(
            snapshot_path, snapshot_sidecar_path, "formal-v3 snapshot"
        )
    except Exception as exc:
        _translate_error(exc, "snapshot_changed")
    if declared_snapshot_sha != snapshot.get("manifest_sha256"):
        raise FormalV3PreflightError(
            "snapshot_changed", "snapshot sidecar no longer binds the finalized manifest"
        )
    leak = _inside(root, heldout.get("leak_audit_path"), "heldout leak metadata")
    leak_sha = _require_sha(
        heldout.get("leak_audit_sha256"), "heldout leak metadata digest"
    )
    if not leak.is_file() or sha256_file(leak) != leak_sha:
        raise FormalV3PreflightError(
            "artifact_changed", "heldout leakage metadata changed after finalization"
        )
    if heldout.get("case_content_read") is not False or heldout.get("metadata_only") is not True:
        raise FormalV3PreflightError(
            "heldout_isolation_changed", "registry lost the heldout metadata-only contract"
        )

    weights = base.get("weights")
    if not isinstance(weights, list) or not weights:
        raise FormalV3PreflightError("base_changed", "base weight inventory is empty")
    for index, raw in enumerate(weights):
        _verify_file(root, _mapping(raw, f"base weight {index}"), f"base weight {index}")

    for arm in ("E", "F"):
        allocation = _mapping(allocations.get(arm), f"arm {arm} allocation")
        try:
            verified = _allocation_binding(
                root,
                _inside(root, allocation.get("path"), f"arm {arm} allocation"),
                snapshot,
                arm,
            )
        except Exception as exc:
            _translate_error(exc, "allocation_changed")
        if verified != allocation:
            raise FormalV3PreflightError(
                "allocation_changed",
                f"arm {arm} allocation differs from the finalized registry",
            )
    if allocations["E"]["mechanism_id"] != allocations["F"]["mechanism_id"]:
        raise FormalV3PreflightError(
            "allocation_changed", "E/F adaptive mechanisms no longer match"
        )

    if set(arms) != set(ARMS):
        raise FormalV3PreflightError("arm_set_changed", "registry must contain A through F")
    if arms["A"].get("adapters") != []:
        raise FormalV3PreflightError("arm_shape_changed", "A must have no adapter")
    for arm in ARMS[1:]:
        arm_value = _mapping(arms[arm], f"registry arm {arm}")
        schedule = _mapping(arm_value.get("schedule"), f"arm {arm} schedule")
        schedule_path = _inside(root, schedule.get("path"), f"arm {arm} schedule")
        if not schedule_path.is_file() or sha256_file(schedule_path) != schedule.get("sha256"):
            raise FormalV3PreflightError(
                "schedule_changed", f"arm {arm} schedule changed after finalization"
            )
        schedule_sidecar = _mapping(
            schedule.get("sidecar"), f"arm {arm} schedule sidecar"
        )
        schedule_sidecar_path = _verify_file(
            root, schedule_sidecar, f"arm {arm} schedule sidecar"
        )
        try:
            declared_schedule_sha = _verify_sidecar(
                schedule_path,
                schedule_sidecar_path,
                f"arm {arm} training schedule",
            )
        except Exception as exc:
            _translate_error(exc, "schedule_changed")
        if declared_schedule_sha != schedule.get("sha256"):
            raise FormalV3PreflightError(
                "schedule_changed",
                f"arm {arm} schedule sidecar no longer binds the finalized schedule",
            )
        records = arm_value.get("adapters")
        if not isinstance(records, list) or not records:
            raise FormalV3PreflightError(
                "training_incomplete", f"arm {arm} has no finalized adapters"
            )
        for index, raw in enumerate(records):
            record = _mapping(raw, f"arm {arm} adapter {index}")
            adapter_dir = _inside(root, record.get("adapter_dir"), "adapter directory")
            if not adapter_dir.is_dir():
                raise FormalV3PreflightError(
                    "artifact_changed", f"arm {arm} adapter directory is missing"
                )
            files = _mapping(record.get("files"), f"arm {arm} adapter files")
            for label, item in files.items():
                _verify_file(
                    root,
                    _mapping(item, f"arm {arm} {label}"),
                    f"arm {arm} {record.get('artifact_name')} {label}",
                )
            if record.get("adapter_sha256") != _digest(files):
                raise FormalV3PreflightError(
                    "artifact_changed", f"arm {arm} adapter tree binding changed"
                )


def _runtime_bindings(registry: Mapping[str, Any]) -> dict[str, Any]:
    arms = _mapping(registry.get("arms"), "registry arms")
    version = str(registry.get("version_id", ""))
    runtime: dict[str, Any] = {}
    for arm in ARMS:
        value = _mapping(arms[arm], f"registry arm {arm}")
        records = value.get("adapters", [])
        by_name = {
            str(item["adapter_name"]): item
            for item in records
            if isinstance(item, Mapping)
        }
        group: dict[str, Any] = {}
        for stage in STAGES:
            training_name = TRAINING_NAMES[stage]
            if arm == "A":
                model_id = f"fmv3-{version}-base-q4"
                record = None
            elif arm == "B":
                record = by_name.get("mixed_all")
                model_id = f"fmv3-{version}-b-mixed-r16"
            else:
                record = by_name.get(training_name)
                if record is None:
                    raise FormalV3PreflightError(
                        "arm_shape_changed", f"arm {arm} lacks {training_name}"
                    )
                model_id = (
                    f"fmv3-{version}-{arm.lower()}-{stage}-"
                    f"{str(record['artifact_name']).rsplit('-r', 1)[-1]}"
                )
            group[stage] = {
                "model_id": model_id,
                "adapter_artifact": (
                    None if record is None else record["artifact_name"]
                ),
                "adapter_dir": None if record is None else record["adapter_dir"],
                "adapter_sha256": (
                    None if record is None else record["adapter_sha256"]
                ),
            }
        runtime[arm] = group
    if len({runtime["B"][stage]["model_id"] for stage in STAGES}) != 1:
        raise FormalV3PreflightError(
            "arm_shape_changed", "B must reuse one mixed adapter at all five stages"
        )
    for arm in ("C", "D", "E", "F"):
        if len({runtime[arm][stage]["model_id"] for stage in STAGES}) != 5:
            raise FormalV3PreflightError(
                "arm_shape_changed", f"arm {arm} must hot-swap five distinct adapters"
            )
    return runtime


def preflight(
    benchmark_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Verify a finalized bundle without opening held-out cases or fixtures."""

    root = Path(project_root).resolve()
    benchmark_path = Path(benchmark_path).resolve()
    benchmark = _load_json(benchmark_path, "formal-v3 benchmark")
    if benchmark.get("schema_version") != BENCHMARK_SCHEMA:
        raise FormalV3PreflightError(
            "invalid_benchmark", "unsupported formal-v3 benchmark schema"
        )
    try:
        _forbid_formal_v2(benchmark, "formal-v3 benchmark")
    except Exception as exc:
        _translate_error(exc, "formal_v2_input_forbidden")
    if not (
        benchmark.get("formal_version") == "formal-v3"
        and benchmark.get("formal_v2_inputs_allowed") is False
    ):
        raise FormalV3PreflightError(
            "version_isolation_changed", "benchmark is not isolated to formal-v3"
        )
    binding = _mapping(benchmark.get("registry_binding"), "registry binding")
    registry_path = _inside(root, binding.get("path"), "formal-v3 registry")
    if registry_path.parent != benchmark_path.parent:
        raise FormalV3PreflightError(
            "bundle_mismatch", "benchmark and registry must share one version bundle"
        )
    registry_sha = _require_sha(binding.get("sha256"), "registry digest")
    if not registry_path.is_file() or sha256_file(registry_path) != registry_sha:
        raise FormalV3PreflightError(
            "registry_changed", "formal-v3 registry is missing or changed"
        )
    registry = _load_json(registry_path, "formal-v3 registry")
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise FormalV3PreflightError("registry_changed", "unsupported registry schema")
    contract = {key: registry.get(key) for key in REGISTRY_CONTRACT_KEYS}
    if (
        registry.get("contract_sha256") != _digest(contract)
        or binding.get("contract_sha256") != registry.get("contract_sha256")
        or registry.get("immutable") is not True
        or registry.get("formal_v2_inputs_allowed") is not False
    ):
        raise FormalV3PreflightError(
            "registry_changed", "formal-v3 registry contract digest mismatch"
        )
    if (
        registry.get("version_id") != benchmark.get("version_id")
        or registry.get("output_namespace")
        != f"runs/formal-v3/evaluation/{registry.get('version_id')}"
    ):
        raise FormalV3PreflightError(
            "version_isolation_changed", "formal-v3 version/output namespace changed"
        )
    bundle_path = benchmark_path.parent / "bundle_manifest.json"
    bundle = _load_json(bundle_path, "formal-v3 bundle manifest")
    if not (
        bundle.get("schema_version") == BUNDLE_SCHEMA
        and bundle.get("version_id") == registry.get("version_id")
        and bundle.get("registry_sha256") == registry_sha
        and bundle.get("benchmark_sha256") == sha256_file(benchmark_path)
        and bundle.get("formal_v2_inputs_allowed") is False
    ):
        raise FormalV3PreflightError(
            "bundle_mismatch", "formal-v3 bundle manifest changed"
        )
    _verify_registry_sources(root, registry)
    specs = load_specs(benchmark_path)
    try:
        validate_primary_specs(specs, require_verified_q4=True)
    except HeldoutGateError as exc:
        raise FormalV3PreflightError("fairness_contract_changed", str(exc)) from exc
    if len(specs) != 6 or [spec.name for spec in specs] != [
        "base_matched_calls",
        "mixed_matched_calls",
        "c_pipeline",
        "d_budget_matched_pipeline",
        "e_adaptive_pareto_pipeline",
        "f_adaptive_budget_matched_pipeline",
    ]:
        raise FormalV3PreflightError(
            "arm_set_changed", "benchmark must contain ordered A through F"
        )
    protocol = _mapping(benchmark.get("token_contract"), "token contract")
    sampling = _mapping(protocol.get("sampling"), "sampling contract")
    if tuple(protocol.get("stages", ())) != STAGES or not (
        sampling.get("temperature") == 0.0 and sampling.get("top_p") == 1.0
    ):
        raise FormalV3PreflightError(
            "fairness_contract_changed", "A--F protocol/sampling contract changed"
        )
    heldout = _mapping(benchmark.get("heldout_binding"), "heldout binding")
    if heldout.get("case_content_read") is not False or heldout.get("metadata_only") is not True:
        raise FormalV3PreflightError(
            "heldout_isolation_changed", "offline preflight may read metadata only"
        )
    runtime = _runtime_bindings(registry)
    runtime_sha = _digest(runtime)
    execution_contract = {
        "formal_version": "formal-v3",
        "version_id": registry["version_id"],
        "registry_sha256": registry_sha,
        "registry_contract_sha256": registry["contract_sha256"],
        "snapshot_sha256": registry["snapshot"]["snapshot_sha256"],
        "base_manifest_sha256": registry["base"]["manifest_sha256"],
        "heldout_ids_sha256": heldout["ids_sha256"],
        "heldout_manifest_sha256": heldout["manifest_sha256"],
        "leak_audit_sha256": heldout["leak_audit_sha256"],
        "protocol_sha256": _digest(protocol),
        "runtime_bindings_sha256": runtime_sha,
        "output_namespace": registry["output_namespace"],
        "resume_scope": registry["resume_scope"],
        "formal_v2_inputs_allowed": False,
    }
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "READY",
        "execution_authorized": True,
        "offline_only": True,
        "heldout_case_content_read": False,
        "formal_version": "formal-v3",
        "version_id": registry["version_id"],
        "config_sha256": sha256_file(benchmark_path),
        "run_manifest_sha256": registry_sha,
        "registry_contract_sha256": registry["contract_sha256"],
        "dataset_snapshot_sha256": registry["snapshot"]["snapshot_sha256"],
        "base_q4_artifact_sha256": registry["base"]["manifest_sha256"],
        "runtime_bindings": runtime,
        "runtime_bindings_sha256": runtime_sha,
        "serial_runtime_contract": {
            "base_model_id": runtime["A"]["planner"]["model_id"],
            "maximum_active_loras": 1,
            "maximum_cpu_loras": 1,
            "catalog_gate": "base_model_only_before_heldout",
            "allow_static_lora_modules": False,
            "require_localhost_admin": True,
            "server_project_root_transport": "explicit_absolute_posix",
        },
        "comparison_plan": {
            "A_index": 100,
            "equal_budget": ["B", "D", "F"],
            "capacity": ["C", "E"],
        },
        "execution_contract": execution_contract,
        "execution_contract_sha256": _digest(execution_contract),
        "output_namespace": registry["output_namespace"],
        "resume_scope": registry["resume_scope"],
    }


def inspect_preflight(
    benchmark_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    try:
        return preflight(benchmark_path, project_root)
    except FormalV3PreflightError as exc:
        return {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "BLOCKED",
            "execution_authorized": False,
            "offline_only": True,
            "heldout_case_content_read": False,
            "code": exc.code,
            "message": str(exc),
            "details": exc.details,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--project-root", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = inspect_preflight(args.benchmark, args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
