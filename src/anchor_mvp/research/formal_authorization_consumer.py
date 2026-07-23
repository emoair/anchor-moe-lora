"""Fail-closed formal-authorization overlay for the diagnostic research stack.

This consumer authenticates and executes exact source snapshots of the Qwen
prerequisite v2 consumer and the multi-seed planner.  It also authenticates the
legacy generic release v2 consumer/schema.  The latter is permanently scoped
to ``research_proxy_only`` and therefore cannot authorize formal training.

No model, GPU, network, provider, protected body, or training operation is
performed.  This v1 overlay is deliberately incapable of returning an
authorized decision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import yaml


CONFIG_VERSION = "anchor.formal-authorization-consumer-config.v1"
DECISION_VERSION = "anchor.formal-authorization-decision.v1"
CONFIG_PATH = "configs/research/formal_authorization_consumer_v1.yaml"
CONFIG_SHA256 = "26d1e502875a9dac102f6b353df1d9ffd16b7e12ee8686d3e3c5c5eeea83369e"
IMPLEMENTATION_PATH = "src/anchor_mvp/research/formal_authorization_consumer.py"
DECISION_SCHEMA_PATH = "configs/research/formal_authorization_decision_v1.schema.json"

_ROOT = Path(__file__).resolve().parents[3]
_THIS_FILE = Path(__file__).resolve()
_MAX_BYTES = 2 * 1024 * 1024
_REPARSE_POINT = 0x0400
_PATH_KEYS = (
    "qwen_prerequisite_config",
    "qwen_prerequisite_implementation",
    "qwen_prerequisite_manifest",
    "qwen_prerequisite_manifest_sidecar",
    "multiseed_plan_config",
    "multiseed_plan_implementation",
    "training_release_consumer_implementation",
    "generic_release_v2_schema",
    "formal_decision_schema",
)
_BINDING_KEYS = (
    "qwen_prerequisite_config_sha256",
    "qwen_prerequisite_implementation_sha256",
    "qwen_prerequisite_manifest_sha256",
    "qwen_prerequisite_manifest_sidecar_physical_sha256",
    "multiseed_plan_config_sha256",
    "multiseed_plan_implementation_sha256",
    "training_release_consumer_implementation_sha256",
    "generic_release_v2_schema_sha256",
    "formal_decision_schema_sha256",
)
_POLICY = {
    "decision_status": "blocked_formal_authorization_inputs_unavailable",
    "prerequisite_status": "blocked",
    "formal_v3_ready_count": 0,
    "formal_v3_total": 5,
    "protected_inventory_ready_count": 2,
    "protected_inventory_total": 6,
    "secondary_factorial_may_satisfy_independent_confirmation": False,
    "secondary_factorial_may_satisfy_bundle_generalization": False,
    "old_generic_release_schema_version": "anchor.generic-train-release-lock.v2",
    "old_generic_release_claim_scope": "research_proxy_only",
    "old_generic_release_eligible_for_formal": False,
    "execution_ready": False,
    "materialization_ready": False,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal": False,
}
_ZERO_AUDIT = {
    "provider_requests": 0,
    "network_requests": 0,
    "model_loads": 0,
    "gpu_requests": 0,
    "protected_content_reads": 0,
    "dataset_body_reads": 0,
    "training_runs": 0,
}
_PREREQUISITE_ZERO_AUDIT = {
    "protected_dataset_files_read": 0,
    "protected_content_reads": 0,
    "provider_requests": 0,
    "network_requests": 0,
    "model_loads": 0,
    "gpu_requests": 0,
}
_FORMAL_V3_MISSING = frozenset(
    {
        "final_projector",
        "formal_release_lock",
        "formal_snapshot",
        "generic_execution_contract",
        "source_disjoint_manifest",
    }
)


class FormalAuthorizationConsumerError(RuntimeError):
    """Stable fail-closed formal-authorization error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise FormalAuthorizationConsumerError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_link(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0)) & _REPARSE_POINT
    )


def _safe_path(relative: object, code: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail(code)
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        _fail(code)
    current = _ROOT
    if _is_link(current):
        _fail(code)
    for part in candidate.parts:
        current /= part
        if current.exists() and _is_link(current):
            _fail(code)
    try:
        current.resolve(strict=False).relative_to(_ROOT.resolve())
    except ValueError:
        _fail(code)
    return current


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]

    def assert_unchanged(self, code: str) -> None:
        current = _read_snapshot(self.path, code)
        if (
            current.data != self.data
            or current.sha256 != self.sha256
            or current.identity != self.identity
        ):
            _fail(code)


def _read_snapshot(path: Path, code: str) -> _Snapshot:
    try:
        path.resolve(strict=False).relative_to(_ROOT.resolve())
    except ValueError:
        _fail(code)
    try:
        if not path.is_file() or _is_link(path):
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size > _MAX_BYTES:
                _fail(code)
            data = handle.read(_MAX_BYTES + 1)
            after = os.fstat(handle.fileno())
        final = path.stat()
    except FormalAuthorizationConsumerError:
        raise
    except OSError as exc:
        raise FormalAuthorizationConsumerError(code) from exc
    identity = _identity(after)
    if (
        len(data) > _MAX_BYTES
        or len(data) != after.st_size
        or _identity(before) != identity
        or _identity(final) != identity
        or _is_link(path)
    ):
        _fail(code)
    return _Snapshot(path, data, _sha256(data), identity)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            _fail("config_duplicate_key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact(value: Mapping[str, Any], fields: set[str], code: str) -> None:
    if set(value) != fields:
        _fail(code)


def _load_yaml(data: bytes) -> Mapping[str, Any]:
    try:
        value = yaml.load(data.decode("utf-8"), Loader=_UniqueLoader)
    except FormalAuthorizationConsumerError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FormalAuthorizationConsumerError("config_invalid") from exc
    return _mapping(value, "config_invalid")


def _load_json(data: bytes, code: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(code)
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _: _fail(code),
        )
    except FormalAuthorizationConsumerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalAuthorizationConsumerError(code) from exc
    return _mapping(value, code)


def _validate_config(config: Mapping[str, Any]) -> None:
    _exact(
        config,
        {"schema_version", "claim_scope", "paths", "bindings", "policy", "audit"},
        "config_fields_invalid",
    )
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope") != "immutable_non_authorizing_overlay_only"
    ):
        _fail("config_identity_invalid")
    paths = _mapping(config.get("paths"), "paths_invalid")
    _exact(paths, {"repository_root", *_PATH_KEYS}, "paths_invalid")
    if paths.get("repository_root") != "../..":
        _fail("repository_root_invalid")
    bindings = _mapping(config.get("bindings"), "bindings_invalid")
    _exact(bindings, set(_BINDING_KEYS), "bindings_invalid")
    if not all(
        isinstance(bindings.get(key), str)
        and len(bindings[key]) == 64
        and all(character in "0123456789abcdef" for character in bindings[key])
        for key in _BINDING_KEYS
    ):
        _fail("bindings_invalid")
    policy = _mapping(config.get("policy"), "policy_invalid")
    if dict(policy) != _POLICY:
        _fail("policy_invalid")
    audit = _mapping(config.get("audit"), "audit_invalid")
    if dict(audit) != _ZERO_AUDIT:
        _fail("audit_invalid")


def _validate_sidecar(
    manifest: _Snapshot, sidecar: _Snapshot, expected_name: str
) -> None:
    expected = f"{manifest.sha256}  {expected_name}\n".encode("ascii")
    if sidecar.data != expected:
        _fail("qwen_prerequisite_manifest_sidecar_invalid")


def _execute_snapshot(
    snapshot: _Snapshot, module_name: str, evaluator_name: str, *args: object
) -> object:
    """Execute only the already-authenticated bytes supplied by ``snapshot``."""

    module = ModuleType(module_name)
    module.__file__ = str(snapshot.path)
    module.__package__ = "anchor_mvp.research"
    previous = sys.modules.get(module_name)
    try:
        sys.modules[module_name] = module
        code = compile(
            snapshot.data.decode("utf-8"),
            str(snapshot.path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        evaluator = getattr(module, evaluator_name, None)
        if not callable(evaluator):
            _fail("authenticated_evaluator_missing")
        return evaluator(*args)
    except FormalAuthorizationConsumerError:
        raise
    except Exception as exc:
        raise FormalAuthorizationConsumerError(
            "authenticated_evaluator_failed"
        ) from exc
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def _validate_prerequisite(value: object) -> Mapping[str, Any]:
    decision = _mapping(value, "prerequisite_decision_invalid")
    inventory = _mapping(
        decision.get("inventory_status"), "prerequisite_inventory_invalid"
    )
    audit = _mapping(decision.get("audit"), "prerequisite_audit_invalid")
    missing = decision.get("missing_artifacts")
    if (
        decision.get("schema_version") != "anchor.qwen-train-prerequisite-decision.v2"
        or decision.get("status") != "blocked"
        or decision.get("training_authorized") is not False
        or decision.get("formal_training_authorized") is not False
        or inventory.get("coverage_ready_count") != 2
        or inventory.get("coverage_total") != 6
        or inventory.get("zero_intersection_claimed") is not False
        or inventory.get("v1_attestation_emitted") is not False
        or not isinstance(missing, list)
        or not _FORMAL_V3_MISSING.issubset(set(missing))
        or dict(audit) != _PREREQUISITE_ZERO_AUDIT
    ):
        _fail("prerequisite_decision_invalid")
    return decision


def _validate_plan(value: object) -> Mapping[str, Any]:
    plan = _mapping(value, "multiseed_plan_invalid")
    boundaries = _mapping(plan.get("study_boundaries"), "study_boundaries_invalid")
    independent = _mapping(
        boundaries.get("producer_independent_confirmation"),
        "independent_confirmation_invalid",
    )
    secondary = _mapping(
        boundaries.get("secondary_controlled_factorial_probe"),
        "secondary_factorial_invalid",
    )
    gates = _mapping(plan.get("gates"), "multiseed_gates_invalid")
    audit = _mapping(plan.get("audit"), "multiseed_audit_invalid")
    replication = _mapping(plan.get("replication"), "replication_invalid")
    if (
        plan.get("schema_version") != "anchor.qwen-multiseed-independent-bundle-plan.v1"
        or plan.get("status")
        != "blocked_controlled_factorial_confirmation_inputs_unavailable"
        or independent.get("independent_confirmation_validated") is not False
        or independent.get("bundle_generalization_validated") is not False
        or independent.get("satisfied_by_secondary_controlled_factorial_probe")
        is not False
        or secondary.get("may_satisfy_independent_confirmation") is not False
        or secondary.get("may_satisfy_bundle_generalization") is not False
        or gates.get("formal_v3_ready_count") != 0
        or gates.get("formal_v3_total") != 5
        or gates.get("protected_inventory_ready_count") != 2
        or gates.get("protected_inventory_total") != 6
        or gates.get("training_authorized") is not False
        or gates.get("formal_training_authorized") is not False
        or gates.get("formal") is not False
        or replication.get("planned_independent_training_jobs") != 40
        or replication.get("planned_trainable_checkpoint_receipts") != 200
        or replication.get("planned_throughput_order_slots") != 240
        or dict(audit) != _ZERO_AUDIT
    ):
        _fail("multiseed_plan_invalid")
    return plan


def _validate_release_policy(
    implementation: _Snapshot, schema: _Snapshot
) -> dict[str, Any]:
    schema_document = _load_json(schema.data, "generic_release_schema_invalid")
    properties = _mapping(
        schema_document.get("properties"), "generic_release_schema_invalid"
    )
    version = _mapping(
        properties.get("schema_version"), "generic_release_schema_invalid"
    )
    scope = _mapping(properties.get("claim_scope"), "generic_release_schema_invalid")
    if (
        version.get("const") != "anchor.generic-train-release-lock.v2"
        or scope.get("const") != "research_proxy_only"
    ):
        _fail("generic_release_schema_invalid")
    validated = _execute_snapshot(
        implementation,
        "anchor_mvp.research._authenticated_training_release_consumer_"
        + implementation.sha256,
        "validate_release_lock_schema",
        schema.path,
        schema.sha256,
    )
    if validated != schema.sha256:
        _fail("generic_release_consumer_invalid")
    return {
        "schema_version": "anchor.generic-train-release-lock.v2",
        "claim_scope": "research_proxy_only",
        "eligible_for_formal": False,
        "reason": "legacy_release_v2_is_research_proxy_only",
    }


def _validate_decision_schema(
    schema_snapshot: _Snapshot, decision: Mapping[str, Any]
) -> None:
    schema = _load_json(schema_snapshot.data, "formal_decision_schema_invalid")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema).iter_errors(decision),
            key=lambda item: list(item.absolute_path),
        )
    except Exception as exc:
        raise FormalAuthorizationConsumerError(
            "formal_decision_schema_invalid"
        ) from exc
    if errors:
        _fail("formal_decision_invalid")


def _requested_config(path: str | Path) -> Path:
    requested = Path(path)
    if ".." in requested.parts:
        _fail("config_path_invalid")
    candidate = requested if requested.is_absolute() else _ROOT / requested
    canonical = (_ROOT / CONFIG_PATH).resolve(strict=False)
    if candidate.resolve(strict=False) != canonical:
        _fail("config_path_invalid")
    return canonical


def evaluate_formal_authorization(
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Return the authenticated, permanently blocked v1 authorization decision."""

    requested = _requested_config(config_path)
    config_snapshot = _read_snapshot(requested, "config_unreadable")
    implementation_snapshot = _read_snapshot(
        _THIS_FILE, "consumer_implementation_unreadable"
    )
    if config_snapshot.sha256 != CONFIG_SHA256 or b"\r" in config_snapshot.data:
        _fail("config_sha256_mismatch")
    if b"\r" in implementation_snapshot.data:
        _fail("consumer_implementation_line_endings_invalid")
    config = _load_yaml(config_snapshot.data)
    _validate_config(config)
    paths = _mapping(config["paths"], "paths_invalid")
    bindings = _mapping(config["bindings"], "bindings_invalid")

    role_bindings = {
        "qwen_prerequisite_config": "qwen_prerequisite_config_sha256",
        "qwen_prerequisite_implementation": ("qwen_prerequisite_implementation_sha256"),
        "qwen_prerequisite_manifest": "qwen_prerequisite_manifest_sha256",
        "qwen_prerequisite_manifest_sidecar": (
            "qwen_prerequisite_manifest_sidecar_physical_sha256"
        ),
        "multiseed_plan_config": "multiseed_plan_config_sha256",
        "multiseed_plan_implementation": "multiseed_plan_implementation_sha256",
        "training_release_consumer_implementation": (
            "training_release_consumer_implementation_sha256"
        ),
        "generic_release_v2_schema": "generic_release_v2_schema_sha256",
        "formal_decision_schema": "formal_decision_schema_sha256",
    }
    snapshots: dict[str, _Snapshot] = {}
    for role, binding in role_bindings.items():
        snapshot = _read_snapshot(
            _safe_path(paths[role], f"{role}_path_invalid"), f"{role}_unreadable"
        )
        if snapshot.sha256 != bindings.get(binding):
            _fail(f"{role}_sha256_mismatch")
        snapshots[role] = snapshot
    _validate_sidecar(
        snapshots["qwen_prerequisite_manifest"],
        snapshots["qwen_prerequisite_manifest_sidecar"],
        "manifest.json",
    )

    prerequisite = _validate_prerequisite(
        _execute_snapshot(
            snapshots["qwen_prerequisite_implementation"],
            "anchor_mvp.research._authenticated_qwen_prerequisite_v2_"
            + snapshots["qwen_prerequisite_implementation"].sha256,
            "evaluate_prerequisites",
            paths["qwen_prerequisite_config"],
        )
    )
    plan = _validate_plan(
        _execute_snapshot(
            snapshots["multiseed_plan_implementation"],
            "anchor_mvp.research._authenticated_multiseed_plan_"
            + snapshots["multiseed_plan_implementation"].sha256,
            "build_dry_run_plan",
            paths["multiseed_plan_config"],
        )
    )
    release_policy = _validate_release_policy(
        snapshots["training_release_consumer_implementation"],
        snapshots["generic_release_v2_schema"],
    )

    result: dict[str, Any] = {
        "schema_version": DECISION_VERSION,
        "status": _POLICY["decision_status"],
        "claim_scope": "immutable_non_authorizing_overlay_only",
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "self_bindings": {
            "config": {"path": CONFIG_PATH, "sha256": config_snapshot.sha256},
            "decision_schema": {
                "path": DECISION_SCHEMA_PATH,
                "sha256": snapshots["formal_decision_schema"].sha256,
            },
            "implementation": {
                "path": IMPLEMENTATION_PATH,
                "sha256": implementation_snapshot.sha256,
            },
        },
        "authenticated_dependencies": {key: bindings[key] for key in sorted(bindings)},
        "qwen_prerequisite": {
            "status": prerequisite["status"],
            "formal_v3_ready_count": 0,
            "formal_v3_total": 5,
            "protected_inventory_ready_count": 2,
            "protected_inventory_total": 6,
            "training_authorized": False,
            "formal_training_authorized": False,
        },
        "multiseed_plan": {
            "status": plan["status"],
            "plan_sha256": plan["plan_sha256"],
            "planned_independent_training_jobs": 40,
            "planned_trainable_checkpoint_receipts": 200,
            "planned_throughput_order_slots": 240,
            "secondary_factorial_may_satisfy_independent_confirmation": False,
            "secondary_factorial_may_satisfy_bundle_generalization": False,
        },
        "legacy_generic_release_v2": release_policy,
        "gates": {
            "execution_ready": False,
            "materialization_ready": False,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal": False,
            "independent_confirmation_validated": False,
            "bundle_generalization_validated": False,
        },
        "audit": dict(_ZERO_AUDIT),
    }
    result["decision_sha256"] = _canonical_sha256(result)
    _validate_decision_schema(snapshots["formal_decision_schema"], result)

    for role, snapshot in snapshots.items():
        snapshot.assert_unchanged(f"{role}_changed")
    config_snapshot.assert_unchanged("config_changed")
    implementation_snapshot.assert_unchanged("consumer_implementation_changed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        decision = evaluate_formal_authorization(args.config)
    except FormalAuthorizationConsumerError as exc:
        print(json.dumps({"status": "blocked", "error": exc.code}, sort_keys=True))
        return 2
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True, indent=2))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
