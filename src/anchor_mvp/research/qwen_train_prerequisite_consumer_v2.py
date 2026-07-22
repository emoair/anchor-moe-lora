"""Conjunctive consumer for frozen Qwen prerequisite v1 plus companion v2.

The module authenticates two independent, content-free facts.  The frozen v1
consumer remains byte-for-byte unchanged and blocked.  Companion v2 upgrades
only the request-local trigger fact to ``ready_diagnostic_only``.  It cannot
authorize training while the protected inventory, attestation, and formal-v3
release gates remain incomplete.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import yaml


CONFIG_VERSION = "anchor.qwen-train-prerequisite-consumer-overlay-config.v2"
DECISION_VERSION = "anchor.qwen-train-prerequisite-decision.v2"
CONFIG_PATH = "configs/research/qwen_train_prerequisite_consumer_v2.yaml"
CONFIG_SHA256 = "616d685c35e044d5e52f87b2a7868d6e4bd25b3c7119e7f5680634905bb07004"

V1_CONFIG_SHA256 = "4fdc8173baaa9f14d93a288b18f38691be62bb1fb8e646c579a06d9c78bc1a8a"
V1_IMPLEMENTATION_SHA256 = (
    "28224d9f065844c8bed04a1dc850e2d67888aa08ca080fbd6a017733864dfc60"
)
TOY_V1_CONFIG_SHA256 = (
    "ac2c522015798b379566c8c2aa96e5689398fb303240db916818c3a06e667811"
)
TOY_V1_IMPLEMENTATION_SHA256 = (
    "f49f19a74555aa7393d228b6a83649d6bd47208409f4f5cf12adafa5a0f510db"
)
COMPANION_RELEASE_COMMIT = "2648129d599a5041100278cb04b12291ffd8a482"
COMPANION_RELEASE_TREE = "f9bbe821b2a3a683376f7bb565fa31bcda86b119"
COMPANION_RELEASE_PARENT = "744e23f975b13923903f5fabe04c32e74ea25dc4"
COMPANION_CONFIG_SHA256 = (
    "21d483bfbfdab61a48996664b9221443c794902c4dcc547522b29b0c2346e50f"
)
COMPANION_SCHEMA_SHA256 = (
    "596898946d773617ec4f0f2dc86ef8c261aa5c3f7bdc382e07114abc98382119"
)
COMPANION_IMPLEMENTATION_SHA256 = (
    "dc96b2690769f14397393af0f6cfc07450dd58e9073ca5982adc2a9cc84c905e"
)
COMPANION_MANIFEST_SHA256 = (
    "7de94ff167db5cbae678e1c5f9049bc9abe5f8156ad4ab3acb4f65f52cf1d115"
)
COMPANION_SIDECAR_SHA256 = (
    "f17631d15ec34561c9fcc7c0f29aad1b2fd8d302c2cdc4553e88562701673095"
)
SOURCE_RECEIPT_SHA256 = (
    "ae15b7a0edad2527348189fa65532865bdbbe6be0f32757714f2b0fe6582089e"
)
SOURCE_SIDECAR_SHA256 = (
    "ca54594ac067286e5a8619f26870537291fb5e1c5556e8b8efbccc4a67a58a8a"
)
ORDERED_ARTIFACT_INVENTORY_SHA256 = (
    "bf38c88fd993d4804f9a624bbf98330a9d4572a6803a80c2f57a6beb05b5f567"
)

OVERLAY_SEMANTICS = (
    "non_mutating_conjunctive_overlay_preserve_v1_pending_add_authenticated_"
    "ready_trigger"
)
MISSING_SOURCE_CLASSES = (
    "gold_partition",
    "partial_gold_export",
    "legacy_heldout_cases",
    "synthetic_scaffold",
)
READY_SOURCE_CLASSES = ("swebench_source", "heldout")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_THIS_FILE = Path(__file__).resolve()
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_MAX_CONTRACT_BYTES = 2_000_000
_FORBIDDEN_BODY_KEYS = frozenset(
    {
        "answer",
        "body",
        "content",
        "input_ids",
        "preview",
        "prompt",
        "target",
        "token_ids",
        "token_indices",
    }
)


class QwenPrerequisiteOverlayError(RuntimeError):
    """A stable fail-closed overlay error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise QwenPrerequisiteOverlayError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    attributes = int(getattr(value, "st_file_attributes", 0))
    return stat.S_ISLNK(value.st_mode) or bool(
        attributes & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _assert_physical_ancestry(path: Path, code: str) -> None:
    try:
        relative = path.relative_to(_REPOSITORY_ROOT)
    except ValueError:
        _fail(code)
    current = _REPOSITORY_ROOT
    if _is_reparse_or_symlink(current):
        _fail(code)
    for part in relative.parts:
        current = current / part
        if current.exists() and _is_reparse_or_symlink(current):
            _fail(code)


@dataclass(frozen=True)
class _BytesSnapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]

    def assert_unchanged(self, code: str) -> None:
        current = _read_bytes_snapshot(self.path, code)
        if (
            current.identity != self.identity
            or current.sha256 != self.sha256
            or current.data != self.data
        ):
            _fail(code)


def _read_bytes_snapshot(
    path: Path, code: str, *, max_bytes: int = _MAX_CONTRACT_BYTES
) -> _BytesSnapshot:
    _assert_physical_ancestry(path, code)
    try:
        if not path.is_file() or path.is_symlink():
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size > max_bytes:
                _fail(code)
            data = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
        path_after = path.stat()
        _assert_physical_ancestry(path, code)
    except QwenPrerequisiteOverlayError:
        raise
    except OSError as exc:
        raise QwenPrerequisiteOverlayError(code) from exc
    if len(data) > max_bytes:
        _fail(code)
    identity = _stat_identity(after)
    if (
        _stat_identity(before) != identity
        or identity != _stat_identity(path_after)
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        _fail(code)
    return _BytesSnapshot(path, data, _sha256(data), identity)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_fields(value: Mapping[str, Any], fields: set[str], code: str) -> None:
    if set(value) != fields:
        _fail(code)


def _strict_json(data: bytes, code: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(code)
            result[key] = value
        return result

    def reject_constant(_: str) -> object:
        _fail(code)

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except QwenPrerequisiteOverlayError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenPrerequisiteOverlayError(code) from exc
    return _mapping(value, code)


def _canonical_json_document(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _decode_yaml(snapshot: _BytesSnapshot, code: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise QwenPrerequisiteOverlayError(code) from exc
    return _mapping(value, code)


def _reject_external_refs(value: object, code: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "$ref" and (
                not isinstance(item, str) or not item.startswith("#")
            ):
                _fail(code)
            _reject_external_refs(item, code)
    elif isinstance(value, list):
        for item in value:
            _reject_external_refs(item, code)


def _validate_schema(schema: Mapping[str, Any], instance: Mapping[str, Any]) -> None:
    _reject_external_refs(schema, "companion_schema_external_ref")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance)
    except QwenPrerequisiteOverlayError:
        raise
    except ImportError as exc:
        raise QwenPrerequisiteOverlayError("jsonschema_dependency_unavailable") from exc
    except Exception as exc:
        raise QwenPrerequisiteOverlayError(
            "companion_manifest_schema_validation_failed"
        ) from exc


def _reject_content_fields(value: object, code: str) -> None:
    if isinstance(value, Mapping):
        if any(str(key) in _FORBIDDEN_BODY_KEYS for key in value):
            _fail(code)
        for item in value.values():
            _reject_content_fields(item, code)
    elif isinstance(value, list):
        for item in value:
            _reject_content_fields(item, code)


def _safe_path(value: object, code: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(code)
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        _fail(code)
    lexical = _REPOSITORY_ROOT / relative
    # Inspect the lexical chain before resolving it.  Resolving first would
    # erase the evidence that an ancestor was a symlink/junction.
    _assert_physical_ancestry(lexical, code)
    candidate = lexical.resolve(strict=False)
    try:
        candidate.relative_to(_REPOSITORY_ROOT)
    except ValueError:
        _fail(code)
    _assert_physical_ancestry(candidate, code)
    return candidate


def _validate_config(config: Mapping[str, Any]) -> None:
    _exact_fields(
        config,
        {
            "schema_version",
            "claim_scope",
            "paths",
            "producer_companion_contract",
            "bindings",
            "policy",
        },
        "overlay_config_shape_invalid",
    )
    if config.get("schema_version") != CONFIG_VERSION:
        _fail("overlay_config_version_invalid")
    if config.get("claim_scope") != (
        "content_free_conjunctive_overlay_only_no_training_authority"
    ):
        _fail("overlay_config_claim_scope_invalid")

    paths = _mapping(config.get("paths"), "overlay_paths_invalid")
    if dict(paths) != {
        "repository_root": "../..",
        "frozen_v1_consumer_config": (
            "configs/research/qwen_train_prerequisite_consumer_v1.yaml"
        ),
        "frozen_v1_consumer_implementation": (
            "src/anchor_mvp/research/qwen_train_prerequisite_consumer.py"
        ),
        "frozen_toy_v1_consumer_config": (
            "configs/research/qwen_toy_prerequisite_consumer_v1.yaml"
        ),
        "frozen_toy_v1_consumer_implementation": (
            "src/anchor_mvp/research/qwen_toy_prerequisite_consumer.py"
        ),
        "companion_config": (
            "configs/research/qwen_toy_prerequisite_companion_v2.json"
        ),
        "companion_schema": (
            "configs/research/qwen_toy_prerequisite_companion_v2.schema.json"
        ),
        "companion_implementation": (
            "src/anchor_mvp/swebench/qwen_toy_prerequisite_companion_v2.py"
        ),
        "companion_artifact_root": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2"
        ),
        "canonical_source_receipt": (
            "fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json"
        ),
        "canonical_source_receipt_sidecar": (
            "fixtures/research/qwen_request_local_trigger_receipt_v2/"
            "receipt.json.sha256"
        ),
    }:
        _fail("overlay_paths_drift")

    producer = _mapping(
        config.get("producer_companion_contract"), "overlay_producer_contract_invalid"
    )
    if dict(producer) != {
        "release_commit": COMPANION_RELEASE_COMMIT,
        "release_tree": COMPANION_RELEASE_TREE,
        "release_parent": COMPANION_RELEASE_PARENT,
        "manifest_schema_version": (
            "anchor.qwen-toy-prerequisite-companion-manifest.v2"
        ),
        "overlay_semantics": OVERLAY_SEMANTICS,
    }:
        _fail("overlay_producer_contract_drift")

    bindings = _mapping(config.get("bindings"), "overlay_bindings_invalid")
    if dict(bindings) != {
        "frozen_v1_consumer_config_sha256": V1_CONFIG_SHA256,
        "frozen_v1_consumer_implementation_sha256": V1_IMPLEMENTATION_SHA256,
        "frozen_toy_v1_consumer_config_sha256": TOY_V1_CONFIG_SHA256,
        "frozen_toy_v1_consumer_implementation_sha256": (TOY_V1_IMPLEMENTATION_SHA256),
        "companion_config_sha256": COMPANION_CONFIG_SHA256,
        "companion_schema_sha256": COMPANION_SCHEMA_SHA256,
        "companion_implementation_sha256": COMPANION_IMPLEMENTATION_SHA256,
        "companion_manifest_sha256": COMPANION_MANIFEST_SHA256,
        "companion_manifest_sidecar_physical_sha256": COMPANION_SIDECAR_SHA256,
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "source_receipt_sidecar_physical_sha256": SOURCE_SIDECAR_SHA256,
        "ordered_source_artifact_inventory_sha256": (ORDERED_ARTIFACT_INVENTORY_SHA256),
    }:
        _fail("overlay_bindings_drift")

    policy = _mapping(config.get("policy"), "overlay_policy_invalid")
    if dict(policy) != {
        "effective_condition": "frozen_v1_and_authenticated_companion_v2",
        "unknown_or_missing_result": "fail_closed",
        "require_mandatory_sha256_sidecars": True,
        "trigger_materialization_status": "ready_diagnostic_only",
        "inventory_ready_count": 2,
        "inventory_total": 6,
        "training_authorized": False,
        "formal_training_authorized": False,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
        "provider_requests": 0,
        "protected_content_reads": 0,
    }:
        _fail("overlay_policy_drift")


def _validate_sidecar(
    document: _BytesSnapshot,
    sidecar: _BytesSnapshot,
    expected_sidecar_sha256: str,
    filename: str,
    code: str,
) -> None:
    if sidecar.sha256 != expected_sidecar_sha256:
        _fail(f"{code}_physical_sha256_mismatch")
    expected = f"{document.sha256}  {filename}\n".encode("ascii")
    if sidecar.data != expected:
        _fail(f"{code}_invalid")


def _load_snapshot_module(
    snapshot: _BytesSnapshot, name: str, package: str
) -> ModuleType:
    try:
        source = snapshot.data.decode("utf-8")
        spec = importlib.util.spec_from_loader(
            name, loader=None, origin=str(snapshot.path)
        )
        if spec is None:
            _fail("snapshot_module_spec_invalid")
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(snapshot.path)
        module.__package__ = package
        sys.modules[name] = module
        exec(compile(source, str(snapshot.path), "exec"), module.__dict__)
    except QwenPrerequisiteOverlayError:
        sys.modules.pop(name, None)
        raise
    except Exception as exc:
        sys.modules.pop(name, None)
        raise QwenPrerequisiteOverlayError("snapshot_module_execution_failed") from exc
    return module


def _evaluate_frozen_v1(
    implementation: _BytesSnapshot, config_path: Path
) -> Mapping[str, Any]:
    name = "anchor_mvp.research._frozen_qwen_train_prerequisite_consumer_v1"
    module = _load_snapshot_module(implementation, name, "anchor_mvp.research")
    try:
        result = module.evaluate_prerequisites(config_path)
    except Exception as exc:
        raise QwenPrerequisiteOverlayError("frozen_v1_evaluation_failed") from exc
    finally:
        sys.modules.pop(name, None)
    return _mapping(result, "frozen_v1_decision_invalid")


def _evaluate_frozen_toy_v1(
    implementation: _BytesSnapshot, config_path: Path
) -> Mapping[str, Any]:
    name = "anchor_mvp.research._frozen_qwen_toy_prerequisite_consumer_v1"
    module = _load_snapshot_module(implementation, name, "anchor_mvp.research")
    try:
        result = module.evaluate_toy_prerequisite(config_path)
    except Exception as exc:
        raise QwenPrerequisiteOverlayError("frozen_toy_v1_evaluation_failed") from exc
    finally:
        sys.modules.pop(name, None)
    return _mapping(result, "frozen_toy_v1_decision_invalid")


def _audit_companion(
    implementation: _BytesSnapshot, config_path: Path, artifact_root: Path
) -> Mapping[str, Any]:
    name = "anchor_mvp.swebench._frozen_qwen_toy_prerequisite_companion_v2"
    module = _load_snapshot_module(implementation, name, "anchor_mvp.swebench")
    try:
        result = module.audit_qwen_toy_prerequisite_companion(
            _REPOSITORY_ROOT, config_path, artifact_root
        )
    except Exception as exc:
        raise QwenPrerequisiteOverlayError("companion_audit_failed") from exc
    finally:
        sys.modules.pop(name, None)
    return _mapping(result, "companion_audit_result_invalid")


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _run_git(
    arguments: Sequence[str], code: str, *, binary: bool = False
) -> bytes | str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=_REPOSITORY_ROOT,
            env=_git_environment(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QwenPrerequisiteOverlayError(code) from exc
    return result.stdout


def _validate_release_git_blobs(snapshots: Mapping[str, _BytesSnapshot]) -> None:
    replace_refs = str(
        _run_git(
            ["for-each-ref", "--format=%(refname)", "refs/replace/"],
            "companion_git_controls_invalid",
        )
    ).strip()
    if replace_refs:
        _fail("companion_git_replace_refs_present")
    graft_path_raw = str(
        _run_git(
            ["rev-parse", "--git-path", "info/grafts"],
            "companion_git_controls_invalid",
        )
    ).strip()
    graft_path = Path(graft_path_raw)
    if not graft_path.is_absolute():
        graft_path = _REPOSITORY_ROOT / graft_path
    if graft_path.is_file() and graft_path.stat().st_size:
        _fail("companion_git_grafts_present")

    tree = str(
        _run_git(
            ["rev-parse", f"{COMPANION_RELEASE_COMMIT}^{{tree}}"],
            "companion_release_commit_unavailable",
        )
    ).strip()
    parents = str(
        _run_git(
            ["show", "-s", "--format=%P", COMPANION_RELEASE_COMMIT],
            "companion_release_commit_unavailable",
        )
    ).strip()
    if tree != COMPANION_RELEASE_TREE or parents != COMPANION_RELEASE_PARENT:
        _fail("companion_release_commit_identity_mismatch")

    release_paths = {
        "companion_config": "configs/research/qwen_toy_prerequisite_companion_v2.json",
        "companion_schema": (
            "configs/research/qwen_toy_prerequisite_companion_v2.schema.json"
        ),
        "companion_implementation": (
            "src/anchor_mvp/swebench/qwen_toy_prerequisite_companion_v2.py"
        ),
        "companion_manifest": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/manifest.json"
        ),
        "companion_manifest_sidecar": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/manifest.json.sha256"
        ),
        "copied_source_receipt": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/source/"
            "qwen_request_local_trigger_receipt_v2/receipt.json"
        ),
        "copied_source_sidecar": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/source/"
            "qwen_request_local_trigger_receipt_v2/receipt.json.sha256"
        ),
    }
    for role, path in release_paths.items():
        raw = _run_git(
            ["cat-file", "blob", f"{COMPANION_RELEASE_COMMIT}:{path}"],
            "companion_release_blob_unavailable",
            binary=True,
        )
        if not isinstance(raw, bytes) or raw != snapshots[role].data:
            _fail("companion_release_blob_mismatch")


def _validate_companion_semantics(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != (
        "anchor.qwen-toy-prerequisite-companion-manifest.v2"
    ):
        _fail("companion_manifest_version_mismatch")
    if manifest.get("status") != "trigger_ready_diagnostic_only_inventory_incomplete":
        _fail("companion_status_drift")
    if manifest.get("overlay_semantics") != OVERLAY_SEMANTICS:
        _fail("companion_overlay_semantics_drift")

    claims = _mapping(manifest.get("claims"), "companion_claims_invalid")
    if dict(claims) != {
        "diagnostic_only": True,
        "formal": False,
        "full_generation_kv_shared_claimed": False,
        "inventory_complete": False,
        "multistream_claimed": False,
        "numeric_equivalence": False,
        "physical_kv_claimed": False,
        "proxy_signal_passed": True,
        "quality_validated": False,
        "thresholds_formal": False,
        "training_authorized": False,
        "trigger_materialization_ready": True,
        "zero_copy_claimed": False,
    }:
        _fail("companion_claims_drift")

    execution = _mapping(manifest.get("execution"), "companion_execution_invalid")
    if dict(execution) != {
        "consumer_git_blob_final_recheck_reads": 8,
        "consumer_git_blob_initial_reads": 8,
        "consumer_worktree_file_reads": 0,
        "gpu_requests": 0,
        "model_loads": 0,
        "network_requests": 0,
        "protected_content_reads": 0,
        "provider_requests": 0,
        "source_materialization_runs": 0,
    }:
        _fail("companion_execution_drift")

    inventory = _mapping(
        manifest.get("inventory_status"), "companion_inventory_invalid"
    )
    if (
        inventory.get("coverage_ready_count") != 2
        or inventory.get("coverage_total") != 6
    ):
        _fail("companion_inventory_coverage_drift")
    if tuple(inventory.get("ready_source_classes", ())) != READY_SOURCE_CLASSES:
        _fail("companion_ready_sources_drift")
    if tuple(inventory.get("missing_source_classes", ())) != MISSING_SOURCE_CLASSES:
        _fail("companion_missing_sources_drift")
    if inventory.get("inventories_modified_by_companion") is not False:
        _fail("companion_inventory_mutation_claimed")

    proof = _mapping(manifest.get("proof"), "companion_proof_invalid")
    if dict(proof) != {
        "formal_training_authorized": False,
        "status": "blocked_incomplete_protected_inventories",
        "v1_attestation_emitted": False,
        "zero_intersection_claimed": False,
    }:
        _fail("companion_proof_drift")

    trigger = _mapping(
        manifest.get("trigger_materialization"), "companion_trigger_invalid"
    )
    if trigger.get("status") != "ready_diagnostic_only":
        _fail("companion_trigger_status_drift")
    if trigger.get("activation_semantics") != "next_request_input_activation_only":
        _fail("companion_activation_semantics_drift")
    if trigger.get("total_tokens") != 44 or trigger.get("trigger_span_width") != 8:
        _fail("companion_trigger_count_drift")
    if dict(
        _mapping(
            trigger.get("trigger_span_zero_based_exclusive"), "trigger_span_invalid"
        )
    ) != {"end": 33, "end_semantics": "exclusive", "index_base": "zero", "start": 25}:
        _fail("companion_trigger_span_drift")
    if dict(_mapping(trigger.get("boundary_overhang"), "trigger_overhang_invalid")) != {
        "leading_codepoints": 0,
        "leading_utf8_bytes": 0,
        "trailing_codepoints": 1,
        "trailing_utf8_bytes": 1,
    }:
        _fail("companion_trigger_overhang_drift")
    for field in (
        "raw_token_ids_emitted",
        "global_token_index_emitted",
        "planner_request1_private_kv_reused",
        "isolated_trigger_encoding_authoritative",
        "source_materialization_reexecuted_by_companion",
    ):
        if trigger.get(field) is not False:
            _fail(f"companion_{field}_must_be_false")

    consumer = _mapping(
        manifest.get("consumer_dependency"), "companion_consumer_dependency_invalid"
    )
    if consumer.get("consumer_release_commit") != (
        "7cb1f7454a76fa3c8c9f46d64da9f11244b51c54"
    ) or consumer.get("consumer_release_tree") != (
        "67ca22bd2f9d50642bf88e484408082abebe2126"
    ):
        _fail("companion_consumer_release_identity_drift")
    if consumer.get("artifact_inventory_sha256") != ORDERED_ARTIFACT_INVENTORY_SHA256:
        _fail("companion_artifact_inventory_drift")

    producer = _mapping(manifest.get("producer"), "companion_producer_invalid")
    if (
        _mapping(producer.get("config"), "companion_producer_config_invalid").get(
            "sha256"
        )
        != COMPANION_CONFIG_SHA256
    ):
        _fail("companion_producer_config_drift")
    if (
        _mapping(
            producer.get("manifest_schema"), "companion_producer_schema_invalid"
        ).get("sha256")
        != COMPANION_SCHEMA_SHA256
    ):
        _fail("companion_producer_schema_drift")
    if (
        _mapping(
            producer.get("implementation"), "companion_producer_implementation_invalid"
        ).get("sha256")
        != COMPANION_IMPLEMENTATION_SHA256
    ):
        _fail("companion_producer_implementation_drift")


def _requested_config_path(config_path: str | Path) -> Path:
    requested = Path(config_path)
    if ".." in requested.parts:
        _fail("overlay_config_path_invalid")
    canonical_lexical = _REPOSITORY_ROOT / CONFIG_PATH
    if requested.is_absolute():
        requested_lexical = requested
    else:
        if requested.as_posix() != CONFIG_PATH:
            _fail("overlay_config_path_invalid")
        requested_lexical = _REPOSITORY_ROOT / requested
    _assert_physical_ancestry(requested_lexical, "overlay_config_path_invalid")
    _assert_physical_ancestry(canonical_lexical, "overlay_config_path_invalid")
    resolved = requested_lexical.resolve(strict=False)
    canonical = canonical_lexical.resolve(strict=False)
    if resolved != canonical:
        _fail("overlay_config_path_invalid")
    return canonical


def evaluate_prerequisites(config_path: str | Path) -> dict[str, Any]:
    """Authenticate v1 AND companion v2 and return the still-blocked decision."""

    config_file = _requested_config_path(config_path)
    own_snapshot = _read_bytes_snapshot(_THIS_FILE, "overlay_implementation_unreadable")
    config_snapshot = _read_bytes_snapshot(config_file, "overlay_config_unreadable")
    if config_snapshot.sha256 != CONFIG_SHA256:
        _fail("overlay_config_sha256_mismatch")
    config = _decode_yaml(config_snapshot, "overlay_config_invalid")
    _validate_config(config)
    _reject_content_fields(config, "overlay_config_contains_content")
    paths = _mapping(config["paths"], "overlay_paths_invalid")

    v1_config_path = _safe_path(
        paths["frozen_v1_consumer_config"], "v1_config_path_invalid"
    )
    v1_impl_path = _safe_path(
        paths["frozen_v1_consumer_implementation"], "v1_impl_path_invalid"
    )
    toy_v1_config_path = _safe_path(
        paths["frozen_toy_v1_consumer_config"], "toy_v1_config_path_invalid"
    )
    toy_v1_impl_path = _safe_path(
        paths["frozen_toy_v1_consumer_implementation"],
        "toy_v1_impl_path_invalid",
    )
    companion_config_path = _safe_path(
        paths["companion_config"], "companion_config_path_invalid"
    )
    companion_schema_path = _safe_path(
        paths["companion_schema"], "companion_schema_path_invalid"
    )
    companion_impl_path = _safe_path(
        paths["companion_implementation"], "companion_impl_path_invalid"
    )
    artifact_root = _safe_path(
        paths["companion_artifact_root"], "companion_artifact_path_invalid"
    )
    if not artifact_root.is_dir() or _is_reparse_or_symlink(artifact_root):
        _fail("companion_artifact_path_invalid")
    canonical_receipt_path = _safe_path(
        paths["canonical_source_receipt"], "canonical_receipt_path_invalid"
    )
    canonical_sidecar_path = _safe_path(
        paths["canonical_source_receipt_sidecar"], "canonical_sidecar_path_invalid"
    )

    copied_receipt_path = artifact_root / (
        "source/qwen_request_local_trigger_receipt_v2/receipt.json"
    )
    copied_sidecar_path = copied_receipt_path.with_name("receipt.json.sha256")
    manifest_path = artifact_root / "manifest.json"
    manifest_sidecar_path = artifact_root / "manifest.json.sha256"

    snapshots = {
        "overlay_implementation": own_snapshot,
        "overlay_config": config_snapshot,
        "v1_config": _read_bytes_snapshot(v1_config_path, "v1_config_unreadable"),
        "v1_implementation": _read_bytes_snapshot(v1_impl_path, "v1_impl_unreadable"),
        "toy_v1_config": _read_bytes_snapshot(
            toy_v1_config_path, "toy_v1_config_unreadable"
        ),
        "toy_v1_implementation": _read_bytes_snapshot(
            toy_v1_impl_path, "toy_v1_impl_unreadable"
        ),
        "companion_config": _read_bytes_snapshot(
            companion_config_path, "companion_config_unreadable"
        ),
        "companion_schema": _read_bytes_snapshot(
            companion_schema_path, "companion_schema_unreadable"
        ),
        "companion_implementation": _read_bytes_snapshot(
            companion_impl_path, "companion_impl_unreadable"
        ),
        "companion_manifest": _read_bytes_snapshot(
            manifest_path, "companion_manifest_unreadable"
        ),
        "companion_manifest_sidecar": _read_bytes_snapshot(
            manifest_sidecar_path,
            "companion_manifest_sidecar_unreadable",
            max_bytes=1024,
        ),
        "copied_source_receipt": _read_bytes_snapshot(
            copied_receipt_path, "copied_source_receipt_unreadable"
        ),
        "copied_source_sidecar": _read_bytes_snapshot(
            copied_sidecar_path, "copied_source_sidecar_unreadable", max_bytes=1024
        ),
        "canonical_source_receipt": _read_bytes_snapshot(
            canonical_receipt_path, "canonical_source_receipt_unreadable"
        ),
        "canonical_source_sidecar": _read_bytes_snapshot(
            canonical_sidecar_path,
            "canonical_source_sidecar_unreadable",
            max_bytes=1024,
        ),
    }

    expected_hashes = {
        "v1_config": V1_CONFIG_SHA256,
        "v1_implementation": V1_IMPLEMENTATION_SHA256,
        "toy_v1_config": TOY_V1_CONFIG_SHA256,
        "toy_v1_implementation": TOY_V1_IMPLEMENTATION_SHA256,
        "companion_config": COMPANION_CONFIG_SHA256,
        "companion_schema": COMPANION_SCHEMA_SHA256,
        "companion_implementation": COMPANION_IMPLEMENTATION_SHA256,
        "companion_manifest": COMPANION_MANIFEST_SHA256,
        "companion_manifest_sidecar": COMPANION_SIDECAR_SHA256,
        "copied_source_receipt": SOURCE_RECEIPT_SHA256,
        "copied_source_sidecar": SOURCE_SIDECAR_SHA256,
        "canonical_source_receipt": SOURCE_RECEIPT_SHA256,
        "canonical_source_sidecar": SOURCE_SIDECAR_SHA256,
    }
    for role, expected in expected_hashes.items():
        if snapshots[role].sha256 != expected:
            _fail(f"{role}_sha256_mismatch")

    _validate_sidecar(
        snapshots["companion_manifest"],
        snapshots["companion_manifest_sidecar"],
        COMPANION_SIDECAR_SHA256,
        "manifest.json",
        "companion_manifest_sidecar",
    )
    for prefix in ("copied_source", "canonical_source"):
        _validate_sidecar(
            snapshots[f"{prefix}_receipt"],
            snapshots[f"{prefix}_sidecar"],
            SOURCE_SIDECAR_SHA256,
            "receipt.json",
            f"{prefix}_sidecar",
        )
    if (
        snapshots["copied_source_receipt"].data
        != snapshots["canonical_source_receipt"].data
        or snapshots["copied_source_sidecar"].data
        != snapshots["canonical_source_sidecar"].data
    ):
        _fail("companion_source_copy_mismatch")

    manifest = _strict_json(
        snapshots["companion_manifest"].data, "companion_manifest_invalid"
    )
    if _canonical_json_document(manifest) != snapshots["companion_manifest"].data:
        _fail("companion_manifest_not_canonical")
    schema = _strict_json(
        snapshots["companion_schema"].data, "companion_schema_invalid"
    )
    _validate_schema(schema, manifest)
    _reject_content_fields(manifest, "companion_manifest_contains_content")
    _validate_companion_semantics(manifest)
    _validate_release_git_blobs(snapshots)

    v1_decision = _evaluate_frozen_v1(snapshots["v1_implementation"], v1_config_path)
    if (
        v1_decision.get("schema_version")
        != "anchor.qwen-train-prerequisite-decision.v1"
        or v1_decision.get("status") != "blocked"
        or v1_decision.get("training_authorized") is not False
        or v1_decision.get("formal_training_authorized") is not False
    ):
        _fail("frozen_v1_decision_drift")

    toy_v1_decision = _evaluate_frozen_toy_v1(
        snapshots["toy_v1_implementation"], toy_v1_config_path
    )
    if (
        toy_v1_decision.get("schema_version")
        != "anchor.qwen-toy-prerequisite-consumer-decision.v1"
        or toy_v1_decision.get("status") != "blocked"
        or toy_v1_decision.get("training_authorized") is not False
        or toy_v1_decision.get("formal_training_authorized") is not False
        or toy_v1_decision.get("zero_intersection_claimed") is not False
        or toy_v1_decision.get("v1_attestation_emitted") is not False
    ):
        _fail("frozen_toy_v1_decision_drift")
    toy_coverage = _mapping(
        toy_v1_decision.get("protected_inventory_coverage"),
        "frozen_toy_v1_coverage_invalid",
    )
    if dict(toy_coverage) != {
        "ready": 2,
        "total": 6,
        "ready_source_classes": list(READY_SOURCE_CLASSES),
        "unavailable_source_classes": list(MISSING_SOURCE_CLASSES),
    }:
        _fail("frozen_toy_v1_coverage_drift")
    toy_trigger = _mapping(
        toy_v1_decision.get("trigger_receipt"), "frozen_toy_v1_trigger_invalid"
    )
    if dict(toy_trigger) != {
        "status": "pending_request_local_materialization",
        "bound_identity_count": 0,
        "token_ids_emitted": False,
        "planner_request1_private_kv_reused": False,
    }:
        _fail("frozen_toy_v1_trigger_drift")

    audited_manifest = _audit_companion(
        snapshots["companion_implementation"], companion_config_path, artifact_root
    )
    if audited_manifest != manifest:
        _fail("companion_audit_manifest_mismatch")

    for role, snapshot in snapshots.items():
        snapshot.assert_unchanged(f"{role}_changed_during_evaluation")
    _validate_release_git_blobs(snapshots)

    decision = {
        "schema_version": DECISION_VERSION,
        "status": "blocked",
        "effective_condition": "frozen_v1_and_authenticated_companion_v2",
        "training_authorized": False,
        "formal_training_authorized": False,
        "trigger_materialization": {
            "status": "ready_diagnostic_only",
            "frozen_toy_v1_status": "pending_request_local_materialization",
            "companion_v2_status": "ready_diagnostic_only",
            "total_tokens": 44,
            "trigger_span_zero_based_exclusive": {"start": 25, "end": 33},
            "trigger_span_width": 8,
            "activation_semantics": "next_request_input_activation_only",
        },
        "inventory_status": {
            "coverage_ready_count": 2,
            "coverage_total": 6,
            "ready_source_classes": list(READY_SOURCE_CLASSES),
            "missing_source_classes": list(MISSING_SOURCE_CLASSES),
            "zero_intersection_claimed": False,
            "v1_attestation_emitted": False,
        },
        "bindings": {
            "frozen_v1_consumer_config_sha256": V1_CONFIG_SHA256,
            "frozen_v1_consumer_implementation_sha256": V1_IMPLEMENTATION_SHA256,
            "frozen_toy_v1_consumer_config_sha256": TOY_V1_CONFIG_SHA256,
            "frozen_toy_v1_consumer_implementation_sha256": (
                TOY_V1_IMPLEMENTATION_SHA256
            ),
            "producer_companion_release_commit": COMPANION_RELEASE_COMMIT,
            "producer_companion_release_tree": COMPANION_RELEASE_TREE,
            "companion_config_sha256": COMPANION_CONFIG_SHA256,
            "companion_schema_sha256": COMPANION_SCHEMA_SHA256,
            "companion_implementation_sha256": COMPANION_IMPLEMENTATION_SHA256,
            "companion_manifest_sha256": COMPANION_MANIFEST_SHA256,
            "companion_manifest_sidecar_physical_sha256": (COMPANION_SIDECAR_SHA256),
            "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
            "source_receipt_sidecar_physical_sha256": SOURCE_SIDECAR_SHA256,
            "ordered_source_artifact_inventory_sha256": (
                ORDERED_ARTIFACT_INVENTORY_SHA256
            ),
        },
        "missing_artifacts": sorted(
            set(v1_decision.get("missing_artifacts", ()))
            | {
                "gold_partition_source_id_inventory",
                "partial_gold_export_source_id_inventory",
                "legacy_heldout_cases_source_id_inventory",
                "synthetic_scaffold_source_id_inventory",
                "six_source_zero_intersection_proof",
                "v1_toy_source_disjoint_attestation",
            }
        ),
        "audit": {
            "protected_dataset_files_read": 0,
            "protected_content_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
        },
        # This is deliberately named on-disk: the already-imported overlay
        # cannot use itself as an independent execution trust root.
        "on_disk_consumer_implementation_sha256": own_snapshot.sha256,
    }
    _reject_content_fields(decision, "overlay_decision_contains_content")
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate frozen Qwen prerequisite v1 AND companion v2"
    )
    parser.add_argument("--config", required=True)
    exact = "optional exact identity assertion; this does not override the frozen path"
    parser.add_argument("--frozen-v1-config", help=exact)
    parser.add_argument("--frozen-v1-config-sha256", help=exact)
    parser.add_argument("--frozen-toy-v1-config", help=exact)
    parser.add_argument("--frozen-toy-v1-config-sha256", help=exact)
    parser.add_argument("--companion-manifest", help=exact)
    parser.add_argument("--companion-manifest-sha256", help=exact)
    parser.add_argument("--companion-manifest-sidecar-sha256", help=exact)
    parser.add_argument("--producer-companion-release-commit", help=exact)
    return parser


def _validate_cli_overrides(namespace: argparse.Namespace) -> None:
    expected = {
        "frozen_v1_config": (
            "configs/research/qwen_train_prerequisite_consumer_v1.yaml"
        ),
        "frozen_v1_config_sha256": V1_CONFIG_SHA256,
        "frozen_toy_v1_config": (
            "configs/research/qwen_toy_prerequisite_consumer_v1.yaml"
        ),
        "frozen_toy_v1_config_sha256": TOY_V1_CONFIG_SHA256,
        "companion_manifest": (
            "fixtures/research/qwen_toy_prerequisite_companion_v2/manifest.json"
        ),
        "companion_manifest_sha256": COMPANION_MANIFEST_SHA256,
        "companion_manifest_sidecar_sha256": COMPANION_SIDECAR_SHA256,
        "producer_companion_release_commit": COMPANION_RELEASE_COMMIT,
    }
    groups = (
        ("frozen_v1_config", "frozen_v1_config_sha256"),
        ("frozen_toy_v1_config", "frozen_toy_v1_config_sha256"),
        (
            "companion_manifest",
            "companion_manifest_sha256",
            "companion_manifest_sidecar_sha256",
        ),
    )
    for group in groups:
        supplied = [getattr(namespace, field) is not None for field in group]
        if any(supplied) and not all(supplied):
            _fail("overlay_cli_override_incomplete")
    for field, frozen in expected.items():
        supplied = getattr(namespace, field)
        if supplied is not None and supplied != frozen:
            _fail("overlay_cli_override_drift")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli_overrides(args)
    decision = evaluate_prerequisites(args.config)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
