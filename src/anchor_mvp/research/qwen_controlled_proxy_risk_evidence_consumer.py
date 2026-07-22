"""Fail-closed consumer for additive Q+O controlled-proxy risk evidence.

This overlay deliberately does not modify the frozen Qwen prerequisite-v2
consumer.  It authenticates that still-blocked decision, then executes the
two frozen Producer auditors.  Passing risk evidence is informative only: it
can never authorize training or a formal release.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
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


CONFIG_VERSION = "anchor.qwen-controlled-proxy-risk-evidence-consumer-config.v1"
DECISION_VERSION = "anchor.qwen-controlled-proxy-risk-evidence-decision.v1"
CONFIG_PATH = "configs/research/qwen_controlled_proxy_risk_evidence_consumer_v1.yaml"
CONFIG_SHA256 = "30bf5cb93b1ee31facfb4139a1ff387d24ce9c335bb8f0ddaee6d84258e1e844"

PREREQUISITE_CONFIG_SHA256 = (
    "616d685c35e044d5e52f87b2a7868d6e4bd25b3c7119e7f5680634905bb07004"
)
PREREQUISITE_IMPLEMENTATION_SHA256 = (
    "01bf5fcbc16bdd781ce7ac1c209b2fbe9c52c3a7e8196ba6d3c146226ff28f90"
)
FOLLOWUP_RELEASE_COMMIT = "23194f7b3c707e3531ac92a64863c2b2f523f81d"
FOLLOWUP_RELEASE_PARENT = "901a7e17f1373c50ceff3381691956c08890696c"
FOLLOWUP_RELEASE_TREE = "7c580ea4f24a6a34d95f03c766f95affe29aa34e"
RISK_RELEASE_COMMIT = "7acf75a4d0198230e9ac4b1d894fdf4d93721c96"
RISK_RELEASE_PARENT = FOLLOWUP_RELEASE_COMMIT
RISK_RELEASE_TREE = "e7aed3b5934ca4bd17b9c46d566442c04c22806a"
CONSUMER_EVIDENCE_COMMIT = "58e9cd0c021ac0f01250746d44f199c1f616261d"
CONSUMER_EVIDENCE_PARENT = "6ef29f1e0e9e110d59f9f2a09c1a8151f04b2465"
CONSUMER_EVIDENCE_TREE = "1f275e2451c08f540d82ac5f24ee8b7beace060b"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_THIS_FILE = Path(__file__).resolve()
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_FORBIDDEN_BODY_KEYS = frozenset(
    {
        "answer",
        "body",
        "content",
        "input_ids",
        "preview",
        "prompt",
        "prompt_text",
        "raw_text",
        "target",
        "target_text",
        "token_ids",
    }
)


class ControlledProxyRiskConsumerError(RuntimeError):
    """A stable, content-free fail-closed error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise ControlledProxyRiskConsumerError(code)


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
        current = _read_bytes_snapshot(
            self.path, code, max_bytes=max(len(self.data), 1)
        )
        if (
            current.identity != self.identity
            or current.sha256 != self.sha256
            or current.data != self.data
        ):
            _fail(code)


def _read_bytes_snapshot(
    path: Path, code: str, *, max_bytes: int = _MAX_METADATA_BYTES
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
    except ControlledProxyRiskConsumerError:
        raise
    except OSError as exc:
        raise ControlledProxyRiskConsumerError(code) from exc
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


def _sequence(value: object, code: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(code)
    return value


def _strict_json(data: bytes, code: str) -> Mapping[str, Any]:
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
    except ControlledProxyRiskConsumerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlledProxyRiskConsumerError(code) from exc
    return _mapping(value, code)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            _fail("consumer_config_duplicate_key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _decode_yaml(snapshot: _BytesSnapshot) -> Mapping[str, Any]:
    try:
        value = yaml.load(snapshot.data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except ControlledProxyRiskConsumerError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ControlledProxyRiskConsumerError("consumer_config_invalid") from exc
    return _mapping(value, "consumer_config_invalid")


def _reject_body_fields(value: object, code: str) -> None:
    if isinstance(value, Mapping):
        if _FORBIDDEN_BODY_KEYS.intersection(str(key) for key in value):
            _fail(code)
        for child in value.values():
            _reject_body_fields(child, code)
    elif isinstance(value, list):
        for child in value:
            _reject_body_fields(child, code)


def _safe_path(value: object, code: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(code)
    relative = Path(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        _fail(code)
    lexical = _REPOSITORY_ROOT / relative
    _assert_physical_ancestry(lexical, code)
    candidate = lexical.resolve(strict=False)
    try:
        candidate.relative_to(_REPOSITORY_ROOT)
    except ValueError:
        _fail(code)
    _assert_physical_ancestry(candidate, code)
    return candidate


def _validate_sidecar(
    document: _BytesSnapshot,
    sidecar: _BytesSnapshot,
    expected_sidecar_sha256: str,
    filename: str,
    code: str,
) -> None:
    if sidecar.sha256 != expected_sidecar_sha256:
        _fail(f"{code}_physical_sha256_mismatch")
    if sidecar.data != f"{document.sha256}  {filename}\n".encode("ascii"):
        _fail(f"{code}_invalid")


def _artifact_map(
    release: Mapping[str, Any], code: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(release.get("artifacts"), code):
        item = _mapping(raw, code)
        role = item.get("role")
        if not isinstance(role, str) or role in result:
            _fail(code)
        if set(item) != {"role", "path", "sha256", "git_blob_sha1"}:
            _fail(code)
        result[role] = item
    return result


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "claim_scope",
        "paths",
        "prerequisite_binding",
        "producer_releases",
        "consumer_evidence_binding",
        "policy",
    }:
        _fail("consumer_config_shape_invalid")
    if config.get("schema_version") != CONFIG_VERSION or config.get("claim_scope") != (
        "additive_diagnostic_risk_evidence_only_no_training_authority"
    ):
        _fail("consumer_config_identity_invalid")

    paths = _mapping(config.get("paths"), "consumer_paths_invalid")
    expected_paths = {
        "repository_root": "../..",
        "prerequisite_config": "configs/research/qwen_train_prerequisite_consumer_v2.yaml",
        "prerequisite_implementation": "src/anchor_mvp/research/qwen_train_prerequisite_consumer_v2.py",
        "followup_contract": "configs/research/synthetic_scaffold_controlled_proxy_followup_v1.json",
        "followup_contract_sidecar": "configs/research/synthetic_scaffold_controlled_proxy_followup_v1.json.sha256",
        "followup_schema": "configs/research/synthetic_scaffold_controlled_proxy_followup_v1.schema.json",
        "followup_implementation": "src/anchor_mvp/swebench/synthetic_scaffold_controlled_proxy_followup.py",
        "followup_source": "fixtures/research/synthetic_scaffold_controlled_proxy_followup_v1/source/qwen_budget_matched_ablation_v1/comparison.json",
        "followup_source_sidecar": "fixtures/research/synthetic_scaffold_controlled_proxy_followup_v1/source/qwen_budget_matched_ablation_v1/comparison.json.sha256",
        "risk_contract": "configs/research/synthetic_scaffold_controlled_proxy_risk_evidence_companion_v1.json",
        "risk_contract_sidecar": "configs/research/synthetic_scaffold_controlled_proxy_risk_evidence_companion_v1.json.sha256",
        "risk_schema": "configs/research/synthetic_scaffold_controlled_proxy_risk_evidence_companion_v1.schema.json",
        "risk_implementation": "src/anchor_mvp/swebench/synthetic_scaffold_controlled_proxy_risk_evidence.py",
    }
    if dict(paths) != expected_paths:
        _fail("consumer_paths_drift")

    prerequisite = _mapping(
        config.get("prerequisite_binding"), "prerequisite_binding_invalid"
    )
    if dict(prerequisite) != {
        "config_sha256": PREREQUISITE_CONFIG_SHA256,
        "implementation_sha256": PREREQUISITE_IMPLEMENTATION_SHA256,
        "required_status": "blocked",
        "training_authorized": False,
        "formal_training_authorized": False,
    }:
        _fail("prerequisite_binding_drift")

    releases = _mapping(config.get("producer_releases"), "release_binding_invalid")
    expected_release_ids = {
        "followup_v1": (
            FOLLOWUP_RELEASE_COMMIT,
            FOLLOWUP_RELEASE_PARENT,
            FOLLOWUP_RELEASE_TREE,
            {
                "schema",
                "contract",
                "contract_sidecar",
                "implementation",
                "source",
                "source_sidecar",
            },
        ),
        "risk_companion_v1": (
            RISK_RELEASE_COMMIT,
            RISK_RELEASE_PARENT,
            RISK_RELEASE_TREE,
            {"schema", "contract", "contract_sidecar", "implementation"},
        ),
    }
    if set(releases) != set(expected_release_ids):
        _fail("release_binding_drift")
    for name, (commit, parent, tree, roles) in expected_release_ids.items():
        release = _mapping(releases[name], "release_binding_invalid")
        if set(release) != {"commit", "parent", "tree", "artifacts"}:
            _fail("release_binding_drift")
        if (release.get("commit"), release.get("parent"), release.get("tree")) != (
            commit,
            parent,
            tree,
        ):
            _fail("release_binding_drift")
        if set(_artifact_map(release, "release_artifacts_invalid")) != roles:
            _fail("release_artifacts_drift")

    evidence = _mapping(
        config.get("consumer_evidence_binding"), "consumer_evidence_invalid"
    )
    if (
        evidence.get("commit") != CONSUMER_EVIDENCE_COMMIT
        or evidence.get("parent") != CONSUMER_EVIDENCE_PARENT
        or evidence.get("tree") != CONSUMER_EVIDENCE_TREE
    ):
        _fail("consumer_evidence_identity_drift")
    receipts = _sequence(evidence.get("receipts"), "consumer_receipts_invalid")
    expected_receipts = {
        "q_o_branch_ablation": (
            "docs/research/results/qwen_qo_memory_ablation_audit_v1_receipt.json",
            "59750842e7bbad7fb06fcc64a1b9956dbd449e5591ba85f4abca021616da8ca3",
            "ad18c248e9bb091797f229927febddd11c0520d1497943119e0da2401b657a31",
        ),
        "spectral_risk": (
            "docs/research/results/qwen_qo_spectral_memory_audit_v1_receipt.json",
            "c2fddd98ece4127ad3f17a19ffbd5bfa6e8d7f95588964b389e7e2970cfc8dd3",
            "61b829f963b48dfe1b3f98337b6992dce0e5944595ec436e296b37b2986ac74a",
        ),
        "attention_hook": (
            "docs/research/results/qwen_attention_weight_hook_qpluso_v1_summary.json",
            "fc1ce0168cacfd1ed46a7ffcc1b482e7593253e224e780ab7dc6f7b701bb58a4",
            "1238605cf7abff130096e78071dd1f6ab2f916af93d85b29fdf3e054e1a13120",
        ),
    }
    if len(receipts) != len(expected_receipts):
        _fail("consumer_receipts_drift")
    for raw in receipts:
        item = _mapping(raw, "consumer_receipts_invalid")
        role = item.get("role")
        if role not in expected_receipts or dict(item) != {
            "role": role,
            "path": expected_receipts[str(role)][0],
            "sha256": expected_receipts[str(role)][1],
            "sidecar_sha256": expected_receipts[str(role)][2],
        }:
            _fail("consumer_receipts_drift")
    if {str(item["role"]) for item in receipts} != set(expected_receipts):
        _fail("consumer_receipts_drift")

    policy = _mapping(config.get("policy"), "consumer_policy_invalid")
    expected_policy = {
        "effective_condition": "prerequisite_v2_and_followup_v1_and_risk_companion_v1",
        "unknown_or_missing_result": "fail_closed",
        "require_single_bytes_snapshot_and_final_recheck": True,
        "require_exact_sha256_sidecars": True,
        "evidence_status": "authenticated_non_authorizing_diagnostic",
        "branch_ablation_semantics": "jointly_trained_q_plus_o_checkpoint_retained_branches",
        "retained_branch_label": "o_branch_retained",
        "independently_trained_o_only_claimed": False,
        "branch_effects_additive_claimed": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "formal_v3_ready_count": 0,
        "formal_v3_total": 5,
        "protected_inventory_ready_count": 2,
        "protected_inventory_total": 6,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
        "provider_requests": 0,
        "protected_content_reads": 0,
    }
    if dict(policy) != expected_policy:
        _fail("consumer_policy_drift")


def _git_environment() -> dict[str, str]:
    result = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    result.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return result


def _run_git(
    arguments: Sequence[str], code: str, *, binary: bool = False
) -> bytes | str:
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-c",
                "protocol.allow=never",
                "-C",
                os.fspath(_REPOSITORY_ROOT),
                *arguments,
            ],
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=15,
            text=not binary,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ControlledProxyRiskConsumerError(code) from exc
    return result.stdout


def _validate_git_controls() -> None:
    replace_refs = str(
        _run_git(
            ["for-each-ref", "--format=%(refname)", "refs/replace/"],
            "git_controls_invalid",
        )
    ).strip()
    if replace_refs:
        _fail("git_replace_refs_present")
    graft_path_raw = str(
        _run_git(["rev-parse", "--git-path", "info/grafts"], "git_controls_invalid")
    ).strip()
    graft_path = Path(graft_path_raw)
    if not graft_path.is_absolute():
        graft_path = _REPOSITORY_ROOT / graft_path
    if graft_path.is_file() and graft_path.stat().st_size:
        _fail("git_grafts_present")


def _validate_release_git_blobs(
    releases: Mapping[str, Any], snapshots_by_path: Mapping[str, _BytesSnapshot]
) -> None:
    _validate_git_controls()
    for name in ("followup_v1", "risk_companion_v1"):
        release = _mapping(releases[name], "release_binding_invalid")
        commit = str(release["commit"])
        tree = str(
            _run_git(["rev-parse", f"{commit}^{{tree}}"], "release_commit_unavailable")
        ).strip()
        parents = str(
            _run_git(
                ["show", "-s", "--format=%P", commit], "release_commit_unavailable"
            )
        ).strip()
        if tree != release["tree"] or parents != release["parent"]:
            _fail("release_commit_identity_mismatch")
        for raw in _sequence(release["artifacts"], "release_artifacts_invalid"):
            artifact = _mapping(raw, "release_artifacts_invalid")
            path = str(artifact["path"])
            listing = bytes(
                _run_git(
                    ["ls-tree", "-z", str(release["tree"]), "--", path],
                    "release_blob_unavailable",
                    binary=True,
                )
            )
            records = [item for item in listing.split(b"\0") if item]
            if len(records) != 1:
                _fail("release_blob_identity_mismatch")
            try:
                metadata, actual_path = records[0].split(b"\t", 1)
                mode, kind, oid = metadata.decode("ascii").split(" ")
                decoded_path = actual_path.decode("utf-8")
            except (UnicodeDecodeError, ValueError) as exc:
                raise ControlledProxyRiskConsumerError(
                    "release_blob_identity_mismatch"
                ) from exc
            if (
                mode != "100644"
                or kind != "blob"
                or oid != artifact["git_blob_sha1"]
                or decoded_path != path
            ):
                _fail("release_blob_identity_mismatch")
            blob = bytes(
                _run_git(
                    ["cat-file", "blob", oid],
                    "release_blob_unavailable",
                    binary=True,
                )
            )
            local = snapshots_by_path.get(path)
            if (
                local is None
                or local.sha256 != artifact["sha256"]
                or _sha256(blob) != artifact["sha256"]
                or local.data != blob
            ):
                _fail("release_blob_content_mismatch")


def _load_snapshot_module(
    snapshot: _BytesSnapshot, name: str, package: str
) -> ModuleType:
    try:
        spec = importlib.util.spec_from_loader(
            name, loader=None, origin=str(snapshot.path)
        )
        if spec is None:
            _fail("snapshot_module_spec_invalid")
        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(snapshot.path)
        module.__package__ = package
        sys.modules[name] = module
        exec(
            compile(snapshot.data.decode("utf-8"), str(snapshot.path), "exec"),
            module.__dict__,
        )
        return module
    except ControlledProxyRiskConsumerError:
        sys.modules.pop(name, None)
        raise
    except Exception as exc:
        sys.modules.pop(name, None)
        raise ControlledProxyRiskConsumerError(
            "snapshot_module_execution_failed"
        ) from exc


def _evaluate_prerequisite(
    implementation: _BytesSnapshot, config_path: Path
) -> Mapping[str, Any]:
    name = "anchor_mvp.research._controlled_proxy_risk_prerequisite_v2"
    module = _load_snapshot_module(implementation, name, "anchor_mvp.research")
    try:
        result = module.evaluate_prerequisites(config_path)
    except Exception as exc:
        raise ControlledProxyRiskConsumerError(
            "prerequisite_evaluation_failed"
        ) from exc
    finally:
        sys.modules.pop(name, None)
    return _mapping(result, "prerequisite_decision_invalid")


def _run_producer_audits(
    followup_implementation: _BytesSnapshot,
    risk_implementation: _BytesSnapshot,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    package = importlib.import_module("anchor_mvp.swebench")
    followup_name = "anchor_mvp.swebench.synthetic_scaffold_controlled_proxy_followup"
    risk_name = "anchor_mvp.swebench.synthetic_scaffold_controlled_proxy_risk_evidence"
    attribute = "synthetic_scaffold_controlled_proxy_followup"
    old_followup_module = sys.modules.get(followup_name)
    old_risk_module = sys.modules.get(risk_name)
    had_attribute = hasattr(package, attribute)
    old_attribute = getattr(package, attribute, None)
    try:
        followup = _load_snapshot_module(
            followup_implementation, followup_name, "anchor_mvp.swebench"
        )
        setattr(package, attribute, followup)
        risk = _load_snapshot_module(
            risk_implementation, risk_name, "anchor_mvp.swebench"
        )
        followup_result = _mapping(
            followup.audit_followup(_REPOSITORY_ROOT), "followup_audit_result_invalid"
        )
        risk_result = _mapping(
            risk.audit_risk_evidence(_REPOSITORY_ROOT), "risk_audit_result_invalid"
        )
        return followup_result, risk_result
    except ControlledProxyRiskConsumerError:
        raise
    except Exception as exc:
        raise ControlledProxyRiskConsumerError("producer_audit_failed") from exc
    finally:
        if old_followup_module is None:
            sys.modules.pop(followup_name, None)
        else:
            sys.modules[followup_name] = old_followup_module
        if old_risk_module is None:
            sys.modules.pop(risk_name, None)
        else:
            sys.modules[risk_name] = old_risk_module
        if had_attribute:
            setattr(package, attribute, old_attribute)
        elif hasattr(package, attribute):
            delattr(package, attribute)


def _validate_risk_interpretation(
    risk_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    claims = _mapping(risk_contract.get("claims"), "risk_contract_claims_invalid")
    constraints = _mapping(
        risk_contract.get("followup_constraints"), "risk_constraints_invalid"
    )
    projection = _mapping(
        risk_contract.get("closed_projection"), "risk_projection_invalid"
    )
    branch = _mapping(
        projection.get("q_o_branch_ablation"), "risk_branch_projection_invalid"
    )
    if (
        claims.get("independent_o_only_mechanism_proven") is not False
        or claims.get("causal_mechanism_proven") is not False
        or claims.get("memorization_proven") is not False
        or claims.get("formal") is not False
        or claims.get("training_authorized") is not False
        or claims.get("formal_training_authorized") is not False
        or constraints.get("branch_ablation_interpretation")
        != "jointly_trained_q_plus_o_checkpoint_o_branch_retention_not_independently_trained_o_only_mechanism"
        or branch.get("branch_effects_additive_claimed") is not False
    ):
        _fail("risk_interpretation_boundary_drift")
    return branch


def _requested_config_path(config_path: str | Path) -> Path:
    requested = Path(config_path)
    if ".." in requested.parts:
        _fail("consumer_config_path_invalid")
    canonical = (_REPOSITORY_ROOT / CONFIG_PATH).resolve(strict=False)
    lexical = requested if requested.is_absolute() else _REPOSITORY_ROOT / requested
    _assert_physical_ancestry(lexical, "consumer_config_path_invalid")
    if lexical.resolve(strict=False) != canonical:
        _fail("consumer_config_path_invalid")
    return canonical


def evaluate_risk_evidence(config_path: str | Path) -> dict[str, Any]:
    """Authenticate all three layers and return a still-blocked decision."""

    config_file = _requested_config_path(config_path)
    snapshots: dict[str, _BytesSnapshot] = {
        "consumer_implementation": _read_bytes_snapshot(
            _THIS_FILE, "consumer_implementation_unreadable"
        ),
        "consumer_config": _read_bytes_snapshot(
            config_file, "consumer_config_unreadable"
        ),
    }
    if snapshots["consumer_config"].sha256 != CONFIG_SHA256:
        _fail("consumer_config_sha256_mismatch")
    config = _decode_yaml(snapshots["consumer_config"])
    _validate_config(config)
    _reject_body_fields(config, "consumer_config_contains_body")
    paths = _mapping(config["paths"], "consumer_paths_invalid")

    path_roles = {
        role: str(value) for role, value in paths.items() if role != "repository_root"
    }
    for role, relative in path_roles.items():
        snapshots[role] = _read_bytes_snapshot(
            _safe_path(relative, f"{role}_path_invalid"), f"{role}_unreadable"
        )

    if snapshots["prerequisite_config"].sha256 != PREREQUISITE_CONFIG_SHA256:
        _fail("prerequisite_config_sha256_mismatch")
    if (
        snapshots["prerequisite_implementation"].sha256
        != PREREQUISITE_IMPLEMENTATION_SHA256
    ):
        _fail("prerequisite_implementation_sha256_mismatch")

    releases = _mapping(config["producer_releases"], "release_binding_invalid")
    snapshots_by_path = {
        str(paths[role]): snapshots[role]
        for role in (
            "followup_contract",
            "followup_contract_sidecar",
            "followup_schema",
            "followup_implementation",
            "followup_source",
            "followup_source_sidecar",
            "risk_contract",
            "risk_contract_sidecar",
            "risk_schema",
            "risk_implementation",
        )
    }
    _validate_release_git_blobs(releases, snapshots_by_path)

    followup_artifacts = _artifact_map(
        _mapping(releases["followup_v1"], "release_binding_invalid"),
        "release_artifacts_invalid",
    )
    risk_artifacts = _artifact_map(
        _mapping(releases["risk_companion_v1"], "release_binding_invalid"),
        "release_artifacts_invalid",
    )
    _validate_sidecar(
        snapshots["followup_contract"],
        snapshots["followup_contract_sidecar"],
        str(followup_artifacts["contract_sidecar"]["sha256"]),
        "synthetic_scaffold_controlled_proxy_followup_v1.json",
        "followup_contract_sidecar",
    )
    _validate_sidecar(
        snapshots["followup_source"],
        snapshots["followup_source_sidecar"],
        str(followup_artifacts["source_sidecar"]["sha256"]),
        "comparison.json",
        "followup_source_sidecar",
    )
    _validate_sidecar(
        snapshots["risk_contract"],
        snapshots["risk_contract_sidecar"],
        str(risk_artifacts["contract_sidecar"]["sha256"]),
        "synthetic_scaffold_controlled_proxy_risk_evidence_companion_v1.json",
        "risk_contract_sidecar",
    )

    prerequisite = _evaluate_prerequisite(
        snapshots["prerequisite_implementation"],
        snapshots["prerequisite_config"].path,
    )
    if (
        prerequisite.get("schema_version")
        != "anchor.qwen-train-prerequisite-decision.v2"
        or prerequisite.get("status") != "blocked"
        or prerequisite.get("training_authorized") is not False
        or prerequisite.get("formal_training_authorized") is not False
    ):
        _fail("prerequisite_decision_drift")

    followup_result, risk_result = _run_producer_audits(
        snapshots["followup_implementation"], snapshots["risk_implementation"]
    )
    if (
        followup_result.get("status") != "passed"
        or followup_result.get("training_authorized") is not False
        or followup_result.get("formal_training_authorized") is not False
        or followup_result.get("diagnostic_only") is not True
    ):
        _fail("followup_audit_semantics_drift")
    if (
        risk_result.get("status") != "passed"
        or risk_result.get("consumer_commit") != CONSUMER_EVIDENCE_COMMIT
        or risk_result.get("receipt_count") != 3
        or risk_result.get("primary_endpoint") != "step_80"
        or risk_result.get("training_authorized") is not False
        or risk_result.get("formal_training_authorized") is not False
        or risk_result.get("multi_seed_validated") is not False
        or risk_result.get("bundle_generalization_validated") is not False
        or risk_result.get("diagnostic_only") is not True
    ):
        _fail("risk_audit_semantics_drift")

    risk_contract = _strict_json(
        snapshots["risk_contract"].data, "risk_contract_invalid"
    )
    _reject_body_fields(risk_contract, "risk_contract_contains_body")
    branch = _validate_risk_interpretation(risk_contract)

    for role, snapshot in snapshots.items():
        snapshot.assert_unchanged(f"{role}_changed_during_evaluation")
    _validate_release_git_blobs(releases, snapshots_by_path)

    decision = {
        "schema_version": DECISION_VERSION,
        "status": "blocked",
        "effective_condition": (
            "prerequisite_v2_and_followup_v1_and_risk_companion_v1"
        ),
        "evidence_status": "authenticated_non_authorizing_diagnostic",
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "prerequisite_status": prerequisite["status"],
        "producer_audits": {
            "followup_v1": followup_result["status"],
            "risk_companion_v1": risk_result["status"],
        },
        "branch_ablation": {
            "checkpoint_semantics": (
                "jointly_trained_q_plus_o_checkpoint_retained_branches"
            ),
            "retained_branch_label": "o_branch_retained",
            "o_branch_retained_fraction_of_off_to_full_reduction": branch[
                "o_branch_retained_fraction_of_off_to_full_reduction"
            ],
            "independently_trained_o_only_claimed": False,
            "independent_o_only_mechanism_proven": False,
            "branch_effects_additive_claimed": False,
            "primary_endpoint": "step_80",
        },
        "promotion_gates": {
            "formal_v3_ready_count": 0,
            "formal_v3_total": 5,
            "protected_inventory_ready_count": 2,
            "protected_inventory_total": 6,
            "multi_seed_validated": False,
            "bundle_generalization_validated": False,
        },
        "bindings": {
            "prerequisite_config_sha256": PREREQUISITE_CONFIG_SHA256,
            "prerequisite_implementation_sha256": (PREREQUISITE_IMPLEMENTATION_SHA256),
            "producer_followup_release_commit": FOLLOWUP_RELEASE_COMMIT,
            "producer_risk_release_commit": RISK_RELEASE_COMMIT,
            "consumer_evidence_commit": CONSUMER_EVIDENCE_COMMIT,
            "followup_contract_sha256": snapshots["followup_contract"].sha256,
            "risk_contract_sha256": snapshots["risk_contract"].sha256,
            "risk_schema_sha256": snapshots["risk_schema"].sha256,
        },
        "audit": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "protected_content_reads": 0,
            "consumer_source_worktree_receipts_read": 0,
        },
        "on_disk_consumer_implementation_sha256": snapshots[
            "consumer_implementation"
        ].sha256,
    }
    _reject_body_fields(decision, "consumer_decision_contains_body")
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate non-authorizing Q+O controlled-proxy risk evidence"
    )
    parser.add_argument("--config", required=True)
    exact = "optional exact identity assertion; never overrides canonical paths"
    parser.add_argument("--prerequisite-config", help=exact)
    parser.add_argument("--prerequisite-config-sha256", help=exact)
    parser.add_argument("--followup-contract", help=exact)
    parser.add_argument("--followup-contract-sha256", help=exact)
    parser.add_argument("--followup-contract-sidecar-sha256", help=exact)
    parser.add_argument("--risk-contract", help=exact)
    parser.add_argument("--risk-contract-sha256", help=exact)
    parser.add_argument("--risk-contract-sidecar-sha256", help=exact)
    parser.add_argument("--producer-risk-release-commit", help=exact)
    parser.add_argument("--consumer-evidence-commit", help=exact)
    return parser


def _validate_cli_overrides(namespace: argparse.Namespace) -> None:
    expected = {
        "prerequisite_config": (
            "configs/research/qwen_train_prerequisite_consumer_v2.yaml"
        ),
        "prerequisite_config_sha256": PREREQUISITE_CONFIG_SHA256,
        "followup_contract": (
            "configs/research/synthetic_scaffold_controlled_proxy_followup_v1.json"
        ),
        "followup_contract_sha256": (
            "06ed121b23570546eb00088c891b273806dab1a1f764c9e40fba527cbf6447df"
        ),
        "followup_contract_sidecar_sha256": (
            "bf346eb634d8f00f36eb069707069f2983d9083c9f8c1f416dc93795cb4f7209"
        ),
        "risk_contract": (
            "configs/research/"
            "synthetic_scaffold_controlled_proxy_risk_evidence_companion_v1.json"
        ),
        "risk_contract_sha256": (
            "352870bbea976c0b97df722fd3b188d731a8d463ec33f27bcd15bdb2e292ac28"
        ),
        "risk_contract_sidecar_sha256": (
            "c75a91a78123fc1f583e9d053525c290fa39c3e6a811b22ecb48b11581c5503b"
        ),
        "producer_risk_release_commit": RISK_RELEASE_COMMIT,
        "consumer_evidence_commit": CONSUMER_EVIDENCE_COMMIT,
    }
    groups = (
        ("prerequisite_config", "prerequisite_config_sha256"),
        (
            "followup_contract",
            "followup_contract_sha256",
            "followup_contract_sidecar_sha256",
        ),
        (
            "risk_contract",
            "risk_contract_sha256",
            "risk_contract_sidecar_sha256",
        ),
    )
    for group in groups:
        supplied = [getattr(namespace, field) is not None for field in group]
        if any(supplied) and not all(supplied):
            _fail("consumer_cli_override_incomplete")
    for field, frozen in expected.items():
        supplied = getattr(namespace, field)
        if supplied is not None and supplied != frozen:
            _fail("consumer_cli_override_drift")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli_overrides(args)
    decision = evaluate_risk_evidence(args.config)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
