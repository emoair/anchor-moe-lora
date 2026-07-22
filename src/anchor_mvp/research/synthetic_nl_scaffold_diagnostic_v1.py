"""Build and audit a 100-record synthetic scaffold diagnostic dataset.

The producer is deliberately content-independent: every task, route, context
segment, and target is derived from the versioned closed grammar in its config.
It never opens Gold, held-out, provider, model, tokenizer, or GPU resources.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


CONFIG_VERSION = "anchor.synthetic-nl-scaffold-diagnostic-config.v1"
RECORD_VERSION = "anchor.synthetic-nl-scaffold-diagnostic-record.v1"
MANIFEST_VERSION = "anchor.synthetic-nl-scaffold-diagnostic-manifest.v1"
PRODUCER_VERSION = "anchor.synthetic-nl-scaffold-diagnostic-producer.v1"
CLAIM_SCOPE = "synthetic_diagnostic_only_no_formal_or_training_authority"
CONFIG_PATH = "configs/research/synthetic_nl_scaffold_diagnostic_v1.yaml"
CONFIG_SCHEMA_PATH = (
    "configs/research/synthetic_nl_scaffold_diagnostic_v1_config.schema.json"
)
CLOSED_GRAMMAR_PATH = (
    "configs/research/synthetic_nl_scaffold_diagnostic_v1_closed_grammar.json"
)
CLOSED_GRAMMAR_SCHEMA_PATH = (
    "configs/research/synthetic_nl_scaffold_diagnostic_v1_closed_grammar.schema.json"
)
RECORD_SCHEMA_PATH = "configs/research/synthetic_nl_scaffold_diagnostic_v1.schema.json"
MANIFEST_SCHEMA_PATH = (
    "configs/research/synthetic_nl_scaffold_diagnostic_v1_manifest.schema.json"
)
IMPLEMENTATION_PATH = "src/anchor_mvp/research/synthetic_nl_scaffold_diagnostic_v1.py"
CANONICAL_FIXTURE_PATH = "fixtures/research/synthetic_nl_scaffold_diagnostic_v1"

ROLES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
VARIANTS = ("json_only", "concise_rationale_plus_json")
LANGUAGES = ("en", "zh-CN")
ABLATION_LABELS = ("q_only", "q_plus_o", "wide_lora")
PARTITION_KEYS = (
    ("train", "json_only"),
    ("train", "concise_rationale_plus_json"),
    ("eval_proxy", "json_only"),
    ("eval_proxy", "concise_rationale_plus_json"),
)
PARTITION_PATHS = tuple(f"{split}/{variant}.jsonl" for split, variant in PARTITION_KEYS)
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_MAX_CONTRACT_BYTES = 2_000_000
_MAX_ARTIFACT_BYTES = 50_000_000


class SyntheticScaffoldDiagnosticError(RuntimeError):
    """Stable fail-closed diagnostic dataset error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise SyntheticScaffoldDiagnosticError(code)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (_canonical_json(value) + suffix).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_json_bytes(value))


def _inventory_sha256(domain: str, values: Sequence[str]) -> str:
    return _canonical_sha256({"domain": domain, "values": sorted(set(values))})


def _id(prefix: str, payload: object) -> str:
    return f"{prefix}:{_canonical_sha256(payload)}"


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory and fail if the destination exists."""

    try:
        if os.name == "nt":
            os.rename(source, destination)
            return
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL(None, use_errno=True)
            try:
                renameat2 = libc.renameat2
            except AttributeError:
                _fail("synthetic_atomic_no_replace_unavailable")
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            at_fdcwd = -100
            rename_noreplace = 1
            result = renameat2(
                at_fdcwd,
                os.fsencode(source),
                at_fdcwd,
                os.fsencode(destination),
                rename_noreplace,
            )
            if result == 0:
                return
            error = ctypes.get_errno()
            if error in (errno.EEXIST, errno.ENOTEMPTY):
                _fail("synthetic_output_already_exists")
            raise OSError(error, os.strerror(error), str(destination))
        _fail("synthetic_atomic_no_replace_unavailable")
    except SyntheticScaffoldDiagnosticError:
        raise
    except FileExistsError as exc:
        raise SyntheticScaffoldDiagnosticError(
            "synthetic_output_already_exists"
        ) from exc
    except OSError as exc:
        raise SyntheticScaffoldDiagnosticError(
            "synthetic_atomic_publish_failed"
        ) from exc


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


def _assert_no_reparse_absolute_ancestry(path: Path, code: str) -> None:
    absolute = path.absolute()
    for current in reversed((absolute, *absolute.parents)):
        if _path_lexists(current) and _is_reparse_or_symlink(current):
            _fail(code)


def _assert_exact_artifact_layout(artifact: Path) -> None:
    expected_files = {"manifest.json", "manifest.json.sha256", *PARTITION_PATHS}
    expected_directories = {"train", "eval_proxy"}
    _assert_no_reparse_absolute_ancestry(
        artifact, "synthetic_artifact_reparse_ancestry"
    )
    try:
        root_stat = artifact.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or _is_reparse_or_symlink(artifact):
            _fail("synthetic_artifact_invalid")
        directories: set[str] = set()
        files: set[str] = set()
        pending = [artifact]
        while pending:
            current = pending.pop()
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
            for entry in entries:
                entry_path = Path(entry.path)
                relative = entry_path.relative_to(artifact).as_posix()
                value = entry.stat(follow_symlinks=False)
                attributes = int(getattr(value, "st_file_attributes", 0))
                if stat.S_ISLNK(value.st_mode) or bool(
                    attributes & _FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    _fail("synthetic_artifact_reparse_entry")
                if stat.S_ISDIR(value.st_mode):
                    directories.add(relative)
                    pending.append(entry_path)
                elif stat.S_ISREG(value.st_mode):
                    files.add(relative)
                else:
                    _fail("synthetic_artifact_special_entry")
    except SyntheticScaffoldDiagnosticError:
        raise
    except OSError as exc:
        raise SyntheticScaffoldDiagnosticError(
            "synthetic_artifact_layout_unreadable"
        ) from exc
    if directories != expected_directories or files != expected_files:
        _fail("synthetic_artifact_layout_invalid")


def _capture_artifact_snapshots(artifact: Path) -> dict[str, _Snapshot]:
    _assert_exact_artifact_layout(artifact)
    snapshots = {
        "manifest": _read_snapshot(
            artifact / "manifest.json",
            "synthetic_manifest_unreadable",
            max_bytes=_MAX_ARTIFACT_BYTES,
        ),
        "sidecar": _read_snapshot(
            artifact / "manifest.json.sha256",
            "synthetic_manifest_sidecar_unreadable",
            max_bytes=1024,
        ),
    }
    for relative in PARTITION_PATHS:
        snapshots[relative] = _read_snapshot(
            artifact / relative,
            "synthetic_partition_unreadable",
            max_bytes=_MAX_ARTIFACT_BYTES,
        )
    _assert_exact_artifact_layout(artifact)
    return snapshots


def _assert_artifact_snapshot_unchanged(
    artifact: Path,
    root_identity: tuple[int, int, int, int],
    snapshots: Mapping[str, _Snapshot],
) -> None:
    _assert_exact_artifact_layout(artifact)
    try:
        if _stat_identity(artifact.stat()) != root_identity:
            _fail("synthetic_artifact_root_changed_before_publish")
    except SyntheticScaffoldDiagnosticError:
        raise
    except OSError as exc:
        raise SyntheticScaffoldDiagnosticError(
            "synthetic_artifact_root_changed_before_publish"
        ) from exc
    for role, snapshot in snapshots.items():
        snapshot.assert_unchanged(f"synthetic_{role}_changed_before_publish")
    _assert_exact_artifact_layout(artifact)


def _assert_no_reparse_ancestry(path: Path, stop: Path, code: str) -> None:
    try:
        relative = path.relative_to(stop)
    except ValueError:
        _fail(code)
    current = stop
    if _is_reparse_or_symlink(current):
        _fail(code)
    for part in relative.parts:
        current = current / part
        if current.exists() and _is_reparse_or_symlink(current):
            _fail(code)


def _safe_repo_path(repo_root: Path, relative: object, code: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(code)
    requested = Path(relative.replace("\\", "/"))
    if requested.is_absolute() or ".." in requested.parts:
        _fail(code)
    lexical = repo_root / requested
    _assert_no_reparse_ancestry(lexical, repo_root, code)
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        _fail(code)
    _assert_no_reparse_ancestry(resolved, repo_root, code)
    return resolved


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]

    def assert_unchanged(self, code: str) -> None:
        current = _read_snapshot(self.path, code, max_bytes=max(len(self.data), 1))
        if (
            current.identity != self.identity
            or current.sha256 != self.sha256
            or current.data != self.data
        ):
            _fail(code)


def _read_snapshot(
    path: Path, code: str, *, max_bytes: int = _MAX_CONTRACT_BYTES
) -> _Snapshot:
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
    except SyntheticScaffoldDiagnosticError:
        raise
    except OSError as exc:
        raise SyntheticScaffoldDiagnosticError(code) from exc
    if len(data) > max_bytes:
        _fail(code)
    identity = _stat_identity(after)
    if (
        _stat_identity(before) != identity
        or _stat_identity(path_after) != identity
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        _fail(code)
    return _Snapshot(path, data, _sha256(data), identity)


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
    except SyntheticScaffoldDiagnosticError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticScaffoldDiagnosticError(code) from exc
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _strict_jsonl(data: bytes, code: str) -> list[Mapping[str, Any]]:
    if not data or not data.endswith(b"\n") or b"\r" in data:
        _fail(code)
    records: list[Mapping[str, Any]] = []
    for raw in data.splitlines():
        record = _strict_json(raw, code)
        if _canonical_json_bytes(record) != raw:
            _fail(code)
        records.append(record)
    return records


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


def _validate_schema(schema: Mapping[str, Any], instance: object, code: str) -> None:
    _reject_external_refs(schema, f"{code}_external_ref")
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance)
    except SyntheticScaffoldDiagnosticError:
        raise
    except Exception as exc:
        raise SyntheticScaffoldDiagnosticError(code) from exc


def _load_contract_snapshots(
    repo_root: Path, config_path: Path
) -> tuple[
    dict[str, _Snapshot],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    config_snapshot = _read_snapshot(config_path, "synthetic_config_unreadable")
    try:
        config = yaml.load(
            config_snapshot.data.decode("utf-8"), Loader=_UniqueKeySafeLoader
        )
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SyntheticScaffoldDiagnosticError("synthetic_config_invalid") from exc
    if not isinstance(config, Mapping):
        _fail("synthetic_config_invalid")
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        _fail("synthetic_config_paths_invalid")
    snapshots = {
        "config": config_snapshot,
        "config_schema": _read_snapshot(
            _safe_repo_path(
                repo_root, paths.get("config_schema"), "config_schema_path_invalid"
            ),
            "config_schema_unreadable",
        ),
        "closed_grammar": _read_snapshot(
            _safe_repo_path(
                repo_root, paths.get("closed_grammar"), "closed_grammar_path_invalid"
            ),
            "closed_grammar_unreadable",
        ),
        "closed_grammar_schema": _read_snapshot(
            _safe_repo_path(
                repo_root,
                paths.get("closed_grammar_schema"),
                "closed_grammar_schema_path_invalid",
            ),
            "closed_grammar_schema_unreadable",
        ),
        "record_schema": _read_snapshot(
            _safe_repo_path(
                repo_root, paths.get("record_schema"), "record_schema_path_invalid"
            ),
            "record_schema_unreadable",
        ),
        "manifest_schema": _read_snapshot(
            _safe_repo_path(
                repo_root, paths.get("manifest_schema"), "manifest_schema_path_invalid"
            ),
            "manifest_schema_unreadable",
        ),
        "implementation": _read_snapshot(
            _safe_repo_path(
                repo_root, paths.get("implementation"), "implementation_path_invalid"
            ),
            "implementation_unreadable",
        ),
    }
    config_schema = _strict_json(
        snapshots["config_schema"].data, "config_schema_invalid"
    )
    grammar = _strict_json(snapshots["closed_grammar"].data, "closed_grammar_invalid")
    grammar_schema = _strict_json(
        snapshots["closed_grammar_schema"].data, "closed_grammar_schema_invalid"
    )
    record_schema = _strict_json(
        snapshots["record_schema"].data, "record_schema_invalid"
    )
    manifest_schema = _strict_json(
        snapshots["manifest_schema"].data, "manifest_schema_invalid"
    )
    _validate_schema(config_schema, config, "synthetic_config_schema_validation_failed")
    _validate_schema(
        grammar_schema, grammar, "synthetic_closed_grammar_schema_validation_failed"
    )
    _validate_config(config)
    _validate_grammar(grammar)
    return snapshots, config, grammar, record_schema, manifest_schema


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_VERSION:
        _fail("synthetic_config_version_invalid")
    if config.get("claim_scope") != CLAIM_SCOPE:
        _fail("synthetic_config_claim_scope_invalid")
    paths = config["paths"]
    if dict(paths) != {
        "config_schema": CONFIG_SCHEMA_PATH,
        "closed_grammar": CLOSED_GRAMMAR_PATH,
        "closed_grammar_schema": CLOSED_GRAMMAR_SCHEMA_PATH,
        "record_schema": RECORD_SCHEMA_PATH,
        "manifest_schema": MANIFEST_SCHEMA_PATH,
        "implementation": IMPLEMENTATION_PATH,
    }:
        _fail("synthetic_config_paths_drift")
    dataset = config["dataset_contract"]
    if dict(dataset) != {
        "producer_version": PRODUCER_VERSION,
        "record_schema_version": RECORD_VERSION,
        "manifest_schema_version": MANIFEST_VERSION,
        "bundle_count": 10,
        "roles_per_bundle": 5,
        "variants_per_role": 2,
        "record_count": 100,
        "pair_count": 50,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_body_reads": 0,
    }:
        _fail("synthetic_dataset_contract_drift")
    if dict(config["generation_contract"]) != {
        "seed_id": "anchor.synthetic-nl-scaffold-diagnostic.seed.v1",
        "seed_text": "anchor-synthetic-nl-scaffold-diagnostic-generation-seed-v1",
        "source_namespace": "anchor.synthetic-nl-scaffold-diagnostic.v1",
        "augmentation": "none",
        "split_before_augmentation": True,
        "declared_semantic_read_set_only": True,
    }:
        _fail("synthetic_generation_contract_drift")
    if dict(config["protected_inventory_contract"]) != {
        "consumes_protected_inventories": False,
        "statuses": {
            "swebench_source": "unavailable_not_read",
            "gold_partition": "unavailable_not_read",
            "partial_gold_export": "unavailable_not_read",
            "heldout": "unavailable_not_read",
            "legacy_heldout_cases": "unavailable_not_read",
            "synthetic_scaffold": "unavailable_not_read",
        },
        "zero_intersection_claimed": False,
        "source_disjoint_attestation_emitted": False,
        "formal_source_disjoint_proven": False,
    }:
        _fail("synthetic_protected_inventory_contract_drift")
    if dict(config["split_contract"]) != {
        "group_key": "source_bundle_id",
        "split_before_role_and_variant_expansion": True,
        "algorithm": (
            "sha256_utf8_salt_nul_language_nul_bundle_key_sort_ascending_"
            "eval_one_per_language_v1"
        ),
        "salt": "anchor-synthetic-nl-scaffold-diagnostic-split-v1",
        "train_bundle_count": 8,
        "eval_proxy_bundle_count": 2,
        "eval_proxy_is_heldout": False,
        "all_role_variant_views_same_split": True,
        "language_stratified": {
            "en": {"train_bundles": 4, "eval_proxy_bundles": 1},
            "zh-CN": {"train_bundles": 4, "eval_proxy_bundles": 1},
        },
    }:
        _fail("synthetic_split_contract_drift")
    if dict(config["pair_contract"]) != {
        "variants": list(VARIANTS),
        "same_input_sha256": True,
        "same_canonical_json_sha256": True,
        "same_allowed_and_forbidden_segments": True,
        "concise_rationale_max_utf8_bytes": 512,
        "concise_rationale_is_hidden_chain_of_thought": False,
        "concise_rationale_is_auditable_decision_summary": True,
    }:
        _fail("synthetic_pair_contract_drift")
    if dict(config["role_contract"]) != {
        "ordered_roles": list(ROLES),
        "role_stage_index": {role: index for index, role in enumerate(ROLES)},
        "current_target_content_in_prompt": False,
        "future_target_content_in_prompt": False,
        "whole_task_board_stringification_allowed": False,
        "previous_committed_segments_allowed": True,
    }:
        _fail("synthetic_role_contract_drift")
    if dict(config["ablation_contract"]) != {
        "labels": list(ABLATION_LABELS),
        "all_records_eligible_for_all_labels": True,
        "arm_assignment_location": "diagnostic_run_manifest_only",
        "identical_record_inventory_required": True,
        "producer_selects_winner": False,
        "producer_claims_training_outcome": False,
    }:
        _fail("synthetic_ablation_contract_drift")
    if dict(config["route_contract"]) != {
        "required_fields": [
            "role",
            "expert",
            "goal",
            "constraints",
            "allowed_segment_refs",
            "evidence_segment_refs",
            "tool_plan",
            "acceptance_criteria",
        ],
        "canonical_json_policy": "utf8_sort_keys_compact_no_normalization_v1",
        "tool_trace_kind": "harmless_synthetic_fixture_trace",
        "real_tool_execution": False,
        "explicit_two_request_commit": True,
        "activation_semantics": "next_request_input_activation_only",
        "planner_request1_private_kv_reused": False,
    }:
        _fail("synthetic_route_contract_drift")
    if dict(config["audit_contract"]) != {
        "mandatory_manifest_sha256_sidecar": True,
        "sidecar_format": "sha256sum_manifest_json_lf",
        "single_bytes_snapshot_for_hash_parse_count": True,
        "final_toctou_recheck": True,
        "atomic_directory_publish": True,
        "maximum_artifact_file_bytes": 50_000_000,
        "utf8_required": True,
        "lf_required": True,
        "duplicate_json_keys_rejected": True,
        "protected_source_paths_allowed": False,
        "raw_token_ids_allowed": False,
        "global_token_indices_allowed": False,
        "formal_claim_allowed": False,
        "training_authorized_claim_allowed": False,
    }:
        _fail("synthetic_audit_contract_drift")
    bundles = config["bundles"]
    if not isinstance(bundles, list) or len(bundles) != 10:
        _fail("synthetic_bundle_count_invalid")
    keys = [bundle["bundle_key"] for bundle in bundles]
    if len(set(keys)) != 10:
        _fail("synthetic_bundle_key_duplicate")
    languages = Counter(bundle["language"] for bundle in bundles)
    if languages != Counter({"en": 5, "zh-CN": 5}):
        _fail("synthetic_bundle_language_imbalance")


def _validate_grammar(grammar: Mapping[str, Any]) -> None:
    if grammar.get("schema_version") != (
        "anchor.synthetic-nl-scaffold-closed-grammar.v1"
    ):
        _fail("synthetic_closed_grammar_version_invalid")
    if grammar.get("source_namespace") != (
        "anchor.synthetic-nl-scaffold-diagnostic.v1"
    ):
        _fail("synthetic_closed_grammar_namespace_invalid")
    if tuple(grammar.get("languages", ())) != LANGUAGES:
        _fail("synthetic_closed_grammar_languages_invalid")
    if tuple(grammar.get("roles", ())) != ROLES:
        _fail("synthetic_closed_grammar_roles_invalid")
    if tuple(grammar.get("variants", ())) != VARIANTS:
        _fail("synthetic_closed_grammar_variants_invalid")
    if tuple(grammar.get("ablation_labels", ())) != ABLATION_LABELS:
        _fail("synthetic_closed_grammar_ablation_invalid")


def _bundle_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "domain": "anchor.synthetic-nl-scaffold-bundle.v1",
        "bundle_key": bundle["bundle_key"],
        "language": bundle["language"],
        "archetype": bundle["archetype"],
        "task_text": bundle["task_text"],
        "constraints": list(bundle["constraints"]),
        "closed_grammar_id": "anchor.synthetic-nl-scaffold-closed-grammar.v1",
        "generation_seed_id": "anchor.synthetic-nl-scaffold-diagnostic.seed.v1",
    }


def _bundle_identity(bundle: Mapping[str, Any]) -> tuple[str, str]:
    digest = _canonical_sha256(_bundle_payload(bundle))
    return f"syn-nl-bundle-v1:{digest}", digest


def _split_bundles(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], list[str], list[str]]:
    salt = config["split_contract"]["salt"]
    assignments: dict[str, str] = {}
    for language in LANGUAGES:
        candidates: list[tuple[str, str]] = []
        for bundle in config["bundles"]:
            if bundle["language"] != language:
                continue
            bundle_id, _ = _bundle_identity(bundle)
            score = _sha256(
                f"{salt}\0{language}\0{bundle['bundle_key']}".encode("utf-8")
            )
            candidates.append((score, bundle_id))
        candidates.sort()
        for index, (_, bundle_id) in enumerate(candidates):
            assignments[bundle_id] = "eval_proxy" if index == 0 else "train"
    train_ids = sorted(key for key, value in assignments.items() if value == "train")
    eval_ids = sorted(
        key for key, value in assignments.items() if value == "eval_proxy"
    )
    if len(train_ids) != 8 or len(eval_ids) != 2:
        _fail("synthetic_bundle_split_invalid")
    return assignments, train_ids, eval_ids


def _segment_id(
    bundle_id: str, role: str, stage_index: int, content_sha256: str
) -> str:
    return _id(
        "syn-nl-segment-v1",
        {
            "domain": "anchor.synthetic-nl-scaffold-segment-id.v1",
            "source_bundle_id": bundle_id,
            "role": role,
            "stage_index": stage_index,
            "content_sha256": content_sha256,
        },
    )


def _role_text(
    grammar: Mapping[str, Any], language: str, role: str
) -> tuple[str, str, list[str]]:
    language_templates = grammar["templates"][language]
    template = language_templates[role]
    return (
        str(template["instruction"]),
        str(template["summary"]),
        list(language_templates["acceptance_criteria"]),
    )


def _role_goal(
    grammar: Mapping[str, Any], language: str, role: str, archetype: str
) -> str:
    prefix = str(grammar["templates"][language][role]["goal_prefix"])
    if language == "en":
        return f"{prefix} the synthetic {archetype.replace('_', ' ')} task."
    return f"{prefix}合成任务 {archetype}。"


def _make_segments(
    bundle: Mapping[str, Any], bundle_id: str, grammar: Mapping[str, Any]
) -> list[dict[str, Any]]:
    segment_refs = [f"S{index}" for index in range(len(ROLES))]
    segments: list[dict[str, Any]] = []
    for index, role in enumerate(ROLES):
        _, commit_summary, generic_criteria = _role_text(
            grammar, bundle["language"], role
        )
        allowed = segment_refs[:index]
        route = {
            "role": role,
            "expert": f"{role}_expert",
            "goal": _role_goal(grammar, bundle["language"], role, bundle["archetype"]),
            "constraints": list(bundle["constraints"]),
            "allowed_segment_refs": allowed,
            "evidence_segment_refs": allowed,
            "tool_plan": [
                {
                    "action": (f"{grammar['tool_action_prefix']}{role}_contract"),
                    "mode": grammar["tool_trace_mode"],
                }
            ],
            "acceptance_criteria": generic_criteria,
        }
        content = _canonical_json(route)
        content_sha256 = _sha256(content.encode("utf-8"))
        segments.append(
            {
                "segment_id": _segment_id(bundle_id, role, index, content_sha256),
                "segment_ref": segment_refs[index],
                "role": role,
                "stage_index": index,
                "content": content,
                "content_sha256": content_sha256,
                "commit_summary": commit_summary,
                "commit_summary_sha256": _sha256(commit_summary.encode("utf-8")),
                "route": route,
            }
        )
    return segments


def build_training_view(
    bundle: Mapping[str, Any],
    role: str,
    segments: Sequence[Mapping[str, Any]],
    grammar: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the real prompt view used by generated records."""

    if role not in ROLES:
        _fail("synthetic_training_view_role_invalid")
    stage_index = ROLES.index(role)
    if len(segments) != len(ROLES):
        _fail("synthetic_training_view_segments_invalid")
    role_instruction, _, _ = _role_text(grammar, bundle["language"], role)
    visible = [dict(segment) for segment in segments[:stage_index]]
    allowed_context = [
        {
            "segment_id": segment["segment_id"],
            "segment_ref": segment["segment_ref"],
            "role": segment["role"],
            "stage_index": segment["stage_index"],
            "committed_summary": segment["commit_summary"],
            "committed_summary_sha256": segment["commit_summary_sha256"],
            "source_target_sha256": segment["content_sha256"],
        }
        for segment in visible
    ]
    lines = [
        "[SYNTHETIC_DIAGNOSTIC_TASK]",
        str(bundle["task_text"]),
        "[CONSTRAINTS]",
        *[f"- {item}" for item in bundle["constraints"]],
        "[ROLE]",
        role,
        "[ROLE_INSTRUCTION]",
        role_instruction,
        "[COMMITTED_CONTEXT]",
    ]
    if not allowed_context:
        lines.append("(none)")
    else:
        for item in allowed_context:
            lines.extend(
                [
                    f"[{item['segment_ref']}|{item['role']}]",
                    str(item["committed_summary"]),
                ]
            )
    lines.extend(
        [
            "[OUTPUT_CONTRACT]",
            "Return the role route using the requested scaffold variant.",
        ]
    )
    materialized_prompt = "\n".join(lines) + "\n"
    payload = {
        "task_text": bundle["task_text"],
        "constraints": list(bundle["constraints"]),
        "role_instruction": role_instruction,
        "allowed_context_segments": allowed_context,
        "materialized_prompt": materialized_prompt,
    }
    payload["input_sha256"] = _canonical_sha256(payload)
    return payload


def _record_pair(
    bundle: Mapping[str, Any],
    bundle_id: str,
    bundle_sha256: str,
    split: str,
    role: str,
    segments: Sequence[Mapping[str, Any]],
    config_sha256: str,
    implementation_sha256: str,
    grammar: Mapping[str, Any],
    grammar_sha256: str,
    generation_seed_sha256: str,
) -> list[dict[str, Any]]:
    stage_index = ROLES.index(role)
    input_view = build_training_view(bundle, role, segments, grammar)
    forbidden = [segment["segment_id"] for segment in segments[stage_index:]]
    board_inventory = [
        {
            "segment_id": segment["segment_id"],
            "segment_ref": segment["segment_ref"],
            "role": segment["role"],
            "stage_index": segment["stage_index"],
            "content_sha256": segment["content_sha256"],
            "visibility": (
                "previous_committed"
                if segment["stage_index"] < stage_index
                else "current_target"
                if segment["stage_index"] == stage_index
                else "future_target"
            ),
        }
        for segment in segments
    ]
    route = dict(segments[stage_index]["route"])
    canonical_route = _canonical_json(route)
    canonical_route_sha = _sha256(canonical_route.encode("utf-8"))
    pair_id = _id(
        "syn-nl-pair-v1",
        {
            "domain": "anchor.synthetic-nl-scaffold-pair-id.v1",
            "source_bundle_id": bundle_id,
            "role": role,
            "input_sha256": input_view["input_sha256"],
            "canonical_json_sha256": canonical_route_sha,
            "generator_config_sha256": config_sha256,
            "generator_implementation_sha256": implementation_sha256,
            "closed_grammar_sha256": grammar_sha256,
            "generation_seed_sha256": generation_seed_sha256,
        },
    )
    _, rationale, _ = _role_text(grammar, bundle["language"], role)
    records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        summary = rationale if variant == "concise_rationale_plus_json" else None
        output = canonical_route if summary is None else f"{summary}\n{canonical_route}"
        output_sha256 = _sha256(output.encode("utf-8"))
        record_id = _id(
            "syn-nl-record-v1",
            {
                "domain": "anchor.synthetic-nl-scaffold-record-id.v1",
                "pair_id": pair_id,
                "variant": variant,
                "output_sha256": output_sha256,
            },
        )
        records.append(
            {
                "schema_version": RECORD_VERSION,
                "record_id": record_id,
                "pair_id": pair_id,
                "source_bundle_id": bundle_id,
                "source_bundle_sha256": bundle_sha256,
                "split": split,
                "language": bundle["language"],
                "role": role,
                "stage_index": stage_index,
                "variant": variant,
                "synthetic_source": {
                    "kind": "closed_grammar_synthetic_no_external_body",
                    "bundle_key": bundle["bundle_key"],
                    "archetype": bundle["archetype"],
                    "closed_grammar_id": (
                        "anchor.synthetic-nl-scaffold-closed-grammar.v1"
                    ),
                    "closed_grammar_sha256": grammar_sha256,
                    "generation_seed_id": (
                        "anchor.synthetic-nl-scaffold-diagnostic.seed.v1"
                    ),
                    "generation_seed_sha256": generation_seed_sha256,
                    "generator_config_sha256": config_sha256,
                    "generator_implementation_sha256": implementation_sha256,
                },
                "input": input_view,
                "board_segment_inventory": board_inventory,
                "forbidden_segment_ids": forbidden,
                "target": {
                    "canonical_routing_json": route,
                    "canonical_json_sha256": canonical_route_sha,
                    "concise_rationale_summary": summary,
                    "serialized_assistant_output": output,
                    "output_sha256": output_sha256,
                },
                "route_boundary": {
                    "semantics": "explicit_two_request_commit",
                    "activation_semantics": "next_request_input_activation_only",
                    "committed_scaffold_reencode_required": True,
                    "planner_request1_private_kv_reused": False,
                },
                "ablation": {
                    "eligible_labels": list(ABLATION_LABELS),
                    "assignment_location": "diagnostic_run_manifest_only",
                    "identical_inventory_required": True,
                    "producer_selects_winner": False,
                },
                "claims": {
                    "diagnostic_only": True,
                    "formal": False,
                    "training_authorized": False,
                    "quality_validated": False,
                    "proxy_signal_only": True,
                    "physical_kv_reuse_claimed": False,
                    "numeric_equivalence_claimed": False,
                },
                "audit": {
                    "protected_body_reads": 0,
                    "provider_requests": 0,
                    "network_requests": 0,
                    "model_loads": 0,
                    "gpu_requests": 0,
                    "real_tool_executions": 0,
                },
            }
        )
    return records


def _generate_records(
    config: Mapping[str, Any],
    grammar: Mapping[str, Any],
    snapshots: Mapping[str, _Snapshot],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    assignments, train_ids, eval_ids = _split_bundles(config)
    records: list[dict[str, Any]] = []
    for bundle in config["bundles"]:
        bundle_id, bundle_sha = _bundle_identity(bundle)
        segments = _make_segments(bundle, bundle_id, grammar)
        for role in ROLES:
            records.extend(
                _record_pair(
                    bundle,
                    bundle_id,
                    bundle_sha,
                    assignments[bundle_id],
                    role,
                    segments,
                    snapshots["config"].sha256,
                    snapshots["implementation"].sha256,
                    grammar,
                    snapshots["closed_grammar"].sha256,
                    _sha256(config["generation_contract"]["seed_text"].encode("utf-8")),
                )
            )
    if len(records) != 100:
        _fail("synthetic_record_count_invalid")
    return records, train_ids, eval_ids


def _artifact_descriptor(repo_root: Path, snapshot: _Snapshot) -> dict[str, Any]:
    return {
        "path": snapshot.path.relative_to(repo_root).as_posix(),
        "bytes": len(snapshot.data),
        "sha256": snapshot.sha256,
    }


def _partition_bytes(records: Sequence[Mapping[str, Any]]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for split, variant in PARTITION_KEYS:
        selected = [
            record
            for record in records
            if record["split"] == split and record["variant"] == variant
        ]
        selected.sort(
            key=lambda item: (
                item["source_bundle_id"],
                ROLES.index(item["role"]),
                item["record_id"],
            )
        )
        result[f"{split}/{variant}.jsonl"] = b"".join(
            _canonical_json_bytes(record, newline=True) for record in selected
        )
    return result


def _build_manifest(
    repo_root: Path,
    config: Mapping[str, Any],
    snapshots: Mapping[str, _Snapshot],
    records: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, bytes],
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
) -> dict[str, Any]:
    record_ids = [str(record["record_id"]) for record in records]
    pair_ids = [str(record["pair_id"]) for record in records]
    bundle_ids = [str(record["source_bundle_id"]) for record in records]
    record_id_inventory = _inventory_sha256(
        "anchor.synthetic-nl-scaffold-record-inventory.v1", record_ids
    )
    record_content_inventory = _inventory_sha256(
        "anchor.synthetic-nl-scaffold-record-content-inventory.v1",
        [_sha256(_canonical_json_bytes(record)) for record in records],
    )
    partition_entries = []
    for split, variant in PARTITION_KEYS:
        path = f"{split}/{variant}.jsonl"
        raw = partitions[path]
        partition_entries.append(
            {
                "path": path,
                "split": split,
                "variant": variant,
                "records": len(raw.splitlines()),
                "bytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
    read_set = [
        _artifact_descriptor(repo_root, snapshots[role])
        for role in (
            "config",
            "config_schema",
            "closed_grammar",
            "closed_grammar_schema",
            "record_schema",
            "manifest_schema",
            "implementation",
        )
    ]
    read_set_sha = _inventory_sha256(
        "anchor.synthetic-nl-scaffold-read-set.v1",
        [f"{item['path']}:{item['sha256']}" for item in read_set],
    )
    return {
        "schema_version": MANIFEST_VERSION,
        "status": "dataset_proxy_ready_training_not_authorized",
        "claim_scope": CLAIM_SCOPE,
        "producer": {
            "producer_version": PRODUCER_VERSION,
            "config": _artifact_descriptor(repo_root, snapshots["config"]),
            "config_schema": _artifact_descriptor(
                repo_root, snapshots["config_schema"]
            ),
            "closed_grammar": _artifact_descriptor(
                repo_root, snapshots["closed_grammar"]
            ),
            "closed_grammar_schema": _artifact_descriptor(
                repo_root, snapshots["closed_grammar_schema"]
            ),
            "record_schema": _artifact_descriptor(
                repo_root, snapshots["record_schema"]
            ),
            "manifest_schema": _artifact_descriptor(
                repo_root, snapshots["manifest_schema"]
            ),
            "implementation": _artifact_descriptor(
                repo_root, snapshots["implementation"]
            ),
            "canonical_json_policy": ("utf8_sort_keys_compact_no_normalization_lf_v1"),
            "atomic_publish": True,
            "final_toctou_recheck": True,
        },
        "counts": {
            "records": 100,
            "pairs": 50,
            "source_bundles": 10,
            "roles": 5,
            "languages": 2,
            "variants": 2,
            "train_records": 80,
            "eval_proxy_records": 20,
        },
        "generation_contract": {
            "seed_id": config["generation_contract"]["seed_id"],
            "seed_sha256": _sha256(
                config["generation_contract"]["seed_text"].encode("utf-8")
            ),
            "source_namespace": config["generation_contract"]["source_namespace"],
            "augmentation": "none",
            "split_before_augmentation": True,
        },
        "split_contract": {
            "group_key": "source_bundle_id",
            "algorithm": config["split_contract"]["algorithm"],
            "salt_sha256": _sha256(config["split_contract"]["salt"].encode("utf-8")),
            "train_bundle_ids": list(train_ids),
            "eval_proxy_bundle_ids": list(eval_ids),
            "all_role_variant_views_same_split": True,
            "eval_proxy_is_heldout": False,
        },
        "pair_contract": {
            "variants": list(VARIANTS),
            "same_input_sha256": True,
            "same_canonical_json_sha256": True,
            "same_allowed_and_forbidden_segments": True,
            "rationale_is_hidden_chain_of_thought": False,
        },
        "partitions": partition_entries,
        "inventories": {
            "record_ids_sha256": record_id_inventory,
            "record_content_sha256": record_content_inventory,
            "pair_ids_sha256": _inventory_sha256(
                "anchor.synthetic-nl-scaffold-pair-inventory.v1", pair_ids
            ),
            "source_bundle_ids_sha256": _inventory_sha256(
                "anchor.synthetic-nl-scaffold-bundle-inventory.v1", bundle_ids
            ),
            "arm_record_inventory_sha256": {
                label: record_content_inventory for label in ABLATION_LABELS
            },
        },
        "read_set": {
            "scope": "declared_semantic_generation_inputs_only",
            "ordered_artifacts": read_set,
            "ordered_artifacts_sha256": read_set_sha,
            "protected_source_paths_read": 0,
        },
        "protected_inventory_status": {
            "consumes_protected_inventories": False,
            "statuses": dict(config["protected_inventory_contract"]["statuses"]),
        },
        "source_disjoint_boundary": {
            "source_namespace": "anchor.synthetic-nl-scaffold-diagnostic.v1",
            "zero_intersection_claimed": False,
            "source_disjoint_attestation_emitted": False,
            "formal_source_disjoint_proven": False,
            "status": "unavailable_without_protected_inventory_identities",
        },
        "token_length_contract": {
            "status": "tokenizer_unbound",
            "token_counts_emitted": False,
            "truncation_allowed": False,
            "run_preflight_must_bind_tokenizer_and_full_chat_plus_target_lengths": True,
            "diagnostic_target_max_tokens": 1024,
            "diagnostic_preferred_p95_tokens": 768,
        },
        "ablation_contract": {
            "labels": list(ABLATION_LABELS),
            "same_record_inventory_for_all_arms": True,
            "assignment_location": "diagnostic_run_manifest_only",
            "producer_selects_winner": False,
            "target_modules_bound_by_dataset": False,
        },
        "claims": {
            "dataset_proxy_ready": True,
            "diagnostic_only": True,
            "formal": False,
            "training_authorized": False,
            "quality_validated": False,
            "eval_proxy_is_heldout": False,
            "physical_kv_reuse_claimed": False,
            "numeric_equivalence_claimed": False,
        },
        "audit": {
            "record_schema_validated": True,
            "manifest_schema_validated": True,
            "bundle_split_disjoint": True,
            "pair_invariants_validated": True,
            "forbidden_content_excluded": True,
            "mandatory_sidecar": True,
            "protected_body_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "real_tool_executions": 0,
        },
    }


def _validate_records(
    config: Mapping[str, Any],
    grammar: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    record_schema: Mapping[str, Any],
) -> None:
    for record in records:
        _validate_schema(record_schema, record, "synthetic_record_schema_failed")
    if len(records) != 100 or len({record["record_id"] for record in records}) != 100:
        _fail("synthetic_record_identity_invalid")
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    bundles: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        pairs[str(record["pair_id"])].append(record)
        bundles[str(record["source_bundle_id"])].append(record)
    if len(pairs) != 50 or len(bundles) != 10:
        _fail("synthetic_pair_or_bundle_count_invalid")
    for pair_records in pairs.values():
        if len(pair_records) != 2 or {item["variant"] for item in pair_records} != set(
            VARIANTS
        ):
            _fail("synthetic_pair_variant_invalid")
        left, right = pair_records
        if (
            left["input"] != right["input"]
            or left["forbidden_segment_ids"] != right["forbidden_segment_ids"]
            or left["board_segment_inventory"] != right["board_segment_inventory"]
            or left["target"]["canonical_json_sha256"]
            != right["target"]["canonical_json_sha256"]
            or left["target"]["canonical_routing_json"]
            != right["target"]["canonical_routing_json"]
        ):
            _fail("synthetic_pair_invariant_failed")
    for bundle_records in bundles.values():
        if (
            len(bundle_records) != 10
            or len({item["split"] for item in bundle_records}) != 1
        ):
            _fail("synthetic_bundle_split_leakage")
        if Counter(item["role"] for item in bundle_records) != Counter(
            {role: 2 for role in ROLES}
        ):
            _fail("synthetic_bundle_role_imbalance")
    counts = Counter(
        (record["split"], record["variant"], record["role"], record["language"])
        for record in records
    )
    for language in LANGUAGES:
        for role in ROLES:
            for variant in VARIANTS:
                if counts[("train", variant, role, language)] != 4:
                    _fail("synthetic_train_balance_invalid")
                if counts[("eval_proxy", variant, role, language)] != 1:
                    _fail("synthetic_eval_balance_invalid")
    bundle_by_key = {bundle["bundle_key"]: bundle for bundle in config["bundles"]}
    for record in records:
        bundle = bundle_by_key[record["synthetic_source"]["bundle_key"]]
        bundle_id, _ = _bundle_identity(bundle)
        segments = _make_segments(bundle, bundle_id, grammar)
        expected_input = build_training_view(
            bundle, str(record["role"]), segments, grammar
        )
        if record["input"] != expected_input:
            _fail("synthetic_materialized_input_mismatch")
        target = record["target"]
        canonical_route = _canonical_json(target["canonical_routing_json"])
        canonical_route_sha256 = _sha256(canonical_route.encode("utf-8"))
        if target["canonical_json_sha256"] != canonical_route_sha256:
            _fail("synthetic_canonical_route_hash_mismatch")
        summary = target["concise_rationale_summary"]
        if record["variant"] == "json_only":
            if summary is not None:
                _fail("synthetic_json_only_rationale_invalid")
            expected_output = canonical_route
        else:
            if (
                not isinstance(summary, str)
                or not summary
                or "\n" in summary
                or "\r" in summary
                or len(summary.encode("utf-8")) > 512
            ):
                _fail("synthetic_concise_rationale_invalid")
            expected_output = f"{summary}\n{canonical_route}"
        if target["serialized_assistant_output"] != expected_output or target[
            "output_sha256"
        ] != _sha256(expected_output.encode("utf-8")):
            _fail("synthetic_output_normal_form_invalid")
        source = record["synthetic_source"]
        expected_pair_id = _id(
            "syn-nl-pair-v1",
            {
                "domain": "anchor.synthetic-nl-scaffold-pair-id.v1",
                "source_bundle_id": record["source_bundle_id"],
                "role": record["role"],
                "input_sha256": record["input"]["input_sha256"],
                "canonical_json_sha256": canonical_route_sha256,
                "generator_config_sha256": source["generator_config_sha256"],
                "generator_implementation_sha256": source[
                    "generator_implementation_sha256"
                ],
                "closed_grammar_sha256": source["closed_grammar_sha256"],
                "generation_seed_sha256": source["generation_seed_sha256"],
            },
        )
        expected_record_id = _id(
            "syn-nl-record-v1",
            {
                "domain": "anchor.synthetic-nl-scaffold-record-id.v1",
                "pair_id": expected_pair_id,
                "variant": record["variant"],
                "output_sha256": target["output_sha256"],
            },
        )
        if (
            record["pair_id"] != expected_pair_id
            or record["record_id"] != expected_record_id
        ):
            _fail("synthetic_content_bound_identity_mismatch")
        stage = int(record["stage_index"])
        prompt = str(record["input"]["materialized_prompt"])
        expected_forbidden = [item["segment_id"] for item in segments[stage:]]
        if record["forbidden_segment_ids"] != expected_forbidden:
            _fail("synthetic_forbidden_inventory_mismatch")
        for segment in segments[stage:]:
            if (
                segment["segment_id"] in prompt
                or segment["segment_ref"] in prompt
                or segment["content"] in prompt
            ):
                _fail("synthetic_forbidden_content_leaked")
        for segment in segments[:stage]:
            if (
                segment["segment_ref"] not in prompt
                or segment["commit_summary"] not in prompt
                or segment["segment_id"] in prompt
                or segment["content"] in prompt
            ):
                _fail("synthetic_allowed_content_missing")
        rationale = record["target"]["concise_rationale_summary"]
        if rationale is not None and len(str(rationale).encode("utf-8")) > 512:
            _fail("synthetic_rationale_too_large")


def _expected_materialization(
    repo_root: Path,
    config: Mapping[str, Any],
    grammar: Mapping[str, Any],
    snapshots: Mapping[str, _Snapshot],
    record_schema: Mapping[str, Any],
    manifest_schema: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    records, train_ids, eval_ids = _generate_records(config, grammar, snapshots)
    _validate_records(config, grammar, records, record_schema)
    partitions = _partition_bytes(records)
    manifest = _build_manifest(
        repo_root, config, snapshots, records, partitions, train_ids, eval_ids
    )
    _validate_schema(
        manifest_schema, manifest, "synthetic_manifest_schema_validation_failed"
    )
    return partitions, manifest


def _write_atomic_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_dataset(
    repo_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Build the fixture atomically. Existing outputs are never overwritten."""

    root = Path(repo_root).resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve(strict=True)
    snapshots, config, grammar, record_schema, manifest_schema = (
        _load_contract_snapshots(root, config_file)
    )
    partitions, manifest = _expected_materialization(
        root, config, grammar, snapshots, record_schema, manifest_schema
    )
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_absolute_ancestry(
        output.parent, "synthetic_output_parent_invalid"
    )
    if _path_lexists(output):
        _fail("synthetic_output_already_exists")
    if _is_reparse_or_symlink(output.parent):
        _fail("synthetic_output_parent_invalid")
    parent_before = output.parent.stat()
    parent_identity = (parent_before.st_dev, parent_before.st_ino)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        for relative, raw in partitions.items():
            _write_atomic_file(temporary / relative, raw)
        manifest_raw = _canonical_json_bytes(manifest, newline=True)
        _write_atomic_file(temporary / "manifest.json", manifest_raw)
        sidecar = f"{_sha256(manifest_raw)}  manifest.json\n".encode("ascii")
        _write_atomic_file(temporary / "manifest.json.sha256", sidecar)
        audited = audit_dataset(root, config_file, temporary)
        if audited != manifest:
            _fail("synthetic_prepublication_audit_mismatch")
        temporary_identity = _stat_identity(temporary.stat())
        temporary_snapshots = _capture_artifact_snapshots(temporary)
        for role, snapshot in snapshots.items():
            snapshot.assert_unchanged(f"synthetic_{role}_changed_during_build")
        parent_after = output.parent.stat()
        if (
            _path_lexists(output)
            or _is_reparse_or_symlink(output.parent)
            or (parent_after.st_dev, parent_after.st_ino) != parent_identity
        ):
            _fail("synthetic_output_publish_race")
        _assert_artifact_snapshot_unchanged(
            temporary, temporary_identity, temporary_snapshots
        )
        _rename_directory_no_replace(temporary, output)
        return audit_dataset(root, config_file, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def audit_dataset(
    repo_root: str | Path,
    config_path: str | Path,
    artifact_dir: str | Path,
) -> Mapping[str, Any]:
    """Audit one artifact with single-byte snapshots and final TOCTOU rechecks."""

    root = Path(repo_root).resolve(strict=True)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve(strict=True)
    artifact = Path(artifact_dir)
    if not artifact.is_absolute():
        artifact = root / artifact
    artifact = artifact.absolute()
    snapshots, config, grammar, record_schema, manifest_schema = (
        _load_contract_snapshots(root, config_file)
    )
    artifact_snapshots = _capture_artifact_snapshots(artifact)
    expected_sidecar = (
        f"{artifact_snapshots['manifest'].sha256}  manifest.json\n".encode("ascii")
    )
    if artifact_snapshots["sidecar"].data != expected_sidecar:
        _fail("synthetic_manifest_sidecar_invalid")
    manifest = _strict_json(
        artifact_snapshots["manifest"].data, "synthetic_manifest_invalid"
    )
    if (
        _canonical_json_bytes(manifest, newline=True)
        != artifact_snapshots["manifest"].data
    ):
        _fail("synthetic_manifest_not_canonical")
    _validate_schema(
        manifest_schema, manifest, "synthetic_manifest_schema_validation_failed"
    )
    observed_records: list[Mapping[str, Any]] = []
    for relative in PARTITION_PATHS:
        observed_records.extend(
            _strict_jsonl(
                artifact_snapshots[relative].data, "synthetic_partition_invalid"
            )
        )
    _validate_records(config, grammar, observed_records, record_schema)
    expected_partitions, expected_manifest = _expected_materialization(
        root, config, grammar, snapshots, record_schema, manifest_schema
    )
    for relative, expected in expected_partitions.items():
        if artifact_snapshots[relative].data != expected:
            _fail("synthetic_partition_materialization_mismatch")
    if dict(manifest) != expected_manifest:
        _fail("synthetic_manifest_materialization_mismatch")
    _assert_exact_artifact_layout(artifact)
    for role, snapshot in snapshots.items():
        snapshot.assert_unchanged(f"synthetic_{role}_changed_during_audit")
    for role, snapshot in artifact_snapshots.items():
        snapshot.assert_unchanged(f"synthetic_{role}_changed_during_audit")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or audit the synthetic scaffold diagnostic v1 fixture"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("--repo-root", default=".")
        child.add_argument("--config", default=CONFIG_PATH)
        child.add_argument("--artifact", default=CANONICAL_FIXTURE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        manifest = build_dataset(args.repo_root, args.config, args.artifact)
    else:
        manifest = audit_dataset(args.repo_root, args.config, args.artifact)
    print(_canonical_json(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
