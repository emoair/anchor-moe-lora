"""Strict model-free consumer binding for Gemma 3 Chat unbalanced-v2.

The preflight authenticates the exact physical artifact tree and aggregate
metadata only.  JSONL payloads are streamed solely into SHA-256; their bodies
are never parsed, retained, or emitted.  This module never copies the dataset,
loads a model, requests a GPU, or contacts a provider.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


NAMESPACE = "gemma3_chat_five_expert_qonly_unbalanced_v2"
BINDING_VERSION = "anchor.gemma3-chat-unbalanced-v2-consumer-binding.v1"
RECEIPT_VERSION = "anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.v1"
MANIFEST_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-manifest.v1"
)
BUILD_RECEIPT_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-build-receipt.v1"
)
SCHEMA_RELATIVE_PATH = (
    "configs/research/gemma3_chat_unbalanced_v2_consumer_binding_v1.schema.json"
)
TREE_DIGEST_DOMAIN = "anchor.gemma3-chat-unbalanced-v2-physical-tree.v1"
MAX_PAYLOAD_BYTES = 52_428_799

PAYLOAD_PATHS = (
    "build_receipt.json",
    "components/planner_eval/build_receipt.json",
    "components/planner_eval/manifest.json",
    "components/planner_eval/negative_inventory.jsonl",
    "components/planner_eval/records.jsonl",
    "components/planner_eval/token_inventory.jsonl",
    "components/router/build_receipt.json",
    "components/router/manifest.json",
    "components/router/records.jsonl",
    "components/router/token_inventory.jsonl",
    "components/tool_eval/build_receipt.json",
    "components/tool_eval/manifest.json",
    "components/tool_eval/negative_inventory.jsonl",
    "components/tool_eval/records.jsonl",
    "components/tool_eval/token_inventory.jsonl",
    "eval_proxy/chat.jsonl",
    "identity_eval/probe_inventory.jsonl",
    "manifest.json",
    "reviews/planner_eval_independent_review.json",
    "reviews/tool_eval_independent_review.json",
    "serialization_inventory.jsonl",
    "train/chat.jsonl",
)
PAYLOAD_PATH_SET = frozenset(PAYLOAD_PATHS)
SIDECAR_PATHS = tuple(f"{path}.sha256" for path in PAYLOAD_PATHS)
EXACT_TREE_PATHS = frozenset((*PAYLOAD_PATHS, *SIDECAR_PATHS))
MANIFEST_LISTED_PATHS = EXACT_TREE_PATHS.difference(
    {
        "manifest.json",
        "manifest.json.sha256",
        "build_receipt.json",
        "build_receipt.json.sha256",
    }
)
RECORD_COUNTS: Mapping[str, int | None] = {
    "train/chat.jsonl": 3440,
    "eval_proxy/chat.jsonl": 860,
    "serialization_inventory.jsonl": 4300,
    "identity_eval/probe_inventory.jsonl": 50,
    "components/router/records.jsonl": 100,
    "components/router/token_inventory.jsonl": 100,
    "components/tool_eval/records.jsonl": 400,
    "components/tool_eval/token_inventory.jsonl": 400,
    "components/tool_eval/negative_inventory.jsonl": 44,
    "components/planner_eval/records.jsonl": 240,
    "components/planner_eval/token_inventory.jsonl": 240,
    "components/planner_eval/negative_inventory.jsonl": 53,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ZERO_RESOURCES = {
    "provider_requests": 0,
    "network_requests": 0,
    "model_loads": 0,
    "gpu_requests": 0,
    "gold_body_reads": 0,
    "heldout_body_reads": 0,
    "protected_body_reads": 0,
}


class ConsumerPreflightError(RuntimeError):
    """Body-free fail-closed preflight error."""


class _DuplicateJsonKey(ValueError):
    pass


def _fail(code: str) -> None:
    raise ConsumerPreflightError(code)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(code)
    return value


def _strict_json(raw: bytes, label: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJsonKey(key)
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(value)

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKey,
        ValueError,
    ) as exc:
        raise ConsumerPreflightError(f"{label}_json_invalid") from exc
    return _mapping(parsed, f"{label}_mapping_invalid")


def _schema_validator(
    *,
    ignore_payload_size_maximum: bool = False,
) -> Draft202012Validator:
    schema_path = _project_root() / SCHEMA_RELATIVE_PATH
    try:
        raw = schema_path.read_bytes()
    except OSError as exc:
        raise ConsumerPreflightError("consumer_binding_schema_unreadable") from exc
    schema = _strict_json(raw, "consumer_binding_schema")
    if ignore_payload_size_maximum:
        schema = copy.deepcopy(schema)
        try:
            del schema["$defs"]["payload_pin"]["properties"]["bytes"]["maximum"]
        except (KeyError, TypeError) as exc:
            raise ConsumerPreflightError(
                "consumer_binding_schema_size_gate_missing"
            ) from exc
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ConsumerPreflightError("consumer_binding_schema_invalid") from exc
    return Draft202012Validator(schema)


def _validate_binding_document(
    value: object,
    *,
    ignore_payload_size_maximum: bool,
) -> Mapping[str, Any]:
    binding = _mapping(value, "binding_document_mapping_invalid")
    try:
        _schema_validator(
            ignore_payload_size_maximum=ignore_payload_size_maximum
        ).validate(binding)
    except ValidationError as exc:
        raise ConsumerPreflightError("binding_document_schema_mismatch") from exc
    if binding.get("schema_version") != BINDING_VERSION:
        _fail("binding_document_version_mismatch")
    pins = _sequence(binding.get("payload_files"), "binding_payload_files_invalid")
    paths: list[str] = []
    for pin_value in pins:
        pin = _mapping(pin_value, "binding_payload_pin_invalid")
        path = pin.get("path")
        if not isinstance(path, str):
            _fail("binding_payload_path_invalid")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in path:
            _fail("binding_payload_path_escape")
        paths.append(path)
        expected_kind = "jsonl" if path.endswith(".jsonl") else "json"
        if pin.get("kind") != expected_kind:
            _fail("binding_payload_kind_mismatch")
    if len(paths) != len(set(paths)):
        _fail("binding_payload_path_duplicate")
    if frozenset(paths) != PAYLOAD_PATH_SET:
        _fail("binding_payload_inventory_mismatch")
    return binding


def validate_binding_document(value: object) -> Mapping[str, Any]:
    return _validate_binding_document(
        value,
        ignore_payload_size_maximum=False,
    )


def load_binding_document(path: str | Path) -> Mapping[str, Any]:
    snapshot = _snapshot_file(Path(path), capture=True)
    if snapshot.data is None:
        _fail("binding_document_bytes_missing")
    binding = _validate_binding_document(
        _strict_json(snapshot.data, "binding_document"),
        ignore_payload_size_maximum=True,
    )
    terminal = _snapshot_file(snapshot.path, capture=False)
    if not snapshot.same_identity_and_bytes(terminal):
        _fail("binding_document_toctou")
    return binding


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_signature(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _assert_directory(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ConsumerPreflightError(f"{label}_unreadable") from exc
    if path.is_symlink() or _is_reparse(value):
        _fail(f"{label}_symlink_or_reparse")
    if not stat.S_ISDIR(value.st_mode):
        _fail(f"{label}_not_directory")
    return value


def _assert_no_symlink_or_reparse_ancestors(path: Path, label: str) -> None:
    current = path
    while True:
        try:
            value = current.lstat()
        except OSError as exc:
            raise ConsumerPreflightError(f"{label}_ancestor_unreadable") from exc
        if current.is_symlink() or _is_reparse(value):
            _fail(f"{label}_ancestor_symlink_or_reparse")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _assert_regular(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ConsumerPreflightError(f"{label}_unreadable") from exc
    if path.is_symlink() or _is_reparse(value):
        _fail(f"{label}_symlink_or_reparse")
    if not stat.S_ISREG(value.st_mode):
        _fail(f"{label}_not_regular_file")
    return value


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    bytes: int
    stat_signature: tuple[int, int, int, int, int]
    data: bytes | None

    def same_identity_and_bytes(self, other: FileSnapshot) -> bool:
        return (
            self.path == other.path
            and self.sha256 == other.sha256
            and self.bytes == other.bytes
            and self.stat_signature == other.stat_signature
        )


@dataclass(frozen=True)
class TreeSnapshot:
    root_signature: tuple[int, int]
    directory_signatures: Mapping[str, tuple[int, int]]
    files: Mapping[str, FileSnapshot]


def _snapshot_file(path: Path, *, capture: bool) -> FileSnapshot:
    before_path = _assert_regular(path, "artifact_file")
    before_path_signature = _stat_signature(before_path)
    digest = hashlib.sha256()
    captured: list[bytes] | None = [] if capture else None
    total = 0
    try:
        with path.open("rb") as handle:
            before_handle = os.fstat(handle.fileno())
            if _stat_signature(before_handle) != before_path_signature:
                _fail("artifact_file_open_identity_drift")
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                if captured is not None:
                    captured.append(chunk)
            after_handle = os.fstat(handle.fileno())
    except OSError as exc:
        raise ConsumerPreflightError("artifact_file_stream_failed") from exc
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise ConsumerPreflightError("artifact_file_terminal_stat_failed") from exc
    signatures = {
        before_path_signature,
        _stat_signature(before_handle),
        _stat_signature(after_handle),
        _stat_signature(after_path),
    }
    if (
        len(signatures) != 1
        or total != before_path.st_size
        or path.is_symlink()
        or _is_reparse(after_path)
    ):
        _fail("artifact_file_stream_toctou")
    return FileSnapshot(
        path=path,
        sha256=digest.hexdigest(),
        bytes=total,
        stat_signature=before_path_signature,
        data=b"".join(captured) if captured is not None else None,
    )


def _scan_tree(
    root: Path,
) -> tuple[tuple[int, int], dict[str, tuple[int, int]], set[str]]:
    root_value = _assert_directory(root, "artifact_root")
    root_signature = _directory_signature(root_value)
    directories: dict[str, tuple[int, int]] = {"": root_signature}
    files: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ConsumerPreflightError("artifact_tree_scan_failed") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
                _fail("artifact_tree_path_escape")
            try:
                value = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ConsumerPreflightError("artifact_tree_lstat_failed") from exc
            if entry.is_symlink() or _is_reparse(value):
                _fail("artifact_tree_symlink_or_reparse")
            if stat.S_ISDIR(value.st_mode):
                directories[relative] = _directory_signature(value)
                stack.append(path)
            elif stat.S_ISREG(value.st_mode):
                files.add(relative)
            else:
                _fail("artifact_tree_special_file")
    return root_signature, directories, files


def _snapshot_tree(root: Path, *, capture_metadata: bool) -> TreeSnapshot:
    root_signature, directories, files = _scan_tree(root)
    if files != EXACT_TREE_PATHS:
        _fail("artifact_tree_file_inventory_mismatch")
    expected_directories = {""}
    for relative in EXACT_TREE_PATHS:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if set(directories) != expected_directories:
        _fail("artifact_tree_directory_inventory_mismatch")
    snapshots: dict[str, FileSnapshot] = {}
    for relative in sorted(EXACT_TREE_PATHS):
        capture = capture_metadata and (
            relative.endswith(".json") or relative.endswith(".sha256")
        )
        snapshots[relative] = _snapshot_file(
            root.joinpath(*PurePosixPath(relative).parts),
            capture=capture,
        )
    return TreeSnapshot(
        root_signature=root_signature,
        directory_signatures=directories,
        files=snapshots,
    )


def _terminal_tree_recheck(root: Path, baseline: TreeSnapshot) -> None:
    root_signature, directories, files = _scan_tree(root)
    if (
        root_signature != baseline.root_signature
        or directories != baseline.directory_signatures
        or files != EXACT_TREE_PATHS
    ):
        _fail("artifact_tree_terminal_identity_drift")
    for relative, previous in baseline.files.items():
        current = _assert_regular(
            root.joinpath(*PurePosixPath(relative).parts),
            "artifact_terminal_file",
        )
        if _stat_signature(current) != previous.stat_signature:
            _fail("artifact_terminal_file_identity_drift")


def _tree_digest(files: Mapping[str, FileSnapshot]) -> str:
    inventory = [
        {
            "path": path,
            "sha256": files[path].sha256,
            "bytes": files[path].bytes,
        }
        for path in sorted(files)
    ]
    preimage = {"domain": TREE_DIGEST_DOMAIN, "files": inventory}
    return _sha256(_canonical_json(preimage))


def _kind(path: str) -> str:
    if path.endswith(".sha256"):
        return "sha256_sidecar"
    if path.endswith(".jsonl"):
        return "jsonl"
    return "json"


def _payload_pins(binding: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    pins: dict[str, Mapping[str, Any]] = {}
    for value in _sequence(binding["payload_files"], "binding_payload_files_invalid"):
        pin = _mapping(value, "binding_payload_pin_invalid")
        path = str(pin["path"])
        if path in pins:
            _fail("binding_payload_path_duplicate")
        pins[path] = pin
    return pins


def _authenticate_pins(
    binding: Mapping[str, Any],
    snapshot: TreeSnapshot,
) -> None:
    pins = _payload_pins(binding)
    for path in PAYLOAD_PATHS:
        payload = snapshot.files[path]
        sidecar_path = f"{path}.sha256"
        sidecar = snapshot.files[sidecar_path]
        pin = pins[path]
        if (
            pin["sha256"] != payload.sha256
            or pin["bytes"] != payload.bytes
            or pin["sidecar_sha256"] != sidecar.sha256
            or pin["sidecar_bytes"] != sidecar.bytes
        ):
            _fail("artifact_binding_file_identity_drift")
        if sidecar.data is None:
            _fail("artifact_sidecar_bytes_missing")
        expected_sidecar = f"{payload.sha256}  {PurePosixPath(path).name}\n".encode(
            "ascii"
        )
        if sidecar.data != expected_sidecar:
            _fail("artifact_sidecar_content_mismatch")
    observed_tree_digest = _tree_digest(snapshot.files)
    if observed_tree_digest != binding["tree_digest_sha256"]:
        _fail("artifact_tree_digest_drift")


def _assert_payload_size(size: int) -> None:
    if size < 0 or size > MAX_PAYLOAD_BYTES:
        _fail("consumer_payload_size_limit_exceeded")


def _validate_payload_sizes(files: Mapping[str, FileSnapshot]) -> None:
    for path in PAYLOAD_PATHS:
        _assert_payload_size(files[path].bytes)


def _claims_gate(value: object, label: str) -> Mapping[str, Any]:
    claims = _mapping(value, f"{label}_claims_invalid")
    required = {
        "candidate": False,
        "final": True,
        "diagnostic_only": True,
        "proxy_only": True,
        "training_authorized": True,
        "formal_training_authorized": False,
        "live_authorized": False,
        "release_authorized": True,
    }
    for key, expected in required.items():
        if claims.get(key) is not expected:
            _fail(f"{label}_{key}_gate_failed")
    return claims


def _require_equal(actual: object, expected: object, code: str) -> None:
    if actual != expected:
        _fail(code)


def _validate_counts(manifest: Mapping[str, Any]) -> None:
    expected = {
        "records": 4300,
        "task_bundles": 1700,
        "task_semantics": 1700,
        "roles": {
            "humor": 300,
            "serious": 300,
            "angry_style": 300,
            "tool_call": 1700,
            "review_audit": 1700,
        },
        "splits": {"train": 3440, "eval_proxy": 860},
        "languages": {"en": 2150, "zh-CN": 2150},
        "language_split": {
            "en:train": 1720,
            "zh-CN:train": 1720,
            "en:eval_proxy": 430,
            "zh-CN:eval_proxy": 430,
        },
        "bundle_classes": {
            "full_five_role_core": 300,
            "tool_review_depth": 1400,
        },
        "review_verdicts": {"pass": 850, "fail": 850},
        "review_fault_types": {
            "format_schema": 170,
            "tool_argument": 170,
            "evidence_mismatch": 170,
            "grounding_mismatch": 170,
            "routing_scope": 170,
        },
        "identity_bundles": 30,
        "identity_role_records": 150,
        "identity_eval_probe_records": 50,
        "tool_direct_no_tool": 30,
        "tool_local_executions": 1670,
    }
    _require_equal(manifest.get("counts"), expected, "manifest_counts_mismatch")
    expected_identity = {
        "source_pool": "full_five_role_core",
        "rows_added_to_training_4300": 0,
        "identity_bundles": 30,
        "train_bundles": 20,
        "eval_proxy_bundles": 10,
        "en_bundles": 15,
        "zh_cn_bundles": 15,
        "intents": {
            "direct_self_identity": 6,
            "provenance_fact_check": 6,
            "no_tool_identity_decision": 6,
            "pre_coding_attribution": 6,
            "false_attribution_correction": 6,
        },
        "role_records": 150,
        "eval_probe_records": 50,
        "tool_behavior": "direct_answer_no_tool",
        "review_rejects_google_openai": True,
        "wrong_route_fact_invariant": True,
    }
    _require_equal(
        manifest.get("identity"),
        expected_identity,
        "manifest_identity_summary_mismatch",
    )
    expected_review = {
        "records": 1700,
        "pass": 850,
        "fail": 850,
        "fault_types": expected["review_fault_types"],
        "same_bundle_tool_dependency": True,
        "candidate_projection_hash_bound": True,
        "tool_record_never_mutated_by_review_fixture": True,
    }
    _require_equal(
        manifest.get("review"),
        expected_review,
        "manifest_review_summary_mismatch",
    )


def _validate_components(
    manifest: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    components = _mapping(manifest.get("components"), "manifest_components_invalid")
    if set(components) != {"router", "tool_eval", "planner_eval"}:
        _fail("manifest_component_inventory_mismatch")
    router = _mapping(components["router"], "manifest_router_invalid")
    router_expected = {
        "namespace": "gemma3_chat_emotion_router_qonly_v1",
        "records": 100,
        "train": 80,
        "eval_proxy": 20,
        "en": 50,
        "zh_cn": 50,
        "labels": {"humor": 34, "serious": 33, "angry_style": 33},
        "not_a_sixth_expert": True,
        "single_total_to_branch_then_router_exits": True,
        "independent_from_generation_experts": True,
    }
    for key, expected in router_expected.items():
        _require_equal(router.get(key), expected, f"manifest_router_{key}_mismatch")
    router_hashes = {
        "manifest_sha256": "components/router/manifest.json",
        "records_sha256": "components/router/records.jsonl",
        "token_inventory_sha256": "components/router/token_inventory.jsonl",
        "build_receipt_sha256": "components/router/build_receipt.json",
    }
    for key, path in router_hashes.items():
        _require_equal(
            router.get(key), files[path].sha256, f"manifest_router_{key}_drift"
        )
    for name, records, slots, namespace in (
        (
            "tool_eval",
            400,
            1200,
            "gemma3_chat_unbalanced_v2_tool_comparison_eval_v1",
        ),
        (
            "planner_eval",
            240,
            960,
            "gemma3_chat_unbalanced_v2_planner_comparison_eval_v1",
        ),
    ):
        component = _mapping(components[name], f"manifest_{name}_component_invalid")
        expected_values = {
            "namespace": namespace,
            "records": records,
            "execution_slots": slots,
            "arm_neutral": True,
            "rows_replicated_by_arms": False,
            "oracle_unchanged": True,
            "strata_unchanged": True,
            "language_unchanged": True,
            "negative_inventory_unchanged": True,
            "independent_review_status": "passed",
        }
        for key, expected in expected_values.items():
            _require_equal(
                component.get(key),
                expected,
                f"manifest_{name}_{key}_mismatch",
            )
        review_path = f"reviews/{name}_independent_review.json"
        hash_paths = {
            "manifest_sha256": f"components/{name}/manifest.json",
            "records_sha256": f"components/{name}/records.jsonl",
            "token_inventory_sha256": (f"components/{name}/token_inventory.jsonl"),
            "build_receipt_sha256": f"components/{name}/build_receipt.json",
            "independent_review_receipt_sha256": review_path,
        }
        for key, path in hash_paths.items():
            _require_equal(
                component.get(key),
                files[path].sha256,
                f"manifest_{name}_{key}_drift",
            )


def _validate_adapter_and_kv(manifest: Mapping[str, Any]) -> None:
    adapter = _mapping(
        manifest.get("adapter_contract"), "manifest_adapter_contract_invalid"
    )
    expected_adapter = {
        "tool_arms": ["base", "tool_q_only", "tool_q_plus_micro_o"],
        "planner_arms": [
            "base",
            "planner_q_only",
            "planner_o_only",
            "planner_q_plus_micro_o",
        ],
        "tool_execution_slots": 1200,
        "planner_execution_slots": 960,
        "q_branch": {
            "target_module": "q_proj",
            "rank": 1024,
            "alpha": 2048,
            "trainable_params": 57933824,
        },
        "micro_o_branch": {
            "target_module": "o_proj",
            "rank": 64,
            "alpha": 128,
            "trainable_params": 3620864,
            "max_lr_ratio_to_q": "1/10",
        },
        "non_tool_q_only_roles": [
            "humor",
            "serious",
            "angry_style",
            "review_audit",
        ],
        "records_added_by_arms": 0,
        "tool_records_arm_neutral": True,
        "planner_records_arm_neutral": True,
        "o_learning_rate_lte_q_over_10_every_step": True,
        "planner_o_only_q_plus_o_o_branch_identity_equal": True,
        "planner_o_only_q_plus_o_o_init_equal": True,
        "planner_o_only_q_plus_o_data_order_steps_seeds_optimizer_equal": True,
        "humor_serious_angry_review_q_only": True,
    }
    _require_equal(adapter, expected_adapter, "manifest_adapter_contract_mismatch")
    kv = _mapping(manifest.get("kv_contract"), "manifest_kv_contract_invalid")
    required_kv = {
        "producer_mode": "frozen_base_adapter_off_single_shared_prefix_prefill",
        "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
        "current_physical_claim": "prefill_compute_reuse",
        "route_plan_commits_immutable_and_hash_bound": True,
        "gemma_canonical_serialization": True,
        "expert_private_tail_append_only": True,
        "planner_live_private_kv_transfer": False,
        "full_generation_kv_shared": False,
        "persistent_zero_copy_verified": False,
        "q8_kv_exact": False,
    }
    for key, expected in required_kv.items():
        _require_equal(kv.get(key), expected, f"manifest_kv_{key}_mismatch")
    if kv.get("persistent_zero_copy_blocked_on") != [
        "data_ptr_identity",
        "storage_identity",
        "cache_identity",
    ]:
        _fail("manifest_kv_zero_copy_blockers_mismatch")


def _validate_manifest_file_inventory(
    manifest: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    entries = _sequence(manifest.get("files"), "manifest_files_invalid")
    observed: dict[str, Mapping[str, Any]] = {}
    for value in entries:
        entry = _mapping(value, "manifest_file_entry_invalid")
        if set(entry) != {"path", "sha256", "bytes", "kind", "records"}:
            _fail("manifest_file_entry_schema_mismatch")
        path = entry.get("path")
        if not isinstance(path, str) or path in observed:
            _fail("manifest_file_path_duplicate_or_invalid")
        observed[path] = entry
    if set(observed) != MANIFEST_LISTED_PATHS:
        _fail("manifest_file_inventory_mismatch")
    for path, entry in observed.items():
        snapshot = files[path]
        if (
            entry["sha256"] != snapshot.sha256
            or entry["bytes"] != snapshot.bytes
            or entry["kind"] != _kind(path)
            or entry["records"] != RECORD_COUNTS.get(path)
        ):
            _fail("manifest_file_binding_drift")


def _validate_manifest(
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    if manifest.get("schema_version") != MANIFEST_VERSION:
        _fail("manifest_schema_version_mismatch")
    if manifest.get("namespace") != NAMESPACE:
        _fail("manifest_namespace_mismatch")
    status = manifest.get("status")
    if (
        not isinstance(status, str)
        or status != binding["expected_manifest_status"]
        or "candidate" in status
        or "pending" in status
    ):
        _fail("manifest_release_status_not_final")
    _claims_gate(manifest.get("claims"), "manifest")
    _validate_manifest_file_inventory(manifest, files)
    _validate_counts(manifest)
    _validate_components(manifest, files)
    _validate_adapter_and_kv(manifest)
    integrity = _mapping(manifest.get("integrity"), "manifest_integrity_invalid")
    required_integrity = {
        "single_authenticated_physical_bytes_snapshot_per_input": True,
        "terminal_toctou_snapshot_recheck_required": True,
        "mandatory_sidecar_for_every_non_sidecar_file": True,
        "checksum_sidecars_are_only_sidecar_exemption": True,
        "raw_token_ids_persisted": False,
        "strict_json_duplicate_and_nonfinite_rejection": True,
        "cross_field_validator": True,
    }
    for key, expected in required_integrity.items():
        _require_equal(
            integrity.get(key), expected, f"manifest_integrity_{key}_mismatch"
        )
    review = _mapping(manifest.get("release_review"), "manifest_release_review_invalid")
    if (
        review.get("implementer_release_signoff") is not False
        or review.get("independent_release_review_required") is not True
    ):
        _fail("manifest_release_review_contract_mismatch")


def _validate_build_receipt(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    files: Mapping[str, FileSnapshot],
) -> None:
    if receipt.get("schema_version") != BUILD_RECEIPT_VERSION:
        _fail("build_receipt_schema_version_mismatch")
    if receipt.get("status") != manifest.get("status"):
        _fail("build_receipt_status_mismatch")
    _claims_gate(receipt.get("claims"), "build_receipt")
    manifest_binding = _mapping(
        receipt.get("manifest"), "build_receipt_manifest_binding_invalid"
    )
    manifest_snapshot = files["manifest.json"]
    if (
        manifest_binding.get("path") != "manifest.json"
        or manifest_binding.get("sha256") != manifest_snapshot.sha256
        or manifest_binding.get("bytes") != manifest_snapshot.bytes
    ):
        _fail("build_receipt_manifest_binding_drift")
    resources = _mapping(
        receipt.get("resource_counters"), "build_receipt_resources_invalid"
    )
    for key, expected in _ZERO_RESOURCES.items():
        if resources.get(key) != expected:
            _fail("build_receipt_nonzero_resource_counter")
    audits = _mapping(
        receipt.get("official_model_free_audits"),
        "build_receipt_official_audits_invalid",
    )
    if audits.get("independent_release_review") != "passed":
        _fail("build_receipt_independent_release_review_not_passed")


def _metadata_json(
    snapshot: FileSnapshot,
    label: str,
) -> Mapping[str, Any]:
    if snapshot.data is None:
        _fail(f"{label}_authenticated_bytes_missing")
    return _strict_json(snapshot.data, label)


def _receipt(
    *,
    operation: str,
    binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    build_receipt: Mapping[str, Any],
    snapshot: TreeSnapshot,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_VERSION,
        "status": "passed",
        "operation": operation,
        "namespace": NAMESPACE,
        "model_free": True,
        "binding_contract_sha256": _sha256(_canonical_json(binding)),
        "producer_git_commit": binding["producer_git_commit"],
        "producer_manifest_schema_sha256": binding["producer_manifest_schema_sha256"],
        "producer_build_receipt_schema_sha256": binding[
            "producer_build_receipt_schema_sha256"
        ],
        "release_review_receipt_sha256": binding["release_review"]["receipt_sha256"],
        "tree_digest_sha256": _tree_digest(snapshot.files),
        "file_counts": {"total": 44, "payload": 22, "sidecar": 22},
        "manifest": {
            "schema_version": str(manifest["schema_version"]),
            "sha256": snapshot.files["manifest.json"].sha256,
            "bytes": snapshot.files["manifest.json"].bytes,
        },
        "build_receipt": {
            "schema_version": str(build_receipt["schema_version"]),
            "sha256": snapshot.files["build_receipt.json"].sha256,
            "bytes": snapshot.files["build_receipt.json"].bytes,
        },
        "aggregate_counts": {
            "training_records": 4300,
            "training_train_records": 3440,
            "training_eval_proxy_records": 860,
            "router_records": 100,
            "router_train_records": 80,
            "router_eval_proxy_records": 20,
            "tool_eval_records": 400,
            "planner_eval_records": 240,
            "identity_probe_records": 50,
        },
        "release": {
            "manifest_status": manifest["status"],
            "independent_release_review": "passed",
            "final": True,
            "release_authorized": True,
            "training_authorized": True,
            "formal_training_authorized": False,
        },
        "terminal_recheck": {
            "two_snapshot_bytes_equal": True,
            "stat_identity_equal": True,
            "physical_inventory_equal": True,
        },
        "resource_counters": dict(_ZERO_RESOURCES),
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "training_started": False,
            "data_copied": False,
            "sample_bodies_parsed": False,
            "raw_token_ids_parsed": False,
        },
    }


def validate_artifact(
    artifact_root: str | Path,
    binding_document: Mapping[str, Any],
    *,
    observed_producer_commit: str,
    operation: str = "validate",
    before_terminal_recheck: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Authenticate the exact tree and return a body-free receipt.

    ``before_terminal_recheck`` exists for deterministic race-negative tests;
    production CLI callers cannot supply it.
    """

    if operation not in {"validate", "dry-run"}:
        _fail("consumer_operation_invalid")
    binding = _validate_binding_document(
        binding_document,
        ignore_payload_size_maximum=True,
    )
    if (
        _COMMIT_RE.fullmatch(observed_producer_commit) is None
        or observed_producer_commit == "0" * 40
    ):
        _fail("observed_producer_commit_invalid")
    if observed_producer_commit != binding["producer_git_commit"]:
        _fail("producer_git_commit_drift")
    root = Path(artifact_root).absolute()
    _assert_no_symlink_or_reparse_ancestors(root, "artifact_root")
    first = _snapshot_tree(root, capture_metadata=True)
    _authenticate_pins(binding, first)
    manifest = _metadata_json(first.files["manifest.json"], "manifest")
    build_receipt = _metadata_json(first.files["build_receipt.json"], "build_receipt")
    _validate_manifest(manifest, binding, first.files)
    _validate_build_receipt(build_receipt, manifest, first.files)
    _validate_payload_sizes(first.files)
    validate_binding_document(binding)
    if before_terminal_recheck is not None:
        before_terminal_recheck()
    second = _snapshot_tree(root, capture_metadata=False)
    if (
        first.root_signature != second.root_signature
        or first.directory_signatures != second.directory_signatures
        or set(first.files) != set(second.files)
    ):
        _fail("artifact_two_snapshot_tree_identity_drift")
    for relative, previous in first.files.items():
        if not previous.same_identity_and_bytes(second.files[relative]):
            _fail("artifact_two_snapshot_byte_drift")
    _terminal_tree_recheck(root, second)
    result = _receipt(
        operation=operation,
        binding=binding,
        manifest=manifest,
        build_receipt=build_receipt,
        snapshot=second,
    )
    try:
        _schema_validator().validate(result)
    except ValidationError as exc:
        raise ConsumerPreflightError("preflight_receipt_schema_mismatch") from exc
    return result


def _atomic_write_receipt(
    output: str | Path,
    receipt: Mapping[str, Any],
    *,
    artifact_root: str | Path,
) -> None:
    target = Path(output).absolute()
    root = Path(artifact_root).absolute()
    try:
        target.relative_to(root)
    except ValueError:
        pass
    else:
        _fail("receipt_output_inside_authenticated_artifact")
    parent = target.parent
    _assert_directory(parent, "receipt_output_parent")
    raw = _canonical_json(receipt)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise ConsumerPreflightError("receipt_atomic_write_failed") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Model-free strict consumer authentication for Gemma 3 Chat unbalanced-v2."
        )
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("validate", "dry-run"):
        command = subparsers.add_parser(operation)
        command.add_argument("--artifact", required=True)
        command.add_argument("--binding", required=True)
        command.add_argument("--producer-commit", required=True)
        command.add_argument(
            "--output",
            help=(
                "Explicit path for an atomic body-free receipt. No file is "
                "written when omitted."
            ),
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        binding = load_binding_document(args.binding)
        receipt = validate_artifact(
            args.artifact,
            binding,
            observed_producer_commit=args.producer_commit,
            operation=args.operation,
        )
        if args.output:
            _atomic_write_receipt(
                args.output,
                receipt,
                artifact_root=args.artifact,
            )
    except ConsumerPreflightError as exc:
        failure = {"status": "failed", "error_code": str(exc)}
        print(
            json.dumps(failure, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
