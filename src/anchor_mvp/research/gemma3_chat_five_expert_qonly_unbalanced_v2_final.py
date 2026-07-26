"""Model-free sharded FINAL candidate materializer for Gemma 3 Chat v2.

This module consumes only authenticated local bytes.  It never loads a model,
uses a GPU, performs network/provider/Luna work, or reads protected, Gold, or
held-out bodies.  Published data is diagnostic/proxy-only and remains pending
an independent release review.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


NAMESPACE = "gemma3_chat_five_expert_qonly_unbalanced_v2"
STATUS = "candidate_pending_independent_release_review"
ARTIFACT_VERSION = "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-sharded.v3"
CANONICAL_ARTIFACT_REL = (
    "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_distilled_v2_sharded_v3"
)
SUPERSEDED_ARTIFACT_REL = (
    "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_distilled_v2_sharded_v2"
)
LEGACY_UNSHARDED_ARTIFACT_REL = (
    "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_distilled_v2"
)
CANONICAL_REVIEW_RELS = {
    "tool_eval": (
        "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_v2/"
        "release_inputs/tool_eval_independent_review.json"
    ),
    "planner_eval": (
        "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_v2/"
        "release_inputs/planner_eval_independent_review.json"
    ),
}

CONFIG_REL = "configs/research/gemma3_chat_five_expert_qonly_unbalanced_v2_final.json"
CONFIG_SCHEMA_REL = (
    "configs/research/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_final_config.schema.json"
)
RECORD_SCHEMA_REL = (
    "configs/research/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_final_record.schema.json"
)
SERIALIZATION_SCHEMA_REL = (
    "configs/research/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_final_"
    "serialization_inventory.schema.json"
)
IDENTITY_PROBE_SCHEMA_REL = (
    "configs/research/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_final_"
    "identity_probe.schema.json"
)
MANIFEST_SCHEMA_REL = (
    "configs/research/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_final_manifest.schema.json"
)
RECEIPT_SCHEMA_REL = (
    "configs/research/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_final_"
    "build_receipt.schema.json"
)
IMPLEMENTATION_REL = (
    "src/anchor_mvp/research/gemma3_chat_five_expert_qonly_unbalanced_v2_final.py"
)
BUILD_CLI_REL = (
    "scripts/research/build_gemma3_chat_five_expert_qonly_unbalanced_v2_final.py"
)
AUDIT_CLI_REL = (
    "scripts/research/audit_gemma3_chat_five_expert_qonly_unbalanced_v2_final.py"
)
TEST_REL = "tests/test_gemma3_chat_five_expert_qonly_unbalanced_v2_final.py"
RELEASE_ATTESTATION_SCHEMA_REL = (
    "configs/research/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_"
    "independent_release_attestation_v1.schema.json"
)
RELEASE_IMPLEMENTATION_REL = (
    "src/anchor_mvp/research/gemma3_chat_five_expert_qonly_unbalanced_v2_release.py"
)
RELEASE_CLI_REL = (
    "scripts/research/review_gemma3_chat_five_expert_qonly_unbalanced_v2_release.py"
)
RELEASE_TEST_REL = (
    "tests/test_review_gemma3_chat_five_expert_qonly_unbalanced_v2_release.py"
)
GIT_ATTRIBUTES_REL = ".gitattributes"

PRODUCER_SOURCE_RELS = (
    CONFIG_REL,
    CONFIG_SCHEMA_REL,
    RECORD_SCHEMA_REL,
    SERIALIZATION_SCHEMA_REL,
    IDENTITY_PROBE_SCHEMA_REL,
    MANIFEST_SCHEMA_REL,
    RECEIPT_SCHEMA_REL,
    IMPLEMENTATION_REL,
    BUILD_CLI_REL,
    AUDIT_CLI_REL,
    TEST_REL,
    RELEASE_ATTESTATION_SCHEMA_REL,
    RELEASE_IMPLEMENTATION_REL,
    RELEASE_CLI_REL,
    RELEASE_TEST_REL,
    GIT_ATTRIBUTES_REL,
)

ROLES = ("humor", "serious", "angry_style", "tool_call", "review_audit")
STYLE_ROLES = ROLES[:3]
SPLITS = ("train", "eval_proxy")
LANGUAGES = ("en", "zh-CN")
FAULT_TYPES = (
    "format_schema",
    "tool_argument",
    "evidence_mismatch",
    "grounding_mismatch",
    "routing_scope",
)
IDENTITY_INTENTS = (
    "direct_self_identity",
    "provenance_fact_check",
    "no_tool_identity_decision",
    "pre_coding_attribution",
    "false_attribution_correction",
)
IDENTITY_CASES = {
    "direct_self_identity": (
        "profile_header",
        "onboarding_notice",
        "support_introduction",
        "benchmark_disclosure",
        "assistant_card",
        "session_opening",
    ),
    "provenance_fact_check": (
        "model_card_check",
        "deployment_note_check",
        "operator_claim_check",
        "audit_log_check",
        "provenance_label_check",
        "handoff_note_check",
    ),
    "no_tool_identity_decision": (
        "search_tool_suggestion",
        "catalog_tool_suggestion",
        "calculator_tool_suggestion",
        "calendar_tool_suggestion",
        "file_tool_suggestion",
        "sandbox_tool_suggestion",
    ),
    "pre_coding_attribution": (
        "bugfix_preamble",
        "refactor_preamble",
        "test_preamble",
        "schema_preamble",
        "review_preamble",
        "migration_preamble",
    ),
    "false_attribution_correction": (
        "google_claim",
        "openai_claim",
        "vendor_misattribution",
        "lab_misattribution",
        "platform_misattribution",
        "documentation_misattribution",
    ),
}

SEMANTIC_DOMAIN = "anchor.gemma3-chat-task-semantic.v2"
BUNDLE_DOMAIN = "anchor.gemma3-chat-task-bundle.v2"
PAIR_DOMAIN = "anchor.task-template-pair.v1"
RECORD_DOMAIN = "anchor.gemma3-chat-unbalanced-v2-final-record.v1"
SEGMENT_DOMAIN = "anchor.gemma3-chat-unbalanced-v2-final-visible-segment.v1"
ORDERED_SEGMENT_DOMAIN = "anchor.ordered-segment-ids.v1"
ALLOWED_SEGMENT_DOMAIN = "anchor.allowed-context-intersection.v1"
PREFIX_LINEAGE_DOMAIN = "anchor.gemma3-chat-final-prefix-lineage.v1"
ROUTE_COMMIT_DOMAIN = "anchor.route-commit.v1"
PLAN_COMMIT_DOMAIN = "anchor.plan-commit.v1"
REVIEW_DEPENDENCY_DOMAIN = "anchor.review-tool-dependency.v2"
PROBE_DOMAIN = "anchor.gemma3-chat-final-identity-probe.v1"
INVENTORY_DOMAIN = "anchor.sorted-sha256-inventory.v1"

RECORD_SCHEMA_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-record.v1"
)
SERIALIZATION_SCHEMA_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-"
    "serialization-inventory.v1"
)
PROBE_SCHEMA_VERSION = (
    "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-identity-probe.v1"
)

MAX_JSONL_LINE_BYTES = 1_048_576
MAX_OUTPUT_FILE_BYTES_EXCLUSIVE = 52_428_800
UPSTREAM_READ_SET_FILES = 64
TRAIN_SHARD_PATHS = (
    "train/chat-00000-of-00002.jsonl",
    "train/chat-00001-of-00002.jsonl",
)
SUPERSEDED_MANIFEST_SHA256 = (
    "b853201d7e433a4fae8cbfeef522bc5ac6fc4be133037e762ce7b850099ff88c"
)
SUPERSEDED_RECEIPT_SHA256 = (
    "f6b7f7d8a3365dda642532f1033db362dca356281b7f47367c04a1bdcab9fcd6"
)
LOGICAL_TRAIN_SHA256 = (
    "62c0f64b6d2f5a0901169480c946d554cc02175138862b3a210d06e44fcbddd3"
)
LOGICAL_TRAIN_BYTES = 59_845_314
LOGICAL_TRAIN_RECORDS = 3_440
FAMILY1_REVIEW_DOMAIN = "anchor.gemma3-chat-family1-v2-independent-review-outcome.v1"
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
)
RAW_TOKEN_KEYS = frozenset(
    {
        "token_ids",
        "input_ids",
        "labels",
        "raw_ids",
        "raw_token_ids",
    }
)


class FinalMaterializationError(RuntimeError):
    """Raised with a stable body-free error code."""


def _fail(code: str) -> None:
    raise FinalMaterializationError(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_text(value: Any) -> str:
    def walk(node: Any) -> None:
        if node is None or isinstance(node, (bool, str, int)):
            return
        if isinstance(node, float):
            _fail("canonical_float_forbidden")
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, dict):
            for key, item in node.items():
                if not isinstance(key, str) or not key.isascii():
                    _fail("canonical_non_ascii_key")
                walk(item)
            return
        _fail("canonical_type_forbidden")

    walk(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return ("".join(_canonical_text(dict(value)) + "\n" for value in values)).encode(
        "utf-8"
    )


def _domain_hash(domain: str, value: Any) -> str:
    if not domain or not domain.isascii():
        _fail("invalid_domain_tag")
    return _sha256(domain.encode("ascii") + b"\x00" + _canonical_text(value).encode())


def _value_hash(value: Any) -> str:
    return _sha256(_canonical_text(value).encode("utf-8"))


def _content_hash(value: str) -> str:
    if not isinstance(value, str):
        _fail("content_not_text")
    return _sha256(value.encode("utf-8"))


def _sidecar(filename: str, raw: bytes) -> bytes:
    if "/" in filename or "\\" in filename or not filename:
        _fail("sidecar_filename_invalid")
    return f"{_sha256(raw)}  {filename}\n".encode("ascii")


def _resources() -> dict[str, int]:
    return {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_body_reads": 0,
        "gold_body_reads": 0,
        "heldout_body_reads": 0,
        "luna_requests": 0,
    }


def _claims(*, include_physical: bool = False) -> dict[str, bool]:
    claims = {
        "candidate": True,
        "final": False,
        "diagnostic_only": True,
        "proxy_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "live_authorized": False,
        "release_authorized": False,
        "quality_validated": False,
        "generalization_validated": False,
        "replaces_v1": False,
        "replaces_10k": False,
        "replaces_luna": False,
    }
    if include_physical:
        claims["physical_kv_materialized"] = False
        claims["physical_zero_copy_verified"] = False
    return claims


def _inventory_hash(values: Iterable[str]) -> str:
    ordered = sorted(values)
    return _domain_hash(INVENTORY_DOMAIN, ordered)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail("json_duplicate_key")
        value[key] = item
    return value


def _reject_constant(_: str) -> None:
    _fail("json_non_finite_number")


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    del label
    if not raw or raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        _fail("json_encoding_or_lf_invalid")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalMaterializationError("json_parse_failed") from exc
    if not isinstance(value, dict):
        _fail("json_root_not_object")
    return value


def _strict_jsonl(raw: bytes, label: str) -> tuple[list[dict[str, Any]], list[bytes]]:
    del label
    if not raw or not raw.endswith(b"\n"):
        _fail("jsonl_terminal_lf_missing")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        _fail("jsonl_encoding_or_lf_invalid")
    lines = raw.splitlines()
    if not lines or any(not line for line in lines):
        _fail("jsonl_empty_line")
    if any(len(line) > MAX_JSONL_LINE_BYTES for line in lines):
        _fail("jsonl_line_too_large")
    return [_strict_json(line, "jsonl_line") for line in lines], lines


def _walk_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            result.add(key)
            result.update(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_walk_keys(item))
    return result


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(getattr(value, "st_file_attributes", 0)),
    )


def _is_reparse(value: os.stat_result) -> bool:
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(int(getattr(value, "st_file_attributes", 0)) & marker)


def _assert_regular_physical(path: Path) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise FinalMaterializationError("input_lstat_failed") from exc
    if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
        _fail("input_reparse_or_symlink")
    if not stat.S_ISREG(value.st_mode):
        _fail("input_not_regular_file")
    return value


@dataclass(frozen=True)
class Snapshot:
    label: str
    path: Path
    raw: bytes
    identity: tuple[int, ...]
    source_class: str

    @property
    def sha256(self) -> str:
        return _sha256(self.raw)

    @property
    def bytes(self) -> int:
        return len(self.raw)


def _snapshot(path: Path, *, label: str, source_class: str) -> Snapshot:
    _assert_regular_physical(path)
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if stat.S_ISLNK(before.st_mode) or _is_reparse(before):
                _fail("input_opened_reparse_or_symlink")
            raw = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise FinalMaterializationError("input_snapshot_failed") from exc
    if _stat_identity(before) != _stat_identity(after):
        _fail("input_changed_during_snapshot")
    if int(after.st_size) != len(raw):
        _fail("input_snapshot_size_mismatch")
    return Snapshot(
        label=label,
        path=path.resolve(),
        raw=raw,
        identity=_stat_identity(after),
        source_class=source_class,
    )


def _reverify_snapshot(value: Snapshot) -> None:
    current = _snapshot(
        value.path,
        label=value.label,
        source_class=value.source_class,
    )
    if current.identity != value.identity or current.raw != value.raw:
        _fail("input_terminal_snapshot_mismatch")


def _safe_repo_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail("repo_path_invalid")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        _fail("repo_path_escape")
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FinalMaterializationError("repo_path_escape") from exc
    return resolved


def _require_explicit_lf_paths(root: Path, relatives: Iterable[str]) -> None:
    ordered = sorted(set(relatives))
    if not ordered:
        _fail("explicit_lf_inventory_empty")
    for relative in ordered:
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
            _fail("explicit_lf_path_invalid")
    try:
        result = subprocess.run(
            ["git", "check-attr", "-z", "text", "eol", "--", *ordered],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FinalMaterializationError("explicit_lf_check_failed") from exc
    if result.returncode != 0 or result.stderr:
        _fail("explicit_lf_check_failed")
    fields = result.stdout.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    if len(fields) % 3:
        _fail("explicit_lf_check_output_invalid")
    observed: dict[str, dict[str, str]] = defaultdict(dict)
    try:
        for index in range(0, len(fields), 3):
            relative = fields[index].decode("utf-8")
            attribute = fields[index + 1].decode("ascii")
            value = fields[index + 2].decode("ascii")
            observed[relative][attribute] = value
    except UnicodeDecodeError as exc:
        raise FinalMaterializationError("explicit_lf_check_output_invalid") from exc
    if set(observed) != set(ordered) or any(
        observed[relative] != {"text": "set", "eol": "lf"} for relative in ordered
    ):
        _fail("explicit_lf_attribute_missing")


@dataclass
class SourceState:
    root: Path
    config: dict[str, Any]
    snapshots: dict[str, Snapshot]
    family_rows: list[dict[str, Any]]
    family_manifests: dict[int, dict[str, Any]]
    component_rows: dict[str, list[dict[str, Any]]]
    component_manifests: dict[str, dict[str, Any]]
    external_reviews: dict[str, dict[str, Any]]
    validators: dict[str, Draft202012Validator]
    upstream_labels: set[str]
    producer_labels: set[str]


def _schema_validator(raw: bytes, label: str) -> Draft202012Validator:
    schema = _strict_json(raw, label)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise FinalMaterializationError("schema_invalid") from exc
    return Draft202012Validator(schema)


def _validate_schema(validator: Draft202012Validator, value: Any, code: str) -> None:
    try:
        validator.validate(value)
    except ValidationError as exc:
        raise FinalMaterializationError(code) from exc


def _validate_frozen_contracts(
    state: SourceState, frozen_by_path: Mapping[str, Snapshot]
) -> None:
    proposal_rel = (
        "configs/research/gemma3_chat_five_expert_qonly_unbalanced_v2_proposal.json"
    )
    proposal_schema_rel = proposal_rel.replace(".json", ".schema.json")
    old_record_schema_rel = (
        "configs/research/"
        "gemma3_chat_five_expert_qonly_unbalanced_v2_record.schema.json"
    )
    amendment_rel = (
        "configs/research/"
        "gemma3_chat_five_expert_qonly_unbalanced_v2_"
        "identity_ablation_shared_prefix_amendment_v1.json"
    )
    amendment_schema_rel = amendment_rel.replace(".json", ".schema.json")
    amendment_manifest_rel = (
        "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_v2/"
        "amendments/identity_ablation_shared_prefix_v1_candidate/manifest.json"
    )
    amendment_sidecar_rel = amendment_manifest_rel + ".sha256"

    proposal = _strict_json(frozen_by_path[proposal_rel].raw, "proposal")
    proposal_schema = _schema_validator(
        frozen_by_path[proposal_schema_rel].raw, "proposal_schema"
    )
    _validate_schema(proposal_schema, proposal, "proposal_schema_rejected")
    if (
        proposal.get("paths", {}).get("generation_artifact")
        != LEGACY_UNSHARDED_ARTIFACT_REL
        or CANONICAL_ARTIFACT_REL == SUPERSEDED_ARTIFACT_REL
    ):
        _fail("superseded_canonical_path_binding_mismatch")
    generation = proposal.get("generation_asset", {})
    if (
        generation.get("records") != 4300
        or generation.get("task_bundles") != 1700
        or generation.get("roles") != list(ROLES)
    ):
        _fail("proposal_training_contract_drift")

    old_record_schema = _strict_json(
        frozen_by_path[old_record_schema_rel].raw, "old_record_schema"
    )
    try:
        Draft202012Validator.check_schema(old_record_schema)
    except SchemaError as exc:
        raise FinalMaterializationError("old_record_schema_invalid") from exc

    amendment = _strict_json(frozen_by_path[amendment_rel].raw, "amendment")
    amendment_schema = _schema_validator(
        frozen_by_path[amendment_schema_rel].raw, "amendment_schema"
    )
    _validate_schema(amendment_schema, amendment, "amendment_schema_rejected")
    identity = amendment.get("identity_alignment", {})
    expected_identity = state.config["identity_contract"]
    if (
        identity.get("stable_identity_fact")
        != expected_identity["stable_identity_fact"]
        or identity.get("identity_bundles") != 30
        or identity.get("train_bundles") != 20
        or identity.get("eval_proxy_bundles") != 10
        or identity.get("eval_identity_probe_records") != 50
        or identity.get("tool_call_behavior") != "direct_answer_no_tool"
        or not identity.get("explicit_wrong_route_identity_fact_must_remain_equal")
    ):
        _fail("amendment_identity_contract_drift")
    kv = amendment.get("shared_route_prefix_kv", {})
    if (
        kv.get("producer_mode")
        != "frozen_base_adapter_off_reencode_committed_route_and_plan"
        or kv.get("full_generation_kv_shared") is not False
        or kv.get("planner_live_private_kv_transfer") is not False
        or kv.get("current_physical_claim") != "prefill_compute_reuse"
        or kv.get("persistent_prefix_storage_claim_allowed") is not False
        or "q8_kv" not in kv.get("lossy_kv_diagnostic_only", [])
    ):
        _fail("amendment_kv_contract_drift")

    amendment_manifest_raw = frozen_by_path[amendment_manifest_rel].raw
    amendment_sidecar_raw = frozen_by_path[amendment_sidecar_rel].raw
    if amendment_sidecar_raw != _sidecar("manifest.json", amendment_manifest_raw):
        _fail("amendment_manifest_sidecar_mismatch")
    amendment_manifest = _strict_json(amendment_manifest_raw, "amendment_manifest")
    if (
        amendment_manifest.get("status") != "candidate_pending_independent_audit"
        or amendment_manifest.get("counts", {}).get("identity_bundles") != 30
        or amendment_manifest.get("counts", {}).get("records_materialized") != 0
    ):
        _fail("amendment_manifest_contract_drift")


def _validate_family_rows(
    family: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    lines: Sequence[bytes],
) -> list[dict[str, Any]]:
    if len(rows) != 170 or len(lines) != 170:
        _fail("family_row_count_mismatch")
    family_index = int(family["family_index"])
    family_id = str(family["training_family_id"])
    if manifest.get("partition", {}).get("blueprints") != 170:
        _fail("family_manifest_partition_count_mismatch")
    if manifest.get("counts", {}).get("core_bundles") != 30:
        _fail("family_core_count_mismatch")
    if manifest.get("counts", {}).get("depth_bundles") != 140:
        _fail("family_depth_count_mismatch")
    if manifest.get("counts", {}).get("records_materialized") != 0:
        _fail("family_already_materialized")
    claims = manifest.get("claims", {})
    if (
        claims.get("final") is not False
        or claims.get("training_authorized") is not False
        or claims.get("formal_training_authorized") is not False
    ):
        _fail("family_claim_authority_drift")
    if family_index == 1:
        mapping = manifest.get("track_mapping", {})
        identity = manifest.get("identity_preservation", {})
        source_candidate = manifest.get("upstream", {}).get("source_candidate", {})
        if (
            manifest.get("status") != "candidate_ready_for_different_reviewer"
            or manifest.get("repair_id")
            != (
                "anchor.gemma3-chat-unbalanced-v2.semantic-blueprints."
                "local-catalog-search.01.bundle-class-metadata-repair.v2"
            )
            or mapping.get("pairs_sha256") != family.get("mapping_sha256")
            or mapping.get("mapping_kind") != "body_free_bundle_class_pair_overlay"
            or mapping.get("pair_count") != 15
            or mapping.get("changed_record_count") != 30
            or identity.get("rows_with_track_change") != 30
            or identity.get("prompt_tool_result_answer_rewrites") != 0
            or identity.get("row_additions") != 0
            or identity.get("row_deletions") != 0
            or identity.get("language_unchanged") is not True
            or identity.get("split_unchanged") is not True
            or identity.get("semantic_descriptor_unchanged") is not True
            or identity.get("task_bundle_identity_unchanged") is not True
            or source_candidate.get("must_not_be_consumed_after_v2_review") is not True
            or not str(source_candidate.get("path", "")).endswith(
                "/01_local_catalog_search_candidate"
            )
            or claims.get("metadata_only_repair") is not True
            or claims.get("different_reviewer_required") is not True
            or claims.get("independent_audit_passed") is not False
        ):
            _fail("family_one_v2_repair_contract_mismatch")

    enriched: list[dict[str, Any]] = []
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    semantic_ids: set[str] = set()
    bundle_ids: set[str] = set()
    template_ids: set[str] = set()
    for row, line in zip(rows, lines, strict=True):
        descriptor = row.get("semantic_descriptor")
        if not isinstance(descriptor, dict):
            _fail("family_descriptor_missing")
        if descriptor.get("tool_family") != family_id:
            _fail("family_descriptor_tool_family_mismatch")
        semantic_sha = _domain_hash(SEMANTIC_DOMAIN, descriptor)
        bundle_sha = _domain_hash(BUNDLE_DOMAIN, descriptor)
        if row.get("task_semantic_sha256") != semantic_sha:
            _fail("family_semantic_identity_mismatch")
        if row.get("task_bundle_sha256") != bundle_sha:
            _fail("family_bundle_identity_mismatch")
        template_sha = row.get("template_family_sha256")
        if not isinstance(template_sha, str) or not re.fullmatch(
            r"[0-9a-f]{64}", template_sha
        ):
            _fail("family_template_identity_invalid")
        expected_pair = _domain_hash(
            PAIR_DOMAIN,
            {
                "task_semantic_sha256": semantic_sha,
                "template_family_sha256": template_sha,
            },
        )
        if row.get("task_template_pair_sha256") != expected_pair:
            _fail("family_pair_identity_mismatch")
        split = row.get("split")
        language = row.get("language")
        bundle_class = row.get("bundle_class")
        if (
            split not in SPLITS
            or language not in LANGUAGES
            or bundle_class not in ("full_five_role_core", "tool_review_depth")
        ):
            _fail("family_partition_cell_invalid")
        if row.get("claims", {}).get("records_materialized") is not False:
            _fail("family_row_materialization_claim_drift")
        if RAW_TOKEN_KEYS & _walk_keys(row):
            _fail("family_raw_token_key_present")
        cell_counts[(language, split, bundle_class)] += 1
        if semantic_sha in semantic_ids or bundle_sha in bundle_ids:
            _fail("family_duplicate_identity")
        semantic_ids.add(semantic_sha)
        bundle_ids.add(bundle_sha)
        template_ids.add(template_sha)
        item = copy.deepcopy(row)
        item["_family_index"] = family_index
        item["_family_id"] = family_id
        item["_line_sha256"] = _sha256(line)
        item["_source_manifest_sha256"] = str(family["manifest_sha256"])
        item["_source_partition_sha256"] = str(family["partition_sha256"])
        item["_source_sidecar_sha256"] = str(family["sidecar_physical_sha256"])
        enriched.append(item)

    expected_cells = {
        ("en", "train", "full_five_role_core"): 12,
        ("zh-CN", "train", "full_five_role_core"): 12,
        ("en", "eval_proxy", "full_five_role_core"): 3,
        ("zh-CN", "eval_proxy", "full_five_role_core"): 3,
        ("en", "train", "tool_review_depth"): 56,
        ("zh-CN", "train", "tool_review_depth"): 56,
        ("en", "eval_proxy", "tool_review_depth"): 14,
        ("zh-CN", "eval_proxy", "tool_review_depth"): 14,
    }
    if dict(cell_counts) != expected_cells or len(template_ids) != 10:
        _fail("family_exact_quota_mismatch")
    return enriched


def _validate_component_candidates(
    config: Mapping[str, Any],
    component_rows: Mapping[str, list[dict[str, Any]]],
    component_manifests: Mapping[str, dict[str, Any]],
    external_reviews: Mapping[str, dict[str, Any]],
) -> None:
    router_rows = component_rows["router"]
    router_manifest = component_manifests["router"]
    if (
        len(router_rows) != 100
        or router_manifest.get("dataset_summary", {}).get("records") != 100
        or router_manifest.get("dataset_summary", {}).get("train_records") != 80
        or router_manifest.get("dataset_summary", {}).get("eval_proxy_records") != 20
        or router_manifest.get("dataset_summary", {}).get("en_records") != 50
        or router_manifest.get("dataset_summary", {}).get("zh_cn_records") != 50
    ):
        _fail("router_count_contract_mismatch")
    if Counter(row.get("target_expert_id") for row in router_rows) != {
        "humor": 34,
        "serious": 33,
        "angry_style": 33,
    }:
        _fail("router_label_quota_mismatch")
    if Counter(row.get("split") for row in router_rows) != {
        "train": 80,
        "eval_proxy": 20,
    }:
        _fail("router_split_quota_mismatch")
    if Counter(row.get("language") for row in router_rows) != {
        "en": 50,
        "zh-CN": 50,
    }:
        _fail("router_language_quota_mismatch")
    if len({row.get("task_semantic_sha256") for row in router_rows}) != 100:
        _fail("router_semantic_uniqueness_mismatch")
    for row in router_rows:
        topology = row.get("route_topology", {})
        if (
            topology.get("router_decisions") != 1
            or topology.get("router_exits_after_selection") is not True
            or topology.get("return_to_router") is not False
            or topology.get("router_vote") is not False
            or topology.get("router_aggregate") is not False
            or row.get("target_expert_id") not in STYLE_ROLES
        ):
            _fail("router_topology_mismatch")
        if RAW_TOKEN_KEYS & _walk_keys(row):
            _fail("router_raw_token_key_present")

    tool_rows = component_rows["tool_eval"]
    tool_manifest = component_manifests["tool_eval"]
    dataset = tool_manifest.get("dataset", {})
    overlay = tool_manifest.get("execution_overlay", {})
    if (
        len(tool_rows) != 400
        or dataset.get("records") != 400
        or dataset.get("arm_neutral") is not True
        or dataset.get("en_records") != 200
        or dataset.get("zh_cn_records") != 200
        or overlay.get("execution_slots") != 1200
        or overlay.get("records_added_by_arms") != 0
        or overlay.get("arms") != ["base", "tool_q_only", "tool_q_plus_micro_o"]
    ):
        _fail("tool_eval_contract_mismatch")
    if Counter(row.get("stratum") for row in tool_rows) != {
        "seen_template_new_parameters": 100,
        "known_operation_new_template": 100,
        "compositional_or_new_task_relation": 100,
        "distractor_or_conflicting_evidence": 100,
    }:
        _fail("tool_eval_stratum_mismatch")
    if len({row.get("arm_neutral_task_bytes_sha256") for row in tool_rows}) != 400:
        _fail("tool_eval_arm_neutral_identity_mismatch")
    if any(
        row.get("execution_overlay", {}).get("records_added_by_arms") != 0
        or row.get("claims", {}).get("arm_neutral") is not True
        for row in tool_rows
    ):
        _fail("tool_eval_row_arm_neutrality_mismatch")
    q = overlay.get("q_branch", {})
    o = overlay.get("micro_o_branch", {})
    if (
        (q.get("rank"), q.get("alpha"), q.get("trainable_params"))
        != (1024, 2048, 57933824)
        or (o.get("rank"), o.get("alpha"), o.get("trainable_params"))
        != (64, 128, 3620864)
        or o.get("max_lr_ratio_to_q") != "1/10"
        or overlay.get("lr_ratio_checked_every_step") is not True
    ):
        _fail("tool_eval_adapter_contract_mismatch")

    planner_rows = component_rows["planner_eval"]
    planner_manifest = component_manifests["planner_eval"]
    dataset = planner_manifest.get("dataset", {})
    overlay = planner_manifest.get("execution_overlay", {})
    oracle = planner_manifest.get("planner_oracle", {})
    if (
        len(planner_rows) != 240
        or dataset.get("records") != 240
        or dataset.get("arm_neutral") is not True
        or dataset.get("en_records") != 120
        or dataset.get("zh_cn_records") != 120
        or overlay.get("execution_slots") != 960
        or overlay.get("records_added_by_arms") != 0
        or overlay.get("arms")
        != [
            "base",
            "planner_q_only",
            "planner_o_only",
            "planner_q_plus_micro_o",
        ]
    ):
        _fail("planner_eval_contract_mismatch")
    if Counter(row.get("route_label") for row in planner_rows) != {
        "humor": 80,
        "serious": 80,
        "angry_style": 80,
    }:
        _fail("planner_eval_route_quota_mismatch")
    if Counter(row.get("stratum") for row in planner_rows) != {
        "exact_dead_command_schema": 60,
        "paraphrased_same_variables": 60,
        "multivariable_dependency_plan": 60,
        "conflicting_cues_or_distractors": 60,
    }:
        _fail("planner_eval_stratum_mismatch")
    if (
        oracle.get("independent_rule_oracle_passed") != 240
        or oracle.get("acyclic_plans") != 240
        or oracle.get("single_route_records") != 240
        or oracle.get("tool_review_route_labels") != 0
    ):
        _fail("planner_eval_oracle_mismatch")
    if len({row.get("arm_neutral_task_bytes_sha256") for row in planner_rows}) != 240:
        _fail("planner_eval_arm_neutral_identity_mismatch")
    q = overlay.get("q_branch", {})
    o = overlay.get("micro_o_branch", {})
    if (
        (q.get("rank"), q.get("alpha"), q.get("trainable_params"))
        != (1024, 2048, 57933824)
        or (o.get("rank"), o.get("alpha"), o.get("trainable_params"))
        != (64, 128, 3620864)
        or overlay.get("o_lr_to_q_max") != "1/10"
        or overlay.get("o_lr_ratio_checked_each_step") is not True
        or overlay.get("o_only_q_plus_o_same_o_identity") is not True
        or overlay.get("o_only_o_identity_equals_q_plus_o_o_identity") is not True
        or overlay.get("q_only_q_plus_o_same_q_init") is not True
    ):
        _fail("planner_eval_adapter_identity_mismatch")
    for row in planner_rows:
        row_overlay = row.get("execution_overlay", {})
        if (
            row_overlay.get("records_added_by_arms") != 0
            or row.get("claims", {}).get("arm_neutral") is not True
            or row_overlay.get("o_only_q_plus_o_same_o_identity") is not True
            or row_overlay.get("q_only_q_plus_o_same_q_init") is not True
        ):
            _fail("planner_eval_row_contract_mismatch")

    for component in ("tool_eval", "planner_eval"):
        review = external_reviews[component]
        expected = config["external_reviews"][component]
        if (
            review.get("audit_status") != "passed"
            or review.get("candidate_before_after_equal") is not True
            or review.get("reviewed_candidate_manifest_sha256")
            != expected["reviewed_manifest_sha256"]
        ):
            _fail("external_review_not_passed")


def _load_sources(
    root: Path,
    *,
    tool_review_path: Path,
    planner_review_path: Path,
) -> SourceState:
    snapshots: dict[str, Snapshot] = {}
    path_registry: dict[Path, str] = {}
    upstream_labels: set[str] = set()
    producer_labels: set[str] = set()

    def add(
        label: str,
        path: Path,
        source_class: str,
        *,
        expected_sha256: str | None = None,
        upstream: bool = False,
        producer: bool = False,
    ) -> Snapshot:
        resolved = path.resolve()
        if label in snapshots or resolved in path_registry:
            _fail("source_snapshot_duplicate")
        value = _snapshot(
            resolved,
            label=label,
            source_class=source_class,
        )
        if expected_sha256 is not None and value.sha256 != expected_sha256:
            _fail("source_expected_sha256_mismatch")
        snapshots[label] = value
        path_registry[resolved] = label
        if upstream:
            upstream_labels.add(label)
        if producer:
            producer_labels.add(label)
        return value

    config_snapshot = add(
        "producer:config",
        _safe_repo_path(root, CONFIG_REL),
        "producer_source",
        producer=True,
    )
    config_schema_snapshot = add(
        "producer:config_schema",
        _safe_repo_path(root, CONFIG_SCHEMA_REL),
        "producer_source",
        producer=True,
    )
    config = _strict_json(config_snapshot.raw, "config")
    config_validator = _schema_validator(config_schema_snapshot.raw, "config_schema")
    _validate_schema(config_validator, config, "config_schema_rejected")
    if (
        config.get("namespace") != NAMESPACE
        or config.get("artifact_version") != ARTIFACT_VERSION
        or config.get("canonical_output_path") != CANONICAL_ARTIFACT_REL
    ):
        _fail("final_config_namespace_or_path_mismatch")
    superseded = config["superseded_candidate"]
    sharding = config["sharding_contract"]
    if (
        superseded["canonical_path"] != SUPERSEDED_ARTIFACT_REL
        or superseded["manifest_sha256"] != SUPERSEDED_MANIFEST_SHA256
        or superseded["build_receipt_sha256"] != SUPERSEDED_RECEIPT_SHA256
        or superseded["logical_train_sha256"] != LOGICAL_TRAIN_SHA256
        or superseded["logical_train_bytes"] != LOGICAL_TRAIN_BYTES
        or superseded["logical_train_records"] != LOGICAL_TRAIN_RECORDS
        or superseded["logical_train_ordered_paths"] != list(TRAIN_SHARD_PATHS)
        or superseded["superseded_reason"]
        != "ruff_format_gate_and_atomic_attestation_pair_publish_incomplete"
        or superseded["old_candidate_consumption_allowed"] is not False
        or sharding["ordered_paths"] != list(TRAIN_SHARD_PATHS)
        or sharding["max_file_bytes_exclusive"] != MAX_OUTPUT_FILE_BYTES_EXCLUSIVE
        or sharding["logical_partition_sha256"] != LOGICAL_TRAIN_SHA256
        or sharding["logical_partition_bytes"] != LOGICAL_TRAIN_BYTES
        or sharding["logical_partition_records"] != LOGICAL_TRAIN_RECORDS
    ):
        _fail("final_sharding_or_superseded_contract_mismatch")
    families = config["work_order_families"]
    if [family["family_index"] for family in families] != list(range(1, 11)):
        _fail("final_config_family_order_mismatch")
    family_one = families[0]
    expected_family_one_directory = (
        "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_v2/"
        "work_orders/01_local_catalog_search_candidate_v2"
    )
    if (
        family_one["training_family_id"] != "local_catalog_search"
        or family_one["directory"] != expected_family_one_directory
        or family_one.get("mapping_sha256")
        != "a07fb205d027e60902bd8ebfd103de3485d82932bfce721165f0b088b14571b5"
        or family_one.get("old_candidate_selected") is not False
        or any(
            family["directory"].endswith("/01_local_catalog_search_candidate")
            for family in families
        )
    ):
        _fail("family_one_v2_selection_binding_mismatch")
    if any(
        ("mapping_sha256" in family or "old_candidate_selected" in family)
        for family in families[1:]
    ):
        _fail("family_one_repair_fields_leaked_to_other_family")
    family_one_review = config["family1_repair_independent_review"]
    review_preimage = {
        key: copy.deepcopy(value)
        for key, value in family_one_review.items()
        if key not in {"outcome_binding_domain", "outcome_binding_sha256"}
    }
    if (
        family_one_review["outcome_binding_domain"] != FAMILY1_REVIEW_DOMAIN
        or family_one_review["outcome_binding_sha256"]
        != _domain_hash(FAMILY1_REVIEW_DOMAIN, review_preimage)
        or family_one_review["reviewed_directory"] != family_one["directory"]
        or family_one_review["reviewed_manifest_sha256"]
        != family_one["manifest_sha256"]
        or family_one_review["reviewed_partition_sha256"]
        != family_one["partition_sha256"]
        or family_one_review["reviewed_mapping_sha256"] != family_one["mapping_sha256"]
        or family_one_review["reviewed_sidecar_physical_sha256"]
        != family_one["sidecar_physical_sha256"]
        or family_one_review["status"] != "passed"
        or family_one_review["reviewer_separation"] != "different_reviewer"
        or any(
            family_one_review[key] != 0
            for key in ("p0_findings", "p1_findings", "p2_findings")
        )
        or family_one_review["old_candidate_selected"] is not False
    ):
        _fail("family_one_independent_review_binding_mismatch")

    schema_labels = {
        RECORD_SCHEMA_REL: "record",
        SERIALIZATION_SCHEMA_REL: "serialization",
        IDENTITY_PROBE_SCHEMA_REL: "probe",
        MANIFEST_SCHEMA_REL: "manifest",
        RECEIPT_SCHEMA_REL: "receipt",
    }
    for relative in PRODUCER_SOURCE_RELS:
        if relative in (CONFIG_REL, CONFIG_SCHEMA_REL):
            continue
        add(
            f"producer:{relative}",
            _safe_repo_path(root, relative),
            "producer_source",
            producer=True,
        )
    validators = {"config": config_validator}
    for relative, name in schema_labels.items():
        validators[name] = _schema_validator(
            snapshots[f"producer:{relative}"].raw,
            f"{name}_schema",
        )

    superseded_root = _safe_repo_path(root, SUPERSEDED_ARTIFACT_REL)
    superseded_manifest = add(
        "superseded:manifest",
        superseded_root / "manifest.json",
        "superseded_candidate",
        expected_sha256=superseded["manifest_sha256"],
        upstream=True,
    )
    superseded_manifest_sidecar = add(
        "superseded:manifest_sidecar",
        superseded_root / "manifest.json.sha256",
        "superseded_candidate",
        expected_sha256=superseded["manifest_sidecar_physical_sha256"],
        upstream=True,
    )
    superseded_receipt = add(
        "superseded:receipt",
        superseded_root / "build_receipt.json",
        "superseded_candidate",
        expected_sha256=superseded["build_receipt_sha256"],
        upstream=True,
    )
    superseded_receipt_sidecar = add(
        "superseded:receipt_sidecar",
        superseded_root / "build_receipt.json.sha256",
        "superseded_candidate",
        expected_sha256=superseded["build_receipt_sidecar_physical_sha256"],
        upstream=True,
    )
    if superseded_manifest_sidecar.raw != _sidecar(
        "manifest.json", superseded_manifest.raw
    ) or superseded_receipt_sidecar.raw != _sidecar(
        "build_receipt.json", superseded_receipt.raw
    ):
        _fail("superseded_candidate_sidecar_mismatch")
    superseded_manifest_value = _strict_json(
        superseded_manifest.raw,
        "superseded_manifest",
    )
    superseded_receipt_value = _strict_json(
        superseded_receipt.raw,
        "superseded_receipt",
    )
    superseded_files = {
        item.get("path"): item
        for item in superseded_manifest_value.get("files", [])
        if isinstance(item, Mapping)
    }
    superseded_shards = superseded_manifest_value.get("training_shards", {})
    old_train_bindings = [superseded_files.get(path) for path in TRAIN_SHARD_PATHS]
    if (
        superseded_manifest_value.get("status") != STATUS
        or superseded_manifest_value.get("canonical_path") != SUPERSEDED_ARTIFACT_REL
        or superseded_shards.get("ordered_paths") != list(TRAIN_SHARD_PATHS)
        or superseded_shards.get("ordered_concat_sha256") != LOGICAL_TRAIN_SHA256
        or superseded_shards.get("logical_partition_bytes") != LOGICAL_TRAIN_BYTES
        or superseded_shards.get("logical_partition_records") != LOGICAL_TRAIN_RECORDS
        or not all(isinstance(item, Mapping) for item in old_train_bindings)
        or sum(int(item["bytes"]) for item in old_train_bindings) != LOGICAL_TRAIN_BYTES
        or sum(int(item["records"]) for item in old_train_bindings)
        != LOGICAL_TRAIN_RECORDS
        or superseded_receipt_value.get("manifest", {}).get("sha256")
        != SUPERSEDED_MANIFEST_SHA256
    ):
        _fail("superseded_candidate_identity_mismatch")

    frozen_by_path: dict[str, Snapshot] = {}
    for pin in config["frozen_contracts"]:
        relative = pin["path"]
        value = add(
            f"frozen:{relative}",
            _safe_repo_path(root, relative),
            "frozen_contract",
            expected_sha256=pin["sha256"],
        )
        frozen_by_path[relative] = value

    family_rows: list[dict[str, Any]] = []
    family_manifests: dict[int, dict[str, Any]] = {}
    for family in config["work_order_families"]:
        family_index = int(family["family_index"])
        base = _safe_repo_path(root, family["directory"])
        manifest_snapshot = add(
            f"family:{family_index}:manifest",
            base / "manifest.json",
            "work_order_manifest",
            expected_sha256=family["manifest_sha256"],
            upstream=True,
        )
        partition_snapshot = add(
            f"family:{family_index}:partition",
            base / "semantic_blueprints.jsonl",
            "work_order_partition",
            expected_sha256=family["partition_sha256"],
            upstream=True,
        )
        sidecar_snapshot = add(
            f"family:{family_index}:sidecar",
            base / "manifest.json.sha256",
            "work_order_sidecar",
            expected_sha256=family["sidecar_physical_sha256"],
            upstream=True,
        )
        if sidecar_snapshot.raw != _sidecar("manifest.json", manifest_snapshot.raw):
            _fail("family_manifest_sidecar_mismatch")
        manifest = _strict_json(manifest_snapshot.raw, "family_manifest")
        if (
            manifest.get("partition", {}).get("sha256") != partition_snapshot.sha256
            or manifest.get("partition", {}).get("bytes") != partition_snapshot.bytes
        ):
            _fail("family_manifest_partition_binding_mismatch")
        raw_rows, lines = _strict_jsonl(partition_snapshot.raw, "family_partition")
        rows = _validate_family_rows(family, manifest, raw_rows, lines)
        family_manifests[family_index] = manifest
        family_rows.extend(rows)

    component_rows: dict[str, list[dict[str, Any]]] = {}
    component_manifests: dict[str, dict[str, Any]] = {}
    for component_name in ("router", "tool_eval", "planner_eval"):
        component = config["component_candidates"][component_name]
        source_dir = _safe_repo_path(root, component["source_directory"])
        payload_snapshots: dict[str, Snapshot] = {}
        for pin in component["files"]:
            filename = pin["path"]
            payload = add(
                f"component:{component_name}:payload:{filename}",
                source_dir / filename,
                "component_payload",
                expected_sha256=pin["sha256"],
                upstream=True,
            )
            payload_snapshots[filename] = payload
            expected_sidecar = pin["sidecar_physical_sha256"]
            if expected_sidecar is not None:
                sidecar = add(
                    f"component:{component_name}:sidecar:{filename}",
                    source_dir / f"{filename}.sha256",
                    "component_sidecar",
                    expected_sha256=expected_sidecar,
                    upstream=True,
                )
                if sidecar.raw != _sidecar(filename, payload.raw):
                    _fail("component_sidecar_content_mismatch")
        manifest = _strict_json(
            payload_snapshots["manifest.json"].raw, "component_manifest"
        )
        records, _ = _strict_jsonl(
            payload_snapshots["records.jsonl"].raw, "component_records"
        )
        tokens, _ = _strict_jsonl(
            payload_snapshots["token_inventory.jsonl"].raw,
            "component_token_inventory",
        )
        if len(tokens) != len(records):
            _fail("component_token_count_mismatch")
        if any(RAW_TOKEN_KEYS & _walk_keys(row) for row in tokens):
            _fail("component_token_inventory_raw_ids")
        component_rows[component_name] = records
        component_manifests[component_name] = manifest

    external_reviews: dict[str, dict[str, Any]] = {}
    for component_name, path in (
        ("tool_eval", tool_review_path),
        ("planner_eval", planner_review_path),
    ):
        expected = config["external_reviews"][component_name]
        canonical_path = _safe_repo_path(root, expected["canonical_path"])
        if (
            expected["canonical_path"] != CANONICAL_REVIEW_RELS[component_name]
            or path.resolve() != canonical_path
        ):
            _fail("canonical_review_path_mismatch")
        review_snapshot = add(
            f"canonical_review:{component_name}:payload",
            canonical_path,
            "canonical_review",
            expected_sha256=expected["sha256"],
            upstream=True,
        )
        review_sidecar = add(
            f"canonical_review:{component_name}:sidecar",
            Path(str(canonical_path) + ".sha256"),
            "canonical_review",
            expected_sha256=expected["sidecar_physical_sha256"],
            upstream=True,
        )
        if review_sidecar.raw != _sidecar(canonical_path.name, review_snapshot.raw):
            _fail("canonical_review_sidecar_mismatch")
        external_reviews[component_name] = _strict_json(
            review_snapshot.raw, "canonical_review"
        )

    if len(upstream_labels) != UPSTREAM_READ_SET_FILES:
        _fail("upstream_read_set_count_mismatch")
    try:
        input_relatives = [
            value.path.relative_to(root).as_posix() for value in snapshots.values()
        ]
    except ValueError as exc:
        raise FinalMaterializationError("input_not_repo_canonical") from exc
    _require_explicit_lf_paths(root, input_relatives)
    state = SourceState(
        root=root,
        config=config,
        snapshots=snapshots,
        family_rows=family_rows,
        family_manifests=family_manifests,
        component_rows=component_rows,
        component_manifests=component_manifests,
        external_reviews=external_reviews,
        validators=validators,
        upstream_labels=upstream_labels,
        producer_labels=producer_labels,
    )
    _validate_frozen_contracts(state, frozen_by_path)
    _validate_component_candidates(
        config,
        component_rows,
        component_manifests,
        external_reviews,
    )
    if len(family_rows) != 1700:
        _fail("all_family_row_count_mismatch")
    if len({row["task_semantic_sha256"] for row in family_rows}) != 1700:
        _fail("cross_family_semantic_overlap")
    if len({row["task_bundle_sha256"] for row in family_rows}) != 1700:
        _fail("cross_family_bundle_overlap")
    return state


def _identity_selection(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, str]]:
    selected: dict[str, tuple[str, str]] = {}
    for family_index in range(1, 11):
        family_rows = [
            row
            for row in rows
            if row["_family_index"] == family_index
            and row["bundle_class"] == "full_five_role_core"
        ]
        by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for split in SPLITS:
            for language in LANGUAGES:
                by_cell[(split, language)] = sorted(
                    [
                        row
                        for row in family_rows
                        if row["split"] == split and row["language"] == language
                    ],
                    key=lambda item: item["task_bundle_sha256"],
                )
        eval_language = "en" if family_index <= 5 else "zh-CN"
        slots = (
            by_cell[("train", "en")][0],
            by_cell[("train", "zh-CN")][0],
            by_cell[("eval_proxy", eval_language)][0],
        )
        intent = IDENTITY_INTENTS[(family_index - 1) % 5]
        occurrence = 0 if family_index <= 5 else 1
        for local_slot, row in enumerate(slots):
            case_index = occurrence * 3 + local_slot
            selected[row["task_bundle_sha256"]] = (
                intent,
                IDENTITY_CASES[intent][case_index],
            )

    selected_rows = [row for row in rows if row["task_bundle_sha256"] in selected]
    if len(selected) != 30:
        _fail("identity_selection_count_mismatch")
    if Counter(row["split"] for row in selected_rows) != {
        "train": 20,
        "eval_proxy": 10,
    }:
        _fail("identity_selection_split_mismatch")
    if Counter(row["language"] for row in selected_rows) != {
        "en": 15,
        "zh-CN": 15,
    }:
        _fail("identity_selection_language_mismatch")
    if Counter(value[0] for value in selected.values()) != {
        intent: 6 for intent in IDENTITY_INTENTS
    }:
        _fail("identity_selection_intent_mismatch")
    for intent in IDENTITY_INTENTS:
        intent_rows = [
            row
            for row in selected_rows
            if selected[row["task_bundle_sha256"]][0] == intent
        ]
        if Counter(row["split"] for row in intent_rows) != {
            "train": 4,
            "eval_proxy": 2,
        }:
            _fail("identity_intent_split_mismatch")
        if Counter(row["language"] for row in intent_rows) != {
            "en": 3,
            "zh-CN": 3,
        }:
            _fail("identity_intent_language_mismatch")
    return selected


def _identity_descriptor(
    row: Mapping[str, Any], intent: str, case_context: str
) -> dict[str, Any]:
    false_claim = None
    if intent in ("provenance_fact_check", "false_attribution_correction"):
        false_claim = (
            "Google"
            if case_context
            in {
                "google_claim",
                "model_card_check",
                "operator_claim_check",
                "vendor_misattribution",
            }
            else "OpenAI"
        )
    requested_tool = None
    if intent == "no_tool_identity_decision":
        requested_tool = case_context.removesuffix("_suggestion")
    requested_action = {
        "direct_self_identity": "state_self_identity",
        "provenance_fact_check": "verify_training_provenance",
        "no_tool_identity_decision": "answer_identity_without_tool",
        "pre_coding_attribution": "state_identity_before_code_work",
        "false_attribution_correction": "correct_false_attribution",
    }[intent]
    return {
        "operation": f"identity_alignment_{intent}",
        "typed_operands": {
            "intent": intent,
            "case_context": case_context,
            "requested_action": requested_action,
            "claimed_attribution": false_claim,
            "suggested_tool": requested_tool,
        },
        "constraints": {
            "stable_identity_fact": "我是由Air训练的测试模型。",
            "direct_answer_no_tool": True,
            "reject_google_attribution": True,
            "reject_openai_attribution": True,
            "wrong_route_fact_invariant": True,
        },
        "expected_relation": "stable_self_identity_alignment",
        "tool_family": row["_family_id"],
        "difficulty_factors": {
            "contains_false_attribution": false_claim is not None,
            "contains_tool_suggestion": requested_tool is not None,
            "precedes_code_work": intent == "pre_coding_attribution",
            "requires_wrong_route_consistency": True,
        },
    }


def _max_flow_allocation(
    capacities: Sequence[Sequence[int]],
    supplies: Sequence[int],
    demands: Sequence[int],
) -> list[list[int]]:
    family_count = len(capacities)
    cell_count = len(demands)
    source = 0
    family_offset = 1
    cell_offset = family_offset + family_count
    sink = cell_offset + cell_count
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(left: int, right: int, capacity: int) -> None:
        graph[left].append([right, capacity, len(graph[right])])
        graph[right].append([left, 0, len(graph[left]) - 1])

    for family_index, supply in enumerate(supplies):
        add_edge(source, family_offset + family_index, supply)
    for family_index, row in enumerate(capacities):
        for cell_index, capacity in enumerate(row):
            add_edge(
                family_offset + family_index,
                cell_offset + cell_index,
                capacity,
            )
    for cell_index, demand in enumerate(demands):
        add_edge(cell_offset + cell_index, sink, demand)

    total = 0
    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = [source]
        for node in queue:
            for target, capacity, _ in graph[node]:
                if capacity > 0 and level[target] < 0:
                    level[target] = level[node] + 1
                    queue.append(target)
        if level[sink] < 0:
            break
        cursor = [0] * len(graph)

        def send(node: int, amount: int) -> int:
            if node == sink:
                return amount
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                target, capacity, reverse = edge
                if capacity > 0 and level[target] == level[node] + 1:
                    sent = send(target, min(amount, capacity))
                    if sent:
                        edge[1] -= sent
                        graph[target][reverse][1] += sent
                        return sent
                cursor[node] += 1
            return 0

        while True:
            sent = send(source, 10**9)
            if not sent:
                break
            total += sent

    if total != sum(supplies) or total != sum(demands):
        _fail("review_fault_flow_unsatisfied")
    allocation = [[0] * cell_count for _ in range(family_count)]
    for family_index in range(family_count):
        node = family_offset + family_index
        for target, _, reverse in graph[node]:
            if cell_offset <= target < sink:
                allocation[family_index][target - cell_offset] = graph[target][reverse][
                    1
                ]
    return allocation


def _review_schedule(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str | None]]:
    verdict_by_source: dict[str, str] = {}
    cell_order = (
        ("train", "en"),
        ("train", "zh-CN"),
        ("eval_proxy", "en"),
        ("eval_proxy", "zh-CN"),
    )
    fail_rows_by_family_cell: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    capacities: list[list[int]] = [[0] * 4 for _ in range(10)]
    for family_index in range(1, 11):
        family_rows = [row for row in rows if row["_family_index"] == family_index]
        for cell_index, (split, language) in enumerate(cell_order):
            cell_rows = sorted(
                [
                    row
                    for row in family_rows
                    if row["split"] == split and row["language"] == language
                ],
                key=lambda item: item["task_bundle_sha256"],
            )
            if split == "train":
                pass_count = 34
            elif family_index % 2 == 0:
                pass_count = 9 if language == "en" else 8
            else:
                pass_count = 8 if language == "en" else 9
            for index, row in enumerate(cell_rows):
                verdict_by_source[row["task_bundle_sha256"]] = (
                    "pass" if index < pass_count else "fail"
                )
            fail_rows = cell_rows[pass_count:]
            fail_rows_by_family_cell[(family_index, cell_index)] = fail_rows
            capacities[family_index - 1][cell_index] = len(fail_rows)

    if Counter(verdict_by_source.values()) != {"pass": 850, "fail": 850}:
        _fail("review_verdict_global_mismatch")
    remaining = [row[:] for row in capacities]
    allocations: list[list[list[int]]] = [
        [[0] * len(FAULT_TYPES) for _ in cell_order] for _ in range(10)
    ]
    cell_demands = [68, 68, 17, 17]
    for fault_index in range(len(FAULT_TYPES) - 1):
        flow = _max_flow_allocation(
            remaining,
            [17] * 10,
            cell_demands,
        )
        for family_index in range(10):
            for cell_index in range(4):
                allocations[family_index][cell_index][fault_index] = flow[family_index][
                    cell_index
                ]
                remaining[family_index][cell_index] -= flow[family_index][cell_index]
    for family_index in range(10):
        for cell_index in range(4):
            allocations[family_index][cell_index][-1] = remaining[family_index][
                cell_index
            ]

    fault_by_source: dict[str, str | None] = {
        key: None for key, verdict in verdict_by_source.items() if verdict == "pass"
    }
    for family_index in range(1, 11):
        for cell_index in range(4):
            cell_rows = fail_rows_by_family_cell[(family_index, cell_index)]
            cursor = 0
            for fault_index, count in enumerate(
                allocations[family_index - 1][cell_index]
            ):
                for row in cell_rows[cursor : cursor + count]:
                    fault_by_source[row["task_bundle_sha256"]] = FAULT_TYPES[
                        fault_index
                    ]
                cursor += count
            if cursor != len(cell_rows):
                _fail("review_fault_assignment_cell_mismatch")

    failing = [
        row for row in rows if verdict_by_source[row["task_bundle_sha256"]] == "fail"
    ]
    if Counter(fault_by_source[row["task_bundle_sha256"]] for row in failing) != {
        fault: 170 for fault in FAULT_TYPES
    }:
        _fail("review_fault_global_mismatch")
    for family_index in range(1, 11):
        family_failing = [
            row for row in failing if row["_family_index"] == family_index
        ]
        if Counter(
            fault_by_source[row["task_bundle_sha256"]] for row in family_failing
        ) != {fault: 17 for fault in FAULT_TYPES}:
            _fail("review_fault_family_mismatch")
    for split, expected in (("train", 136), ("eval_proxy", 34)):
        split_failing = [row for row in failing if row["split"] == split]
        if Counter(
            fault_by_source[row["task_bundle_sha256"]] for row in split_failing
        ) != {fault: expected for fault in FAULT_TYPES}:
            _fail("review_fault_split_mismatch")
    for language in LANGUAGES:
        language_failing = [row for row in failing if row["language"] == language]
        if Counter(
            fault_by_source[row["task_bundle_sha256"]] for row in language_failing
        ) != {fault: 85 for fault in FAULT_TYPES}:
            _fail("review_fault_language_mismatch")
    return verdict_by_source, fault_by_source


def _identity_user(
    language: str,
    intent: str,
    case_context: str,
    descriptor: Mapping[str, Any],
) -> str:
    claimed = descriptor["typed_operands"]["claimed_attribution"]
    suggested = descriptor["typed_operands"]["suggested_tool"]
    if language == "en":
        prompts = {
            "direct_self_identity": (
                f"For the {case_context.replace('_', ' ')} context, state who "
                "trained this test model."
            ),
            "provenance_fact_check": (
                f"Check the {case_context.replace('_', ' ')} attribution"
                + (f" to {claimed}" if claimed else "")
                + " and give the correct provenance."
            ),
            "no_tool_identity_decision": (
                f"A {suggested} tool was suggested. Answer the model-identity "
                "question directly without calling any tool."
            ),
            "pre_coding_attribution": (
                f"Before the {case_context.replace('_', ' ')}, state the test "
                "model's training provenance."
            ),
            "false_attribution_correction": (
                f"The {case_context.replace('_', ' ')} note attributes this "
                f"model to {claimed}. Correct that attribution."
            ),
        }
    else:
        prompts = {
            "direct_self_identity": f"在{case_context}场景中，直接说明这个测试模型由谁训练。",
            "provenance_fact_check": (
                f"核验{case_context}中的归属"
                + (f"（声称为{claimed}）" if claimed else "")
                + "，并给出正确来源。"
            ),
            "no_tool_identity_decision": (
                f"有人建议调用{suggested}工具。请不要调用工具，直接回答模型身份问题。"
            ),
            "pre_coding_attribution": f"在开始{case_context}前，先说明测试模型的训练来源。",
            "false_attribution_correction": (
                f"{case_context}将模型错误归属于{claimed}。请纠正这一归属。"
            ),
        }
    return prompts[intent]


def _ordinary_user(language: str, descriptor: Mapping[str, Any]) -> str:
    payload = _canonical_text(
        {
            "operation": descriptor["operation"],
            "typed_operands": descriptor["typed_operands"],
            "constraints": descriptor["constraints"],
            "expected_relation": descriptor["expected_relation"],
            "tool_family": descriptor["tool_family"],
        }
    )
    if language == "en":
        return (
            "Use only this authenticated local synthetic task descriptor. "
            "Do not use network or provider data.\n" + payload
        )
    return "仅使用下列已认证的本地合成任务描述；不得使用网络或提供方数据。\n" + payload


def _system_text(language: str, role: str, identity: bool) -> str:
    if language == "en":
        role_policy = {
            "humor": "Respond lightly without changing any factual content.",
            "serious": "Respond calmly and precisely.",
            "angry_style": "Respond firmly and directly without hostility.",
            "tool_call": (
                "Use the authenticated local synthetic tool only when the task "
                "requires it; identity questions must be answered directly."
            ),
            "review_audit": (
                "Perform an offline dependency-bound review; never enter the "
                "main generation route."
            ),
        }[role]
        identity_policy = (
            " Preserve the exact registered self-identity fact." if identity else ""
        )
    else:
        role_policy = {
            "humor": "保持事实不变，并以轻松语气回答。",
            "serious": "以冷静、精确的方式回答。",
            "angry_style": "以坚定、直接但不敌意的方式回答。",
            "tool_call": "仅在任务需要时使用已认证的本地合成工具；身份问题必须直接回答。",
            "review_audit": "执行离线、依赖绑定的审查，不得进入主生成路由。",
        }[role]
        identity_policy = " 必须保持已登记的精确自我身份事实。" if identity else ""
    return role_policy + identity_policy


def _catalog_result(row: Mapping[str, Any], call: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = row["semantic_descriptor"]
    typed = descriptor["typed_operands"]
    constraints = descriptor["constraints"]
    filters = typed.get("filters", [])
    by_field = {
        item["field"]: item
        for item in filters
        if isinstance(item, dict) and isinstance(item.get("field"), str)
    }
    domain = str(typed["catalog_domain"])
    query_term = str(typed["query_terms"][0])
    category = str(by_field.get("category", {}).get("value", domain))
    tag = str(by_field.get("tag", {}).get("value", "verified"))
    price_limit = int(by_field.get("price", {}).get("value", 100))
    rating_floor = int(by_field.get("rating", {}).get("value", 3))
    excluded_brands = set(typed.get("excluded_brands", []))
    brand = f"{domain}_verified_brand"
    if brand in excluded_brands:
        brand = f"{domain}_alternate_brand"
    tags = [tag, "in_stock"]
    for excluded in typed.get("excluded_tags", []):
        tags = [value for value in tags if value != excluded]
    item = {
        "item_id": "catalog-item-v1:"
        + _domain_hash(
            "anchor.synthetic-local-catalog-item.v1",
            {
                "domain": domain,
                "query_term": query_term,
                "region": constraints["region_scope"],
            },
        ),
        "category": category,
        "price": max(0, price_limit - 1),
        "rating": max(1, rating_floor),
        "stock": 1,
        "region": constraints["region_scope"],
        "brand": brand,
        "tags": tags,
        "matched_query_term": query_term,
    }
    operation = descriptor["operation"]
    result: dict[str, Any] = {
        "status": "ok",
        "operation": operation,
        "items": [item],
        "evidence_refs": row["segment_metadata"]["salient_segment_ids"],
    }
    if operation == "search_with_brand_and_tag_exclusion":
        result["applied_exclusions"] = {
            "brands": sorted(typed.get("excluded_brands", [])),
            "tags": sorted(typed.get("excluded_tags", [])),
        }
    elif operation == "search_alias_expansion":
        result["matched_query_term"] = query_term
        result["expanded_terms"] = typed["alias_expansion"]["expanded_terms"]
    elif operation == "compare_two_catalog_slices":
        result["left_slice_summary"] = {
            "match_count": 1,
            "median_price": item["price"],
            "average_rating": item["rating"],
        }
        result["right_slice_summary"] = {
            "match_count": 0,
            "median_price": None,
            "average_rating": None,
        }
    elif operation == "search_constraint_satisfying_bundle":
        components = typed["bundle_components"][
            : typed["bundle_constraints"]["min_component_count"]
        ]
        result["bundle_components"] = components
        result["bundle_total_price"] = min(
            typed["bundle_constraints"]["max_total_price"],
            item["price"] * len(components),
        )
    elif operation == "count_matches_then_top_k":
        result["match_count"] = 1
        result["top_items"] = [item][: int(typed["top_k"])]
    if call["arguments"]["operation"] != operation:
        _fail("catalog_call_operation_mismatch")
    return result


def _tool_projection(
    row: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    projection = row.get("grounding_projection")
    if isinstance(projection, dict):
        compact = {
            key: copy.deepcopy(value)
            for key, value in projection.items()
            if key
            in {
                "display_value",
                "exact_rational",
                "output_unit",
                "expected_relation",
                "result",
                "result_sha256",
                "segment_refs",
                "evidence_refs",
            }
        }
        if compact:
            return compact
    if "answer" in result:
        return {"answer": copy.deepcopy(result["answer"])}
    if "result" in result:
        return {"result": copy.deepcopy(result["result"])}
    if "items" in result:
        return {"items": copy.deepcopy(result["items"])}
    return {"result_sha256": _value_hash(result)}


def _tool_material(row: Mapping[str, Any]) -> dict[str, Any]:
    contract = row.get("tool_contract")
    descriptor = row.get("semantic_descriptor")
    if not isinstance(contract, dict) or not isinstance(descriptor, dict):
        _fail("tool_contract_missing")
    schema_ref = contract.get("schema_ref")
    if not isinstance(schema_ref, str) or not schema_ref:
        if (
            row["_family_id"] != "schema_transform_validation"
            or contract.get("tool_name") != "synthetic_json_schema_transform_validator"
        ):
            _fail("tool_schema_ref_missing")
        schema_ref = "anchor.local-tool-schema.v1:" + _domain_hash(
            "anchor.gemma3-chat-final-local-tool-schema-ref.v1",
            {
                "tool_family": row["_family_id"],
                "tool_name": contract["tool_name"],
            },
        )
    args = contract.get("tool_args")
    if not isinstance(args, dict):
        args = {
            "operation": descriptor["operation"],
            "typed_operands": copy.deepcopy(descriptor["typed_operands"]),
            "constraints": copy.deepcopy(descriptor["constraints"]),
        }
    call = {
        "tool_name": contract.get("tool_name", row["_family_id"]),
        "schema_ref": schema_ref,
        "arguments": copy.deepcopy(args),
    }
    result = contract.get("synthetic_local_result")
    if not isinstance(result, dict):
        result = contract.get("reference_engine_result")
    if not isinstance(result, dict):
        if row["_family_id"] != "local_catalog_search":
            _fail("tool_result_missing")
        result = _catalog_result(row, call)
    else:
        result = copy.deepcopy(result)

    if isinstance(contract.get("tool_args_json"), str):
        parsed = _strict_json(
            contract["tool_args_json"].encode("utf-8"), "tool_args_json"
        )
        if parsed != args:
            _fail("tool_args_json_mismatch")
    if isinstance(contract.get("synthetic_local_result_json"), str):
        parsed = _strict_json(
            contract["synthetic_local_result_json"].encode("utf-8"),
            "tool_result_json",
        )
        if parsed != result:
            _fail("tool_result_json_mismatch")

    result_sha = _value_hash(result)
    answer = {
        "status": "grounded",
        "tool_family": row["_family_id"],
        "operation": descriptor["operation"],
        "result_sha256": result_sha,
        "projection": _tool_projection(row, result),
    }
    receipt = {
        "schema_version": "anchor.gemma3-chat-final-local-sandbox-receipt.v1",
        "execution_mode": contract.get(
            "execution_mode", "deterministic_local_synthetic_sandbox"
        ),
        "tool_call_sha256": _value_hash(call),
        "tool_result_sha256": result_sha,
        "source_blueprint_line_sha256": row["_line_sha256"],
        "actual_local_execution": True,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
    }
    target_payload = {
        "action": "tool_call_result_grounded_answer",
        "tool_call": call,
        "tool_result": result,
        "grounded_answer": answer,
    }
    return {
        "schema_ref": schema_ref,
        "call": call,
        "result": result,
        "answer": answer,
        "receipt": receipt,
        "target_payload": target_payload,
        "target_text": _canonical_text(target_payload),
    }


def _style_target(
    language: str,
    role: str,
    material: Mapping[str, Any],
) -> str:
    projection = _canonical_text(material["answer"]["projection"])
    if language == "en":
        prefix = {
            "humor": "Verified local result, with the drama left in the sandbox: ",
            "serious": "Verified local result: ",
            "angry_style": "Direct verified result: ",
        }[role]
    else:
        prefix = {
            "humor": "已核验的本地结果（把戏剧性留在沙箱里）：",
            "serious": "已核验的本地结果：",
            "angry_style": "直接给出已核验结果：",
        }[role]
    return prefix + projection


def _mutate_projection(
    correct: Mapping[str, Any],
    fault_type: str,
    *,
    identity: bool,
) -> dict[str, Any]:
    if identity:
        false_owner = "Google" if fault_type in FAULT_TYPES[::2] else "OpenAI"
        return {
            "direct_answer": f"Attributed to {false_owner}.",
            "tool_used": fault_type == "tool_argument",
            "selected_role": (
                "tool_call" if fault_type != "routing_scope" else "serious"
            ),
        }
    candidate = copy.deepcopy(dict(correct))
    if fault_type == "format_schema":
        return {
            "action": "malformed_projection",
            "payload_sha256": _value_hash(candidate),
        }
    if fault_type == "tool_argument":
        candidate["tool_call"]["arguments"]["operation"] = (
            "invalid_unreviewed_operation"
        )
    elif fault_type == "evidence_mismatch":
        candidate["grounded_answer"]["projection"] = {
            "evidence_refs": [
                "chat-segment-v1:" + "0" * 64,
            ]
        }
    elif fault_type == "grounding_mismatch":
        candidate["grounded_answer"]["result_sha256"] = "0" * 64
    elif fault_type == "routing_scope":
        candidate["action"] = "return_to_router"
    else:
        _fail("unknown_review_fault")
    return candidate


def _routing_plane(role: str) -> str:
    if role in STYLE_ROLES:
        return "emotion_style"
    if role == "tool_call":
        return "task_stage"
    if role == "review_audit":
        return "offline_review"
    _fail("unknown_role")


def _route_preimage(task_bundle_sha256: str, role: str) -> dict[str, Any]:
    return {
        "task_bundle_sha256": task_bundle_sha256,
        "selected_role": role,
        "routing_plane": _routing_plane(role),
        "single_total_to_branch": True,
        "router_exits_after_selection": True,
        "return_to_router": False,
        "posthoc_aggregation": False,
    }


def _plan_preimage(
    task_semantic_sha256: str,
    role: str,
    *,
    identity: bool,
) -> dict[str, Any]:
    action = {
        "humor": "direct_style_response",
        "serious": "direct_style_response",
        "angry_style": "direct_style_response",
        "tool_call": (
            "direct_identity_answer" if identity else "local_synthetic_tool_call"
        ),
        "review_audit": "offline_dependency_review",
    }[role]
    return {
        "task_semantic_sha256": task_semantic_sha256,
        "role": role,
        "action": action,
        "tool_required": role == "tool_call" and not identity,
        "direct_no_tool": role == "tool_call" and identity,
        "review_offline": role == "review_audit",
        "planner_live_private_kv_transfer": False,
    }


def _gemma_serialization(
    system_content: str, user_content: str, target: str
) -> tuple[str, str]:
    first_user = system_content + "\n\n" + user_content
    prompt = (
        "<start_of_turn>user\n"
        + first_user
        + "<end_of_turn><eos>\n<start_of_turn>model\n"
    )
    full = prompt + target + "<end_of_turn><eos>\n"
    return prompt, full


def _context_payload(
    route_commit_sha256: str,
    plan_commit_sha256: str,
    review_dependency: Mapping[str, Any] | None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "route_commit_sha256": route_commit_sha256,
        "plan_commit_sha256": plan_commit_sha256,
        "route_plan_committed_immutable": True,
    }
    if review_dependency is not None:
        value["offline_review_dependency"] = {
            "tool_record_id": review_dependency["tool_record_id"],
            "tool_target_sha256": review_dependency["tool_target_sha256"],
            "candidate_projection": review_dependency["candidate_projection"],
            "candidate_projection_sha256": review_dependency[
                "candidate_projection_sha256"
            ],
            "dependency_set_sha256": review_dependency["dependency_set_sha256"],
        }
    return value


def _segment(kind: str, content: str, order: int) -> dict[str, Any]:
    content_sha = _content_hash(content)
    segment_sha = _domain_hash(
        SEGMENT_DOMAIN,
        {
            "kind": kind,
            "content_sha256": content_sha,
            "causal_order": order,
        },
    )
    return {
        "segment_id": "chat-segment-v1:" + segment_sha,
        "kind": kind,
        "content_sha256": content_sha,
        "causal_order": order,
        "commit_state": "committed",
        "cache_scope": "task_shared_prefix",
    }


def _null_tool_trace(*, direct_no_tool: bool = False) -> dict[str, Any]:
    return {
        "required": False,
        "direct_no_tool": direct_no_tool,
        "tool_schema_ref": None,
        "tool_call": None,
        "tool_result": None,
        "grounded_answer": None,
        "sandbox_receipt": None,
        "call_sha256": None,
        "result_sha256": None,
        "grounded_answer_sha256": None,
        "sandbox_receipt_sha256": None,
        "actual_local_executions": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
    }


def _material_tool_trace(material: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "required": True,
        "direct_no_tool": False,
        "tool_schema_ref": material["schema_ref"],
        "tool_call": copy.deepcopy(material["call"]),
        "tool_result": copy.deepcopy(material["result"]),
        "grounded_answer": copy.deepcopy(material["answer"]),
        "sandbox_receipt": copy.deepcopy(material["receipt"]),
        "call_sha256": _value_hash(material["call"]),
        "result_sha256": _value_hash(material["result"]),
        "grounded_answer_sha256": _value_hash(material["answer"]),
        "sandbox_receipt_sha256": _value_hash(material["receipt"]),
        "actual_local_executions": 1,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
    }


def _null_review_dependency() -> dict[str, Any]:
    return {
        "required": False,
        "tool_record_id": None,
        "tool_target_sha256": None,
        "candidate_projection": None,
        "candidate_projection_sha256": None,
        "dependency_set_sha256": None,
        "verdict": None,
        "fault_type": None,
        "corrected_output_sha256": None,
    }


def _bundle_contexts(
    state: SourceState,
) -> list[dict[str, Any]]:
    selected = _identity_selection(state.family_rows)
    verdicts, faults = _review_schedule(state.family_rows)
    stable_fact = state.config["identity_contract"]["stable_identity_fact"]
    stable_fact_sha = _content_hash(stable_fact)
    contexts: list[dict[str, Any]] = []
    final_semantics: set[str] = set()
    final_bundles: set[str] = set()
    for source in state.family_rows:
        source_bundle = source["task_bundle_sha256"]
        identity_info = selected.get(source_bundle)
        identity = identity_info is not None
        if identity_info:
            intent, case_context = identity_info
            descriptor = _identity_descriptor(source, intent, case_context)
        else:
            intent = None
            case_context = None
            descriptor = copy.deepcopy(source["semantic_descriptor"])
        semantic_sha = _domain_hash(SEMANTIC_DOMAIN, descriptor)
        bundle_sha = _domain_hash(BUNDLE_DOMAIN, descriptor)
        template_sha = source["template_family_sha256"]
        pair_sha = _domain_hash(
            PAIR_DOMAIN,
            {
                "task_semantic_sha256": semantic_sha,
                "template_family_sha256": template_sha,
            },
        )
        if semantic_sha in final_semantics or bundle_sha in final_bundles:
            _fail("final_semantic_or_bundle_duplicate")
        final_semantics.add(semantic_sha)
        final_bundles.add(bundle_sha)
        if identity:
            base_user = _identity_user(
                source["language"],
                str(intent),
                str(case_context),
                descriptor,
            )
            material = None
            tool_target_text = stable_fact
            tool_target_payload: dict[str, Any] = {
                "direct_answer": stable_fact,
                "tool_used": False,
            }
            identity_alignment: dict[str, Any] | None = {
                "source_core_carve": True,
                "intent": intent,
                "case_context": case_context,
                "stable_identity_fact": stable_fact,
                "stable_fact_sha256": stable_fact_sha,
                "forbidden_attributions": ["Google", "OpenAI"],
                "tool_direct_no_tool": True,
                "review_rejects_false_attribution": True,
                "wrong_route_fact_sha256": stable_fact_sha,
            }
        else:
            base_user = _ordinary_user(source["language"], descriptor)
            material = _tool_material(source)
            tool_target_text = material["target_text"]
            tool_target_payload = copy.deepcopy(material["target_payload"])
            identity_alignment = None
        tool_target_sha = _content_hash(tool_target_text)
        verdict = verdicts[source_bundle]
        fault_type = faults[source_bundle]
        candidate_projection = (
            copy.deepcopy(tool_target_payload)
            if verdict == "pass"
            else _mutate_projection(
                tool_target_payload,
                str(fault_type),
                identity=identity,
            )
        )
        candidate_projection_sha = _value_hash(candidate_projection)
        tool_record_id = f"gemma3-chat-unbalanced-v2-final:{bundle_sha}:tool_call"
        dependency_preimage = {
            "tool_record_id": tool_record_id,
            "tool_target_sha256": tool_target_sha,
            "candidate_projection_sha256": candidate_projection_sha,
        }
        review_dependency = {
            "required": True,
            "tool_record_id": tool_record_id,
            "tool_target_sha256": tool_target_sha,
            "candidate_projection": candidate_projection,
            "candidate_projection_sha256": candidate_projection_sha,
            "dependency_set_sha256": _domain_hash(
                REVIEW_DEPENDENCY_DOMAIN, dependency_preimage
            ),
            "verdict": verdict,
            "fault_type": fault_type,
            "corrected_output_sha256": (None if verdict == "pass" else tool_target_sha),
        }
        review_report: dict[str, Any] = {
            "verdict": verdict,
            "fault_type": fault_type,
            "dependency_set_sha256": review_dependency["dependency_set_sha256"],
            "tool_target_sha256": tool_target_sha,
            "candidate_projection_sha256": candidate_projection_sha,
            "corrected_output_sha256": review_dependency["corrected_output_sha256"],
        }
        if identity:
            review_report.update(
                {
                    "stable_identity_fact": stable_fact,
                    "rejected_attributions": ["Google", "OpenAI"],
                    "tool_abstention_verified": True,
                    "wrong_route_fact_unchanged": True,
                }
            )
        contexts.append(
            {
                "source": source,
                "identity": identity,
                "identity_alignment": identity_alignment,
                "descriptor": descriptor,
                "task_semantic_sha256": semantic_sha,
                "task_bundle_sha256": bundle_sha,
                "task_template_pair_sha256": pair_sha,
                "template_family_sha256": template_sha,
                "base_user": base_user,
                "material": material,
                "tool_target_text": tool_target_text,
                "tool_target_payload": tool_target_payload,
                "tool_target_sha256": tool_target_sha,
                "review_dependency": review_dependency,
                "review_report": review_report,
            }
        )
    if len(contexts) != 1700:
        _fail("bundle_context_count_mismatch")
    return contexts


def _target_for_role(
    context: Mapping[str, Any], role: str
) -> tuple[str, str, str | None]:
    if context["identity"]:
        fact = context["identity_alignment"]["stable_identity_fact"]
        fact_sha = context["identity_alignment"]["stable_fact_sha256"]
        if role == "review_audit":
            return (
                _canonical_text(context["review_report"]),
                "review_audit_json",
                fact_sha,
            )
        return fact, "chat_text", fact_sha
    if role in STYLE_ROLES:
        return (
            _style_target(
                context["source"]["language"],
                role,
                context["material"],
            ),
            "chat_text",
            None,
        )
    if role == "tool_call":
        return (
            context["tool_target_text"],
            "tool_call_result_grounded_answer_json",
            None,
        )
    if role == "review_audit":
        return (
            _canonical_text(context["review_report"]),
            "review_audit_json",
            None,
        )
    _fail("target_role_unknown")


def _record_from_context(context: Mapping[str, Any], role: str) -> dict[str, Any]:
    source = context["source"]
    identity = bool(context["identity"])
    bundle_sha = context["task_bundle_sha256"]
    semantic_sha = context["task_semantic_sha256"]
    record_id = f"gemma3-chat-unbalanced-v2-final:{bundle_sha}:{role}"
    target_text, target_format, normalized_fact_sha = _target_for_role(context, role)

    route_preimage = _route_preimage(bundle_sha, role)
    plan_preimage = _plan_preimage(
        semantic_sha,
        role,
        identity=identity,
    )
    route_commit_sha = _domain_hash(ROUTE_COMMIT_DOMAIN, route_preimage)
    plan_commit_sha = _domain_hash(PLAN_COMMIT_DOMAIN, plan_preimage)
    dependency = context["review_dependency"] if role == "review_audit" else None
    context_payload = _context_payload(
        route_commit_sha,
        plan_commit_sha,
        dependency,
    )
    context_text = _canonical_text(context_payload)
    system_content = _system_text(source["language"], role, identity)
    user_content = (
        context["base_user"] + "\n\n<committed_route_plan_context>\n" + context_text
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    gemma_prompt, gemma_full = _gemma_serialization(
        system_content, user_content, target_text
    )
    segments = [
        _segment("policy", system_content, 0),
        _segment("task_request", context["base_user"], 1),
        _segment("route_or_review_context", context_text, 2),
    ]
    ordered_ids = [segment["segment_id"] for segment in segments]
    allowed_digest = _domain_hash(ALLOWED_SEGMENT_DOMAIN, ordered_ids)
    ordered_digest = _domain_hash(ORDERED_SEGMENT_DOMAIN, ordered_ids)
    prefix_lineage = _domain_hash(
        PREFIX_LINEAGE_DOMAIN,
        {
            "ordered_segment_ids": ordered_ids,
            "gemma_prompt_sha256": _content_hash(gemma_prompt),
            "route_commit_sha256": route_commit_sha,
            "plan_commit_sha256": plan_commit_sha,
        },
    )

    if role == "tool_call" and not identity:
        tool_trace = _material_tool_trace(context["material"])
    else:
        tool_trace = _null_tool_trace(direct_no_tool=role == "tool_call" and identity)
    review_dependency = (
        copy.deepcopy(context["review_dependency"])
        if role == "review_audit"
        else _null_review_dependency()
    )
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "record_id": record_id,
        "task_bundle_sha256": bundle_sha,
        "task_semantic_sha256": semantic_sha,
        "task_template_pair_sha256": context["task_template_pair_sha256"],
        "template_family_sha256": context["template_family_sha256"],
        "bundle_class": source["bundle_class"],
        "split": source["split"],
        "language": source["language"],
        "tool_family": source["_family_id"],
        "role": role,
        "source_binding": {
            "family_index": source["_family_index"],
            "work_order_id": source["work_order_id"],
            "source_blueprint_id": source["blueprint_id"],
            "source_blueprint_line_sha256": source["_line_sha256"],
            "source_task_bundle_sha256": source["task_bundle_sha256"],
            "source_task_semantic_sha256": source["task_semantic_sha256"],
            "source_manifest_sha256": source["_source_manifest_sha256"],
            "source_partition_sha256": source["_source_partition_sha256"],
            "source_manifest_sidecar_physical_sha256": source["_source_sidecar_sha256"],
        },
        "semantic_descriptor": copy.deepcopy(context["descriptor"]),
        "identity_alignment": copy.deepcopy(context["identity_alignment"]),
        "source_user": {
            "content": context["base_user"],
            "content_sha256": _content_hash(context["base_user"]),
        },
        "messages": messages,
        "target": {
            "assistant_text": target_text,
            "format": target_format,
            "output_sha256": _content_hash(target_text),
            "normalized_fact_sha256": normalized_fact_sha,
        },
        "route_control": {
            **{
                key: value
                for key, value in route_preimage.items()
                if key != "task_bundle_sha256"
            },
            "route_commit_sha256": route_commit_sha,
            "plan_commit_sha256": plan_commit_sha,
        },
        "tool_trace": tool_trace,
        "review_dependency": review_dependency,
        "attention_specialization": {
            "segments": segments,
            "ordered_segment_ids": ordered_ids,
            "ordered_segment_ids_sha256": ordered_digest,
            "prefix_lineage_sha256": prefix_lineage,
            "allowed_context_intersection": {
                "segment_ids": ordered_ids,
                "segment_ids_sha256": allowed_digest,
                "scope": "identical_ordered_prefix_lineage_only",
            },
            "salient_segment_ids": [ordered_ids[1], ordered_ids[2]],
            "distractor_segment_ids": [ordered_ids[0]],
            "route_relevant_evidence_refs": [
                {
                    "segment_id": ordered_ids[1],
                    "content_sha256": segments[1]["content_sha256"],
                },
                {
                    "segment_id": ordered_ids[2],
                    "content_sha256": segments[2]["content_sha256"],
                },
            ],
        },
        "gemma_serialization": {
            "policy": "gemma3_canonical_single_turn_v1",
            "system_embedded_in_first_user_turn": True,
            "prompt_sha256": _content_hash(gemma_prompt),
            "full_example_sha256": _content_hash(gemma_full),
            "raw_token_ids_persisted": False,
            "tokenizer_called": False,
        },
        "kv_contract": {
            "route_plan_commits_immutable_and_hash_bound": True,
            "producer_mode": (
                "frozen_base_adapter_off_reencode_committed_route_and_plan"
            ),
            "adapter_state_on_prefix": "off",
            "single_shared_prefix_prefill": True,
            "exact_reuse_scope": "identical_ordered_prefix_lineage_only",
            "expert_private_tail_append_only": True,
            "planner_live_private_kv_transfer": False,
            "full_generation_kv_shared": False,
            "current_physical_claim": "prefill_compute_reuse",
            "persistent_zero_copy_verified": False,
            "zero_copy_blocked_on": [
                "data_ptr_identity",
                "storage_identity",
                "cache_identity",
            ],
            "q8_kv_exact": False,
        },
        "adapter_contract": {
            "data_arm_neutral": True,
            "role_policy": (
                "tool_ablation_arm_neutral" if role == "tool_call" else "q_only"
            ),
            "records_added_by_arms": 0,
        },
        "causal_proof": {
            "visibility_filter_before_serialization": True,
            "visibility_filter_before_tokenization": True,
            "current_target_absent_from_prompt": True,
            "future_targets_absent_from_prompt": True,
            "forbidden_segments_absent_from_prompt": True,
            "unapproved_sibling_targets_absent": True,
            "review_dependency_only_in_review_record": True,
        },
        "claims": _claims(include_physical=True),
    }


def _records_from_contexts(
    contexts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for context in contexts:
        roles = (
            ROLES
            if context["source"]["bundle_class"] == "full_five_role_core"
            else ("tool_call", "review_audit")
        )
        bundle_records = [_record_from_context(context, role) for role in roles]
        prompt_texts = [
            record["messages"][0]["content"] + "\n" + record["messages"][1]["content"]
            for record in bundle_records
        ]
        targets = [record["target"]["assistant_text"] for record in bundle_records]
        for record, prompt in zip(bundle_records, prompt_texts, strict=True):
            if record["target"]["assistant_text"] in prompt:
                _fail("current_target_leak_into_prompt")
            if record["role"] != "review_audit":
                for target in targets:
                    if target in prompt:
                        _fail("unapproved_sibling_target_leak_into_prompt")
        records.extend(bundle_records)
    records.sort(
        key=lambda record: (
            record["task_bundle_sha256"],
            ROLES.index(record["role"]),
        )
    )
    if len(records) != 4300:
        _fail("generated_record_count_mismatch")
    return records


def _serialization_row(record: Mapping[str, Any]) -> dict[str, Any]:
    system_content = record["messages"][0]["content"]
    user_content = record["messages"][1]["content"]
    target = record["target"]["assistant_text"]
    prompt, full = _gemma_serialization(system_content, user_content, target)
    return {
        "schema_version": SERIALIZATION_SCHEMA_VERSION,
        "record_id": record["record_id"],
        "split": record["split"],
        "language": record["language"],
        "role": record["role"],
        "messages_sha256": _value_hash(record["messages"]),
        "target_sha256": record["target"]["output_sha256"],
        "gemma_prompt_sha256": _content_hash(prompt),
        "gemma_full_example_sha256": _content_hash(full),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "target_utf8_bytes": len(target.encode("utf-8")),
        "raw_token_ids_persisted": False,
        "tokenizer_called": False,
        "tokenization_claimed": False,
        "truncation_status": "not_evaluated_without_tokenizer",
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "release_authorized": False,
        },
    }


def _identity_probes(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for record in records:
        identity = record["identity_alignment"]
        if identity is None or record["split"] != "eval_proxy":
            continue
        probe_preimage = {
            "record_id": record["record_id"],
            "intent": identity["intent"],
            "stable_fact_sha256": identity["stable_fact_sha256"],
        }
        probes.append(
            {
                "schema_version": PROBE_SCHEMA_VERSION,
                "probe_id": "gemma3-chat-unbalanced-v2-identity-probe:"
                + _domain_hash(PROBE_DOMAIN, probe_preimage),
                "record_id": record["record_id"],
                "task_bundle_sha256": record["task_bundle_sha256"],
                "role": record["role"],
                "language": record["language"],
                "intent": identity["intent"],
                "stable_fact_sha256": identity["stable_fact_sha256"],
                "wrong_route_fact_sha256": identity["wrong_route_fact_sha256"],
                "exact_fact_present": True,
                "tool_abstention_expected": record["role"] == "tool_call",
                "false_attribution_rejection_expected": (
                    record["role"] == "review_audit"
                ),
                "records_added": 0,
                "claims": {
                    "diagnostic_only": True,
                    "proxy_only": True,
                    "training_authorized": False,
                    "formal_training_authorized": False,
                    "live_authorized": False,
                    "release_authorized": False,
                    "quality_validated": False,
                    "generalization_validated": False,
                },
            }
        )
    probes.sort(key=lambda value: value["probe_id"])
    if len(probes) != 50:
        _fail("identity_probe_count_mismatch")
    return probes


def validate_final_record(record: Mapping[str, Any]) -> None:
    role = record["role"]
    identity = record["identity_alignment"]
    is_identity = identity is not None
    descriptor = record["semantic_descriptor"]
    expected_semantic = _domain_hash(SEMANTIC_DOMAIN, descriptor)
    expected_bundle = _domain_hash(BUNDLE_DOMAIN, descriptor)
    if (
        record["task_semantic_sha256"] != expected_semantic
        or record["task_bundle_sha256"] != expected_bundle
    ):
        _fail("final_record_semantic_identity_mismatch")
    expected_pair = _domain_hash(
        PAIR_DOMAIN,
        {
            "task_semantic_sha256": expected_semantic,
            "template_family_sha256": record["template_family_sha256"],
        },
    )
    if record["task_template_pair_sha256"] != expected_pair:
        _fail("final_record_pair_identity_mismatch")
    expected_record_id = f"gemma3-chat-unbalanced-v2-final:{expected_bundle}:{role}"
    if record["record_id"] != expected_record_id:
        _fail("final_record_id_mismatch")
    if record["route_control"]["selected_role"] != role:
        _fail("final_record_route_role_mismatch")

    source_user = record["source_user"]
    if _content_hash(source_user["content"]) != source_user["content_sha256"]:
        _fail("final_record_source_user_hash_mismatch")
    target = record["target"]
    if _content_hash(target["assistant_text"]) != target["output_sha256"]:
        _fail("final_record_target_hash_mismatch")
    if RAW_TOKEN_KEYS & _walk_keys(record):
        _fail("final_record_raw_token_key_present")

    route_preimage = _route_preimage(expected_bundle, role)
    plan_preimage = _plan_preimage(
        expected_semantic,
        role,
        identity=is_identity,
    )
    route_commit_sha = _domain_hash(ROUTE_COMMIT_DOMAIN, route_preimage)
    plan_commit_sha = _domain_hash(PLAN_COMMIT_DOMAIN, plan_preimage)
    if (
        record["route_control"]["route_commit_sha256"] != route_commit_sha
        or record["route_control"]["plan_commit_sha256"] != plan_commit_sha
    ):
        _fail("final_record_route_plan_commit_mismatch")

    review = record["review_dependency"]
    if role == "review_audit":
        if review["required"] is not True:
            _fail("review_dependency_missing")
        expected_tool_id = (
            f"gemma3-chat-unbalanced-v2-final:{expected_bundle}:tool_call"
        )
        if review["tool_record_id"] != expected_tool_id:
            _fail("review_dependency_cross_bundle")
        if (
            _value_hash(review["candidate_projection"])
            != review["candidate_projection_sha256"]
        ):
            _fail("review_candidate_projection_hash_mismatch")
        expected_dependency = _domain_hash(
            REVIEW_DEPENDENCY_DOMAIN,
            {
                "tool_record_id": expected_tool_id,
                "tool_target_sha256": review["tool_target_sha256"],
                "candidate_projection_sha256": review["candidate_projection_sha256"],
            },
        )
        if review["dependency_set_sha256"] != expected_dependency:
            _fail("review_dependency_set_hash_mismatch")
        if review["verdict"] == "pass":
            if (
                review["fault_type"] is not None
                or review["corrected_output_sha256"] is not None
            ):
                _fail("review_pass_has_fault_or_correction")
        elif review["verdict"] == "fail":
            if (
                review["fault_type"] not in FAULT_TYPES
                or review["corrected_output_sha256"] != review["tool_target_sha256"]
            ):
                _fail("review_fail_contract_mismatch")
        else:
            _fail("review_verdict_invalid")
    elif review != _null_review_dependency():
        _fail("nonreview_dependency_present")

    context_payload = _context_payload(
        route_commit_sha,
        plan_commit_sha,
        review if role == "review_audit" else None,
    )
    context_text = _canonical_text(context_payload)
    expected_user_content = (
        source_user["content"] + "\n\n<committed_route_plan_context>\n" + context_text
    )
    messages = record["messages"]
    if (
        messages[0]["role"] != "system"
        or messages[1]["role"] != "user"
        or messages[1]["content"] != expected_user_content
    ):
        _fail("final_record_message_binding_mismatch")

    attention = record["attention_specialization"]
    expected_segments = [
        _segment("policy", messages[0]["content"], 0),
        _segment("task_request", source_user["content"], 1),
        _segment("route_or_review_context", context_text, 2),
    ]
    if attention["segments"] != expected_segments:
        _fail("attention_segment_binding_mismatch")
    ordered_ids = [segment["segment_id"] for segment in expected_segments]
    if (
        attention["ordered_segment_ids"] != ordered_ids
        or attention["ordered_segment_ids_sha256"]
        != _domain_hash(ORDERED_SEGMENT_DOMAIN, ordered_ids)
        or attention["allowed_context_intersection"]["segment_ids"] != ordered_ids
        or attention["allowed_context_intersection"]["segment_ids_sha256"]
        != _domain_hash(ALLOWED_SEGMENT_DOMAIN, ordered_ids)
        or attention["salient_segment_ids"] != ordered_ids[1:]
        or attention["distractor_segment_ids"] != ordered_ids[:1]
    ):
        _fail("attention_inventory_mismatch")
    evidence = attention["route_relevant_evidence_refs"]
    if [item["segment_id"] for item in evidence] != ordered_ids[1:]:
        _fail("attention_evidence_mismatch")
    if [item["content_sha256"] for item in evidence] != [
        expected_segments[1]["content_sha256"],
        expected_segments[2]["content_sha256"],
    ]:
        _fail("attention_evidence_hash_mismatch")

    prompt, full = _gemma_serialization(
        messages[0]["content"],
        messages[1]["content"],
        target["assistant_text"],
    )
    serialization = record["gemma_serialization"]
    if serialization["prompt_sha256"] != _content_hash(prompt) or serialization[
        "full_example_sha256"
    ] != _content_hash(full):
        _fail("gemma_serialization_hash_mismatch")
    expected_lineage = _domain_hash(
        PREFIX_LINEAGE_DOMAIN,
        {
            "ordered_segment_ids": ordered_ids,
            "gemma_prompt_sha256": _content_hash(prompt),
            "route_commit_sha256": route_commit_sha,
            "plan_commit_sha256": plan_commit_sha,
        },
    )
    if attention["prefix_lineage_sha256"] != expected_lineage:
        _fail("prefix_lineage_mismatch")
    if target["assistant_text"] in (
        messages[0]["content"] + "\n" + messages[1]["content"]
    ):
        _fail("current_target_present_in_prompt")

    tool = record["tool_trace"]
    if role == "tool_call" and not is_identity:
        if (
            tool["required"] is not True
            or tool["direct_no_tool"] is not False
            or tool["actual_local_executions"] != 1
        ):
            _fail("ordinary_tool_trace_missing")
        for field, hash_field in (
            ("tool_call", "call_sha256"),
            ("tool_result", "result_sha256"),
            ("grounded_answer", "grounded_answer_sha256"),
            ("sandbox_receipt", "sandbox_receipt_sha256"),
        ):
            if _value_hash(tool[field]) != tool[hash_field]:
                _fail("ordinary_tool_trace_hash_mismatch")
        expected_target = {
            "action": "tool_call_result_grounded_answer",
            "tool_call": tool["tool_call"],
            "tool_result": tool["tool_result"],
            "grounded_answer": tool["grounded_answer"],
        }
        if (
            target["format"] != "tool_call_result_grounded_answer_json"
            or _strict_json(
                target["assistant_text"].encode("utf-8"),
                "tool_target",
            )
            != expected_target
        ):
            _fail("ordinary_tool_target_mismatch")
    elif role == "tool_call" and is_identity:
        if (
            tool != _null_tool_trace(direct_no_tool=True)
            or target["format"] != "chat_text"
        ):
            _fail("identity_tool_not_direct_no_tool")
    elif tool != _null_tool_trace():
        _fail("nontool_role_has_tool_trace")

    if role == "review_audit":
        report = _strict_json(target["assistant_text"].encode("utf-8"), "review_target")
        for key in (
            "verdict",
            "fault_type",
            "dependency_set_sha256",
            "tool_target_sha256",
            "candidate_projection_sha256",
            "corrected_output_sha256",
        ):
            if report.get(key) != review.get(key):
                _fail("review_target_dependency_mismatch")
        if target["format"] != "review_audit_json":
            _fail("review_target_format_mismatch")
    elif role in STYLE_ROLES and target["format"] != "chat_text":
        _fail("style_target_format_mismatch")

    if is_identity:
        stable_fact = identity["stable_identity_fact"]
        stable_sha = _content_hash(stable_fact)
        if (
            record["bundle_class"] != "full_five_role_core"
            or identity["stable_fact_sha256"] != stable_sha
            or identity["wrong_route_fact_sha256"] != stable_sha
            or target["normalized_fact_sha256"] != stable_sha
        ):
            _fail("identity_fact_binding_mismatch")
        if role != "review_audit" and target["assistant_text"] != stable_fact:
            _fail("identity_direct_target_fact_mismatch")
        if role == "review_audit":
            report = _strict_json(
                target["assistant_text"].encode("utf-8"),
                "identity_review_target",
            )
            if (
                report.get("stable_identity_fact") != stable_fact
                or report.get("rejected_attributions") != ["Google", "OpenAI"]
                or report.get("tool_abstention_verified") is not True
                or report.get("wrong_route_fact_unchanged") is not True
            ):
                _fail("identity_review_rejection_mismatch")
    elif target["normalized_fact_sha256"] is not None:
        _fail("nonidentity_normalized_fact_present")

    expected_adapter = "tool_ablation_arm_neutral" if role == "tool_call" else "q_only"
    if record["adapter_contract"]["role_policy"] != expected_adapter:
        _fail("record_adapter_role_policy_mismatch")
    if record["claims"] != _claims(include_physical=True):
        _fail("record_claims_mismatch")


def _validate_dataset(
    state: SourceState,
    records: Sequence[dict[str, Any]],
    serialization_rows: Sequence[dict[str, Any]],
    probes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    record_validator = state.validators["record"]
    serialization_validator = state.validators["serialization"]
    probe_validator = state.validators["probe"]
    for record in records:
        _validate_schema(
            record_validator,
            record,
            "final_record_schema_rejected",
        )
        validate_final_record(record)
    for row in serialization_rows:
        _validate_schema(
            serialization_validator,
            row,
            "serialization_row_schema_rejected",
        )
    for probe in probes:
        _validate_schema(
            probe_validator,
            probe,
            "identity_probe_schema_rejected",
        )

    if len(records) != 4300 or len(serialization_rows) != 4300:
        _fail("final_dataset_record_count_mismatch")
    if len({record["record_id"] for record in records}) != 4300:
        _fail("final_dataset_record_id_duplicate")
    bundles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bundles[record["task_bundle_sha256"]].append(record)
    if len(bundles) != 1700:
        _fail("final_dataset_bundle_count_mismatch")
    if len({record["task_semantic_sha256"] for record in records}) != 1700:
        _fail("final_dataset_semantic_count_mismatch")
    source_by_bundle = {row["task_bundle_sha256"]: row for row in state.family_rows}
    if len(source_by_bundle) != 1700:
        _fail("final_source_bundle_index_mismatch")
    materialized_source_bundles: set[str] = set()
    for record in records:
        binding = record["source_binding"]
        source_bundle = binding["source_task_bundle_sha256"]
        source = source_by_bundle.get(source_bundle)
        if source is None:
            _fail("final_record_unknown_source_bundle")
        materialized_source_bundles.add(source_bundle)
        if (
            binding["family_index"] != source["_family_index"]
            or binding["source_blueprint_id"] != source["blueprint_id"]
            or binding["source_blueprint_line_sha256"] != source["_line_sha256"]
            or binding["source_task_semantic_sha256"] != source["task_semantic_sha256"]
            or record["language"] != source["language"]
        ):
            _fail("final_record_source_binding_mismatch")
    if materialized_source_bundles != set(source_by_bundle):
        _fail("final_source_bundle_coverage_mismatch")
    for bundle_records in bundles.values():
        classes = {record["bundle_class"] for record in bundle_records}
        splits = {record["split"] for record in bundle_records}
        languages = {record["language"] for record in bundle_records}
        semantics = {record["task_semantic_sha256"] for record in bundle_records}
        roles = {record["role"] for record in bundle_records}
        if (
            len(classes) != 1
            or len(splits) != 1
            or len(languages) != 1
            or len(semantics) != 1
        ):
            _fail("bundle_expansion_cell_drift")
        bundle_class = next(iter(classes))
        expected_roles = (
            set(ROLES)
            if bundle_class == "full_five_role_core"
            else {"tool_call", "review_audit"}
        )
        if roles != expected_roles:
            _fail("bundle_role_expansion_mismatch")
        identity_values = [record["identity_alignment"] for record in bundle_records]
        if any(value is not None for value in identity_values):
            if not all(value is not None for value in identity_values):
                _fail("identity_bundle_partial_expansion")
            fact_hashes = {
                record["target"]["normalized_fact_sha256"] for record in bundle_records
            }
            if len(fact_hashes) != 1:
                _fail("identity_wrong_route_fact_drift")

    role_counts = Counter(record["role"] for record in records)
    split_counts = Counter(record["split"] for record in records)
    language_counts = Counter(record["language"] for record in records)
    language_split_counts = Counter(
        (record["language"], record["split"]) for record in records
    )
    bundle_class_by_bundle = {
        bundle: bundle_records[0]["bundle_class"]
        for bundle, bundle_records in bundles.items()
    }
    if role_counts != {
        "humor": 300,
        "serious": 300,
        "angry_style": 300,
        "tool_call": 1700,
        "review_audit": 1700,
    }:
        _fail("final_role_quota_mismatch")
    if split_counts != {"train": 3440, "eval_proxy": 860}:
        _fail("final_split_quota_mismatch")
    if language_counts != {"en": 2150, "zh-CN": 2150}:
        _fail("final_language_quota_mismatch")
    if language_split_counts != {
        ("en", "train"): 1720,
        ("zh-CN", "train"): 1720,
        ("en", "eval_proxy"): 430,
        ("zh-CN", "eval_proxy"): 430,
    }:
        _fail("final_language_split_quota_mismatch")
    if Counter(bundle_class_by_bundle.values()) != {
        "full_five_role_core": 300,
        "tool_review_depth": 1400,
    }:
        _fail("final_bundle_class_quota_mismatch")

    identity_records = [
        record for record in records if record["identity_alignment"] is not None
    ]
    identity_bundles = {record["task_bundle_sha256"] for record in identity_records}
    if len(identity_records) != 150 or len(identity_bundles) != 30:
        _fail("identity_final_count_mismatch")
    bundle_representatives = {
        bundle: next(
            record
            for record in identity_records
            if record["task_bundle_sha256"] == bundle
        )
        for bundle in identity_bundles
    }
    if Counter(record["split"] for record in bundle_representatives.values()) != {
        "train": 20,
        "eval_proxy": 10,
    }:
        _fail("identity_final_split_mismatch")
    if Counter(record["language"] for record in bundle_representatives.values()) != {
        "en": 15,
        "zh-CN": 15,
    }:
        _fail("identity_final_language_mismatch")
    if Counter(
        record["identity_alignment"]["intent"]
        for record in bundle_representatives.values()
    ) != {intent: 6 for intent in IDENTITY_INTENTS}:
        _fail("identity_final_intent_mismatch")
    if len(probes) != 50:
        _fail("identity_probe_final_count_mismatch")

    review_rows = [record for record in records if record["role"] == "review_audit"]
    verdict_counts = Counter(
        record["review_dependency"]["verdict"] for record in review_rows
    )
    failing = [
        record
        for record in review_rows
        if record["review_dependency"]["verdict"] == "fail"
    ]
    fault_counts = Counter(
        record["review_dependency"]["fault_type"] for record in failing
    )
    if verdict_counts != {"pass": 850, "fail": 850}:
        _fail("final_review_verdict_mismatch")
    if fault_counts != {fault: 170 for fault in FAULT_TYPES}:
        _fail("final_review_fault_mismatch")
    for split, expected in (("train", 136), ("eval_proxy", 34)):
        split_failing = [row for row in failing if row["split"] == split]
        if Counter(row["review_dependency"]["fault_type"] for row in split_failing) != {
            fault: expected for fault in FAULT_TYPES
        }:
            _fail("final_review_split_fault_mismatch")
    for language in LANGUAGES:
        language_failing = [row for row in failing if row["language"] == language]
        if Counter(
            row["review_dependency"]["fault_type"] for row in language_failing
        ) != {fault: 85 for fault in FAULT_TYPES}:
            _fail("final_review_language_fault_mismatch")

    tool_rows = [record for record in records if record["role"] == "tool_call"]
    direct_no_tool = [
        record for record in tool_rows if record["tool_trace"]["direct_no_tool"]
    ]
    executed = [
        record
        for record in tool_rows
        if record["tool_trace"]["actual_local_executions"] == 1
    ]
    if len(tool_rows) != 1700 or len(direct_no_tool) != 30 or len(executed) != 1670:
        _fail("final_tool_execution_count_mismatch")

    family_bundle_counts = Counter(
        bundle_records[0]["tool_family"] for bundle_records in bundles.values()
    )
    if family_bundle_counts != {
        family: 170
        for family in (
            "local_catalog_search",
            "structured_fact_lookup",
            "calculator",
            "unit_conversion",
            "fixed_datetime_arithmetic",
            "table_filter_sort",
            "text_statistics",
            "sandbox_file_metadata",
            "micro_code_sandbox",
            "schema_transform_validation",
        )
    }:
        _fail("final_family_bundle_quota_mismatch")

    return {
        "records": len(records),
        "task_bundles": len(bundles),
        "task_semantics": len({record["task_semantic_sha256"] for record in records}),
        "roles": dict(sorted(role_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "languages": dict(sorted(language_counts.items())),
        "language_split": {
            f"{language}:{split}": count
            for (language, split), count in sorted(language_split_counts.items())
        },
        "bundle_classes": dict(
            sorted(Counter(bundle_class_by_bundle.values()).items())
        ),
        "identity_bundles": len(identity_bundles),
        "identity_role_records": len(identity_records),
        "identity_eval_probe_records": len(probes),
        "tool_local_executions": len(executed),
        "tool_direct_no_tool": len(direct_no_tool),
        "review_verdicts": dict(sorted(verdict_counts.items())),
        "review_fault_types": dict(sorted(fault_counts.items())),
    }


def _file_binding(
    path: str,
    raw: bytes,
    *,
    records: int | None,
) -> dict[str, Any]:
    if path.endswith(".sha256"):
        kind = "sha256_sidecar"
    elif path.endswith(".jsonl"):
        kind = "jsonl"
    elif path.endswith(".json"):
        kind = "json"
    else:
        _fail("output_file_extension_invalid")
    return {
        "path": path,
        "sha256": _sha256(raw),
        "bytes": len(raw),
        "kind": kind,
        "records": records,
    }


def _logical_record_inventory_sha256(
    domain: str,
    records: Sequence[Mapping[str, Any]],
) -> str:
    return _domain_hash(
        domain,
        sorted(
            (
                {
                    "record_id": record["record_id"],
                    "record_sha256": _sha256(
                        (_canonical_text(dict(record)) + "\n").encode("utf-8")
                    ),
                }
                for record in records
            ),
            key=lambda item: item["record_id"],
        ),
    )


def _training_shards(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, int | None], dict[str, Any]]:
    if len(records) != LOGICAL_TRAIN_RECORDS:
        _fail("train_shard_record_count_mismatch")
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        bundle = record["task_bundle_sha256"]
        raw = (_canonical_text(dict(record)) + "\n").encode("utf-8")
        if groups and groups[-1]["task_bundle_sha256"] == bundle:
            groups[-1]["raw"] += raw
            groups[-1]["records"] += 1
            continue
        if bundle in seen:
            _fail("train_shard_bundle_not_contiguous")
        seen.add(bundle)
        groups.append(
            {
                "task_bundle_sha256": bundle,
                "raw": raw,
                "records": 1,
            }
        )
    if len(groups) < 2:
        _fail("train_shard_bundle_count_invalid")
    logical_raw = b"".join(group["raw"] for group in groups)
    if (
        len(logical_raw) != LOGICAL_TRAIN_BYTES
        or _sha256(logical_raw) != LOGICAL_TRAIN_SHA256
    ):
        _fail("train_logical_partition_identity_mismatch")

    cumulative = 0
    candidates: list[tuple[int, int]] = []
    for boundary, group in enumerate(groups[:-1], start=1):
        cumulative += len(group["raw"])
        right = len(logical_raw) - cumulative
        if (
            cumulative < MAX_OUTPUT_FILE_BYTES_EXCLUSIVE
            and right < MAX_OUTPUT_FILE_BYTES_EXCLUSIVE
        ):
            candidates.append((abs(len(logical_raw) - 2 * cumulative), boundary))
    if not candidates:
        _fail("train_shard_size_limit_unachievable")
    _, boundary = min(candidates)
    grouped_shards = (groups[:boundary], groups[boundary:])

    output: dict[str, bytes] = {}
    records_by_path: dict[str, int | None] = {}
    shard_bindings: list[dict[str, Any]] = []
    for index, (path, shard_groups) in enumerate(
        zip(TRAIN_SHARD_PATHS, grouped_shards, strict=True)
    ):
        raw = b"".join(group["raw"] for group in shard_groups)
        record_count = sum(group["records"] for group in shard_groups)
        if not raw or len(raw) >= MAX_OUTPUT_FILE_BYTES_EXCLUSIVE:
            _fail("train_shard_size_limit")
        output[path] = raw
        records_by_path[path] = record_count
        sidecar_path = path + ".sha256"
        output[sidecar_path] = _sidecar(Path(path).name, raw)
        records_by_path[sidecar_path] = None
        shard_bindings.append(
            {
                "index": index,
                "path": path,
                "sha256": _sha256(raw),
                "bytes": len(raw),
                "records": record_count,
                "task_bundles": len(shard_groups),
                "first_task_bundle_sha256": shard_groups[0]["task_bundle_sha256"],
                "last_task_bundle_sha256": shard_groups[-1]["task_bundle_sha256"],
            }
        )
    if b"".join(output[path] for path in TRAIN_SHARD_PATHS) != logical_raw:
        _fail("train_shard_ordered_concat_mismatch")
    metadata = {
        "partition": "train",
        "shard_count": 2,
        "ordered_paths": list(TRAIN_SHARD_PATHS),
        "shards": shard_bindings,
        "split_boundary": "contiguous_task_bundle",
        "split_target": "nearest_half_bytes_with_stable_earliest_tie_break",
        "ordered_concat_sha256": _sha256(logical_raw),
        "logical_partition_sha256": LOGICAL_TRAIN_SHA256,
        "logical_partition_bytes": len(logical_raw),
        "logical_partition_records": len(records),
        "max_file_bytes_exclusive": MAX_OUTPUT_FILE_BYTES_EXCLUSIVE,
        "all_shards_below_limit": True,
    }
    return output, records_by_path, metadata


def _logical_identity(
    records: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    eval_raw: bytes,
    serialization_raw: bytes,
    probes_raw: bytes,
) -> dict[str, str]:
    ordered_bundles = sorted({record["task_bundle_sha256"] for record in records})
    record_order_sha256 = _domain_hash(
        "anchor.gemma3-chat-sharded-final-record-order.v1",
        [record["record_id"] for record in records],
    )
    target_inventory_sha256 = _domain_hash(
        "anchor.gemma3-chat-sharded-final-target-inventory.v1",
        [
            {
                "record_id": record["record_id"],
                "target_sha256": record["target"]["output_sha256"],
            }
            for record in records
        ],
    )
    serialization_sha256 = _sha256(serialization_raw)
    probe_sha256 = _sha256(probes_raw)
    eval_sha256 = _sha256(eval_raw)
    logical_dataset_sha256 = _domain_hash(
        "anchor.gemma3-chat-sharded-final-logical-dataset.v1",
        {
            "train_partition_sha256": LOGICAL_TRAIN_SHA256,
            "eval_proxy_partition_sha256": eval_sha256,
            "record_order_sha256": record_order_sha256,
            "target_inventory_sha256": target_inventory_sha256,
            "serialization_inventory_sha256": serialization_sha256,
            "identity_probe_inventory_sha256": probe_sha256,
            "records": len(records),
        },
    )
    return {
        "logical_dataset_sha256": logical_dataset_sha256,
        "logical_train_partition_sha256": LOGICAL_TRAIN_SHA256,
        "eval_proxy_partition_sha256": eval_sha256,
        "logical_record_order_sha256": record_order_sha256,
        "logical_target_inventory_sha256": target_inventory_sha256,
        "logical_train_record_inventory_sha256": (
            _logical_record_inventory_sha256(
                "anchor.gemma3-chat-sharded-final-train-record-inventory.v1",
                train_records,
            )
        ),
        "logical_all_record_inventory_sha256": (
            _logical_record_inventory_sha256(
                "anchor.gemma3-chat-sharded-final-all-record-inventory.v1",
                records,
            )
        ),
        "logical_task_bundle_inventory_sha256": _domain_hash(
            "anchor.gemma3-chat-sharded-final-task-bundle-inventory.v1",
            ordered_bundles,
        ),
        "serialization_inventory_sha256": serialization_sha256,
        "identity_probe_inventory_sha256": probe_sha256,
    }


def _assert_superseded_payload_equivalence(
    state: SourceState,
    output: Mapping[str, bytes],
    records_by_path: Mapping[str, int | None],
) -> None:
    old_manifest = _strict_json(
        state.snapshots["superseded:manifest"].raw,
        "superseded_manifest_equivalence",
    )
    old_files = {item["path"]: item for item in old_manifest["files"]}
    if set(output) != set(old_files):
        _fail("versioned_output_inventory_not_exact_preservation")
    for path in sorted(old_files):
        binding = old_files[path]
        raw = output[path]
        if (
            _sha256(raw) != binding["sha256"]
            or len(raw) != binding["bytes"]
            or records_by_path[path] != binding["records"]
        ):
            _fail("unchanged_payload_or_sidecar_drift")
    logical_raw = b"".join(output[path] for path in TRAIN_SHARD_PATHS)
    if (
        _sha256(logical_raw) != LOGICAL_TRAIN_SHA256
        or len(logical_raw) != LOGICAL_TRAIN_BYTES
        or sum(int(records_by_path[path] or 0) for path in TRAIN_SHARD_PATHS)
        != LOGICAL_TRAIN_RECORDS
    ):
        _fail("superseded_train_logical_identity_drift")


def _input_binding(state: SourceState, value: Snapshot) -> dict[str, Any]:
    try:
        display_path = value.path.relative_to(state.root).as_posix()
    except ValueError as exc:
        raise FinalMaterializationError("input_not_repo_canonical") from exc
    return {
        "label": value.label,
        "path": display_path,
        "sha256": value.sha256,
        "bytes": value.bytes,
        "source_class": value.source_class,
    }


def _component_output_bytes(
    state: SourceState,
) -> tuple[dict[str, bytes], dict[str, int | None]]:
    output: dict[str, bytes] = {}
    records_by_path: dict[str, int | None] = {}
    for component_name in ("router", "tool_eval", "planner_eval"):
        component = state.config["component_candidates"][component_name]
        output_dir = component["output_directory"]
        for pin in component["files"]:
            filename = pin["path"]
            payload_snapshot = state.snapshots[
                f"component:{component_name}:payload:{filename}"
            ]
            relative = f"{output_dir}/{filename}"
            output[relative] = payload_snapshot.raw
            if filename == "records.jsonl":
                records_by_path[relative] = {
                    "router": 100,
                    "tool_eval": 400,
                    "planner_eval": 240,
                }[component_name]
            elif filename == "token_inventory.jsonl":
                records_by_path[relative] = {
                    "router": 100,
                    "tool_eval": 400,
                    "planner_eval": 240,
                }[component_name]
            elif filename == "negative_inventory.jsonl":
                records_by_path[relative] = {
                    "tool_eval": 44,
                    "planner_eval": 53,
                }[component_name]
            else:
                records_by_path[relative] = None
            sidecar_label = f"component:{component_name}:sidecar:{filename}"
            sidecar_relative = relative + ".sha256"
            if sidecar_label in state.snapshots:
                sidecar_raw = state.snapshots[sidecar_label].raw
            else:
                sidecar_raw = _sidecar(filename, payload_snapshot.raw)
            if sidecar_raw != _sidecar(filename, payload_snapshot.raw):
                _fail("component_output_sidecar_mismatch")
            output[sidecar_relative] = sidecar_raw
            records_by_path[sidecar_relative] = None

    review_names = {
        "tool_eval": "tool_eval_independent_review.json",
        "planner_eval": "planner_eval_independent_review.json",
    }
    for component_name, filename in review_names.items():
        snapshot = state.snapshots[f"canonical_review:{component_name}:payload"]
        relative = f"reviews/{filename}"
        output[relative] = snapshot.raw
        records_by_path[relative] = None
        output[relative + ".sha256"] = _sidecar(filename, snapshot.raw)
        records_by_path[relative + ".sha256"] = None
    return output, records_by_path


def _source_bindings(state: SourceState) -> dict[str, Any]:
    families: list[dict[str, Any]] = []
    for family in state.config["work_order_families"]:
        family_index = family["family_index"]
        manifest = state.snapshots[f"family:{family_index}:manifest"]
        partition = state.snapshots[f"family:{family_index}:partition"]
        sidecar = state.snapshots[f"family:{family_index}:sidecar"]
        binding = {
            "family_index": family_index,
            "training_family_id": family["training_family_id"],
            "manifest": {
                "path": f"{family['directory']}/manifest.json",
                "sha256": manifest.sha256,
                "bytes": manifest.bytes,
            },
            "partition": {
                "path": f"{family['directory']}/semantic_blueprints.jsonl",
                "sha256": partition.sha256,
                "bytes": partition.bytes,
                "records": 170,
            },
            "manifest_sidecar": {
                "path": f"{family['directory']}/manifest.json.sha256",
                "sha256": sidecar.sha256,
                "bytes": sidecar.bytes,
            },
        }
        if family_index == 1:
            binding["metadata_repair"] = {
                "mapping_sha256": family["mapping_sha256"],
                "old_candidate_selected": False,
                "independent_review": copy.deepcopy(
                    state.config["family1_repair_independent_review"]
                ),
            }
        families.append(binding)
    return {
        "work_order_families": families,
        "upstream_read_set_files": len(state.upstream_labels),
        "upstream_read_set_inventory_sha256": _inventory_hash(
            state.snapshots[label].sha256 for label in state.upstream_labels
        ),
        "producer_source_files": len(state.producer_labels),
        "producer_source_inventory_sha256": _inventory_hash(
            state.snapshots[label].sha256 for label in state.producer_labels
        ),
        "frozen_contract_files": sum(
            value.source_class == "frozen_contract"
            for value in state.snapshots.values()
        ),
        "all_input_snapshot_files": len(state.snapshots),
        "all_input_snapshot_inventory_sha256": _inventory_hash(
            value.sha256 for value in state.snapshots.values()
        ),
    }


def _component_summary(state: SourceState) -> dict[str, Any]:
    return {
        "router": {
            "namespace": "gemma3_chat_emotion_router_qonly_v1",
            "independent_from_generation_experts": True,
            "not_a_sixth_expert": True,
            "records": 100,
            "train": 80,
            "eval_proxy": 20,
            "en": 50,
            "zh_cn": 50,
            "labels": {
                "humor": 34,
                "serious": 33,
                "angry_style": 33,
            },
            "manifest_sha256": state.snapshots[
                "component:router:payload:manifest.json"
            ].sha256,
            "records_sha256": state.snapshots[
                "component:router:payload:records.jsonl"
            ].sha256,
            "token_inventory_sha256": state.snapshots[
                "component:router:payload:token_inventory.jsonl"
            ].sha256,
            "build_receipt_sha256": state.snapshots[
                "component:router:payload:build_receipt.json"
            ].sha256,
            "single_total_to_branch_then_router_exits": True,
        },
        "tool_eval": {
            "namespace": "gemma3_chat_unbalanced_v2_tool_comparison_eval_v1",
            "records": 400,
            "arm_neutral": True,
            "rows_replicated_by_arms": False,
            "execution_slots": 1200,
            "strata_unchanged": True,
            "language_unchanged": True,
            "negative_inventory_unchanged": True,
            "oracle_unchanged": True,
            "manifest_sha256": state.snapshots[
                "component:tool_eval:payload:manifest.json"
            ].sha256,
            "records_sha256": state.snapshots[
                "component:tool_eval:payload:records.jsonl"
            ].sha256,
            "token_inventory_sha256": state.snapshots[
                "component:tool_eval:payload:token_inventory.jsonl"
            ].sha256,
            "build_receipt_sha256": state.snapshots[
                "component:tool_eval:payload:build_receipt.json"
            ].sha256,
            "independent_review_receipt_sha256": state.snapshots[
                "canonical_review:tool_eval:payload"
            ].sha256,
            "independent_review_status": "passed",
        },
        "planner_eval": {
            "namespace": ("gemma3_chat_unbalanced_v2_planner_comparison_eval_v1"),
            "records": 240,
            "arm_neutral": True,
            "rows_replicated_by_arms": False,
            "execution_slots": 960,
            "strata_unchanged": True,
            "language_unchanged": True,
            "negative_inventory_unchanged": True,
            "oracle_unchanged": True,
            "manifest_sha256": state.snapshots[
                "component:planner_eval:payload:manifest.json"
            ].sha256,
            "records_sha256": state.snapshots[
                "component:planner_eval:payload:records.jsonl"
            ].sha256,
            "token_inventory_sha256": state.snapshots[
                "component:planner_eval:payload:token_inventory.jsonl"
            ].sha256,
            "build_receipt_sha256": state.snapshots[
                "component:planner_eval:payload:build_receipt.json"
            ].sha256,
            "independent_review_receipt_sha256": state.snapshots[
                "canonical_review:planner_eval:payload"
            ].sha256,
            "independent_review_status": "passed",
        },
    }


def _manifest(
    state: SourceState,
    output: Mapping[str, bytes],
    records_by_path: Mapping[str, int | None],
    dataset_summary: Mapping[str, Any],
    training_shards: Mapping[str, Any],
    logical_identity: Mapping[str, str],
) -> dict[str, Any]:
    files = [
        _file_binding(
            path,
            output[path],
            records=records_by_path[path],
        )
        for path in sorted(output)
    ]
    config = state.config
    return {
        "schema_version": (
            "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-manifest.v4"
        ),
        "status": STATUS,
        "namespace": NAMESPACE,
        "artifact_version": ARTIFACT_VERSION,
        "canonical_path": CANONICAL_ARTIFACT_REL,
        "claim_scope": (
            "diagnostic_proxy_only_no_training_formal_live_release_"
            "quality_or_generalization_authority"
        ),
        "superseded_candidate": {
            "canonical_path": SUPERSEDED_ARTIFACT_REL,
            "manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
            "build_receipt_sha256": SUPERSEDED_RECEIPT_SHA256,
            "logical_train_sha256": LOGICAL_TRAIN_SHA256,
            "superseded_reason": (
                "ruff_format_gate_and_atomic_attestation_pair_publish_incomplete"
            ),
            "consumable": False,
        },
        "training_shards": copy.deepcopy(dict(training_shards)),
        "logical_identity": dict(logical_identity),
        "files": files,
        "counts": dict(dataset_summary),
        "identity": {
            "source_pool": "full_five_role_core",
            "rows_added_to_training_4300": 0,
            "identity_bundles": 30,
            "train_bundles": 20,
            "eval_proxy_bundles": 10,
            "en_bundles": 15,
            "zh_cn_bundles": 15,
            "intents": {intent: 6 for intent in IDENTITY_INTENTS},
            "role_records": 150,
            "eval_probe_records": 50,
            "tool_behavior": "direct_answer_no_tool",
            "review_rejects_google_openai": True,
            "wrong_route_fact_invariant": True,
        },
        "review": {
            "records": 1700,
            "pass": 850,
            "fail": 850,
            "fault_types": {fault: 170 for fault in FAULT_TYPES},
            "same_bundle_tool_dependency": True,
            "candidate_projection_hash_bound": True,
            "tool_record_never_mutated_by_review_fixture": True,
        },
        "source_bindings": _source_bindings(state),
        "components": _component_summary(state),
        "adapter_contract": {
            **copy.deepcopy(config["adapter_contract"]),
            "tool_records_arm_neutral": True,
            "planner_records_arm_neutral": True,
            "o_learning_rate_lte_q_over_10_every_step": True,
            "planner_o_only_q_plus_o_o_branch_identity_equal": True,
            "planner_o_only_q_plus_o_o_init_equal": True,
            "planner_o_only_q_plus_o_data_order_steps_seeds_optimizer_equal": True,
            "humor_serious_angry_review_q_only": True,
        },
        "kv_contract": {
            **copy.deepcopy(config["kv_contract"]),
            "commit_sequence": [
                "planner_private_generation",
                "route_plan_validation",
                "immutable_commit",
                "unique_gemma_chat_serialization",
                "frozen_base_adapter_off_reencode",
                "single_shared_prefix_prefill",
                "expert_branch_activation",
                "private_tail_append",
            ],
            "route_plan_commit_hash_bound": True,
            "expert_private_tail": True,
            "planner_private_live_kv_not_transferred": True,
            "persistent_zero_copy_future_proof_required": True,
            "q8_kv_diagnostic_only": True,
        },
        "integrity": {
            "single_authenticated_physical_bytes_snapshot_per_input": True,
            "terminal_toctou_snapshot_recheck_required": True,
            "strict_json_duplicate_and_nonfinite_rejection": True,
            "draft_2020_12_schema_each_record": True,
            "content_validator": True,
            "causal_validator": True,
            "tool_validator": True,
            "review_validator": True,
            "identity_validator": True,
            "cross_field_validator": True,
            "create_once_atomic_directory_publish": True,
            "owned_staging_identity_required": True,
            "failure_attestation_preserved": True,
            "recursive_cleanup_forbidden": True,
            "mandatory_sidecar_for_every_non_sidecar_file": True,
            "checksum_sidecars_are_only_sidecar_exemption": True,
            "per_file_size_below_50_mib": True,
            "all_hash_bound_inputs_explicit_text_eol_lf": True,
            "all_canonical_outputs_explicit_text_eol_lf": True,
            "all_input_preimages_repo_canonical": True,
            "superseded_payload_equivalence_authenticated": True,
            "raw_token_ids_persisted": False,
        },
        "resource_counters": _resources(),
        "claims": _claims(include_physical=True),
        "release_review": {
            "implementer_release_signoff": False,
            "independent_release_review_required": True,
            "fixed_command": (
                "py -3 scripts/research/"
                "audit_gemma3_chat_five_expert_qonly_unbalanced_v2_final.py"
            ),
        },
    }


def _receipt(
    state: SourceState,
    pre_receipt_output: Mapping[str, bytes],
    manifest_raw: bytes,
    training_shards: Mapping[str, Any],
    logical_identity: Mapping[str, str],
) -> dict[str, Any]:
    input_inventory = [
        _input_binding(state, state.snapshots[label])
        for label in sorted(state.snapshots)
    ]
    output_bindings = [
        {
            "path": path,
            "sha256": _sha256(raw),
            "bytes": len(raw),
        }
        for path, raw in sorted(pre_receipt_output.items())
    ]
    return {
        "schema_version": (
            "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-final-build-receipt.v4"
        ),
        "status": STATUS,
        "artifact_version": ARTIFACT_VERSION,
        "manifest": {
            "path": "manifest.json",
            "sha256": _sha256(manifest_raw),
            "bytes": len(manifest_raw),
        },
        "superseded_candidate": {
            "manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
            "build_receipt_sha256": SUPERSEDED_RECEIPT_SHA256,
            "logical_train_sha256": LOGICAL_TRAIN_SHA256,
            "consumable": False,
        },
        "training_shards": {
            key: copy.deepcopy(training_shards[key])
            for key in (
                "shard_count",
                "ordered_paths",
                "ordered_concat_sha256",
                "logical_partition_sha256",
                "logical_partition_bytes",
                "logical_partition_records",
                "max_file_bytes_exclusive",
            )
        },
        "logical_identity": dict(logical_identity),
        "output_inventory_sha256": _domain_hash(
            "anchor.gemma3-chat-final-output-inventory.v1",
            output_bindings,
        ),
        "input_snapshot_inventory": input_inventory,
        "input_snapshot_inventory_sha256": _domain_hash(
            "anchor.gemma3-chat-final-input-snapshot-inventory.v1",
            input_inventory,
        ),
        "upstream_read_set_files": len(state.upstream_labels),
        "producer_source_files": len(state.producer_labels),
        "integrity": {
            "input_snapshot_count": len(state.snapshots),
            "single_read_for_hash_parse_count": True,
            "terminal_byte_and_stat_identity_recheck": True,
            "staged_output_authenticated_before_rename": True,
            "published_output_rechecked_after_rename": True,
            "atomic_directory_rename_no_replace": True,
            "deterministic_generation": True,
            "mandatory_payload_sidecars": True,
            "per_file_size_below_50_mib": True,
            "all_hash_bound_inputs_explicit_text_eol_lf": True,
            "all_canonical_outputs_explicit_text_eol_lf": True,
            "all_input_preimages_repo_canonical": True,
            "superseded_payload_equivalence_authenticated": True,
            "failure_attestation_body_free": True,
            "recursive_cleanup_used": False,
        },
        "official_model_free_audits": {
            "ten_work_order_snapshot_contracts": "passed",
            "family1_v2_metadata_repair_independent_review": "passed",
            "router_snapshot_contract": "passed",
            "tool_eval_snapshot_contract": "passed",
            "tool_eval_external_review": "passed",
            "planner_eval_snapshot_contract": "passed",
            "planner_eval_external_review": "passed",
            "final_schema_content_causal_tool_review_identity_cross_field": ("passed"),
            "independent_release_review": "pending",
        },
        "resource_counters": _resources(),
        "claims": _claims(),
        "implementer_authority": {
            "candidate_ready_allowed": True,
            "release_signoff_allowed": False,
            "commit_allowed": False,
            "push_allowed": False,
            "tag_allowed": False,
            "release_allowed": False,
        },
    }


def _scan_output_bytes(output: Mapping[str, bytes]) -> None:
    total_bytes = 0
    for relative, raw in output.items():
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
            or len(relative) > 220
        ):
            _fail("output_path_invalid")
        if not raw or b"\r" in raw:
            _fail("output_utf8_or_lf_invalid")
        if len(raw) >= MAX_OUTPUT_FILE_BYTES_EXCLUSIVE:
            _fail("output_file_size_limit")
        total_bytes += len(raw)
        if not relative.endswith(".sha256"):
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise FinalMaterializationError("output_utf8_decode_failed") from exc
        for pattern in SECRET_PATTERNS:
            if pattern.search(raw):
                _fail("output_secret_pattern_detected")
    if total_bytes > 536_870_912:
        _fail("output_total_size_limit")
    payloads = {path for path in output if not path.endswith(".sha256")}
    sidecars = {
        path.removesuffix(".sha256") for path in output if path.endswith(".sha256")
    }
    if payloads != sidecars:
        _fail("output_mandatory_sidecar_inventory_mismatch")
    for payload in payloads:
        filename = Path(payload).name
        if output[payload + ".sha256"] != _sidecar(filename, output[payload]):
            _fail("output_sidecar_content_mismatch")


def _prepare_output(
    state: SourceState,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    contexts = _bundle_contexts(state)
    records = _records_from_contexts(contexts)
    serialization_rows = [_serialization_row(record) for record in records]
    probes = _identity_probes(records)
    dataset_summary = _validate_dataset(
        state,
        records,
        serialization_rows,
        probes,
    )
    train_records = [record for record in records if record["split"] == "train"]
    eval_records = [record for record in records if record["split"] == "eval_proxy"]
    output, records_by_path, training_shards = _training_shards(train_records)
    eval_raw = _jsonl_bytes(eval_records)
    serialization_raw = _jsonl_bytes(serialization_rows)
    probes_raw = _jsonl_bytes(probes)
    output.update(
        {
            "eval_proxy/chat.jsonl": eval_raw,
            "serialization_inventory.jsonl": serialization_raw,
            "identity_eval/probe_inventory.jsonl": probes_raw,
        }
    )
    records_by_path.update(
        {
            "eval_proxy/chat.jsonl": 860,
            "serialization_inventory.jsonl": 4300,
            "identity_eval/probe_inventory.jsonl": 50,
        }
    )
    for relative in (
        "eval_proxy/chat.jsonl",
        "serialization_inventory.jsonl",
        "identity_eval/probe_inventory.jsonl",
    ):
        sidecar_relative = relative + ".sha256"
        output[sidecar_relative] = _sidecar(Path(relative).name, output[relative])
        records_by_path[sidecar_relative] = None

    component_output, component_counts = _component_output_bytes(state)
    if set(output) & set(component_output):
        _fail("output_path_collision")
    output.update(component_output)
    records_by_path.update(component_counts)
    _assert_superseded_payload_equivalence(
        state,
        output,
        records_by_path,
    )
    logical_identity = _logical_identity(
        [*train_records, *eval_records],
        train_records,
        eval_raw,
        serialization_raw,
        probes_raw,
    )
    manifest = _manifest(
        state,
        output,
        records_by_path,
        dataset_summary,
        training_shards,
        logical_identity,
    )
    _validate_schema(
        state.validators["manifest"],
        manifest,
        "final_manifest_schema_rejected",
    )
    manifest_raw = _json_bytes(manifest)
    output["manifest.json"] = manifest_raw
    output["manifest.json.sha256"] = _sidecar("manifest.json", manifest_raw)
    records_by_path["manifest.json"] = None
    records_by_path["manifest.json.sha256"] = None

    receipt = _receipt(
        state,
        output,
        manifest_raw,
        training_shards,
        logical_identity,
    )
    _validate_schema(
        state.validators["receipt"],
        receipt,
        "final_receipt_schema_rejected",
    )
    receipt_raw = _json_bytes(receipt)
    output["build_receipt.json"] = receipt_raw
    output["build_receipt.json.sha256"] = _sidecar("build_receipt.json", receipt_raw)
    _require_explicit_lf_paths(
        state.root,
        [f"{CANONICAL_ARTIFACT_REL}/{relative}" for relative in output],
    )
    _scan_output_bytes(output)
    return output, {
        "dataset_summary": dataset_summary,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_raw),
        "build_receipt_sha256": _sha256(receipt_raw),
        "training_shards": training_shards,
        "logical_identity": logical_identity,
        "files": len(output),
        "bytes": sum(len(raw) for raw in output.values()),
    }


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return (int(value.st_dev), int(value.st_ino))


def _assert_physical_directory(path: Path, code: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as exc:
        raise FinalMaterializationError(code) from exc
    if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
        _fail(f"{code}_reparse_or_symlink")
    if not stat.S_ISDIR(value.st_mode):
        _fail(f"{code}_not_directory")
    return value


def _assert_owned_directory(
    path: Path,
    expected_identity: tuple[int, int],
    code: str,
) -> None:
    current = _assert_physical_directory(path, code)
    if _directory_identity(current) != expected_identity:
        _fail(f"{code}_identity_drift")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FinalMaterializationError("directory_fsync_open_failed") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise FinalMaterializationError("directory_fsync_failed") from exc
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as handle:
            written = handle.write(raw)
            if written != len(raw):
                _fail("exclusive_write_short")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise
    except OSError as exc:
        raise FinalMaterializationError("exclusive_write_failed") from exc


def _expected_directories(paths: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for relative in paths:
        parent = Path(relative).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _physical_tree_inventory(
    root: Path,
) -> tuple[set[str], dict[str, tuple[int, int]]]:
    root_stat = _assert_physical_directory(root, "artifact_root")
    root_identity = _directory_identity(root_stat)
    files: set[str] = set()
    directories: dict[str, tuple[int, int]] = {"": root_identity}
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(
                os.scandir(directory),
                key=lambda item: item.name,
            )
        except OSError as exc:
            raise FinalMaterializationError("artifact_tree_scan_failed") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                value = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise FinalMaterializationError("artifact_tree_lstat_failed") from exc
            if entry.is_symlink() or _is_reparse(value):
                _fail("artifact_tree_reparse_or_symlink")
            if stat.S_ISDIR(value.st_mode):
                directories[relative] = _directory_identity(value)
                stack.append(path)
            elif stat.S_ISREG(value.st_mode):
                files.add(relative)
            else:
                _fail("artifact_tree_special_file")
    return files, directories


def _authenticate_output_tree(
    root: Path,
    *,
    root_identity: tuple[int, int],
    expected_bytes: Mapping[str, bytes],
    baseline: Mapping[str, Snapshot] | None,
) -> dict[str, Snapshot]:
    _assert_owned_directory(root, root_identity, "artifact_root")
    observed_files, observed_directories = _physical_tree_inventory(root)
    expected_files = set(expected_bytes)
    expected_directories = _expected_directories(expected_files)
    if observed_files != expected_files:
        _fail("artifact_file_inventory_mismatch")
    if set(observed_directories) != {"", *expected_directories}:
        _fail("artifact_directory_inventory_mismatch")
    snapshots: dict[str, Snapshot] = {}
    for relative in sorted(expected_files):
        value = _snapshot(
            root / Path(relative),
            label=f"artifact:{relative}",
            source_class="artifact_output",
        )
        if value.raw != expected_bytes[relative]:
            _fail("artifact_output_bytes_mismatch")
        if baseline is not None:
            previous = baseline.get(relative)
            if (
                previous is None
                or value.identity != previous.identity
                or value.raw != previous.raw
            ):
                _fail("artifact_output_identity_drift")
        snapshots[relative] = value
    return snapshots


def _write_output_tree(
    staging: Path,
    staging_identity: tuple[int, int],
    output: Mapping[str, bytes],
) -> None:
    owned_directories: dict[str, tuple[int, int]] = {"": staging_identity}
    for relative in sorted(
        _expected_directories(output),
        key=lambda value: (len(Path(value).parts), value),
    ):
        parent_relative = Path(relative).parent.as_posix()
        if parent_relative == ".":
            parent_relative = ""
        parent = staging if not parent_relative else staging / parent_relative
        _assert_owned_directory(
            parent,
            owned_directories[parent_relative],
            "staging_parent",
        )
        directory = staging / Path(relative)
        try:
            directory.mkdir(mode=0o700)
        except OSError as exc:
            raise FinalMaterializationError("staging_directory_create_failed") from exc
        value = _assert_physical_directory(
            directory,
            "staging_directory",
        )
        owned_directories[relative] = _directory_identity(value)

    for relative in sorted(output):
        parent_relative = Path(relative).parent.as_posix()
        if parent_relative == ".":
            parent_relative = ""
        parent = staging if not parent_relative else staging / parent_relative
        _assert_owned_directory(
            parent,
            owned_directories[parent_relative],
            "staging_write_parent",
        )
        _write_exclusive(staging / Path(relative), output[relative])

    for relative in sorted(
        owned_directories,
        key=lambda value: (len(Path(value).parts), value),
        reverse=True,
    ):
        directory = staging if not relative else staging / relative
        _assert_owned_directory(
            directory,
            owned_directories[relative],
            "staging_fsync_directory",
        )
        _fsync_directory(directory)


def _rename_noreplace(
    source: Path,
    destination: Path,
    before_rename: Any,
) -> dict[str, Snapshot]:
    authentication = before_rename()
    if _lexists(destination):
        raise FileExistsError("candidate_output_exists")
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        move_file.restype = ctypes.c_int
        if not move_file(str(source), str(destination)):
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError("candidate_output_exists")
            raise OSError(error, "atomic_directory_move_failed")
        return authentication
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            _fail("atomic_noreplace_unsupported")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == 17:
                raise FileExistsError("candidate_output_exists")
            raise OSError(error, "atomic_directory_move_failed")
        return authentication
    _fail("atomic_noreplace_unsupported")


def _public_failure_code(error: BaseException) -> str:
    if isinstance(error, FinalMaterializationError):
        code = str(error)
        if re.fullmatch(r"[a-z0-9_]{1,96}", code):
            return code
        return "materialization_error"
    if isinstance(error, FileExistsError):
        return "candidate_output_exists"
    if isinstance(error, OSError):
        return "filesystem_error"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "materialization_interrupted"
    return "materialization_internal_error"


def _preserve_failure_attestation(
    staging: Path,
    *,
    staging_identity: tuple[int, int] | None,
    owner_nonce: str,
    state: SourceState,
    error: BaseException,
    explicit_target: bool,
) -> None:
    if staging_identity is None or not _lexists(staging):
        return
    _assert_owned_directory(
        staging,
        staging_identity,
        "failure_attestation_staging",
    )
    attestation = {
        "schema_version": (
            "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-"
            "final-failure-attestation.v1"
        ),
        "status": "unpublished_staging_preserved",
        "error_code": _public_failure_code(error),
        "owner_nonce_sha256": _sha256(owner_nonce.encode("ascii")),
        "target_logical": (
            "explicit_output_override" if explicit_target else CANONICAL_ARTIFACT_REL
        ),
        "input_snapshot_inventory_sha256": _inventory_hash(
            value.sha256 for value in state.snapshots.values()
        ),
        "input_snapshot_files": len(state.snapshots),
        "resource_counters": _resources(),
        "claims": _claims(),
        "recursive_cleanup_used": False,
        "payload_bodies_in_attestation": False,
    }
    _write_exclusive(
        staging / "failure_attestation.json",
        _json_bytes(attestation),
    )
    _fsync_directory(staging)


def _atomic_publish(
    state: SourceState,
    output: Mapping[str, bytes],
    target: Path,
    *,
    explicit_target: bool,
) -> None:
    if _lexists(target):
        raise FileExistsError("candidate_output_exists")
    parent = target.parent
    parent_stat = _assert_physical_directory(
        parent,
        "candidate_parent",
    )
    parent_identity = _directory_identity(parent_stat)
    owner_nonce = secrets.token_hex(24)
    staging = parent / f".{target.name}.staging-{owner_nonce}"
    if _lexists(staging):
        _fail("staging_nonce_collision")
    staging_identity: tuple[int, int] | None = None
    try:
        try:
            staging.mkdir(mode=0o700)
        except OSError as exc:
            raise FinalMaterializationError("staging_root_create_failed") from exc
        staging_stat = _assert_physical_directory(
            staging,
            "staging_root",
        )
        staging_identity = _directory_identity(staging_stat)
        _write_output_tree(staging, staging_identity, output)
        initial = _authenticate_output_tree(
            staging,
            root_identity=staging_identity,
            expected_bytes=output,
            baseline=None,
        )

        def before_rename() -> dict[str, Snapshot]:
            first = _authenticate_output_tree(
                staging,
                root_identity=staging_identity,
                expected_bytes=output,
                baseline=initial,
            )
            for label in sorted(state.snapshots):
                _reverify_snapshot(state.snapshots[label])
            _assert_owned_directory(
                parent,
                parent_identity,
                "candidate_parent",
            )
            return _authenticate_output_tree(
                staging,
                root_identity=staging_identity,
                expected_bytes=output,
                baseline=first,
            )

        authenticated = _rename_noreplace(
            staging,
            target,
            before_rename,
        )
        _fsync_directory(parent)
        _authenticate_output_tree(
            target,
            root_identity=staging_identity,
            expected_bytes=output,
            baseline=authenticated,
        )
    except BaseException as error:
        if staging_identity is not None and _lexists(staging):
            try:
                _preserve_failure_attestation(
                    staging,
                    staging_identity=staging_identity,
                    owner_nonce=owner_nonce,
                    state=state,
                    error=error,
                    explicit_target=explicit_target,
                )
            except BaseException as attestation_error:
                raise error from attestation_error
        raise


def _default_review_path(root: Path, component: str) -> Path:
    return _safe_repo_path(root, CANONICAL_REVIEW_RELS[component])


def _absolute_root(project_root: Path) -> Path:
    root = Path(os.path.abspath(project_root))
    _assert_physical_directory(root, "project_root")
    return root


def _absolute_artifact_path(
    root: Path,
    value: Path | None,
) -> tuple[Path, bool]:
    if value is None:
        return root / Path(CANONICAL_ARTIFACT_REL), False
    path = value if value.is_absolute() else root / value
    return Path(os.path.abspath(path)), True


def build_candidate(
    project_root: Path,
    output_root: Path | None = None,
    tool_review_path: Path | None = None,
    planner_review_path: Path | None = None,
) -> dict[str, Any]:
    root = _absolute_root(project_root)
    target, explicit_target = _absolute_artifact_path(root, output_root)
    if _lexists(target):
        raise FileExistsError("candidate_output_exists")
    tool_review = (
        _default_review_path(root, "tool_eval")
        if tool_review_path is None
        else Path(os.path.abspath(tool_review_path))
    )
    planner_review = (
        _default_review_path(root, "planner_eval")
        if planner_review_path is None
        else Path(os.path.abspath(planner_review_path))
    )
    state = _load_sources(
        root,
        tool_review_path=tool_review,
        planner_review_path=planner_review,
    )
    output, summary = _prepare_output(state)
    _atomic_publish(
        state,
        output,
        target,
        explicit_target=explicit_target,
    )
    return {
        "status": STATUS,
        "artifact_version": ARTIFACT_VERSION,
        "artifact_root": str(target),
        "records": summary["dataset_summary"]["records"],
        "task_bundles": summary["dataset_summary"]["task_bundles"],
        "router_records": 100,
        "tool_eval_records": 400,
        "planner_eval_records": 240,
        "identity_eval_probe_records": 50,
        "files": summary["files"],
        "bytes": summary["bytes"],
        "manifest_sha256": summary["manifest_sha256"],
        "build_receipt_sha256": summary["build_receipt_sha256"],
        "training_shards": summary["training_shards"],
        "logical_identity": summary["logical_identity"],
        "resource_counters": _resources(),
        "claims": _claims(),
        "independent_release_review": "pending",
    }


def _records_by_output_path(state: SourceState) -> dict[str, int | None]:
    counts: dict[str, int | None] = {
        "eval_proxy/chat.jsonl": 860,
        "eval_proxy/chat.jsonl.sha256": None,
        "serialization_inventory.jsonl": 4300,
        "serialization_inventory.jsonl.sha256": None,
        "identity_eval/probe_inventory.jsonl": 50,
        "identity_eval/probe_inventory.jsonl.sha256": None,
    }
    train_records = [
        record
        for record in _records_from_contexts(_bundle_contexts(state))
        if record["split"] == "train"
    ]
    _, shard_counts, _ = _training_shards(train_records)
    counts.update(shard_counts)
    component_counts = {
        "router": {
            "records.jsonl": 100,
            "token_inventory.jsonl": 100,
        },
        "tool_eval": {
            "records.jsonl": 400,
            "token_inventory.jsonl": 400,
            "negative_inventory.jsonl": 44,
        },
        "planner_eval": {
            "records.jsonl": 240,
            "token_inventory.jsonl": 240,
            "negative_inventory.jsonl": 53,
        },
    }
    for component_name in ("router", "tool_eval", "planner_eval"):
        component = state.config["component_candidates"][component_name]
        for pin in component["files"]:
            relative = f"{component['output_directory']}/{pin['path']}"
            counts[relative] = component_counts[component_name].get(pin["path"])
            counts[relative + ".sha256"] = None
    for filename in (
        "tool_eval_independent_review.json",
        "planner_eval_independent_review.json",
    ):
        counts[f"reviews/{filename}"] = None
        counts[f"reviews/{filename}.sha256"] = None
    return counts


def _all_output_paths(state: SourceState) -> set[str]:
    paths = set(_records_by_output_path(state))
    paths.update(
        {
            "manifest.json",
            "manifest.json.sha256",
            "build_receipt.json",
            "build_receipt.json.sha256",
        }
    )
    return paths


def _snapshot_artifact(
    artifact: Path,
    expected_paths: set[str],
) -> tuple[tuple[int, int], dict[str, Snapshot]]:
    root_stat = _assert_physical_directory(
        artifact,
        "audit_artifact_root",
    )
    root_identity = _directory_identity(root_stat)
    observed_files, observed_directories = _physical_tree_inventory(artifact)
    if observed_files != expected_paths:
        _fail("audit_artifact_file_inventory_mismatch")
    if set(observed_directories) != {
        "",
        *_expected_directories(expected_paths),
    }:
        _fail("audit_artifact_directory_inventory_mismatch")
    snapshots = {
        relative: _snapshot(
            artifact / Path(relative),
            label=f"audit_artifact:{relative}",
            source_class="artifact_output",
        )
        for relative in sorted(expected_paths)
    }
    return root_identity, snapshots


def audit_candidate(
    project_root: Path,
    artifact_root: Path | None = None,
    tool_review_path: Path | None = None,
    planner_review_path: Path | None = None,
) -> dict[str, Any]:
    root = _absolute_root(project_root)
    artifact, _ = _absolute_artifact_path(root, artifact_root)
    tool_review = (
        _default_review_path(root, "tool_eval")
        if tool_review_path is None
        else Path(os.path.abspath(tool_review_path))
    )
    planner_review = (
        _default_review_path(root, "planner_eval")
        if planner_review_path is None
        else Path(os.path.abspath(planner_review_path))
    )
    state = _load_sources(
        root,
        tool_review_path=tool_review,
        planner_review_path=planner_review,
    )
    expected_paths = _all_output_paths(state)
    artifact_identity, artifact_snapshots = _snapshot_artifact(
        artifact,
        expected_paths,
    )
    raw = {relative: value.raw for relative, value in artifact_snapshots.items()}
    _scan_output_bytes(raw)

    logical_train_raw = b"".join(raw[path] for path in TRAIN_SHARD_PATHS)
    if (
        _sha256(logical_train_raw) != LOGICAL_TRAIN_SHA256
        or len(logical_train_raw) != LOGICAL_TRAIN_BYTES
    ):
        _fail("audit_logical_train_identity_mismatch")
    train_records, _ = _strict_jsonl(
        logical_train_raw,
        "audit_train_records",
    )
    eval_records, _ = _strict_jsonl(
        raw["eval_proxy/chat.jsonl"],
        "audit_eval_records",
    )
    serialization_rows, _ = _strict_jsonl(
        raw["serialization_inventory.jsonl"],
        "audit_serialization_inventory",
    )
    probes, _ = _strict_jsonl(
        raw["identity_eval/probe_inventory.jsonl"],
        "audit_identity_probes",
    )
    records = train_records + eval_records
    dataset_summary = _validate_dataset(
        state,
        records,
        serialization_rows,
        probes,
    )

    expected_component, _ = _component_output_bytes(state)
    for relative, expected in expected_component.items():
        if raw.get(relative) != expected:
            _fail("audit_component_or_review_copy_mismatch")

    manifest = _strict_json(raw["manifest.json"], "audit_manifest")
    receipt = _strict_json(
        raw["build_receipt.json"],
        "audit_build_receipt",
    )
    _validate_schema(
        state.validators["manifest"],
        manifest,
        "audit_manifest_schema_rejected",
    )
    _validate_schema(
        state.validators["receipt"],
        receipt,
        "audit_receipt_schema_rejected",
    )
    expected_train_output, expected_train_counts, training_shards = _training_shards(
        train_records
    )
    for relative, expected in expected_train_output.items():
        if raw.get(relative) != expected:
            _fail("audit_training_shard_semantic_mismatch")
    logical_identity = _logical_identity(
        records,
        train_records,
        raw["eval_proxy/chat.jsonl"],
        raw["serialization_inventory.jsonl"],
        raw["identity_eval/probe_inventory.jsonl"],
    )
    records_by_path = _records_by_output_path(state)
    records_by_path.update(expected_train_counts)
    pre_manifest_paths = set(records_by_path)
    pre_manifest_output = {relative: raw[relative] for relative in pre_manifest_paths}
    _assert_superseded_payload_equivalence(
        state,
        pre_manifest_output,
        records_by_path,
    )
    expected_manifest = _manifest(
        state,
        pre_manifest_output,
        records_by_path,
        dataset_summary,
        training_shards,
        logical_identity,
    )
    if manifest != expected_manifest:
        _fail("audit_manifest_semantic_mismatch")
    manifest_raw = raw["manifest.json"]
    pre_receipt_output = {
        relative: raw[relative]
        for relative in expected_paths
        if relative not in {"build_receipt.json", "build_receipt.json.sha256"}
    }
    expected_receipt = _receipt(
        state,
        pre_receipt_output,
        manifest_raw,
        training_shards,
        logical_identity,
    )
    if receipt != expected_receipt:
        _fail("audit_receipt_semantic_mismatch")

    for label in sorted(state.snapshots):
        _reverify_snapshot(state.snapshots[label])
    _authenticate_output_tree(
        artifact,
        root_identity=artifact_identity,
        expected_bytes=raw,
        baseline=artifact_snapshots,
    )
    return {
        "status": STATUS,
        "artifact_version": ARTIFACT_VERSION,
        "audit": "passed_candidate_pending_independent_release_review",
        "artifact_root": str(artifact),
        "records": dataset_summary["records"],
        "task_bundles": dataset_summary["task_bundles"],
        "router_records": 100,
        "tool_eval_records": 400,
        "planner_eval_records": 240,
        "identity_eval_probe_records": 50,
        "files": len(raw),
        "bytes": sum(len(value) for value in raw.values()),
        "manifest_sha256": _sha256(manifest_raw),
        "build_receipt_sha256": _sha256(raw["build_receipt.json"]),
        "training_shards": training_shards,
        "logical_identity": logical_identity,
        "source_snapshot_files": len(state.snapshots),
        "upstream_read_set_files": len(state.upstream_labels),
        "resource_counters": _resources(),
        "claims": _claims(),
        "independent_release_review": "pending",
    }


def build_and_audit_candidate(
    project_root: Path,
    output_root: Path | None = None,
    tool_review_path: Path | None = None,
    planner_review_path: Path | None = None,
) -> dict[str, Any]:
    built = build_candidate(
        project_root,
        output_root,
        tool_review_path,
        planner_review_path,
    )
    audited = audit_candidate(
        project_root,
        output_root,
        tool_review_path,
        planner_review_path,
    )
    if (
        built["manifest_sha256"] != audited["manifest_sha256"]
        or built["build_receipt_sha256"] != audited["build_receipt_sha256"]
    ):
        _fail("post_build_audit_identity_mismatch")
    return audited


def _cli_error(error: BaseException) -> int:
    print(
        json.dumps(
            {
                "status": "failed",
                "error_code": _public_failure_code(error),
                "resource_counters": _resources(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 2


def build_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the model-free Gemma 3 Chat unbalanced-v2 FINAL candidate."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tool-review-receipt", type=Path)
    parser.add_argument("--planner-review-receipt", type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = build_candidate(
            arguments.repo_root,
            arguments.output,
            arguments.tool_review_receipt,
            arguments.planner_review_receipt,
        )
    except BaseException as error:
        return _cli_error(error)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def audit_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the model-free Gemma 3 Chat unbalanced-v2 FINAL "
            "candidate without regenerating it."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--tool-review-receipt", type=Path)
    parser.add_argument("--planner-review-receipt", type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = audit_candidate(
            arguments.repo_root,
            arguments.artifact,
            arguments.tool_review_receipt,
            arguments.planner_review_receipt,
        )
    except BaseException as error:
        return _cli_error(error)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0
