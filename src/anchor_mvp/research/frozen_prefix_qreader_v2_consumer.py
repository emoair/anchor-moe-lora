"""Strict, model-free consumer for the frozen-prefix Q-reader V2 contract.

The consumer authenticates the Producer commit and all 24 added Git blobs,
validates the Producer contracts from those exact byte snapshots, and emits a
non-authorizing dry-run decision. It never reads a dataset record body and
cannot launch a provider, model, GPU job, training job, or release.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
import yaml


CONFIG_VERSION = "anchor.frozen-prefix-qreader-consumer-config.v1"
MANIFEST_VERSION = "anchor.frozen-prefix-qreader-consumer-manifest.v1"
DECISION_VERSION = "anchor.frozen-prefix-qreader-consumer-decision.v1"
CONFIG_PATH = "configs/research/frozen_prefix_qreader_v2_consumer_v1.yaml"
MANIFEST_SCHEMA_PATH = (
    "configs/research/frozen_prefix_qreader_v2_consumer_manifest_v1.schema.json"
)
DECISION_SCHEMA_PATH = (
    "configs/research/frozen_prefix_qreader_v2_consumer_decision_v1.schema.json"
)
FIXTURE_PATH = "fixtures/research/frozen_prefix_qreader_v2_consumer_v1"

CONFIG_SHA256 = "9921297b13017f5776a93073ff08a689a7abbbcc38b43024ef42a3b26819fe79"
MANIFEST_SCHEMA_SHA256 = (
    "4e4a1f20355d0e5513fe2ba2563ee25e9d5f87891313c9d8c52718a4d962d571"
)
DECISION_SCHEMA_SHA256 = (
    "40130cf89b62403434097c89b1bfdf045eb66aa971b00e927eba0cf343463d3d"
)
MANIFEST_SHA256 = "48c76a86a1994e956538191a13decd3590b8721fdd5f0c4c54cfe224a7c0ed2f"
MANIFEST_SIDECAR_PHYSICAL_SHA256 = (
    "da9d626d4e3bc0fc8ad757fccdfcfb97e80f608e51be39c872578c502c629778"
)
PRODUCER_INVENTORY_SHA256 = (
    "d79b6339290e827a3cf4cd921cdaf6daea5a142d8d605fb92999c031d84ab7e1"
)
PRODUCER_COMMIT = "8c9fdfc71b94b5b41d6f3566e9f81baadcc0c267"
PRODUCER_TREE = "0f0c2a25bb338b44b57282d434c09507e7d1ef61"
PRODUCER_PARENT = "524ca359eff128221ef4fa9f5a9e665abf64c7c3"
PRODUCER_TRACKING_REF = (
    "refs/remotes/origin/research/frozen-prefix-qreader-distillation-v2"
)

_ROOT = Path(__file__).resolve().parents[3]
_MAX_LOCAL_BYTES = 2 * 1024 * 1024
_MAX_GIT_BLOB_BYTES = 2 * 1024 * 1024
_REPARSE_POINT = 0x0400
_ROLES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
_STAGE_TO_ROLE = {
    "planner": "planner",
    "tool_policy": "tool_policy",
    "domain_builder": "frontend_gen",
    "domain_review": "frontend_review",
    "security": "security_gate",
}
_ZERO_AUDIT = {
    "provider_requests": 0,
    "network_requests": 0,
    "model_loads": 0,
    "gpu_requests": 0,
    "protected_body_reads": 0,
    "gold_body_reads": 0,
    "heldout_body_reads": 0,
}


class FrozenPrefixQReaderConsumerError(RuntimeError):
    """Stable fail-closed consumer error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise FrozenPrefixQReaderConsumerError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _reject_constant(_: str) -> object:
    _fail("json_non_finite_number")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("json_duplicate_key")
        result[key] = value
    return result


def _load_json(data: bytes, code: str) -> Any:
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except FrozenPrefixQReaderConsumerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenPrefixQReaderConsumerError(code) from exc
    return value


def _mapping(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            _fail("config_duplicate_key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _load_yaml(data: bytes, code: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        value = yaml.load(text, Loader=_UniqueSafeLoader)
    except FrozenPrefixQReaderConsumerError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FrozenPrefixQReaderConsumerError(code) from exc
    return _mapping(value, code)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_link_or_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0)) & _REPARSE_POINT
    )


def _safe_path(root: Path, relative: object, code: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail(code)
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        _fail(code)
    current = root
    if _is_link_or_reparse(current):
        _fail(code)
    for part in candidate.parts:
        current /= part
        if current.exists() and _is_link_or_reparse(current):
            _fail(code)
    try:
        current.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        _fail(code)
    return current


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int]

    def assert_unchanged(self, root: Path, code: str) -> None:
        current = _read_snapshot(root, self.path, code)
        if (
            self.data != current.data
            or self.sha256 != current.sha256
            or self.identity != current.identity
        ):
            _fail(code)


def _read_snapshot(root: Path, path: Path, code: str) -> _Snapshot:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
        if not path.is_file() or _is_link_or_reparse(path):
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size > _MAX_LOCAL_BYTES:
                _fail(code)
            data = handle.read(_MAX_LOCAL_BYTES + 1)
            after = os.fstat(handle.fileno())
        final = path.stat()
    except FrozenPrefixQReaderConsumerError:
        raise
    except OSError as exc:
        raise FrozenPrefixQReaderConsumerError(code) from exc
    identity = _stat_identity(after)
    if (
        len(data) > _MAX_LOCAL_BYTES
        or len(data) != after.st_size
        or _stat_identity(before) != identity
        or _stat_identity(final) != identity
        or _is_link_or_reparse(path)
    ):
        _fail(code)
    return _Snapshot(path, data, _sha256(data), identity)


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "LC_ALL": "C",
        }
    )
    return environment


def _git(
    repo: Path,
    arguments: Sequence[str],
    code: str,
    *,
    max_bytes: int = _MAX_GIT_BLOB_BYTES,
) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            env=_git_environment(),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FrozenPrefixQReaderConsumerError(code) from exc
    if result.returncode != 0 or len(result.stdout) > max_bytes:
        _fail(code)
    return result.stdout


def _assert_clean_git_provenance(repo: Path) -> None:
    git_dir_raw = _git(repo, ["rev-parse", "--absolute-git-dir"], "git_invalid")
    try:
        git_dir = Path(git_dir_raw.decode("utf-8").strip())
    except UnicodeDecodeError as exc:
        raise FrozenPrefixQReaderConsumerError("git_invalid") from exc
    grafts = git_dir / "info" / "grafts"
    if grafts.exists() and grafts.read_bytes().strip():
        _fail("git_grafts_forbidden")
    replace_refs = _git(
        repo,
        ["for-each-ref", "--format=%(refname)", "refs/replace"],
        "git_replace_refs_invalid",
    )
    if replace_refs.strip():
        _fail("git_replace_refs_forbidden")


def _decode_ascii(data: bytes, code: str) -> str:
    try:
        return data.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise FrozenPrefixQReaderConsumerError(code) from exc


def _authenticate_git(
    repo: Path, config: Mapping[str, Any]
) -> tuple[dict[str, bytes], str]:
    _assert_clean_git_provenance(repo)
    producer = _mapping(config.get("producer"), "producer_config_invalid")
    if producer != {
        "branch": "research/frozen-prefix-qreader-distillation-v2",
        "tracking_ref": PRODUCER_TRACKING_REF,
        "commit": PRODUCER_COMMIT,
        "tree": PRODUCER_TREE,
        "parent": PRODUCER_PARENT,
        "exact_added_paths": 24,
    }:
        _fail("producer_config_invalid")
    commit = _decode_ascii(
        _git(
            repo,
            ["rev-parse", "--verify", f"{PRODUCER_COMMIT}^{{commit}}"],
            "producer_commit_unavailable",
        ),
        "producer_commit_invalid",
    )
    tree = _decode_ascii(
        _git(
            repo,
            ["rev-parse", "--verify", f"{PRODUCER_COMMIT}^{{tree}}"],
            "producer_tree_unavailable",
        ),
        "producer_tree_invalid",
    )
    parent = _decode_ascii(
        _git(
            repo,
            ["rev-parse", "--verify", f"{PRODUCER_COMMIT}^"],
            "producer_parent_unavailable",
        ),
        "producer_parent_invalid",
    )
    tracking = _decode_ascii(
        _git(
            repo,
            ["rev-parse", "--verify", PRODUCER_TRACKING_REF],
            "producer_tracking_ref_unavailable",
        ),
        "producer_tracking_ref_invalid",
    )
    if (commit, tree, parent, tracking) != (
        PRODUCER_COMMIT,
        PRODUCER_TREE,
        PRODUCER_PARENT,
        PRODUCER_COMMIT,
    ):
        _fail("producer_provenance_mismatch")

    raw_changes = _git(
        repo,
        [
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            PRODUCER_COMMIT,
        ],
        "producer_diff_invalid",
    )
    try:
        changes = [
            tuple(line.split("\t", 1))
            for line in raw_changes.decode("utf-8").splitlines()
            if line
        ]
    except UnicodeDecodeError as exc:
        raise FrozenPrefixQReaderConsumerError("producer_diff_invalid") from exc
    files = config.get("producer_files")
    if not isinstance(files, list) or len(files) != 24:
        _fail("producer_inventory_invalid")
    expected_paths = [row.get("path") for row in files if isinstance(row, dict)]
    if (
        len(expected_paths) != 24
        or len(set(expected_paths)) != 24
        or changes != [("A", path) for path in expected_paths]
    ):
        _fail("producer_diff_inventory_mismatch")
    if _canonical_sha256(files) != PRODUCER_INVENTORY_SHA256:
        _fail("producer_inventory_digest_mismatch")

    snapshots: dict[str, bytes] = {}
    for row_value in files:
        row = _mapping(row_value, "producer_inventory_invalid")
        path = row.get("path")
        if not isinstance(path, str):
            _fail("producer_inventory_invalid")
        raw_entry = _git(
            repo,
            ["ls-tree", PRODUCER_COMMIT, "--", path],
            "producer_tree_entry_invalid",
        )
        try:
            fields = raw_entry.decode("utf-8").strip().split()
        except UnicodeDecodeError as exc:
            raise FrozenPrefixQReaderConsumerError(
                "producer_tree_entry_invalid"
            ) from exc
        if (
            len(fields) < 4
            or fields[0] != "100644"
            or fields[1] != "blob"
            or fields[2] != row.get("git_blob")
        ):
            _fail("producer_tree_entry_mismatch")
        blob = _git(
            repo,
            ["cat-file", "blob", f"{PRODUCER_COMMIT}:{path}"],
            "producer_blob_unavailable",
        )
        if len(blob) != row.get("bytes") or _sha256(blob) != row.get("sha256"):
            _fail("producer_blob_identity_mismatch")
        snapshots[path] = blob
    return snapshots, _canonical_sha256(files)


def _assert_no_external_schema_refs(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref" and isinstance(item, str):
                if not item.startswith("#"):
                    _fail("schema_external_reference_forbidden")
            _assert_no_external_schema_refs(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_external_schema_refs(item)


def _schema(data: bytes, code: str) -> dict[str, Any]:
    value = _mapping(_load_json(data, code), code)
    _assert_no_external_schema_refs(value)
    try:
        Draft202012Validator.check_schema(value)
    except Exception as exc:
        raise FrozenPrefixQReaderConsumerError(code) from exc
    return value


def _validate(schema: Mapping[str, Any], value: Any, code: str) -> None:
    try:
        Draft202012Validator(schema).validate(value)
    except Exception as exc:
        raise FrozenPrefixQReaderConsumerError(code) from exc


def _get(mapping: Mapping[str, Any], *keys: str, code: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            _fail(code)
        current = current[key]
    return current


def _assert_producer_contracts(blobs: Mapping[str, bytes]) -> int:
    schema_paths = sorted(path for path in blobs if path.endswith(".schema.json"))
    schemas = {
        path: _schema(blobs[path], "producer_schema_invalid") for path in schema_paths
    }
    if len(schemas) != 9:
        _fail("producer_schema_inventory_invalid")

    profile_path = "configs/orchestration/profiles/frozen_prefix_qreader_v2.json"
    profile = _mapping(
        _load_json(blobs[profile_path], "profile_invalid"), "profile_invalid"
    )
    _validate(
        schemas["configs/orchestration/frozen_prefix_qreader_profile.schema.json"],
        profile,
        "profile_schema_invalid",
    )
    overlay_path = "configs/research/frozen_prefix_qreader_release_overlay_v1.json"
    overlay = _mapping(
        _load_json(blobs[overlay_path], "overlay_invalid"), "overlay_invalid"
    )
    materializer_path = "configs/research/swebench_natural_language_scaffold_v2.yaml"
    materializer = _load_yaml(blobs[materializer_path], "materializer_invalid")
    _validate(
        schemas[
            "configs/research/swebench_natural_language_scaffold_v2_config.schema.json"
        ],
        materializer,
        "materializer_schema_invalid",
    )

    post_gold = _mapping(
        _get(profile, "post_gold_pipeline", code="profile_contract_invalid"),
        "profile_contract_invalid",
    )
    if (
        post_gold.get("stage_to_expert") != _STAGE_TO_ROLE
        or post_gold.get("split_group_key") != "task_bundle_sha256"
        or post_gold.get("split_before_augmentation") is not True
        or post_gold.get("primary_scaffold_variant") != "concise_rationale_plus_json"
        or post_gold.get("primary_adapter_label") != "q_only"
        or post_gold.get("diagnostic_adapter_labels") != ["o_only", "q_plus_o"]
        or post_gold.get("adapter_state_on_prefix") != "off"
        or post_gold.get("adapter_state_after_boundary") != "expert_only"
        or post_gold.get("private_tail_kv_required") is not True
        or post_gold.get("expert_private_kv_cross_expert_reuse") is not False
        or post_gold.get("exact_reuse_scope") != "identical_ordered_prefix_lineage_only"
        or post_gold.get("full_generation_kv_shared_claimed") is not False
        or post_gold.get("current_future_forbidden_excluded_before_serialization")
        is not True
        or post_gold.get("token_index_allowed_without_tokenizer_binding") is not False
        or post_gold.get("committed_text_reencoded_by_frozen_base") is not True
        or post_gold.get("source_records_rewritten") is not False
    ):
        _fail("profile_contract_invalid")
    profile_authorization = _mapping(
        _get(profile, "authorization", code="profile_authorization_invalid"),
        "profile_authorization_invalid",
    )
    if any(
        profile_authorization.get(key) is not False
        for key in (
            "live_authorized",
            "training_authorized",
            "formal_training_authorized",
            "release_authorized",
        )
    ):
        _fail("profile_authorization_invalid")
    if any(
        profile_authorization.get(key) != 0
        for key in (
            "provider_requests",
            "model_loads",
            "gpu_requests",
            "network_requests",
        )
    ):
        _fail("profile_resource_audit_invalid")

    architecture = _mapping(
        _get(overlay, "architecture_boundary", code="overlay_contract_invalid"),
        "overlay_contract_invalid",
    )
    if (
        architecture.get("execution_mode")
        != "decoupled_frozen_prefix_producer_required"
        or architecture.get("route_protocol")
        != "two_request_explicit_validate_commit_then_next_request"
        or architecture.get("adapter_state_on_prefix") != "off"
        or architecture.get("adapter_state_after_boundary") != "expert_only"
        or architecture.get("private_tail_kv_required") is not True
        or architecture.get("private_tail_kv_cross_expert_reuse_allowed") is not False
        or architecture.get("exact_reuse_scope")
        != "identical_ordered_prefix_lineage_only"
        or architecture.get("full_generation_kv_shared_claimed") is not False
        or architecture.get("physical_qreader_implemented_claimed") is not False
        or architecture.get("token_level_moe_claimed") is not False
    ):
        _fail("overlay_contract_invalid")
    unresolved = _mapping(
        _get(overlay, "unresolved_gates", code="overlay_gates_invalid"),
        "overlay_gates_invalid",
    )
    if (
        unresolved.get("formal_v3_available") != 0
        or unresolved.get("formal_v3_required") != 5
        or unresolved.get("protected_inventory_available") != 2
        or unresolved.get("protected_inventory_required") != 6
        or any(
            unresolved.get(key) is not False
            for key in (
                "tokenizer_chat_template_binding_authenticated",
                "gemma_runner_binding_authenticated",
                "execution_decision_available",
                "execution_lease_available",
                "data_byte_toctou_lease_available",
                "independent_confirmation_executed",
                "quality_evaluation_complete",
                "performance_evaluation_complete",
            )
        )
    ):
        _fail("overlay_gates_invalid")

    role = _mapping(
        _get(materializer, "role_contract", code="materializer_role_invalid"),
        "materializer_role_invalid",
    )
    visibility = _mapping(
        _get(
            materializer, "visibility_contract", code="materializer_visibility_invalid"
        ),
        "materializer_visibility_invalid",
    )
    route = _mapping(
        _get(
            materializer,
            "route_boundary_contract",
            code="materializer_route_invalid",
        ),
        "materializer_route_invalid",
    )
    cache = _mapping(
        _get(materializer, "cache_contract", code="materializer_cache_invalid"),
        "materializer_cache_invalid",
    )
    adapter = _mapping(
        _get(
            materializer,
            "adapter_control_contract",
            code="materializer_adapter_invalid",
        ),
        "materializer_adapter_invalid",
    )
    safety = _mapping(
        _get(materializer, "safety", code="materializer_safety_invalid"),
        "materializer_safety_invalid",
    )
    if (
        role.get("stage_to_expert") != _STAGE_TO_ROLE
        or role.get("require_all_five_roles_per_bundle") is not True
        or role.get("one_primary_view_per_role") is not True
        or visibility.get("filter_before_serialization") is not True
        or any(
            visibility.get(key) is not False
            for key in (
                "current_target_body_in_prompt",
                "future_block_body_in_prompt",
                "forbidden_block_body_in_prompt",
                "whole_taskboard_stringification_allowed",
                "shared_then_mask_allowed",
            )
        )
        or route.get("semantics") != "explicit_two_request_commit_boundary"
        or route.get("validation_required") is not True
        or route.get("commit_required") is not True
        or route.get("commit_promotes_text_only") is not True
        or route.get("committed_scaffold_reencode_required") is not True
        or route.get("committed_scaffold_reencode_producer") != "frozen_base"
        or route.get("committed_scaffold_reencode_adapter_state") != "off"
        or route.get("expert_activation_request") != "next_request"
        or route.get("token_index_emitted") is not False
        or cache.get("adapter_state_on_shared_prefix") != "off"
        or cache.get("adapter_state_after_boundary") != "expert_only"
        or cache.get("private_tail_kv_required") is not True
        or cache.get("private_tail_append_only") is not True
        or cache.get("private_tail_cross_expert_transfer_allowed") is not False
        or cache.get("exact_reuse_scope") != "identical_ordered_prefix_lineage_only"
        or cache.get("full_generation_kv_shared_claimed") is not False
        or cache.get("physical_kv_tensor_emitted") is not False
        or adapter.get("primary") != "q_only"
        or adapter.get("diagnostic_overlays") != ["o_only", "q_plus_o"]
        or adapter.get("wide_lora_inherited") is not False
        or adapter.get("controls_are_non_authorizing") is not True
        or safety.get("training_authorized") is not False
        or safety.get("formal_training_authorized") is not False
    ):
        _fail("materializer_contract_invalid")
    if any(
        safety.get(key) != 0
        for key in (
            "provider_requests",
            "model_loads",
            "gpu_requests",
            "network_requests",
        )
    ):
        _fail("materializer_resource_audit_invalid")
    return len(schemas)


def _assert_consumer_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_VERSION:
        _fail("consumer_config_version_invalid")
    contract = _mapping(config.get("contract"), "consumer_contract_invalid")
    if (
        tuple(contract.get("exact_roles", ())) != _ROLES
        or contract.get("stage_to_role") != _STAGE_TO_ROLE
        or contract.get("split_group_key") != "task_bundle_sha256"
        or contract.get("split_before_augmentation") is not True
        or contract.get("primary_view") != "concise_rationale_plus_json"
        or contract.get("variants_per_role") != 1
        or contract.get("primary_adapter") != "q_only"
        or contract.get("diagnostic_adapters") != ["o_only", "q_plus_o"]
        or contract.get("filter_before_serialization") is not True
        or any(
            contract.get(key) is not False
            for key in (
                "current_target_body_in_prompt",
                "future_block_body_in_prompt",
                "forbidden_block_body_in_prompt",
                "whole_taskboard_stringification_allowed",
                "shared_then_mask_allowed",
                "private_tail_cross_expert_transfer_allowed",
                "full_generation_kv_shared_claimed",
                "naive_in_stack_q_lora_exact_reuse_claimed",
                "physical_qreader_implemented_claimed",
                "token_level_moe_claimed",
            )
        )
        or contract.get("private_tail_kv_required") is not True
        or contract.get("private_tail_append_only") is not True
    ):
        _fail("consumer_contract_invalid")
    gates = _mapping(config.get("gates"), "consumer_gates_invalid")
    if gates.get("formal_v3_ready_count") != 0 or gates.get("formal_v3_total") != 5:
        _fail("consumer_gates_invalid")
    if (
        gates.get("protected_inventory_ready_count") != 2
        or gates.get("protected_inventory_total") != 6
    ):
        _fail("consumer_gates_invalid")
    for key, value in gates.items():
        if key.endswith(("_available", "_authenticated", "_ready", "_authorized")):
            if value is not False and key != "producer_contract_authenticated":
                _fail("consumer_gates_invalid")
    if gates.get("producer_contract_authenticated") is not True:
        _fail("consumer_gates_invalid")
    if _mapping(config.get("audit"), "consumer_audit_invalid") != _ZERO_AUDIT:
        _fail("consumer_audit_invalid")


def _evaluate(
    root: Path,
    config_path: str | Path,
    expected_config_sha256: str,
    *,
    provenance_repo: Path,
) -> dict[str, Any]:
    config_file = _safe_path(root, str(config_path), "consumer_config_path_invalid")
    config_snapshot = _read_snapshot(root, config_file, "consumer_config_unreadable")
    if config_snapshot.sha256 != expected_config_sha256:
        _fail("consumer_config_sha256_mismatch")
    config = _load_yaml(config_snapshot.data, "consumer_config_invalid")
    _assert_consumer_config(config)

    consumer = _mapping(config.get("consumer"), "consumer_paths_invalid")
    manifest_schema_snapshot = _read_snapshot(
        root,
        _safe_path(
            root, consumer.get("manifest_schema"), "manifest_schema_path_invalid"
        ),
        "manifest_schema_unreadable",
    )
    decision_schema_snapshot = _read_snapshot(
        root,
        _safe_path(
            root, consumer.get("decision_schema"), "decision_schema_path_invalid"
        ),
        "decision_schema_unreadable",
    )
    if manifest_schema_snapshot.sha256 != MANIFEST_SCHEMA_SHA256:
        _fail("manifest_schema_sha256_mismatch")
    if decision_schema_snapshot.sha256 != DECISION_SCHEMA_SHA256:
        _fail("decision_schema_sha256_mismatch")
    manifest_schema = _schema(manifest_schema_snapshot.data, "manifest_schema_invalid")
    decision_schema = _schema(decision_schema_snapshot.data, "decision_schema_invalid")

    fixture = _safe_path(root, consumer.get("fixture"), "fixture_path_invalid")
    manifest_snapshot = _read_snapshot(
        root, fixture / "manifest.json", "manifest_unreadable"
    )
    sidecar_snapshot = _read_snapshot(
        root, fixture / "manifest.json.sha256", "manifest_sidecar_unreadable"
    )
    if manifest_snapshot.sha256 != MANIFEST_SHA256:
        _fail("manifest_sha256_mismatch")
    if sidecar_snapshot.sha256 != MANIFEST_SIDECAR_PHYSICAL_SHA256:
        _fail("manifest_sidecar_physical_sha256_mismatch")
    expected_sidecar = f"{MANIFEST_SHA256}  manifest.json\n".encode("ascii")
    if sidecar_snapshot.data != expected_sidecar:
        _fail("manifest_sidecar_invalid")
    manifest = _mapping(
        _load_json(manifest_snapshot.data, "manifest_invalid"), "manifest_invalid"
    )
    _validate(manifest_schema, manifest, "manifest_schema_validation_failed")
    if manifest.get("consumer_bindings") != {
        "config_sha256": CONFIG_SHA256,
        "manifest_schema_sha256": MANIFEST_SCHEMA_SHA256,
        "decision_schema_sha256": DECISION_SCHEMA_SHA256,
    }:
        _fail("manifest_consumer_bindings_invalid")
    if (
        _get(
            manifest,
            "inventory",
            "ordered_inventory_sha256",
            code="manifest_inventory_invalid",
        )
        != PRODUCER_INVENTORY_SHA256
    ):
        _fail("manifest_inventory_invalid")

    first_blobs, inventory_sha256 = _authenticate_git(provenance_repo, config)
    producer_schema_count = _assert_producer_contracts(first_blobs)

    gates = dict(_mapping(config.get("gates"), "consumer_gates_invalid"))
    gates.pop("producer_contract_authenticated")
    decision: dict[str, Any] = {
        "schema_version": DECISION_VERSION,
        "status": "producer_contract_ready_execution_blocked",
        "claim_scope": "additive_contract_identity_only_non_authorizing",
        "producer_contract_authenticated": True,
        "producer": {
            "commit": PRODUCER_COMMIT,
            "tree": PRODUCER_TREE,
            "files_authenticated": len(first_blobs),
            "inventory_sha256": inventory_sha256,
        },
        "contracts": {
            "role_map_exact": True,
            "bundle_split_exact": True,
            "causal_visibility_exact": True,
            "route_boundary_exact": True,
            "private_tail_exact": True,
            "q_only_primary": True,
            "diagnostic_controls_non_authorizing": True,
            "token_index_not_emitted": True,
            "adapter_off_reencode": True,
            "physical_kv_tensor_not_emitted": True,
            "wide_lora_not_inherited": True,
            "source_records_not_rewritten": True,
        },
        "gates": gates,
        "audit": {
            "single_bytes_snapshot": True,
            "same_bytes_hash_and_parse": True,
            "terminal_toctou_recheck": True,
            "producer_tracking_ref_authenticated": True,
            "producer_git_objects_authenticated": True,
            "json_schemas_validated": producer_schema_count + 2,
            **_ZERO_AUDIT,
        },
    }
    decision["decision_sha256"] = _canonical_sha256(decision)
    _validate(decision_schema, decision, "decision_schema_validation_failed")

    final_blobs, final_inventory_sha256 = _authenticate_git(provenance_repo, config)
    if (
        final_inventory_sha256 != inventory_sha256
        or final_blobs.keys() != first_blobs.keys()
        or any(final_blobs[path] != data for path, data in first_blobs.items())
    ):
        _fail("producer_git_changed")
    for snapshot, code in (
        (config_snapshot, "consumer_config_changed"),
        (manifest_schema_snapshot, "manifest_schema_changed"),
        (decision_schema_snapshot, "decision_schema_changed"),
        (manifest_snapshot, "manifest_changed"),
        (sidecar_snapshot, "manifest_sidecar_changed"),
    ):
        snapshot.assert_unchanged(root, code)
    return decision


def evaluate_frozen_prefix_qreader_v2_consumer(
    config_path: str | Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Authenticate Producer V2 contracts and return a blocked dry-run decision."""

    return _evaluate(
        _ROOT,
        config_path,
        CONFIG_SHA256,
        provenance_repo=_ROOT,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        decision = evaluate_frozen_prefix_qreader_v2_consumer(args.config)
    except FrozenPrefixQReaderConsumerError as exc:
        print(
            json.dumps(
                {
                    "schema_version": DECISION_VERSION,
                    "status": "blocked_invalid_producer_contract",
                    "error": exc.code,
                    "producer_contract_authenticated": False,
                    "training_authorized": False,
                    "formal_training_authorized": False,
                    "release_authorized": False,
                    "live_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
