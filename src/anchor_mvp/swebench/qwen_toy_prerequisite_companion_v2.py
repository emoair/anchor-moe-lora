"""Authenticate the request-local Qwen trigger as a v1 companion overlay.

The companion reads frozen metadata from the Producer worktree and exact Git
blobs from a fixed Consumer commit.  It never executes Consumer code, opens a
protected sample body, invokes a provider, loads a model, or requests a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any

import yaml


CONFIG_PATH = "configs/research/qwen_toy_prerequisite_companion_v2.json"
CONFIG_VERSION = "anchor.qwen-toy-prerequisite-companion-config.v2"
PRODUCER_ID = "anchor.qwen-toy-prerequisite-companion-producer.v2"
MANIFEST_VERSION = "anchor.qwen-toy-prerequisite-companion-manifest.v2"
STATUS = "trigger_ready_diagnostic_only_inventory_incomplete"
PRODUCER_BASELINE_COMMIT = "744e23f975b13923903f5fabe04c32e74ea25dc4"
PRODUCER_BASELINE_TREE = "90cb962f5341717501fcb16caef13db8922f1cb4"
CONSUMER_RELEASE_COMMIT = "7cb1f7454a76fa3c8c9f46d64da9f11244b51c54"
CONSUMER_RELEASE_TREE = "67ca22bd2f9d50642bf88e484408082abebe2126"
CONSUMER_BASELINE_COMMIT = "b0441e6beaa07b180d7fc69e462b4d2babf21792"
CONSUMER_REF = "refs/remotes/origin/research/neural-swarm-kv"
ARTIFACT_INVENTORY_DOMAIN = (
    "anchor.qwen-toy-prerequisite-companion.consumer-artifacts.v2"
)
ARTIFACT_INVENTORY_ALGORITHM = (
    "sha256_canonical_json_domain_and_ordered_artifacts_utf8_sort_keys_compact_v1"
)
READY_SOURCES = ("swebench_source", "heldout")
MISSING_SOURCES = (
    "gold_partition",
    "partial_gold_export",
    "legacy_heldout_cases",
    "synthetic_scaffold",
)
SOURCE_COPY_RECEIPT = "source/qwen_request_local_trigger_receipt_v2/receipt.json"
SOURCE_COPY_SIDECAR = f"{SOURCE_COPY_RECEIPT}.sha256"

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_BODY_KEYS = frozenset(
    {"answer", "content", "input_ids", "preview", "prompt", "token_ids"}
)

EXPECTED_CONSUMER_ARTIFACTS: tuple[dict[str, object], ...] = (
    {
        "role": "config_schema",
        "path": (
            "configs/research/qwen_request_local_trigger_receipt_v2_config.schema.json"
        ),
        "sha256": ("aa2822a7c4ea60d6148858567920a623959fbe5363a21e2a86d5c855b5d6330f"),
        "bytes": 13710,
    },
    {
        "role": "receipt_schema",
        "path": "configs/research/qwen_request_local_trigger_receipt_v2.schema.json",
        "sha256": ("d5324e8a50dc033850ff4301b7723ad3fc84bcaa312a1ff4eabac43e348a98fc"),
        "bytes": 10873,
    },
    {
        "role": "config",
        "path": "configs/research/qwen_request_local_trigger_receipt_v2.yaml",
        "sha256": ("4425ee2bfcf2a01af0097db242ec7d244b622e3cfb34534628b60653297d7ee4"),
        "bytes": 5519,
    },
    {
        "role": "implementation",
        "path": "src/anchor_mvp/research/qwen_request_local_trigger_receipt_v2.py",
        "sha256": ("dc1fb65441ff67837cb31e9a83df7818d0fb6f4b5b9afd8037423040e59e9f6f"),
        "bytes": 31774,
    },
    {
        "role": "receipt",
        "path": (
            "fixtures/research/qwen_request_local_trigger_receipt_v2/receipt.json"
        ),
        "sha256": ("ae15b7a0edad2527348189fa65532865bdbbe6be0f32757714f2b0fe6582089e"),
        "bytes": 5618,
    },
    {
        "role": "receipt_sidecar",
        "path": (
            "fixtures/research/qwen_request_local_trigger_receipt_v2/"
            "receipt.json.sha256"
        ),
        "sha256": ("ca54594ac067286e5a8619f26870537291fb5e1c5556e8b8efbccc4a67a58a8a"),
        "bytes": 79,
    },
    {
        "role": "tf32_proxy_receipt",
        "path": (
            "fixtures/research/qwen_alora_prefix_kv_diagnostic_tf32_v2/"
            "diagnostic_receipt.json"
        ),
        "sha256": ("278a58441e2f4fb7fc46a67dd5acec3851a7b6937685d90c77873e5972b49cce"),
        "bytes": 3294,
    },
    {
        "role": "tf32_proxy_sidecar",
        "path": (
            "fixtures/research/qwen_alora_prefix_kv_diagnostic_tf32_v2/"
            "diagnostic_receipt.json.sha256"
        ),
        "sha256": ("5e958452e82ca351c06ba8c8cf555cb699841c128526aec219ea97a2460c03be"),
        "bytes": 90,
    },
)

EXPECTED_V1 = {
    "manifest_schema_path": (
        "configs/research/qwen_toy_prerequisite_manifest.schema.json"
    ),
    "manifest_schema_sha256": (
        "b55a0200a3945189687dc0363915e5911bbef41eb6aedcf0cb0f0ceb5bb18e20"
    ),
    "manifest_schema_bytes": 12905,
    "pending_trigger_schema_path": (
        "configs/research/qwen_request_local_trigger_materialization.schema.json"
    ),
    "pending_trigger_schema_sha256": (
        "8a8d97c1ef1513999e215fa63883d476ad7d062e7bcff8274971b2388e9c62e9"
    ),
    "pending_trigger_schema_bytes": 3976,
    "manifest_path": "fixtures/research/qwen_toy_prerequisite_v1/manifest.json",
    "manifest_sha256": (
        "99b94d71639e252c2d768b84a444efa09e844d287c691d8ddfa8312481f2f311"
    ),
    "manifest_bytes": 7910,
    "sidecar_path": ("fixtures/research/qwen_toy_prerequisite_v1/manifest.json.sha256"),
    "sidecar_sha256": (
        "b8a3f7f7bec390da842ef35f8c9942a985051400c8e65857d6ba1a906b23c951"
    ),
    "sidecar_bytes": 80,
    "protected_inventory_set_sha256": (
        "d0bd5702a9c6bbbb1db547b826a94d960a518e1f6ef3e60bfdd25dcd93a3fe22"
    ),
}

EXPECTED_TRIGGER = {
    "tokenizer_binding_sha256": (
        "a76b0f60e5c1e2d92b8a8d9131f9afe9edfda3fcbf0221c4234359f70e806425"
    ),
    "chat_template_sha256": (
        "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
    ),
    "exact_r2_serialization_sha256": (
        "ed6adfcbd0052fdda52a5ab8c52ed04d6e55c7f62493f0d326d4e1b29d55c9f3"
    ),
    "ordered_input_token_ids_sha256": (
        "d989d46116cd50f30d5bba1be48a366e2a04efb8c156550d0f11a532f19121e6"
    ),
    "ordered_input_token_ids_digest_algorithm": (
        "sha256_concat_signed_int64_big_endian_v1"
    ),
    "total_tokens": 44,
    "trigger_span_start": 25,
    "trigger_span_end": 33,
    "trigger_span_width": 8,
    "leading_utf8_bytes": 0,
    "trailing_utf8_bytes": 1,
    "leading_codepoints": 0,
    "trailing_codepoints": 1,
    "trigger_text_sha256": (
        "c963be7f4ed297935fb5fc732292ad62f4fce620a75a6906639bcda0edfdec3c"
    ),
    "trigger_covering_token_ids_sha256": (
        "1d6889128be1b4b84ae22999ffe267a1cc862209b7c38ef3f932a5e69851a412"
    ),
}

EXPECTED_SAFETY = {
    "diagnostic_only": True,
    "formal_training_authorized": False,
    "provider_requests": 0,
    "network_requests": 0,
    "model_loads": 0,
    "gpu_requests": 0,
    "protected_content_reads": 0,
    "consumer_worktree_file_reads": 0,
    "consumer_git_blob_initial_reads": 8,
    "consumer_git_blob_final_recheck_reads": 8,
    "source_materialization_runs": 0,
}


class QwenToyPrerequisiteCompanionError(RuntimeError):
    """A stable, content-free companion producer or audit failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise QwenToyPrerequisiteCompanionError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QwenToyPrerequisiteCompanionError(
            "qwen_toy_companion_canonical_json_invalid"
        ) from exc


def _canonical_document(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sidecar_bytes(digest: str, filename: str) -> bytes:
    if not _SHA_RE.fullmatch(digest) or "/" in filename or "\\" in filename:
        _fail("qwen_toy_companion_sidecar_invalid")
    return f"{digest}  {filename}\n".encode("ascii")


def _strict_json_bytes(data: bytes, code: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite number")

    try:
        text = data.decode("utf-8", errors="strict")
        first = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
        second = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise QwenToyPrerequisiteCompanionError(code) from exc
    if not isinstance(first, Mapping) or first != second:
        _fail(code)
    return first


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _walk_keys(value: object) -> Sequence[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.append(str(key))
            found.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_keys(child))
    return found


def _reject_body_fields(value: object, code: str) -> None:
    if _FORBIDDEN_BODY_KEYS.intersection(_walk_keys(value)):
        _fail(code)


def _validate_relative_path(value: object, code: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "//" in value:
        _fail(code)
    relative = Path(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        _fail(code)
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _has_reparse_attribute(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


@dataclass(frozen=True)
class BytesSnapshot:
    path: Path
    data: bytes
    sha256: str
    size: int
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class GitBlob:
    role: str
    path: str
    data: bytes
    sha256: str
    size: int


@dataclass(frozen=True)
class SourceContext:
    root: Path
    config: Mapping[str, Any]
    local_snapshots: Mapping[Path, BytesSnapshot]
    config_snapshot: BytesSnapshot
    schema_snapshot: BytesSnapshot
    implementation_snapshot: BytesSnapshot
    v1_manifest: Mapping[str, Any]
    source_blobs: tuple[GitBlob, ...]
    receipt: Mapping[str, Any]
    trigger_projection: Mapping[str, Any]


def _reject_reparse_chain(root: Path, path: Path, code: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise QwenToyPrerequisiteCompanionError(code) from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current_stat = current.lstat()
        except OSError as exc:
            raise QwenToyPrerequisiteCompanionError(code) from exc
        if stat.S_ISLNK(current_stat.st_mode) or _has_reparse_attribute(current_stat):
            _fail(code)


def _safe_existing_file(root: Path, relative_value: object, code: str) -> Path:
    relative_text = _validate_relative_path(relative_value, code)
    relative = Path(relative_text)
    path = root.joinpath(*relative.parts)
    _reject_reparse_chain(root, path, code)
    try:
        if path.resolve(strict=True) != path.absolute() or not path.is_file():
            _fail(code)
    except OSError as exc:
        raise QwenToyPrerequisiteCompanionError(code) from exc
    return path


def _snapshot(path: Path, code: str, *, max_bytes: int) -> BytesSnapshot:
    try:
        if path.is_symlink():
            _fail(code)
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or _has_reparse_attribute(before)
                or before.st_size > max_bytes
            ):
                _fail(code)
            data = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except QwenToyPrerequisiteCompanionError:
        raise
    except OSError as exc:
        raise QwenToyPrerequisiteCompanionError(code) from exc
    identity = _stat_identity(before)
    if (
        identity != _stat_identity(after)
        or identity != _stat_identity(current)
        or len(data) != after.st_size
    ):
        _fail(code)
    return BytesSnapshot(path, data, _sha256(data), len(data), identity)


def _verify_unchanged(snapshots: Mapping[Path, BytesSnapshot]) -> None:
    for path, expected in snapshots.items():
        current = _snapshot(
            path,
            "qwen_toy_companion_input_changed",
            max_bytes=max(expected.size, 1),
        )
        if current.sha256 != expected.sha256 or current.size != expected.size:
            _fail("qwen_toy_companion_input_changed")


def _parse_config(snapshot: BytesSnapshot) -> Mapping[str, Any]:
    config = _strict_json_bytes(snapshot.data, "qwen_toy_companion_config_invalid")
    expected = {
        "schema_version": CONFIG_VERSION,
        "producer_id": PRODUCER_ID,
        "producer_baseline_commit": PRODUCER_BASELINE_COMMIT,
        "producer_baseline_tree": PRODUCER_BASELINE_TREE,
        "manifest_schema_path": (
            "configs/research/qwen_toy_prerequisite_companion_v2.schema.json"
        ),
        "implementation_path": (
            "src/anchor_mvp/swebench/qwen_toy_prerequisite_companion_v2.py"
        ),
        "v1_dependency": EXPECTED_V1,
        "consumer_dependency": {
            "repository": "anchor-moe-lora",
            "local_remote_tracking_ref": CONSUMER_REF,
            "consumer_release_commit": CONSUMER_RELEASE_COMMIT,
            "consumer_release_tree": CONSUMER_RELEASE_TREE,
            "consumer_baseline_commit": CONSUMER_BASELINE_COMMIT,
            "consumer_baseline_semantics": ("required_ancestor_or_equal_dependency"),
            "artifacts": [dict(item) for item in EXPECTED_CONSUMER_ARTIFACTS],
        },
        "expected_trigger": EXPECTED_TRIGGER,
        "safety": EXPECTED_SAFETY,
    }
    if config != expected:
        _fail("qwen_toy_companion_config_invalid")
    return config


def _run_git(
    root: Path,
    arguments: Sequence[str],
    code: str,
    *,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise QwenToyPrerequisiteCompanionError(code) from exc
    if result.returncode not in allowed_returncodes:
        _fail(code)
    return result


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _git_ascii(root: Path, arguments: Sequence[str], code: str) -> str:
    result = _run_git(root, arguments, code)
    try:
        value = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise QwenToyPrerequisiteCompanionError(code) from exc
    if not value or b"\x00" in result.stdout:
        _fail(code)
    return value


def _git_commit(root: Path, commit: str, code: str) -> str:
    if not _COMMIT_RE.fullmatch(commit):
        _fail(code)
    resolved = _git_ascii(root, ["rev-parse", "--verify", f"{commit}^{{commit}}"], code)
    if resolved != commit:
        _fail(code)
    return resolved


def _git_tree(root: Path, commit: str, expected: str, code: str) -> None:
    value = _git_ascii(root, ["show", "-s", "--format=%T", commit], code)
    if value != expected:
        _fail(code)


def _git_is_ancestor(root: Path, ancestor: str, descendant: str, code: str) -> None:
    result = _run_git(
        root,
        ["merge-base", "--is-ancestor", ancestor, descendant],
        code,
        allowed_returncodes=frozenset({0, 1}),
    )
    if result.returncode != 0:
        _fail(code)


def _validate_git_repository_controls(root: Path) -> None:
    common_value = _git_ascii(
        root,
        ["rev-parse", "--git-common-dir"],
        "qwen_toy_companion_git_controls_invalid",
    )
    common_input = Path(common_value)
    if not common_input.is_absolute():
        common_input = root / common_input
    requested_common = common_input.absolute()
    try:
        common = requested_common.resolve(strict=True)
        common_stat = common.lstat()
    except OSError as exc:
        raise QwenToyPrerequisiteCompanionError(
            "qwen_toy_companion_git_controls_invalid"
        ) from exc
    if (
        common != requested_common
        or not common.is_dir()
        or stat.S_ISLNK(common_stat.st_mode)
        or _has_reparse_attribute(common_stat)
    ):
        _fail("qwen_toy_companion_git_controls_invalid")
    grafts = common / "info" / "grafts"
    if grafts.exists() or grafts.is_symlink():
        _fail("qwen_toy_companion_git_controls_invalid")
    replace_refs = _run_git(
        root,
        ["for-each-ref", "--format=%(refname)", "refs/replace/"],
        "qwen_toy_companion_git_controls_invalid",
    )
    if replace_refs.stdout.strip():
        _fail("qwen_toy_companion_git_controls_invalid")


def _validate_raw_release_commit(root: Path) -> None:
    result = _run_git(
        root,
        ["cat-file", "commit", CONSUMER_RELEASE_COMMIT],
        "qwen_toy_companion_consumer_commit_invalid",
    )
    header = result.stdout.split(b"\n\n", 1)[0]
    try:
        lines = header.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise QwenToyPrerequisiteCompanionError(
            "qwen_toy_companion_consumer_commit_invalid"
        ) from exc
    trees = [line.removeprefix("tree ") for line in lines if line.startswith("tree ")]
    parents = [
        line.removeprefix("parent ") for line in lines if line.startswith("parent ")
    ]
    if trees != [CONSUMER_RELEASE_TREE] or parents != [CONSUMER_BASELINE_COMMIT]:
        _fail("qwen_toy_companion_consumer_commit_invalid")


def _validate_git_identity(root: Path) -> None:
    _validate_git_repository_controls(root)
    _git_commit(root, PRODUCER_BASELINE_COMMIT, "qwen_toy_companion_git_invalid")
    _git_tree(
        root,
        PRODUCER_BASELINE_COMMIT,
        PRODUCER_BASELINE_TREE,
        "qwen_toy_companion_git_invalid",
    )
    _validate_raw_release_commit(root)
    _git_commit(root, CONSUMER_BASELINE_COMMIT, "qwen_toy_companion_git_invalid")
    _git_commit(root, CONSUMER_RELEASE_COMMIT, "qwen_toy_companion_git_invalid")
    _git_tree(
        root,
        CONSUMER_RELEASE_COMMIT,
        CONSUMER_RELEASE_TREE,
        "qwen_toy_companion_git_invalid",
    )
    _git_is_ancestor(
        root,
        CONSUMER_BASELINE_COMMIT,
        CONSUMER_RELEASE_COMMIT,
        "qwen_toy_companion_consumer_baseline_invalid",
    )
    ref_commit = _git_ascii(
        root,
        ["rev-parse", "--verify", f"{CONSUMER_REF}^{{commit}}"],
        "qwen_toy_companion_consumer_ref_invalid",
    )
    if not _COMMIT_RE.fullmatch(ref_commit):
        _fail("qwen_toy_companion_consumer_ref_invalid")
    _git_is_ancestor(
        root,
        CONSUMER_RELEASE_COMMIT,
        ref_commit,
        "qwen_toy_companion_consumer_ref_invalid",
    )


def _read_git_blobs(root: Path) -> tuple[GitBlob, ...]:
    blobs: list[GitBlob] = []
    for expected in EXPECTED_CONSUMER_ARTIFACTS:
        role = str(expected["role"])
        path = _validate_relative_path(
            expected["path"], "qwen_toy_companion_source_path_invalid"
        )
        result = _run_git(
            root,
            ["cat-file", "blob", f"{CONSUMER_RELEASE_COMMIT}:{path}"],
            "qwen_toy_companion_source_blob_invalid",
        )
        data = result.stdout
        digest = _sha256(data)
        if len(data) != expected["bytes"] or digest != expected["sha256"]:
            _fail("qwen_toy_companion_source_blob_invalid")
        blobs.append(GitBlob(role, path, data, digest, len(data)))
    return tuple(blobs)


def _blob_map(blobs: Sequence[GitBlob]) -> Mapping[str, GitBlob]:
    mapped = {blob.role: blob for blob in blobs}
    if set(mapped) != {str(item["role"]) for item in EXPECTED_CONSUMER_ARTIFACTS}:
        _fail("qwen_toy_companion_source_inventory_invalid")
    return mapped


def _reject_external_refs(value: object, code: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "$ref" and (
                not isinstance(child, str) or not child.startswith("#/")
            ):
                _fail(code)
            _reject_external_refs(child, code)
    elif isinstance(value, list):
        for child in value:
            _reject_external_refs(child, code)


def _validate_schema_instance(
    schema: Mapping[str, Any], instance: object, code: str
) -> None:
    _reject_external_refs(schema, code)
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        if next(validator.iter_errors(instance), None) is not None:
            _fail(code)
    except QwenToyPrerequisiteCompanionError:
        raise
    except Exception as exc:
        raise QwenToyPrerequisiteCompanionError(code) from exc


def _parse_yaml_twice(data: bytes, code: str) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
        first = yaml.safe_load(text)
        second = yaml.safe_load(text)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise QwenToyPrerequisiteCompanionError(code) from exc
    if not isinstance(first, Mapping) or first != second:
        _fail(code)
    return first


def _validate_v1(
    snapshots: Mapping[str, BytesSnapshot], expected_config: Mapping[str, Any]
) -> Mapping[str, Any]:
    for name, suffix in (
        ("manifest_schema", "manifest_schema"),
        ("pending_trigger_schema", "pending_trigger_schema"),
        ("manifest", "manifest"),
        ("sidecar", "sidecar"),
    ):
        snapshot = snapshots[name]
        if (
            snapshot.sha256 != expected_config[f"{suffix}_sha256"]
            or snapshot.size != expected_config[f"{suffix}_bytes"]
        ):
            _fail("qwen_toy_companion_v1_identity_invalid")
    manifest_snapshot = snapshots["manifest"]
    expected_sidecar = _sidecar_bytes(manifest_snapshot.sha256, "manifest.json")
    if snapshots["sidecar"].data != expected_sidecar:
        _fail("qwen_toy_companion_v1_sidecar_invalid")
    manifest = _strict_json_bytes(
        manifest_snapshot.data, "qwen_toy_companion_v1_manifest_invalid"
    )
    if _canonical_document(manifest) != manifest_snapshot.data:
        _fail("qwen_toy_companion_v1_manifest_invalid")
    manifest_schema = _strict_json_bytes(
        snapshots["manifest_schema"].data,
        "qwen_toy_companion_v1_schema_invalid",
    )
    trigger_schema = _strict_json_bytes(
        snapshots["pending_trigger_schema"].data,
        "qwen_toy_companion_v1_schema_invalid",
    )
    _validate_schema_instance(
        manifest_schema, manifest, "qwen_toy_companion_v1_manifest_invalid"
    )
    trigger = _mapping(
        manifest.get("request_local_trigger_binding"),
        "qwen_toy_companion_v1_manifest_invalid",
    )
    _validate_schema_instance(
        trigger_schema, trigger, "qwen_toy_companion_v1_manifest_invalid"
    )
    proof = _mapping(manifest.get("proof"), "qwen_toy_companion_v1_manifest_invalid")
    safety = _mapping(manifest.get("safety"), "qwen_toy_companion_v1_manifest_invalid")
    if (
        manifest.get("status")
        != "toy_generation_verified_protected_inventory_incomplete"
        or proof.get("coverage_ready_count") != 2
        or proof.get("coverage_total") != 6
        or tuple(proof.get("missing_source_classes", ())) != MISSING_SOURCES
        or proof.get("protected_inventory_set_sha256")
        != expected_config["protected_inventory_set_sha256"]
        or proof.get("zero_intersection_claimed") is not False
        or proof.get("v1_attestation_emitted") is not False
        or proof.get("formal_training_authorized") is not False
        or trigger.get("status") != "pending_request_local_materialization"
        or trigger.get("formal_training_authorized") is not False
        or safety.get("formal_training_authorized") is not False
        or safety.get("consumable_by_formal_release") is not False
    ):
        _fail("qwen_toy_companion_v1_manifest_invalid")
    return manifest


def _require_path_sha(
    value: object, expected_path: str, expected_sha: str, code: str
) -> None:
    mapping = _mapping(value, code)
    if mapping != {"path": expected_path, "sha256": expected_sha}:
        _fail(code)


def _validate_source_receipt(
    blobs: Sequence[GitBlob], expected_trigger: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    by_role = _blob_map(blobs)
    config_schema = _strict_json_bytes(
        by_role["config_schema"].data,
        "qwen_toy_companion_source_schema_invalid",
    )
    receipt_schema = _strict_json_bytes(
        by_role["receipt_schema"].data,
        "qwen_toy_companion_source_schema_invalid",
    )
    source_config = _parse_yaml_twice(
        by_role["config"].data, "qwen_toy_companion_source_config_invalid"
    )
    receipt = _strict_json_bytes(
        by_role["receipt"].data, "qwen_toy_companion_source_receipt_invalid"
    )
    if _canonical_document(receipt) != by_role["receipt"].data:
        _fail("qwen_toy_companion_source_receipt_invalid")
    _validate_schema_instance(
        config_schema, source_config, "qwen_toy_companion_source_config_invalid"
    )
    _validate_schema_instance(
        receipt_schema, receipt, "qwen_toy_companion_source_receipt_invalid"
    )
    if by_role["receipt_sidecar"].data != _sidecar_bytes(
        by_role["receipt"].sha256, "receipt.json"
    ):
        _fail("qwen_toy_companion_source_sidecar_invalid")
    if by_role["tf32_proxy_sidecar"].data != _sidecar_bytes(
        by_role["tf32_proxy_receipt"].sha256, "diagnostic_receipt.json"
    ):
        _fail("qwen_toy_companion_source_sidecar_invalid")
    tf32 = _strict_json_bytes(
        by_role["tf32_proxy_receipt"].data,
        "qwen_toy_companion_proxy_receipt_invalid",
    )
    if _canonical_document(tf32) != by_role["tf32_proxy_receipt"].data:
        _fail("qwen_toy_companion_proxy_receipt_invalid")
    _reject_body_fields(receipt, "qwen_toy_companion_source_body_field_invalid")
    _reject_body_fields(tf32, "qwen_toy_companion_source_body_field_invalid")

    if receipt.get("schema_version") != "anchor.qwen-request-local-trigger-receipt.v2":
        _fail("qwen_toy_companion_source_receipt_invalid")
    if receipt.get("status") != "ready_diagnostic_only":
        _fail("qwen_toy_companion_source_receipt_invalid")
    source_bindings = _mapping(
        receipt.get("source_bindings"), "qwen_toy_companion_source_receipt_invalid"
    )
    if source_bindings != {
        "consumer_baseline_commit": CONSUMER_BASELINE_COMMIT,
        "consumer_baseline_semantics": "required_ancestor_or_equal_dependency",
        "producer_commit": PRODUCER_BASELINE_COMMIT,
    }:
        _fail("qwen_toy_companion_source_receipt_invalid")
    producer = _mapping(
        receipt.get("producer"), "qwen_toy_companion_source_receipt_invalid"
    )
    _require_path_sha(
        producer.get("config"),
        by_role["config"].path,
        by_role["config"].sha256,
        "qwen_toy_companion_source_receipt_invalid",
    )
    _require_path_sha(
        producer.get("config_schema"),
        by_role["config_schema"].path,
        by_role["config_schema"].sha256,
        "qwen_toy_companion_source_receipt_invalid",
    )
    _require_path_sha(
        producer.get("receipt_schema"),
        by_role["receipt_schema"].path,
        by_role["receipt_schema"].sha256,
        "qwen_toy_companion_source_receipt_invalid",
    )
    _require_path_sha(
        producer.get("implementation"),
        by_role["implementation"].path,
        by_role["implementation"].sha256,
        "qwen_toy_companion_source_receipt_invalid",
    )
    if (
        producer.get("producer_id")
        != "anchor.qwen-request-local-trigger-materializer.v2"
        or producer.get("canonical_json_policy")
        != "utf8_sort_keys_compact_no_normalization_lf_v1"
        or producer.get("receipt_sha256_sidecar_required") is not True
        or producer.get("receipt_self_sha256_in_body") is not False
    ):
        _fail("qwen_toy_companion_source_receipt_invalid")

    tokenizer_binding = _mapping(
        receipt.get("tokenizer_binding"),
        "qwen_toy_companion_source_receipt_invalid",
    )
    if (
        _sha256(_canonical_bytes(tokenizer_binding))
        != expected_trigger["tokenizer_binding_sha256"]
        or receipt.get("tokenizer_binding_sha256")
        != expected_trigger["tokenizer_binding_sha256"]
        or tokenizer_binding.get("training_authorized") is not False
        or tokenizer_binding.get("model_weight_files_read") is not False
        or tokenizer_binding.get("gguf_files_read") is not False
        or tokenizer_binding.get("network_access") is not False
    ):
        _fail("qwen_toy_companion_tokenizer_binding_invalid")

    r2 = _mapping(
        receipt.get("request2_materialization"),
        "qwen_toy_companion_trigger_projection_invalid",
    )
    span = _mapping(
        r2.get("trigger_span_zero_based_exclusive"),
        "qwen_toy_companion_trigger_projection_invalid",
    )
    overhang = _mapping(
        r2.get("boundary_overhang"),
        "qwen_toy_companion_trigger_projection_invalid",
    )
    checks = {
        "chat_template_sha256": expected_trigger["chat_template_sha256"],
        "exact_r2_serialization_sha256": expected_trigger[
            "exact_r2_serialization_sha256"
        ],
        "ordered_input_token_ids_sha256": expected_trigger[
            "ordered_input_token_ids_sha256"
        ],
        "ordered_input_token_ids_digest_algorithm": expected_trigger[
            "ordered_input_token_ids_digest_algorithm"
        ],
        "total_tokens": expected_trigger["total_tokens"],
        "trigger_span_width": expected_trigger["trigger_span_width"],
        "trigger_text_sha256": expected_trigger["trigger_text_sha256"],
        "trigger_covering_token_ids_sha256": expected_trigger[
            "trigger_covering_token_ids_sha256"
        ],
        "trigger_text_occurrences": 1,
        "trigger_token_sequence_occurrences": 1,
        "complete_r2_tokenization_count": 1,
        "activation_semantics": "next_request_input_activation_only",
        "serialization_scope": (
            "exact_full_chat_templated_request2_bytes_single_tokenization"
        ),
        "isolated_trigger_encoding_authoritative": False,
        "raw_token_ids_emitted": False,
        "global_token_index_emitted": False,
        "planner_request1_private_kv_reused": False,
    }
    if any(r2.get(key) != value for key, value in checks.items()):
        _fail("qwen_toy_companion_trigger_projection_invalid")
    expected_span = {
        "start": expected_trigger["trigger_span_start"],
        "end": expected_trigger["trigger_span_end"],
        "index_base": "zero",
        "end_semantics": "exclusive",
    }
    expected_overhang = {
        "leading_utf8_bytes": expected_trigger["leading_utf8_bytes"],
        "trailing_utf8_bytes": expected_trigger["trailing_utf8_bytes"],
        "leading_codepoints": expected_trigger["leading_codepoints"],
        "trailing_codepoints": expected_trigger["trailing_codepoints"],
    }
    if span != expected_span or overhang != expected_overhang:
        _fail("qwen_toy_companion_trigger_projection_invalid")
    start = expected_trigger["trigger_span_start"]
    end = expected_trigger["trigger_span_end"]
    total = expected_trigger["total_tokens"]
    width = expected_trigger["trigger_span_width"]
    if not (0 <= start < end <= total and end - start == width):
        _fail("qwen_toy_companion_trigger_projection_invalid")

    claims = _mapping(
        receipt.get("claims"), "qwen_toy_companion_source_receipt_invalid"
    )
    if claims != {
        "diagnostic_only": True,
        "formal": False,
        "multistream_claimed": False,
        "numeric_equivalence": False,
        "physical_kv_claimed": False,
        "proxy_signal_passed": True,
        "quality_validated": False,
        "thresholds_formal": False,
        "training_authorized": False,
    }:
        _fail("qwen_toy_companion_source_claim_invalid")
    audit = _mapping(receipt.get("audit"), "qwen_toy_companion_source_receipt_invalid")
    if audit != {
        "canonical_gold_read": False,
        "dataset_reads": 0,
        "gguf_files_read": False,
        "gpu_requests": 0,
        "heldout_content_read": False,
        "model_loads": 0,
        "model_weight_files_read": False,
        "network_requests": 0,
        "provider_requests": 0,
        "scaffold_content_read": False,
    }:
        _fail("qwen_toy_companion_source_audit_invalid")
    proxy_source = _mapping(
        receipt.get("tf32_proxy_source"),
        "qwen_toy_companion_proxy_receipt_invalid",
    )
    if proxy_source != {
        "path": by_role["tf32_proxy_receipt"].path,
        "proxy_signal_passed": True,
        "schema_version": "anchor.qwen-alora-prefix-kv-diagnostic-receipt.v2",
        "sha256": by_role["tf32_proxy_receipt"].sha256,
    }:
        _fail("qwen_toy_companion_proxy_receipt_invalid")
    tf32_claims = _mapping(
        tf32.get("claims"), "qwen_toy_companion_proxy_receipt_invalid"
    )
    if (
        tf32.get("schema_version")
        != "anchor.qwen-alora-prefix-kv-diagnostic-receipt.v2"
        or tf32.get("status") != "passed"
        or tf32_claims.get("diagnostic_only") is not True
        or tf32_claims.get("proxy_signal_passed") is not True
        or any(
            tf32_claims.get(key) is not False
            for key in (
                "formal",
                "full_generation_kv_shared",
                "multi_stream",
                "numeric_equivalence",
                "quality_validated",
                "thresholds_formal",
                "training_authorized",
                "zero_copy",
            )
        )
    ):
        _fail("qwen_toy_companion_proxy_claim_invalid")

    projection = {
        "schema_version": receipt["schema_version"],
        "status": receipt["status"],
        "source_receipt_copy": {
            "path": SOURCE_COPY_RECEIPT,
            "sha256": by_role["receipt"].sha256,
            "bytes": by_role["receipt"].size,
            "sidecar_path": SOURCE_COPY_SIDECAR,
            "sidecar_sha256": by_role["receipt_sidecar"].sha256,
            "sidecar_declared_sha256": by_role["receipt"].sha256,
            "source_bytes_equal_git_blob": True,
        },
        "tokenizer_binding_sha256": receipt["tokenizer_binding_sha256"],
        "chat_template_sha256": r2["chat_template_sha256"],
        "exact_r2_serialization_sha256": r2["exact_r2_serialization_sha256"],
        "ordered_input_token_ids_sha256": r2["ordered_input_token_ids_sha256"],
        "ordered_input_token_ids_digest_algorithm": r2[
            "ordered_input_token_ids_digest_algorithm"
        ],
        "total_tokens": r2["total_tokens"],
        "trigger_span_zero_based_exclusive": dict(span),
        "trigger_span_width": r2["trigger_span_width"],
        "boundary_overhang": dict(overhang),
        "trigger_text_sha256": r2["trigger_text_sha256"],
        "trigger_text_occurrences": r2["trigger_text_occurrences"],
        "trigger_token_sequence_occurrences": r2["trigger_token_sequence_occurrences"],
        "trigger_covering_token_ids_sha256": r2["trigger_covering_token_ids_sha256"],
        "complete_r2_tokenization_count": r2["complete_r2_tokenization_count"],
        "activation_semantics": r2["activation_semantics"],
        "serialization_scope": r2["serialization_scope"],
        "isolated_trigger_encoding_authoritative": r2[
            "isolated_trigger_encoding_authoritative"
        ],
        "raw_token_ids_emitted": r2["raw_token_ids_emitted"],
        "global_token_index_emitted": r2["global_token_index_emitted"],
        "planner_request1_private_kv_reused": r2["planner_request1_private_kv_reused"],
        "source_materialization_reexecuted_by_companion": False,
    }
    return receipt, projection


def _load_context(repo_root: Path, config_path: Path) -> SourceContext:
    supplied_root = Path(repo_root)
    if not supplied_root.is_absolute():
        supplied_root = supplied_root.absolute()
    try:
        root = supplied_root.resolve(strict=True)
        root_stat = root.lstat()
    except OSError as exc:
        raise QwenToyPrerequisiteCompanionError(
            "qwen_toy_companion_root_invalid"
        ) from exc
    if (
        root != supplied_root
        or not root.is_dir()
        or stat.S_ISLNK(root_stat.st_mode)
        or _has_reparse_attribute(root_stat)
    ):
        _fail("qwen_toy_companion_root_invalid")
    supplied = Path(config_path)
    if supplied.is_absolute():
        try:
            config_relative = supplied.resolve(strict=True).relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise QwenToyPrerequisiteCompanionError(
                "qwen_toy_companion_config_invalid"
            ) from exc
    else:
        config_relative = supplied.as_posix()
    if config_relative != CONFIG_PATH:
        _fail("qwen_toy_companion_config_invalid")
    config_file = _safe_existing_file(
        root, CONFIG_PATH, "qwen_toy_companion_config_invalid"
    )
    config_snapshot = _snapshot(
        config_file, "qwen_toy_companion_config_invalid", max_bytes=1_000_000
    )
    config = _parse_config(config_snapshot)

    local_snapshots: dict[Path, BytesSnapshot] = {config_file: config_snapshot}
    schema_file = _safe_existing_file(
        root, config["manifest_schema_path"], "qwen_toy_companion_schema_invalid"
    )
    schema_snapshot = _snapshot(
        schema_file, "qwen_toy_companion_schema_invalid", max_bytes=1_000_000
    )
    implementation_file = _safe_existing_file(
        root,
        config["implementation_path"],
        "qwen_toy_companion_implementation_invalid",
    )
    implementation_snapshot = _snapshot(
        implementation_file,
        "qwen_toy_companion_implementation_invalid",
        max_bytes=1_000_000,
    )
    local_snapshots[schema_file] = schema_snapshot
    local_snapshots[implementation_file] = implementation_snapshot

    manifest_schema = _strict_json_bytes(
        schema_snapshot.data, "qwen_toy_companion_schema_invalid"
    )
    try:
        from jsonschema import Draft202012Validator

        _reject_external_refs(manifest_schema, "qwen_toy_companion_schema_invalid")
        Draft202012Validator.check_schema(manifest_schema)
    except QwenToyPrerequisiteCompanionError:
        raise
    except Exception as exc:
        raise QwenToyPrerequisiteCompanionError(
            "qwen_toy_companion_schema_invalid"
        ) from exc

    v1_config = _mapping(config["v1_dependency"], "qwen_toy_companion_config_invalid")
    v1_snapshots: dict[str, BytesSnapshot] = {}
    for role, path_key, bytes_key in (
        ("manifest_schema", "manifest_schema_path", "manifest_schema_bytes"),
        (
            "pending_trigger_schema",
            "pending_trigger_schema_path",
            "pending_trigger_schema_bytes",
        ),
        ("manifest", "manifest_path", "manifest_bytes"),
        ("sidecar", "sidecar_path", "sidecar_bytes"),
    ):
        path = _safe_existing_file(
            root, v1_config[path_key], "qwen_toy_companion_v1_identity_invalid"
        )
        snapshot = _snapshot(
            path,
            "qwen_toy_companion_v1_identity_invalid",
            max_bytes=int(v1_config[bytes_key]),
        )
        v1_snapshots[role] = snapshot
        local_snapshots[path] = snapshot
    v1_manifest = _validate_v1(v1_snapshots, v1_config)

    _validate_git_identity(root)
    source_blobs = _read_git_blobs(root)
    receipt, trigger_projection = _validate_source_receipt(
        source_blobs,
        _mapping(config["expected_trigger"], "qwen_toy_companion_config_invalid"),
    )
    return SourceContext(
        root=root,
        config=config,
        local_snapshots=local_snapshots,
        config_snapshot=config_snapshot,
        schema_snapshot=schema_snapshot,
        implementation_snapshot=implementation_snapshot,
        v1_manifest=v1_manifest,
        source_blobs=source_blobs,
        receipt=receipt,
        trigger_projection=trigger_projection,
    )


def _final_recheck(context: SourceContext) -> None:
    _verify_unchanged(context.local_snapshots)
    _validate_git_identity(context.root)
    current_blobs = _read_git_blobs(context.root)
    if tuple(
        (item.role, item.path, item.sha256, item.size) for item in current_blobs
    ) != tuple(
        (item.role, item.path, item.sha256, item.size) for item in context.source_blobs
    ):
        _fail("qwen_toy_companion_source_changed")
    if any(
        current.data != expected.data
        for current, expected in zip(current_blobs, context.source_blobs, strict=True)
    ):
        _fail("qwen_toy_companion_source_changed")


def _build_manifest(context: SourceContext) -> Mapping[str, Any]:
    config = context.config
    v1 = _mapping(config["v1_dependency"], "qwen_toy_companion_config_invalid")
    artifacts = [
        {
            "role": blob.role,
            "path": blob.path,
            "sha256": blob.sha256,
            "bytes": blob.size,
        }
        for blob in context.source_blobs
    ]
    inventory_preimage = {
        "domain": ARTIFACT_INVENTORY_DOMAIN,
        "artifacts": artifacts,
    }
    manifest: Mapping[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "status": STATUS,
        "claim_scope": (
            "diagnostic_only_trigger_materialization_overlay_no_training_authority"
        ),
        "overlay_semantics": (
            "non_mutating_conjunctive_overlay_preserve_v1_pending_"
            "add_authenticated_ready_trigger"
        ),
        "producer": {
            "producer_id": PRODUCER_ID,
            "repository": "anchor-moe-lora",
            "baseline_commit": PRODUCER_BASELINE_COMMIT,
            "baseline_tree": PRODUCER_BASELINE_TREE,
            "config": {
                "path": CONFIG_PATH,
                "sha256": context.config_snapshot.sha256,
            },
            "manifest_schema": {
                "path": str(config["manifest_schema_path"]),
                "sha256": context.schema_snapshot.sha256,
            },
            "implementation": {
                "path": str(config["implementation_path"]),
                "sha256": context.implementation_snapshot.sha256,
            },
            "canonical_json_policy": ("utf8_sort_keys_compact_no_normalization_lf_v1"),
            "manifest_sidecar_required": True,
            "single_snapshot": True,
            "same_bytes_reparse": True,
            "final_recheck": True,
            "atomic_publish": True,
        },
        "v1_dependency": {
            "producer_commit": PRODUCER_BASELINE_COMMIT,
            "manifest_schema": {
                "path": str(v1["manifest_schema_path"]),
                "sha256": str(v1["manifest_schema_sha256"]),
                "bytes": int(v1["manifest_schema_bytes"]),
            },
            "pending_trigger_schema": {
                "path": str(v1["pending_trigger_schema_path"]),
                "sha256": str(v1["pending_trigger_schema_sha256"]),
                "bytes": int(v1["pending_trigger_schema_bytes"]),
            },
            "manifest": {
                "path": str(v1["manifest_path"]),
                "sha256": str(v1["manifest_sha256"]),
                "bytes": int(v1["manifest_bytes"]),
            },
            "sidecar": {
                "path": str(v1["sidecar_path"]),
                "sha256": str(v1["sidecar_sha256"]),
                "bytes": int(v1["sidecar_bytes"]),
            },
            "sidecar_declared_manifest_sha256": str(v1["manifest_sha256"]),
            "protected_inventory_set_sha256": str(v1["protected_inventory_set_sha256"]),
            "coverage_ready_count": 2,
            "coverage_total": 6,
            "missing_source_classes": list(MISSING_SOURCES),
            "v1_bytes_modified": False,
        },
        "consumer_dependency": {
            "repository": "anchor-moe-lora",
            "local_remote_tracking_ref": CONSUMER_REF,
            "consumer_release_commit": CONSUMER_RELEASE_COMMIT,
            "consumer_release_tree": CONSUMER_RELEASE_TREE,
            "consumer_baseline_commit": CONSUMER_BASELINE_COMMIT,
            "consumer_baseline_semantics": ("required_ancestor_or_equal_dependency"),
            "baseline_is_ancestor_or_equal": True,
            "release_commit_is_ancestor_of_local_remote_tracking_ref": True,
            "release_commit_signature_claimed": False,
            "live_remote_authenticated": False,
            "source_snapshot_semantics": (
                "exact_git_blob_bytes_at_consumer_release_commit"
            ),
            "consumer_worktree_file_read": False,
            "artifacts": artifacts,
            "artifact_inventory_sha256": _sha256(_canonical_bytes(inventory_preimage)),
            "artifact_inventory_hash_algorithm": ARTIFACT_INVENTORY_ALGORITHM,
            "artifact_inventory_hash_domain": ARTIFACT_INVENTORY_DOMAIN,
        },
        "trigger_materialization": dict(context.trigger_projection),
        "inventory_status": {
            "source": "frozen_v1_dependency",
            "coverage_ready_count": 2,
            "coverage_total": 6,
            "ready_source_classes": list(READY_SOURCES),
            "missing_source_classes": list(MISSING_SOURCES),
            "inventories_modified_by_companion": False,
        },
        "proof": {
            "status": "blocked_incomplete_protected_inventories",
            "zero_intersection_claimed": False,
            "v1_attestation_emitted": False,
            "formal_training_authorized": False,
        },
        "verification": {
            "consumer_release_commit_authenticated": True,
            "consumer_release_raw_parent_and_tree_verified": True,
            "consumer_baseline_ancestry_verified": True,
            "git_environment_overrides_cleared": True,
            "git_replace_and_grafts_absent": True,
            "source_artifact_hashes_match": True,
            "source_receipt_schema_validated": True,
            "source_config_schema_validated": True,
            "source_receipt_sidecar_matches": True,
            "tf32_proxy_sidecar_matches": True,
            "tf32_proxy_receipt_matches_source_binding": True,
            "source_receipt_projection_matches": True,
            "v1_manifest_sidecar_matches": True,
            "span_bounds_valid": True,
            "span_width_valid": True,
            "same_bytes_reparse_passed": True,
            "final_recheck_passed": True,
        },
        "execution": {
            key: value
            for key, value in EXPECTED_SAFETY.items()
            if key not in {"diagnostic_only", "formal_training_authorized"}
        },
        "claims": {
            "diagnostic_only": True,
            "trigger_materialization_ready": True,
            "inventory_complete": False,
            "training_authorized": False,
            "formal": False,
            "numeric_equivalence": False,
            "thresholds_formal": False,
            "quality_validated": False,
            "proxy_signal_passed": True,
            "physical_kv_claimed": False,
            "multistream_claimed": False,
            "zero_copy_claimed": False,
            "full_generation_kv_shared_claimed": False,
        },
    }
    _reject_body_fields(manifest, "qwen_toy_companion_manifest_body_field_invalid")
    schema = _strict_json_bytes(
        context.schema_snapshot.data, "qwen_toy_companion_schema_invalid"
    )
    _validate_schema_instance(schema, manifest, "qwen_toy_companion_manifest_invalid")
    return manifest


def _resolve_output(root: Path, output_root: Path) -> Path:
    output = Path(output_root)
    if not output.is_absolute():
        if any(part in {"", ".", ".."} for part in output.parts):
            _fail("qwen_toy_companion_output_invalid")
        output = root / output
    elif any(part in {"", ".", ".."} for part in output.parts[1:]):
        _fail("qwen_toy_companion_output_invalid")
    output = output.absolute()
    try:
        requested_parent = output.parent.absolute()
        parent = requested_parent.resolve(strict=True)
        parent_stat = parent.lstat()
    except OSError as exc:
        raise QwenToyPrerequisiteCompanionError(
            "qwen_toy_companion_output_invalid"
        ) from exc
    if (
        output == root
        or parent != requested_parent
        or stat.S_ISLNK(parent_stat.st_mode)
        or _has_reparse_attribute(parent_stat)
        or output.exists()
        or output.is_symlink()
    ):
        _fail("qwen_toy_companion_output_invalid")
    return output


def _write_new(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise QwenToyPrerequisiteCompanionError(
            "qwen_toy_companion_output_invalid"
        ) from exc


def _validate_temp_output(
    temp_root: Path,
    manifest: Mapping[str, Any],
    context: SourceContext,
) -> None:
    _validate_artifact_layout(temp_root)
    manifest_bytes = _canonical_document(manifest)
    manifest_snapshot = _snapshot(
        temp_root / "manifest.json",
        "qwen_toy_companion_output_invalid",
        max_bytes=1_000_000,
    )
    sidecar_snapshot = _snapshot(
        temp_root / "manifest.json.sha256",
        "qwen_toy_companion_output_invalid",
        max_bytes=1024,
    )
    by_role = _blob_map(context.source_blobs)
    receipt_snapshot = _snapshot(
        temp_root / SOURCE_COPY_RECEIPT,
        "qwen_toy_companion_output_invalid",
        max_bytes=by_role["receipt"].size,
    )
    receipt_sidecar_snapshot = _snapshot(
        temp_root / SOURCE_COPY_SIDECAR,
        "qwen_toy_companion_output_invalid",
        max_bytes=by_role["receipt_sidecar"].size,
    )
    parsed = _strict_json_bytes(
        manifest_snapshot.data, "qwen_toy_companion_output_invalid"
    )
    if (
        manifest_snapshot.data != manifest_bytes
        or parsed != manifest
        or sidecar_snapshot.data
        != _sidecar_bytes(manifest_snapshot.sha256, "manifest.json")
        or receipt_snapshot.data != by_role["receipt"].data
        or receipt_sidecar_snapshot.data != by_role["receipt_sidecar"].data
    ):
        _fail("qwen_toy_companion_output_invalid")


def build_qwen_toy_prerequisite_companion(
    repo_root: Path, config_path: Path, output_root: Path
) -> Mapping[str, Any]:
    """Build and atomically publish the authenticated companion fixture."""

    context = _load_context(Path(repo_root), Path(config_path))
    manifest = _build_manifest(context)
    output = _resolve_output(context.root, Path(output_root))
    temp_path = Path(
        tempfile.mkdtemp(prefix=".qwen-toy-companion-v2-", dir=output.parent)
    )
    published = False
    try:
        by_role = _blob_map(context.source_blobs)
        manifest_bytes = _canonical_document(manifest)
        _write_new(temp_path / "manifest.json", manifest_bytes)
        _write_new(
            temp_path / "manifest.json.sha256",
            _sidecar_bytes(_sha256(manifest_bytes), "manifest.json"),
        )
        _write_new(temp_path / SOURCE_COPY_RECEIPT, by_role["receipt"].data)
        _write_new(
            temp_path / SOURCE_COPY_SIDECAR,
            by_role["receipt_sidecar"].data,
        )
        _validate_temp_output(temp_path, manifest, context)
        _final_recheck(context)
        _validate_temp_output(temp_path, manifest, context)
        try:
            os.replace(temp_path, output)
        except OSError as exc:
            raise QwenToyPrerequisiteCompanionError(
                "qwen_toy_companion_atomic_publish_failed"
            ) from exc
        published = True
    finally:
        if not published:
            shutil.rmtree(temp_path, ignore_errors=True)
    return manifest


def _artifact_file(root: Path, relative: str, max_bytes: int) -> BytesSnapshot:
    path = _safe_existing_file(root, relative, "qwen_toy_companion_artifact_invalid")
    return _snapshot(path, "qwen_toy_companion_artifact_invalid", max_bytes=max_bytes)


def _validate_artifact_layout(root: Path) -> None:
    expected_files = {
        "manifest.json",
        "manifest.json.sha256",
        SOURCE_COPY_RECEIPT,
        SOURCE_COPY_SIDECAR,
    }
    expected_directories = {
        "source",
        "source/qwen_request_local_trigger_receipt_v2",
    }
    seen_files: set[str] = set()
    try:
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            item_stat = path.lstat()
            if stat.S_ISLNK(item_stat.st_mode) or _has_reparse_attribute(item_stat):
                _fail("qwen_toy_companion_artifact_layout_invalid")
            if stat.S_ISREG(item_stat.st_mode):
                if relative not in expected_files:
                    _fail("qwen_toy_companion_artifact_layout_invalid")
                seen_files.add(relative)
            elif stat.S_ISDIR(item_stat.st_mode):
                if relative not in expected_directories:
                    _fail("qwen_toy_companion_artifact_layout_invalid")
            else:
                _fail("qwen_toy_companion_artifact_layout_invalid")
    except QwenToyPrerequisiteCompanionError:
        raise
    except OSError as exc:
        raise QwenToyPrerequisiteCompanionError(
            "qwen_toy_companion_artifact_layout_invalid"
        ) from exc
    if seen_files != expected_files:
        _fail("qwen_toy_companion_artifact_layout_invalid")


def audit_qwen_toy_prerequisite_companion(
    repo_root: Path, config_path: Path, artifact_root: Path
) -> Mapping[str, Any]:
    """Audit a published companion against all frozen local and Git inputs."""

    context = _load_context(Path(repo_root), Path(config_path))
    expected = _build_manifest(context)
    artifact_input = Path(artifact_root)
    if not artifact_input.is_absolute():
        if any(part in {"", ".", ".."} for part in artifact_input.parts):
            _fail("qwen_toy_companion_artifact_invalid")
        artifact_input = context.root / artifact_input
    elif any(part in {"", ".", ".."} for part in artifact_input.parts[1:]):
        _fail("qwen_toy_companion_artifact_invalid")
    requested_artifact = artifact_input.absolute()
    try:
        artifact = requested_artifact.resolve(strict=True)
        artifact_stat = artifact.lstat()
    except OSError as exc:
        raise QwenToyPrerequisiteCompanionError(
            "qwen_toy_companion_artifact_invalid"
        ) from exc
    if (
        artifact != requested_artifact
        or not artifact.is_dir()
        or stat.S_ISLNK(artifact_stat.st_mode)
        or _has_reparse_attribute(artifact_stat)
    ):
        _fail("qwen_toy_companion_artifact_invalid")
    _validate_artifact_layout(artifact)
    manifest_snapshot = _artifact_file(artifact, "manifest.json", 1_000_000)
    sidecar_snapshot = _artifact_file(artifact, "manifest.json.sha256", 1024)
    receipt_snapshot = _artifact_file(artifact, SOURCE_COPY_RECEIPT, 1_000_000)
    receipt_sidecar_snapshot = _artifact_file(artifact, SOURCE_COPY_SIDECAR, 1024)
    if sidecar_snapshot.data != _sidecar_bytes(
        manifest_snapshot.sha256, "manifest.json"
    ):
        _fail("qwen_toy_companion_artifact_sidecar_invalid")
    manifest = _strict_json_bytes(
        manifest_snapshot.data, "qwen_toy_companion_artifact_manifest_invalid"
    )
    if _canonical_document(manifest) != manifest_snapshot.data or manifest != expected:
        _fail("qwen_toy_companion_artifact_manifest_invalid")
    schema = _strict_json_bytes(
        context.schema_snapshot.data, "qwen_toy_companion_schema_invalid"
    )
    _validate_schema_instance(
        schema, manifest, "qwen_toy_companion_artifact_manifest_invalid"
    )
    by_role = _blob_map(context.source_blobs)
    if (
        receipt_snapshot.data != by_role["receipt"].data
        or receipt_sidecar_snapshot.data != by_role["receipt_sidecar"].data
    ):
        _fail("qwen_toy_companion_artifact_source_copy_invalid")
    _final_recheck(context)
    _verify_unchanged(
        {
            manifest_snapshot.path: manifest_snapshot,
            sidecar_snapshot.path: sidecar_snapshot,
            receipt_snapshot.path: receipt_snapshot,
            receipt_sidecar_snapshot.path: receipt_sidecar_snapshot,
        }
    )
    _validate_artifact_layout(artifact)
    return manifest
