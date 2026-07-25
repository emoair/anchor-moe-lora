"""Post-hoc Git provenance for the completed Gemma 3 chat diagnostic.

The overlay authenticates a frozen Producer commit, the sixteen raw byte
copies consumed by the training/evaluation pipeline, and the already-published
diagnostic receipts/adapters.  It never parses dataset JSONL records or token
IDs, never loads a model, and cannot authorize training or formal release.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
from typing import Any, Mapping, Sequence


CONFIG_VERSION = "anchor.gemma3-chat-final-producer-git-overlay-config.v1"
ATTESTATION_VERSION = "anchor.gemma3-chat-final-producer-git-attestation.v1"
CONFIG_PATH = "configs/research/gemma3_chat_final_producer_git_overlay_v1.json"
CONFIG_SCHEMA_PATH = (
    "configs/research/gemma3_chat_final_producer_git_overlay_v1.schema.json"
)
ATTESTATION_SCHEMA_PATH = (
    "configs/research/gemma3_chat_final_producer_git_attestation_v1.schema.json"
)
CLI_PATH = "scripts/research/audit_gemma3_chat_final_producer_git_overlay_v1.py"
IMPLEMENTATION_PATH = (
    "src/anchor_mvp/research/gemma3_chat_final_producer_git_overlay_v1.py"
)

# Updated only when the immutable overlay config itself is intentionally frozen.
CONFIG_SHA256 = "925f4fbf4a7a0293bc936fba5763c9dcc1c17e3e5383dcab90d5c711bfc80b83"

_ROOT = Path(__file__).resolve().parents[3]
_REPARSE_POINT = 0x0400
_SMALL_FILE_LIMIT = 8 * 1024 * 1024
_STREAM_CHUNK = 4 * 1024 * 1024
_ROLE_ORDER = ("humor", "serious", "angry_style", "tool_call", "review_audit")
_FALSE_CLAIMS = (
    "changes_training_identity",
    "retroactive_training_authority",
    "training_authorized",
    "formal_training_authorized",
    "formal",
    "quality_validated",
    "source_disjoint_claimed",
    "physical_kv_reuse_claimed",
    "training_reexecuted",
    "evaluation_reexecuted",
)
_ZERO_AUDIT = (
    "provider_requests",
    "network_requests",
    "model_loads",
    "gpu_requests",
    "protected_body_reads",
    "jsonl_records_parsed",
    "token_ids_parsed",
    "luna_requests",
)


class Gemma3ChatProducerGitOverlayError(RuntimeError):
    """Stable fail-closed audit error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise Gemma3ChatProducerGitOverlayError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _strict_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("json_duplicate_key")
        result[key] = value
    return result


def _load_json(data: bytes, code: str) -> Any:
    if not data or b"\r" in data or data.startswith(b"\xef\xbb\xbf"):
        _fail(code)
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda _value: _fail("json_non_finite_number"),
        )
    except Gemma3ChatProducerGitOverlayError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Gemma3ChatProducerGitOverlayError(code) from exc


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(code)
    return value


def _load_mapping(data: bytes, code: str) -> Mapping[str, Any]:
    return _mapping(_load_json(data, code), code)


def _reject_external_refs(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                not isinstance(child, str) or not child.startswith("#/")
            ):
                _fail("schema_external_reference_forbidden")
            _reject_external_refs(child)
    elif isinstance(value, list):
        for child in value:
            _reject_external_refs(child)


def _validator(schema: Mapping[str, Any], code: str) -> Any:
    _reject_external_refs(schema)
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)
    except Gemma3ChatProducerGitOverlayError:
        raise
    except Exception as exc:
        raise Gemma3ChatProducerGitOverlayError(code) from exc


def _validate(validator: Any, value: object, code: str) -> None:
    if sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path)):
        _fail(code)


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


def _assert_no_reparse_ancestry(root: Path, path: Path, code: str) -> None:
    try:
        relative = path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise Gemma3ChatProducerGitOverlayError(code) from exc
    current = root
    if _is_link_or_reparse(current):
        _fail(code)
    for part in relative.parts:
        current /= part
        if current.exists() and _is_link_or_reparse(current):
            _fail(code)


def _safe_relative(root: Path, relative: object, code: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        _fail(code)
    candidate = Path(relative)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        _fail(code)
    path = root.joinpath(*candidate.parts)
    _assert_no_reparse_ancestry(root, path, code)
    return path


@dataclass(frozen=True)
class _SmallSnapshot:
    path: Path
    data: bytes
    sha256: str
    size: int
    identity: tuple[int, int, int, int]

    def assert_unchanged(self, root: Path, code: str) -> None:
        current = _read_small_snapshot(root, self.path, code)
        if (
            current.data != self.data
            or current.sha256 != self.sha256
            or current.identity != self.identity
        ):
            _fail(code)


@dataclass(frozen=True)
class _BlobSnapshot:
    path: Path
    sha256: str
    size: int
    identity: tuple[int, int, int, int]

    def assert_unchanged(self, root: Path, code: str) -> None:
        current = _read_blob_snapshot(root, self.path, self.size, code)
        if current.sha256 != self.sha256 or current.identity != self.identity:
            _fail(code)


def _read_small_snapshot(root: Path, path: Path, code: str) -> _SmallSnapshot:
    _assert_no_reparse_ancestry(root, path, code)
    try:
        if not path.is_file() or _is_link_or_reparse(path):
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size > _SMALL_FILE_LIMIT:
                _fail(code)
            data = handle.read(_SMALL_FILE_LIMIT + 1)
            after = os.fstat(handle.fileno())
        final = path.stat()
    except Gemma3ChatProducerGitOverlayError:
        raise
    except OSError as exc:
        raise Gemma3ChatProducerGitOverlayError(code) from exc
    identity = _stat_identity(after)
    if (
        len(data) > _SMALL_FILE_LIMIT
        or len(data) != after.st_size
        or _stat_identity(before) != identity
        or _stat_identity(final) != identity
        or _is_link_or_reparse(path)
    ):
        _fail(code)
    return _SmallSnapshot(
        path=path,
        data=data,
        sha256=_sha256(data),
        size=len(data),
        identity=identity,
    )


def _read_blob_snapshot(
    root: Path,
    path: Path,
    expected_size: int,
    code: str,
) -> _BlobSnapshot:
    _assert_no_reparse_ancestry(root, path, code)
    digest = hashlib.sha256()
    total = 0
    try:
        if not path.is_file() or _is_link_or_reparse(path):
            _fail(code)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size != expected_size:
                _fail(code)
            while True:
                chunk = handle.read(_STREAM_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_size:
                    _fail(code)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        final = path.stat()
    except Gemma3ChatProducerGitOverlayError:
        raise
    except OSError as exc:
        raise Gemma3ChatProducerGitOverlayError(code) from exc
    identity = _stat_identity(after)
    if (
        total != expected_size
        or _stat_identity(before) != identity
        or _stat_identity(final) != identity
        or _is_link_or_reparse(path)
    ):
        _fail(code)
    return _BlobSnapshot(
        path=path,
        sha256=digest.hexdigest(),
        size=total,
        identity=identity,
    )


def _assert_binding(
    snapshot: _SmallSnapshot | _BlobSnapshot,
    binding: Mapping[str, Any],
    code: str,
) -> None:
    if snapshot.size != binding.get("bytes") or snapshot.sha256 != binding.get(
        "sha256"
    ):
        _fail(code)


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


def _git(repo: Path, arguments: Sequence[str], code: str) -> bytes:
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
        raise Gemma3ChatProducerGitOverlayError(code) from exc
    if result.returncode != 0:
        _fail(code)
    return result.stdout


def _git_optional(repo: Path, arguments: Sequence[str]) -> bytes | None:
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
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _ascii(raw: bytes, code: str) -> str:
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise Gemma3ChatProducerGitOverlayError(code) from exc


def _git_blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest()


def _assert_git_hygiene(repo: Path) -> None:
    git_dir_raw = _git(repo, ["rev-parse", "--absolute-git-dir"], "git_dir_invalid")
    try:
        git_dir = Path(git_dir_raw.decode("utf-8").strip())
    except UnicodeDecodeError as exc:
        raise Gemma3ChatProducerGitOverlayError("git_dir_invalid") from exc
    if _is_link_or_reparse(git_dir):
        _fail("git_dir_reparse_forbidden")
    grafts = git_dir / "info" / "grafts"
    if grafts.exists():
        snapshot = _read_small_snapshot(git_dir, grafts, "git_grafts_invalid")
        if snapshot.data.strip():
            _fail("git_grafts_forbidden")
    replacements = _git(
        repo,
        ["for-each-ref", "--format=%(refname)", "refs/replace"],
        "git_replace_refs_invalid",
    )
    if replacements.strip():
        _fail("git_replace_refs_forbidden")


def _tree_entry(repo: Path, revision: str, path: str) -> tuple[str, str] | None:
    raw = _git(
        repo,
        ["ls-tree", revision, "--", path],
        "producer_tree_entry_invalid",
    )
    if not raw:
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Gemma3ChatProducerGitOverlayError("producer_tree_entry_invalid") from exc
    prefix = "100644 blob "
    suffix = f"\t{path}\n"
    if not decoded.startswith(prefix) or not decoded.endswith(suffix):
        _fail("producer_tree_entry_invalid")
    blob = decoded[len(prefix) : -len(suffix)]
    if len(blob) != 40 or any(char not in "0123456789abcdef" for char in blob):
        _fail("producer_tree_entry_invalid")
    return ("100644", blob)


def _authenticate_git(
    repo: Path,
    producer: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, str | bool | None]]:
    _assert_git_hygiene(repo)
    commit = producer["commit_sha1"]
    tree = producer["tree_sha1"]
    parent = producer["parent_commit_sha1"]
    parent_tree = producer["parent_tree_sha1"]
    if (
        _ascii(
            _git(
                repo,
                ["rev-parse", "--verify", f"{commit}^{{commit}}"],
                "commit_missing",
            ),
            "commit_invalid",
        )
        != commit
    ):
        _fail("commit_mismatch")
    if (
        _ascii(
            _git(repo, ["rev-parse", "--verify", f"{commit}^{{tree}}"], "tree_missing"),
            "tree_invalid",
        )
        != tree
    ):
        _fail("tree_mismatch")
    if (
        _ascii(
            _git(
                repo,
                ["rev-parse", "--verify", f"{parent}^{{commit}}"],
                "parent_missing",
            ),
            "parent_invalid",
        )
        != parent
    ):
        _fail("parent_mismatch")
    if (
        _ascii(
            _git(
                repo,
                ["rev-parse", "--verify", f"{parent}^{{tree}}"],
                "parent_tree_missing",
            ),
            "parent_tree_invalid",
        )
        != parent_tree
    ):
        _fail("parent_tree_mismatch")
    branch_raw = _git_optional(repo, ["rev-parse", "--verify", producer["branch_ref"]])
    tracking_raw = _git_optional(
        repo,
        ["rev-parse", "--verify", producer["local_tracking_ref"]],
    )
    branch_observed = (
        _ascii(branch_raw, "branch_ref_invalid") if branch_raw is not None else None
    )
    tracking_observed = (
        _ascii(tracking_raw, "local_tracking_ref_invalid")
        if tracking_raw is not None
        else None
    )

    raw_commit = _git(repo, ["cat-file", "commit", commit], "commit_object_invalid")
    headers = raw_commit.split(b"\n\n", 1)[0].splitlines()
    tree_headers = [line for line in headers if line.startswith(b"tree ")]
    parent_headers = [line for line in headers if line.startswith(b"parent ")]
    if tree_headers != [f"tree {tree}".encode("ascii")] or parent_headers != [
        f"parent {parent}".encode("ascii")
    ]:
        _fail("commit_lineage_mismatch")

    artifacts = list(producer["artifacts"])
    expected_diff = b"".join(
        f"{row['status']}\t{row['path']}\n".encode("utf-8") for row in artifacts
    )
    actual_diff = _git(
        repo,
        ["diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", commit],
        "producer_diff_invalid",
    )
    if actual_diff != expected_diff:
        _fail("producer_diff_mismatch")
    if (
        len(artifacts) != producer["changed_file_count"]
        or sum(row["status"] == "A" for row in artifacts)
        != producer["added_file_count"]
        or sum(row["status"] == "M" for row in artifacts)
        != producer["modified_file_count"]
        or [row["path"] for row in artifacts]
        != sorted(row["path"] for row in artifacts)
    ):
        _fail("producer_artifact_inventory_invalid")

    blobs: dict[str, bytes] = {}
    for row in artifacts:
        path = row["path"]
        entry = _tree_entry(repo, commit, path)
        if entry != ("100644", row["blob_sha1"]):
            _fail("producer_blob_identity_mismatch")
        raw = _git(
            repo,
            ["cat-file", "blob", row["blob_sha1"]],
            "producer_blob_unavailable",
        )
        if (
            len(raw) != row["bytes"]
            or _sha256(raw) != row["sha256"]
            or _git_blob_sha1(raw) != row["blob_sha1"]
        ):
            _fail("producer_blob_bytes_mismatch")
        parent_entry = _tree_entry(repo, parent, path)
        if row["status"] == "A":
            if row["parent_blob_sha1"] is not None or parent_entry is not None:
                _fail("producer_added_path_parent_mismatch")
        elif parent_entry != ("100644", row["parent_blob_sha1"]):
            _fail("producer_modified_path_parent_mismatch")
        blobs[path] = raw
    return blobs, {
        "branch_ref_observed_sha1": branch_observed,
        "branch_ref_equal_at_attestation": branch_observed == commit,
        "local_tracking_ref_observed_sha1": tracking_observed,
        "local_tracking_ref_equal_at_attestation": tracking_observed == commit,
    }


def _sidecar_bytes(payload: _SmallSnapshot) -> bytes:
    return f"{payload.sha256}  {payload.path.name}\n".encode("ascii")


def _verify_sidecar(
    payload: _SmallSnapshot,
    sidecar: _SmallSnapshot,
    code: str,
) -> None:
    if sidecar.data != _sidecar_bytes(payload):
        _fail(code)


def _validate_config_contract(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_VERSION
        or config.get("claim_scope")
        != "additive_posthoc_git_provenance_non_authorizing"
    ):
        _fail("config_contract_invalid")
    claims = _mapping(config.get("claims"), "config_claims_invalid")
    if (
        claims.get("diagnostic_only") is not True
        or claims.get("posthoc_provenance_only") is not True
        or any(claims.get(key) is not False for key in _FALSE_CLAIMS)
    ):
        _fail("config_claims_invalid")
    audit = _mapping(config.get("resource_audit"), "config_resource_audit_invalid")
    if any(audit.get(key) != 0 for key in _ZERO_AUDIT):
        _fail("config_resource_audit_invalid")
    producer = _mapping(config.get("producer_git"), "producer_config_invalid")
    runtime = config.get("runtime_copies")
    if not isinstance(runtime, list) or len(runtime) != 16:
        _fail("runtime_copy_inventory_invalid")
    producer_artifacts = producer.get("artifacts")
    if not isinstance(producer_artifacts, list) or len(producer_artifacts) != 24:
        _fail("producer_artifact_inventory_invalid")
    artifact_by_path = {
        row["path"]: row
        for row in producer_artifacts
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    if len(artifact_by_path) != 24:
        _fail("producer_artifact_inventory_invalid")
    seen_consumer: set[str] = set()
    for row in runtime:
        copy = _mapping(row, "runtime_copy_inventory_invalid")
        source = artifact_by_path.get(copy.get("producer_path"))
        if source is None or any(
            copy.get(key) != source.get(key) for key in ("blob_sha1", "bytes", "sha256")
        ):
            _fail("runtime_copy_producer_binding_invalid")
        consumer_path = copy.get("consumer_path")
        if not isinstance(consumer_path, str) or consumer_path in seen_consumer:
            _fail("runtime_copy_inventory_invalid")
        seen_consumer.add(consumer_path)
    legacy = _mapping(config.get("legacy_results"), "legacy_config_invalid")
    roles = _mapping(legacy.get("training_run"), "training_run_config_invalid").get(
        "roles"
    )
    if not isinstance(roles, list) or [
        row.get("role") for row in roles if isinstance(row, Mapping)
    ] != list(_ROLE_ORDER):
        _fail("training_role_inventory_invalid")


def _snapshot_binding(
    root: Path,
    binding: Mapping[str, Any],
    code: str,
    *,
    stream: bool = False,
) -> _SmallSnapshot | _BlobSnapshot:
    path = _safe_relative(root, binding.get("path"), f"{code}_path_invalid")
    if stream:
        snapshot = _read_blob_snapshot(root, path, binding["bytes"], code)
    else:
        snapshot = _read_small_snapshot(root, path, code)
    _assert_binding(snapshot, binding, f"{code}_identity_mismatch")
    return snapshot


def _authenticate_runtime_copies(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    git_blobs: Mapping[str, bytes],
) -> tuple[list[dict[str, Any]], list[_SmallSnapshot]]:
    result: list[dict[str, Any]] = []
    snapshots: list[_SmallSnapshot] = []
    for index, row in enumerate(rows):
        path = _safe_relative(
            root,
            row.get("consumer_path"),
            f"runtime_copy_{index}_path_invalid",
        )
        snapshot = _read_small_snapshot(root, path, f"runtime_copy_{index}_unreadable")
        source = git_blobs.get(row.get("producer_path"))
        if (
            source is None
            or snapshot.data != source
            or snapshot.size != row.get("bytes")
            or snapshot.sha256 != row.get("sha256")
            or _git_blob_sha1(source) != row.get("blob_sha1")
        ):
            _fail("runtime_copy_bytes_mismatch")
        snapshots.append(snapshot)
        result.append(dict(row))
    by_name = {
        Path(row["consumer_path"]).name: (row, snap)
        for row, snap in zip(rows, snapshots)
    }
    for sidecar_name, payload_name in (
        ("manifest.json.sha256", "manifest.json"),
        ("build_receipt.json.sha256", "build_receipt.json"),
    ):
        if sidecar_name not in by_name or payload_name not in by_name:
            _fail("runtime_sidecar_inventory_invalid")
        _verify_sidecar(
            by_name[payload_name][1],
            by_name[sidecar_name][1],
            "runtime_sidecar_invalid",
        )
    return result, snapshots


def _authenticate_receipt_metadata(
    config: Mapping[str, Any],
    training_receipt: _SmallSnapshot,
    evaluation_receipt: _SmallSnapshot,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    training = _load_mapping(training_receipt.data, "training_receipt_invalid")
    evaluation = _load_mapping(evaluation_receipt.data, "evaluation_receipt_invalid")
    legacy = _mapping(config["legacy_results"], "legacy_config_invalid")
    train_cfg = _mapping(legacy["training_run"], "training_run_config_invalid")
    eval_cfg = _mapping(legacy["evaluation_run"], "evaluation_run_config_invalid")
    if (
        training.get("schema_version")
        != "anchor.gemma3-1b-it-chat-five-expert-qonly-run-receipt.v1"
        or training.get("status") != "passed_diagnostic_training_only"
        or training.get("run_id") != train_cfg["run_id"]
    ):
        _fail("training_receipt_contract_invalid")
    identity = _mapping(training.get("identity"), "training_identity_invalid")
    if (
        identity.get("config_sha256") != legacy["training_config"]["sha256"]
        or identity.get("implementation_sha256")
        != legacy["consumer_implementation"]["sha256"]
        or identity.get("producer_config_sha256")
        != "2e93ab6b53f68814a4ae017643ddc4c7113716e871b2fa4ef4346358d2269cee"
        or identity.get("producer_implementation_sha256")
        != "98eeac4150a5c51dd422bfe6163e34ce26d691038b497343c20af0a19552fb6f"
        or identity.get("dataset_manifest_sha256")
        != "e71fb5f239d7d1abb94ff50d5db2d57cfd101ccfad3c2c4e34cc128a7e91317b"
        or identity.get("partition_sha256")
        != {
            "eval_proxy/chat.jsonl": (
                "2e1434229482b21ee908cfee8f389523c6730691ade177f47948ad8350821a86"
            ),
            "train/chat.jsonl": (
                "26fd3f5c30d402ab684d67977b8e790470c85f2c33436047d1cdb237c00379dc"
            ),
        }
    ):
        _fail("training_identity_invalid")
    train_claims = _mapping(training.get("claims"), "training_claims_invalid")
    if (
        train_claims.get("diagnostic_only") is not True
        or train_claims.get("training_executed") is not True
        or train_claims.get("training_authorized") is not False
        or train_claims.get("formal_training_authorized") is not False
        or train_claims.get("formal") is not False
        or train_claims.get("quality_claimed") is not False
    ):
        _fail("training_claims_invalid")
    train_audit = _mapping(training.get("audit"), "training_audit_invalid")
    if any(
        train_audit.get(key) != 0
        for key in ("network_requests", "protected_body_reads", "provider_requests")
    ):
        _fail("training_audit_invalid")

    if (
        evaluation.get("schema_version")
        != "anchor.gemma3-chat-five-expert-qonly-generation-eval-receipt.v1"
        or evaluation.get("status") != "passed_diagnostic_eval_proxy_generation_only"
        or evaluation.get("run_id") != eval_cfg["run_id"]
        or evaluation.get("source_training_run_id") != train_cfg["run_id"]
    ):
        _fail("evaluation_receipt_contract_invalid")
    source_hashes = _mapping(
        evaluation.get("source_hashes"), "evaluation_source_hashes_invalid"
    )
    if source_hashes != {
        "consumer_config_sha256": legacy["training_config"]["sha256"],
        "consumer_implementation_sha256": legacy["consumer_implementation"]["sha256"],
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "partition_sha256": identity["partition_sha256"],
        "training_run_receipt_sha256": train_cfg["receipt"]["sha256"],
    }:
        _fail("evaluation_source_hashes_invalid")
    eval_claims = _mapping(evaluation.get("claims"), "evaluation_claims_invalid")
    if (
        eval_claims.get("diagnostic_only") is not True
        or eval_claims.get("eval_proxy_is_heldout") is not False
        or eval_claims.get("formal") is not False
        or eval_claims.get("quality_validated") is not False
    ):
        _fail("evaluation_claims_invalid")
    eval_audit = _mapping(evaluation.get("audit"), "evaluation_audit_invalid")
    if any(
        eval_audit.get(key) != 0
        for key in (
            "generated_bodies_persisted",
            "heldout_reads",
            "network_requests",
            "per_record_metrics_persisted",
            "protected_body_reads",
            "provider_requests",
            "raw_token_ids_persisted",
        )
    ):
        _fail("evaluation_audit_invalid")
    evaluation_spec = _mapping(evaluation.get("evaluation"), "evaluation_spec_invalid")
    if (
        evaluation_spec.get("total_generations") != 75
        or evaluation_spec.get("total_unique_records") != 25
        or evaluation_spec.get("records_per_role") != 5
    ):
        _fail("evaluation_spec_invalid")
    return training, evaluation


def _authenticate_legacy_results(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[_SmallSnapshot], list[_BlobSnapshot]]:
    legacy = _mapping(config["legacy_results"], "legacy_config_invalid")
    small: list[_SmallSnapshot] = []
    large: list[_BlobSnapshot] = []
    training_config = _snapshot_binding(
        root,
        _mapping(legacy["training_config"], "training_config_binding_invalid"),
        "training_config",
    )
    consumer_impl = _snapshot_binding(
        root,
        _mapping(
            legacy["consumer_implementation"],
            "consumer_implementation_binding_invalid",
        ),
        "consumer_implementation",
    )
    assert isinstance(training_config, _SmallSnapshot)
    assert isinstance(consumer_impl, _SmallSnapshot)
    small.extend((training_config, consumer_impl))

    train_cfg = _mapping(legacy["training_run"], "training_run_config_invalid")
    eval_cfg = _mapping(legacy["evaluation_run"], "evaluation_run_config_invalid")
    train_receipt = _snapshot_binding(
        root,
        _mapping(train_cfg["receipt"], "training_receipt_binding_invalid"),
        "training_receipt",
    )
    train_sidecar = _snapshot_binding(
        root,
        _mapping(
            train_cfg["receipt_sidecar"],
            "training_receipt_sidecar_binding_invalid",
        ),
        "training_receipt_sidecar",
    )
    eval_receipt = _snapshot_binding(
        root,
        _mapping(eval_cfg["receipt"], "evaluation_receipt_binding_invalid"),
        "evaluation_receipt",
    )
    eval_sidecar = _snapshot_binding(
        root,
        _mapping(
            eval_cfg["receipt_sidecar"],
            "evaluation_receipt_sidecar_binding_invalid",
        ),
        "evaluation_receipt_sidecar",
    )
    assert isinstance(train_receipt, _SmallSnapshot)
    assert isinstance(train_sidecar, _SmallSnapshot)
    assert isinstance(eval_receipt, _SmallSnapshot)
    assert isinstance(eval_sidecar, _SmallSnapshot)
    _verify_sidecar(train_receipt, train_sidecar, "training_receipt_sidecar_invalid")
    _verify_sidecar(eval_receipt, eval_sidecar, "evaluation_receipt_sidecar_invalid")
    small.extend((train_receipt, train_sidecar, eval_receipt, eval_sidecar))
    training, _evaluation = _authenticate_receipt_metadata(
        config, train_receipt, eval_receipt
    )
    receipt_roles = training.get("roles")
    if not isinstance(receipt_roles, list) or [
        row.get("role") for row in receipt_roles if isinstance(row, Mapping)
    ] != list(_ROLE_ORDER):
        _fail("training_receipt_roles_invalid")
    receipt_role_map = {row["role"]: row for row in receipt_roles}

    adapter_inventory: list[dict[str, Any]] = []
    for role_cfg_raw in train_cfg["roles"]:
        role_cfg = _mapping(role_cfg_raw, "role_config_invalid")
        role = role_cfg["role"]
        role_receipt = _mapping(receipt_role_map[role], "training_role_receipt_invalid")
        role_summary: dict[str, Any] = {"role": role}
        for phase in ("smoke", "full"):
            payload_key = f"{phase}_receipt"
            sidecar_key = f"{phase}_receipt_sidecar"
            payload = _snapshot_binding(
                root,
                _mapping(role_cfg[payload_key], f"{payload_key}_binding_invalid"),
                f"{role}_{payload_key}",
            )
            sidecar = _snapshot_binding(
                root,
                _mapping(role_cfg[sidecar_key], f"{sidecar_key}_binding_invalid"),
                f"{role}_{sidecar_key}",
            )
            assert isinstance(payload, _SmallSnapshot)
            assert isinstance(sidecar, _SmallSnapshot)
            _verify_sidecar(payload, sidecar, f"{role}_{phase}_sidecar_invalid")
            if role_receipt.get(f"{phase}_receipt_sha256") != payload.sha256:
                _fail(f"{role}_{phase}_receipt_cross_binding_invalid")
            small.extend((payload, sidecar))
            role_summary[f"{phase}_receipt_sha256"] = payload.sha256
        adapter_config = _snapshot_binding(
            root,
            _mapping(role_cfg["adapter_config"], "adapter_config_binding_invalid"),
            f"{role}_adapter_config",
        )
        adapter_model = _snapshot_binding(
            root,
            _mapping(role_cfg["adapter_model"], "adapter_model_binding_invalid"),
            f"{role}_adapter_model",
            stream=True,
        )
        assert isinstance(adapter_config, _SmallSnapshot)
        assert isinstance(adapter_model, _BlobSnapshot)
        adapter_receipt = _mapping(
            role_receipt.get("adapter_artifact_sha256"),
            "adapter_receipt_binding_invalid",
        )
        if adapter_receipt != {
            "adapter_config.json": adapter_config.sha256,
            "adapter_model.safetensors": adapter_model.sha256,
        }:
            _fail("adapter_receipt_binding_invalid")
        small.append(adapter_config)
        large.append(adapter_model)
        role_summary.update(
            {
                "adapter_config_sha256": adapter_config.sha256,
                "adapter_model_sha256": adapter_model.sha256,
                "adapter_model_bytes": adapter_model.size,
            }
        )
        adapter_inventory.append(role_summary)

    result = {
        "training_config_sha256": training_config.sha256,
        "consumer_implementation_sha256": consumer_impl.sha256,
        "training_run_id": train_cfg["run_id"],
        "training_receipt_sha256": train_receipt.sha256,
        "training_receipt_sidecar_physical_sha256": train_sidecar.sha256,
        "evaluation_run_id": eval_cfg["run_id"],
        "evaluation_receipt_sha256": eval_receipt.sha256,
        "evaluation_receipt_sidecar_physical_sha256": eval_sidecar.sha256,
        "roles_authenticated": 5,
        "phase_receipts_authenticated": 10,
        "phase_sidecars_authenticated": 10,
        "adapter_configs_authenticated": 5,
        "adapter_models_authenticated": 5,
        "adapter_inventory_sha256": _canonical_sha256(adapter_inventory),
    }
    return result, small, large


def _ensure_output_parent(path: Path) -> None:
    missing: list[Path] = []
    current = path.parent
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            _fail("output_parent_invalid")
        current = current.parent
    if _is_link_or_reparse(current):
        _fail("output_parent_reparse_forbidden")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except OSError as exc:
            raise Gemma3ChatProducerGitOverlayError("output_parent_invalid") from exc
        if _is_link_or_reparse(directory):
            _fail("output_parent_reparse_forbidden")


def _write_exclusive(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise Gemma3ChatProducerGitOverlayError("output_write_failed") from exc


def _atomic_publish(output_dir: Path, attestation_bytes: bytes) -> dict[str, str]:
    if output_dir.exists():
        _fail("output_already_exists")
    _ensure_output_parent(output_dir)
    staging = output_dir.parent / (
        f".{output_dir.name}.staging-{os.getpid()}-{secrets.token_hex(6)}"
    )
    if staging.exists():
        _fail("output_staging_exists")
    try:
        staging.mkdir()
    except OSError as exc:
        raise Gemma3ChatProducerGitOverlayError("output_staging_create_failed") from exc
    attestation_path = staging / "attestation.json"
    sidecar_path = staging / "attestation.json.sha256"
    attestation_sha = _sha256(attestation_bytes)
    sidecar_bytes = f"{attestation_sha}  attestation.json\n".encode("ascii")
    _write_exclusive(attestation_path, attestation_bytes)
    _write_exclusive(sidecar_path, sidecar_bytes)
    attestation_snapshot = _read_small_snapshot(
        staging, attestation_path, "output_attestation_invalid"
    )
    sidecar_snapshot = _read_small_snapshot(
        staging, sidecar_path, "output_sidecar_invalid"
    )
    if (
        attestation_snapshot.data != attestation_bytes
        or sidecar_snapshot.data != sidecar_bytes
    ):
        _fail("output_bytes_mismatch")
    try:
        staging.rename(output_dir)
    except OSError as exc:
        raise Gemma3ChatProducerGitOverlayError("output_publish_failed") from exc
    if not output_dir.is_dir() or _is_link_or_reparse(output_dir):
        _fail("output_publish_invalid")
    return {
        "attestation_file_sha256": attestation_sha,
        "attestation_sidecar_physical_sha256": _sha256(sidecar_bytes),
    }


def _evaluate(
    root: Path,
    config_path: str | Path,
    expected_config_sha256: str,
    *,
    producer_repo: Path | None,
    output_dir: Path | None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config_file = (
        _safe_relative(root, config_path, "config_path_invalid")
        if isinstance(config_path, str)
        else Path(config_path)
    )
    config_snapshot = _read_small_snapshot(root, config_file, "config_unreadable")
    if config_snapshot.sha256 != expected_config_sha256:
        _fail("config_sha256_mismatch")
    config = _load_mapping(config_snapshot.data, "config_invalid")

    self_bindings = {
        "config_schema": CONFIG_SCHEMA_PATH,
        "attestation_schema": ATTESTATION_SCHEMA_PATH,
        "implementation": IMPLEMENTATION_PATH,
        "cli": CLI_PATH,
    }
    self_snapshots: dict[str, _SmallSnapshot] = {}
    for role, relative in self_bindings.items():
        self_snapshots[role] = _read_small_snapshot(
            root,
            _safe_relative(root, relative, f"{role}_path_invalid"),
            f"{role}_unreadable",
        )
    config_schema = _load_mapping(
        self_snapshots["config_schema"].data, "config_schema_invalid"
    )
    attestation_schema = _load_mapping(
        self_snapshots["attestation_schema"].data,
        "attestation_schema_invalid",
    )
    config_validator = _validator(config_schema, "config_schema_invalid")
    attestation_validator = _validator(attestation_schema, "attestation_schema_invalid")
    _validate(config_validator, config, "config_schema_validation_failed")
    _validate_config_contract(config)

    paths = _mapping(config["paths"], "config_paths_invalid")
    if producer_repo is None:
        producer_repo = root.parent / paths["producer_repo_hint"]
    try:
        producer_repo = producer_repo.resolve(strict=True)
    except OSError as exc:
        raise Gemma3ChatProducerGitOverlayError("producer_repo_invalid") from exc
    if not producer_repo.is_dir() or _is_link_or_reparse(producer_repo):
        _fail("producer_repo_invalid")
    if output_dir is None:
        output_dir = _safe_relative(
            root, paths["default_output_dir"], "output_dir_invalid"
        )
    else:
        output_dir = output_dir.resolve(strict=False)

    producer_config = _mapping(config["producer_git"], "producer_config_invalid")
    git_blobs, ref_observation = _authenticate_git(producer_repo, producer_config)
    runtime_rows = [
        _mapping(row, "runtime_copy_inventory_invalid")
        for row in config["runtime_copies"]
    ]
    runtime_result, runtime_snapshots = _authenticate_runtime_copies(
        root, runtime_rows, git_blobs
    )
    legacy_result, legacy_small, legacy_large = _authenticate_legacy_results(
        root, config
    )

    artifact_rows = [dict(row) for row in producer_config["artifacts"]]
    attestation: dict[str, Any] = {
        "schema_version": ATTESTATION_VERSION,
        "status": "passed_posthoc_git_provenance_only",
        "claim_scope": "additive_posthoc_git_provenance_non_authorizing",
        "producer_git": {
            "repository_id": producer_config["repository_id"],
            "commit_sha1": producer_config["commit_sha1"],
            "tree_sha1": producer_config["tree_sha1"],
            "parent_commit_sha1": producer_config["parent_commit_sha1"],
            "parent_tree_sha1": producer_config["parent_tree_sha1"],
            **ref_observation,
            "network_remote_checked": False,
            "changed_file_count": 24,
            "added_file_count": 23,
            "modified_file_count": 1,
            "artifact_inventory_sha256": _canonical_sha256(artifact_rows),
            "artifacts": artifact_rows,
        },
        "runtime_binding": {
            "copy_count": 16,
            "all_equal": True,
            "jsonl_parsed": False,
            "token_ids_parsed": False,
            "inventory_sha256": _canonical_sha256(runtime_result),
            "copies": runtime_result,
        },
        "legacy_results": legacy_result,
        "auditor_identity": {
            "config_sha256": config_snapshot.sha256,
            "config_schema_sha256": self_snapshots["config_schema"].sha256,
            "attestation_schema_sha256": self_snapshots["attestation_schema"].sha256,
            "implementation_sha256": self_snapshots["implementation"].sha256,
            "cli_sha256": self_snapshots["cli"].sha256,
        },
        "claims": dict(config["claims"]),
        "audit": {
            **dict(config["resource_audit"]),
            "single_bytes_snapshot": True,
            "stat_identity_checked": True,
            "reparse_points_rejected": True,
            "final_toctou_recheck": True,
            "git_environment_sanitized": True,
            "git_no_replace_objects": True,
            "git_no_lazy_fetch": True,
            "replace_refs_rejected": True,
            "grafts_rejected": True,
            "git_commit_tree_parent_authenticated": True,
            "git_diff_authenticated": True,
            "git_blobs_authenticated": 24,
            "runtime_copies_authenticated": 16,
            "legacy_receipts_authenticated": 12,
            "output_atomic_publish": True,
        },
    }
    attestation["attestation_sha256"] = _canonical_sha256(attestation)
    _validate(
        attestation_validator,
        attestation,
        "attestation_schema_validation_failed",
    )

    # Full end-of-audit identity recheck.  Git blobs and every consumed/runtime
    # file are authenticated a second time before the output is published.
    second_git_blobs, _second_ref_observation = _authenticate_git(
        producer_repo, producer_config
    )
    if second_git_blobs != git_blobs:
        _fail("producer_git_toctou_detected")
    for index, snapshot in enumerate(runtime_snapshots):
        snapshot.assert_unchanged(root, f"runtime_copy_{index}_changed")
    for index, snapshot in enumerate(legacy_small):
        snapshot.assert_unchanged(root, f"legacy_small_{index}_changed")
    for index, snapshot in enumerate(legacy_large):
        snapshot.assert_unchanged(root, f"legacy_large_{index}_changed")
    config_snapshot.assert_unchanged(root, "config_changed")
    for role, snapshot in self_snapshots.items():
        snapshot.assert_unchanged(root, f"{role}_changed")

    attestation_bytes = _canonical_bytes(attestation, newline=True)
    output_hashes = _atomic_publish(output_dir, attestation_bytes)
    return {
        "status": attestation["status"],
        "claim_scope": attestation["claim_scope"],
        "output_dir": str(output_dir),
        "attestation_sha256": attestation["attestation_sha256"],
        **output_hashes,
        "producer_commit": producer_config["commit_sha1"],
        "git_blobs_authenticated": 24,
        "runtime_copies_authenticated": 16,
        "roles_authenticated": 5,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_body_reads": 0,
        "luna_requests": 0,
    }


def audit_gemma3_chat_final_producer_git_overlay(
    config_path: str | Path = CONFIG_PATH,
    *,
    producer_repo: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate and publish the non-authorizing post-hoc attestation."""

    return _evaluate(
        _ROOT,
        config_path,
        CONFIG_SHA256,
        producer_repo=Path(producer_repo) if producer_repo is not None else None,
        output_dir=Path(output_dir) if output_dir is not None else None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--producer-repo")
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_gemma3_chat_final_producer_git_overlay(
            args.config,
            producer_repo=args.producer_repo,
            output_dir=args.output_dir,
        )
    except Gemma3ChatProducerGitOverlayError as exc:
        print(
            json.dumps(
                {
                    "schema_version": ATTESTATION_VERSION,
                    "status": "blocked_invalid_posthoc_provenance",
                    "error": exc.code,
                    "training_authorized": False,
                    "formal_training_authorized": False,
                    "formal": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
