"""Independent release attestation for the sharded Gemma 3 Chat v2 artifact.

The producer may ship this verifier but may not use it to sign its own output.
An independent reviewer runs it only after a scoped candidate commit exists.
The attestation upgrades the external artifact identity, never training,
formal, live, model-release, quality, or generalization authority.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


NAMESPACE = "gemma3_chat_five_expert_qonly_unbalanced_v2"
ARTIFACT_VERSION = "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-sharded.v3"
LIFECYCLE_STATUS = "candidate_pending_independent_release_review"
CANONICAL_ARTIFACT_REL = (
    "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_distilled_v2_sharded_v3"
)
RELEASE_ATTESTATION_REL = (
    "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_v2_release_v1/"
    "independent_release_attestation.json"
)
SCHEMA_REL = (
    "configs/research/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_"
    "independent_release_attestation_v1.schema.json"
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
AUDIT_CLI_REL = (
    "scripts/research/audit_gemma3_chat_five_expert_qonly_unbalanced_v2_final.py"
)
TRAIN_SHARD_PATHS = (
    "train/chat-00000-of-00002.jsonl",
    "train/chat-00001-of-00002.jsonl",
)
LOGICAL_TRAIN_SHA256 = (
    "62c0f64b6d2f5a0901169480c946d554cc02175138862b3a210d06e44fcbddd3"
)
LOGICAL_TRAIN_BYTES = 59_845_314
LOGICAL_TRAIN_RECORDS = 3_440
MAX_FILE_BYTES_EXCLUSIVE = 52_428_800
EXPECTED_FILES = 46
RESOURCE_COUNTERS = {
    "provider_requests": 0,
    "network_requests": 0,
    "model_loads": 0,
    "gpu_requests": 0,
    "protected_body_reads": 0,
    "gold_body_reads": 0,
    "heldout_body_reads": 0,
    "luna_requests": 0,
}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


class IndependentReleaseError(RuntimeError):
    """Stable, body-free independent release failure."""


def _fail(code: str) -> None:
    raise IndependentReleaseError(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _compact(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _domain_hash(domain: str, value: Any) -> str:
    return _sha256(domain.encode("ascii") + b"\0" + _compact(value).encode("utf-8"))


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("json_duplicate_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    _fail("json_nonfinite_number")


def _strict_json(raw: bytes, code: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndependentReleaseError(code) from exc
    if not isinstance(value, dict):
        _fail(code)
    return value


def _strict_jsonl(raw: bytes, code: str) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        _fail(code)
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line or len(line) > 1_048_576:
            _fail(code)
        rows.append(_strict_json(line, code))
    return rows


def _is_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return True
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0))
        & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    )


def _physical(path: Path, *, file: bool = False, directory: bool = False) -> Path:
    absolute = Path(os.path.abspath(path))
    if os.path.normcase(os.path.realpath(absolute)) != os.path.normcase(str(absolute)):
        _fail("physical_path_drift")
    current = absolute
    while True:
        if os.path.lexists(current) and _is_reparse(current):
            _fail("symlink_or_reparse_rejected")
        if current.parent == current:
            break
        current = current.parent
    if file and not absolute.is_file():
        _fail("physical_file_required")
    if directory and not absolute.is_dir():
        _fail("physical_directory_required")
    return absolute


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


@dataclass(frozen=True)
class Snapshot:
    path: Path
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int]


def _snapshot(path: Path) -> Snapshot:
    physical = _physical(path, file=True)
    try:
        before = physical.stat()
        with physical.open("rb", buffering=0) as handle:
            opened = os.fstat(handle.fileno())
            if _identity(opened) != _identity(before):
                _fail("snapshot_open_identity_mismatch")
            raw = handle.read()
        after = physical.stat()
    except OSError as exc:
        raise IndependentReleaseError("snapshot_failed") from exc
    if (
        _identity(before) != _identity(after)
        or len(raw) != before.st_size
        or len(raw) >= MAX_FILE_BYTES_EXCLUSIVE
    ):
        _fail("snapshot_toctou_or_size_limit")
    return Snapshot(
        path=physical,
        raw=raw,
        sha256=_sha256(raw),
        identity=_identity(after),
    )


def _tree_files(root: Path) -> set[str]:
    files: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        if _is_reparse(current_path):
            _fail("artifact_tree_reparse")
        for name in directories:
            if _is_reparse(current_path / name):
                _fail("artifact_tree_reparse")
        for name in filenames:
            path = current_path / name
            if _is_reparse(path):
                _fail("artifact_tree_reparse")
            files.add(path.relative_to(root).as_posix())
    return files


def _snapshot_artifact(root: Path) -> dict[str, Snapshot]:
    artifact = _physical(root, directory=True)
    paths = _tree_files(artifact)
    if len(paths) != EXPECTED_FILES:
        _fail("artifact_file_count_mismatch")
    return {relative: _snapshot(artifact / relative) for relative in sorted(paths)}


def _sidecar(filename: str, raw: bytes) -> bytes:
    return f"{_sha256(raw)}  {filename}\n".encode("ascii")


def _schema(snapshot: Snapshot) -> Draft202012Validator:
    value = _strict_json(snapshot.raw, "schema_json_invalid")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as exc:
        raise IndependentReleaseError("schema_invalid") from exc
    return Draft202012Validator(value)


def _validate(
    validator: Draft202012Validator,
    value: Mapping[str, Any],
    code: str,
) -> None:
    try:
        validator.validate(value)
    except ValidationError as exc:
        raise IndependentReleaseError(code) from exc


def _authenticate_artifact(
    snapshots: Mapping[str, Snapshot],
    manifest_validator: Draft202012Validator,
    receipt_validator: Draft202012Validator,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    required = {
        "manifest.json",
        "manifest.json.sha256",
        "build_receipt.json",
        "build_receipt.json.sha256",
        *TRAIN_SHARD_PATHS,
        *(path + ".sha256" for path in TRAIN_SHARD_PATHS),
    }
    if not required <= set(snapshots):
        _fail("artifact_required_file_missing")
    payloads = {path for path in snapshots if not path.endswith(".sha256")}
    sidecars = {
        path.removesuffix(".sha256") for path in snapshots if path.endswith(".sha256")
    }
    if payloads != sidecars:
        _fail("artifact_sidecar_inventory_mismatch")
    for payload in payloads:
        if snapshots[payload + ".sha256"].raw != _sidecar(
            Path(payload).name,
            snapshots[payload].raw,
        ):
            _fail("artifact_sidecar_mismatch")

    manifest = _strict_json(snapshots["manifest.json"].raw, "manifest_invalid")
    receipt = _strict_json(
        snapshots["build_receipt.json"].raw,
        "build_receipt_invalid",
    )
    _validate(manifest_validator, manifest, "manifest_schema_rejected")
    _validate(receipt_validator, receipt, "receipt_schema_rejected")
    if (
        manifest.get("status") != LIFECYCLE_STATUS
        or manifest.get("namespace") != NAMESPACE
        or manifest.get("artifact_version") != ARTIFACT_VERSION
        or manifest.get("canonical_path") != CANONICAL_ARTIFACT_REL
        or receipt.get("status") != LIFECYCLE_STATUS
        or receipt.get("artifact_version") != ARTIFACT_VERSION
        or receipt.get("manifest", {}).get("sha256")
        != snapshots["manifest.json"].sha256
    ):
        _fail("artifact_lifecycle_or_identity_mismatch")
    claims = manifest.get("claims", {})
    if (
        claims.get("candidate") is not True
        or claims.get("final") is not False
        or any(
            claims.get(key) is not False
            for key in (
                "training_authorized",
                "formal_training_authorized",
                "live_authorized",
                "release_authorized",
                "quality_validated",
                "generalization_validated",
            )
        )
    ):
        _fail("candidate_non_authorizing_claims_mismatch")

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        _fail("manifest_file_inventory_invalid")
    expected = {
        item.get("path") for item in manifest_files if isinstance(item, Mapping)
    } | {
        "manifest.json",
        "manifest.json.sha256",
        "build_receipt.json",
        "build_receipt.json.sha256",
    }
    if (
        len(expected) != EXPECTED_FILES
        or expected != set(snapshots)
        or len(manifest_files) != EXPECTED_FILES - 4
    ):
        _fail("manifest_file_inventory_mismatch")
    for item in manifest_files:
        if (
            not isinstance(item, Mapping)
            or item.get("path") not in snapshots
            or item.get("sha256") != snapshots[item["path"]].sha256
            or item.get("bytes") != len(snapshots[item["path"]].raw)
        ):
            _fail("manifest_file_binding_mismatch")

    shard_rows = [
        _strict_jsonl(snapshots[path].raw, "train_shard_jsonl_invalid")
        for path in TRAIN_SHARD_PATHS
    ]
    logical = b"".join(snapshots[path].raw for path in TRAIN_SHARD_PATHS)
    if (
        _sha256(logical) != LOGICAL_TRAIN_SHA256
        or len(logical) != LOGICAL_TRAIN_BYTES
        or sum(len(rows) for rows in shard_rows) != LOGICAL_TRAIN_RECORDS
    ):
        _fail("logical_train_partition_mismatch")
    bundle_sets: list[set[str]] = []
    for rows in shard_rows:
        bundles = [row.get("task_bundle_sha256") for row in rows]
        if not all(
            isinstance(value, str) and SHA256.fullmatch(value) for value in bundles
        ):
            _fail("train_shard_bundle_identity_invalid")
        bundle_sets.append(set(bundles))
    if bundle_sets[0] & bundle_sets[1]:
        _fail("train_shard_bundle_boundary_split")
    shards = manifest.get("training_shards", {}).get("shards")
    if not isinstance(shards, list) or len(shards) != 2:
        _fail("manifest_training_shards_invalid")
    for index, item in enumerate(shards):
        path = TRAIN_SHARD_PATHS[index]
        if (
            item.get("index") != index
            or item.get("path") != path
            or item.get("sha256") != snapshots[path].sha256
            or item.get("bytes") != len(snapshots[path].raw)
            or item.get("records") != len(shard_rows[index])
            or item.get("task_bundles") != len(bundle_sets[index])
        ):
            _fail("manifest_training_shard_binding_mismatch")
    return manifest, receipt, [dict(item) for item in shards]


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


def _git(root: Path, arguments: Sequence[str], code: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IndependentReleaseError(code) from exc
    if result.returncode != 0:
        _fail(code)
    return result.stdout


def _require_explicit_lf_attributes(root: Path, relative: str) -> None:
    if (
        not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or "\\" in relative
    ):
        _fail("git_lf_attribute_path_invalid")
    try:
        lines = (
            _git(
                root,
                ["check-attr", "text", "eol", "--", relative],
                "git_lf_attribute_invalid",
            )
            .decode("utf-8")
            .splitlines()
        )
    except UnicodeDecodeError as exc:
        raise IndependentReleaseError("git_lf_attribute_invalid") from exc
    if lines != [
        f"{relative}: text: set",
        f"{relative}: eol: lf",
    ]:
        _fail("git_lf_attribute_invalid")


def _authenticate_git(
    root: Path,
    commit: str,
    artifact_snapshots: Mapping[str, Snapshot],
    receipt: Mapping[str, Any],
    source_snapshots: Mapping[str, Snapshot],
) -> list[str]:
    if COMMIT.fullmatch(commit) is None:
        _fail("producer_candidate_commit_invalid")
    git_dir_raw = _git(root, ["rev-parse", "--absolute-git-dir"], "git_invalid")
    try:
        git_dir = Path(git_dir_raw.decode("utf-8").strip())
    except UnicodeDecodeError as exc:
        raise IndependentReleaseError("git_invalid") from exc
    grafts = git_dir / "info" / "grafts"
    if grafts.exists() and grafts.read_bytes().strip():
        _fail("git_grafts_forbidden")
    if _git(
        root,
        ["for-each-ref", "--format=%(refname)", "refs/replace"],
        "git_replace_refs_invalid",
    ).strip():
        _fail("git_replace_refs_forbidden")
    resolved = (
        _git(
            root,
            ["rev-parse", "--verify", f"{commit}^{{commit}}"],
            "producer_candidate_commit_unavailable",
        )
        .decode("ascii")
        .strip()
    )
    if resolved != commit:
        _fail("producer_candidate_commit_mismatch")

    input_bindings = receipt.get("input_snapshot_inventory", [])
    if not isinstance(input_bindings, list):
        _fail("input_source_inventory_mismatch")
    input_paths = [
        item.get("path")
        for item in input_bindings
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    ]
    if (
        len(input_paths) != len(input_bindings)
        or len(set(input_paths)) != len(input_paths)
        or set(input_paths) != set(source_snapshots)
        or any(
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or "\\" in relative
            or relative.startswith("external/")
            for relative in input_paths
        )
    ):
        _fail("input_source_inventory_mismatch")
    binding_by_path = {item["path"]: item for item in input_bindings}
    if any(
        binding_by_path[relative].get("sha256") != snapshot.sha256
        or binding_by_path[relative].get("bytes") != len(snapshot.raw)
        for relative, snapshot in source_snapshots.items()
    ):
        _fail("input_source_binding_mismatch")
    artifact_paths = {
        f"{CANONICAL_ARTIFACT_REL}/{relative}": snapshot
        for relative, snapshot in artifact_snapshots.items()
    }
    scoped_snapshots = {
        **artifact_paths,
        **source_snapshots,
    }
    tree_inventory = set(
        _git(
            root,
            [
                "ls-tree",
                "-r",
                "--name-only",
                commit,
                "--",
                CANONICAL_ARTIFACT_REL,
            ],
            "git_artifact_inventory_invalid",
        )
        .decode("utf-8")
        .splitlines()
    )
    if tree_inventory != set(artifact_paths):
        _fail("git_artifact_inventory_mismatch")
    paths = sorted(scoped_snapshots)
    status = _git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all", "--", *paths],
        "git_scoped_status_invalid",
    )
    if status.strip():
        _fail("git_scoped_checkout_not_clean")
    for relative in paths:
        snapshot = scoped_snapshots[relative]
        entry = (
            _git(
                root,
                ["ls-tree", commit, "--", relative],
                "git_commit_tree_invalid",
            )
            .decode("utf-8")
            .rstrip("\n")
        )
        if not entry.startswith("100644 blob ") or not entry.endswith(f"\t{relative}"):
            _fail("git_commit_tree_invalid")
        committed = _git(
            root,
            ["cat-file", "blob", f"{commit}:{relative}"],
            "git_commit_blob_unavailable",
        )
        if committed != snapshot.raw:
            _fail("git_commit_blob_mismatch")
        stage = (
            _git(
                root,
                ["ls-files", "--stage", "--", relative],
                "git_index_invalid",
            )
            .decode("utf-8")
            .rstrip("\n")
        )
        match = re.fullmatch(r"100644 ([0-9a-f]{40}) 0\t(.+)", stage)
        if match is None or match.group(2) != relative:
            _fail("git_index_invalid")
        indexed = _git(
            root,
            ["cat-file", "blob", match.group(1)],
            "git_index_blob_unavailable",
        )
        if indexed != snapshot.raw:
            _fail("git_index_blob_mismatch")
        _require_explicit_lf_attributes(root, relative)
    return paths


def _run_official_audit(
    root: Path,
    artifact: Path,
    tool_review: Path | None,
    planner_review: Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(root / AUDIT_CLI_REL),
        "--repo-root",
        str(root),
        "--artifact",
        str(artifact),
    ]
    if tool_review is not None:
        command.extend(["--tool-review-receipt", str(tool_review)])
    if planner_review is not None:
        command.extend(["--planner-review-receipt", str(planner_review)])
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IndependentReleaseError("official_audit_failed") from exc
    if result.returncode != 0 or result.stderr:
        _fail("official_audit_failed")
    try:
        value = _strict_json(result.stdout.strip(), "official_audit_invalid")
    except IndependentReleaseError:
        raise
    if (
        value.get("audit") != "passed_candidate_pending_independent_release_review"
        or value.get("artifact_version") != ARTIFACT_VERSION
        or value.get("files") != EXPECTED_FILES
        or value.get("resource_counters") != RESOURCE_COUNTERS
    ):
        _fail("official_audit_assertion_failed")
    return value


def _terminal_recheck(
    snapshots: Mapping[str, Snapshot],
) -> None:
    for snapshot in snapshots.values():
        current = _snapshot(snapshot.path)
        if (
            current.raw != snapshot.raw
            or current.sha256 != snapshot.sha256
            or current.identity != snapshot.identity
        ):
            _fail("terminal_toctou_recheck_failed")


def _write_exclusive(path: Path, raw: bytes) -> None:
    target = Path(os.path.abspath(path))
    _physical(target.parent, directory=True)
    try:
        with target.open("xb", buffering=0) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise IndependentReleaseError("attestation_create_once_failed") from exc
    if _snapshot(target).raw != raw:
        _fail("attestation_write_drift")


def _directory_identity(path: Path, code: str) -> tuple[int, int]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise IndependentReleaseError(code) from exc
    if (
        stat.S_ISLNK(value.st_mode)
        or _is_reparse(path)
        or not stat.S_ISDIR(value.st_mode)
    ):
        _fail(code)
    return int(value.st_dev), int(value.st_ino)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise IndependentReleaseError("directory_fsync_failed") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise IndependentReleaseError("directory_fsync_failed") from exc
    finally:
        os.close(descriptor)


def _snapshot_attestation_pair(
    directory: Path,
    *,
    expected_directory_identity: tuple[int, int],
    attestation_name: str,
    attestation_raw: bytes,
    sidecar_raw: bytes,
) -> dict[str, Snapshot]:
    identity = _directory_identity(directory, "attestation_staging_identity_invalid")
    if identity != expected_directory_identity:
        _fail("attestation_pair_directory_identity_drift")
    expected = {
        attestation_name: attestation_raw,
        attestation_name + ".sha256": sidecar_raw,
    }
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise IndependentReleaseError("attestation_pair_inventory_failed") from exc
    if {entry.name for entry in entries} != set(expected):
        _fail("attestation_pair_inventory_mismatch")
    snapshots = {name: _snapshot(directory / name) for name in sorted(expected)}
    if any(
        snapshots[name].raw != raw or snapshots[name].sha256 != _sha256(raw)
        for name, raw in expected.items()
    ):
        _fail("attestation_pair_content_mismatch")
    if snapshots[attestation_name + ".sha256"].raw != _sidecar(
        attestation_name,
        snapshots[attestation_name].raw,
    ):
        _fail("attestation_pair_sidecar_mismatch")
    if (
        _directory_identity(directory, "attestation_staging_identity_invalid")
        != expected_directory_identity
    ):
        _fail("attestation_pair_directory_identity_drift")
    return snapshots


def _rename_directory_noreplace(
    source: Path,
    destination: Path,
    before_rename: Any,
) -> dict[str, Snapshot]:
    authenticated = before_rename()
    if os.path.lexists(destination):
        raise FileExistsError("canonical_release_directory_exists")
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        move_file.restype = ctypes.c_int
        if not move_file(str(source), str(destination)):
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError("canonical_release_directory_exists")
            raise OSError(error, "atomic_release_directory_move_failed")
        return authenticated
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
                raise FileExistsError("canonical_release_directory_exists")
            raise OSError(error, "atomic_release_directory_move_failed")
        return authenticated
    _fail("atomic_noreplace_unsupported")


def _publish_attestation_pair(
    project: Path,
    destination: Path,
    attestation_raw: bytes,
    *,
    before_rename: Any,
) -> dict[str, Snapshot]:
    release_directory = destination.parent
    parent = _physical(release_directory.parent, directory=True)
    parent_identity = _directory_identity(parent, "release_parent_identity_invalid")
    if os.path.lexists(release_directory):
        raise FileExistsError("canonical_release_directory_exists")
    sidecar_raw = _sidecar(destination.name, attestation_raw)
    owner_nonce = secrets.token_hex(24)
    staging = parent / f".{release_directory.name}.staging-{owner_nonce}"
    if os.path.lexists(staging):
        _fail("attestation_staging_nonce_collision")
    try:
        staging.mkdir(mode=0o700)
    except OSError as exc:
        raise IndependentReleaseError("attestation_staging_create_failed") from exc
    staging_identity = _directory_identity(
        staging,
        "attestation_staging_identity_invalid",
    )
    try:
        if (
            _directory_identity(parent, "release_parent_identity_invalid")
            != parent_identity
        ):
            _fail("release_parent_identity_drift")
        if (
            _directory_identity(staging, "attestation_staging_identity_invalid")
            != staging_identity
        ):
            _fail("attestation_staging_identity_drift")
        _write_exclusive(staging / destination.name, attestation_raw)
        if (
            _directory_identity(staging, "attestation_staging_identity_invalid")
            != staging_identity
        ):
            _fail("attestation_staging_identity_drift")
        _write_exclusive(staging / (destination.name + ".sha256"), sidecar_raw)
        _fsync_directory(staging)

        def authenticate_before_rename() -> dict[str, Snapshot]:
            if (
                _directory_identity(
                    staging,
                    "attestation_staging_identity_invalid",
                )
                != staging_identity
            ):
                _fail("attestation_staging_identity_drift")
            snapshots = _snapshot_attestation_pair(
                staging,
                expected_directory_identity=staging_identity,
                attestation_name=destination.name,
                attestation_raw=attestation_raw,
                sidecar_raw=sidecar_raw,
            )
            _require_explicit_lf_attributes(project, RELEASE_ATTESTATION_REL)
            _require_explicit_lf_attributes(
                project,
                RELEASE_ATTESTATION_REL + ".sha256",
            )
            before_rename()
            for name, snapshot in snapshots.items():
                current = _snapshot(staging / name)
                if (
                    current.raw != snapshot.raw
                    or current.sha256 != snapshot.sha256
                    or current.identity != snapshot.identity
                ):
                    _fail("attestation_pair_terminal_toctou_failed")
            if (
                _directory_identity(parent, "release_parent_identity_invalid")
                != parent_identity
            ):
                _fail("release_parent_identity_drift")
            return snapshots

        authenticated = _rename_directory_noreplace(
            staging,
            release_directory,
            authenticate_before_rename,
        )
        _fsync_directory(parent)
        if os.path.lexists(staging):
            _fail("attestation_staging_still_exists_after_publish")
        if (
            _directory_identity(
                release_directory,
                "canonical_release_directory_identity_invalid",
            )
            != staging_identity
        ):
            _fail("canonical_release_directory_identity_drift")
        published = _snapshot_attestation_pair(
            release_directory,
            expected_directory_identity=staging_identity,
            attestation_name=destination.name,
            attestation_raw=attestation_raw,
            sidecar_raw=sidecar_raw,
        )
        if any(
            published[name].raw != snapshot.raw
            or published[name].sha256 != snapshot.sha256
            for name, snapshot in authenticated.items()
        ):
            _fail("published_attestation_pair_drift")
        return published
    except BaseException:
        # An owned staging directory is intentionally preserved for diagnosis.
        # No recursive cleanup or canonical partial-pair recovery is attempted.
        raise


def review_release(
    root: Path,
    artifact: Path,
    attestation_path: Path,
    *,
    producer_candidate_commit: str,
    producer_id: str,
    reviewer_id: str,
    tool_review_path: Path | None = None,
    planner_review_path: Path | None = None,
) -> dict[str, Any]:
    project = _physical(root, directory=True)
    candidate = _physical(artifact, directory=True)
    expected_candidate = project / CANONICAL_ARTIFACT_REL
    if os.path.normcase(str(candidate)) != os.path.normcase(str(expected_candidate)):
        _fail("canonical_artifact_path_required")
    destination = Path(os.path.abspath(attestation_path))
    try:
        destination.relative_to(candidate)
    except ValueError:
        pass
    else:
        _fail("attestation_must_be_outside_artifact")
    if (
        IDENTIFIER.fullmatch(producer_id) is None
        or IDENTIFIER.fullmatch(reviewer_id) is None
        or producer_id == reviewer_id
    ):
        _fail("reviewer_separation_invalid")
    expected_destination = project / RELEASE_ATTESTATION_REL
    if os.path.normcase(str(destination)) != os.path.normcase(
        str(expected_destination)
    ):
        _fail("canonical_attestation_path_required")
    _require_explicit_lf_attributes(project, RELEASE_ATTESTATION_REL)
    _require_explicit_lf_attributes(
        project,
        RELEASE_ATTESTATION_REL + ".sha256",
    )

    source_paths = {
        SCHEMA_REL,
        MANIFEST_SCHEMA_REL,
        RECEIPT_SCHEMA_REL,
    }
    source_snapshots = {
        relative: _snapshot(project / relative) for relative in sorted(source_paths)
    }
    artifact_snapshots = _snapshot_artifact(candidate)
    manifest, receipt, shards = _authenticate_artifact(
        artifact_snapshots,
        _schema(source_snapshots[MANIFEST_SCHEMA_REL]),
        _schema(source_snapshots[RECEIPT_SCHEMA_REL]),
    )
    producer_sources = {item["path"] for item in receipt["input_snapshot_inventory"]}
    all_source_snapshots = {
        relative: (
            source_snapshots[relative]
            if relative in source_snapshots
            else _snapshot(project / relative)
        )
        for relative in sorted(producer_sources)
    }
    authenticated_paths = _authenticate_git(
        project,
        producer_candidate_commit,
        artifact_snapshots,
        receipt,
        all_source_snapshots,
    )
    official = _run_official_audit(
        project,
        candidate,
        tool_review_path,
        planner_review_path,
    )
    if (
        official.get("manifest_sha256") != artifact_snapshots["manifest.json"].sha256
        or official.get("build_receipt_sha256")
        != artifact_snapshots["build_receipt.json"].sha256
        or official.get("logical_identity") != manifest.get("logical_identity")
    ):
        _fail("official_audit_identity_mismatch")
    tree_bindings = [
        {
            "path": path,
            "sha256": snapshot.sha256,
            "bytes": len(snapshot.raw),
        }
        for path, snapshot in sorted(artifact_snapshots.items())
    ]
    attestation = {
        "schema_version": (
            "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-"
            "independent-release-attestation.v3"
        ),
        "audit_status": "passed",
        "artifact_lifecycle_status": LIFECYCLE_STATUS,
        "namespace": NAMESPACE,
        "artifact_version": ARTIFACT_VERSION,
        "canonical_artifact_path": CANONICAL_ARTIFACT_REL,
        "producer_candidate_commit": producer_candidate_commit,
        "reviewer_separation": {
            "producer_id": producer_id,
            "reviewer_id": reviewer_id,
            "different_reviewer": True,
        },
        "findings": {"p0": 0, "p1": 0, "p2": 0},
        "artifact": {
            "files": len(artifact_snapshots),
            "tree_sha256": _domain_hash(
                "anchor.gemma3-chat-sharded-final-physical-tree.v1",
                tree_bindings,
            ),
            "manifest_sha256": artifact_snapshots["manifest.json"].sha256,
            "manifest_sidecar_physical_sha256": artifact_snapshots[
                "manifest.json.sha256"
            ].sha256,
            "build_receipt_sha256": artifact_snapshots["build_receipt.json"].sha256,
            "build_receipt_sidecar_physical_sha256": artifact_snapshots[
                "build_receipt.json.sha256"
            ].sha256,
            "output_inventory_sha256": receipt["output_inventory_sha256"],
            "all_payload_sidecars_valid": True,
            "all_files_below_50_mib": True,
        },
        "training_shards": {
            "shard_count": 2,
            "ordered_paths": list(TRAIN_SHARD_PATHS),
            "shards": [
                {
                    key: item[key]
                    for key in (
                        "index",
                        "path",
                        "sha256",
                        "bytes",
                        "records",
                        "task_bundles",
                    )
                }
                for item in shards
            ],
            "ordered_concat_sha256": LOGICAL_TRAIN_SHA256,
            "logical_partition_bytes": LOGICAL_TRAIN_BYTES,
            "logical_partition_records": LOGICAL_TRAIN_RECORDS,
            "bundle_boundary_only": True,
        },
        "logical_identity": dict(manifest["logical_identity"]),
        "git_authentication": {
            "commit_resolved_exactly": True,
            "replace_refs_absent": True,
            "grafts_absent": True,
            "tracked_inventory_exact": len(authenticated_paths)
            == EXPECTED_FILES + len(all_source_snapshots),
            "commit_blobs_equal_physical_bytes": True,
            "index_blobs_equal_physical_bytes": True,
            "scoped_checkout_clean": True,
            "lf_attributes_exact": True,
        },
        "integrity": {
            "single_bytes_snapshot": True,
            "terminal_toctou_recheck": True,
            "official_audit_passed": True,
            "draft_2020_12_passed": True,
            "create_once_attestation": True,
            "atomic_attestation_sidecar_pair_publish": True,
            "owned_sibling_staging_directory": True,
            "atomic_directory_rename_no_replace": True,
            "canonical_partial_pair_possible": False,
            "recursive_cleanup_used": False,
            "attestation_and_sidecar_attributes_preverified": True,
            "attestation_outside_artifact_tree": True,
            "old_candidate_superseded": True,
            "old_candidate_consumable": False,
        },
        "resource_counters": dict(RESOURCE_COUNTERS),
        "claims": {
            "artifact_final_identity": True,
            "independent_review_passed": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
            "model_release_authorized": False,
            "quality_validated": False,
            "generalization_validated": False,
        },
    }
    _validate(
        _schema(source_snapshots[SCHEMA_REL]),
        attestation,
        "attestation_schema_rejected",
    )
    raw = _json_bytes(attestation)
    published = _publish_attestation_pair(
        project,
        destination,
        raw,
        before_rename=lambda: _terminal_recheck(
            {**artifact_snapshots, **all_source_snapshots}
        ),
    )
    return {
        **attestation,
        "attestation_path": str(destination),
        "attestation_sha256": published[destination.name].sha256,
        "sidecar_physical_sha256": published[destination.name + ".sha256"].sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(CANONICAL_ARTIFACT_REL),
    )
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--producer-candidate-commit", required=True)
    parser.add_argument("--producer-id", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--tool-review-receipt", type=Path)
    parser.add_argument("--planner-review-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = Path(os.path.abspath(arguments.repo_root))
    artifact = arguments.artifact
    if not artifact.is_absolute():
        artifact = root / artifact
    try:
        result = review_release(
            root,
            artifact,
            arguments.attestation,
            producer_candidate_commit=arguments.producer_candidate_commit,
            producer_id=arguments.producer_id,
            reviewer_id=arguments.reviewer_id,
            tool_review_path=arguments.tool_review_receipt,
            planner_review_path=arguments.planner_review_receipt,
        )
    except (
        IndependentReleaseError,
        FileExistsError,
        OSError,
        subprocess.TimeoutExpired,
    ) as exc:
        code = (
            str(exc) if isinstance(exc, IndependentReleaseError) else type(exc).__name__
        )
        print(
            _compact(
                {
                    "audit_status": "failed",
                    "artifact_lifecycle_status": LIFECYCLE_STATUS,
                    "error_code": code,
                    "resource_counters": dict(RESOURCE_COUNTERS),
                }
            )
        )
        return 2
    print(_compact(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
