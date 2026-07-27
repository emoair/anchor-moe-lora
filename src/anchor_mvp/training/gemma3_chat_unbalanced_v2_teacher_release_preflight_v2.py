"""Model-free authentication for the Teacher release implementation.

This gate authenticates immutable Git blobs for the already reviewed
controller/finalizer/release implementation.  It deliberately does not claim
that the post-distillation Teacher FINAL artifact exists or is consumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[3]
BINDING_PATH = Path(
    "configs/training/"
    "gemma3_chat_unbalanced_v2_teacher_release_implementation_binding_v2.json"
)
BINDING_SCHEMA_PATH = BINDING_PATH.with_name(
    "gemma3_chat_unbalanced_v2_teacher_release_implementation_binding_v2.schema.json"
)
RECEIPT_PATH = Path(
    "fixtures/research/"
    "gemma3_chat_unbalanced_v2_teacher_release_implementation_preflight_v2/"
    "receipt.json"
)
RECEIPT_SCHEMA_PATH = Path(
    "configs/training/"
    "gemma3_chat_unbalanced_v2_teacher_release_implementation_"
    "preflight_receipt_v2.schema.json"
)
PRODUCER_COMMIT = "b88e749d4d19b6366d4b841c95a4fe46285c9c15"
PRODUCER_TREE = "b9d83b10eb5e62e5661ec8f57e9152fc11317156"
CANONICAL_BINDING_SHA256 = (
    "096c33110a1a3d0941154e89270319a7cf48dd33432296cae298b8ec469ce21d"
)
BINDING_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-release-implementation-binding.v2"
)
RECEIPT_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-teacher-release-implementation-"
    "preflight-receipt.v2"
)


class TeacherReleasePreflightError(RuntimeError):
    """Body-free fail-closed preflight error."""


def _fail(code: str) -> None:
    raise TeacherReleasePreflightError(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _load_json(path: Path, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeacherReleasePreflightError(code) from exc
    if not isinstance(value, dict):
        _fail(code)
    return value, raw


def _validator(path: Path, code: str) -> tuple[Draft202012Validator, str]:
    value, raw = _load_json(path, code)
    try:
        Draft202012Validator.check_schema(value)
    except Exception as exc:  # noqa: BLE001 - normalized fail-closed boundary
        raise TeacherReleasePreflightError(code) from exc
    return Draft202012Validator(value), _sha256(raw)


def load_binding(
    path: str | os.PathLike[str] = ROOT / BINDING_PATH,
) -> tuple[dict[str, Any], str, str]:
    binding_path = Path(path)
    binding, raw = _load_json(binding_path, "teacher_release_binding_invalid")
    validator, schema_sha = _validator(
        ROOT / BINDING_SCHEMA_PATH,
        "teacher_release_binding_schema_invalid",
    )
    errors = tuple(validator.iter_errors(binding))
    if errors or binding.get("schema_version") != BINDING_VERSION:
        _fail("teacher_release_binding_invalid")
    expected = binding["producer"]
    if (
        expected["commit"] != PRODUCER_COMMIT
        or expected["tree"] != PRODUCER_TREE
        or expected["exact_files"] != len(binding["exact_files"])
        or len({item["path"] for item in binding["exact_files"]}) != 20
    ):
        _fail("teacher_release_binding_invalid")
    inventory = _sha256(_canonical_json(binding["exact_files"]))
    if inventory != expected["file_inventory_sha256"]:
        _fail("teacher_release_file_inventory_invalid")
    return binding, _sha256(raw), schema_sha


def _git(
    repository: Path,
    args: Sequence[str],
    *,
    stdin: bytes | None = None,
) -> bytes:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("GIT_"):
            environment.pop(key, None)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    command = [
        "git",
        "--no-replace-objects",
        "-c",
        "core.useReplaceRefs=false",
        "-C",
        str(repository),
        *args,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            input=stdin,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TeacherReleasePreflightError(
            "teacher_release_git_authentication_failed"
        ) from exc
    return completed.stdout


def _assert_no_replacements(repository: Path) -> None:
    replacements = _git(
        repository, ["for-each-ref", "--format=%(refname)", "refs/replace"]
    )
    if replacements.strip():
        _fail("teacher_release_git_replacement_ref_forbidden")
    git_dir = Path(
        _git(repository, ["rev-parse", "--absolute-git-dir"]).decode("utf-8").strip()
    )
    if (git_dir / "info" / "grafts").exists():
        _fail("teacher_release_git_grafts_forbidden")


def _git_blob_inventory(
    repository: Path,
    paths: Sequence[str],
) -> dict[str, bytes]:
    tree = _git(
        repository,
        [
            "ls-tree",
            "-rz",
            "--full-tree",
            PRODUCER_COMMIT,
            "--",
            *paths,
        ],
    )
    objects: dict[str, str] = {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            _mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise TeacherReleasePreflightError(
                "teacher_release_git_tree_invalid"
            ) from exc
        if object_type != "blob" or path in objects:
            _fail("teacher_release_git_object_type_invalid")
        objects[path] = object_id
    if set(objects) != set(paths):
        _fail("teacher_release_git_tree_invalid")
    output = _git(
        repository,
        ["cat-file", "--batch"],
        stdin=b"".join(f"{objects[path]}\n".encode("ascii") for path in paths),
    )
    result: dict[str, bytes] = {}
    offset = 0
    for path in paths:
        header_end = output.find(b"\n", offset)
        if header_end < 0:
            _fail("teacher_release_git_blob_batch_invalid")
        try:
            object_id, object_type, raw_size = output[offset:header_end].split(b" ")
            size = int(raw_size)
        except ValueError as exc:
            raise TeacherReleasePreflightError(
                "teacher_release_git_blob_batch_invalid"
            ) from exc
        if object_id.decode("ascii") != objects[path] or object_type != b"blob":
            _fail("teacher_release_git_object_type_invalid")
        start = header_end + 1
        end = start + size
        if end >= len(output) or output[end : end + 1] != b"\n":
            _fail("teacher_release_git_blob_batch_invalid")
        result[path] = output[start:end]
        offset = end + 1
    if offset != len(output):
        _fail("teacher_release_git_blob_batch_invalid")
    return result


def _git_commit_change_paths(repository: Path) -> tuple[str, ...]:
    raw = _git(
        repository,
        [
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            PRODUCER_COMMIT,
        ],
    )
    paths: list[str] = []
    try:
        records = raw.split(b"\0")
        if records[-1] != b"":
            _fail("teacher_release_git_change_path_inventory_invalid")
        for record in records[:-1]:
            path = record.decode("utf-8")
            if not path or path.startswith("/") or "\\" in path or path in paths:
                _fail("teacher_release_git_change_path_inventory_invalid")
            paths.append(path)
    except UnicodeDecodeError as exc:
        raise TeacherReleasePreflightError(
            "teacher_release_git_change_path_inventory_invalid"
        ) from exc
    return tuple(paths)


def authenticate_release_implementation(
    producer_repository: str | os.PathLike[str],
    *,
    binding_path: str | os.PathLike[str] = ROOT / BINDING_PATH,
) -> dict[str, Any]:
    binding, binding_sha, binding_schema_sha = load_binding(binding_path)
    if binding_sha != CANONICAL_BINDING_SHA256:
        _fail("teacher_release_binding_identity_mismatch")
    repository = Path(producer_repository).resolve()
    _assert_no_replacements(repository)
    observed_commit = (
        _git(repository, ["rev-parse", "--verify", f"{PRODUCER_COMMIT}^{{commit}}"])
        .decode("ascii")
        .strip()
    )
    observed_tree = (
        _git(repository, ["show", "-s", "--format=%T", PRODUCER_COMMIT])
        .decode("ascii")
        .strip()
    )
    if observed_commit != PRODUCER_COMMIT or observed_tree != PRODUCER_TREE:
        _fail("teacher_release_commit_identity_mismatch")
    paths = [str(item["path"]) for item in binding["exact_files"]]
    changed_paths = _git_commit_change_paths(repository)
    if len(changed_paths) != 20 or len(paths) != 20 or set(changed_paths) != set(paths):
        _fail("teacher_release_git_change_path_inventory_mismatch")
    blobs = _git_blob_inventory(repository, paths)
    observed_files: list[dict[str, Any]] = []
    for expected in binding["exact_files"]:
        path = str(expected["path"])
        raw = blobs[path]
        observed = {"path": path, "sha256": _sha256(raw), "bytes": len(raw)}
        if observed != expected:
            _fail("teacher_release_git_blob_identity_mismatch")
        observed_files.append(observed)
    inventory_sha = _sha256(_canonical_json(observed_files))
    if inventory_sha != binding["producer"]["file_inventory_sha256"]:
        _fail("teacher_release_file_inventory_invalid")
    implementation_path = ROOT / Path(
        "src/anchor_mvp/training/"
        "gemma3_chat_unbalanced_v2_teacher_release_preflight_v2.py"
    )
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "status": "passed",
        "operation": "immutable_git_release_implementation_preflight",
        "binding_sha256": binding_sha,
        "binding_schema_sha256": binding_schema_sha,
        "implementation_sha256": _sha256(implementation_path.read_bytes()),
        "producer": {
            "commit": observed_commit,
            "tree": observed_tree,
            "exact_files": len(observed_files),
            "file_inventory_sha256": inventory_sha,
            "replace_refs_used": False,
            "worktree_bytes_used": False,
        },
        "independent_review": {"status": "passed", "p0": 0, "p1": 0, "p2": 0},
        "post_distillation_teacher_final": {
            "state": "pending_materialization_and_external_release",
            "artifact_authenticated": False,
            "training_consumable": False,
        },
        "resource_counters": {
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "teacher_record_bodies_read": 0,
            "raw_token_ids_read": 0,
        },
    }
    receipt_validator, _ = _validator(
        ROOT / RECEIPT_SCHEMA_PATH,
        "teacher_release_receipt_schema_invalid",
    )
    if tuple(receipt_validator.iter_errors(receipt)):
        _fail("teacher_release_receipt_invalid")
    return receipt


def validate_checked_receipt(
    *,
    binding_path: str | os.PathLike[str] = ROOT / BINDING_PATH,
    receipt_path: str | os.PathLike[str] = ROOT / RECEIPT_PATH,
) -> dict[str, Any]:
    _, binding_sha, binding_schema_sha = load_binding(binding_path)
    if binding_sha != CANONICAL_BINDING_SHA256:
        _fail("teacher_release_binding_identity_mismatch")
    receipt_file = Path(receipt_path)
    receipt, raw = _load_json(receipt_file, "teacher_release_receipt_invalid")
    validator, _ = _validator(
        ROOT / RECEIPT_SCHEMA_PATH,
        "teacher_release_receipt_schema_invalid",
    )
    if tuple(validator.iter_errors(receipt)):
        _fail("teacher_release_receipt_invalid")
    if (
        receipt["binding_sha256"] != binding_sha
        or receipt["binding_schema_sha256"] != binding_schema_sha
        or receipt["implementation_sha256"]
        != _sha256(
            (
                ROOT / "src/anchor_mvp/training/"
                "gemma3_chat_unbalanced_v2_teacher_release_preflight_v2.py"
            ).read_bytes()
        )
    ):
        _fail("teacher_release_receipt_identity_mismatch")
    sidecar = receipt_file.with_name(receipt_file.name + ".sha256")
    expected_sha = _sha256(raw)
    try:
        expected_sidecar = f"{expected_sha}  {receipt_file.name}\n".encode("ascii")
        if sidecar.read_bytes() != expected_sidecar:
            _fail("teacher_release_receipt_sidecar_mismatch")
    except OSError as exc:
        raise TeacherReleasePreflightError(
            "teacher_release_receipt_sidecar_mismatch"
        ) from exc
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-repository", required=True)
    parser.add_argument("--binding", default=str(ROOT / BINDING_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = authenticate_release_implementation(
            args.producer_repository,
            binding_path=args.binding,
        )
    except TeacherReleasePreflightError as exc:
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "error_code": str(exc),
                    "provider_requests": 0,
                    "network_requests": 0,
                    "model_loads": 0,
                    "gpu_requests": 0,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
