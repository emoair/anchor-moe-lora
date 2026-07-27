"""Additive consumer-identity rollover for the single-process GLM controller."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_integrated_controller_v2 as integrated,
)
from anchor_mvp.data import gemma3_chat_unbalanced_v2_live_controller as base

CONFIG_PATH = (
    batch.REPO_ROOT
    / "configs/data/gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.json"
)
SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2.schema.json"
)
INTEGRATED_CONFIG_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_"
    "consumer_rollover_v2.json"
)
CONSUMER_REPOSITORY_ENV = "ANCHOR_GEMMA3_UNBALANCED_V2_CONSUMER_REPOSITORY"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _snapshot(path: Path, *, reason: str) -> bytes:
    return base._stable_read(path, reason=reason)


def _document(raw: bytes, *, reason: str) -> dict[str, Any]:
    value = base._strict_json(raw, reason=reason)
    if not isinstance(value, dict):
        raise batch.AdapterError(reason)
    return value


def _validate(schema: Mapping[str, Any], value: Any, *, reason: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except (SchemaError, ValidationError) as error:
        raise batch.AdapterError(reason) from error


def _physical_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(getattr(value, "st_file_attributes", 0)),
    )


def _authenticated_git_executable(config: Mapping[str, Any]) -> Path:
    pin = config["repository"]["git_executable"]
    raw_path = pin["path"]
    if not isinstance(raw_path, str) or raw_path != raw_path.strip() or not raw_path:
        raise batch.AdapterError("consumer_rollover_git_executable_path_invalid")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or path.as_posix() != raw_path
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise batch.AdapterError("consumer_rollover_git_executable_path_invalid")
    base._reject_reparse_chain(path)
    try:
        lexical_before = path.lstat()
        if not stat.S_ISREG(lexical_before.st_mode):
            raise batch.AdapterError("consumer_rollover_git_executable_type_invalid")
        with path.open("rb", buffering=0) as handle:
            handle_before = os.fstat(handle.fileno())
            if not stat.S_ISREG(handle_before.st_mode):
                raise batch.AdapterError(
                    "consumer_rollover_git_executable_type_invalid"
                )
            raw = handle.read()
            handle_after = os.fstat(handle.fileno())
        lexical_after = path.lstat()
    except batch.AdapterError:
        raise
    except OSError as error:
        raise batch.AdapterError(
            "consumer_rollover_git_executable_read_failed"
        ) from error
    identities = {
        _physical_identity(lexical_before),
        _physical_identity(handle_before),
        _physical_identity(handle_after),
        _physical_identity(lexical_after),
    }
    if (
        len(identities) != 1
        or len(raw) != pin["bytes"]
        or _sha256(raw) != pin["sha256"]
    ):
        raise batch.AdapterError("consumer_rollover_git_executable_identity_drift")
    return path


def _git(
    executable: Path,
    repo: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        result = subprocess.run(
            [str(executable), "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=text,
            timeout=15,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise batch.AdapterError("consumer_rollover_git_identity_invalid") from error
    return result.stdout


def _repo_path(config: Mapping[str, Any]) -> Path:
    raw = os.environ.get(
        CONSUMER_REPOSITORY_ENV,
        str(config["repository"]["path"]),
    )
    path = Path(raw).absolute()
    if not path.is_dir():
        raise batch.AdapterError("consumer_rollover_repository_invalid")
    base._reject_reparse_chain(path)
    return path


def _load_contract() -> tuple[dict[str, Any], bytes, bytes]:
    config_raw = _snapshot(
        CONFIG_PATH, reason="consumer_rollover_config_snapshot_drift"
    )
    schema_raw = _snapshot(
        SCHEMA_PATH, reason="consumer_rollover_schema_snapshot_drift"
    )
    config = _document(config_raw, reason="consumer_rollover_config_invalid")
    schema = _document(schema_raw, reason="consumer_rollover_schema_invalid")
    _validate(schema, config, reason="consumer_rollover_contract_schema_rejected")
    return config, config_raw, schema_raw


def _exact_git_identity(
    executable: Path,
    repo: Path,
    config: Mapping[str, Any],
) -> None:
    repository = config["repository"]
    commit = str(repository["commit"])
    branch = str(repository["branch"])
    observed = {
        "head": str(_git(executable, repo, "rev-parse", "HEAD")).strip(),
        "branch": str(_git(executable, repo, "branch", "--show-current")).strip(),
        "upstream": str(
            _git(
                executable,
                repo,
                "rev-parse",
                f"refs/remotes/origin/{branch}",
            )
        ).strip(),
        "parent": str(
            _git(executable, repo, "show", "-s", "--format=%P", commit)
        ).strip(),
        "tree": str(_git(executable, repo, "rev-parse", f"{commit}^{{tree}}")).strip(),
        "replace": str(
            _git(
                executable,
                repo,
                "for-each-ref",
                "--format=%(refname)",
                "refs/replace",
            )
        ).strip(),
        "grafts": str(
            _git(executable, repo, "rev-parse", "--git-path", "info/grafts")
        ).strip(),
    }
    grafts = (repo / observed["grafts"]).absolute()
    if observed["replace"] or (grafts.is_file() and grafts.stat().st_size):
        raise batch.AdapterError("consumer_rollover_git_rewrite_forbidden")
    if (
        observed["head"] != commit
        or observed["branch"] != branch
        or observed["upstream"] != commit
        or observed["parent"] != repository["parent"]
        or observed["tree"] != repository["tree"]
    ):
        raise batch.AdapterError("consumer_rollover_git_identity_drift")

    expected = list(config["exact_files"])
    inventory_sha256 = _sha256(_canonical(expected))
    if (
        len(expected) != repository["exact_file_count"]
        or inventory_sha256 != repository["exact_file_inventory_sha256"]
    ):
        raise batch.AdapterError("consumer_rollover_inventory_identity_drift")
    diff_lines = str(
        _git(
            executable,
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            commit,
        )
    ).splitlines()
    observed_diff = sorted(
        (line.split("\t", 1)[0], line.split("\t", 1)[1]) for line in diff_lines
    )
    expected_diff = sorted(
        (str(item["change"]), str(item["path"])) for item in expected
    )
    if observed_diff != expected_diff:
        raise batch.AdapterError("consumer_rollover_exact_file_set_drift")


def _authenticate_files(
    executable: Path,
    repo: Path,
    config: Mapping[str, Any],
) -> dict[str, bytes]:
    commit = str(config["repository"]["commit"])
    snapshots: dict[str, bytes] = {}
    for item in config["exact_files"]:
        relative = str(item["path"])
        path = (repo / relative).absolute()
        if repo not in path.parents or not path.is_file():
            raise batch.AdapterError("consumer_rollover_physical_path_invalid")
        base._reject_reparse_chain(path, stop=repo)
        raw = _snapshot(path, reason="consumer_rollover_physical_snapshot_drift")
        blob = _git(
            executable,
            repo,
            "cat-file",
            "blob",
            f"{commit}:{relative}",
            text=False,
        )
        if (
            not isinstance(blob, bytes)
            or raw != blob
            or len(raw) != item["bytes"]
            or _sha256(raw) != item["sha256"]
        ):
            raise batch.AdapterError("consumer_rollover_git_blob_worktree_drift")
        snapshots[relative] = raw
    return snapshots


def _authenticate_documents(
    config: Mapping[str, Any],
    snapshots: Mapping[str, bytes],
) -> None:
    pins = config["authenticated_documents"]
    for pin in pins.values():
        if snapshots.get(str(pin["path"])) is None:
            raise batch.AdapterError("consumer_rollover_document_not_in_exact_files")
        if _sha256(snapshots[str(pin["path"])]) != pin["sha256"]:
            raise batch.AdapterError("consumer_rollover_document_hash_drift")
    binding = _document(
        snapshots[pins["binding"]["path"]],
        reason="consumer_rollover_binding_invalid",
    )
    binding_schema = _document(
        snapshots[pins["binding_schema"]["path"]],
        reason="consumer_rollover_binding_schema_invalid",
    )
    receipt = _document(
        snapshots[pins["preflight_receipt"]["path"]],
        reason="consumer_rollover_preflight_receipt_invalid",
    )
    receipt_schema = _document(
        snapshots[pins["preflight_receipt_schema"]["path"]],
        reason="consumer_rollover_preflight_receipt_schema_invalid",
    )
    _validate(
        binding_schema,
        binding,
        reason="consumer_rollover_binding_schema_rejected",
    )
    _validate(
        receipt_schema,
        receipt,
        reason="consumer_rollover_preflight_receipt_schema_rejected",
    )
    receipt_sha256 = pins["preflight_receipt"]["sha256"]
    expected_sidecar = f"{receipt_sha256}  receipt.json\n".encode("ascii")
    if (
        snapshots[pins["preflight_receipt_sidecar"]["path"]] != expected_sidecar
        or receipt.get("binding_sha256") != pins["binding"]["sha256"]
        or receipt.get("binding_schema_sha256") != pins["binding_schema"]["sha256"]
        or receipt.get("implementation_sha256")
        != pins["preflight_implementation"]["sha256"]
        or binding.get("independent_review", {}).get("status") != "passed"
        or receipt.get("independent_review", {}).get("status") != "passed"
    ):
        raise batch.AdapterError("consumer_rollover_semantic_identity_drift")


def validate_consumer_identity_rollover(
    _controller: base.ControllerConfig | None = None,
) -> None:
    config, config_raw, schema_raw = _load_contract()
    repo = _repo_path(config)
    executable = _authenticated_git_executable(config)
    _exact_git_identity(executable, repo, config)
    snapshots = _authenticate_files(executable, repo, config)
    _authenticate_documents(config, snapshots)
    terminal_executable = _authenticated_git_executable(config)
    if terminal_executable != executable:
        raise batch.AdapterError("consumer_rollover_git_executable_terminal_drift")
    _exact_git_identity(terminal_executable, repo, config)
    terminal = _authenticate_files(terminal_executable, repo, config)
    if terminal != snapshots:
        raise batch.AdapterError("consumer_rollover_terminal_identity_drift")
    if (
        _snapshot(CONFIG_PATH, reason="consumer_rollover_config_terminal_drift")
        != config_raw
        or _snapshot(SCHEMA_PATH, reason="consumer_rollover_schema_terminal_drift")
        != schema_raw
    ):
        raise batch.AdapterError("consumer_rollover_contract_terminal_drift")


def preflight(*, validate_source: bool) -> dict[str, Any]:
    contract, contract_raw, _ = _load_contract()
    value = integrated.load_integrated_config(INTEGRATED_CONFIG_PATH)
    bound = integrated.bind_profiles(value)
    result = dict(
        base._preflight(
            bound,
            validate_source=validate_source,
            consumer_validator=validate_consumer_identity_rollover,
        )
    )
    integrated._recheck_integrated(value)
    result.update(
        {
            "schema_version": integrated.STATUS_SCHEMA_VERSION,
            "integrated_controller_config_sha256": value.physical_sha256,
            "integrated_controller_implementation_sha256": value.implementation_sha256,
            "teacher_finalizer_config_sha256": value.teacher_finalizer["config_sha256"],
            "teacher_finalizer_implementation_sha256": value.teacher_finalizer[
                "implementation_sha256"
            ],
            "consumer_identity_rollover_sha256": _sha256(contract_raw),
            "consumer_commit": contract["repository"]["commit"],
            "consumer_exact_files": contract["repository"]["exact_file_count"],
            "teacher_chain_consumer_consistent": len(
                {
                    contract["teacher_chain"]["finalizer"]["commit"],
                    contract["teacher_chain"]["final_manifest"]["commit"],
                    contract["teacher_chain"]["external_binding"]["consumer_commit"],
                }
            )
            == 1,
            "teacher_finalization_position": "before_runtime_slots_close",
            "external_release_review_required": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
        }
    )
    return result


async def _execute() -> dict[str, Any]:
    value = integrated.load_integrated_config(INTEGRATED_CONFIG_PATH)
    bound = integrated.bind_profiles(value)
    base._reject_environment_credentials(os.environ)
    report = preflight(validate_source=True)
    non_secret = [
        item
        for item in report["blockers"]
        if item
        not in {
            "controller_credential_slot_unloaded",
            "runtime_hmac_slot_unloaded",
        }
    ]
    if non_secret:
        raise batch.AdapterError(non_secret[0])
    base._guard_output_root(bound.values["smoke_exact1"], os.environ)
    base._validate_anonymous_os_channel(sys.stdin.buffer)
    slots = batch.RuntimeSecretSlots.from_anonymous_stdin(sys.stdin.buffer)
    controller = integrated.IntegratedSingleProcessController(
        value,
        bound,
        slots,
        consumer_validator=validate_consumer_identity_rollover,
    )
    return await controller.run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consumer-rollover single-process GLM controller"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--credential-stdin", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        base._reject_argv_credentials(raw_argv)
    except batch.AdapterError as error:
        print(json.dumps(base._blocked(error.reason_code), sort_keys=True))
        return 2
    parser = _parser()
    args = parser.parse_args(raw_argv)
    if args.execute != args.credential_stdin:
        parser.error("--execute requires --credential-stdin and vice versa")
    try:
        if args.dry_run:
            result = preflight(validate_source=False)
        elif args.validate_only:
            result = preflight(validate_source=True)
        else:
            result = asyncio.run(_execute())
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("state") == "complete" else 2
    except BaseException as error:
        print(
            json.dumps(
                base._blocked(base._body_free_reason(error)),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
