#!/usr/bin/env python3
"""Validate the final train-only workspace state without reading hidden tests.

The command emits a final, machine-readable JSON line.  It intentionally binds
validation to the current Git diff, including untracked files, so a later edit
cannot reuse an earlier successful result.  This is execution evidence, not an
official SWE-bench verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


SCHEMA = "anchor.train-sandbox-validation.v1"
VERSION = "1.0.1"
SUPPORTED_CODE_SUFFIXES = frozenset({".cjs", ".js", ".mjs", ".py"})
IGNORED_NON_CODE_SUFFIXES = frozenset(
    {
        "",
        ".css",
        ".csv",
        ".html",
        ".json",
        ".lock",
        ".md",
        ".rst",
        ".svg",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)


class ValidationError(RuntimeError):
    """Fail-closed validation error with a stable content-free code."""


def _run_bytes(args: list[str], *, timeout: int = 120) -> bytes:
    result = subprocess.run(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise ValidationError("git_workspace_inspection_failed")
    return result.stdout


def _git_args(root: Path, *args: str) -> list[str]:
    """Build a Git command that trusts only this mounted workspace."""

    resolved = root.resolve(strict=True)
    return [
        "git",
        "-c",
        f"safe.directory={resolved}",
        "-C",
        str(resolved),
        *args,
    ]


def _nul_paths(payload: bytes) -> tuple[str, ...]:
    values: list[str] = []
    for raw in payload.split(b"\0"):
        if not raw:
            continue
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError("workspace_path_not_utf8") from exc
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or "\x00" in value:
            raise ValidationError("workspace_path_invalid")
        values.append(path.as_posix())
    return tuple(values)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _workspace_state(root: Path) -> tuple[tuple[str, ...], str]:
    if not (root / ".git").exists():
        raise ValidationError("git_workspace_missing")
    tracked = _nul_paths(
        _run_bytes(
            _git_args(
                root,
                "diff",
                "--name-only",
                "--diff-filter=ACMRTUXB",
                "-z",
                "HEAD",
                "--",
            )
        )
    )
    deleted = _nul_paths(
        _run_bytes(
            _git_args(
                root,
                "diff",
                "--name-only",
                "--diff-filter=D",
                "-z",
                "HEAD",
                "--",
            )
        )
    )
    if deleted:
        raise ValidationError("changed_code_deletion_not_validated")
    untracked = _nul_paths(
        _run_bytes(
            _git_args(
                root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
            )
        )
    )
    changed = tuple(sorted(set((*tracked, *untracked))))
    if not changed:
        raise ValidationError("workspace_has_no_final_changes")
    patch = _run_bytes(
        _git_args(
            root,
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "HEAD",
            "--",
        )
    )
    untracked_bindings: list[dict[str, object]] = []
    for relative in sorted(set(untracked)):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValidationError("untracked_path_not_regular_file")
        data = candidate.read_bytes()
        untracked_bindings.append(
            {"path": relative, "sha256": _sha256(data), "size": len(data)}
        )
    state = {
        "changed_paths": list(changed),
        "tracked_patch_sha256": _sha256(patch),
        "untracked": untracked_bindings,
    }
    return changed, _sha256(_canonical(state))


def _changed_code_paths(changed: Iterable[str]) -> tuple[str, ...]:
    supported: list[str] = []
    unsupported: list[str] = []
    for value in changed:
        suffix = Path(value).suffix.casefold()
        if suffix in SUPPORTED_CODE_SUFFIXES:
            supported.append(value)
        elif suffix not in IGNORED_NON_CODE_SUFFIXES:
            unsupported.append(value)
    if unsupported:
        raise ValidationError("changed_code_language_unsupported")
    if not supported:
        raise ValidationError("changed_code_missing")
    return tuple(sorted(supported))


def _stream(args: list[str], *, root: Path, timeout: int) -> None:
    result = subprocess.run(
        args,
        cwd=root,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise ValidationError("workspace_validation_command_failed")


def _compile(root: Path, code_paths: Iterable[str]) -> tuple[str, ...]:
    validators: list[str] = []
    for relative in code_paths:
        suffix = Path(relative).suffix.casefold()
        if suffix == ".py":
            candidate = root / relative
            try:
                # Built-in compile validates the exact final bytes without
                # writing __pycache__ into the evidence-bound workspace.
                compile(candidate.read_bytes(), relative, "exec")
            except (OSError, SyntaxError, ValueError) as exc:
                raise ValidationError("workspace_validation_command_failed") from exc
            validators.append("python-compile")
        else:
            _stream(["node", "--check", relative], root=root, timeout=120)
            validators.append("node-check")
    return tuple(sorted(set(validators)))


def _native_test(root: Path, code_paths: tuple[str, ...]) -> str:
    suffixes = {Path(value).suffix.casefold() for value in code_paths}
    if suffixes <= {".cjs", ".js", ".mjs"}:
        package = root / "package.json"
        if not package.is_file() or not (root / "node_modules").is_dir():
            raise ValidationError("native_test_dependencies_unavailable")
        try:
            payload = json.loads(package.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError("package_json_invalid") from exc
        scripts = payload.get("scripts") if isinstance(payload, dict) else None
        if not isinstance(scripts, dict) or not isinstance(scripts.get("test"), str):
            raise ValidationError("native_test_command_missing")
        _stream(["npm", "test", "--", "--runInBand"], root=root, timeout=600)
        return "npm-test"
    if suffixes <= {".py"}:
        tests_present = (root / "tests").is_dir() or any(root.glob("test_*.py"))
        if not tests_present:
            raise ValidationError("native_test_command_missing")
        _stream(["python3", "-m", "pytest", "-q"], root=root, timeout=600)
        return "pytest"
    raise ValidationError("mixed_language_native_test_unsupported")


def _emit(*, mode: str, success: bool, **extra: object) -> None:
    print(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "validator_version": VERSION,
                "mode": mode,
                "success": success,
                "not_official_swebench_pass": True,
                **extra,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=("compile", "test"))
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print(VERSION)
        return 0
    mode = args.mode or "compile"
    root = Path.cwd().resolve()
    try:
        changed, state_sha256 = _workspace_state(root)
        code_paths = _changed_code_paths(changed)
        validators = list(_compile(root, code_paths))
        validation_level = "syntax"
        if mode == "test":
            validators.append(_native_test(root, code_paths))
            validation_level = "native_test"
        _emit(
            mode=mode,
            success=True,
            validation_level=validation_level,
            changed_paths=list(changed),
            changed_paths_sha256=_sha256(_canonical(list(changed))),
            final_state_sha256=state_sha256,
            validators=sorted(set(validators)),
        )
        return 0
    except (OSError, subprocess.TimeoutExpired, ValidationError) as exc:
        code = str(exc) if isinstance(exc, ValidationError) else "validator_runtime_failed"
        _emit(mode=mode, success=False, error_code=code)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
