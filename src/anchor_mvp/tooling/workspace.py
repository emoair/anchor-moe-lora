from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import uuid

from .models import FileChange


_IGNORED_DIRECTORIES = {".git", ".hg", ".svn", ".anchor"}


def safe_sample_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise ValueError("sample_id must contain a letter or digit")
    return cleaned[:80]


def _assert_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes isolated workspace: {path}") from exc


def _copy_tree_no_links(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        if item.name in _IGNORED_DIRECTORIES:
            continue
        if item.is_symlink():
            raise ValueError(f"symlinks are not accepted in sample sources: {item}")
        target = destination / item.name
        if item.is_dir():
            target.mkdir()
            _copy_tree_no_links(item, target)
        else:
            shutil.copy2(item, target)


class WorkspaceManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def prepare(self, sample_id: str, source_dir: str | Path) -> Path:
        source = Path(source_dir).resolve()
        if not source.is_dir():
            raise ValueError(f"sample source is not a directory: {source}")
        workspace = self.root / f"{safe_sample_id(sample_id)}--{uuid.uuid4().hex[:10]}"
        _assert_within(workspace, self.root)
        workspace.mkdir()
        _copy_tree_no_links(source, workspace)
        return workspace


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(root: str | Path) -> dict[str, str]:
    base = Path(root).resolve()
    snapshot: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base)
        if any(part in _IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"symlink appeared inside isolated workspace: {path}")
        if path.is_file():
            snapshot[relative.as_posix()] = _file_digest(path)
    return snapshot


def diff_snapshots(before: dict[str, str], after: dict[str, str]) -> tuple[FileChange, ...]:
    changes: list[FileChange] = []
    for path in sorted(set(before) | set(after)):
        before_hash = before.get(path)
        after_hash = after.get(path)
        if before_hash == after_hash:
            continue
        operation = "added" if before_hash is None else "deleted" if after_hash is None else "modified"
        changes.append(FileChange(path, operation, before_hash, after_hash))
    return tuple(changes)
