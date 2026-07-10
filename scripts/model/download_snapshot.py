"""Resumable Hugging Face snapshot download with a small status manifest."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, object] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(values)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--download-timeout", type=float, default=120.0)
    args = parser.parse_args()

    write_status(
        args.status,
        state="running",
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(args.local_dir.resolve()),
        pid=os.getpid(),
        started_at=utc_now(),
        finished_at=None,
        error=None,
    )
    try:
        kwargs: dict[str, object] = {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "max_workers": args.max_workers,
        }
        if args.cache_dir:
            kwargs["cache_dir"] = str(args.cache_dir)
        else:
            kwargs["local_dir"] = str(args.local_dir)
        parameters = inspect.signature(snapshot_download).parameters
        # huggingface_hub >=1.x resumes local-dir downloads automatically. These
        # switches keep the helper compatible with the older Anaconda base copy.
        if "local_dir_use_symlinks" in parameters:
            kwargs["local_dir_use_symlinks"] = False
        if "resume_download" in parameters:
            kwargs["resume_download"] = True
            # huggingface_hub 0.16 hard-codes a short streaming read timeout and
            # has no public environment override. Patch only that legacy helper;
            # newer releases honor HF_HUB_DOWNLOAD_TIMEOUT themselves.
            import huggingface_hub.file_download as file_download

            original_http_get = file_download.http_get

            def http_get_with_timeout(*http_args: object, **http_kwargs: object) -> object:
                http_kwargs["timeout"] = max(
                    float(http_kwargs.get("timeout", 0) or 0), args.download_timeout
                )
                return original_http_get(*http_args, **http_kwargs)

            file_download.http_get = http_get_with_timeout
        result = snapshot_download(**kwargs)
    except Exception as exc:
        write_status(
            args.status,
            state="failed",
            finished_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    snapshot_path = Path(result).resolve()
    if args.cache_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        for source in snapshot_path.rglob("*"):
            if not source.is_file():
                continue
            target = args.local_dir / source.relative_to(snapshot_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != source.stat().st_size:
                    raise RuntimeError(f"existing target differs from snapshot: {target}")
                continue
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)

    write_status(
        args.status,
        state="complete",
        finished_at=utc_now(),
        snapshot_path=str(snapshot_path),
        materialized_path=str(args.local_dir.resolve()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
