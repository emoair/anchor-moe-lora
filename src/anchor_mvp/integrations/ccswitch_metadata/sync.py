"""Verified snapshot resolution and transactional local metadata installation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import resources
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from .constants import (
    ALLOWED_SOURCE_URLS,
    EXPECTED_SOURCE_FILES,
    MAX_SNAPSHOT_BYTES,
    MAX_SOURCE_BYTES,
    SOURCE_COMMIT,
    SOURCE_TAG,
)
from .schema import (
    SchemaError,
    load_snapshot,
    parse_snapshot_bytes,
    safe_json_bytes,
    validate_snapshot,
)


class MetadataSyncError(RuntimeError):
    """Base error for sync, storage, or trust failures."""


class NetworkUnavailable(MetadataSyncError):
    """Pinned sources could not be reached, so a verified snapshot may be reused."""


class IntegrityError(MetadataSyncError):
    """Remote bytes or local rollback state failed a trust check."""


Fetch = Callable[[str, str | None, int], tuple[int, bytes, str | None, str]]


@dataclass(frozen=True)
class Candidate:
    snapshot: dict[str, Any]
    origin: str
    verification: str
    warning: str | None = None

    @property
    def sha256(self) -> str:
        return snapshot_sha256(self.snapshot)


class SourceVerifier:
    """Hash official, commit-pinned source anchors without importing their code."""

    def __init__(self, state_dir: str | Path, *, fetch: Fetch | None = None) -> None:
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.cache_path = self.state_dir / "verification_cache.json"
        self.fetch = fetch or _https_fetch

    def verify(self, snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
        validate_snapshot(snapshot)
        old_records = self._load_cache()
        new_records: dict[str, dict[str, Any]] = {}
        results: list[dict[str, Any]] = []
        for source in snapshot["source"]["files"]:
            path = source["path"]
            previous = old_records.get(path, {})
            previous_etag = previous.get("etag")
            status, data, etag, final_url = self.fetch(
                source["url"], previous_etag, MAX_SOURCE_BYTES
            )
            if final_url != source["url"] or final_url not in ALLOWED_SOURCE_URLS:
                raise IntegrityError(f"source redirect escaped the allowlist: {path}")
            if status == 304:
                if not _cached_record_matches(previous, source):
                    raise IntegrityError(
                        f"received 304 without a matching verified cache record: {path}"
                    )
                record = dict(previous)
                record["verified_at"] = _utc_now()
                response_etag = _safe_etag(etag)
                record["etag"] = (
                    response_etag if response_etag is not None else previous.get("etag")
                )
                result_status = "not_modified"
            elif status == 200:
                actual_size = len(data)
                actual_sha = sha256(data).hexdigest()
                if actual_size != source["size"]:
                    raise IntegrityError(
                        f"source size mismatch for {path}: {actual_size} != {source['size']}"
                    )
                if actual_sha != source["sha256"]:
                    raise IntegrityError(f"source SHA-256 mismatch for {path}")
                record = {
                    "url": source["url"],
                    "sha256": actual_sha,
                    "size": actual_size,
                    "etag": _safe_etag(etag),
                    "verified_at": _utc_now(),
                }
                result_status = "verified"
            else:
                raise NetworkUnavailable(f"unexpected HTTP status {status} for {path}")
            new_records[path] = record
            results.append(
                {
                    "path": path,
                    "status": result_status,
                    "sha256": record["sha256"],
                    "size": record["size"],
                }
            )

        cache = {
            "schema_version": 1,
            "source_tag": SOURCE_TAG,
            "source_commit": SOURCE_COMMIT,
            "records": new_records,
        }
        _atomic_write(self.cache_path, _internal_json_bytes(cache))
        return results

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            if self.cache_path.is_symlink():
                return {}
            data = self.cache_path.read_bytes()
            if len(data) > MAX_SNAPSHOT_BYTES:
                return {}
            value = json.loads(data)
            if not isinstance(value, dict) or set(value) != {
                "schema_version",
                "source_tag",
                "source_commit",
                "records",
            }:
                return {}
            if (
                value["schema_version"] != 1
                or value["source_tag"] != SOURCE_TAG
                or value["source_commit"] != SOURCE_COMMIT
                or not isinstance(value["records"], dict)
            ):
                return {}
            records: dict[str, dict[str, Any]] = {}
            for path, record in value["records"].items():
                if path not in EXPECTED_SOURCE_FILES or not isinstance(record, dict):
                    return {}
                if set(record) != {"url", "sha256", "size", "etag", "verified_at"}:
                    return {}
                expected = EXPECTED_SOURCE_FILES[path]
                if (
                    record["url"] not in ALLOWED_SOURCE_URLS
                    or record["sha256"] != expected["sha256"]
                    or record["size"] != expected["size"]
                    or (
                        record["etag"] is not None
                        and _safe_etag(record["etag"]) is None
                    )
                ):
                    return {}
                records[path] = record
            return records
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}


def load_bundled_snapshot() -> dict[str, Any]:
    fixture = resources.files(__package__).joinpath(
        "fixtures", "cc_switch_v3_16_5.json"
    )
    return parse_snapshot_bytes(fixture.read_bytes(), origin=str(fixture))


def resolve_candidate(
    state_dir: str | Path,
    *,
    offline: bool = False,
    fetch: Fetch | None = None,
) -> Candidate:
    """Resolve a verified candidate, falling back only on network unavailability."""

    state = Path(state_dir).expanduser().resolve()
    _ensure_private_dir(state)
    bundled = load_bundled_snapshot()
    last_path = state / "last_verified_snapshot.json"

    if offline:
        try:
            if last_path.exists() and not last_path.is_symlink():
                return Candidate(
                    load_snapshot(last_path),
                    origin="last_verified_snapshot",
                    verification="offline",
                )
        except SchemaError:
            pass
        return Candidate(
            bundled,
            origin="bundled_verified_snapshot",
            verification="offline",
        )

    try:
        SourceVerifier(state, fetch=fetch).verify(bundled)
    except NetworkUnavailable as exc:
        try:
            if last_path.exists() and not last_path.is_symlink():
                return Candidate(
                    load_snapshot(last_path),
                    origin="last_verified_snapshot",
                    verification="network_fallback",
                    warning=str(exc),
                )
        except SchemaError:
            pass
        return Candidate(
            bundled,
            origin="bundled_verified_snapshot",
            verification="network_fallback",
            warning=str(exc),
        )
    _atomic_write(last_path, safe_json_bytes(bundled))
    return Candidate(
        bundled,
        origin="pinned_github_verified_snapshot",
        verification="verified",
    )


def snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    return sha256(safe_json_bytes(snapshot)).hexdigest()


def semantic_diff(
    current: Mapping[str, Any] | None, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    validate_snapshot(candidate)
    if current is not None:
        validate_snapshot(current)
    sections: dict[str, dict[str, list[str]]] = {}
    for name, key in (
        ("providers", lambda item: item["id"]),
        ("models", lambda item: item["id"]),
        (
            "model_aliases",
            lambda item: f"{item['provider_id']}:{item['alias']}",
        ),
        ("pricing", lambda item: item["model_id"]),
    ):
        before = {} if current is None else {key(item): item for item in current[name]}
        after = {key(item): item for item in candidate[name]}
        sections[name] = {
            "added": sorted(after.keys() - before.keys()),
            "removed": sorted(before.keys() - after.keys()),
            "changed": sorted(
                item_key
                for item_key in before.keys() & after.keys()
                if before[item_key] != after[item_key]
            ),
        }
    sections["contracts"] = {
        "added": [],
        "removed": [],
        "changed": [
            name
            for name in ("source", "token_billing")
            if current is None or current[name] != candidate[name]
        ],
    }
    current_sha = snapshot_sha256(current) if current is not None else None
    candidate_sha = snapshot_sha256(candidate)
    return {
        "changed": current_sha != candidate_sha,
        "current_sha256": current_sha,
        "candidate_sha256": candidate_sha,
        "sections": sections,
    }


class MetadataStore:
    """Atomic active-snapshot store with a target-specific rollback journal."""

    def __init__(self, state_dir: str | Path, target: str | Path | None = None) -> None:
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.target = (
            Path(target).expanduser().resolve()
            if target is not None
            else self.state_dir / "active.json"
        )
        self.backup_dir = self.state_dir / "backups"
        self.index_path = self.state_dir / "backup_index.json"
        self.lock_path = self.state_dir / ".sync.lock"
        self.target_id = sha256(os.fsencode(str(self.target))).hexdigest()

    def current(self) -> dict[str, Any] | None:
        if not self.target.exists():
            return None
        if self.target.is_symlink():
            raise IntegrityError("active metadata target may not be a symbolic link")
        return load_snapshot(self.target)

    def apply(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        validate_snapshot(snapshot)
        _ensure_private_dir(self.state_dir)
        _ensure_private_dir(self.backup_dir)
        with _exclusive_lock(self.lock_path):
            before = self.current()
            before_sha = snapshot_sha256(before) if before is not None else None
            after_sha = snapshot_sha256(snapshot)
            if before_sha == after_sha:
                return {
                    "changed": False,
                    "target": str(self.target),
                    "sha256": after_sha,
                    "backup_id": None,
                }

            backup_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup_id += "-" + uuid4().hex[:12]
            snapshot_file: str | None = None
            if before is not None:
                snapshot_file = backup_id + ".json"
                _atomic_write(
                    self.backup_dir / snapshot_file,
                    safe_json_bytes(before),
                )
            entry = {
                "id": backup_id,
                "target_id": self.target_id,
                "created_at": _utc_now(),
                "previous_exists": before is not None,
                "snapshot_file": snapshot_file,
                "sha256": before_sha,
            }
            index = self._load_index()
            index["entries"].append(entry)
            _atomic_write(self.index_path, _internal_json_bytes(index))
            _atomic_write(self.target, safe_json_bytes(snapshot))
            return {
                "changed": True,
                "target": str(self.target),
                "sha256": after_sha,
                "backup_id": backup_id,
            }

    def rollback(self) -> dict[str, Any]:
        _ensure_private_dir(self.state_dir)
        with _exclusive_lock(self.lock_path):
            index = self._load_index()
            selected_index = next(
                (
                    position
                    for position in range(len(index["entries"]) - 1, -1, -1)
                    if index["entries"][position]["target_id"] == self.target_id
                ),
                None,
            )
            if selected_index is None:
                raise MetadataSyncError("no rollback snapshot exists for this target")
            entry = index["entries"][selected_index]
            if entry["previous_exists"]:
                snapshot_name = entry["snapshot_file"]
                if (
                    not isinstance(snapshot_name, str)
                    or Path(snapshot_name).name != snapshot_name
                    or not snapshot_name.endswith(".json")
                ):
                    raise IntegrityError("rollback snapshot path is invalid")
                restored = load_snapshot(self.backup_dir / snapshot_name)
                restored_sha = snapshot_sha256(restored)
                if restored_sha != entry["sha256"]:
                    raise IntegrityError("rollback snapshot SHA-256 mismatch")
                _atomic_write(self.target, safe_json_bytes(restored))
                target_exists = True
            else:
                if self.target.is_symlink():
                    raise IntegrityError(
                        "active metadata target may not be a symbolic link"
                    )
                if self.target.exists():
                    self.target.unlink()
                restored_sha = None
                target_exists = False
            del index["entries"][selected_index]
            _atomic_write(self.index_path, _internal_json_bytes(index))
            return {
                "changed": True,
                "target": str(self.target),
                "target_exists": target_exists,
                "sha256": restored_sha,
                "backup_id": entry["id"],
            }

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"schema_version": 1, "entries": []}
        if self.index_path.is_symlink():
            raise IntegrityError("backup index may not be a symbolic link")
        try:
            value = json.loads(self.index_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"invalid backup index: {exc}") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "entries"}
            or value["schema_version"] != 1
            or not isinstance(value["entries"], list)
        ):
            raise IntegrityError("backup index schema is invalid")
        for entry in value["entries"]:
            if not isinstance(entry, dict) or set(entry) != {
                "id",
                "target_id",
                "created_at",
                "previous_exists",
                "snapshot_file",
                "sha256",
            }:
                raise IntegrityError("backup index entry schema is invalid")
            if not isinstance(entry["previous_exists"], bool):
                raise IntegrityError("backup index previous_exists must be boolean")
            if not isinstance(entry["target_id"], str) or len(entry["target_id"]) != 64:
                raise IntegrityError("backup index target id is invalid")
            if entry["previous_exists"]:
                if not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64:
                    raise IntegrityError("backup index SHA-256 is invalid")
                if not isinstance(entry["snapshot_file"], str):
                    raise IntegrityError("backup index snapshot file is invalid")
            elif entry["sha256"] is not None or entry["snapshot_file"] is not None:
                raise IntegrityError("absence rollback entry must not name a snapshot")
        return value


def default_state_dir() -> Path:
    configured = os.environ.get("ANCHOR_CCSWITCH_METADATA_STATE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".anchor-moe-lora" / "ccswitch-metadata"


def _https_fetch(
    url: str, etag: str | None, maximum_bytes: int
) -> tuple[int, bytes, str | None, str]:
    if url not in ALLOWED_SOURCE_URLS:
        raise IntegrityError("network URL is not on the pinned source allowlist")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "raw.githubusercontent.com":
        raise IntegrityError("network URL must use official GitHub HTTPS raw content")
    headers = {
        "Accept": "text/plain, application/octet-stream",
        "Accept-Encoding": "identity",
        "User-Agent": "anchor-moe-lora-ccswitch-metadata/1",
    }
    if etag:
        headers["If-None-Match"] = etag
    request = Request(url, headers=headers, method="GET")
    try:
        response = urlopen(request, timeout=15)
    except HTTPError as exc:
        if exc.code == 304:
            return 304, b"", exc.headers.get("ETag"), url
        raise NetworkUnavailable(f"GitHub returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise NetworkUnavailable(
            f"GitHub source verification unavailable: {exc}"
        ) from exc
    try:
        final_url = response.geturl()
        status = getattr(response, "status", response.getcode())
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > maximum_bytes:
                    raise IntegrityError("source Content-Length exceeds the size limit")
            except ValueError as exc:
                raise IntegrityError("source Content-Length is invalid") from exc
        data = response.read(maximum_bytes + 1)
        if len(data) > maximum_bytes:
            raise IntegrityError("source body exceeds the size limit")
        return status, data, response.headers.get("ETag"), final_url
    except (TimeoutError, socket.timeout, OSError) as exc:
        raise NetworkUnavailable(f"GitHub source read failed: {exc}") from exc
    finally:
        response.close()


def _cached_record_matches(
    record: Mapping[str, Any], source: Mapping[str, Any]
) -> bool:
    return (
        record.get("url") == source["url"]
        and record.get("sha256") == source["sha256"]
        and record.get("size") == source["size"]
        and isinstance(record.get("verified_at"), str)
    )


def _safe_etag(value: Any) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        return None
    return value


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _atomic_write(path: Path, data: bytes) -> None:
    _ensure_private_dir(path.parent)
    if path.is_symlink():
        raise IntegrityError(f"refusing to replace symbolic link: {path}")
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        for attempt in range(3):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))
        _fsync_directory(path.parent)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(descriptor, f"pid={os.getpid()} created_at={_utc_now()}\n".encode())
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise MetadataSyncError(
            f"metadata sync is already locked; inspect and remove stale lock {path}"
        ) from exc
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _internal_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
