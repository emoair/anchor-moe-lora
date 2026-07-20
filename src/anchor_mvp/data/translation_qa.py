"""Offline, fail-closed QA for the zh-CN training corpus copy.

The translation workflow deliberately has no provider or network integration.  It
binds four human/agent-edited JSONL shards to one immutable source registry, audits
every translated record against its source row, and atomically publishes a
bilingual snapshot plus the five canonical training files.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from anchor_mvp.training.schema import DatasetValidationError, validate_record


REGISTRY_SCHEMA = "anchor.compact-mvp-v2b-dataset-snapshot.v1"
TRANSLATION_MANIFEST_SCHEMA = "anchor.translation-snapshot.v1"
BILINGUAL_RECORD_SCHEMA = "anchor.bilingual-record.v1"
TARGET_LOCALE = "zh-CN"
TARGET_ID_SUFFIX = "::zh-CN"
COMPACT_ARTIFACT_PROTOCOL = "single_file_tsx_segmented_v1"
SHARD_NAMES = tuple(f"part-{index:03d}.jsonl" for index in range(4))
SOURCE_FILES = (
    ("data_plan.jsonl", "planner"),
    ("data_tool_policy.jsonl", "tool_policy"),
    ("data_frontend.jsonl", "frontend_gen"),
    ("data_review.jsonl", "frontend_review"),
    ("data_security.jsonl", "security_gate"),
)
ENVELOPE_FIELDS = {
    "source_path",
    "source_line",
    "source_id",
    "source_record_sha256",
    "target_locale",
    "translated_record",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_FORBIDDEN_PATH_PARTS = {"heldout", "holdout", "benchmark", "eval", "evaluation"}
_PROTECTED_FIELD_NAMES = {
    "artifact_protocol",
    "artifact_sha256",
    "candidate_artifact_sha256",
    "candidate_excerpt",
    "candidate_schema",
    "cap",
    "capability",
    "code",
    "code_security_synopsis",
    "corrected_artifact_sha256_prefix",
    "decision",
    "effect",
    "expert",
    "id",
    "language",
    "label",
    "method",
    "model",
    "path",
    "protocol",
    "proposal_labels",
    "provider",
    "role",
    "schema_version",
    "segment_count",
    "segment_index",
    "selection",
    "severity",
    "scope",
    "source_id",
    "status",
    "target_locale",
}
_PROTECTED_FIELD_SUFFIXES = (
    "_count",
    "_id",
    "_index",
    "_path",
    "_sha256",
    "_sha256_prefix",
)
_WHOLE_PROTOCOL_RE = re.compile(
    r"^(?:\[(?:BLOCK|PASS)\]|APPROVE|BLOCK|ESCALATE|"
    r"(?:PLAN|TOOL_POLICY|GENERATE_TSX_SEGMENT|REVIEW_TSX_SEGMENT|"
    r"SECURITY_GATE)\|)$"
)
_PROTECTED_TOKEN_PATTERNS = (
    re.compile(r"(?ms)(?:```[^\n]*\n.*?```|~~~[^\n]*\n.*?~~~)"),
    re.compile(r"(?<!`)`[^`\n]+`(?!`)"),
    re.compile(r"https?://[^\s<>\"'),.;:!?\]，。；：！？、（）【】《》]+"),
    re.compile(r"</?[A-Za-z][^>\n]*>"),
    re.compile(r"\[(?:BLOCK|PASS|APPROVE|ESCALATE|[A-Z][A-Z0-9_-]{1,31})\]"),
    re.compile(
        r"(?<![A-Z0-9_])(?:PLAN|TOOL_POLICY|GENERATE_TSX_SEGMENT|"
        r"REVIEW_TSX_SEGMENT|SECURITY_GATE)\|"
    ),
    re.compile(r"(?m)^(?:REQ|DEFECT|CANDIDATE):"),
    re.compile(r"\b(?:artifact|sha)=[A-Za-z0-9_.:-]+"),
)


class TranslationAuditError(ValueError):
    """Raised when a translation shard or source binding is unsafe."""


@dataclass(frozen=True)
class SourceRow:
    source_path: str
    source_line: int
    expert: str
    record_id: str
    record_sha256: str
    record: Mapping[str, Any]

    @property
    def locator(self) -> tuple[str, int]:
        return (self.source_path, self.source_line)


@dataclass(frozen=True)
class SourceInventory:
    source_dir: Path
    registry_path: Path
    registry_sha256: str
    snapshot_sha256: str
    manifest_entries: Mapping[str, Mapping[str, Any]]
    rows: tuple[SourceRow, ...]


@dataclass
class _RecordStats:
    translatable_fields: int = 0
    changed_fields: int = 0
    cjk_fields: int = 0
    protected_fields: int = 0
    protected_tokens: int = 0


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical bytes used for every row-level SHA-256 binding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def record_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _expected_compact_user_message(record: Mapping[str, Any]) -> str:
    """Rebuild the canonical compact-v2 prompt from structured fields."""

    expert = record.get("expert")
    inputs = record.get("input")
    if not isinstance(inputs, Mapping):
        raise TranslationAuditError("compact record input must be an object")
    try:
        if expert == "planner":
            return (
                f"PLAN|artifact={COMPACT_ARTIFACT_PROTOCOL}\n"
                f"requirement={inputs['requirement']}"
            )
        if expert == "tool_policy":
            return "TOOL_POLICY|" + _compact_json(inputs)
        if expert == "frontend_gen":
            return "GENERATE_TSX_SEGMENT|" + _compact_json(inputs)
        if expert == "frontend_review":
            return (
                f"REVIEW_TSX_SEGMENT|{inputs['segment_index'] + 1}/"
                f"{inputs['segment_count']}|"
                f"sha={inputs['corrected_artifact_sha256_prefix']}\n"
                f"REQ:{inputs['requirement']}\n"
                f"DEFECT:{inputs['known_benign_defect']}\n"
                f"CANDIDATE:\n{inputs['candidate_excerpt']}"
            )
        if expert == "security_gate":
            return "SECURITY_GATE|" + _compact_json(inputs)
    except (KeyError, TypeError, ValueError) as error:
        raise TranslationAuditError(
            f"{expert}: compact prompt fields are incomplete"
        ) from error
    raise TranslationAuditError(f"unsupported compact expert: {expert!r}")


def _audit_compact_message_binding(
    record: Mapping[str, Any], *, source: str
) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise TranslationAuditError(
            f"{source}: compact messages must be exactly user + assistant"
        )
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise TranslationAuditError(
            f"{source}: compact message roles must be user then assistant"
        )
    if messages[0].get("content") != _expected_compact_user_message(record):
        raise TranslationAuditError(
            f"{source}: user message is not canonical for translated input"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_manifest_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TranslationAuditError(f"{label}: source path must be non-empty text")
    normalised = value.replace("\\", "/")
    path = PurePosixPath(normalised)
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if path.is_absolute() or ".." in parts or (parts and ":" in parts[0]):
        raise TranslationAuditError(f"{label}: source path must be repository-relative")
    lowered = {part.casefold() for part in parts}
    forbidden = lowered.intersection(_FORBIDDEN_PATH_PARTS)
    if forbidden:
        raise TranslationAuditError(
            f"{label}: heldout/benchmark path is forbidden: {sorted(forbidden)}"
        )
    return "/".join(parts)


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TranslationAuditError(f"{label}: invalid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise TranslationAuditError(f"{label}: JSON root must be an object")
    return value


def _load_source_inventory(
    *,
    source_dir: str | Path,
    registry_path: str | Path,
    expected_snapshot_sha256: str | None,
) -> SourceInventory:
    root = Path(source_dir).expanduser().resolve(strict=True)
    registry_file = Path(registry_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise TranslationAuditError("source_dir must be a directory")
    if not registry_file.is_file() or registry_file.is_symlink():
        raise TranslationAuditError("source registry must be a regular file")
    if root.name != "candidate_dataset":
        raise TranslationAuditError("source_dir must be the candidate_dataset root")
    if (
        registry_file.parent != root
        or registry_file.name != "manifest.registry-formal-v2.json"
    ):
        raise TranslationAuditError(
            "source registry must be manifest.registry-formal-v2.json in source_dir"
        )
    _normalise_manifest_path(registry_file.name, label="source registry")
    registry = _read_json_object(registry_file, label="source registry")
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise TranslationAuditError("source registry schema is not supported")
    snapshot_sha256 = registry.get("snapshot_sha256")
    if not isinstance(snapshot_sha256, str) or not _SHA256_RE.fullmatch(
        snapshot_sha256
    ):
        raise TranslationAuditError("source registry snapshot_sha256 is invalid")
    if (
        expected_snapshot_sha256 is not None
        and snapshot_sha256 != expected_snapshot_sha256
    ):
        raise TranslationAuditError("source snapshot SHA-256 does not match expected")
    if registry.get("heldout_content_read") is not False:
        raise TranslationAuditError("source registry must prove heldout_content_read=false")
    if registry.get("benchmark_record_content_read") is not False:
        raise TranslationAuditError(
            "source registry must prove benchmark_record_content_read=false"
        )
    if registry.get("artifact_protocol") != COMPACT_ARTIFACT_PROTOCOL:
        raise TranslationAuditError("source registry artifact protocol is invalid")
    source_manifest = registry.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise TranslationAuditError("source registry source_manifest binding is missing")
    _normalise_manifest_path(
        source_manifest.get("path"), label="source registry source_manifest"
    )
    source_manifest_sha256 = source_manifest.get("sha256")
    if not isinstance(source_manifest_sha256, str) or not _SHA256_RE.fullmatch(
        source_manifest_sha256
    ):
        raise TranslationAuditError("source registry source_manifest SHA-256 is invalid")

    raw_files = registry.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != len(SOURCE_FILES):
        raise TranslationAuditError("source registry must contain exactly five files")
    expected_by_name = dict(SOURCE_FILES)
    entries_by_name: dict[str, Mapping[str, Any]] = {}
    source_paths: set[str] = set()
    for index, raw_entry in enumerate(raw_files):
        label = f"source registry files[{index}]"
        if not isinstance(raw_entry, Mapping):
            raise TranslationAuditError(f"{label}: entry must be an object")
        source_path = _normalise_manifest_path(raw_entry.get("path"), label=label)
        name = PurePosixPath(source_path).name
        if name not in expected_by_name:
            raise TranslationAuditError(f"{label}: non-training source file is forbidden")
        if name in entries_by_name or source_path in source_paths:
            raise TranslationAuditError(f"{label}: duplicate source file binding")
        if raw_entry.get("expert") != expected_by_name[name]:
            raise TranslationAuditError(f"{label}: expert/file binding is invalid")
        digest = raw_entry.get("sha256")
        size = raw_entry.get("bytes")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise TranslationAuditError(f"{label}: sha256 is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise TranslationAuditError(f"{label}: bytes is invalid")
        entries_by_name[name] = dict(raw_entry, path=source_path)
        source_paths.add(source_path)
    if set(entries_by_name) != set(expected_by_name):
        raise TranslationAuditError("source registry file allowlist is incomplete")

    rows: list[SourceRow] = []
    seen_ids: set[str] = set()
    for filename, expert in SOURCE_FILES:
        entry = entries_by_name[filename]
        path = root / filename
        if not path.is_file() or path.is_symlink():
            raise TranslationAuditError(f"source file is not regular: {filename}")
        if path.stat().st_size != entry["bytes"]:
            raise TranslationAuditError(f"source file byte size drifted: {filename}")
        if _file_sha256(path) != entry["sha256"]:
            raise TranslationAuditError(f"source file SHA-256 drifted: {filename}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise TranslationAuditError(
                            f"{filename}:{line_number}: blank source rows are forbidden"
                        )
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise TranslationAuditError(
                            f"{filename}:{line_number}: invalid source JSON"
                        ) from error
                    try:
                        observed_expert = validate_record(
                            record, source=f"{filename}:{line_number}"
                        )
                    except DatasetValidationError as error:
                        raise TranslationAuditError(str(error)) from error
                    if observed_expert != expert:
                        raise TranslationAuditError(
                            f"{filename}:{line_number}: expert/file mismatch"
                        )
                    _audit_compact_message_binding(
                        record, source=f"{filename}:{line_number}"
                    )
                    record_id = str(record["id"])
                    if record_id in seen_ids:
                        raise TranslationAuditError(
                            f"{filename}:{line_number}: duplicate source id {record_id!r}"
                        )
                    if record_id.endswith(TARGET_ID_SUFFIX):
                        raise TranslationAuditError(
                            f"{filename}:{line_number}: source already has target suffix"
                        )
                    seen_ids.add(record_id)
                    rows.append(
                        SourceRow(
                            source_path=str(entry["path"]),
                            source_line=line_number,
                            expert=expert,
                            record_id=record_id,
                            record_sha256=record_sha256(record),
                            record=record,
                        )
                    )
        except UnicodeDecodeError as error:
            raise TranslationAuditError(f"source file is not UTF-8: {filename}") from error
    if not rows:
        raise TranslationAuditError("source registry resolved to an empty corpus")
    return SourceInventory(
        source_dir=root,
        registry_path=registry_file,
        registry_sha256=_file_sha256(registry_file),
        snapshot_sha256=snapshot_sha256,
        manifest_entries=entries_by_name,
        rows=tuple(rows),
    )


def _atomic_directory(target: Path) -> tuple[Path, Path]:
    resolved = target.expanduser().resolve(strict=False)
    if resolved.exists():
        raise TranslationAuditError(f"output already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{resolved.name}.staging-", dir=resolved.parent)
    ).resolve()
    return resolved, staging


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(value) + b"\n" for value in values)


def prepare_translation_shards(
    *,
    source_dir: str | Path,
    registry_path: str | Path,
    shard_dir: str | Path,
    expected_snapshot_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Create four deterministic offline shard templates with source bindings."""

    inventory = _load_source_inventory(
        source_dir=source_dir,
        registry_path=registry_path,
        expected_snapshot_sha256=expected_snapshot_sha256,
    )
    requested_target = Path(shard_dir).expanduser().resolve(strict=False)
    if requested_target == inventory.source_dir or requested_target.is_relative_to(
        inventory.source_dir
    ):
        raise TranslationAuditError("translation shards must be outside source_dir")
    target, initial_staging = _atomic_directory(requested_target)
    staging: Path | None = initial_staging
    shards: list[list[Mapping[str, Any]]] = [[] for _ in SHARD_NAMES]
    for index, row in enumerate(inventory.rows):
        translated = copy.deepcopy(row.record)
        translated["id"] = row.record_id + TARGET_ID_SUFFIX
        shards[index % len(SHARD_NAMES)].append(
            {
                "source_path": row.source_path,
                "source_line": row.source_line,
                "source_id": row.record_id,
                "source_record_sha256": row.record_sha256,
                "target_locale": TARGET_LOCALE,
                "translated_record": translated,
            }
        )
    try:
        shard_entries: list[Mapping[str, Any]] = []
        for name, values in zip(SHARD_NAMES, shards):
            content = _jsonl_bytes(values)
            _write_bytes(staging / name, content)
            shard_entries.append(
                {
                    "path": name,
                    "records": len(values),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        work_manifest = {
            "schema_version": "anchor.translation-work-shards.v1",
            "target_locale": TARGET_LOCALE,
            "source_registry_sha256": inventory.registry_sha256,
            "source_snapshot_sha256": inventory.snapshot_sha256,
            "record_hash_algorithm": (
                "sha256(utf8(json.dumps(sort_keys=true,separators=(',',':'),"
                "ensure_ascii=false)))"
            ),
            "partition": "source registry order, global_index modulo 4",
            "records": len(inventory.rows),
            "shards": shard_entries,
            "heldout_content_read": False,
            "benchmark_record_content_read": False,
        }
        manifest_content = (
            json.dumps(work_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        _write_bytes(staging / "work_manifest.json", manifest_content)
        os.rename(staging, target)
        staging = None
        return work_manifest
    finally:
        if staging is not None and staging.is_dir():
            shutil.rmtree(staging)


def _protected_tokens(value: str) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for pattern in _PROTECTED_TOKEN_PATTERNS:
        tokens.update(pattern.findall(value))
    return tokens


def _path_text(path: Sequence[str | int]) -> str:
    result = "$"
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def _is_protected_field(path: Sequence[str | int], source: str) -> bool:
    string_parts = tuple(part for part in path if isinstance(part, str))
    if "provenance" in string_parts or "compact_v2" in string_parts:
        return True
    field = string_parts[-1] if string_parts else ""
    if field in _PROTECTED_FIELD_NAMES or field.endswith(_PROTECTED_FIELD_SUFFIXES):
        return True
    return bool(_WHOLE_PROTOCOL_RE.fullmatch(source.strip()))


def _audit_nodes(
    source: Any,
    target: Any,
    *,
    path: tuple[str | int, ...],
    stats: _RecordStats,
) -> None:
    label = _path_text(path)
    if type(source) is not type(target):
        raise TranslationAuditError(f"{label}: JSON value type changed")
    if isinstance(source, Mapping):
        if set(source) != set(target):
            raise TranslationAuditError(f"{label}: JSON object keys changed")
        for key in source:
            _audit_nodes(
                source[key],
                target[key],
                path=path + (str(key),),
                stats=stats,
            )
        return
    if isinstance(source, list):
        if len(source) != len(target):
            raise TranslationAuditError(f"{label}: JSON list length changed")
        for index, (source_item, target_item) in enumerate(zip(source, target)):
            _audit_nodes(
                source_item,
                target_item,
                path=path + (index,),
                stats=stats,
            )
        return
    if isinstance(source, str):
        if _is_protected_field(path, source):
            stats.protected_fields += 1
            if source != target:
                raise TranslationAuditError(f"{label}: protected field changed")
            return
        stats.translatable_fields += 1
        if source and not target.strip():
            raise TranslationAuditError(f"{label}: translation is empty")
        if not source and target:
            raise TranslationAuditError(f"{label}: empty source field changed")
        source_tokens = _protected_tokens(source)
        target_tokens = _protected_tokens(target)
        stats.protected_tokens += sum(source_tokens.values())
        if source_tokens != target_tokens:
            raise TranslationAuditError(
                f"{label}: code/URL/protocol tokens changed"
            )
        if source != target:
            stats.changed_fields += 1
        if _CJK_RE.search(target):
            stats.cjk_fields += 1
        return
    if source != target:
        raise TranslationAuditError(f"{label}: non-text value changed")


def _audit_translated_record(row: SourceRow, translated: Any) -> _RecordStats:
    if not isinstance(translated, Mapping):
        raise TranslationAuditError("translated_record must be a JSON object")
    expected_id = row.record_id + TARGET_ID_SUFFIX
    if translated.get("id") != expected_id:
        raise TranslationAuditError(
            f"{row.source_path}:{row.source_line}: translated id must be "
            f"{expected_id!r}"
        )
    if set(translated) != set(row.record):
        raise TranslationAuditError(
            f"{row.source_path}:{row.source_line}: top-level fields changed"
        )
    stats = _RecordStats(protected_fields=1)
    for key in row.record:
        if key == "id":
            continue
        _audit_nodes(
            row.record[key],
            translated[key],
            path=(str(key),),
            stats=stats,
        )
    if stats.translatable_fields == 0:
        raise TranslationAuditError(
            f"{row.source_path}:{row.source_line}: no translatable text fields"
        )
    if stats.changed_fields == 0:
        raise TranslationAuditError(
            f"{row.source_path}:{row.source_line}: untranslated record"
        )
    if stats.cjk_fields == 0:
        raise TranslationAuditError(
            f"{row.source_path}:{row.source_line}: no Chinese text detected"
        )
    try:
        observed_expert = validate_record(
            translated, source=f"{row.source_path}:{row.source_line}:zh-CN"
        )
    except DatasetValidationError as error:
        raise TranslationAuditError(str(error)) from error
    if observed_expert != row.expert:
        raise TranslationAuditError(
            f"{row.source_path}:{row.source_line}: translated expert changed"
        )
    _audit_compact_message_binding(
        translated, source=f"{row.source_path}:{row.source_line}:zh-CN"
    )
    return stats


def _resolve_shards(shard_paths: Sequence[str | Path]) -> tuple[Path, ...]:
    if len(shard_paths) != len(SHARD_NAMES):
        raise TranslationAuditError("exactly four translation shards are required")
    resolved: dict[str, Path] = {}
    for raw_path in shard_paths:
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_file() or path.is_symlink():
            raise TranslationAuditError("every translation shard must be a regular file")
        if path.name not in SHARD_NAMES or path.name in resolved:
            raise TranslationAuditError(
                f"translation shards must be exactly {list(SHARD_NAMES)}"
            )
        resolved[path.name] = path
    if set(resolved) != set(SHARD_NAMES):
        raise TranslationAuditError(
            f"translation shards must be exactly {list(SHARD_NAMES)}"
        )
    if len({path.parent for path in resolved.values()}) != 1:
        raise TranslationAuditError("all four translation shards must share one directory")
    return tuple(resolved[name] for name in SHARD_NAMES)


def _load_and_audit_shards(
    inventory: SourceInventory,
    shard_paths: Sequence[str | Path],
) -> tuple[Mapping[tuple[str, int], Mapping[str, Any]], list[Mapping[str, Any]], _RecordStats]:
    shards = _resolve_shards(shard_paths)
    expected = {row.locator: row for row in inventory.rows}
    by_locator: dict[tuple[str, int], Mapping[str, Any]] = {}
    seen_source_ids: set[str] = set()
    seen_target_ids: set[str] = set()
    seen_target_payloads: dict[str, str] = {}
    shard_entries: list[Mapping[str, Any]] = []
    totals = _RecordStats()
    allowed_paths = {row.source_path for row in inventory.rows}
    for shard in shards:
        rows_in_shard = 0
        try:
            with shard.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: blank shard rows are forbidden"
                        )
                    try:
                        envelope = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: invalid JSON"
                        ) from error
                    if not isinstance(envelope, Mapping):
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: envelope must be an object"
                        )
                    if set(envelope) != ENVELOPE_FIELDS:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: envelope fields are invalid"
                        )
                    if envelope.get("target_locale") != TARGET_LOCALE:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: target_locale must be zh-CN"
                        )
                    source_path = _normalise_manifest_path(
                        envelope.get("source_path"),
                        label=f"{shard.name}:{line_number}",
                    )
                    if source_path not in allowed_paths:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: source_path is not allowlisted"
                        )
                    source_line = envelope.get("source_line")
                    if (
                        not isinstance(source_line, int)
                        or isinstance(source_line, bool)
                        or source_line < 1
                    ):
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: source_line is invalid"
                        )
                    locator = (source_path, source_line)
                    row = expected.get(locator)
                    if row is None:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: source locator does not exist"
                        )
                    if locator in by_locator:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: duplicate source locator"
                        )
                    if envelope.get("source_id") != row.record_id:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: source_id binding mismatch"
                        )
                    if envelope.get("source_record_sha256") != row.record_sha256:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: source record SHA-256 mismatch"
                        )
                    translated = envelope["translated_record"]
                    stats = _audit_translated_record(row, translated)
                    target_id = str(translated["id"])
                    if row.record_id in seen_source_ids:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: duplicate source_id"
                        )
                    if target_id in seen_target_ids:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: duplicate translated id"
                        )
                    semantic_payload = {
                        key: value
                        for key, value in translated.items()
                        if key not in {"id", "provenance"}
                    }
                    semantic_sha256 = record_sha256(semantic_payload)
                    prior_target = seen_target_payloads.get(semantic_sha256)
                    if prior_target is not None:
                        raise TranslationAuditError(
                            f"{shard.name}:{line_number}: duplicate translated "
                            f"training payload also used by {prior_target!r}"
                        )
                    seen_source_ids.add(row.record_id)
                    seen_target_ids.add(target_id)
                    seen_target_payloads[semantic_sha256] = target_id
                    by_locator[locator] = translated
                    rows_in_shard += 1
                    totals.translatable_fields += stats.translatable_fields
                    totals.changed_fields += stats.changed_fields
                    totals.cjk_fields += stats.cjk_fields
                    totals.protected_fields += stats.protected_fields
                    totals.protected_tokens += stats.protected_tokens
        except UnicodeDecodeError as error:
            raise TranslationAuditError(f"{shard.name}: shard is not UTF-8") from error
        shard_entries.append(
            {
                "path": shard.name,
                "records": rows_in_shard,
                "bytes": shard.stat().st_size,
                "sha256": _file_sha256(shard),
            }
        )
    missing = set(expected).difference(by_locator)
    if missing:
        sample = sorted(missing)[:3]
        raise TranslationAuditError(
            f"translation is not one-to-one: {len(missing)} source rows are missing; "
            f"sample={sample}"
        )
    if len(by_locator) != len(expected):
        raise TranslationAuditError("translation row count differs from source")
    return by_locator, shard_entries, totals


def _build_publication(
    inventory: SourceInventory,
    translated: Mapping[tuple[str, int], Mapping[str, Any]],
    shard_entries: Sequence[Mapping[str, Any]],
    stats: _RecordStats,
) -> tuple[Mapping[str, bytes], Mapping[str, Any]]:
    output_files: dict[str, bytes] = {}
    file_entries: dict[str, Mapping[str, Any]] = {}
    bilingual_rows: list[Mapping[str, Any]] = []
    for filename, expert in SOURCE_FILES:
        source_entry = inventory.manifest_entries[filename]
        rows = [
            row
            for row in inventory.rows
            if PurePosixPath(row.source_path).name == filename
        ]
        target_records = [translated[row.locator] for row in rows]
        content = _jsonl_bytes(target_records)
        output_files[filename] = content
        file_entries[filename] = {
            "expert": expert,
            "source_path": source_entry["path"],
            "source_records": len(rows),
            "source_bytes": source_entry["bytes"],
            "source_sha256": source_entry["sha256"],
            "target_path": filename,
            "target_records": len(target_records),
            "target_bytes": len(content),
            "target_sha256": hashlib.sha256(content).hexdigest(),
        }
        for row in rows:
            target_record = translated[row.locator]
            bilingual_rows.append(
                {
                    "schema_version": BILINGUAL_RECORD_SCHEMA,
                    "source_locale": "en",
                    "target_locale": TARGET_LOCALE,
                    "source_path": row.source_path,
                    "source_line": row.source_line,
                    "source_id": row.record_id,
                    "source_record_sha256": row.record_sha256,
                    "target_id": target_record["id"],
                    "target_record_sha256": record_sha256(target_record),
                    "source_record": row.record,
                    "translated_record": target_record,
                }
            )
    bilingual_content = _jsonl_bytes(bilingual_rows)
    bilingual_name = "bilingual_snapshot.jsonl"
    output_files[bilingual_name] = bilingual_content
    bilingual_sha256 = hashlib.sha256(bilingual_content).hexdigest()
    snapshot_parts = [
        f"source:{inventory.snapshot_sha256}",
        *(f"{name}:{file_entries[name]['target_sha256']}" for name, _ in SOURCE_FILES),
        f"{bilingual_name}:{bilingual_sha256}",
    ]
    manifest = {
        "schema_version": TRANSLATION_MANIFEST_SCHEMA,
        "source_locale": "en",
        "target_locale": TARGET_LOCALE,
        "source": {
            "registry_path": inventory.registry_path.as_posix(),
            "registry_sha256": inventory.registry_sha256,
            "snapshot_sha256": inventory.snapshot_sha256,
            "record_hash_algorithm": (
                "sha256(utf8(json.dumps(sort_keys=true,separators=(',',':'),"
                "ensure_ascii=false)))"
            ),
        },
        "counts": {
            "source_records": len(inventory.rows),
            "translated_records": len(translated),
            "bilingual_records": len(bilingual_rows),
            "shards": len(shard_entries),
        },
        "files": file_entries,
        "shards": list(shard_entries),
        "quality": {
            "one_to_one": len(inventory.rows) == len(translated),
            "source_hashes_verified": True,
            "json_and_training_schema_valid": True,
            "json_shape_preserved": True,
            "source_id_lineage_preserved": True,
            "target_id_suffix": TARGET_ID_SUFFIX,
            "duplicate_source_locators": 0,
            "duplicate_source_ids": 0,
            "duplicate_target_ids": 0,
            "duplicate_translated_training_payloads": 0,
            "empty_or_untranslated_records": 0,
            "translatable_fields": stats.translatable_fields,
            "changed_fields": stats.changed_fields,
            "fields_with_cjk": stats.cjk_fields,
            "protected_fields_verified": stats.protected_fields,
            "protected_code_url_protocol_tokens_verified": stats.protected_tokens,
        },
        "bilingual_snapshot": {
            "path": bilingual_name,
            "records": len(bilingual_rows),
            "bytes": len(bilingual_content),
            "sha256": bilingual_sha256,
        },
        "exclusions": {
            "source_file_allowlist": [name for name, _ in SOURCE_FILES],
            "directory_scan_used": False,
            "heldout_content_read": False,
            "benchmark_record_content_read": False,
        },
        "snapshot_sha256": hashlib.sha256(
            "\n".join(snapshot_parts).encode("utf-8")
        ).hexdigest(),
    }
    return output_files, manifest


def merge_translation_shards(
    *,
    source_dir: str | Path,
    registry_path: str | Path,
    shard_paths: Sequence[str | Path],
    output_dir: str | Path,
    expected_snapshot_sha256: str | None = None,
    dry_run: bool = False,
) -> Mapping[str, Any]:
    """Audit four shards and atomically publish the zh-CN + bilingual snapshot."""

    inventory = _load_source_inventory(
        source_dir=source_dir,
        registry_path=registry_path,
        expected_snapshot_sha256=expected_snapshot_sha256,
    )
    requested_target = Path(output_dir).expanduser().resolve(strict=False)
    if requested_target == inventory.source_dir or requested_target.is_relative_to(
        inventory.source_dir
    ):
        raise TranslationAuditError("translation output must be outside source_dir")
    translated, shard_entries, stats = _load_and_audit_shards(
        inventory, shard_paths
    )
    output_files, manifest = _build_publication(
        inventory, translated, shard_entries, stats
    )
    if dry_run:
        return manifest

    target, initial_staging = _atomic_directory(requested_target)
    staging: Path | None = initial_staging
    try:
        for name, content in output_files.items():
            _write_bytes(staging / name, content)
        # Re-validate the exact bytes that will be published.
        for filename, expert in SOURCE_FILES:
            path = staging / filename
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        observed = validate_record(
                            json.loads(line), source=f"{filename}:{line_number}"
                        )
                        if observed != expert:
                            raise TranslationAuditError(
                                f"{filename}:{line_number}: published expert mismatch"
                            )
            except (DatasetValidationError, json.JSONDecodeError) as error:
                raise TranslationAuditError(
                    f"published training schema validation failed: {filename}"
                ) from error
        manifest_name = "manifest.translation-zh-CN.json"
        manifest_content = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _write_bytes(staging / manifest_name, manifest_content)
        manifest_digest = hashlib.sha256(manifest_content).hexdigest()
        _write_bytes(
            staging / f"{manifest_name}.sha256",
            f"{manifest_digest}  {manifest_name}\n".encode("ascii"),
        )
        os.rename(staging, target)
        staging = None
        return manifest
    finally:
        if staging is not None and staging.is_dir():
            shutil.rmtree(staging)
