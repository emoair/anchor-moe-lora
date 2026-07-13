"""Fail-closed, atomic merging for offline distillation shards."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from ..training.schema import DatasetValidationError, validate_record
from .cleaning import contains_secret_material, validate_safe_payload
from .schema import (
    DataValidationError,
    SeedDemand,
    canonical_input,
    normalized_text,
    stable_id,
    validate_output,
)
from .task_cards import assignment_for_seed, load_task_card_catalog


TASK_ORDER = ("plan", "tool_policy", "frontend", "review", "security")
FILES = ("seeds.jsonl", *(f"data_{task}.jsonl" for task in TASK_ORDER))
EXPECTED_EXPERT = {
    "plan": "planner",
    "tool_policy": "tool_policy",
    "frontend": "frontend_gen",
    "review": "frontend_review",
    "security": "security_gate",
}
SOURCE_FIELDS = {
    "plan": {},
    "tool_policy": {"source_plan_record_id": "plan"},
    "frontend": {
        "source_plan_record_id": "plan",
        "source_tool_policy_record_id": "tool_policy",
    },
    "review": {"source_frontend_record_id": "frontend"},
    "security": {"source_review_record_id": "review"},
}
RECORD_FIELDS = {
    "id",
    "expert",
    "messages",
    "input",
    "provenance",
    "decision_trace",
    "output",
    "schema_version",
}
INPUT_FIELDS = {
    "plan": {"requirement"},
    "tool_policy": {"requirement", "plan", "tool_proposals"},
    "frontend": {"requirement", "plan", "tool_policy"},
    "review": {"requirement", "candidate_code", "known_benign_defect"},
    "security": {"requirement", "reviewed_code"},
}
EMPTY_SHA256 = sha256(b"").hexdigest()
PARTIAL_CHAIN_LABELS = (
    "seed_only",
    "through_plan",
    "through_tool_policy",
    "through_frontend",
    "through_review",
)


class ShardMergeError(ValueError):
    """A shard cannot be merged without weakening corpus integrity."""


@dataclass(frozen=True)
class _Audit:
    rows: dict[str, tuple[dict[str, Any], ...]]
    manifest: dict[str, Any]


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ShardMergeError("record is not canonically JSON serializable") from error


def _content_fingerprint(value: Mapping[str, Any]) -> str:
    without_id = dict(value)
    without_id.pop("id", None)
    return sha256(_canonical_bytes(without_id)).hexdigest()


def _file_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _source_label(source_index: int, filename: str, line_number: int) -> str:
    return f"source[{source_index}]/{filename}:{line_number}"


def _load_jsonl(
    path: Path,
    *,
    source_index: int,
) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ShardMergeError(
            f"source[{source_index}]/{path.name} must be a regular non-symlink file"
        )
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            label = _source_label(source_index, path.name, line_number)
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ShardMergeError(f"{label}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ShardMergeError(f"{label}: row must be an object")
            rows.append((line_number, value))
    return rows


def _validate_seed(value: Mapping[str, Any], *, label: str) -> SeedDemand:
    legacy_fields = {"seed_id", "title", "request", "category", "tags"}
    old_slot_fields = {*legacy_fields, "card_id", "seed_index"}
    canonical_fields = {
        *old_slot_fields,
        "template_id",
        "source_kind",
    }
    external_fields = {*canonical_fields, "source_digest"}
    if set(value) not in {
        frozenset(legacy_fields),
        frozenset(old_slot_fields),
        frozenset(canonical_fields),
        frozenset(external_fields),
    }:
        raise ShardMergeError(f"{label}: seed schema is not canonical")
    if contains_secret_material(value):
        raise ShardMergeError(f"{label}: credential-like material detected")
    try:
        seed = SeedDemand.from_mapping(value)
    except ValueError as error:
        raise ShardMergeError(f"{label}: invalid seed schema") from error
    if not seed.seed_id.strip() or not seed.title.strip():
        raise ShardMergeError(f"{label}: seed id and title must be non-empty")
    if not isinstance(value.get("tags"), list):
        raise ShardMergeError(f"{label}: seed tags must be a list")
    if seed.card_id is not None:
        try:
            assignment_for_seed(seed, load_task_card_catalog())
        except ValueError as error:
            raise ShardMergeError(
                f"{label}: seed task-card binding is invalid"
            ) from error
    return seed


def _validate_task_record(
    task: str,
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if contains_secret_material(value):
        raise ShardMergeError(f"{label}: credential-like material detected")
    if set(value) != RECORD_FIELDS:
        raise ShardMergeError(f"{label}: record schema is not canonical")
    try:
        expert = validate_record(value, source=label)
    except DatasetValidationError as error:
        raise ShardMergeError(
            f"{label}: canonical dataset validation failed"
        ) from error
    if expert != EXPECTED_EXPERT[task]:
        raise ShardMergeError(f"{label}: expert does not match dataset task")
    messages = value["messages"]
    if len(messages) != 2 or [message["role"] for message in messages] != [
        "user",
        "assistant",
    ]:
        raise ShardMergeError(f"{label}: message sequence is not canonical")
    if any(set(message) != {"role", "content"} for message in messages):
        raise ShardMergeError(f"{label}: message schema is not canonical")
    trace = value["decision_trace"]
    if any(set(step) != {"check", "evidence", "action"} for step in trace):
        raise ShardMergeError(f"{label}: decision trace schema is not canonical")
    task_input = value.get("input")
    if not isinstance(task_input, Mapping) or set(task_input) != INPUT_FIELDS[task]:
        raise ShardMergeError(f"{label}: task input schema is not canonical")
    output = value.get("output")
    if not isinstance(output, Mapping):
        raise ShardMergeError(f"{label}: output must be an object")
    try:
        validate_output(task, output)  # type: ignore[arg-type]
    except DataValidationError as error:
        raise ShardMergeError(f"{label}: output schema is not canonical") from error
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ShardMergeError(f"{label}: provenance must be an object")
    seed_id = provenance.get("seed_id")
    if not isinstance(seed_id, str) or not seed_id.strip():
        raise ShardMergeError(f"{label}: provenance.seed_id is required")
    teacher = provenance.get("teacher")
    if not isinstance(teacher, Mapping):
        raise ShardMergeError(f"{label}: provenance.teacher is required")
    model = teacher.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ShardMergeError(f"{label}: provenance.teacher.model is required")
    observed_source_fields = {
        str(key)
        for key in provenance
        if str(key).startswith("source_") and str(key).endswith("_record_id")
    }
    expected_source_fields = set(SOURCE_FIELDS[task])
    if observed_source_fields != expected_source_fields:
        raise ShardMergeError(f"{label}: source reference schema is not canonical")
    for field in expected_source_fields:
        source_id = provenance.get(field)
        if not isinstance(source_id, str) or not source_id.strip():
            raise ShardMergeError(f"{label}: source reference is empty")
    try:
        validate_safe_payload(
            task,  # type: ignore[arg-type]
            {
                "input": value.get("input", {}),
                "output": value.get("output", {}),
            },
        )
    except ValueError as error:
        raise ShardMergeError(
            f"{label}: task payload failed safety validation"
        ) from error


def _audit_lineage(
    seeds: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    indexes = {
        task: {str(record["id"]): record for record in records[task]}
        for task in TASK_ORDER
    }
    expected_seed_ids = set(seeds)
    stage_seed_ids: dict[str, set[str]] = {}
    for task in TASK_ORDER:
        ordered_seed_ids = [
            str(record["provenance"]["seed_id"]) for record in records[task]
        ]
        task_seed_ids = set(ordered_seed_ids)
        if len(task_seed_ids) != len(ordered_seed_ids):
            raise ShardMergeError(
                f"{task}: merged corpus contains multiple records for one seed"
            )
        if not task_seed_ids.issubset(expected_seed_ids):
            raise ShardMergeError(
                f"{task}: record references a seed absent from merged seeds"
            )
        stage_seed_ids[task] = task_seed_ids
    for task in TASK_ORDER:
        for upstream_task in set(SOURCE_FIELDS[task].values()):
            if not stage_seed_ids[task].issubset(stage_seed_ids[upstream_task]):
                raise ShardMergeError(
                    f"{task}: dependency closure requires an existing "
                    f"{upstream_task} record for the same seed"
                )
    for task in TASK_ORDER:
        for record in records[task]:
            provenance = record["provenance"]
            seed_id = str(provenance["seed_id"])
            if seed_id not in seeds:
                raise ShardMergeError(
                    f"{task}: record references a seed absent from merged seeds"
                )
            seed = SeedDemand.from_mapping(seeds[seed_id])
            assignment = assignment_for_seed(seed, load_task_card_catalog())
            if not assignment.legacy:
                expected_card = assignment.provenance(seed_id)
                card_fields = set(expected_card) | {
                    "card_id",
                    "card_tags",
                    "alignment_id",
                    "task_card_legacy",
                    "task_card_catalog_sha256",
                    "seed_index",
                    "template_id",
                    "source_kind",
                    "source_digest",
                }
                observed_card = {
                    field: provenance[field]
                    for field in card_fields
                    if field in provenance
                }
                if observed_card != expected_card:
                    raise ShardMergeError(
                        f"{task}: task-card provenance does not match its seed"
                    )
            try:
                canonical_task_input, canonical_user = canonical_input(
                    task,  # type: ignore[arg-type]
                    seed,
                    {},
                    canonical_task_input=record["input"],
                )
            except DataValidationError as error:
                raise ShardMergeError(
                    f"{task}: record input cannot be reconstructed canonically"
                ) from error
            if canonical_task_input != record["input"]:
                raise ShardMergeError(f"{task}: task input is not canonical")
            if record["messages"][0]["content"] != canonical_user:
                raise ShardMergeError(
                    f"{task}: user message does not match canonical task input"
                )
            resolved: dict[str, Mapping[str, Any]] = {}
            for field, upstream_task in SOURCE_FIELDS[task].items():
                source_id = str(provenance[field])
                upstream = indexes[upstream_task].get(source_id)
                if upstream is None:
                    raise ShardMergeError(
                        f"{task}: source reference does not resolve uniquely"
                    )
                upstream_seed = str(upstream["provenance"]["seed_id"])
                if upstream_seed != seed_id:
                    raise ShardMergeError(
                        f"{task}: source reference crosses seed lineage"
                    )
                resolved[upstream_task] = upstream
            if task == "tool_policy" and (
                record["input"]["plan"] != resolved["plan"]["output"]
            ):
                raise ShardMergeError(
                    "tool_policy: embedded plan differs from source record"
                )
            if task == "frontend":
                plan_id = str(provenance["source_plan_record_id"])
                policy_id = str(provenance["source_tool_policy_record_id"])
                policy = indexes["tool_policy"][policy_id]
                if str(policy["provenance"]["source_plan_record_id"]) != plan_id:
                    raise ShardMergeError(
                        "frontend: planner lineage forks across tool-policy input"
                    )
                if record["input"]["plan"] != resolved["plan"]["output"]:
                    raise ShardMergeError(
                        "frontend: embedded plan differs from source record"
                    )
                if record["input"]["tool_policy"] != resolved["tool_policy"]["output"]:
                    raise ShardMergeError(
                        "frontend: embedded tool policy differs from source record"
                    )

    # A raw collection is a stage funnel, not a rectangular table. Existing
    # downstream records have already proven their complete transitive source
    # closure above; absent downstream stages remain valid partial chains.
    partial_distribution = {label: 0 for label in PARTIAL_CHAIN_LABELS}
    complete_seed_ids = set.intersection(*(stage_seed_ids[task] for task in TASK_ORDER))
    for seed_id in expected_seed_ids:
        present = tuple(task for task in TASK_ORDER if seed_id in stage_seed_ids[task])
        expected_prefix = TASK_ORDER[: len(present)]
        if present != expected_prefix:
            # Defensive redundancy: source resolution should make this branch
            # unreachable, but an explicit shape check keeps future DAG edits
            # fail-closed.
            raise ShardMergeError(
                "merged corpus contains a partial chain with a missing dependency stage"
            )
        if len(present) < len(TASK_ORDER):
            partial_distribution[PARTIAL_CHAIN_LABELS[len(present)]] += 1

    complete_ids = sorted(complete_seed_ids)
    return {
        "seed_count": len(expected_seed_ids),
        "stage_counts": {task: len(stage_seed_ids[task]) for task in TASK_ORDER},
        "partial_chain_distribution": partial_distribution,
        "partial_chain_count": sum(partial_distribution.values()),
        "raw_complete_chain_intersection": {
            "count": len(complete_ids),
            "seed_ids_sha256": sha256(
                json.dumps(complete_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
    }


def _resolve_sources(
    base_dir: str | Path,
    shard_dirs: Sequence[str | Path],
    target_dir: str | Path,
) -> tuple[tuple[Path, ...], Path]:
    if not shard_dirs:
        raise ShardMergeError("at least one shard directory is required")
    sources: list[Path] = []
    for raw in (base_dir, *shard_dirs):
        path = Path(raw).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ShardMergeError("every source must be a directory")
        sources.append(path)
    if len(set(sources)) != len(sources):
        raise ShardMergeError("base and shard directories must be distinct")
    target = Path(target_dir).expanduser().resolve(strict=False)
    if target.exists():
        raise ShardMergeError("target directory already exists; overwrite refused")
    if any(target == source or target.is_relative_to(source) for source in sources):
        raise ShardMergeError("target must not be inside an input directory")
    return tuple(sources), target


def _audit_sources(sources: Sequence[Path]) -> _Audit:
    rows: dict[str, list[dict[str, Any]]] = {filename: [] for filename in FILES}
    input_counts: dict[str, int] = {filename: 0 for filename in FILES}
    input_hashes: list[dict[str, str]] = []
    duplicate_stats: dict[str, dict[str, int]] = {
        filename: {"eliminated": 0, "id": 0, "fingerprint": 0} for filename in FILES
    }

    seeds_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    seeds_by_fingerprint: dict[str, str] = {}
    seeds_by_index: dict[int, str] = {}
    records_by_id: dict[str, tuple[str, str, dict[str, Any]]] = {}
    records_by_fingerprint: dict[str, tuple[str, str]] = {}
    task_seed_ids: dict[str, dict[str, str]] = {task: {} for task in TASK_ORDER}

    for source_index, source in enumerate(sources):
        file_hashes: dict[str, str] = {}
        for filename in FILES:
            path = source / filename
            if path.exists() and (path.is_symlink() or not path.is_file()):
                raise ShardMergeError(
                    f"source[{source_index}]/{filename} must be a regular file"
                )
            raw = path.read_bytes() if path.is_file() else b""
            file_hashes[filename] = sha256(raw).hexdigest() if raw else EMPTY_SHA256
            loaded = _load_jsonl(path, source_index=source_index)
            input_counts[filename] += len(loaded)
            for line_offset, value in loaded:
                label = _source_label(source_index, filename, line_offset)
                canonical_hash = sha256(_canonical_bytes(value)).hexdigest()
                if filename == "seeds.jsonl":
                    seed = _validate_seed(value, label=label)
                    fingerprint = stable_id("request", normalized_text(seed.request))
                    prior = seeds_by_id.get(seed.seed_id)
                    if prior is not None:
                        if prior[0] != canonical_hash:
                            raise ShardMergeError(
                                f"{label}: same seed_id has different content"
                            )
                        duplicate_stats[filename]["eliminated"] += 1
                        duplicate_stats[filename]["id"] += 1
                        duplicate_stats[filename]["fingerprint"] += 1
                        continue
                    prior_id = seeds_by_fingerprint.get(fingerprint)
                    if prior_id is not None and prior_id != seed.seed_id:
                        raise ShardMergeError(
                            f"{label}: request fingerprint maps to multiple seed IDs"
                        )
                    copied = dict(value)
                    if seed.seed_index is not None:
                        prior_index_seed = seeds_by_index.get(seed.seed_index)
                        if (
                            prior_index_seed is not None
                            and prior_index_seed != seed.seed_id
                        ):
                            raise ShardMergeError(
                                f"{label}: seed_index is shared by different seeds"
                            )
                        seeds_by_index[seed.seed_index] = seed.seed_id
                    seeds_by_id[seed.seed_id] = (canonical_hash, copied)
                    seeds_by_fingerprint[fingerprint] = seed.seed_id
                    rows[filename].append(copied)
                    continue

                task = filename.removeprefix("data_").removesuffix(".jsonl")
                _validate_task_record(task, value, label=label)
                record_id = str(value["id"])
                seed_id = str(value["provenance"]["seed_id"])
                fingerprint = _content_fingerprint(value)
                prior_global = records_by_id.get(record_id)
                if prior_global is not None:
                    prior_task, prior_hash, _ = prior_global
                    if prior_task != task or prior_hash != canonical_hash:
                        raise ShardMergeError(
                            f"{label}: same record id has different or cross-task content"
                        )
                    duplicate_stats[filename]["eliminated"] += 1
                    duplicate_stats[filename]["id"] += 1
                    duplicate_stats[filename]["fingerprint"] += 1
                    continue
                prior_fingerprint = records_by_fingerprint.get(fingerprint)
                if prior_fingerprint is not None:
                    raise ShardMergeError(
                        f"{label}: content fingerprint maps to multiple record IDs"
                    )
                prior_seed_record = task_seed_ids[task].get(seed_id)
                if prior_seed_record is not None:
                    raise ShardMergeError(
                        f"{label}: task contains multiple records for one seed"
                    )
                copied = dict(value)
                records_by_id[record_id] = (task, canonical_hash, copied)
                records_by_fingerprint[fingerprint] = (task, record_id)
                task_seed_ids[task][seed_id] = record_id
                rows[filename].append(copied)
        input_hashes.append(file_hashes)

    records = {task: rows[f"data_{task}.jsonl"] for task in TASK_ORDER}
    lineage_manifest = _audit_lineage(
        {seed_id: value for seed_id, (_, value) in seeds_by_id.items()},
        records,
    )

    output_bytes = {filename: _file_bytes(rows[filename]) for filename in FILES}
    output_hashes = {
        filename: sha256(content).hexdigest()
        for filename, content in output_bytes.items()
    }
    combined_hash = sha256(
        b"".join(
            filename.encode("utf-8") + b"\0" + bytes.fromhex(output_hashes[filename])
            for filename in FILES
        )
    ).hexdigest()
    duplicates_total = sum(stats["eliminated"] for stats in duplicate_stats.values())
    merged_assignments = [
        assignment_for_seed(SeedDemand.from_mapping(value), load_task_card_catalog())
        for _seed_id, (_digest, value) in sorted(seeds_by_id.items())
    ]
    template_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for assignment in merged_assignments:
        if assignment.template_id is not None:
            template_counts[assignment.template_id] = (
                template_counts.get(assignment.template_id, 0) + 1
            )
        source_counts[assignment.source_kind] = (
            source_counts.get(assignment.source_kind, 0) + 1
        )
    card_ids = sorted(assignment.card_id for assignment in merged_assignments)
    if len(set(card_ids)) != len(card_ids):
        raise ShardMergeError("merged seeds do not map one-to-one to final task cards")
    manifest = {
        "schema_version": "anchor.distillation-shard-merge.v2",
        "counts": {
            "sources": len(sources),
            "input_rows": input_counts,
            "output_rows": {filename: len(rows[filename]) for filename in FILES},
            "input_total": sum(input_counts.values()),
            "output_total": sum(len(rows[filename]) for filename in FILES),
        },
        "hashes": {
            "inputs": input_hashes,
            "outputs": output_hashes,
            "combined_output_sha256": combined_hash,
        },
        "duplicates": {
            "eliminated_total": duplicates_total,
            "by_file": duplicate_stats,
        },
        "lineage": lineage_manifest,
        "strict_gold": {
            "decision": "deferred_to_partition",
            "partition_required": True,
        },
        "task_cards": {
            "card_count": len(set(card_ids)),
            "card_ids_sha256": sha256(
                json.dumps(card_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "legacy_count": sum(item.legacy for item in merged_assignments),
            "template_counts": dict(sorted(template_counts.items())),
            "source_kind_counts": dict(sorted(source_counts.items())),
            "seed_index_min": min(seeds_by_index) if seeds_by_index else None,
            "seed_index_max": max(seeds_by_index) if seeds_by_index else None,
        },
    }
    return _Audit(
        rows={filename: tuple(rows[filename]) for filename in FILES},
        manifest=manifest,
    )


def _write_fsync(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def merge_distillation_shards(
    *,
    base_dir: str | Path,
    shard_dirs: Sequence[str | Path],
    target_dir: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Audit shards and optionally publish one new merged directory atomically."""

    sources, target = _resolve_sources(base_dir, shard_dirs, target_dir)
    audit = _audit_sources(sources)
    if dry_run:
        return audit.manifest

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.merge.lock"
    lock_descriptor: int | None = None
    lock_created = False
    staging: Path | None = None
    try:
        try:
            lock_descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            lock_created = True
        except FileExistsError as error:
            raise ShardMergeError(
                "another merge is already publishing this target"
            ) from error
        if target.exists():
            raise ShardMergeError("target appeared during audit; overwrite refused")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.merge-",
                dir=target.parent,
            )
        ).resolve()
        for filename in FILES:
            _write_fsync(staging / filename, _file_bytes(audit.rows[filename]))
        _write_fsync(
            staging / "manifest.json",
            (
                json.dumps(
                    audit.manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        if target.exists():
            raise ShardMergeError("target appeared before publish; overwrite refused")
        os.rename(staging, target)
        staging = None
        return audit.manifest
    finally:
        if staging is not None and staging.is_dir():
            shutil.rmtree(staging)
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if lock_created and lock_path.is_file():
            lock_path.unlink()
