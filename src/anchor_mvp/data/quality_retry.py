"""Offline planning and crash-safe active projections for quality retries.

Raw teacher history is never edited in place without first being copied byte-for-byte
to an immutable generation archive.  The files at the collection root are an active
projection used by the resumable pipeline; archived generations remain recoverable.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ORDER = ("plan", "tool_policy", "frontend", "review", "security")
_TASK_INDEX = {task: index for index, task in enumerate(TASK_ORDER)}
_DOWNSTREAM = {
    "plan": ("plan", "tool_policy", "frontend", "review", "security"),
    "tool_policy": ("tool_policy", "frontend", "review", "security"),
    "frontend": ("frontend", "review", "security"),
    "review": ("review", "security"),
    "security": ("security",),
}

# These identifiers are the complete public feedback vocabulary.  Never put raw
# validator output, provider errors, prompts, answers, or exception text in a plan.
QUALITY_FEEDBACK_CODES = frozenset(
    {
        "artifact_build_or_test_failed",
        "artifact_missing_tsx",
        "deterministic_oracle_mismatch",
        "duplicate_prompt",
        "duplicate_record",
        "lineage_rebuild_required",
        "public_rubric_disagreement",
        "public_safety_contract",
        "raw_record_missing",
        "review_mutation_unavailable",
        "review_repair_not_canonical",
        "task_card_alignment",
        "task_card_axis",
        "upstream_quality_rebuild",
    }
)

_LABEL_TO_CODE = {
    "artifact_validation_failed": "artifact_build_or_test_failed",
    "deterministic_oracle_mismatch": "deterministic_oracle_mismatch",
    "duplicate_prompt": "duplicate_prompt",
    "duplicate_record_id": "duplicate_record",
    "teacher_label_disagreement": "public_rubric_disagreement",
    "teacher_label_disagreement_oracle_label_only": "public_rubric_disagreement",
    "task_card_alignment_invalid": "task_card_alignment",
    "task_card_assignment_invalid": "task_card_alignment",
    "task_card_axis_mismatch": "task_card_axis",
    "near_duplicate_requirement": "duplicate_prompt",
    "secret_detected": "public_safety_contract",
    "unsafe_payload": "public_safety_contract",
    "heldout_source_excluded": "public_safety_contract",
    "invalid_quality_retry_metadata": "public_safety_contract",
}


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required: {path.name}:{line_number}")
        records.append(value)
    return records


def _seed_id(record: Mapping[str, Any]) -> str | None:
    provenance = record.get("provenance")
    value = provenance.get("seed_id") if isinstance(provenance, Mapping) else None
    return value if isinstance(value, str) and value else None


def _canonical_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        for record in records
    )


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (
            json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )


def _artifact_feedback(
    task: str, failures: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    codes: set[str] = set()
    root = task
    for failure in failures:
        # Only compare fixed prefixes.  The suffix may contain an exception class
        # and is deliberately neither retained nor copied into feedback.
        if failure.startswith("missing_tsx_artifact"):
            codes.add("artifact_missing_tsx")
        elif failure.startswith("mutation_recompute_failed"):
            codes.add("review_mutation_unavailable")
            root = "frontend"
        elif failure.startswith(
            "candidate_or_mutation_not_canonical"
        ) or failure.startswith("repair_does_not_restore_frontend_source"):
            codes.add("review_repair_not_canonical")
        else:
            codes.add("artifact_build_or_test_failed")
    return root, tuple(sorted(codes))


def build_quality_retry_plan(
    config: Any,
    partition: Mapping[str, Any],
    *,
    generation: int,
    artifact_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a content-free earliest-root retry plan for one partition."""

    active: dict[str, list[dict[str, Any]]] = {
        task: _read_jsonl(config.output_dir / f"data_{task}.jsonl")
        for task in TASK_ORDER
    }
    record_index: dict[tuple[str, str], dict[str, Any]] = {}
    records_by_seed: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task, records in active.items():
        for record in records:
            record_id = str(record.get("id", ""))
            record_index[(task, record_id)] = record
            seed_id = _seed_id(record)
            if seed_id:
                records_by_seed[seed_id][task].append(record_id)

    root_by_seed: dict[str, str] = {}
    feedback_by_seed_task: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    def schedule(seed_id: str, task: str, codes: Sequence[str]) -> None:
        if task not in _TASK_INDEX:
            return
        current = root_by_seed.get(seed_id)
        if current is None or _TASK_INDEX[task] < _TASK_INDEX[current]:
            root_by_seed[seed_id] = task
        public_codes = set(codes).intersection(QUALITY_FEEDBACK_CODES)
        feedback_by_seed_task[seed_id][task].update(public_codes)

    staging = _read_jsonl(config.quality_staging_path)
    for item in staging:
        if item.get("disposition") == "gold":
            continue
        task = str(item.get("task_type", ""))
        source_id = str(item.get("source_record_id", ""))
        raw = record_index.get((task, source_id))
        seed_id = _seed_id(raw) if raw is not None else None
        if not seed_id:
            continue
        quality = item.get("quality")
        labels = quality.get("labels", []) if isinstance(quality, Mapping) else []
        codes = {
            _LABEL_TO_CODE[str(label)]
            for label in labels
            if str(label) in _LABEL_TO_CODE
        }
        schedule(seed_id, task, sorted(codes or {"lineage_rebuild_required"}))

    raw_failures = (
        artifact_gate.get("record_failures", {})
        if isinstance(artifact_gate, Mapping)
        else {}
    )
    if isinstance(raw_failures, Mapping):
        for identity, raw_codes in raw_failures.items():
            if not isinstance(identity, str) or ":" not in identity:
                continue
            task, record_id = identity.split(":", 1)
            record = record_index.get((task, record_id))
            seed_id = _seed_id(record) if record is not None else None
            if not seed_id or not isinstance(raw_codes, list):
                continue
            root, codes = _artifact_feedback(task, [str(code) for code in raw_codes])
            schedule(seed_id, root, codes)

    lineage_errors = partition.get("lineage_edge_errors", [])
    if isinstance(lineage_errors, list):
        for error in lineage_errors:
            if not isinstance(error, Mapping):
                continue
            downstream_task = str(error.get("downstream_task", ""))
            upstream_task = str(error.get("upstream_task", ""))
            downstream_id = str(error.get("downstream_record_id", ""))
            downstream = record_index.get((downstream_task, downstream_id))
            seed_id = _seed_id(downstream) if downstream is not None else None
            if not seed_id:
                source_id = str(error.get("source_record_id", ""))
                upstream = record_index.get((upstream_task, source_id))
                seed_id = _seed_id(upstream) if upstream is not None else None
            if not seed_id:
                continue
            contract = str(error.get("contract", ""))
            root = (
                "frontend"
                if contract == "review_mutation_unavailable"
                else upstream_task
                if error.get("code")
                in {"missing_source_record_id", "source_not_strict_gold"}
                else downstream_task
            )
            code = (
                "review_mutation_unavailable"
                if contract == "review_mutation_unavailable"
                else "lineage_rebuild_required"
            )
            schedule(seed_id, root, (code,))

    # Raw shortfalls are converted to concrete missing seed/task jobs rather than
    # replaying arbitrary already-complete examples.
    seeds = _read_jsonl(config.output_dir / "seeds.jsonl")
    seeds.sort(
        key=lambda item: (
            item.get("seed_index")
            if isinstance(item.get("seed_index"), int)
            else 2**63 - 1,
            str(item.get("seed_id", "")),
        )
    )
    target = int(partition.get("raw_collection_target", config.raw_records_per_task))
    for seed in seeds[:target]:
        seed_id = str(seed.get("seed_id", ""))
        if not seed_id:
            continue
        missing = next(
            (task for task in TASK_ORDER if not records_by_seed[seed_id].get(task)),
            None,
        )
        if missing is not None:
            schedule(seed_id, missing, ("raw_record_missing",))

    jobs: list[dict[str, Any]] = []
    seeds_plan: list[dict[str, Any]] = []
    for seed_id in sorted(root_by_seed):
        root = root_by_seed[seed_id]
        tasks = _DOWNSTREAM[root]
        seed_feedback: dict[str, list[str]] = {}
        retry_of: dict[str, list[str]] = {}
        for task in tasks:
            codes = feedback_by_seed_task[seed_id].get(task, set())
            if not codes:
                codes = {"upstream_quality_rebuild"}
            clean_codes = sorted(codes.intersection(QUALITY_FEEDBACK_CODES))
            prior = sorted(records_by_seed[seed_id].get(task, []))
            seed_feedback[task] = clean_codes
            retry_of[task] = prior
            jobs.append(
                {
                    "seed_id": seed_id,
                    "task_type": task,
                    "generation": generation,
                    "retry_of": prior,
                    "quality_feedback_codes": clean_codes,
                }
            )
        seeds_plan.append(
            {
                "seed_id": seed_id,
                "root_task": root,
                "tasks": list(tasks),
                "retry_of": retry_of,
                "quality_feedback_codes": seed_feedback,
            }
        )

    return {
        "schema_version": "anchor.automation-quality-retry-plan.v1",
        "generation": generation,
        "created_at": _iso(),
        "seed_count": len(seeds_plan),
        "job_count": len(jobs),
        "seeds": seeds_plan,
        "jobs": jobs,
        "feedback_code_vocabulary": sorted(QUALITY_FEEDBACK_CODES),
        "content_retained": False,
    }


def quality_feedback_map(
    plan: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index a validated public plan for DistillationPipeline."""

    indexed: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw in plan.get("jobs", []):
        if not isinstance(raw, Mapping):
            continue
        task = str(raw.get("task_type", ""))
        seed_id = str(raw.get("seed_id", ""))
        codes = raw.get("quality_feedback_codes", [])
        if task not in _TASK_INDEX or not seed_id or not isinstance(codes, list):
            raise ValueError("invalid quality retry job")
        clean = sorted({str(code) for code in codes})
        if not set(clean).issubset(QUALITY_FEEDBACK_CODES):
            raise ValueError("quality retry job contains non-public feedback")
        retry_of = raw.get("retry_of", [])
        if not isinstance(retry_of, list) or any(
            not isinstance(value, str) for value in retry_of
        ):
            raise ValueError("quality retry retry_of must contain record IDs")
        indexed[task][seed_id] = {
            "generation": int(raw.get("generation", 0)),
            "retry_of": list(retry_of),
            "quality_feedback_codes": clean,
        }
    return {task: dict(values) for task, values in indexed.items()}


def prepare_quality_retry_projection(
    config: Any,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Archive the full active corpus then atomically remove planned rows.

    A persisted manifest makes every per-file transition replayable: on resume an
    active file must equal either its before or after digest.  Any third value
    aborts rather than overwriting concurrent/user data.
    """

    generation = int(plan.get("generation", 0))
    if generation < 1:
        raise ValueError("quality retry generation must be positive")
    indexed = quality_feedback_map(plan)
    generation_dir = config.state_dir / "quality_retry" / f"generation-{generation}"
    manifest_path = generation_dir / "manifest.json"
    plan_path = generation_dir / "plan.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version")
            != "anchor.automation-quality-retry-archive.v1"
            or manifest.get("generation") != generation
        ):
            raise ValueError("quality retry archive manifest mismatch")
        expected_plan_sha = str(manifest.get("plan_sha256", ""))
        if (
            not plan_path.is_file()
            or _sha256_bytes(plan_path.read_bytes()) != expected_plan_sha
        ):
            raise ValueError("quality retry plan archive hash mismatch")
        files = manifest.get("files")
        if not isinstance(files, Mapping) or set(files) != set(TASK_ORDER):
            raise ValueError("quality retry archive must bind all task files")
        for task, metadata in files.items():
            if task not in _TASK_INDEX or not isinstance(metadata, Mapping):
                raise ValueError("quality retry archive file manifest is invalid")
            expected_names = {
                "archive_path": f"active_data_{task}.jsonl",
                "projection_path": f"projection_data_{task}.jsonl",
                "removed_path": f"removed_data_{task}.jsonl",
            }
            if any(
                metadata.get(field) != name for field, name in expected_names.items()
            ):
                raise ValueError("quality retry archive path contract mismatch")
            archive = generation_dir / expected_names["archive_path"]
            if not archive.is_file() or _sha256_bytes(
                archive.read_bytes()
            ) != metadata.get("before_sha256"):
                raise ValueError("quality retry full archive hash mismatch")
            removed = generation_dir / expected_names["removed_path"]
            if not removed.is_file() or _sha256_bytes(
                removed.read_bytes()
            ) != metadata.get("removed_sha256"):
                raise ValueError("quality retry removed archive hash mismatch")
            projection = generation_dir / expected_names["projection_path"]
            if not projection.is_file() or _sha256_bytes(
                projection.read_bytes()
            ) != metadata.get("after_sha256"):
                raise ValueError("quality retry projection archive hash mismatch")
            active_path = config.output_dir / f"data_{task}.jsonl"
            active = active_path.read_bytes() if active_path.is_file() else b""
            digest = _sha256_bytes(active)
            before = str(metadata["before_sha256"])
            after = str(metadata["after_sha256"])
            if digest == before:
                _atomic_write_bytes(active_path, projection.read_bytes())
            elif digest != after:
                raise ValueError(
                    "active collection changed during quality retry prepare"
                )
        return manifest

    plan_bytes = (
        json.dumps(dict(plan), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(plan_path, plan_bytes)
    files: dict[str, dict[str, Any]] = {}
    prepared: dict[str, bytes] = {}
    for task in TASK_ORDER:
        active_path = config.output_dir / f"data_{task}.jsonl"
        before_bytes = active_path.read_bytes() if active_path.is_file() else b""
        records = _read_jsonl(active_path)
        remove = set(indexed.get(task, {}))
        retained = [record for record in records if _seed_id(record) not in remove]
        removed = [record for record in records if _seed_id(record) in remove]
        after_bytes = _canonical_jsonl(retained)
        archive_name = f"active_data_{task}.jsonl"
        projection_name = f"projection_data_{task}.jsonl"
        removed_name = f"removed_data_{task}.jsonl"
        _atomic_write_bytes(generation_dir / archive_name, before_bytes)
        _atomic_write_bytes(generation_dir / projection_name, after_bytes)
        _atomic_write_bytes(generation_dir / removed_name, _canonical_jsonl(removed))
        files[task] = {
            "active_path": f"data_{task}.jsonl",
            "archive_path": archive_name,
            "projection_path": projection_name,
            "removed_path": removed_name,
            "before_sha256": _sha256_bytes(before_bytes),
            "after_sha256": _sha256_bytes(after_bytes),
            "removed_sha256": _sha256_bytes(_canonical_jsonl(removed)),
            "before_records": len(records),
            "after_records": len(retained),
            "removed_records": len(removed),
        }
        prepared[task] = after_bytes

    manifest = {
        "schema_version": "anchor.automation-quality-retry-archive.v1",
        "generation": generation,
        "created_at": _iso(),
        "plan_path": "plan.json",
        "plan_sha256": _sha256_bytes(plan_bytes),
        "files": files,
        "raw_history_recoverable": True,
    }
    # Persist intent before changing any active projection.  A crash after this
    # point is completed idempotently by the manifest replay branch above.
    _atomic_write_json(manifest_path, manifest)
    for task in TASK_ORDER:
        _atomic_write_bytes(config.output_dir / f"data_{task}.jsonl", prepared[task])
    return manifest


def restore_quality_retry_projection(config: Any, generation: int) -> dict[str, Any]:
    """Abandon the latest active retry projection and restore its full snapshot."""

    retry_root = config.state_dir / "quality_retry"
    generations = sorted(
        int(path.name.removeprefix("generation-"))
        for path in retry_root.glob("generation-*")
        if path.name.removeprefix("generation-").isdigit()
        and (path / "manifest.json").is_file()
    )
    if not generations or generation != generations[-1]:
        raise ValueError("only the latest quality retry generation may be restored")
    generation_dir = retry_root / f"generation-{generation}"
    manifest = json.loads(
        (generation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version")
        != "anchor.automation-quality-retry-archive.v1"
        or manifest.get("generation") != generation
        or not isinstance(files, Mapping)
        or set(files) != set(TASK_ORDER)
    ):
        raise ValueError("quality retry restore manifest is invalid")

    desired: dict[str, bytes] = {}
    for task in TASK_ORDER:
        metadata = files[task]
        expected_names = {
            "archive_path": f"active_data_{task}.jsonl",
            "projection_path": f"projection_data_{task}.jsonl",
            "removed_path": f"removed_data_{task}.jsonl",
        }
        if not isinstance(metadata, Mapping) or any(
            metadata.get(field) != name for field, name in expected_names.items()
        ):
            raise ValueError("quality retry restore path contract mismatch")
        for field, hash_field in (
            ("archive_path", "before_sha256"),
            ("projection_path", "after_sha256"),
            ("removed_path", "removed_sha256"),
        ):
            path = generation_dir / expected_names[field]
            if not path.is_file() or _sha256_bytes(path.read_bytes()) != metadata.get(
                hash_field
            ):
                raise ValueError("quality retry restore archive hash mismatch")
        desired[task] = (generation_dir / expected_names["archive_path"]).read_bytes()

    restore_path = generation_dir / "restore-manifest.json"
    if restore_path.is_file():
        restore = json.loads(restore_path.read_text(encoding="utf-8"))
        if not isinstance(restore, Mapping) or restore.get("generation") != generation:
            raise ValueError("quality retry restore intent is invalid")
    else:
        abandoned: dict[str, dict[str, Any]] = {}
        for task in TASK_ORDER:
            active_path = config.output_dir / f"data_{task}.jsonl"
            active = active_path.read_bytes() if active_path.is_file() else b""
            abandoned_name = f"abandoned_data_{task}.jsonl"
            _atomic_write_bytes(generation_dir / abandoned_name, active)
            abandoned[task] = {
                "path": abandoned_name,
                "sha256": _sha256_bytes(active),
                "restore_sha256": _sha256_bytes(desired[task]),
            }
        restore = {
            "schema_version": "anchor.automation-quality-retry-restore.v1",
            "generation": generation,
            "created_at": _iso(),
            "files": abandoned,
            "partial_retry_history_recoverable": True,
        }
        _atomic_write_json(restore_path, restore)

    restore_files = restore.get("files")
    if not isinstance(restore_files, Mapping) or set(restore_files) != set(TASK_ORDER):
        raise ValueError("quality retry restore intent must bind all task files")
    for task in TASK_ORDER:
        metadata = restore_files[task]
        if not isinstance(metadata, Mapping):
            raise ValueError("quality retry restore file intent is invalid")
        abandoned_name = f"abandoned_data_{task}.jsonl"
        if metadata.get("path") != abandoned_name:
            raise ValueError("quality retry abandoned archive path mismatch")
        abandoned_path = generation_dir / abandoned_name
        if not abandoned_path.is_file() or _sha256_bytes(
            abandoned_path.read_bytes()
        ) != metadata.get("sha256"):
            raise ValueError("quality retry abandoned archive hash mismatch")
        desired_sha = _sha256_bytes(desired[task])
        if metadata.get("restore_sha256") != desired_sha:
            raise ValueError("quality retry restore target hash mismatch")
        active_path = config.output_dir / f"data_{task}.jsonl"
        active = active_path.read_bytes() if active_path.is_file() else b""
        if _sha256_bytes(active) not in {str(metadata["sha256"]), desired_sha}:
            raise ValueError("active collection changed during quality retry restore")
        _atomic_write_bytes(active_path, desired[task])
    return dict(restore)
