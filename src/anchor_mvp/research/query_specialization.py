"""Training-side contracts for role-conditioned query specialization.

The module is intentionally lightweight.  It validates task-board records,
builds visibility-safe training views, selects LoRA target modules, and exposes
an auxiliary block-attention loss that can later be attached to a full LLM
trainer.  The CPU probe lives in ``scripts/research`` and uses the same record
contract without loading a foundation model.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "anchor.query-specialization.v1"
SIDECAR_SCHEMA_VERSION = "anchor.swebench-taskboard-sidecar.v1"
PROJECTOR_VERSION = "anchor.swebench-taskboard-projector.v1"
MANIFEST_SCHEMA_VERSION = "anchor.swebench-taskboard-projector-manifest.v1"
BLOCK_KINDS = (
    "requirement",
    "constraint",
    "plan",
    "repository",
    "code",
    "tool_call",
    "tool_result",
    "test_result",
    "review",
    "history",
)
COMMIT_STATES = ("candidate", "verified", "committed", "rejected")
ROLES = (
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate",
)
LANGUAGES = ("en", "zh-CN")
VARIANTS = ("clean", "noisy")
SPLITS = ("train", "calibration", "eval")
SIDECAR_SPLITS = ("train", "calibration")
STAGE_TO_EXPERT = {
    "planner": "planner",
    "tool_policy": "tool_policy",
    "domain_builder": "frontend_gen",
    "domain_review": "frontend_review",
    "security": "security_gate",
}
FIXED_SIDECAR_PARTITIONS = (
    ("train/clean.jsonl", "train", "clean"),
    ("train/noisy.jsonl", "train", "noisy"),
    ("calibration/clean.jsonl", "calibration", "clean"),
)
LORA_PROFILES: dict[str, tuple[str, ...]] = {
    "q_only": ("q_proj",),
    "q_o": ("q_proj", "o_proj"),
    "q_o_mlp": ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}
TOP_LEVEL_FIELDS = {
    "schema_version",
    "id",
    "pair_id",
    "variant",
    "language",
    "split",
    "role",
    "task_board",
    "attention_targets",
    "target",
}
SIDECAR_FIELDS = {
    "schema_version",
    "id",
    "pair_id",
    "variant",
    "split",
    "stage",
    "expert",
    "source_gold_record_id",
    "source_gold_sha256",
    "source_gold_file_sha256",
    "source_snapshot_sha256",
    "source_snapshot_manifest_sha256",
    "task_bundle_sha256",
    "base_task_board_sha256",
    "projector_version",
    "config_sha256",
    "sidecar_schema_sha256",
    "augmentation",
    "training_record",
}
AUGMENTATION_FIELDS = {
    "kind",
    "same_task_only",
    "split_before_augmentation",
    "source_block_ids",
    "overlay_block_ids",
}
TASK_BOARD_FIELDS = {"task_id", "generation", "blocks"}
BLOCK_FIELDS = {"id", "kind", "content", "commit_state", "visible_to"}
ATTENTION_TARGET_FIELDS = {
    "relevant_block_ids",
    "distractor_block_ids",
    "forbidden_block_ids",
}
TARGET_FIELDS = {"selected_block_ids", "action", "answer"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class QuerySpecializationError(ValueError):
    """Raised when a training record or profile violates the MVP contract."""


@dataclass(frozen=True)
class TaskBlock:
    """One immutable or speculative block on the shared task board."""

    block_id: str
    kind: str
    content: str
    commit_state: str
    visible_to: tuple[str, ...]

    def is_visible_to(self, role: str) -> bool:
        return "all" in self.visible_to or role in self.visible_to


@dataclass(frozen=True)
class AttentionTargets:
    """Block-level supervision; the sets must be pairwise disjoint."""

    relevant: tuple[str, ...]
    distractors: tuple[str, ...]
    forbidden: tuple[str, ...]


@dataclass(frozen=True)
class QueryTrainingRecord:
    """Canonical input expected from the redesigned distillation pipeline."""

    record_id: str
    pair_id: str
    variant: str
    language: str
    split: str
    task_id: str
    generation: int
    role: str
    blocks: tuple[TaskBlock, ...]
    targets: AttentionTargets
    target_output: str


@dataclass(frozen=True)
class TaskBoardAugmentation:
    """Causal augmentation metadata owned by the outer projector sidecar."""

    kind: str
    same_task_only: bool
    split_before_augmentation: bool
    source_block_ids: tuple[str, ...]
    overlay_block_ids: tuple[str, ...]


@dataclass(frozen=True)
class TaskBoardSidecar:
    """Immutable projector wrapper that binds provenance to one inner record."""

    record_id: str
    pair_id: str
    variant: str
    split: str
    stage: str
    expert: str
    source_gold_record_id: str
    source_gold_sha256: str
    source_gold_file_sha256: str
    source_snapshot_sha256: str
    source_snapshot_manifest_sha256: str
    task_bundle_sha256: str
    base_task_board_sha256: str
    projector_version: str
    config_sha256: str
    sidecar_schema_sha256: str
    augmentation: TaskBoardAugmentation
    training_record: QueryTrainingRecord


@dataclass(frozen=True)
class TrainingView:
    """Visibility-filtered material consumed by a role-specific trainer."""

    record_id: str
    role: str
    prompt: str
    target_output: str
    visible_block_ids: tuple[str, ...]
    relevant_mask: tuple[bool, ...]
    distractor_mask: tuple[bool, ...]


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuerySpecializationError(f"{path} must be non-empty text")
    return value.strip()


def _identifier(value: Any, path: str) -> str:
    result = _required_text(value, path)
    if _IDENTIFIER.fullmatch(result) is None:
        raise QuerySpecializationError(
            f"{path} must match the projector identifier contract"
        )
    return result


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: set[str], path: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise QuerySpecializationError(
            f"{path} contains unknown fields: {sorted(unknown)}"
        )


def _sha256_text(value: Any, path: str) -> str:
    result = _required_text(value, path)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise QuerySpecializationError(f"{path} must be a lowercase SHA-256 hex digest")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_bytes_snapshot(path: Path) -> bytes:
    """Read one immutable in-memory snapshot or fail closed."""

    try:
        return path.read_bytes()
    except OSError as exc:
        raise QuerySpecializationError(
            f"{path}: could not read bytes snapshot"
        ) from exc


def _decode_utf8_snapshot(snapshot: bytes, source: str) -> str:
    try:
        return snapshot.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QuerySpecializationError(f"{source}: invalid UTF-8") from exc


def _parse_json_snapshot(snapshot: bytes, source: str) -> Any:
    try:
        return json.loads(_decode_utf8_snapshot(snapshot, source))
    except json.JSONDecodeError as exc:
        raise QuerySpecializationError(f"{source}: invalid JSON: {exc.msg}") from exc


def _text_list(value: Any, path: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise QuerySpecializationError(f"{path} must be a list of non-empty strings")
    result = tuple(item.strip() for item in value)
    if not allow_empty and not result:
        raise QuerySpecializationError(f"{path} must not be empty")
    if len(set(result)) != len(result):
        raise QuerySpecializationError(f"{path} must not contain duplicates")
    return result


def _identifier_list(value: Any, path: str, *, allow_empty: bool) -> tuple[str, ...]:
    values = _text_list(value, path, allow_empty=allow_empty)
    for index, item in enumerate(values):
        _identifier(item, f"{path}[{index}]")
    return values


def parse_query_training_record(
    value: Any, *, source: str = "<record>"
) -> QueryTrainingRecord:
    """Validate and parse one task-board attention-supervision record."""

    if not isinstance(value, Mapping):
        raise QuerySpecializationError(f"{source}: record must be an object")
    _reject_unknown_fields(value, TOP_LEVEL_FIELDS, source)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise QuerySpecializationError(
            f"{source}: schema_version must be {SCHEMA_VERSION!r}"
        )
    record_id = _identifier(value.get("id"), f"{source}.id")
    pair_id = _identifier(value.get("pair_id"), f"{source}.pair_id")
    variant = _required_text(value.get("variant"), f"{source}.variant")
    if variant not in VARIANTS:
        raise QuerySpecializationError(f"{source}.variant must be one of {VARIANTS}")
    language = _required_text(value.get("language"), f"{source}.language")
    if language not in LANGUAGES:
        raise QuerySpecializationError(f"{source}.language must be one of {LANGUAGES}")
    split = _required_text(value.get("split"), f"{source}.split")
    if split not in SPLITS:
        raise QuerySpecializationError(f"{source}.split must be one of {SPLITS}")
    role = _required_text(value.get("role"), f"{source}.role")
    if role not in ROLES:
        raise QuerySpecializationError(f"{source}.role must be one of {ROLES}")

    task_board = value.get("task_board")
    if not isinstance(task_board, Mapping):
        raise QuerySpecializationError(f"{source}.task_board must be an object")
    _reject_unknown_fields(task_board, TASK_BOARD_FIELDS, f"{source}.task_board")
    task_id = _identifier(task_board.get("task_id"), f"{source}.task_board.task_id")
    generation = task_board.get("generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        raise QuerySpecializationError(
            f"{source}.task_board.generation must be a positive integer"
        )
    raw_blocks = task_board.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise QuerySpecializationError(
            f"{source}.task_board.blocks must be a non-empty list"
        )
    blocks: list[TaskBlock] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_blocks):
        path = f"{source}.task_board.blocks[{index}]"
        if not isinstance(raw, Mapping):
            raise QuerySpecializationError(f"{path} must be an object")
        _reject_unknown_fields(raw, BLOCK_FIELDS, path)
        block_id = _identifier(raw.get("id"), f"{path}.id")
        if block_id in seen_ids:
            raise QuerySpecializationError(f"{path}.id is duplicated: {block_id}")
        seen_ids.add(block_id)
        kind = _required_text(raw.get("kind"), f"{path}.kind")
        if kind not in BLOCK_KINDS:
            raise QuerySpecializationError(f"{path}.kind must be one of {BLOCK_KINDS}")
        commit_state = _required_text(raw.get("commit_state"), f"{path}.commit_state")
        if commit_state not in COMMIT_STATES:
            raise QuerySpecializationError(
                f"{path}.commit_state must be one of {COMMIT_STATES}"
            )
        visible_to = _text_list(
            raw.get("visible_to"), f"{path}.visible_to", allow_empty=False
        )
        unknown_visibility = set(visible_to) - set(ROLES) - {"all"}
        if unknown_visibility:
            raise QuerySpecializationError(
                f"{path}.visible_to contains unknown roles: {sorted(unknown_visibility)}"
            )
        blocks.append(
            TaskBlock(
                block_id=block_id,
                kind=kind,
                content=_required_text(raw.get("content"), f"{path}.content"),
                commit_state=commit_state,
                visible_to=visible_to,
            )
        )

    raw_targets = value.get("attention_targets")
    if not isinstance(raw_targets, Mapping):
        raise QuerySpecializationError(f"{source}.attention_targets must be an object")
    _reject_unknown_fields(
        raw_targets, ATTENTION_TARGET_FIELDS, f"{source}.attention_targets"
    )
    targets = AttentionTargets(
        relevant=_identifier_list(
            raw_targets.get("relevant_block_ids"),
            f"{source}.attention_targets.relevant_block_ids",
            allow_empty=False,
        ),
        distractors=_identifier_list(
            raw_targets.get("distractor_block_ids"),
            f"{source}.attention_targets.distractor_block_ids",
            allow_empty=True,
        ),
        forbidden=_identifier_list(
            raw_targets.get("forbidden_block_ids"),
            f"{source}.attention_targets.forbidden_block_ids",
            allow_empty=True,
        ),
    )
    target_sets = (
        set(targets.relevant),
        set(targets.distractors),
        set(targets.forbidden),
    )
    if any(target_sets[i] & target_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise QuerySpecializationError(
            f"{source}.attention_targets sets must be pairwise disjoint"
        )
    referenced = set().union(*target_sets)
    unknown = referenced - seen_ids
    if unknown:
        raise QuerySpecializationError(
            f"{source}.attention_targets references unknown blocks: {sorted(unknown)}"
        )
    by_id = {block.block_id: block for block in blocks}
    hidden_relevant = [
        block_id
        for block_id in targets.relevant
        if not by_id[block_id].is_visible_to(role)
        or by_id[block_id].commit_state == "rejected"
    ]
    if hidden_relevant:
        raise QuerySpecializationError(
            f"{source}: relevant blocks are not visible training evidence: {hidden_relevant}"
        )

    raw_target = value.get("target")
    if not isinstance(raw_target, Mapping):
        raise QuerySpecializationError(f"{source}.target must be an object")
    _reject_unknown_fields(raw_target, TARGET_FIELDS, f"{source}.target")
    selected = _identifier_list(
        raw_target.get("selected_block_ids"),
        f"{source}.target.selected_block_ids",
        allow_empty=False,
    )
    if selected != targets.relevant:
        raise QuerySpecializationError(
            f"{source}.target.selected_block_ids must exactly match relevant_block_ids "
            "in the same order"
        )
    canonical_target = {
        "action": _required_text(raw_target.get("action"), f"{source}.target.action"),
        "answer": _required_text(raw_target.get("answer"), f"{source}.target.answer"),
        "selected_block_ids": list(selected),
    }

    return QueryTrainingRecord(
        record_id=record_id,
        pair_id=pair_id,
        variant=variant,
        language=language,
        split=split,
        task_id=task_id,
        generation=generation,
        role=role,
        blocks=tuple(blocks),
        targets=targets,
        target_output=json.dumps(
            canonical_target, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    )


def iter_query_training_records(path: str | Path) -> Iterable[QueryTrainingRecord]:
    """Stream records without retaining the source JSONL in memory."""

    dataset_path = Path(path).expanduser().resolve()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QuerySpecializationError(
                    f"{dataset_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            yield parse_query_training_record(
                raw, source=f"{dataset_path}:{line_number}"
            )


def canonical_query_training_record(record: QueryTrainingRecord) -> dict[str, Any]:
    """Reconstruct the normalized inner training row for semantic hashing."""

    return {
        "schema_version": SCHEMA_VERSION,
        "id": record.record_id,
        "pair_id": record.pair_id,
        "variant": record.variant,
        "language": record.language,
        "split": record.split,
        "role": record.role,
        "task_board": {
            "task_id": record.task_id,
            "generation": record.generation,
            "blocks": [
                {
                    "id": block.block_id,
                    "kind": block.kind,
                    "content": block.content,
                    "commit_state": block.commit_state,
                    "visible_to": list(block.visible_to),
                }
                for block in record.blocks
            ],
        },
        "attention_targets": {
            "relevant_block_ids": list(record.targets.relevant),
            "distractor_block_ids": list(record.targets.distractors),
            "forbidden_block_ids": list(record.targets.forbidden),
        },
        "target": json.loads(record.target_output),
    }


def query_training_record_sha256(record: QueryTrainingRecord) -> str:
    """Hash the normalized semantic row independent of JSON whitespace."""

    return _canonical_sha256(canonical_query_training_record(record))


def parse_taskboard_sidecar(
    value: Any,
    *,
    source: str = "<sidecar>",
    expected_split: str | None = None,
    expected_variant: str | None = None,
    expected_config_sha256: str | None = None,
    expected_sidecar_schema_sha256: str | None = None,
) -> TaskBoardSidecar:
    """Validate one provenance wrapper and its embedded training record.

    Provenance deliberately remains outside ``anchor.query-specialization.v1``.
    The wrapper/inner equality checks make it impossible for a consumer to pair
    valid provenance with a different role view or augmentation variant.
    """

    if not isinstance(value, Mapping):
        raise QuerySpecializationError(f"{source}: sidecar must be an object")
    _reject_unknown_fields(value, SIDECAR_FIELDS, source)
    if set(value) != SIDECAR_FIELDS:
        missing = sorted(SIDECAR_FIELDS - set(value))
        raise QuerySpecializationError(f"{source} is missing fields: {missing}")
    if value.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise QuerySpecializationError(
            f"{source}.schema_version must be {SIDECAR_SCHEMA_VERSION!r}"
        )

    record_id = _identifier(value.get("id"), f"{source}.id")
    pair_id = _identifier(value.get("pair_id"), f"{source}.pair_id")
    variant = _required_text(value.get("variant"), f"{source}.variant")
    if variant not in VARIANTS:
        raise QuerySpecializationError(f"{source}.variant must be one of {VARIANTS}")
    split = _required_text(value.get("split"), f"{source}.split")
    if split not in SIDECAR_SPLITS:
        raise QuerySpecializationError(
            f"{source}.split must be one of {SIDECAR_SPLITS}"
        )
    if expected_split is not None and split != expected_split:
        raise QuerySpecializationError(
            f"{source}.split is {split!r}, expected {expected_split!r}"
        )
    if expected_variant is not None and variant != expected_variant:
        raise QuerySpecializationError(
            f"{source}.variant is {variant!r}, expected {expected_variant!r}"
        )
    if split == "calibration" and variant != "clean":
        raise QuerySpecializationError(
            f"{source}: calibration permits only the clean variant"
        )
    if variant == "noisy" and split != "train":
        raise QuerySpecializationError(
            f"{source}: noisy augmentation is permitted only in train"
        )

    stage = _required_text(value.get("stage"), f"{source}.stage")
    if stage not in STAGE_TO_EXPERT:
        raise QuerySpecializationError(
            f"{source}.stage must be one of {tuple(STAGE_TO_EXPERT)}"
        )
    expert = _required_text(value.get("expert"), f"{source}.expert")
    if expert != STAGE_TO_EXPERT[stage]:
        raise QuerySpecializationError(
            f"{source}: stage {stage!r} must map to expert {STAGE_TO_EXPERT[stage]!r}"
        )

    source_gold_record_id = _identifier(
        value.get("source_gold_record_id"), f"{source}.source_gold_record_id"
    )
    source_gold_sha256 = _sha256_text(
        value.get("source_gold_sha256"), f"{source}.source_gold_sha256"
    )
    source_gold_file_sha256 = _sha256_text(
        value.get("source_gold_file_sha256"),
        f"{source}.source_gold_file_sha256",
    )
    source_snapshot_sha256 = _sha256_text(
        value.get("source_snapshot_sha256"),
        f"{source}.source_snapshot_sha256",
    )
    source_snapshot_manifest_sha256 = _sha256_text(
        value.get("source_snapshot_manifest_sha256"),
        f"{source}.source_snapshot_manifest_sha256",
    )
    task_bundle_sha256 = _sha256_text(
        value.get("task_bundle_sha256"), f"{source}.task_bundle_sha256"
    )
    base_task_board_sha256 = _sha256_text(
        value.get("base_task_board_sha256"),
        f"{source}.base_task_board_sha256",
    )
    projector_version = _required_text(
        value.get("projector_version"), f"{source}.projector_version"
    )
    if projector_version != PROJECTOR_VERSION:
        raise QuerySpecializationError(
            f"{source}.projector_version must be {PROJECTOR_VERSION!r}"
        )
    config_sha256 = _sha256_text(value.get("config_sha256"), f"{source}.config_sha256")
    sidecar_schema_sha256 = _sha256_text(
        value.get("sidecar_schema_sha256"),
        f"{source}.sidecar_schema_sha256",
    )
    if expected_config_sha256 is not None and config_sha256 != _sha256_text(
        expected_config_sha256, "expected_config_sha256"
    ):
        raise QuerySpecializationError(f"{source}: projector config hash mismatch")
    if (
        expected_sidecar_schema_sha256 is not None
        and sidecar_schema_sha256
        != _sha256_text(
            expected_sidecar_schema_sha256, "expected_sidecar_schema_sha256"
        )
    ):
        raise QuerySpecializationError(f"{source}: sidecar schema hash mismatch")

    raw_augmentation = value.get("augmentation")
    if not isinstance(raw_augmentation, Mapping):
        raise QuerySpecializationError(f"{source}.augmentation must be an object")
    _reject_unknown_fields(
        raw_augmentation, AUGMENTATION_FIELDS, f"{source}.augmentation"
    )
    if set(raw_augmentation) != AUGMENTATION_FIELDS:
        missing = sorted(AUGMENTATION_FIELDS - set(raw_augmentation))
        raise QuerySpecializationError(
            f"{source}.augmentation is missing fields: {missing}"
        )
    if raw_augmentation.get("same_task_only") is not True:
        raise QuerySpecializationError(
            f"{source}.augmentation.same_task_only must be true"
        )
    if raw_augmentation.get("split_before_augmentation") is not True:
        raise QuerySpecializationError(
            f"{source}.augmentation.split_before_augmentation must be true"
        )
    augmentation = TaskBoardAugmentation(
        kind=_required_text(
            raw_augmentation.get("kind"), f"{source}.augmentation.kind"
        ),
        same_task_only=True,
        split_before_augmentation=True,
        source_block_ids=_identifier_list(
            raw_augmentation.get("source_block_ids"),
            f"{source}.augmentation.source_block_ids",
            allow_empty=True,
        ),
        overlay_block_ids=_identifier_list(
            raw_augmentation.get("overlay_block_ids"),
            f"{source}.augmentation.overlay_block_ids",
            allow_empty=True,
        ),
    )

    inner = parse_query_training_record(
        value.get("training_record"), source=f"{source}.training_record"
    )
    wrapper_inner = (
        (record_id, inner.record_id, "id"),
        (pair_id, inner.pair_id, "pair_id"),
        (variant, inner.variant, "variant"),
        (split, inner.split, "split"),
        (expert, inner.role, "expert/role"),
    )
    mismatches = [name for outer, nested, name in wrapper_inner if outer != nested]
    if mismatches:
        raise QuerySpecializationError(
            f"{source}: wrapper/training_record mismatch for {mismatches}"
        )

    by_id = {block.block_id: block for block in inner.blocks}
    if variant == "clean":
        if (
            augmentation.kind != "clean"
            or augmentation.source_block_ids
            or augmentation.overlay_block_ids
            or inner.targets.distractors
        ):
            raise QuerySpecializationError(
                f"{source}: clean sidecars cannot contain augmentation overlays"
            )
    else:
        if (
            augmentation.kind != "stale_duplicate_overlay"
            or not augmentation.source_block_ids
            or not augmentation.overlay_block_ids
            or len(augmentation.source_block_ids) != len(augmentation.overlay_block_ids)
        ):
            raise QuerySpecializationError(
                f"{source}: noisy sidecars require paired stale duplicate overlays"
            )
        if set(augmentation.source_block_ids) - set(inner.targets.relevant):
            raise QuerySpecializationError(
                f"{source}: augmentation sources must be relevant same-task blocks"
            )
        if set(augmentation.overlay_block_ids) != set(inner.targets.distractors):
            raise QuerySpecializationError(
                f"{source}: augmentation overlays must exactly match distractors"
            )
        for source_id, overlay_id in zip(
            augmentation.source_block_ids,
            augmentation.overlay_block_ids,
            strict=True,
        ):
            source_block = by_id.get(source_id)
            overlay_block = by_id.get(overlay_id)
            if source_block is None or overlay_block is None:
                raise QuerySpecializationError(
                    f"{source}: augmentation references unknown blocks"
                )
            if (
                overlay_block.kind != "history"
                or overlay_block.commit_state != "candidate"
                or overlay_block.visible_to != (expert,)
                or overlay_block.content != source_block.content
            ):
                raise QuerySpecializationError(
                    f"{source}: stale overlay does not duplicate its declared source"
                )

    overlay_ids = set(augmentation.overlay_block_ids)
    base_board = {
        "task_id": inner.task_id,
        "generation": inner.generation,
        "blocks": [
            {
                "id": block.block_id,
                "kind": block.kind,
                "content": block.content,
                "commit_state": block.commit_state,
                "visible_to": list(block.visible_to),
            }
            for block in inner.blocks
            if block.block_id not in overlay_ids
        ],
    }
    if _canonical_sha256(base_board) != base_task_board_sha256:
        raise QuerySpecializationError(
            f"{source}: base_task_board_sha256 does not bind the clean board"
        )

    return TaskBoardSidecar(
        record_id=record_id,
        pair_id=pair_id,
        variant=variant,
        split=split,
        stage=stage,
        expert=expert,
        source_gold_record_id=source_gold_record_id,
        source_gold_sha256=source_gold_sha256,
        source_gold_file_sha256=source_gold_file_sha256,
        source_snapshot_sha256=source_snapshot_sha256,
        source_snapshot_manifest_sha256=source_snapshot_manifest_sha256,
        task_bundle_sha256=task_bundle_sha256,
        base_task_board_sha256=base_task_board_sha256,
        projector_version=projector_version,
        config_sha256=config_sha256,
        sidecar_schema_sha256=sidecar_schema_sha256,
        augmentation=augmentation,
        training_record=inner,
    )


def canonical_taskboard_sidecar(sidecar: TaskBoardSidecar) -> dict[str, Any]:
    """Reconstruct the normalized outer wrapper, preserving inner separation."""

    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "id": sidecar.record_id,
        "pair_id": sidecar.pair_id,
        "variant": sidecar.variant,
        "split": sidecar.split,
        "stage": sidecar.stage,
        "expert": sidecar.expert,
        "source_gold_record_id": sidecar.source_gold_record_id,
        "source_gold_sha256": sidecar.source_gold_sha256,
        "source_gold_file_sha256": sidecar.source_gold_file_sha256,
        "source_snapshot_sha256": sidecar.source_snapshot_sha256,
        "source_snapshot_manifest_sha256": sidecar.source_snapshot_manifest_sha256,
        "task_bundle_sha256": sidecar.task_bundle_sha256,
        "base_task_board_sha256": sidecar.base_task_board_sha256,
        "projector_version": sidecar.projector_version,
        "config_sha256": sidecar.config_sha256,
        "sidecar_schema_sha256": sidecar.sidecar_schema_sha256,
        "augmentation": {
            "kind": sidecar.augmentation.kind,
            "same_task_only": sidecar.augmentation.same_task_only,
            "split_before_augmentation": sidecar.augmentation.split_before_augmentation,
            "source_block_ids": list(sidecar.augmentation.source_block_ids),
            "overlay_block_ids": list(sidecar.augmentation.overlay_block_ids),
        },
        "training_record": canonical_query_training_record(sidecar.training_record),
    }


def taskboard_sidecar_sha256(sidecar: TaskBoardSidecar) -> str:
    """Hash the complete normalized wrapper, including outer provenance."""

    return _canonical_sha256(canonical_taskboard_sidecar(sidecar))


def _parse_taskboard_sidecars_snapshot(
    snapshot: bytes,
    *,
    source: str,
    expected_split: str | None = None,
    expected_variant: str | None = None,
    expected_config_sha256: str | None = None,
    expected_sidecar_schema_sha256: str | None = None,
) -> tuple[TaskBoardSidecar, ...]:
    """Parse JSONL records from the exact bytes used for authentication."""

    text = _decode_utf8_snapshot(snapshot, source)
    sidecars: list[TaskBoardSidecar] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QuerySpecializationError(
                f"{source}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        sidecars.append(
            parse_taskboard_sidecar(
                raw,
                source=f"{source}:{line_number}",
                expected_split=expected_split,
                expected_variant=expected_variant,
                expected_config_sha256=expected_config_sha256,
                expected_sidecar_schema_sha256=expected_sidecar_schema_sha256,
            )
        )
    return tuple(sidecars)


def iter_taskboard_sidecars(
    path: str | Path,
    *,
    expected_split: str | None = None,
    expected_variant: str | None = None,
    expected_config_sha256: str | None = None,
    expected_sidecar_schema_sha256: str | None = None,
) -> Iterable[TaskBoardSidecar]:
    """Parse one immutable projector-partition bytes snapshot."""

    dataset_path = Path(path).expanduser().resolve()
    snapshot = _read_bytes_snapshot(dataset_path)
    yield from _parse_taskboard_sidecars_snapshot(
        snapshot,
        source=str(dataset_path),
        expected_split=expected_split,
        expected_variant=expected_variant,
        expected_config_sha256=expected_config_sha256,
        expected_sidecar_schema_sha256=expected_sidecar_schema_sha256,
    )


def build_training_view(record: QueryTrainingRecord) -> TrainingView:
    """Apply hard visibility first, retaining distractors for soft supervision."""

    forbidden = set(record.targets.forbidden)
    relevant = set(record.targets.relevant)
    distractors = set(record.targets.distractors)
    visible = tuple(
        block
        for block in record.blocks
        if block.block_id not in forbidden
        and block.commit_state != "rejected"
        and block.is_visible_to(record.role)
    )
    visible_ids = tuple(block.block_id for block in visible)
    if not relevant.issubset(visible_ids):
        raise QuerySpecializationError(
            f"{record.record_id}: hard visibility removed relevant evidence"
        )
    prompt_payload = {
        "blocks": [
            {
                "content": block.content,
                "id": block.block_id,
                "kind": block.kind,
                "state": block.commit_state,
            }
            for block in visible
        ],
        "generation": record.generation,
        "role": record.role,
        "task_id": record.task_id,
    }
    return TrainingView(
        record_id=record.record_id,
        role=record.role,
        prompt=json.dumps(
            prompt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        target_output=record.target_output,
        visible_block_ids=visible_ids,
        relevant_mask=tuple(block_id in relevant for block_id in visible_ids),
        distractor_mask=tuple(block_id in distractors for block_id in visible_ids),
    )


def lora_target_modules(profile: str) -> tuple[str, ...]:
    """Resolve the checked-in ablation profile to PEFT target-module names."""

    try:
        return LORA_PROFILES[profile]
    except KeyError as exc:
        raise QuerySpecializationError(
            f"unknown LoRA profile {profile!r}; choose one of {tuple(LORA_PROFILES)}"
        ) from exc


def validate_paired_records(records: Sequence[QueryTrainingRecord]) -> dict[str, Any]:
    """Require a clean board and a strictly additive noisy counterfactual."""

    groups: dict[str, list[QueryTrainingRecord]] = {}
    record_ids: set[str] = set()
    for record in records:
        if record.record_id in record_ids:
            raise QuerySpecializationError(
                f"duplicate query-specialization record id: {record.record_id}"
            )
        record_ids.add(record.record_id)
        groups.setdefault(record.pair_id, []).append(record)
    for pair_id, group in groups.items():
        variants = {record.variant for record in group}
        if len(group) != 2 or variants != set(VARIANTS):
            raise QuerySpecializationError(
                f"pair {pair_id!r} must contain exactly one clean and one noisy record"
            )
        baseline = group[0]
        for record in group[1:]:
            invariant = (
                record.task_id,
                record.generation,
                record.role,
                record.language,
                record.split,
                record.target_output,
                set(record.targets.relevant),
                set(record.targets.forbidden),
            )
            expected = (
                baseline.task_id,
                baseline.generation,
                baseline.role,
                baseline.language,
                baseline.split,
                baseline.target_output,
                set(baseline.targets.relevant),
                set(baseline.targets.forbidden),
            )
            if invariant != expected:
                raise QuerySpecializationError(
                    f"pair {pair_id!r} changed task, role, split, evidence, or target"
                )
        noisy = next(record for record in group if record.variant == "noisy")
        clean = next(record for record in group if record.variant == "clean")
        if clean.targets.distractors:
            raise QuerySpecializationError(
                f"pair {pair_id!r} clean record must not contain distractor labels"
            )
        if not noisy.targets.distractors:
            raise QuerySpecializationError(
                f"pair {pair_id!r} noisy record must contain distractor labels"
            )
        clean_by_id = {block.block_id: block for block in clean.blocks}
        noisy_by_id = {block.block_id: block for block in noisy.blocks}
        missing = set(clean_by_id) - set(noisy_by_id)
        if missing:
            raise QuerySpecializationError(
                f"pair {pair_id!r} noisy record removed clean blocks: {sorted(missing)}"
            )
        changed = [
            block_id
            for block_id, block in clean_by_id.items()
            if block != noisy_by_id[block_id]
        ]
        if changed:
            raise QuerySpecializationError(
                f"pair {pair_id!r} changed clean blocks in noisy record: {changed}"
            )
        additions = set(noisy_by_id) - set(clean_by_id)
        if additions != set(noisy.targets.distractors):
            raise QuerySpecializationError(
                f"pair {pair_id!r} noisy additions must exactly match distractor labels"
            )
        invalid_distractors = [
            block_id
            for block_id in noisy.targets.distractors
            if not noisy_by_id[block_id].is_visible_to(noisy.role)
            or noisy_by_id[block_id].commit_state == "rejected"
        ]
        if invalid_distractors:
            raise QuerySpecializationError(
                f"pair {pair_id!r} distractors must be visible non-rejected blocks: "
                f"{invalid_distractors}"
            )
    return {"pairs": len(groups), "records": len(records)}


def validate_taskboard_sidecar_dataset(
    sidecars: Sequence[TaskBoardSidecar],
    *,
    expected_config_sha256: str | None = None,
    expected_sidecar_schema_sha256: str | None = None,
    require_all_roles: bool = True,
) -> dict[str, Any]:
    """Validate the fixed train/noise/calibration projector contract.

    Grouping is performed by ``task_bundle_sha256`` because each of the five
    role views intentionally has a distinct source Gold record.  Hashes and
    task IDs are split-isolated before train-only augmentation is considered.
    """

    if not sidecars:
        raise QuerySpecializationError("taskboard sidecar dataset must not be empty")
    expected_config = (
        _sha256_text(expected_config_sha256, "expected_config_sha256")
        if expected_config_sha256 is not None
        else None
    )
    expected_schema = (
        _sha256_text(expected_sidecar_schema_sha256, "expected_sidecar_schema_sha256")
        if expected_sidecar_schema_sha256 is not None
        else None
    )

    record_ids: set[str] = set()
    bundle_splits: dict[str, set[str]] = {}
    task_id_splits: dict[str, set[str]] = {}
    task_id_bundles: dict[str, set[str]] = {}
    base_board_bindings: dict[str, set[tuple[str, str]]] = {}
    source_hash_bindings: dict[str, set[tuple[str, str, str]]] = {}
    groups: dict[tuple[str, str], list[TaskBoardSidecar]] = {}
    pair_groups: dict[str, list[TaskBoardSidecar]] = {}
    for sidecar in sidecars:
        if not isinstance(sidecar, TaskBoardSidecar):
            raise QuerySpecializationError(
                "dataset validation requires outer TaskBoardSidecar records"
            )
        if sidecar.record_id in record_ids:
            raise QuerySpecializationError(
                f"duplicate taskboard sidecar record id: {sidecar.record_id}"
            )
        record_ids.add(sidecar.record_id)
        if expected_config is not None and sidecar.config_sha256 != expected_config:
            raise QuerySpecializationError("projector config hash binding mismatch")
        if (
            expected_schema is not None
            and sidecar.sidecar_schema_sha256 != expected_schema
        ):
            raise QuerySpecializationError("sidecar schema hash binding mismatch")
        bundle_splits.setdefault(sidecar.task_bundle_sha256, set()).add(sidecar.split)
        task_id_splits.setdefault(sidecar.training_record.task_id, set()).add(
            sidecar.split
        )
        task_id_bundles.setdefault(sidecar.training_record.task_id, set()).add(
            sidecar.task_bundle_sha256
        )
        base_board_bindings.setdefault(sidecar.base_task_board_sha256, set()).add(
            (sidecar.task_bundle_sha256, sidecar.split)
        )
        source_hash_bindings.setdefault(sidecar.source_gold_sha256, set()).add(
            (
                sidecar.source_gold_record_id,
                sidecar.source_gold_file_sha256,
                sidecar.split,
            )
        )
        groups.setdefault((sidecar.task_bundle_sha256, sidecar.split), []).append(
            sidecar
        )
        pair_groups.setdefault(sidecar.pair_id, []).append(sidecar)

    for splits in bundle_splits.values():
        if len(splits) != 1:
            raise QuerySpecializationError("task bundle hash crosses dataset splits")
    for splits in task_id_splits.values():
        if len(splits) != 1:
            raise QuerySpecializationError("task id crosses dataset splits")
    for bundles in task_id_bundles.values():
        if len(bundles) != 1:
            raise QuerySpecializationError("task id aliases multiple task bundles")
    for bindings in base_board_bindings.values():
        if len(bindings) != 1:
            raise QuerySpecializationError(
                "base task-board hash aliases tasks or crosses splits"
            )
    for bindings in source_hash_bindings.values():
        if len(bindings) != 1:
            raise QuerySpecializationError(
                "source Gold hash aliases source records/files or crosses splits"
            )

    global_bindings = {
        (
            sidecar.source_snapshot_sha256,
            sidecar.source_snapshot_manifest_sha256,
            sidecar.projector_version,
            sidecar.config_sha256,
            sidecar.sidecar_schema_sha256,
        )
        for sidecar in sidecars
    }
    if len(global_bindings) != 1:
        raise QuerySpecializationError(
            "sidecar dataset mixes snapshot, projector, config, or schema bindings"
        )

    inverse_stage = {expert: stage for stage, expert in STAGE_TO_EXPERT.items()}
    source_counts = Counter(split for _, split in groups)
    train_pairs = 0
    for (_, split), group in groups.items():
        task_ids = {sidecar.training_record.task_id for sidecar in group}
        languages = {sidecar.training_record.language for sidecar in group}
        base_boards = {sidecar.base_task_board_sha256 for sidecar in group}
        if len(task_ids) != 1 or len(base_boards) != 1:
            raise QuerySpecializationError(
                "one task bundle must bind exactly one task id and base board"
            )
        if len(languages) != 1:
            raise QuerySpecializationError("one task bundle mixes role languages")
        roles = {sidecar.expert for sidecar in group}
        if require_all_roles and roles != set(ROLES):
            raise QuerySpecializationError(
                f"task bundle is missing role views: {sorted(set(ROLES) - roles)}"
            )
        for role in roles:
            role_rows = [sidecar for sidecar in group if sidecar.expert == role]
            expected_variants = {"clean", "noisy"} if split == "train" else {"clean"}
            if (
                len(role_rows) != len(expected_variants)
                or {row.variant for row in role_rows} != expected_variants
            ):
                raise QuerySpecializationError(
                    f"{split}/{role} has invalid clean/noisy cardinality"
                )
            if {row.stage for row in role_rows} != {inverse_stage[role]}:
                raise QuerySpecializationError("role views changed projector stage")
            if len({row.pair_id for row in role_rows}) != 1:
                raise QuerySpecializationError("role views changed pair_id")
            provenance = {
                (
                    row.source_gold_record_id,
                    row.source_gold_sha256,
                    row.source_gold_file_sha256,
                    row.source_snapshot_sha256,
                    row.source_snapshot_manifest_sha256,
                    row.task_bundle_sha256,
                    row.base_task_board_sha256,
                    row.projector_version,
                    row.config_sha256,
                    row.sidecar_schema_sha256,
                )
                for row in role_rows
            }
            if len(provenance) != 1:
                raise QuerySpecializationError(
                    "clean/noisy role views changed outer provenance"
                )
            if split == "train":
                validate_paired_records([row.training_record for row in role_rows])
                train_pairs += 1

    for pair_id, pair in pair_groups.items():
        if len({(row.task_bundle_sha256, row.expert) for row in pair}) != 1:
            raise QuerySpecializationError(f"pair {pair_id!r} aliases tasks or experts")
        expected = 2 if pair[0].split == "train" else 1
        if len(pair) != expected:
            raise QuerySpecializationError(
                f"pair {pair_id!r} has invalid partition cardinality"
            )

    split_counts = Counter(sidecar.split for sidecar in sidecars)
    variant_counts = Counter(sidecar.variant for sidecar in sidecars)
    stage_counts = Counter(sidecar.stage for sidecar in sidecars)
    expert_counts = Counter(sidecar.expert for sidecar in sidecars)
    language_counts = Counter(sidecar.training_record.language for sidecar in sidecars)
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "records": len(sidecars),
        "source_tasks": len(groups),
        "source_tasks_by_split": dict(sorted(source_counts.items())),
        "train_pairs": train_pairs,
        "by_split": dict(sorted(split_counts.items())),
        "by_variant": dict(sorted(variant_counts.items())),
        "by_stage": dict(sorted(stage_counts.items())),
        "by_expert": dict(sorted(expert_counts.items())),
        "by_language": {
            language: language_counts.get(language, 0) for language in LANGUAGES
        },
        "all_roles_required": require_all_roles,
        "split_before_augmentation": True,
        "hash_bindings_validated": True,
    }


def validate_source_task_partition(
    sidecars: Sequence[TaskBoardSidecar], *, require_all_roles: bool = True
) -> dict[str, Any]:
    """Compatibility name for the outer sidecar partition validator."""

    return validate_taskboard_sidecar_dataset(
        sidecars, require_all_roles=require_all_roles
    )


def _manifest_mapping(value: Any, path: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QuerySpecializationError(f"{path} must be an object")
    _reject_unknown_fields(value, fields, path)
    missing = fields - set(value)
    if missing:
        raise QuerySpecializationError(f"{path} is missing fields: {sorted(missing)}")
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QuerySpecializationError(f"{path} must be a positive integer")
    return value


def load_taskboard_sidecar_dataset(
    root: str | Path,
    manifest_path: str | Path | None = None,
    *,
    expected_config_sha256: str | None = None,
    expected_sidecar_schema_sha256: str | None = None,
    expected_manifest_schema_sha256: str | None = None,
) -> tuple[tuple[TaskBoardSidecar, ...], dict[str, Any], dict[str, Any]]:
    """Load and bind the three fixed projector files to their manifest.

    This is deliberately filesystem-aware: it validates output-file byte
    lengths and hashes before returning any records to a trainer.
    """

    dataset_root = Path(root).expanduser().resolve()
    resolved_manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else dataset_root / "manifest.json"
    )
    manifest_snapshot = _read_bytes_snapshot(resolved_manifest)
    manifest_snapshot_sha256 = hashlib.sha256(manifest_snapshot).hexdigest()
    sha_sidecar_path = resolved_manifest.with_name(resolved_manifest.name + ".sha256")
    try:
        sha_sidecar_snapshot = _read_bytes_snapshot(sha_sidecar_path)
    except QuerySpecializationError as exc:
        raise QuerySpecializationError(
            "manifest.json.sha256 SHA-256 sidecar is required and must be readable"
        ) from exc
    expected_sha_declaration = f"{manifest_snapshot_sha256}  manifest.json".encode(
        "ascii"
    )
    if sha_sidecar_snapshot not in (
        expected_sha_declaration,
        expected_sha_declaration + b"\n",
    ):
        raise QuerySpecializationError(
            "manifest SHA-256 sidecar must be exactly "
            "'<64 lowercase hex>  manifest.json' with at most one trailing LF"
        )
    raw_manifest = _parse_json_snapshot(manifest_snapshot, str(resolved_manifest))
    top_fields = {
        "schema_version",
        "input",
        "producer",
        "files",
        "counts",
        "split_group_key",
        "task_id_cross_binding_key",
        "all_five_role_views_same_split",
        "canonical_gold_written",
        "provider_requests",
        "heldout_content_read",
        "heldout_content_emitted",
        "split_preserved",
        "augmentation_applied_after_split",
        "claim_scope",
    }
    manifest = _manifest_mapping(raw_manifest, "manifest", top_fields)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise QuerySpecializationError("manifest schema_version is unsupported")
    expected_grouping_contract = {
        "split_group_key": "task_bundle_sha256",
        "task_id_cross_binding_key": "training_record.task_board.task_id",
        "all_five_role_views_same_split": True,
    }
    if any(
        manifest.get(key) != value
        for key, value in expected_grouping_contract.items()
    ):
        raise QuerySpecializationError("manifest task-bundle grouping contract changed")
    expected_flags = {
        "canonical_gold_written": False,
        "provider_requests": 0,
        "heldout_content_read": False,
        "heldout_content_emitted": False,
        "split_preserved": True,
        "augmentation_applied_after_split": True,
        "claim_scope": "research_proxy_only",
    }
    if any(manifest.get(key) != value for key, value in expected_flags.items()):
        raise QuerySpecializationError("manifest safety or claim-scope flags changed")

    producer = _manifest_mapping(
        manifest.get("producer"),
        "manifest.producer",
        {
            "name",
            "projector_version",
            "config_sha256",
            "sidecar_schema_sha256",
            "manifest_schema_sha256",
            "record_schema_version",
        },
    )
    if (
        producer.get("name") != "anchor.swebench-taskboard-projector"
        or producer.get("projector_version") != PROJECTOR_VERSION
        or producer.get("record_schema_version") != SCHEMA_VERSION
    ):
        raise QuerySpecializationError("manifest producer contract changed")
    manifest_config_sha = _sha256_text(
        producer.get("config_sha256"), "manifest.producer.config_sha256"
    )
    manifest_sidecar_schema_sha = _sha256_text(
        producer.get("sidecar_schema_sha256"),
        "manifest.producer.sidecar_schema_sha256",
    )
    producer_manifest_schema_sha = _sha256_text(
        producer.get("manifest_schema_sha256"),
        "manifest.producer.manifest_schema_sha256",
    )
    if expected_config_sha256 is not None and manifest_config_sha != _sha256_text(
        expected_config_sha256, "expected_config_sha256"
    ):
        raise QuerySpecializationError("manifest projector config hash mismatch")
    if (
        expected_sidecar_schema_sha256 is not None
        and manifest_sidecar_schema_sha
        != _sha256_text(
            expected_sidecar_schema_sha256, "expected_sidecar_schema_sha256"
        )
    ):
        raise QuerySpecializationError("manifest sidecar schema hash mismatch")
    if (
        expected_manifest_schema_sha256 is not None
        and producer_manifest_schema_sha
        != _sha256_text(
            expected_manifest_schema_sha256,
            "expected_manifest_schema_sha256",
        )
    ):
        raise QuerySpecializationError("manifest schema hash mismatch")

    input_binding = _manifest_mapping(
        manifest.get("input"),
        "manifest.input",
        {
            "snapshot_schema_version",
            "snapshot_sha256",
            "snapshot_manifest_path",
            "snapshot_manifest_sha256",
            "snapshot_sha256_sidecar_path",
            "snapshot_sha256_sidecar_sha256",
            "splits",
        },
    )
    if input_binding.get(
        "snapshot_schema_version"
    ) != "anchor.training-snapshot.v2" or input_binding.get("splits") != [
        "train",
        "calibration",
    ]:
        raise QuerySpecializationError("manifest input split contract changed")
    snapshot_sha = _sha256_text(
        input_binding.get("snapshot_sha256"), "manifest.input.snapshot_sha256"
    )
    snapshot_manifest_sha = _sha256_text(
        input_binding.get("snapshot_manifest_sha256"),
        "manifest.input.snapshot_manifest_sha256",
    )
    _sha256_text(
        input_binding.get("snapshot_sha256_sidecar_sha256"),
        "manifest.input.snapshot_sha256_sidecar_sha256",
    )
    for path_field in ("snapshot_manifest_path", "snapshot_sha256_sidecar_path"):
        path_value = _required_text(
            input_binding.get(path_field), f"manifest.input.{path_field}"
        )
        if Path(path_value).is_absolute() or ".." in Path(path_value).parts:
            raise QuerySpecializationError(
                f"manifest.input.{path_field} must be a safe relative path"
            )

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(FIXED_SIDECAR_PARTITIONS):
        raise QuerySpecializationError("manifest must list exactly three partitions")
    all_sidecars: list[TaskBoardSidecar] = []
    authenticated_file_sha256 = {"manifest.json": manifest_snapshot_sha256}
    for index, (relative, split, variant) in enumerate(FIXED_SIDECAR_PARTITIONS):
        entry = _manifest_mapping(
            files[index],
            f"manifest.files[{index}]",
            {"path", "sha256", "bytes", "records", "split", "variant"},
        )
        if (
            entry.get("path") != relative
            or entry.get("split") != split
            or entry.get("variant") != variant
        ):
            raise QuerySpecializationError("manifest fixed partition order changed")
        partition_path = dataset_root / relative
        expected_file_sha = _sha256_text(
            entry.get("sha256"), f"manifest.files[{index}].sha256"
        )
        expected_bytes = _positive_int(
            entry.get("bytes"), f"manifest.files[{index}].bytes"
        )
        expected_records = _positive_int(
            entry.get("records"), f"manifest.files[{index}].records"
        )
        partition_snapshot = _read_bytes_snapshot(partition_path)
        if len(partition_snapshot) != expected_bytes:
            raise QuerySpecializationError(f"{relative}: byte-length binding mismatch")
        partition_snapshot_sha256 = hashlib.sha256(partition_snapshot).hexdigest()
        if partition_snapshot_sha256 != expected_file_sha:
            raise QuerySpecializationError(f"{relative}: file hash binding mismatch")
        authenticated_file_sha256[relative] = partition_snapshot_sha256
        partition = _parse_taskboard_sidecars_snapshot(
            partition_snapshot,
            source=str(partition_path),
            expected_split=split,
            expected_variant=variant,
            expected_config_sha256=manifest_config_sha,
            expected_sidecar_schema_sha256=manifest_sidecar_schema_sha,
        )
        if len(partition) != expected_records:
            raise QuerySpecializationError(f"{relative}: record-count binding mismatch")
        all_sidecars.extend(partition)

    if any(
        sidecar.source_snapshot_sha256 != snapshot_sha
        or sidecar.source_snapshot_manifest_sha256 != snapshot_manifest_sha
        for sidecar in all_sidecars
    ):
        raise QuerySpecializationError("sidecars do not bind the manifest snapshot")
    summary = validate_taskboard_sidecar_dataset(
        all_sidecars,
        expected_config_sha256=manifest_config_sha,
        expected_sidecar_schema_sha256=manifest_sidecar_schema_sha,
    )

    counts = _manifest_mapping(
        manifest.get("counts"),
        "manifest.counts",
        {
            "total",
            "unique_task_bundles",
            "task_ids_sha256",
            "by_split",
            "by_variant",
            "by_stage",
            "by_expert",
            "by_language",
        },
    )
    task_ids_sha = hashlib.sha256(
        "\n".join(
            sorted({sidecar.training_record.task_id for sidecar in all_sidecars})
        ).encode("utf-8")
    ).hexdigest()
    expected_counts = {
        "total": len(all_sidecars),
        "unique_task_bundles": len(
            {sidecar.task_bundle_sha256 for sidecar in all_sidecars}
        ),
        "task_ids_sha256": task_ids_sha,
        "by_split": summary["by_split"],
        "by_variant": summary["by_variant"],
        "by_stage": summary["by_stage"],
        "by_expert": summary["by_expert"],
        "by_language": summary["by_language"],
    }
    if dict(counts) != expected_counts:
        raise QuerySpecializationError(
            "manifest aggregate counts do not match sidecars"
        )
    summary = dict(summary)
    summary["manifest_sha256"] = manifest_snapshot_sha256
    summary["authenticated_file_sha256"] = authenticated_file_sha256
    return tuple(all_sidecars), dict(manifest), summary


def block_attention_auxiliary_loss(
    attention: Any,
    relevant_mask: Any,
    distractor_mask: Any,
    *,
    distractor_weight: float = 0.25,
    epsilon: float = 1e-8,
) -> tuple[Any, dict[str, Any]]:
    """Compute block-level relevance and distractor losses with PyTorch tensors.

    ``attention`` is expected to have shape ``[batch, ..., blocks]`` and to be
    normalized along its final dimension.  Masks have shape ``[batch, blocks]``.
    Averaging intermediate dimensions avoids forcing every attention head to
    imitate the same role-level distribution.
    """

    if distractor_weight < 0:
        raise QuerySpecializationError("distractor_weight must be non-negative")
    if epsilon <= 0:
        raise QuerySpecializationError("epsilon must be positive")
    if attention.ndim < 2:
        raise QuerySpecializationError("attention must include batch and block axes")
    if relevant_mask.ndim != 2 or distractor_mask.ndim != 2:
        raise QuerySpecializationError("block masks must have shape [batch, blocks]")
    if (
        attention.shape[0] != relevant_mask.shape[0]
        or attention.shape[-1] != relevant_mask.shape[1]
    ):
        raise QuerySpecializationError("attention and relevance mask shapes disagree")
    if tuple(relevant_mask.shape) != tuple(distractor_mask.shape):
        raise QuerySpecializationError("relevance and distractor mask shapes disagree")
    if bool((relevant_mask & distractor_mask).any()):
        raise QuerySpecializationError(
            "relevance and distractor masks must be disjoint"
        )
    reduce_dims = tuple(range(1, attention.ndim - 1))
    if not bool(attention.isfinite().all()):
        raise QuerySpecializationError("attention must contain only finite values")
    if bool((attention < 0).any()):
        raise QuerySpecializationError("attention probabilities must be non-negative")
    probability_sums = attention.sum(dim=-1)
    if not bool(((probability_sums - 1.0).abs() <= 1e-4).all()):
        raise QuerySpecializationError("attention must sum to one along the block axis")
    if not bool(relevant_mask.any(dim=-1).all()):
        raise QuerySpecializationError("every sample must include relevant blocks")
    block_mass = attention.mean(dim=reduce_dims) if reduce_dims else attention
    relevant = relevant_mask.to(dtype=block_mass.dtype)
    distractor = distractor_mask.to(dtype=block_mass.dtype)
    relevant_mass = (block_mass * relevant).sum(dim=-1)
    distractor_mass = (block_mass * distractor).sum(dim=-1)
    target_distribution = relevant / relevant.sum(dim=-1, keepdim=True)
    relevance_loss = (
        -(target_distribution * block_mass.clamp_min(epsilon).log()).sum(dim=-1).mean()
    )
    distractor_loss = distractor_mass.mean()
    total = relevance_loss + distractor_weight * distractor_loss
    return total, {
        "relevance_loss": relevance_loss.detach(),
        "distractor_loss": distractor_loss.detach(),
        "relevant_mass": relevant_mass.detach().mean(),
        "distractor_mass": distractor_mass.detach().mean(),
    }


def dataset_summary(records: Sequence[QueryTrainingRecord]) -> dict[str, Any]:
    """Return content-free coverage metadata for preflight manifests."""

    role_counts = {role: 0 for role in ROLES}
    block_kind_counts = {kind: 0 for kind in BLOCK_KINDS}
    for record in records:
        role_counts[record.role] += 1
        for block in record.blocks:
            block_kind_counts[block.kind] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "records": len(records),
        "pairs": len({record.pair_id for record in records}),
        "roles": {key: value for key, value in role_counts.items() if value},
        "block_kinds": {
            key: value for key, value in block_kind_counts.items() if value
        },
        "relevant_labels": sum(len(record.targets.relevant) for record in records),
        "distractor_labels": sum(len(record.targets.distractors) for record in records),
        "forbidden_labels": sum(len(record.targets.forbidden) for record in records),
    }
