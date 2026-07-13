"""Local-only metadata importer for SWE-bench/SWE-smith style task cards."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .partition import (
    PartitionGuard,
    file_sha256,
    iter_jsonl_mappings,
)
from .schema import (
    ChainIndexEntry,
    SWEBenchValidationError,
    TaskCard,
    canonical_json,
    digest_string_set,
    source_fingerprint,
    validate_complete_chain_index,
    validate_dataset_id,
    validate_immutable_revision,
    validate_instance_id,
)


IMPORT_MANIFEST_SCHEMA_VERSION = "anchor.swebench-import-manifest.v1"

_MANIFEST_CONTENT_KEYS = frozenset(
    {
        "problem_statement",
        "patch",
        "test_patch",
        "hints_text",
        "fail_to_pass",
        "pass_to_pass",
        "tests",
        "test_cases",
        "instance_id",
        "card_id",
        "alignment_id",
        "repo",
        "base_commit",
        "image_name",
    }
)


@dataclass(frozen=True)
class ImportConfig:
    source_jsonl: Path
    dataset_id: str
    dataset_revision: str
    train_allowlist: Path
    heldout_registry: Path
    license_ledger: Path
    domain_id: str = "python-repository"
    language: str = "python"
    task_kind: str = "issue-resolution"
    builder_expert_id: str = "swe-shared-builder"
    reviewer_expert_id: str = "swe-shared-reviewer"
    split: str = "train"
    chain_index: Path | None = None
    cards_output: Path | None = None
    manifest_output: Path | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class ImportResult:
    cards: tuple[TaskCard, ...]
    manifest: Mapping[str, Any]


def import_metadata_cards(config: ImportConfig) -> ImportResult:
    dataset_id = validate_dataset_id(config.dataset_id)
    dataset_revision = validate_immutable_revision(config.dataset_revision)
    if config.split != "train":
        raise SWEBenchValidationError(
            "metadata task-card import is restricted to split=train"
        )
    if not config.dry_run and (
        config.cards_output is None or config.manifest_output is None
    ):
        raise SWEBenchValidationError(
            "non-dry import requires cards_output and manifest_output"
        )
    if (
        config.cards_output is not None
        and config.manifest_output is not None
        and config.cards_output.resolve() == config.manifest_output.resolve()
    ):
        raise SWEBenchValidationError("card and manifest outputs must differ")

    guard = PartitionGuard.load(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        allowlist_path=config.train_allowlist,
        heldout_registry_path=config.heldout_registry,
        license_ledger_path=config.license_ledger,
    )

    cards: list[TaskCard] = []
    selected_ids: set[str] = set()
    selected_alignments: set[str] = set()
    source_row_count = 0
    skipped_not_allowlisted = 0
    for _, row in iter_jsonl_mappings(config.source_jsonl):
        source_row_count += 1
        instance_id = _required_string(row, "instance_id")
        if instance_id not in guard.allowlist.instance_ids:
            skipped_not_allowlisted += 1
            continue
        if instance_id in selected_ids:
            raise SWEBenchValidationError(
                "selected source metadata repeats an instance_id"
            )
        _validate_optional_source_identity(
            row,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            split=config.split,
        )
        repo = _required_string(row, "repo")
        problem_statement = _required_string(row, "problem_statement")
        base_commit = _optional_string(row, "base_commit")
        image_name = _optional_string(row, "image_name")
        fingerprint = source_fingerprint(
            repo=repo,
            problem_statement=problem_statement,
            base_commit=base_commit,
            image_name=image_name,
        )
        guard.validate_train_source(
            instance_id=instance_id,
            repo=repo,
            fingerprint=fingerprint,
        )
        card = TaskCard.from_metadata(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            split=config.split,
            instance_id=instance_id,
            repo=repo,
            problem_statement=problem_statement,
            base_commit=base_commit,
            image_name=image_name,
            license_reference=guard.license_ledger.reference_for(repo),
            domain_id=config.domain_id,
            language=config.language,
            task_kind=config.task_kind,
            builder_expert_id=config.builder_expert_id,
            reviewer_expert_id=config.reviewer_expert_id,
        )
        if card.alignment_id in selected_alignments:
            raise SWEBenchValidationError(
                "two selected source rows map to one alignment_id"
            )
        selected_ids.add(validate_instance_id(instance_id))
        selected_alignments.add(card.alignment_id)
        cards.append(card)

    if source_row_count == 0:
        raise SWEBenchValidationError("source metadata JSONL is empty")
    missing = guard.allowlist.instance_ids - selected_ids
    if missing:
        raise SWEBenchValidationError(
            "train allowlist contains instance_ids absent from source metadata"
        )
    cards.sort(key=lambda item: item.alignment_id)

    complete_chain_count = 0
    chain_index_sha256: str | None = None
    if config.chain_index is not None:
        entries = [
            ChainIndexEntry.from_mapping(row)
            for _, row in iter_jsonl_mappings(config.chain_index)
        ]
        complete_chain_count = validate_complete_chain_index(cards, entries)
        chain_index_sha256 = file_sha256(config.chain_index)

    cards_file_sha256 = _cards_sha256(cards)
    manifest = _build_manifest(
        config=config,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        guard=guard,
        cards=cards,
        source_row_count=source_row_count,
        skipped_not_allowlisted=skipped_not_allowlisted,
        cards_file_sha256=cards_file_sha256,
        complete_chain_count=complete_chain_count,
        chain_index_sha256=chain_index_sha256,
    )
    assert_content_free_manifest(manifest)
    if not config.dry_run:
        assert config.cards_output is not None
        assert config.manifest_output is not None
        _write_outputs(
            cards,
            manifest,
            cards_path=config.cards_output,
            manifest_path=config.manifest_output,
        )
    return ImportResult(cards=tuple(cards), manifest=manifest)


def _build_manifest(
    *,
    config: ImportConfig,
    dataset_id: str,
    dataset_revision: str,
    guard: PartitionGuard,
    cards: list[TaskCard],
    source_row_count: int,
    skipped_not_allowlisted: int,
    cards_file_sha256: str,
    complete_chain_count: int,
    chain_index_sha256: str | None,
) -> dict[str, Any]:
    alignment_ids = [card.alignment_id for card in cards]
    fingerprints = [card.source_fingerprint for card in cards]
    manifest: dict[str, Any] = {
        "schema_version": IMPORT_MANIFEST_SCHEMA_VERSION,
        "source": {
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "split": "train",
            "metadata_file_sha256": file_sha256(config.source_jsonl),
            "row_count": source_row_count,
            "skipped_not_allowlisted": skipped_not_allowlisted,
        },
        "partition": {
            "train_allowlist_file_sha256": guard.allowlist.file_sha256,
            "train_allowlist_ids_sha256": guard.allowlist.ids_sha256,
            "train_allowlist_count": len(guard.allowlist.instance_ids),
            "heldout_registry_file_sha256": guard.denylist.file_sha256,
            "heldout_instance_ids_sha256": digest_string_set(
                guard.denylist.instance_ids
            ),
            "heldout_source_fingerprints_sha256": digest_string_set(
                guard.denylist.source_fingerprints
            ),
            "heldout_repositories_sha256": digest_string_set(
                guard.denylist.repositories
            ),
            "heldout_unique_instance_count": len(guard.denylist.instance_ids),
            "heldout_variant_row_counts": dict(
                sorted(guard.denylist.variant_row_counts.items())
            ),
            "strict_repo_isolation": guard.strict_repo_isolation,
            "full_lite_verified_permanent_deny": True,
        },
        "license_gate": {
            "ledger_file_sha256": guard.license_ledger.file_sha256,
            "approved_repository_count": len(guard.license_ledger.approvals),
            "unknown_repository_policy": "fail_closed",
        },
        "routing": {
            "domain_id": config.domain_id,
            "language": config.language,
            "task_kind": config.task_kind,
            "builder_expert_id": config.builder_expert_id,
            "reviewer_expert_id": config.reviewer_expert_id,
            "planner_selection_must_match_execution": True,
        },
        "cards": {
            "card_count": len(cards),
            "unique_alignment_count": len(set(alignment_ids)),
            "planned_chain_count": len(cards),
            "cards_file_sha256": cards_file_sha256,
            "alignment_ids_sha256": digest_string_set(alignment_ids),
            "source_fingerprints_sha256": digest_string_set(fingerprints),
            "one_card_per_alignment": len(cards) == len(set(alignment_ids)),
            "forbidden_oracle_fields_absent": True,
        },
        "complete_chains": {
            "complete_chain_count": complete_chain_count,
            "coverage_complete": complete_chain_count == len(cards),
            "chain_index_file_sha256": chain_index_sha256,
            "sandbox_audit_required": True,
        },
        "content_emitted": False,
    }
    return manifest


def assert_content_free_manifest(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _MANIFEST_CONTENT_KEYS:
                raise SWEBenchValidationError(
                    "import manifest contains source/card content"
                )
            assert_content_free_manifest(child)
    elif isinstance(value, list):
        for child in value:
            assert_content_free_manifest(child)


def _cards_sha256(cards: list[TaskCard]) -> str:
    digest = sha256()
    for card in cards:
        digest.update((canonical_json(card.to_dict()) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _write_outputs(
    cards: list[TaskCard],
    manifest: Mapping[str, Any],
    *,
    cards_path: Path,
    manifest_path: Path,
) -> None:
    cards_path = cards_path.resolve()
    manifest_path = manifest_path.resolve()
    cards_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cards_tmp = cards_path.with_name(cards_path.name + ".tmp")
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        with cards_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for card in cards:
                handle.write(canonical_json(card.to_dict()) + "\n")
        manifest_tmp.write_text(
            json.dumps(
                dict(manifest),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        cards_tmp.replace(cards_path)
        manifest_tmp.replace(manifest_path)
        manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
            f"{file_sha256(manifest_path)}  {manifest_path.name}\n",
            encoding="ascii",
            newline="\n",
        )
    finally:
        cards_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SWEBenchValidationError(f"source metadata requires string field {key}")
    return value


def _optional_string(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SWEBenchValidationError(f"source metadata field {key} must be a string")
    return value


def _validate_optional_source_identity(
    row: Mapping[str, Any],
    *,
    dataset_id: str,
    dataset_revision: str,
    split: str,
) -> None:
    expected = {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "split": split,
    }
    for key, expected_value in expected.items():
        value = row.get(key)
        if value is not None and value != expected_value:
            raise SWEBenchValidationError(
                f"source metadata {key} disagrees with the pinned import"
            )
