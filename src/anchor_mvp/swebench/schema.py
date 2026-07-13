"""Strict, metadata-only task-card contracts for SWE-style datasets."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping


CARD_SCHEMA_VERSION = "anchor.swebench-card.v1"
CHAIN_INDEX_SCHEMA_VERSION = "anchor.swebench-chain-index.v1"
ALIGNMENT_POLICY_ID = "anchor.swebench-alignment.v1"
SOURCE_FINGERPRINT_POLICY_ID = "anchor.swebench-source-fingerprint.v1"
CHAIN_STAGES = (
    "planner",
    "tool_policy",
    "domain_builder",
    "domain_review",
    "security",
)

_DATASET_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/+~-]{0,254}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_COMMIT = re.compile(r"^[0-9a-fA-F]{7,64}$")
_IMAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,299}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPDX_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-.+]{0,79}$")
_ROUTING_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,79}$")

# These source fields may be present in local input metadata, but they may never
# cross into a task card or a content-free manifest.
FORBIDDEN_CARD_KEYS = frozenset(
    {
        "patch",
        "test_patch",
        "hints_text",
        "fail_to_pass",
        "pass_to_pass",
        "tests",
        "test_cases",
        "gold_patch",
        "gold_solution",
        "oracle_text",
    }
)


class SWEBenchValidationError(ValueError):
    """A SWE-style card, partition, or source contract failed closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest_value(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_string_set(values: Iterable[str]) -> str:
    """Digest a set without ambiguous string concatenation."""

    return digest_value(sorted(set(values)))


def validate_dataset_id(value: str) -> str:
    candidate = value.strip()
    if candidate != value or not _DATASET_ID.fullmatch(candidate):
        raise SWEBenchValidationError("dataset_id must be one owner/name identifier")
    return candidate


def validate_immutable_revision(value: str) -> str:
    candidate = value.strip().casefold()
    if candidate != value.casefold() or not _IMMUTABLE_REVISION.fullmatch(candidate):
        raise SWEBenchValidationError(
            "dataset_revision must be a full immutable 40-hex commit"
        )
    return candidate


def normalize_problem_statement(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


def clean_problem_statement(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized or len(normalized) > 200_000:
        raise SWEBenchValidationError(
            "problem_statement must contain 1..200000 characters"
        )
    if any(character == "\x00" for character in normalized):
        raise SWEBenchValidationError("problem_statement contains a NUL character")
    return normalized


def validate_instance_id(value: str) -> str:
    candidate = value.strip()
    if candidate != value or not _INSTANCE_ID.fullmatch(candidate):
        raise SWEBenchValidationError("instance_id is empty or unsafe")
    return candidate


def validate_repository(value: str) -> str:
    candidate = value.strip()
    if candidate != value or not _REPOSITORY.fullmatch(candidate):
        raise SWEBenchValidationError("repo must be one owner/name identifier")
    return candidate


def alignment_id(instance_id: str, repo: str) -> str:
    """Return a split/revision-independent identity for one upstream task."""

    identity = {
        "policy": ALIGNMENT_POLICY_ID,
        "instance_id": validate_instance_id(instance_id).casefold(),
        "repo": validate_repository(repo).casefold(),
    }
    return f"swe-align-v1:{digest_value(identity)}"


def source_fingerprint(
    *,
    repo: str,
    problem_statement: str,
    base_commit: str | None = None,
    image_name: str | None = None,
) -> str:
    """Bind a task to its workspace locator and normalized issue text."""

    clean_repo = validate_repository(repo)
    clean_problem = clean_problem_statement(problem_statement)
    clean_commit = _clean_base_commit(base_commit)
    clean_image = _clean_image_name(image_name)
    if clean_commit is None and clean_image is None:
        raise SWEBenchValidationError(
            "source metadata requires base_commit or image_name"
        )
    identity = {
        "policy": SOURCE_FINGERPRINT_POLICY_ID,
        "repo": clean_repo.casefold(),
        "base_commit": clean_commit,
        "image_name": clean_image,
        "problem_statement": normalize_problem_statement(clean_problem),
    }
    return digest_value(identity)


def _clean_base_commit(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    candidate = value.strip().casefold()
    if candidate != value.casefold() or not _COMMIT.fullmatch(candidate):
        raise SWEBenchValidationError("base_commit must be a 7..64 hex identifier")
    return candidate


def _clean_image_name(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    candidate = value.strip()
    if candidate != value or not _IMAGE_NAME.fullmatch(candidate):
        raise SWEBenchValidationError("image_name is empty or unsafe")
    return candidate


@dataclass(frozen=True)
class LicenseReference:
    spdx_id: str
    license_file_sha256: str
    ledger_sha256: str

    def __post_init__(self) -> None:
        if not _SPDX_ID.fullmatch(self.spdx_id):
            raise SWEBenchValidationError("license reference has an invalid SPDX id")
        for label, value in (
            ("license_file_sha256", self.license_file_sha256),
            ("ledger_sha256", self.ledger_sha256),
        ):
            if not _SHA256.fullmatch(value):
                raise SWEBenchValidationError(f"{label} must be one SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "spdx_id": self.spdx_id,
            "license_file_sha256": self.license_file_sha256,
            "ledger_sha256": self.ledger_sha256,
        }


@dataclass(frozen=True)
class SourceReference:
    dataset_id: str
    dataset_revision: str
    split: str
    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str | None = None
    image_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", validate_dataset_id(self.dataset_id))
        object.__setattr__(
            self,
            "dataset_revision",
            validate_immutable_revision(self.dataset_revision),
        )
        if self.split != "train":
            raise SWEBenchValidationError("task cards may only be imported from train")
        object.__setattr__(self, "instance_id", validate_instance_id(self.instance_id))
        object.__setattr__(self, "repo", validate_repository(self.repo))
        object.__setattr__(
            self,
            "problem_statement",
            clean_problem_statement(self.problem_statement),
        )
        object.__setattr__(self, "base_commit", _clean_base_commit(self.base_commit))
        object.__setattr__(self, "image_name", _clean_image_name(self.image_name))
        if self.base_commit is None and self.image_name is None:
            raise SWEBenchValidationError(
                "source metadata requires base_commit or image_name"
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "instance_id": self.instance_id,
            "repo": self.repo,
        }
        if self.base_commit is not None:
            result["base_commit"] = self.base_commit
        if self.image_name is not None:
            result["image_name"] = self.image_name
        return result


@dataclass(frozen=True)
class TaskCard:
    card_id: str
    alignment_id: str
    source_fingerprint: str
    source: SourceReference
    problem_statement: str
    license: LicenseReference
    domain_id: str
    language: str
    task_kind: str
    builder_expert_id: str
    reviewer_expert_id: str
    schema_version: str = CARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CARD_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported SWE task-card schema")
        expected_alignment = alignment_id(self.source.instance_id, self.source.repo)
        if self.alignment_id != expected_alignment:
            raise SWEBenchValidationError("task-card alignment_id is not canonical")
        if self.card_id != f"swe-card-v1:{expected_alignment.rsplit(':', 1)[1]}":
            raise SWEBenchValidationError("task-card card_id is not canonical")
        expected_fingerprint = source_fingerprint(
            repo=self.source.repo,
            problem_statement=self.problem_statement,
            base_commit=self.source.base_commit,
            image_name=self.source.image_name,
        )
        if self.source_fingerprint != expected_fingerprint:
            raise SWEBenchValidationError(
                "task-card source_fingerprint is not canonical"
            )
        if (
            clean_problem_statement(self.problem_statement)
            != self.source.problem_statement
        ):
            raise SWEBenchValidationError(
                "source and card problem_statement values disagree"
            )
        for label, value in (
            ("domain_id", self.domain_id),
            ("language", self.language),
            ("task_kind", self.task_kind),
            ("builder_expert_id", self.builder_expert_id),
            ("reviewer_expert_id", self.reviewer_expert_id),
        ):
            if not _ROUTING_ID.fullmatch(value):
                raise SWEBenchValidationError(f"{label} is not a safe routing id")

    @classmethod
    def from_metadata(
        cls,
        *,
        dataset_id: str,
        dataset_revision: str,
        split: str,
        instance_id: str,
        repo: str,
        problem_statement: str,
        license_reference: LicenseReference,
        domain_id: str,
        language: str,
        task_kind: str,
        builder_expert_id: str,
        reviewer_expert_id: str,
        base_commit: str | None = None,
        image_name: str | None = None,
    ) -> "TaskCard":
        source = SourceReference(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            split=split,
            instance_id=instance_id,
            repo=repo,
            problem_statement=problem_statement,
            base_commit=base_commit,
            image_name=image_name,
        )
        identity = alignment_id(source.instance_id, source.repo)
        fingerprint = source_fingerprint(
            repo=source.repo,
            problem_statement=source.problem_statement,
            base_commit=source.base_commit,
            image_name=source.image_name,
        )
        return cls(
            card_id=f"swe-card-v1:{identity.rsplit(':', 1)[1]}",
            alignment_id=identity,
            source_fingerprint=fingerprint,
            source=source,
            problem_statement=source.problem_statement,
            license=license_reference,
            domain_id=domain_id,
            language=language,
            task_kind=task_kind,
            builder_expert_id=builder_expert_id,
            reviewer_expert_id=reviewer_expert_id,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "card_id": self.card_id,
            "alignment_id": self.alignment_id,
            "source_fingerprint": self.source_fingerprint,
            "source": self.source.to_dict(),
            "problem_statement": self.problem_statement,
            "license": self.license.to_dict(),
            "domain_id": self.domain_id,
            "language": self.language,
            "task_kind": self.task_kind,
            "routing_contract": {
                "builder_expert_id": self.builder_expert_id,
                "reviewer_expert_id": self.reviewer_expert_id,
            },
            "chain_contract": {
                "stages": list(CHAIN_STAGES),
                "execution_sandbox_required": True,
                "model_tool_policy_is_authority": False,
            },
        }
        reject_forbidden_card_keys(result)
        return result


@dataclass(frozen=True)
class ChainIndexEntry:
    alignment_id: str
    completed_stages: tuple[str, ...]
    execution_sandbox_audit_sha256: str
    domain_id: str
    planner_builder_expert_id: str
    planner_reviewer_expert_id: str
    executed_builder_expert_id: str
    executed_reviewer_expert_id: str
    schema_version: str = CHAIN_INDEX_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ChainIndexEntry":
        expected = {
            "schema_version",
            "alignment_id",
            "completed_stages",
            "execution_sandbox_audit_sha256",
            "planner_route",
            "executed_route",
        }
        if set(value) != expected:
            raise SWEBenchValidationError("chain index row has unexpected fields")
        raw_stages = value.get("completed_stages")
        if not isinstance(raw_stages, list) or not all(
            isinstance(item, str) for item in raw_stages
        ):
            raise SWEBenchValidationError("completed_stages must be a string list")
        planner_route = value.get("planner_route")
        executed_route = value.get("executed_route")
        route_fields = {
            "domain_id",
            "builder_expert_id",
            "reviewer_expert_id",
        }
        if not isinstance(planner_route, Mapping) or set(planner_route) != route_fields:
            raise SWEBenchValidationError("planner_route has unexpected fields")
        if (
            not isinstance(executed_route, Mapping)
            or set(executed_route) != route_fields
        ):
            raise SWEBenchValidationError("executed_route has unexpected fields")
        entry = cls(
            schema_version=str(value.get("schema_version", "")),
            alignment_id=str(value.get("alignment_id", "")),
            completed_stages=tuple(raw_stages),
            execution_sandbox_audit_sha256=str(
                value.get("execution_sandbox_audit_sha256", "")
            ),
            domain_id=str(planner_route.get("domain_id", "")),
            planner_builder_expert_id=str(planner_route.get("builder_expert_id", "")),
            planner_reviewer_expert_id=str(planner_route.get("reviewer_expert_id", "")),
            executed_builder_expert_id=str(executed_route.get("builder_expert_id", "")),
            executed_reviewer_expert_id=str(
                executed_route.get("reviewer_expert_id", "")
            ),
        )
        if str(executed_route.get("domain_id", "")) != entry.domain_id:
            raise SWEBenchValidationError(
                "planner-selected and executed domain_id values disagree"
            )
        entry.validate()
        return entry

    def validate(self) -> None:
        if self.schema_version != CHAIN_INDEX_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported chain-index schema")
        if not re.fullmatch(r"swe-align-v1:[0-9a-f]{64}", self.alignment_id):
            raise SWEBenchValidationError("chain index has an invalid alignment_id")
        if self.completed_stages != CHAIN_STAGES:
            raise SWEBenchValidationError(
                "chain index does not contain the complete ordered stage contract"
            )
        if not _SHA256.fullmatch(self.execution_sandbox_audit_sha256):
            raise SWEBenchValidationError(
                "execution sandbox audit must be bound by SHA-256"
            )
        for label, value in (
            ("domain_id", self.domain_id),
            ("planner_builder_expert_id", self.planner_builder_expert_id),
            ("planner_reviewer_expert_id", self.planner_reviewer_expert_id),
            ("executed_builder_expert_id", self.executed_builder_expert_id),
            ("executed_reviewer_expert_id", self.executed_reviewer_expert_id),
        ):
            if not _ROUTING_ID.fullmatch(value):
                raise SWEBenchValidationError(f"{label} is not a safe routing id")
        if (
            self.planner_builder_expert_id != self.executed_builder_expert_id
            or self.planner_reviewer_expert_id != self.executed_reviewer_expert_id
        ):
            raise SWEBenchValidationError(
                "planner-selected experts differ from executed experts"
            )


def reject_forbidden_card_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_CARD_KEYS:
                raise SWEBenchValidationError(
                    "forbidden patch, hint, or test material entered a task card"
                )
            reject_forbidden_card_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_forbidden_card_keys(child)


def validate_complete_chain_index(
    cards: Iterable[TaskCard], entries: Iterable[ChainIndexEntry]
) -> int:
    card_list = list(cards)
    cards_by_alignment = {card.alignment_id: card for card in card_list}
    if len(cards_by_alignment) != len(card_list):
        raise SWEBenchValidationError("task cards repeat one alignment_id")
    card_ids = set(cards_by_alignment)
    index_ids: set[str] = set()
    for entry in entries:
        entry.validate()
        if entry.alignment_id in index_ids:
            raise SWEBenchValidationError("chain index repeats one alignment_id")
        index_ids.add(entry.alignment_id)
        card = cards_by_alignment.get(entry.alignment_id)
        if card is None:
            continue
        if entry.domain_id != card.domain_id:
            raise SWEBenchValidationError(
                "planner-selected domain differs from the task-card domain"
            )
        if (
            entry.planner_builder_expert_id != card.builder_expert_id
            or entry.planner_reviewer_expert_id != card.reviewer_expert_id
        ):
            raise SWEBenchValidationError(
                "planner-selected experts differ from the task-card routing contract"
            )
    if index_ids != card_ids:
        raise SWEBenchValidationError(
            "complete chain count must equal task-card/alignment count"
        )
    return len(index_ids)
