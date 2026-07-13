"""Fail-closed train/held-out and repository-license partition gates."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .schema import (
    LicenseReference,
    SWEBenchValidationError,
    digest_string_set,
    source_fingerprint,
    validate_dataset_id,
    validate_immutable_revision,
    validate_instance_id,
    validate_repository,
)


ALLOWLIST_SCHEMA_VERSION = "anchor.swebench-train-allowlist.v1"
HELDOUT_REGISTRY_SCHEMA_VERSION = "anchor.swebench-heldout-registry.v1"
LICENSE_LEDGER_SCHEMA_VERSION = "anchor.swebench-license-ledger.v1"

_PERMANENT_VARIANTS = frozenset({"full", "lite", "verified"})


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def permanent_heldout_variant(dataset_id: str) -> str | None:
    """Classify official Full/Lite/Verified aliases as held-out forever."""

    candidate = validate_dataset_id(dataset_id)
    owner, name = candidate.casefold().split("/", 1)
    if owner not in {"swe-bench", "princeton-nlp"}:
        return None
    normalized = name.replace("-", "_")
    return {
        "swe_bench": "full",
        "swe_bench_full": "full",
        "swe_bench_lite": "lite",
        "swe_bench_verified": "verified",
    }.get(normalized)


def is_supported_train_dataset(dataset_id: str) -> bool:
    candidate = validate_dataset_id(dataset_id)
    owner, name = candidate.casefold().split("/", 1)
    return owner in {"swe-bench", "princeton-nlp"} and (
        (
            owner == "swe-bench"
            and (name == "swe-smith" or name.startswith("swe-smith-"))
        )
        or name.replace("-", "_") == "swe_bench"
    )


def iter_jsonl_mappings(path: str | Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise SWEBenchValidationError("a configured metadata JSONL is missing") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SWEBenchValidationError(
                    f"metadata JSONL line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise SWEBenchValidationError(
                    f"metadata JSONL line {line_number} is not an object"
                )
            yield line_number, value


def _load_json_object(path: str | Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SWEBenchValidationError(f"{label} is missing or invalid") from exc
    if not isinstance(value, Mapping):
        raise SWEBenchValidationError(f"{label} root must be an object")
    return value


@dataclass(frozen=True)
class TrainAllowlist:
    dataset_id: str
    dataset_revision: str
    instance_ids: frozenset[str]
    file_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "TrainAllowlist":
        value = _load_json_object(path, "train allowlist")
        expected = {
            "schema_version",
            "dataset_id",
            "dataset_revision",
            "split",
            "instance_ids",
        }
        if set(value) != expected:
            raise SWEBenchValidationError("train allowlist has unexpected fields")
        if value.get("schema_version") != ALLOWLIST_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported train-allowlist schema")
        if value.get("split") != "train":
            raise SWEBenchValidationError("train allowlist must declare split=train")
        raw_ids = value.get("instance_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise SWEBenchValidationError(
                "train allowlist instance_ids must be non-empty"
            )
        ids = [validate_instance_id(str(item)) for item in raw_ids]
        if len(set(ids)) != len(ids):
            raise SWEBenchValidationError("train allowlist repeats an instance_id")
        return cls(
            dataset_id=validate_dataset_id(str(value.get("dataset_id", ""))),
            dataset_revision=validate_immutable_revision(
                str(value.get("dataset_revision", ""))
            ),
            instance_ids=frozenset(ids),
            file_sha256=file_sha256(path),
        )

    @property
    def ids_sha256(self) -> str:
        return digest_string_set(self.instance_ids)


@dataclass(frozen=True)
class LicenseApproval:
    spdx_id: str
    license_file_sha256: str


@dataclass(frozen=True)
class LicenseLedger:
    dataset_id: str
    dataset_revision: str
    approvals: Mapping[str, LicenseApproval]
    file_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "LicenseLedger":
        value = _load_json_object(path, "license ledger")
        expected = {
            "schema_version",
            "dataset_id",
            "dataset_revision",
            "repositories",
        }
        if set(value) != expected:
            raise SWEBenchValidationError("license ledger has unexpected fields")
        if value.get("schema_version") != LICENSE_LEDGER_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported license-ledger schema")
        raw_repositories = value.get("repositories")
        if not isinstance(raw_repositories, Mapping) or not raw_repositories:
            raise SWEBenchValidationError(
                "license ledger repositories must be non-empty"
            )
        approvals: dict[str, LicenseApproval] = {}
        entry_fields = {
            "spdx_id",
            "license_file_sha256",
            "reviewed",
            "training_allowed",
            "metadata_redistribution_allowed",
            "attribution",
        }
        for raw_repo, raw_entry in raw_repositories.items():
            repo = validate_repository(str(raw_repo))
            if not isinstance(raw_entry, Mapping) or set(raw_entry) != entry_fields:
                raise SWEBenchValidationError(
                    "a license ledger repository has unexpected fields"
                )
            if (
                raw_entry.get("reviewed") is not True
                or raw_entry.get("training_allowed") is not True
                or raw_entry.get("metadata_redistribution_allowed") is not True
            ):
                raise SWEBenchValidationError(
                    "every repository license must be reviewed and explicitly approved"
                )
            attribution = raw_entry.get("attribution")
            if not isinstance(attribution, str) or not attribution.strip():
                raise SWEBenchValidationError(
                    "every repository license requires non-empty attribution"
                )
            approval = LicenseApproval(
                spdx_id=str(raw_entry.get("spdx_id", "")),
                license_file_sha256=str(raw_entry.get("license_file_sha256", "")),
            )
            # Reuse the public card validator for SPDX/hash validation without
            # persisting the ledger attribution in task cards.
            LicenseReference(
                spdx_id=approval.spdx_id,
                license_file_sha256=approval.license_file_sha256,
                ledger_sha256=file_sha256(path),
            )
            approvals[repo.casefold()] = approval
        return cls(
            dataset_id=validate_dataset_id(str(value.get("dataset_id", ""))),
            dataset_revision=validate_immutable_revision(
                str(value.get("dataset_revision", ""))
            ),
            approvals=approvals,
            file_sha256=file_sha256(path),
        )

    def reference_for(self, repo: str) -> LicenseReference:
        approval = self.approvals.get(validate_repository(repo).casefold())
        if approval is None:
            raise SWEBenchValidationError(
                "repository is absent from the reviewed license ledger"
            )
        return LicenseReference(
            spdx_id=approval.spdx_id,
            license_file_sha256=approval.license_file_sha256,
            ledger_sha256=self.file_sha256,
        )


@dataclass(frozen=True)
class HeldoutDenylist:
    instance_ids: frozenset[str]
    source_fingerprints: frozenset[str]
    repositories: frozenset[str]
    variant_row_counts: Mapping[str, int]
    file_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "HeldoutDenylist":
        registry_path = Path(path).resolve()
        value = _load_json_object(registry_path, "held-out registry")
        if set(value) != {"schema_version", "sources"}:
            raise SWEBenchValidationError("held-out registry has unexpected fields")
        if value.get("schema_version") != HELDOUT_REGISTRY_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported held-out registry schema")
        raw_sources = value.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise SWEBenchValidationError("held-out registry sources must be non-empty")

        instance_to_fingerprint: dict[str, str] = {}
        fingerprints: set[str] = set()
        repositories: set[str] = set()
        variant_row_counts = {variant: 0 for variant in _PERMANENT_VARIANTS}
        source_fields = {
            "dataset_id",
            "dataset_revision",
            "split",
            "metadata_jsonl",
        }
        seen_variants: set[str] = set()
        for raw_source in raw_sources:
            if not isinstance(raw_source, Mapping) or set(raw_source) != source_fields:
                raise SWEBenchValidationError(
                    "held-out registry source has unexpected fields"
                )
            dataset_id = validate_dataset_id(str(raw_source.get("dataset_id", "")))
            variant = permanent_heldout_variant(dataset_id)
            if variant is None:
                raise SWEBenchValidationError(
                    "held-out registry accepts only SWE-bench Full/Lite/Verified"
                )
            seen_variants.add(variant)
            validate_immutable_revision(str(raw_source.get("dataset_revision", "")))
            if raw_source.get("split") not in {"dev", "test"}:
                raise SWEBenchValidationError(
                    "held-out registry sources must declare dev or test"
                )
            raw_path = raw_source.get("metadata_jsonl")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise SWEBenchValidationError(
                    "held-out registry metadata_jsonl path is missing"
                )
            metadata_path = Path(raw_path)
            if not metadata_path.is_absolute():
                metadata_path = registry_path.parent / metadata_path
            row_count = 0
            for _, row in iter_jsonl_mappings(metadata_path):
                instance_id = _row_string(row, "instance_id")
                repo = _row_string(row, "repo")
                problem_statement = _row_string(row, "problem_statement")
                fingerprint = source_fingerprint(
                    repo=repo,
                    problem_statement=problem_statement,
                    base_commit=_row_optional_string(row, "base_commit"),
                    image_name=_row_optional_string(row, "image_name"),
                )
                existing = instance_to_fingerprint.get(instance_id)
                if existing is not None and existing != fingerprint:
                    raise SWEBenchValidationError(
                        "one held-out instance_id maps to conflicting source metadata"
                    )
                instance_to_fingerprint[instance_id] = fingerprint
                fingerprints.add(fingerprint)
                repositories.add(validate_repository(repo).casefold())
                row_count += 1
            if row_count == 0:
                raise SWEBenchValidationError(
                    "held-out registry metadata source is empty"
                )
            variant_row_counts[variant] += row_count

        if seen_variants != _PERMANENT_VARIANTS:
            raise SWEBenchValidationError(
                "held-out registry must include Full, Lite, and Verified sources"
            )
        return cls(
            instance_ids=frozenset(instance_to_fingerprint),
            source_fingerprints=frozenset(fingerprints),
            repositories=frozenset(repositories),
            variant_row_counts=variant_row_counts,
            file_sha256=file_sha256(registry_path),
        )


@dataclass(frozen=True)
class PartitionGuard:
    allowlist: TrainAllowlist
    denylist: HeldoutDenylist
    license_ledger: LicenseLedger
    strict_repo_isolation: bool = True

    @classmethod
    def load(
        cls,
        *,
        dataset_id: str,
        dataset_revision: str,
        allowlist_path: str | Path,
        heldout_registry_path: str | Path,
        license_ledger_path: str | Path,
    ) -> "PartitionGuard":
        clean_dataset = validate_dataset_id(dataset_id)
        clean_revision = validate_immutable_revision(dataset_revision)
        variant = permanent_heldout_variant(clean_dataset)
        if variant in {"lite", "verified"}:
            raise SWEBenchValidationError(
                "SWE-bench Lite/Verified are permanent held-out datasets"
            )
        if not is_supported_train_dataset(clean_dataset):
            raise SWEBenchValidationError(
                "MVP train import accepts only official SWE-smith datasets or "
                "the ordinary SWE-bench train split"
            )
        allowlist = TrainAllowlist.load(allowlist_path)
        ledger = LicenseLedger.load(license_ledger_path)
        for label, candidate_id, candidate_revision in (
            (
                "train allowlist",
                allowlist.dataset_id,
                allowlist.dataset_revision,
            ),
            ("license ledger", ledger.dataset_id, ledger.dataset_revision),
        ):
            if (
                candidate_id.casefold() != clean_dataset.casefold()
                or candidate_revision != clean_revision
            ):
                raise SWEBenchValidationError(
                    f"{label} does not match the pinned dataset revision"
                )
        denylist = HeldoutDenylist.load(heldout_registry_path)
        overlap = allowlist.instance_ids & denylist.instance_ids
        if overlap:
            raise SWEBenchValidationError(
                "train allowlist intersects the permanent held-out denylist"
            )
        return cls(
            allowlist=allowlist,
            denylist=denylist,
            license_ledger=ledger,
        )

    def validate_train_source(
        self,
        *,
        instance_id: str,
        repo: str,
        fingerprint: str,
    ) -> None:
        clean_id = validate_instance_id(instance_id)
        clean_repo = validate_repository(repo).casefold()
        if clean_id in self.denylist.instance_ids:
            raise SWEBenchValidationError(
                "train instance_id is present in the held-out denylist"
            )
        if fingerprint in self.denylist.source_fingerprints:
            raise SWEBenchValidationError(
                "train source fingerprint is present in the held-out denylist"
            )
        if self.strict_repo_isolation and clean_repo in self.denylist.repositories:
            raise SWEBenchValidationError(
                "train repository is present in the held-out repository denylist"
            )
        self.license_ledger.reference_for(repo)


def _row_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SWEBenchValidationError(f"metadata row requires string field {key}")
    return value


def _row_optional_string(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SWEBenchValidationError(f"metadata field {key} must be a string")
    return value
