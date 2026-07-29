from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from jsonschema import Draft202012Validator
import pytest

from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_minimal320_consumer_v1 as consumer,
)
from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1 as legacy_matrix,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical_bytes(value)
    return hashlib.sha256(raw).hexdigest()


def _blob(raw: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _write(repo: Path, relative: str, raw: bytes) -> None:
    path = repo / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _json_raw(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _row(
    sequence: int,
    *,
    selection_kind: str,
    role: str,
    language: str,
    bundle: str,
    tool_family: str | None = None,
    identity_class: str | None = None,
    router_label: str | None = None,
    review_verdict: str | None = None,
    review_fault_type: str | None = None,
) -> dict[str, Any]:
    source_record_id_sha256 = _sha(f"record-{sequence:04d}".encode())
    task_bundle_sha256 = _sha(bundle.encode())
    identity_derivative_spec_sha256 = (
        _sha(f"identity-spec-{sequence:04d}".encode())
        if selection_kind == "identity"
        else None
    )
    row = {
        "schema_version": consumer._SELECTOR_CANDIDATE_SCHEMA_VERSION,
        "idempotency_key": _sha(f"idempotency-{sequence:04d}".encode()),
        "teacher_request_idempotency_key": _sha(f"idempotency-{sequence:04d}".encode()),
        "source_record_id_sha256": source_record_id_sha256,
        "source_content_sha256": _sha(f"content-{sequence:04d}".encode()),
        "source_serialization_identity_sha256": _sha(
            f"serialization-{sequence:04d}".encode()
        ),
        "teacher_request_record_id_sha256": source_record_id_sha256,
        "parent_source_record_id_sha256": _sha(
            f"parent-record-{sequence:04d}".encode()
        ),
        "task_bundle_sha256": task_bundle_sha256,
        "parent_task_bundle_sha256": _sha(f"parent-bundle-{sequence:04d}".encode()),
        "source_semantic_sha256": _sha(f"semantic-{sequence:04d}".encode()),
        "overlay_row_sha256": _sha(f"overlay-{sequence:04d}".encode()),
        "identity_derivative_spec_sha256": identity_derivative_spec_sha256,
        "selection_kind": selection_kind,
        "output_expert_role": role,
        "training_asset": consumer._ROLE_TO_ASSET[role],
        "validation_contract_role": (
            "identity" if selection_kind == "identity" else role
        ),
        "language": language,
        "tool_family": tool_family,
        "identity_class": identity_class,
        "identity_parent_class": (
            identity_class if selection_kind == "identity" else None
        ),
        "identity_provenance_intent": (
            "air_provenance_invariant" if selection_kind == "identity" else None
        ),
        "router_label": router_label,
        "review_verdict": review_verdict,
        "review_fault_type": review_fault_type,
        "content_retained": False,
    }
    row["candidate_id_sha256"] = _sha(consumer._candidate_identity_payload(row))
    return row


def _refresh_candidate_identity(row: dict[str, Any]) -> None:
    row["candidate_id_sha256"] = _sha(consumer._candidate_identity_payload(row))


def _exact_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    sequence = 0
    roles = ("humor", "serious", "angry", "tool", "review")
    families = tuple(f"family_{index:02d}" for index in range(10))

    for family in families:
        for language in ("en", "zh"):
            bundle = f"core|{family}|{language}"
            for role in roles:
                rows.append(
                    _row(
                        sequence,
                        selection_kind="full_core",
                        role=role,
                        language=language,
                        bundle=bundle,
                        tool_family=family,
                    )
                )
                sequence += 1

    faults = tuple(consumer._REVIEW_FAULT_COUNTS)
    fault_schedule = (
        faults[0],
        faults[0],
        faults[1],
        faults[1],
        faults[2],
        faults[2],
        faults[3],
        faults[3],
        faults[4],
        faults[4],
    )
    fail_cells = [
        {
            "tool_family": families[index],
            "language": "en" if index < 5 else "zh",
            "fault": fault_schedule[index],
        }
        for index in range(10)
    ]
    fail_by_cell = {
        (cell["tool_family"], cell["language"]): cell["fault"] for cell in fail_cells
    }
    for family in families:
        for language in ("en", "zh"):
            bundle = f"depth|{family}|{language}"
            fault = fail_by_cell.get((family, language))
            rows.append(
                _row(
                    sequence,
                    selection_kind="tool_review_depth",
                    role="tool",
                    language=language,
                    bundle=bundle,
                    tool_family=family,
                )
            )
            sequence += 1
            rows.append(
                _row(
                    sequence,
                    selection_kind="tool_review_depth",
                    role="review",
                    language=language,
                    bundle=bundle,
                    tool_family=family,
                    review_verdict="fail" if fault else "pass",
                    review_fault_type=fault,
                )
            )
            sequence += 1

    identity_counts = {
        "air_attribution": 4,
        "false_google_attribution": 3,
        "false_openai_attribution": 3,
    }
    for identity_class, count in identity_counts.items():
        for language in ("en", "zh"):
            for index in range(count):
                bundle = f"identity|{identity_class}|{language}|{index}"
                for role in roles:
                    rows.append(
                        _row(
                            sequence,
                            selection_kind="identity",
                            role=role,
                            language=language,
                            bundle=bundle,
                            identity_class=identity_class,
                        )
                    )
                    sequence += 1

    router_counts = {
        "humor": {"en": 14, "zh": 13},
        "serious": {"en": 13, "zh": 13},
        "angry": {"en": 13, "zh": 14},
    }
    for label, language_counts in router_counts.items():
        for language in ("en", "zh"):
            for index in range(language_counts[language]):
                rows.append(
                    _row(
                        sequence,
                        selection_kind="router",
                        role="router",
                        language=language,
                        bundle=f"router|{label}|{language}|{index}",
                        router_label=label,
                    )
                )
                sequence += 1
    assert sequence == 320
    return (
        sorted(rows, key=lambda item: item["candidate_id_sha256"]),
        sorted(
            fail_cells,
            key=lambda item: (
                item["tool_family"],
                item["language"],
                item["fault"],
            ),
        ),
    )


def _strata() -> dict[str, Any]:
    return {
        "partition_records": {
            "full_core": 100,
            "tool_review_depth": 40,
            "identity": 100,
            "router": 80,
        },
        "partition_bundles": {
            "full_core": 20,
            "tool_review_depth": 20,
            "identity": 20,
        },
        "tool_family_count": 10,
        "identity_bundle_cells": {
            "air_attribution": {"en": 4, "zh": 4},
            "false_google_attribution": {"en": 3, "zh": 3},
            "false_openai_attribution": {"en": 3, "zh": 3},
        },
        "router_record_cells": {
            "humor": {"en": 14, "zh": 13},
            "serious": {"en": 13, "zh": 13},
            "angry": {"en": 13, "zh": 14},
        },
        "review_depth": {
            "pass": 10,
            "fail": 10,
            "fault_types": {
                "format_schema": 2,
                "tool_argument": 2,
                "evidence_mismatch": 2,
                "grounding_mismatch": 2,
                "routing_scope": 2,
            },
        },
    }


def _teacher_target(row: dict[str, Any]) -> str:
    role = row["output_expert_role"]
    if row["selection_kind"] == "identity":
        template = consumer._IDENTITY_TEMPLATES[row["identity_class"]]
        if role == "review":
            return _canonical_bytes(
                {
                    "verdict": "fail",
                    "faults": ["missing_identity_answer"],
                    "correction": template,
                }
            ).decode()
        return template
    if role in {"humor", "serious", "angry"}:
        return f"Minimal training answer {row['candidate_id_sha256'][:12]}."
    if role == "tool":
        return _canonical_bytes(
            {
                "final_answer": "Grounded synthetic result.",
                "tool_call": {
                    "name": "local_catalog_search",
                    "arguments": {"query": row["candidate_id_sha256"][:8]},
                },
                "evidence_ids": ["synthetic_evidence"],
            }
        ).decode()
    if role == "review":
        verdict = (
            row["review_verdict"]
            if row["selection_kind"] == "tool_review_depth"
            else "pass"
        )
        fault = row["review_fault_type"]
        return _canonical_bytes(
            {
                "verdict": verdict,
                "faults": [fault] if verdict == "fail" else [],
                "correction": "Correct the audited fault."
                if verdict == "fail"
                else None,
            }
        ).decode()
    if role == "router":
        return _canonical_bytes(
            {
                "route": row["router_label"],
                "plan": ["hand off exactly once"],
                "stop": True,
            }
        ).decode()
    raise AssertionError(f"unsupported fixture role: {role}")


def _teacher_join_fixture(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    schema = json.loads(
        (
            consumer.ROOT / "configs/training/"
            "gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    teacher_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        target = _teacher_target(row)
        record = {
            "schema_version": consumer._TEACHER_RECORD_SCHEMA_VERSION,
            "namespace": consumer._TEACHER_RECORD_NAMESPACE,
            "record_id_sha256": row["source_record_id_sha256"],
            "source_content_sha256": row["source_content_sha256"],
            "source_serialization_identity_sha256": row[
                "source_serialization_identity_sha256"
            ],
            "task_bundle_sha256": row["task_bundle_sha256"],
            "source_asset": row["training_asset"],
            "decision": "accepted",
            "teacher_target": target,
            "teacher_target_sha256": _sha(target.encode("utf-8")),
        }
        teacher_rows.append(record)
        receipt_sha256 = _sha(f"receipt-{row['idempotency_key']}".encode())
        output_record_sha256 = _sha(f"output-{row['idempotency_key']}".encode())
        accepted_rows.append(
            {
                "schema_version": consumer._ACCEPTED_INVENTORY_SCHEMA_VERSION,
                "idempotency_key": row["idempotency_key"],
                "outcome": "accepted",
                "reason_code": "full_replay_quality_accepted",
                "receipt_sha256": receipt_sha256,
                "output_record_sha256": output_record_sha256,
                "stratum": {
                    "role": row["output_expert_role"],
                    "validation_contract_role": row["validation_contract_role"],
                    "language": row["language"],
                    "tool_family": row["tool_family"],
                    "identity_class": row["identity_class"],
                    "identity_parent_class": row["identity_parent_class"],
                    "selection_kind": row["selection_kind"],
                    "review_verdict": row["review_verdict"],
                    "review_fault_type": row["review_fault_type"],
                    "router_label": row["router_label"],
                    "source_record_id_sha256": row["source_record_id_sha256"],
                },
                "content_retained": False,
            }
        )
        training_rows.append(
            {
                "schema_version": consumer._TRAINING_RECORD_SCHEMA_VERSION,
                "training_ordinal": ordinal,
                "selected_ordinal": ordinal,
                "candidate_id_sha256": row["candidate_id_sha256"],
                "selected_candidate_sha256": _sha(row),
                "idempotency_key": row["idempotency_key"],
                "source_record_id_sha256": row["source_record_id_sha256"],
                "source_content_sha256": row["source_content_sha256"],
                "source_serialization_identity_sha256": row[
                    "source_serialization_identity_sha256"
                ],
                "task_bundle_sha256": row["task_bundle_sha256"],
                "source_asset": record["source_asset"],
                "decision": "accepted",
                "receipt_sha256": receipt_sha256,
                "output_record_sha256": output_record_sha256,
                "teacher_target_sha256": record["teacher_target_sha256"],
                "teacher_record_sha256": _sha(record),
                "content_retained": False,
                "raw_token_ids_retained": False,
            }
        )
    accepted_rows.sort(key=lambda row: row["idempotency_key"])
    return schema, teacher_rows, accepted_rows, training_rows


def _replace_teacher_target(
    teacher_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    index: int,
    target: str,
) -> None:
    teacher_rows[index]["teacher_target"] = target
    teacher_rows[index]["teacher_target_sha256"] = _sha(target.encode("utf-8"))
    training_rows[index]["teacher_target_sha256"] = teacher_rows[index][
        "teacher_target_sha256"
    ]
    training_rows[index]["teacher_record_sha256"] = _sha(teacher_rows[index])


def _artifact_entry(
    repo: Path,
    path: str,
    kind: str,
    stage: str,
) -> dict[str, Any]:
    raw = (repo / path).read_bytes()
    entry = {
        "path": path,
        "kind": kind,
        "git_stage": stage,
        "git_blob_oid": _blob(raw),
        "bytes": len(raw),
        "sha256": _sha(raw),
    }
    if kind == "teacher_shard":
        entry["records"] = raw.count(b"\n")
    return entry


def _build_producer_repo(
    root: Path,
    loaded: consumer.LoadedConfig,
    *,
    rows_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
    component_mismatch: bool = False,
    duplicate_artifact_path: bool = False,
    release_mutation: str = "none",
    stage_mutation: str = "none",
) -> tuple[Path, dict[str, Any]]:
    consumer_runtime_commit = "a" * 40
    consumer_runtime_tree = "b" * 40
    consumer_implementation_sha256 = "c" * 64
    consumer_schema_inventory_sha256 = "d" * 64
    repo = root / "producer"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Minimal320 Test")
    _git(repo, "config", "user.email", "minimal320@example.invalid")
    _git(repo, "config", "core.filemode", "true")

    component_paths = {
        "generation_implementation": "producer/generation.py",
        "selector_implementation": "producer/selector.py",
        "selector_contract": "producer/selector-contract.json",
        "selector_schema": "producer/selector.schema.json",
        "teacher_record_schema": "producer/teacher-record.schema.json",
        "source_truth_contract": "producer/source-truth-vnext.json",
    }
    for name, path in component_paths.items():
        if name != "source_truth_contract":
            _write(repo, path, f"{name}\n".encode())

    rows, fault_cells = _exact_rows()
    if rows_mutator is not None:
        rows_mutator(rows)
        rows.sort(key=lambda item: item["candidate_id_sha256"])
    strata = _strata()
    source_truth_body = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.minimal320-source-truth.vnext"
        ),
        "strata": strata,
        "review_fault_cells": fault_cells,
    }
    source_truth = {
        "contract_id": _sha(source_truth_body),
        **source_truth_body,
    }
    _write(repo, component_paths["source_truth_contract"], _json_raw(source_truth))

    _teacher_schema, teacher_rows, accepted_rows, training_rows = _teacher_join_fixture(
        rows
    )
    selected_inventory_raw = b"".join(_json_raw(row) for row in rows)
    accepted_inventory_raw = b"".join(_json_raw(row) for row in accepted_rows)
    training_inventory_raw = b"".join(_json_raw(row) for row in training_rows)
    teacher_shard_raw = b"".join(_json_raw(row) for row in teacher_rows)
    metadata_paths = {
        "selection_manifest": "producer/selection-manifest.json",
        "selected_candidate_inventory": "producer/selected.jsonl",
        "accepted_inventory": "producer/accepted.jsonl",
        "rejected_inventory": "producer/rejected.json",
        "quarantine_inventory": "producer/quarantine.json",
        "training_record_inventory": "producer/training-records.jsonl",
        "checkpoint_manifest": "producer/checkpoint-manifest.json",
        "freeze_attestation": "release/freeze-attestation.json",
        "external_attestation": "release/external-attestation.json",
    }
    for kind in (
        "selected_candidate_inventory",
        "accepted_inventory",
        "training_record_inventory",
    ):
        raw = (
            selected_inventory_raw
            if kind == "selected_candidate_inventory"
            else (
                accepted_inventory_raw
                if kind == "accepted_inventory"
                else training_inventory_raw
            )
        )
        _write(repo, metadata_paths[kind], raw)
    _write(repo, metadata_paths["rejected_inventory"], b"[]\n")
    _write(repo, metadata_paths["quarantine_inventory"], b"[]\n")
    teacher_path = "producer/teacher-shard-00000.jsonl"
    _write(
        repo,
        teacher_path,
        teacher_shard_raw,
    )

    component_pins = {
        f"{kind}_sha256": _sha((repo / path).read_bytes())
        for kind, path in component_paths.items()
    }
    if component_mismatch:
        component_pins["generation_implementation_sha256"] = "f" * 64
    selected_inventory_root = _sha(rows)
    accepted_inventory_root = _sha(accepted_rows)
    training_inventory_root = _sha(training_rows)
    selection = {
        "selector_identity_sha256": component_pins["selector_implementation_sha256"],
        "selected_set_root_sha256": selected_inventory_root,
        "counts": {
            "selected": 320,
            "by_role": {
                "angry": 40,
                "humor": 40,
                "review": 60,
                "router": 80,
                "serious": 40,
                "tool": 60,
            },
            "by_language": {"en": 160, "zh": 160},
        },
        "identity_training": {
            "records": 100,
            "bundles": 20,
            "validation_contract_role": "identity",
            "expert_role_counts": {
                "angry": 20,
                "humor": 20,
                "review": 20,
                "serious": 20,
                "tool": 20,
            },
            "included_in_role_counts": True,
            "separate_output_expert": False,
            "separate_training_arm": False,
        },
        "inventory_roots": {
            "selected_candidates_sha256": selected_inventory_root,
            "accepted_sha256": accepted_inventory_root,
            "training_records_sha256": training_inventory_root,
        },
        "strata": strata,
        "source_truth_contract_sha256": component_pins["source_truth_contract_sha256"],
        "review_fault_cells": fault_cells,
    }
    _write(repo, metadata_paths["selection_manifest"], _json_raw(selection))
    checkpoint = {
        "schema_version": ("anchor.gemma3-chat-unbalanced-v2.minimal320-checkpoint.v1"),
        "state": "minimal320_checkpoint_frozen",
        "records": 320,
        "selected_set_root_sha256": selected_inventory_root,
        "inventory_roots": selection["inventory_roots"],
    }
    _write(repo, metadata_paths["checkpoint_manifest"], _json_raw(checkpoint))

    candidate_kinds = {
        **component_paths,
        **{
            kind: path
            for kind, path in metadata_paths.items()
            if kind not in {"freeze_attestation", "external_attestation"}
        },
        "teacher_shard": teacher_path,
    }
    payload_inventory = sorted(
        (
            _artifact_entry(
                repo,
                path,
                kind,
                "candidate",
            )
            for kind, path in candidate_kinds.items()
        ),
        key=lambda item: item["path"],
    )
    payload_inventory_sha = _sha(payload_inventory)
    _git(repo, "add", "--", "producer")
    _git(repo, "commit", "--quiet", "-m", "candidate payload")
    candidate_commit = _git(repo, "rev-parse", "HEAD")
    candidate_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if release_mutation == "non_direct":
        _write(repo, "intermediate.txt", b"intermediate release lineage\n")
        _git(repo, "add", "--", "intermediate.txt")
        _git(repo, "commit", "--quiet", "-m", "intermediate release commit")

    selection_sha = _sha(selection)
    freeze = {
        "state": "frozen_pending_external_review",
        "selection_manifest_sha256": selection_sha,
        "selector_identity_sha256": selection["selector_identity_sha256"],
        "selected_set_root_sha256": selected_inventory_root,
        "payload_inventory_sha256": payload_inventory_sha,
        "checkpoint_manifest_sha256": _sha(
            (repo / metadata_paths["checkpoint_manifest"]).read_bytes()
        ),
        "counts": {
            "requested": 320,
            "completed": 320,
            "accepted": 320,
            "rejected": 0,
            "quarantine": 0,
        },
        "uncertain": 0,
    }
    freeze_raw = _json_raw(freeze)
    _write(repo, metadata_paths["freeze_attestation"], freeze_raw)
    freeze_sha = _sha(freeze)
    external = {
        "state": "independent_release_review_passed",
        "producer_commit": candidate_commit,
        "producer_tree": candidate_tree,
        "freeze_attestation_sha256": freeze_sha,
        "selection_manifest_sha256": selection_sha,
        "payload_inventory_sha256": payload_inventory_sha,
        "component_pins_sha256": _sha(component_pins),
        "independent_review": {
            "p0": 0,
            "p1": 0,
            "p2": 0,
            "generator_is_reviewer": False,
        },
    }
    external_raw = _json_raw(external)
    _write(repo, metadata_paths["external_attestation"], external_raw)
    artifact_inventory = sorted(
        [
            *payload_inventory,
            _artifact_entry(
                repo,
                metadata_paths["freeze_attestation"],
                "freeze_attestation",
                "release",
            ),
            _artifact_entry(
                repo,
                metadata_paths["external_attestation"],
                "external_attestation",
                "release",
            ),
        ],
        key=lambda item: item["path"],
    )
    if duplicate_artifact_path:
        artifact_inventory.append(deepcopy(artifact_inventory[-1]))
        artifact_inventory.sort(key=lambda item: item["path"])
    if stage_mutation != "none":
        if stage_mutation == "candidate_as_release":
            target_kind = "generation_implementation"
            replacement = "release"
        elif stage_mutation == "release_as_candidate":
            target_kind = "freeze_attestation"
            replacement = "candidate"
        elif stage_mutation == "arbitrary":
            target_kind = "generation_implementation"
            replacement = "untrusted"
        else:
            raise AssertionError(f"unsupported stage mutation: {stage_mutation}")
        target_entry = next(
            entry for entry in artifact_inventory if entry["kind"] == target_kind
        )
        target_entry["git_stage"] = replacement
    binding_body = {
        "schema_version": consumer.PRODUCER_BINDING_SCHEMA_VERSION,
        "state": "released_for_strict_consumer_preflight",
        "producer_commit": candidate_commit,
        "producer_tree": candidate_tree,
        "component_pins": component_pins,
        "component_pins_sha256": _sha(component_pins),
        "artifact_bindings": {
            "components": component_paths,
            "metadata": metadata_paths,
            "teacher_shards": [teacher_path],
        },
        "artifact_inventory": artifact_inventory,
        "artifact_inventory_sha256": _sha(artifact_inventory),
        "selection_manifest": selection,
        "selection_manifest_sha256": selection_sha,
        "freeze_attestation": freeze,
        "freeze_attestation_sha256": freeze_sha,
        "external_attestation": external,
        "external_attestation_sha256": _sha(external),
        "claims": {
            "candidate": False,
            "final": True,
            "diagnostic_training_authorized": True,
            "formal_training_authorized": False,
            "live_model_release_authorized": False,
        },
        "content_safety": {
            "content_retained_in_binding": False,
            "protected_body_read_required": False,
            "raw_token_ids_retained": False,
            "credential_retained": False,
        },
    }
    binding = {"binding_id": _sha(binding_body), **binding_body}
    binding_path = "release/producer-final-binding.json"
    binding_raw = _json_raw(binding)
    _write(repo, binding_path, binding_raw)
    release_pin = {
        "schema_version": consumer.RELEASE_PIN_SCHEMA_VERSION,
        "state": "producer_final_binding_frozen",
        "producer_binding_schema_version": consumer.PRODUCER_BINDING_SCHEMA_VERSION,
        "producer_binding_schema_sha256": loaded.value["schemas"][
            "producer_final_binding"
        ]["sha256"],
        "producer_binding_path": binding_path,
        "producer_binding_sha256": _sha(binding_raw),
        "producer_binding_blob_oid": _blob(binding_raw),
        "producer_binding_id": binding["binding_id"],
        "candidate_commit": candidate_commit,
        "candidate_tree": candidate_tree,
        "freeze_attestation_path": metadata_paths["freeze_attestation"],
        "freeze_attestation_sha256": _sha(freeze_raw),
        "freeze_attestation_blob_oid": _blob(freeze_raw),
        "external_attestation_path": metadata_paths["external_attestation"],
        "external_attestation_sha256": _sha(external_raw),
        "external_attestation_blob_oid": _blob(external_raw),
        "consumer_runtime_commit": consumer_runtime_commit,
        "consumer_runtime_tree": consumer_runtime_tree,
        "consumer_implementation_sha256": consumer_implementation_sha256,
        "consumer_schema_inventory_sha256": consumer_schema_inventory_sha256,
        "reviewed_by_different_agent": True,
        "claims": {
            "candidate": False,
            "final": True,
            "release_authorized": True,
            "formal_training_authorized": False,
        },
    }
    release_pin_path = "release/producer-release-pin.json"
    release_pin_raw = _json_raw(release_pin)
    _write(repo, release_pin_path, release_pin_raw)
    _git(repo, "add", "--", "release")
    if release_mutation == "extra":
        _write(repo, "release/extra.txt", b"not in the release allowlist\n")
        _git(repo, "add", "--", "release/extra.txt")
    elif release_mutation == "rename":
        _git(
            repo,
            "mv",
            component_paths["generation_implementation"],
            "producer/generation-renamed.py",
        )
    elif release_mutation == "mode":
        _git(
            repo,
            "update-index",
            "--chmod=+x",
            component_paths["generation_implementation"],
        )
    elif release_mutation == "submodule":
        _git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{candidate_commit},{component_paths['generation_implementation']}",
        )
    elif release_mutation not in {"none", "non_direct", "merge", "untracked"}:
        raise AssertionError(f"unsupported release mutation: {release_mutation}")
    _git(repo, "commit", "--quiet", "-m", "release attestation")
    release_commit = _git(repo, "rev-parse", "HEAD")
    if release_mutation == "merge":
        direct_release_commit = release_commit
        _git(repo, "checkout", "--quiet", "-b", "release-side", candidate_commit)
        _write(repo, "release-side.txt", b"merge parent is not allowed\n")
        _git(repo, "add", "--", "release-side.txt")
        _git(repo, "commit", "--quiet", "-m", "release side")
        _git(
            repo,
            "checkout",
            "--quiet",
            "-b",
            "release-merged",
            direct_release_commit,
        )
        _git(repo, "merge", "--quiet", "--no-ff", "release-side", "-m", "merge")
        release_commit = _git(repo, "rev-parse", "HEAD")
    release_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if release_mutation == "untracked":
        _write(repo, "release-untracked.txt", b"global status must reject this\n")
    anchor = {
        "candidate_commit": candidate_commit,
        "candidate_tree": candidate_tree,
        "release_commit": release_commit,
        "release_tree": release_tree,
        "release_pin_path": release_pin_path,
        "release_pin_sha256": _sha(release_pin_raw),
        "release_pin_blob_oid": _blob(release_pin_raw),
        "producer_binding_path": binding_path,
        "producer_binding_sha256": _sha(binding_raw),
        "producer_binding_blob_oid": _blob(binding_raw),
        "producer_binding_id": binding["binding_id"],
        "freeze_attestation_path": metadata_paths["freeze_attestation"],
        "freeze_attestation_sha256": _sha(freeze_raw),
        "freeze_attestation_blob_oid": _blob(freeze_raw),
        "external_attestation_path": metadata_paths["external_attestation"],
        "external_attestation_sha256": _sha(external_raw),
        "external_attestation_blob_oid": _blob(external_raw),
        "consumer_runtime_commit": consumer_runtime_commit,
        "consumer_runtime_tree": consumer_runtime_tree,
        "consumer_implementation_sha256": consumer_implementation_sha256,
        "consumer_schema_inventory_sha256": consumer_schema_inventory_sha256,
    }
    return repo, anchor


def _build_producer_repo_vnext(
    root: Path,
    loaded: consumer.LoadedConfig,
    *,
    rows_mutator: Callable[[list[dict[str, Any]]], None] | None = None,
    component_mismatch: bool = False,
    duplicate_artifact_path: bool = False,
    release_mutation: str = "none",
    stage_mutation: str = "none",
    teacher_rows_mutator: (
        Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None
    ) = None,
    teacher_schema_mutator: Callable[[dict[str, Any]], None] | None = None,
    teacher_rows_from_unmutated: bool = False,
    committed_manifest_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    consumer_runtime_commit = "a" * 40
    consumer_runtime_tree = "b" * 40
    consumer_implementation_sha256 = "c" * 64
    consumer_schema_inventory_sha256 = "d" * 64
    repo = root / "producer"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Minimal320 Test")
    _git(repo, "config", "user.email", "minimal320@example.invalid")
    _git(repo, "config", "core.filemode", "true")
    _git(repo, "config", "core.autocrlf", "false")

    teacher_record_schema_path = (
        "configs/data/gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json"
    )
    teacher_record_schema = json.loads(
        (
            consumer.ROOT
            / "configs/training/gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    if teacher_schema_mutator is not None:
        teacher_schema_mutator(teacher_record_schema)
    teacher_record_schema_raw = _json_raw(teacher_record_schema)
    _write(repo, ".gitattributes", b"* text=auto\n")
    _write(repo, teacher_record_schema_path, teacher_record_schema_raw)
    _git(repo, "add", "--", ".gitattributes", teacher_record_schema_path)
    _git(repo, "commit", "--quiet", "-m", "audited base B")
    base_commit = _git(repo, "rev-parse", "HEAD")
    base_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    component_paths = {
        "generation_implementation": (
            "src/anchor_mvp/data/"
            "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py"
        ),
        "selector_implementation": (
            "src/anchor_mvp/data/"
            "gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2.py"
        ),
        "selector_contract": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_contract_v2.json"
        ),
        "selector_schema": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_selector_v2.schema.json"
        ),
        "teacher_record_schema": teacher_record_schema_path,
        "source_truth_contract": (
            "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_contract_v1.json"
        ),
    }
    rows, fault_cells = _exact_rows()
    unmutated_rows = deepcopy(rows)
    if rows_mutator is not None:
        rows_mutator(rows)
        rows.sort(key=lambda item: item["candidate_id_sha256"])
    strata = _strata()
    source_truth_body = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.minimal320-source-truth.vnext"
        ),
        "strata": strata,
        "review_fault_cells": fault_cells,
    }
    source_truth = {"contract_id": _sha(source_truth_body), **source_truth_body}
    for path in consumer._PRODUCER_L_PATHS:
        if path == ".gitattributes":
            raw = b"* text=auto\n*.json text eol=lf\n*.jsonl text eol=lf\n"
        elif path == component_paths["source_truth_contract"]:
            raw = _json_raw(source_truth)
        else:
            raw = f"launch:{path}\n".encode()
        _write(repo, path, raw)
    _git(repo, "add", "--", *consumer._PRODUCER_L_PATHS)
    _git(repo, "commit", "--quiet", "-m", "producer launch L21")
    launch_commit = _git(repo, "rev-parse", "HEAD")
    launch_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    teacher_source_rows = unmutated_rows if teacher_rows_from_unmutated else rows
    _teacher_schema, teacher_rows, accepted_rows, training_rows = _teacher_join_fixture(
        teacher_source_rows
    )
    if teacher_rows_mutator is not None:
        teacher_rows_mutator(teacher_rows, training_rows)
    selected_inventory_raw = b"".join(_json_raw(row) for row in rows)
    accepted_inventory_raw = b"".join(_json_raw(row) for row in accepted_rows)
    training_inventory_raw = b"".join(_json_raw(row) for row in training_rows)
    teacher_shard_raw = b"".join(_json_raw(row) for row in teacher_rows)
    metadata_paths = dict(consumer._EXACT_METADATA_PATHS)
    teacher_path = consumer._TEACHER_SHARD_PATH
    component_pins = {
        f"{kind}_sha256": _sha((repo / path).read_bytes())
        for kind, path in component_paths.items()
    }
    if component_mismatch:
        component_pins["generation_implementation_sha256"] = "f" * 64
    selected_inventory_root = _sha(rows)
    accepted_inventory_root = _sha(accepted_rows)
    training_inventory_root = _sha(training_rows)
    selection = {
        "selector_identity_sha256": component_pins["selector_implementation_sha256"],
        "selected_set_root_sha256": selected_inventory_root,
        "counts": {
            "selected": 320,
            "by_role": {
                "angry": 40,
                "humor": 40,
                "review": 60,
                "router": 80,
                "serious": 40,
                "tool": 60,
            },
            "by_language": {"en": 160, "zh": 160},
        },
        "identity_training": {
            "records": 100,
            "bundles": 20,
            "validation_contract_role": "identity",
            "expert_role_counts": {
                "angry": 20,
                "humor": 20,
                "review": 20,
                "serious": 20,
                "tool": 20,
            },
            "included_in_role_counts": True,
            "separate_output_expert": False,
            "separate_training_arm": False,
        },
        "inventory_roots": {
            "selected_candidates_sha256": selected_inventory_root,
            "accepted_sha256": accepted_inventory_root,
            "training_records_sha256": training_inventory_root,
        },
        "strata": strata,
        "source_truth_contract_sha256": component_pins["source_truth_contract_sha256"],
        "review_fault_cells": fault_cells,
    }
    checkpoint = {
        "schema_version": "anchor.gemma3-chat-unbalanced-v2.minimal320-checkpoint.v1",
        "state": "minimal320_checkpoint_frozen",
        "records": 320,
        "selected_set_root_sha256": selected_inventory_root,
        "inventory_roots": selection["inventory_roots"],
    }
    teacher_shard_sidecar_raw = (
        f"{_sha(teacher_shard_raw)}  teacher_training_shard-00000.jsonl\n"
    ).encode()
    training_inventory_sidecar_raw = (
        f"{_sha(training_inventory_raw)}  training_record_inventory.jsonl\n"
    ).encode()
    committed_files = [
        {
            "path": consumer._TEACHER_SHARD_PATH,
            "sha256": _sha(teacher_shard_raw),
            "bytes": len(teacher_shard_raw),
            "kind": "teacher_training_shard",
        },
        {
            "path": f"{consumer._TEACHER_SHARD_PATH}.sha256",
            "sha256": _sha(teacher_shard_sidecar_raw),
            "bytes": len(teacher_shard_sidecar_raw),
            "kind": "sha256_sidecar",
        },
        {
            "path": consumer._TRAINING_RECORD_INVENTORY_PATH,
            "sha256": _sha(training_inventory_raw),
            "bytes": len(training_inventory_raw),
            "kind": "training_record_inventory",
        },
        {
            "path": f"{consumer._TRAINING_RECORD_INVENTORY_PATH}.sha256",
            "sha256": _sha(training_inventory_sidecar_raw),
            "bytes": len(training_inventory_sidecar_raw),
            "kind": "sha256_sidecar",
        },
    ]
    committed_shards = {
        "schema_version": consumer._COMMITTED_SHARDS_SCHEMA_VERSION,
        "release_root": consumer._PRODUCER_RELEASE_ROOT,
        "shard_count": 1,
        "files": committed_files,
        "file_count": 4,
        "file_inventory_root_sha256": consumer._merkle_root(committed_files),
        "teacher_training_shard": {
            "path": consumer._TEACHER_SHARD_PATH,
            "sidecar_path": f"{consumer._TEACHER_SHARD_PATH}.sha256",
            "schema_version": consumer._TEACHER_RECORD_SCHEMA_VERSION,
            "schema_sha256": component_pins["teacher_record_schema_sha256"],
            "row_count": len(teacher_rows),
            "bytes": len(teacher_shard_raw),
            "max_bytes_exclusive": consumer._MAX_PRODUCER_PAYLOAD_BYTES_EXCLUSIVE,
            "sha256": _sha(teacher_shard_raw),
            "record_inventory_root_sha256": consumer._merkle_root(teacher_rows),
        },
        "training_record_inventory": {
            "path": consumer._TRAINING_RECORD_INVENTORY_PATH,
            "sidecar_path": f"{consumer._TRAINING_RECORD_INVENTORY_PATH}.sha256",
            "schema_version": consumer._TRAINING_RECORD_SCHEMA_VERSION,
            "row_count": len(training_rows),
            "bytes": len(training_inventory_raw),
            "sha256": _sha(training_inventory_raw),
            "inventory_root_sha256": consumer._merkle_root(training_rows),
        },
        "counts": {
            "requested": 320,
            "completed": 320,
            "accepted": 320,
            "rejected": 0,
            "quarantine": 0,
        },
        "partial_accepted_subset": False,
        "exact_requested_set_complete": True,
        "accepted_selected_order_root_sha256": _sha(
            [row["candidate_id_sha256"] for row in training_rows]
        ),
        "teacher_target_order_root_sha256": _sha(
            [row["teacher_target_sha256"] for row in training_rows]
        ),
        "zip_alignment": {
            "row_count": 320,
            "training_ordinals_contiguous": True,
            "selected_order_filtered_accepted": True,
            "teacher_record_sha256_bound": True,
        },
        "accepted_inventory_sha256": _sha(accepted_inventory_raw),
        "selected_set_root_sha256": selected_inventory_root,
        "accepted_strata_root_sha256": _sha([row["stratum"] for row in accepted_rows]),
        "final_chain_tip_sha256": "e" * 64,
        "final_wal_entries": 320,
        "content_retained": False,
        "raw_token_ids_retained": False,
    }
    if committed_manifest_mutator is not None:
        committed_manifest_mutator(committed_shards)
    primary_raw: dict[str, bytes] = {
        "accepted_inventory.jsonl": accepted_inventory_raw,
        "authenticated_metadata_overlay.jsonl": selected_inventory_raw,
        "authenticated_metadata_overlay_manifest.json": b'{"count":320}\n',
        "candidate_inventory.jsonl": selected_inventory_raw,
        "checkpoint_manifest.json": _json_raw(checkpoint),
        "checkpoints_inventory.json": b'{"count":1}\n',
        "committed_shards_manifest.json": _json_raw(committed_shards),
        "producer_launch_inventory.json": _json_raw(
            {"commit": launch_commit, "tree": launch_tree}
        ),
        "quarantine_inventory.jsonl": b"",
        "rejected_inventory.jsonl": b"",
        "selected_inventory.jsonl": selected_inventory_raw,
        "selection_manifest.json": _json_raw(selection),
        "selection_preimage.json": b'{"selected":320}\n',
        "selector_preimage_manifest.json": b'{"selected":320}\n',
        "teacher_training_shard-00000.jsonl": teacher_shard_raw,
        "training_record_inventory.jsonl": training_inventory_raw,
    }
    for name, raw in primary_raw.items():
        _write(repo, f"{consumer._PRODUCER_RELEASE_ROOT}/{name}", raw)
        _write(
            repo,
            f"{consumer._PRODUCER_RELEASE_ROOT}/{name}.sha256",
            f"{_sha(raw)}  {name}\n".encode(),
        )
    if tuple(sorted(path for path in consumer._PRODUCER_P_PATHS)) != (
        consumer._PRODUCER_P_PATHS
    ):
        raise AssertionError("producer P paths must stay sorted")
    _git(repo, "add", "--", *consumer._PRODUCER_P_PATHS)
    _git(repo, "commit", "--quiet", "-m", "producer payload P32")
    payload_commit = _git(repo, "rev-parse", "HEAD")
    payload_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if release_mutation == "non_direct":
        _write(repo, "intermediate.txt", b"intermediate release lineage\n")
        _git(repo, "add", "--", "intermediate.txt")
        _git(repo, "commit", "--quiet", "-m", "intermediate release commit")

    candidate_kinds = {
        **component_paths,
        **{
            kind: path
            for kind, path in metadata_paths.items()
            if kind not in {"freeze_attestation", "external_attestation"}
        },
        "teacher_shard": teacher_path,
    }
    payload_inventory = sorted(
        (
            _artifact_entry(
                repo,
                path,
                kind,
                (
                    "producer_launch_L"
                    if kind in component_paths
                    else "producer_payload_P"
                ),
            )
            for kind, path in candidate_kinds.items()
        ),
        key=lambda item: item["path"],
    )
    payload_inventory_sha = _sha(payload_inventory)
    selection_sha = _sha(selection)
    freeze = {
        "state": "frozen_pending_external_review",
        "selection_manifest_sha256": selection_sha,
        "selector_identity_sha256": selection["selector_identity_sha256"],
        "selected_set_root_sha256": selected_inventory_root,
        "payload_inventory_sha256": payload_inventory_sha,
        "checkpoint_manifest_sha256": _sha(
            (repo / metadata_paths["checkpoint_manifest"]).read_bytes()
        ),
        "counts": {
            "requested": 320,
            "completed": 320,
            "accepted": 320,
            "rejected": 0,
            "quarantine": 0,
        },
        "uncertain": 0,
    }
    freeze_raw = _json_raw(freeze)
    external = {
        "state": "independent_release_review_passed",
        "producer_commit": payload_commit,
        "producer_tree": payload_tree,
        "freeze_attestation_sha256": _sha(freeze),
        "selection_manifest_sha256": selection_sha,
        "payload_inventory_sha256": payload_inventory_sha,
        "component_pins_sha256": _sha(component_pins),
        "independent_review": {
            "p0": 0,
            "p1": 0,
            "p2": 0,
            "generator_is_reviewer": False,
        },
    }
    external_raw = _json_raw(external)
    for name, raw in (
        ("external_attestation.json", external_raw),
        ("freeze_attestation.json", freeze_raw),
    ):
        _write(repo, f"{consumer._PRODUCER_RELEASE_ROOT}/{name}", raw)
        _write(
            repo,
            f"{consumer._PRODUCER_RELEASE_ROOT}/{name}.sha256",
            f"{_sha(raw)}  {name}\n".encode(),
        )
    if release_mutation == "extra":
        _write(
            repo,
            f"{consumer._PRODUCER_RELEASE_ROOT}/extra.txt",
            b"not allowed\n",
        )
    elif release_mutation == "rename":
        _git(
            repo,
            "mv",
            component_paths["generation_implementation"],
            "generation-renamed.py",
        )
    elif release_mutation in {"mode", "submodule"}:
        pass
    elif release_mutation not in {"none", "non_direct", "merge", "untracked"}:
        raise AssertionError(f"unsupported release mutation: {release_mutation}")
    _git(repo, "add", "-A")
    if release_mutation == "mode":
        _git(
            repo,
            "update-index",
            "--chmod=+x",
            component_paths["generation_implementation"],
        )
    elif release_mutation == "submodule":
        _git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{payload_commit},{component_paths['generation_implementation']}",
        )
    _git(repo, "commit", "--quiet", "-m", "producer final R4")
    final_commit = _git(repo, "rev-parse", "HEAD")
    if release_mutation == "merge":
        direct_final = final_commit
        _git(repo, "checkout", "--quiet", "-b", "release-side", payload_commit)
        _write(repo, "release-side.txt", b"merge parent is not allowed\n")
        _git(repo, "add", "--", "release-side.txt")
        _git(repo, "commit", "--quiet", "-m", "release side")
        _git(repo, "checkout", "--quiet", "-b", "release-merged", direct_final)
        _git(repo, "merge", "--quiet", "--no-ff", "release-side", "-m", "merge")
        final_commit = _git(repo, "rev-parse", "HEAD")
    final_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if release_mutation == "untracked":
        _write(repo, "release-untracked.txt", b"global status must reject this\n")

    stages = {
        "audited_base": {"commit": base_commit, "tree": base_tree},
        "producer_launch_L": {
            "commit": launch_commit,
            "tree": launch_tree,
            "direct_parent": base_commit,
            "path_count": 21,
            "path_set_root_sha256": consumer._PRODUCER_L_PATH_ROOT,
        },
        "producer_payload_P": {
            "commit": payload_commit,
            "tree": payload_tree,
            "direct_parent": launch_commit,
            "path_count": 32,
            "path_set_root_sha256": consumer._PRODUCER_P_PATH_ROOT,
        },
        "producer_final_R": {
            "commit": final_commit,
            "tree": final_tree,
            "direct_parent": payload_commit,
            "path_count": 4,
            "path_set_root_sha256": consumer._PRODUCER_R_PATH_ROOT,
        },
        "global_clean_tree": True,
        "unique_direct_children": True,
        "exact_raw_diff_allowlist": True,
        "release_self_pin_present": False,
    }
    artifact_inventory = sorted(
        [
            *payload_inventory,
            _artifact_entry(
                repo,
                metadata_paths["freeze_attestation"],
                "freeze_attestation",
                "producer_final_R",
            ),
            _artifact_entry(
                repo,
                metadata_paths["external_attestation"],
                "external_attestation",
                "producer_final_R",
            ),
        ],
        key=lambda item: item["path"],
    )
    if duplicate_artifact_path:
        artifact_inventory.append(deepcopy(artifact_inventory[-1]))
        artifact_inventory.sort(key=lambda item: item["path"])
    if stage_mutation != "none":
        if stage_mutation == "candidate_as_release":
            target_kind, replacement = "generation_implementation", "producer_final_R"
        elif stage_mutation == "release_as_candidate":
            target_kind, replacement = "freeze_attestation", "producer_payload_P"
        elif stage_mutation == "arbitrary":
            target_kind, replacement = "generation_implementation", "untrusted"
        else:
            raise AssertionError(f"unsupported stage mutation: {stage_mutation}")
        next(entry for entry in artifact_inventory if entry["kind"] == target_kind)[
            "git_stage"
        ] = replacement
    binding_body = {
        "schema_version": consumer.PRODUCER_BINDING_SCHEMA_VERSION,
        "state": "released_for_strict_consumer_preflight",
        "producer_commit": payload_commit,
        "producer_tree": payload_tree,
        "producer_stages": stages,
        "component_pins": component_pins,
        "component_pins_sha256": _sha(component_pins),
        "artifact_bindings": {
            "components": component_paths,
            "metadata": metadata_paths,
            "teacher_shards": [teacher_path],
        },
        "artifact_inventory": artifact_inventory,
        "artifact_inventory_sha256": _sha(artifact_inventory),
        "selection_manifest": selection,
        "selection_manifest_sha256": selection_sha,
        "freeze_attestation": freeze,
        "freeze_attestation_sha256": _sha(freeze),
        "external_attestation": external,
        "external_attestation_sha256": _sha(external),
        "claims": {
            "producer_payload_P_is_final": False,
            "producer_final_R_authenticated": True,
            "diagnostic_training_authorized": True,
            "formal_training_authorized": False,
            "live_model_release_authorized": False,
        },
        "content_safety": {
            "content_retained_in_binding": False,
            "protected_body_read_required": False,
            "raw_token_ids_retained": False,
            "credential_retained": False,
        },
    }
    binding = {"binding_id": _sha(binding_body), **binding_body}
    anchor = {
        "schema_version": consumer.RELEASE_PIN_SCHEMA_VERSION,
        "state": "producer_final_R_frozen",
        "producer_stages": stages,
        "producer_binding_schema_version": consumer.PRODUCER_BINDING_SCHEMA_VERSION,
        "producer_binding_schema_sha256": loaded.value["schemas"][
            "producer_final_binding"
        ]["sha256"],
        "producer_binding_canonical_json": _canonical_bytes(binding).decode(),
        "producer_binding_sha256": _sha(_canonical_bytes(binding)),
        "producer_binding_id": binding["binding_id"],
        "consumer_runtime_commit": consumer_runtime_commit,
        "consumer_runtime_tree": consumer_runtime_tree,
        "consumer_implementation_sha256": consumer_implementation_sha256,
        "consumer_schema_inventory_sha256": consumer_schema_inventory_sha256,
        "reviewed_by_different_agent": True,
        "claims": {
            "consumer_code_P_only": True,
            "consumer_pin_C_required": True,
            "producer_final_R_authenticated": True,
            "release_authorized": True,
            "formal_training_authorized": False,
        },
    }
    return repo, anchor


_build_producer_repo = _build_producer_repo_vnext


def _mutate_embedded_binding(
    anchor: dict[str, Any],
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    binding = json.loads(anchor["producer_binding_canonical_json"])
    mutator(binding)
    binding_body = dict(binding)
    binding_body.pop("binding_id")
    binding["binding_id"] = _sha(binding_body)
    binding_raw = _canonical_bytes(binding)
    anchor["producer_binding_canonical_json"] = binding_raw.decode()
    anchor["producer_binding_sha256"] = _sha(binding_raw)
    anchor["producer_binding_id"] = binding["binding_id"]


def _frozen_loaded(
    loaded: consumer.LoadedConfig,
    anchor: dict[str, Any],
) -> consumer.LoadedConfig:
    value = deepcopy(loaded.value)
    value["state"] = "producer_final_release_frozen"
    value["producer_release_stage"] = {
        "state": "producer_final_release_frozen",
        "trusted_release": anchor,
        "placeholder_values_accepted": False,
        "candidate_binding_accepted": False,
    }
    return consumer.LoadedConfig(value=value, raw_sha256=loaded.raw_sha256)


def _consumer_git() -> consumer.ConsumerGitIdentity:
    return consumer.ConsumerGitIdentity(
        head="e" * 40,
        tree="f" * 40,
        runtime_commit="a" * 40,
        runtime_tree="b" * 40,
        implementation_sha256="c" * 64,
        schema_inventory_sha256="d" * 64,
    )


def _build_consumer_config_commit(
    root: Path,
    source_root: Path,
    loaded_template: consumer.LoadedConfig,
    *,
    mutation: str = "none",
) -> tuple[Path, dict[str, Any], consumer.LoadedConfig]:
    repo = root / f"consumer-{mutation.replace('/', '-')}"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Minimal320 Consumer Test")
    _git(repo, "config", "user.email", "consumer@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "core.filemode", "true")
    paths = [consumer._CONSUMER_CONFIG_PATH, *consumer._CONSUMER_RUNTIME_PATHS]
    for path_text in paths:
        _write(repo, path_text, (source_root / path_text).read_bytes())
    _git(repo, "add", "--", *paths)
    _git(repo, "commit", "--quiet", "-m", "consumer runtime P")
    runtime_commit = _git(repo, "rev-parse", "HEAD")
    runtime_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    implementation_raw = (
        repo / "src/anchor_mvp/training/"
        "gemma3_chat_unbalanced_v2_minimal320_consumer_v1.py"
    ).read_bytes()
    schema_identities = {
        name: _sha((repo / path_text).read_bytes())
        for name, path_text in consumer._CONSUMER_SCHEMA_PATHS.items()
    }
    producer_stages = {
        "audited_base": {"commit": "1" * 40, "tree": "2" * 40},
        "producer_launch_L": {
            "commit": "3" * 40,
            "tree": "4" * 40,
            "direct_parent": "1" * 40,
            "path_count": 21,
            "path_set_root_sha256": consumer._PRODUCER_L_PATH_ROOT,
        },
        "producer_payload_P": {
            "commit": "5" * 40,
            "tree": "6" * 40,
            "direct_parent": "3" * 40,
            "path_count": 32,
            "path_set_root_sha256": consumer._PRODUCER_P_PATH_ROOT,
        },
        "producer_final_R": {
            "commit": "7" * 40,
            "tree": "8" * 40,
            "direct_parent": "5" * 40,
            "path_count": 4,
            "path_set_root_sha256": consumer._PRODUCER_R_PATH_ROOT,
        },
        "global_clean_tree": True,
        "unique_direct_children": True,
        "exact_raw_diff_allowlist": True,
        "release_self_pin_present": False,
    }
    anchor = {
        "schema_version": consumer.RELEASE_PIN_SCHEMA_VERSION,
        "state": "producer_final_R_frozen",
        "producer_stages": producer_stages,
        "producer_binding_schema_version": consumer.PRODUCER_BINDING_SCHEMA_VERSION,
        "producer_binding_schema_sha256": "9" * 64,
        "producer_binding_canonical_json": ('{"binding_id":"' + ("a" * 64) + '"}'),
        "producer_binding_sha256": "b" * 64,
        "producer_binding_id": "a" * 64,
        "consumer_runtime_commit": runtime_commit,
        "consumer_runtime_tree": runtime_tree,
        "consumer_implementation_sha256": _sha(implementation_raw),
        "consumer_schema_inventory_sha256": _sha(schema_identities),
        "reviewed_by_different_agent": True,
        "claims": {
            "consumer_code_P_only": True,
            "consumer_pin_C_required": True,
            "producer_final_R_authenticated": True,
            "release_authorized": True,
            "formal_training_authorized": False,
        },
    }

    if mutation == "non_direct":
        _write(repo, "intermediate.txt", b"not allowed\n")
        _git(repo, "add", "--", "intermediate.txt")
        _git(repo, "commit", "--quiet", "-m", "intermediate")

    config = deepcopy(loaded_template.value)
    config["state"] = "producer_final_release_frozen"
    config["producer_release_stage"] = {
        "state": "producer_final_release_frozen",
        "trusted_release": anchor,
        "placeholder_values_accepted": False,
        "candidate_binding_accepted": False,
    }
    config_raw = _json_raw(config)
    _write(repo, consumer._CONSUMER_CONFIG_PATH, config_raw)

    if mutation in {"sitecustomize.py", "src/anchor_mvp/__init__.py"}:
        _write(repo, mutation, b"raise RuntimeError('not allowed')\n")
    elif mutation == "rename":
        _git(
            repo,
            "mv",
            "tests/test_gemma3_chat_unbalanced_v2_minimal320_consumer_v1.py",
            "tests/test_gemma3_chat_unbalanced_v2_minimal320_consumer_v1.renamed",
        )
    _git(repo, "add", "-A")
    if mutation == "mode":
        _git(
            repo,
            "update-index",
            "--chmod=+x",
            consumer._CONSUMER_CONFIG_PATH,
        )
    _git(repo, "commit", "--quiet", "-m", "consumer config C")

    if mutation == "merge":
        config_commit = _git(repo, "rev-parse", "HEAD")
        _git(repo, "checkout", "--quiet", "-b", "side", runtime_commit)
        _write(repo, "side.txt", b"merge is not allowed\n")
        _git(repo, "add", "--", "side.txt")
        _git(repo, "commit", "--quiet", "-m", "side")
        _git(repo, "checkout", "--quiet", "-b", "merged", config_commit)
        _git(repo, "merge", "--quiet", "--no-ff", "side", "-m", "merge")

    loaded = consumer.LoadedConfig(
        value=config,
        raw_sha256=_sha(config_raw),
    )
    return repo, anchor, loaded


def _install_frozen_config(
    monkeypatch: pytest.MonkeyPatch,
    frozen: consumer.LoadedConfig,
) -> None:
    monkeypatch.setattr(consumer, "load_config", lambda: frozen)
    monkeypatch.setattr(
        consumer,
        "_consumer_git_identity",
        lambda *_arguments: _consumer_git(),
    )


def _authenticate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **fixture_options: Any,
) -> tuple[consumer.LoadedConfig, consumer.AuthenticatedRelease, Path]:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        **fixture_options,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    release = consumer.authenticate_release(producer_repo=repo)
    return frozen, release, repo


def _build_b_l_p_r_repo(
    root: Path,
    *,
    extra_payload_path: bool = False,
) -> tuple[Path, dict[str, Any]]:
    repo = root / "producer-stage-chain"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Minimal320 Stage Test")
    _git(repo, "config", "user.email", "stage@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    _write(repo, ".gitattributes", b"* text=auto\n")
    _git(repo, "add", "--", ".gitattributes")
    _git(repo, "commit", "--quiet", "-m", "audited base B")
    base_commit = _git(repo, "rev-parse", "HEAD")
    base_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    for path in consumer._PRODUCER_L_PATHS:
        raw = (
            b"* text=auto\n*.json text eol=lf\n"
            if path == ".gitattributes"
            else (f"launch:{path}\n".encode())
        )
        _write(repo, path, raw)
    _git(repo, "add", "--", *consumer._PRODUCER_L_PATHS)
    _git(repo, "commit", "--quiet", "-m", "producer launch L21")
    launch_commit = _git(repo, "rev-parse", "HEAD")
    launch_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    for primary_name in (
        name for name in consumer._PRODUCER_P_NAMES if not name.endswith(".sha256")
    ):
        primary_path = f"{consumer._PRODUCER_RELEASE_ROOT}/{primary_name}"
        raw = (
            b""
            if primary_name
            in {"rejected_inventory.jsonl", "quarantine_inventory.jsonl"}
            else f"payload:{primary_path}\n".encode()
        )
        _write(repo, primary_path, raw)
        _write(
            repo,
            f"{primary_path}.sha256",
            f"{_sha(raw)}  {primary_name}\n".encode(),
        )
    if extra_payload_path:
        _write(
            repo,
            f"{consumer._PRODUCER_RELEASE_ROOT}/unexpected.json",
            b"unexpected\n",
        )
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "producer payload P32")
    payload_commit = _git(repo, "rev-parse", "HEAD")
    payload_tree = _git(repo, "rev-parse", "HEAD^{tree}")

    for path in consumer._PRODUCER_R_PATHS:
        _write(repo, path, f"final:{path}\n".encode())
    _git(repo, "add", "--", *consumer._PRODUCER_R_PATHS)
    _git(repo, "commit", "--quiet", "-m", "producer final R4")
    final_commit = _git(repo, "rev-parse", "HEAD")
    final_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    stages = {
        "audited_base": {
            "commit": base_commit,
            "tree": base_tree,
        },
        "producer_launch_L": {
            "commit": launch_commit,
            "tree": launch_tree,
            "direct_parent": base_commit,
            "path_count": 21,
            "path_set_root_sha256": consumer._PRODUCER_L_PATH_ROOT,
        },
        "producer_payload_P": {
            "commit": payload_commit,
            "tree": payload_tree,
            "direct_parent": launch_commit,
            "path_count": 32,
            "path_set_root_sha256": consumer._PRODUCER_P_PATH_ROOT,
        },
        "producer_final_R": {
            "commit": final_commit,
            "tree": final_tree,
            "direct_parent": payload_commit,
            "path_count": 4,
            "path_set_root_sha256": consumer._PRODUCER_R_PATH_ROOT,
        },
        "global_clean_tree": True,
        "unique_direct_children": True,
        "exact_raw_diff_allowlist": True,
        "release_self_pin_present": False,
    }
    return repo, stages


def test_checked_in_config_is_explicitly_not_live_ready() -> None:
    loaded = consumer.load_config()
    assert loaded.value["state"] == "awaiting_producer_final_binding"
    assert loaded.value["producer_release_stage"] == {
        "state": "awaiting_producer_final_release",
        "trusted_release": None,
        "placeholder_values_accepted": False,
        "candidate_binding_accepted": False,
    }
    assert loaded.value["claims"]["live_ready"] is False
    assert loaded.value["claims"]["gpu_authorized"] is False
    assert loaded.value["claims"]["api_authorized"] is False


def test_all_new_schemas_are_valid_draft_2020_12() -> None:
    loaded = consumer.load_config()
    paths = [consumer.CONFIG_SCHEMA_PATH]
    paths.extend(
        consumer._repo_path(pin["path"], reason="test")
        for pin in loaded.value["schemas"].values()
    )
    for path in paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_teacher_shards_require_a_closed_exact320_semantic_join() -> None:
    rows, _fault_cells = _exact_rows()
    schema, teacher_rows, accepted_rows, training_rows = _teacher_join_fixture(rows)
    observed = consumer._validate_teacher_shards(
        schema=schema,
        shard_rows=teacher_rows,
        selected_rows=rows,
        accepted_rows=accepted_rows,
        training_rows=training_rows,
    )
    assert observed == rows


def test_teacher_shards_reject_wrong_target_hash() -> None:
    rows, _fault_cells = _exact_rows()
    schema, teacher_rows, accepted_rows, training_rows = _teacher_join_fixture(rows)
    teacher_rows[0]["teacher_target"] += " changed"
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_teacher_target_identity_drift",
    ):
        consumer._validate_teacher_shards(
            schema=schema,
            shard_rows=teacher_rows,
            selected_rows=rows,
            accepted_rows=accepted_rows,
            training_rows=training_rows,
        )


def test_teacher_shards_reject_reordered_physical_rows() -> None:
    rows, _fault_cells = _exact_rows()
    schema, teacher_rows, accepted_rows, training_rows = _teacher_join_fixture(rows)
    teacher_rows[0], teacher_rows[1] = teacher_rows[1], teacher_rows[0]
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_teacher_record_join_drift",
    ):
        consumer._validate_teacher_shards(
            schema=schema,
            shard_rows=teacher_rows,
            selected_rows=rows,
            accepted_rows=accepted_rows,
            training_rows=training_rows,
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("fake_field", "minimal320_teacher_shard_invalid"),
        ("wrong_target", "minimal320_teacher_target_identity_drift"),
        ("reorder", "minimal320_teacher_record_join_drift"),
        ("duplicate", "minimal320_teacher_record_duplicate"),
        ("omitted", "minimal320_artifact_kind_inventory_invalid"),
        ("extra", "minimal320_producer_binding_schema_mismatch"),
    ],
)
def test_physical_teacher_shard_tamper_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: str,
) -> None:
    def mutate(
        teacher_rows: list[dict[str, Any]],
        _joined_rows: list[dict[str, Any]],
    ) -> None:
        if mutation == "fake_field":
            teacher_rows[0]["unexpected"] = "not allowed"
        elif mutation == "wrong_target":
            teacher_rows[0]["teacher_target"] += " changed"
        elif mutation == "reorder":
            teacher_rows[0], teacher_rows[1] = teacher_rows[1], teacher_rows[0]
        elif mutation == "duplicate":
            teacher_rows[1] = deepcopy(teacher_rows[0])
        elif mutation == "omitted":
            teacher_rows.pop()
        elif mutation == "extra":
            teacher_rows.append(deepcopy(teacher_rows[-1]))
        else:
            raise AssertionError(f"unsupported teacher mutation: {mutation}")

    with pytest.raises(consumer.Minimal320ConsumerError, match=reason):
        _authenticate(
            tmp_path,
            monkeypatch,
            teacher_rows_mutator=mutate,
        )


def test_physical_teacher_record_schema_tamper_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(schema: dict[str, Any]) -> None:
        schema["additionalProperties"] = True

    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_teacher_record_schema_contract_drift",
    ):
        _authenticate(
            tmp_path,
            monkeypatch,
            teacher_schema_mutator=mutate,
        )


@pytest.mark.parametrize(
    ("inventory_name", "reason"),
    [
        ("accepted", "minimal320_accepted_inventory_order_invalid"),
        ("training", "minimal320_teacher_record_join_drift"),
    ],
)
def test_teacher_join_inventory_reorder_is_rejected(
    inventory_name: str,
    reason: str,
) -> None:
    rows, _fault_cells = _exact_rows()
    schema, teacher_rows, accepted_rows, training_rows = _teacher_join_fixture(rows)
    accepted = deepcopy(accepted_rows)
    training = deepcopy(training_rows)
    target = accepted if inventory_name == "accepted" else training
    target[0], target[1] = target[1], target[0]
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match=reason,
    ):
        consumer._validate_teacher_shards(
            schema=schema,
            shard_rows=teacher_rows,
            selected_rows=rows,
            accepted_rows=accepted,
            training_rows=training,
        )


def test_teacher_identity_and_review_semantics_are_recomputed() -> None:
    rows, _fault_cells = _exact_rows()
    schema, teacher_rows, accepted_rows, training_rows = _teacher_join_fixture(rows)
    identity_index = next(
        index
        for index, row in enumerate(rows)
        if row["selection_kind"] == "identity" and row["output_expert_role"] == "tool"
    )
    _replace_teacher_target(
        teacher_rows,
        training_rows,
        identity_index,
        consumer._IDENTITY_TEMPLATES["air_attribution"],
    )
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_teacher_identity_semantics_invalid",
    ):
        consumer._validate_teacher_shards(
            schema=schema,
            shard_rows=teacher_rows,
            selected_rows=rows,
            accepted_rows=accepted_rows,
            training_rows=training_rows,
        )

    schema, teacher_rows, accepted_rows, training_rows = _teacher_join_fixture(rows)
    review_index = next(
        index
        for index, row in enumerate(rows)
        if row["selection_kind"] == "tool_review_depth"
        and row["output_expert_role"] == "review"
        and row["review_verdict"] == "fail"
    )
    target = _canonical_bytes(
        {
            "verdict": "fail",
            "faults": ["wrong_fault"],
            "correction": "Correct the audited fault.",
        }
    ).decode()
    _replace_teacher_target(teacher_rows, training_rows, review_index, target)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_teacher_review_semantics_invalid",
    ):
        consumer._validate_teacher_shards(
            schema=schema,
            shard_rows=teacher_rows,
            selected_rows=rows,
            accepted_rows=accepted_rows,
            training_rows=training_rows,
        )


def test_legacy_nine_arm_files_are_physically_unchanged() -> None:
    loaded = consumer.load_config()
    for pin in loaded.value["legacy_matrix"]["pinned_files"]:
        raw = consumer._repo_path(pin["path"], reason="test").read_bytes()
        assert len(raw) == pin["bytes"]
        assert _sha(raw) == pin["sha256"]
    assert tuple(loaded.value["arms"]["ordered"]) == legacy_matrix.TRAINED_ARMS
    assert tuple(loaded.value["arms"]["eval_only"]) == legacy_matrix.EVAL_ONLY_ARMS


def test_default_anchor_blocks_even_with_a_repo(tmp_path: Path) -> None:
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="producer_final_release_anchor_required",
    ):
        consumer.authenticate_release(
            producer_repo=tmp_path,
        )


def test_consumer_c_is_exactly_one_config_only_child_of_runtime_p(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = consumer.ROOT
    template = consumer.load_config()
    repo, anchor, loaded = _build_consumer_config_commit(
        tmp_path,
        source_root,
        template,
    )
    monkeypatch.setattr(consumer, "ROOT", repo)
    identity = consumer._consumer_git_identity(anchor, loaded)
    assert identity.runtime_commit == anchor["consumer_runtime_commit"]
    assert identity.runtime_tree == anchor["consumer_runtime_tree"]
    assert identity.head != identity.runtime_commit


@pytest.mark.parametrize(
    "extra_path",
    ["sitecustomize.py", "src/anchor_mvp/__init__.py"],
)
def test_consumer_c_rejects_extra_import_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_path: str,
) -> None:
    source_root = consumer.ROOT
    template = consumer.load_config()
    repo, anchor, loaded = _build_consumer_config_commit(
        tmp_path,
        source_root,
        template,
        mutation=extra_path,
    )
    monkeypatch.setattr(consumer, "ROOT", repo)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_consumer_config_only_diff_drift",
    ):
        consumer._consumer_git_identity(anchor, loaded)


@pytest.mark.parametrize("mutation", ["non_direct", "merge"])
def test_consumer_c_rejects_non_direct_or_merge_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source_root = consumer.ROOT
    template = consumer.load_config()
    repo, anchor, loaded = _build_consumer_config_commit(
        tmp_path,
        source_root,
        template,
        mutation=mutation,
    )
    monkeypatch.setattr(consumer, "ROOT", repo)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_consumer_config_commit_not_direct",
    ):
        consumer._consumer_git_identity(anchor, loaded)


@pytest.mark.parametrize("mutation", ["rename", "mode"])
def test_consumer_c_rejects_rename_or_mode_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source_root = consumer.ROOT
    template = consumer.load_config()
    repo, anchor, loaded = _build_consumer_config_commit(
        tmp_path,
        source_root,
        template,
        mutation=mutation,
    )
    monkeypatch.setattr(consumer, "ROOT", repo)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_consumer_config_only_diff_drift",
    ):
        consumer._consumer_git_identity(anchor, loaded)


def test_real_git_release_and_exact_320_build_a_closed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, release, _repo = _authenticate(tmp_path, monkeypatch)
    receipt = consumer.build_preflight_receipt(loaded, release)
    assert receipt["state"] == "model_free_preflight_passed"
    assert receipt["records"] == 320
    assert receipt["consumer_git_head"] == "e" * 40
    assert receipt["consumer_git_tree"] == "f" * 40
    assert receipt["consumer_runtime_commit"] == "a" * 40
    assert receipt["consumer_runtime_tree"] == "b" * 40
    assert receipt["consumer_implementation_sha256"] == "c" * 64
    assert receipt["consumer_schema_inventory_sha256"] == "d" * 64
    assert (
        receipt["producer_release_commit"]
        == release.trust_anchor["producer_stages"]["producer_final_R"]["commit"]
    )
    assert (
        receipt["producer_evidence"]["selected_candidates_sha256"]
        == (release.binding["selection_manifest"]["selected_set_root_sha256"])
    )
    assert release.binding["selection_manifest"]["strata"]["router_record_cells"] == {
        "humor": {"en": 14, "zh": 13},
        "serious": {"en": 13, "zh": 13},
        "angry": {"en": 13, "zh": 14},
    }
    payload = dict(receipt)
    receipt_id = payload.pop("receipt_id")
    assert receipt_id == _sha(payload)
    assert all("identity" not in arm for arm in receipt["arms"]["ordered"])
    assert receipt["claims"]["api_called"] is False
    assert receipt["claims"]["gpu_started"] is False


def test_authenticated_metadata_overlay_cannot_be_substituted_as_teacher_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(tmp_path, loaded)
    overlay_path = (
        f"{consumer._PRODUCER_RELEASE_ROOT}/authenticated_metadata_overlay.jsonl"
    )
    _mutate_embedded_binding(
        anchor,
        lambda binding: binding["artifact_bindings"].update(
            {"teacher_shards": [overlay_path]}
        ),
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_binding_schema_mismatch",
    ):
        consumer.authenticate_release(producer_repo=repo)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "minimal320_training_record_inventory_invalid"),
        ("tamper", "minimal320_teacher_record_join_drift"),
        ("reorder", "minimal320_teacher_record_join_drift"),
        ("duplicate", "minimal320_teacher_record_join_drift"),
    ],
)
def test_physical_training_inventory_is_closed_and_order_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: str,
) -> None:
    def mutate(
        _teacher_rows: list[dict[str, Any]],
        training_rows: list[dict[str, Any]],
    ) -> None:
        if mutation == "missing":
            training_rows.pop()
        elif mutation == "tamper":
            training_rows[0]["source_content_sha256"] = "f" * 64
        elif mutation == "reorder":
            training_rows[0], training_rows[1] = (
                training_rows[1],
                training_rows[0],
            )
        elif mutation == "duplicate":
            training_rows[1] = deepcopy(training_rows[0])
        else:
            raise AssertionError(f"unsupported training mutation: {mutation}")

    with pytest.raises(consumer.Minimal320ConsumerError, match=reason):
        _authenticate(
            tmp_path,
            monkeypatch,
            teacher_rows_mutator=mutate,
        )


def test_committed_shards_P_cannot_self_pin_future_R(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(manifest: dict[str, Any]) -> None:
        manifest["future_producer_final_R_sha256"] = "f" * 64

    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_committed_shards_manifest_invalid",
    ):
        _authenticate(
            tmp_path,
            monkeypatch,
            committed_manifest_mutator=mutate,
        )


def test_legacy_exact3520_binding_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(tmp_path, loaded)

    def mutate(binding: dict[str, Any]) -> None:
        binding["selection_manifest"]["counts"]["selected"] = 3520

    _mutate_embedded_binding(anchor, mutate)
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_binding_schema_mismatch",
    ):
        consumer.authenticate_release(producer_repo=repo)


@pytest.mark.parametrize("release_mutation", ["non_direct", "merge"])
def test_producer_final_r_must_be_the_unique_direct_child_of_payload_p(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_mutation: str,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        release_mutation=release_mutation,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match=("minimal320_producer_(?:stage_parent_drift|final_parent_invalid)"),
    ):
        consumer.authenticate_release(producer_repo=repo)


@pytest.mark.parametrize(
    "release_mutation",
    ["extra", "rename", "mode", "submodule"],
)
def test_producer_release_diff_is_the_exact_four_file_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_mutation: str,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        release_mutation=release_mutation,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_(?:stage_diff_drift|final_diff_invalid)",
    ):
        consumer.authenticate_release(producer_repo=repo)


@pytest.mark.parametrize(
    "stage_mutation",
    ["candidate_as_release", "release_as_candidate", "arbitrary"],
)
def test_artifact_kind_cannot_be_relabelled_to_another_git_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_mutation: str,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        stage_mutation=stage_mutation,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_binding_schema_mismatch",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_producer_repository_must_be_globally_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        release_mutation="untracked",
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_git_identity_drift",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_caller_cannot_substitute_a_fabricated_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(tmp_path, loaded)
    anchor["producer_stages"]["producer_payload_P"]["commit"] = "1" * 40
    anchor["producer_stages"]["producer_final_R"]["direct_parent"] = "1" * 40
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_stage_commit_invalid",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_component_hash_must_map_to_the_physical_git_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        component_mismatch=True,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_component_artifact_pin_drift",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_two_dummy_rows_cannot_authenticate_as_320(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def keep_two(rows: list[dict[str, Any]]) -> None:
        del rows[2:]

    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        rows_mutator=keep_two,
        teacher_rows_from_unmutated=True,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_selected_candidate_inventory_invalid",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_duplicate_artifact_paths_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        duplicate_artifact_path=True,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_binding_schema_mismatch",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_same_physical_file_via_hardlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(tmp_path, loaded)
    quarantine = repo / consumer._PRODUCER_RELEASE_ROOT / "quarantine_inventory.jsonl"
    quarantine.unlink()
    os.link(
        repo / consumer._PRODUCER_RELEASE_ROOT / "rejected_inventory.jsonl",
        quarantine,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_artifact_hardlink_alias_detected",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_symlinked_artifact_is_rejected_when_platform_supports_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(tmp_path, loaded)
    accepted = repo / consumer._PRODUCER_RELEASE_ROOT / "accepted_inventory.jsonl"
    accepted.unlink()
    try:
        accepted.symlink_to("selected_inventory.jsonl")
    except OSError:
        pytest.skip("symlink capability unavailable")
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_artifact_path_invalid",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_noncanonical_embedded_binding_is_rejected_before_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(tmp_path, loaded)
    anchor["producer_binding_canonical_json"] = (
        anchor["producer_binding_canonical_json"] + " "
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_release_pin_schema_mismatch",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_authenticated_source_fault_cells_cannot_be_reassigned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def swap_fault_cells(rows: list[dict[str, Any]]) -> None:
        failures = [row for row in rows if row["review_fault_type"] is not None]
        first = failures[0]
        second = next(
            row
            for row in failures[1:]
            if row["review_fault_type"] != first["review_fault_type"]
        )
        first["review_fault_type"], second["review_fault_type"] = (
            second["review_fault_type"],
            first["review_fault_type"],
        )
        _refresh_candidate_identity(first)
        _refresh_candidate_identity(second)

    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        rows_mutator=swap_fault_cells,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_review_fault_cell_drift",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_identity_bundle_cell_quota_is_recomputed_from_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def move_identity_bundle(rows: list[dict[str, Any]]) -> None:
        target = next(
            row["task_bundle_sha256"]
            for row in rows
            if row["selection_kind"] == "identity"
            and row["identity_class"] == "air_attribution"
            and row["language"] == "en"
        )
        for row in rows:
            if row["task_bundle_sha256"] == target:
                row["identity_class"] = "false_google_attribution"
                row["identity_parent_class"] = "false_google_attribution"
                _refresh_candidate_identity(row)

    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        rows_mutator=move_identity_bundle,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_identity_cell_distribution_invalid",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_router_cell_quota_is_recomputed_from_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def move_router_record(rows: list[dict[str, Any]]) -> None:
        target = next(
            row
            for row in rows
            if row["selection_kind"] == "router"
            and row["router_label"] == "humor"
            and row["language"] == "en"
        )
        target["router_label"] = "serious"
        _refresh_candidate_identity(target)

    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        rows_mutator=move_router_record,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_router_cell_distribution_invalid",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_obsolete_review_fault_vocabulary_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def obsolete_fault(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["review_fault_type"] is not None:
                row["review_fault_type"] = "missing_evidence"
                _refresh_candidate_identity(row)
                return
        raise AssertionError("fixture has no fault row")

    loaded = consumer.load_config()
    repo, anchor = _build_producer_repo(
        tmp_path,
        loaded,
        rows_mutator=obsolete_fault,
    )
    frozen = _frozen_loaded(loaded, anchor)
    _install_frozen_config(monkeypatch, frozen)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_record_review_fault_invalid",
    ):
        consumer.authenticate_release(producer_repo=repo)


def test_cli_has_no_caller_supplied_pin_or_hash_arguments() -> None:
    options = {action.dest for action in consumer._parser()._actions}
    assert "release_pin" not in options
    assert "release_pin_sha256" not in options
    assert "producer_repo" in options


def test_cli_without_a_producer_repo_is_body_free_blocked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert consumer.main([]) == 2
    status = json.loads(capsys.readouterr().out)
    assert status["blockers"] == ["producer_repository_required"]
    assert status["api_called"] is False
    assert status["gpu_started"] is False


def test_exact_b_l21_p32_r4_stage_chain_is_physically_recomputed(
    tmp_path: Path,
) -> None:
    repo, stages = _build_b_l_p_r_repo(tmp_path)
    launch, payload, final = consumer._verify_b_l_p_r_stage_chain(repo, stages)
    assert all(path in launch for path in consumer._PRODUCER_L_PATHS)
    assert all(path in payload for path in consumer._PRODUCER_P_PATHS)
    assert all(path in final for path in consumer._PRODUCER_R_PATHS)


def test_old_candidate_release_stage_names_are_rejected(tmp_path: Path) -> None:
    repo, stages = _build_b_l_p_r_repo(tmp_path)
    old = deepcopy(stages)
    old["candidate"] = old.pop("producer_payload_P")
    old["release"] = old.pop("producer_final_R")
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_stage_shape_invalid",
    ):
        consumer._verify_b_l_p_r_stage_chain(repo, old)


@pytest.mark.parametrize("legacy_path_count", [28, 30, 3520])
def test_legacy_payload_stage_counts_are_rejected(
    tmp_path: Path,
    legacy_path_count: int,
) -> None:
    repo, stages = _build_b_l_p_r_repo(tmp_path)
    old = deepcopy(stages)
    old["producer_payload_P"]["path_count"] = legacy_path_count
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_stage_contract_drift",
    ):
        consumer._verify_b_l_p_r_stage_chain(repo, old)


def test_extra_payload_file_is_rejected_from_exact_p32(tmp_path: Path) -> None:
    repo, stages = _build_b_l_p_r_repo(tmp_path, extra_payload_path=True)
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_stage_diff_drift",
    ):
        consumer._verify_b_l_p_r_stage_chain(repo, stages)


def test_stage_parent_or_count_assertion_cannot_replace_git_evidence(
    tmp_path: Path,
) -> None:
    repo, stages = _build_b_l_p_r_repo(tmp_path)
    forged = deepcopy(stages)
    forged["producer_payload_P"]["direct_parent"] = "f" * 40
    with pytest.raises(
        consumer.Minimal320ConsumerError,
        match="minimal320_producer_stage_contract_drift",
    ):
        consumer._verify_b_l_p_r_stage_chain(repo, forged)
