from __future__ import annotations

import asyncio
import base64
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2 as selector,
)
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1 as vnext,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _policy(
    *,
    rejection_window: int = 120,
    rejection_min_samples: int = 30,
    rejection_max_rate: float = 0.35,
) -> vnext.VNextPolicy:
    config_dir = REPO_ROOT / "configs" / "data"
    consumer_code_P = {
        "stage": "consumer_code_P",
        "status": "audited_compatible",
        "blocker_reason_code": None,
        "commit_sha1": "1" * 40,
        "tree_sha1": "2" * 40,
        "file_inventory_root_sha256": "3" * 64,
        "implementation_sha256": "4" * 64,
        "schema_set_root_sha256": "5" * 64,
        "contract_sha256": "6" * 64,
        "router_record_cells": {
            "humor": {"en": 14, "zh": 13},
            "serious": {"en": 13, "zh": 13},
            "angry": {"en": 13, "zh": 14},
        },
        "independently_audited": True,
        "content_retained": False,
    }
    return vnext.VNextPolicy(
        path=config_dir / "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.json",
        physical_sha256="1" * 64,
        schema_path=config_dir
        / "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.schema.json",
        contract_sha256="2" * 64,
        schema_sha256="3" * 64,
        implementation_path=REPO_ROOT / "src/anchor_mvp/data/"
        "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
        implementation_sha256="c" * 64,
        checkpoint_schema_path=config_dir
        / (
            "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_"
            "checkpoint_v1.schema.json"
        ),
        checkpoint_schema_sha256="4" * 64,
        receipt_schema_path=config_dir
        / ("gemma3_chat_unbalanced_v2_teacher_alignment_vnext_receipt_v1.schema.json"),
        receipt_schema_sha256="5" * 64,
        selector_contract_sha256="6" * 64,
        selector_schema_path=config_dir
        / (
            "gemma3_chat_unbalanced_v2_teacher_alignment_"
            "minimal320_selector_v2.schema.json"
        ),
        selector_schema_sha256="7" * 64,
        selector_implementation_sha256="8" * 64,
        teacher_record_schema_path=config_dir
        / "gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json",
        teacher_record_schema_sha256="9" * 64,
        transition_schema_path=config_dir
        / (
            "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_"
            "transition_v1.schema.json"
        ),
        transition_schema_sha256="b" * 64,
        base_profile_path=config_dir
        / (
            "gemma3_chat_five_expert_qonly_unbalanced_v2_"
            "teacher_alignment.bulk_c30.yaml"
        ),
        base_profile_sha256="d" * 64,
        output_root_parent=REPO_ROOT / "data" / "vnext-test-runs",
        old_canonical_root=REPO_ROOT / "data" / "vnext-old-canonical",
        old_archive_parent=REPO_ROOT / "data" / "vnext-old-archive",
        controller_lease_name=".vnext-test-controller.lock",
        rejection_window=rejection_window,
        rejection_min_samples=rejection_min_samples,
        rejection_max_rate=rejection_max_rate,
        claims={
            "candidate": True,
            "training_authorized": False,
        },
        consumer_code_P=consumer_code_P,
    )


def _source(
    *,
    role: str = "serious",
    language: str = "en",
    identity_class: str | None = None,
    ordinal: int = 1,
) -> batch.SourceRecord:
    tools = ["calendar.lookup"] if role == "tool" else []
    return batch.SourceRecord(
        record_id=f"record-{ordinal}",
        task_bundle=f"bundle-{ordinal}",
        semantic=f"semantic-{ordinal}",
        role=role,
        language=language,
        teacher_input="synthetic input not emitted by vNext metadata",
        gemma_serialization_identity_sha256="a" * 64,
        teacher_contract={
            "identity_class": identity_class,
            "prompt_template_id": f"template-{role}",
            "output_schema_id": f"schema-{role}",
            "allowed_tools": tools,
            "allowed_evidence_ids": (["evidence-1"] if role == "tool" else []),
            "router_options": (
                ["humor", "serious", "angry"] if role == "router" else []
            ),
        },
        guards={
            "target_leakage_sha256": [],
            "references": [],
        },
        partition_kind="train",
        line_number=ordinal,
        source_line_sha256="b" * 64,
        content_identity_sha256="c" * 64,
    )


def _job(source: batch.SourceRecord, ordinal: int = 1) -> batch.AlignmentJob:
    return batch.AlignmentJob(
        source=source,
        idempotency_key=_hash_text(f"idempotency-{ordinal}"),
        prompt_template_sha256=_hash_text(batch._system_prompt(source.role)),
        output_schema_sha256="e" * 64,
    )


def _view(
    job: batch.AlignmentJob,
    *,
    output_role: str | None = None,
    tool_family: str | None = "calendar.lookup",
    selection_kind: str = "full_core",
) -> dict[str, Any]:
    return {
        "source_record_id_sha256": _hash_text(job.source.record_id),
        "source_content_sha256": job.source.content_identity_sha256,
        "source_serialization_identity_sha256": (
            job.source.gemma_serialization_identity_sha256
        ),
        "task_bundle_sha256": _hash_text(job.source.task_bundle),
        "selection_kind": selection_kind,
        "output_expert_role": output_role or job.source.role,
        "training_asset": selector.TRAINING_ASSET_BY_OUTPUT_ROLE[
            output_role or job.source.role
        ],
        "validation_contract_role": job.source.role,
        "language": "zh" if job.source.language == "zh-CN" else "en",
        "tool_family": tool_family,
        "identity_class": job.source.identity_class,
        "identity_parent_class": (
            job.source.identity_class if selection_kind == "identity" else None
        ),
        "review_verdict": None,
        "review_fault_type": None,
        "router_label": None,
    }


def _selection_manifest() -> dict[str, Any]:
    return {
        "selector_identity_sha256": "f" * 64,
        "selected_set_root_sha256": "0" * 64,
        "counts": {
            "requested": 320,
            "selected": 320,
            "requested_by_role": {
                "angry": 40,
                "humor": 40,
                "review": 60,
                "router": 80,
                "serious": 40,
                "tool": 60,
            },
            "selected_by_role": {
                "angry": 40,
                "humor": 40,
                "review": 60,
                "router": 80,
                "serious": 40,
                "tool": 60,
            },
            "requested_by_language": {"en": 160, "zh": 160},
            "selected_by_language": {"en": 160, "zh": 160},
            "requested_by_partition": {
                "full_core": 100,
                "identity": 100,
                "router": 80,
                "tool_review_depth": 40,
            },
            "selected_by_partition": {
                "full_core": 100,
                "identity": 100,
                "router": 80,
                "tool_review_depth": 40,
            },
        },
    }


def _inventory() -> batch.SourceInventory:
    return batch.SourceInventory(
        records=(),
        manifest_sha256="1" * 64,
        approval_sha256="2" * 64,
        partition_set_sha256="3" * 64,
        readonly_test_identity_sha256="4" * 64,
        source_identity_sha256="5" * 64,
        role_counts={},
        language_counts={},
        identity_counts={},
    )


def _job_provenance(
    profile: batch.AdapterConfig,
    inventory: batch.SourceInventory,
    policy: vnext.VNextPolicy,
) -> dict[str, Any]:
    payload = {
        "schema_version": vnext.JOB_PROVENANCE_SCHEMA_VERSION,
        "runtime": {
            "vnext_config_sha256": policy.physical_sha256,
            "vnext_config_schema_sha256": policy.schema_sha256,
            "vnext_contract_sha256": policy.contract_sha256,
            "vnext_implementation_sha256": policy.implementation_sha256,
            "receipt_schema_sha256": policy.receipt_schema_sha256,
            "checkpoint_schema_sha256": policy.checkpoint_schema_sha256,
            "base_profile_sha256": profile.physical_sha256,
            "campaign_sha256": profile.campaign_sha256,
            "model_binding_sha256": profile.model_binding_sha256,
            "teacher_implementation_sha256": profile.contract_hashes[
                "teacher_implementation_path"
            ],
        },
        "source": {
            "source_identity_sha256": inventory.source_identity_sha256,
            "manifest_sha256": inventory.manifest_sha256,
            "protected_test_identity_sha256": (inventory.readonly_test_identity_sha256),
        },
        "selector": {
            "selector_contract_sha256": policy.selector_contract_sha256,
            "selector_schema_sha256": policy.selector_schema_sha256,
            "selector_implementation_sha256": (policy.selector_implementation_sha256),
            "overlay_manifest_sha256": "a" * 64,
            "overlay_rows_root_sha256": "b" * 64,
            "overlay_row_count": 3520,
            "preimage_manifest_sha256": "c" * 64,
            "preimage_schema_sha256": "d" * 64,
            "candidate_schema_sha256": "e" * 64,
            "candidate_set_root_sha256": "f" * 64,
            "candidate_count": 3960,
            "selected_set_root_sha256": "0" * 64,
            "selected_count": 320,
            "selector_identity_sha256": "f" * 64,
            "identity_derivative_plan_sha256": "1" * 64,
        },
        "producer_launch_L": {
            "stage": "producer_launch_L",
            "audited_base_commit_sha1": "6" * 40,
            "audited_base_tree_sha1": "5" * 40,
            "audited_base_path_status_root_sha256": "4" * 64,
            "commit_sha1": "7" * 40,
            "tree_sha1": "8" * 40,
            "reviewed_file_inventory_root_sha256": "9" * 64,
            "reviewed_file_count": len(vnext.PRODUCER_LAUNCH_REVIEWED_PATHS),
            "global_clean_tree": True,
            "procedurally_independent_review_present": False,
        },
        "consumer_code_P": dict(policy.consumer_code_P),
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }
    return {
        **payload,
        "provenance_identity_sha256": hashlib.sha256(
            vnext._canonical_bytes(payload)
        ).hexdigest(),
    }


def _state(
    tmp_path: Path,
    *,
    policy: vnext.VNextPolicy | None = None,
    source: batch.SourceRecord | None = None,
    output_role: str | None = None,
    tool_family: str | None = "calendar.lookup",
    selection_kind: str = "full_core",
    stop_probe: Any = None,
    final_quality_validator: Any = vnext.final_quality_validate,
    monotonic_clock: Any = time.monotonic,
    observed_at_clock: Any = batch._iso,
) -> tuple[
    vnext.VNextAuthenticatedBatchState,
    batch.SourceInventory,
    batch.AlignmentJob,
    bytes,
]:
    profile = batch.load_config(
        REPO_ROOT / "configs/data/"
        "gemma3_chat_five_expert_qonly_unbalanced_v2_"
        "teacher_alignment.smoke_exact1.yaml"
    )
    output = dict(profile.output)
    output["root"] = str(tmp_path / "data" / batch.DATASET_KIND / "vnext-shard")
    profile = replace(profile, output=output)
    source = source or _source()
    job = _job(source)
    view = _view(
        job,
        output_role=output_role,
        tool_family=tool_family,
        selection_kind=selection_kind,
    )
    inventory = _inventory()
    key = b"k" * 32
    binding = {
        "controller_config_sha256": "6" * 64,
        "controller_implementation_sha256": "7" * 64,
        "teacher_implementation_sha256": "8" * 64,
        "source_identity_sha256": inventory.source_identity_sha256,
        "manifest_sha256": inventory.manifest_sha256,
        "protected_test_identity_sha256": (inventory.readonly_test_identity_sha256),
        "controller_run_id": "9" * 32,
        "terminal_transition_recheck_sha256": "a" * 64,
        "accepted_profiles": {
            profile.profile_id: profile.physical_sha256,
        },
    }
    active_policy = policy or _policy()
    state = vnext.VNextAuthenticatedBatchState(
        profile,
        hmac_key=key,
        controller_binding=binding,
        policy=active_policy,
        selection_manifest=_selection_manifest(),
        selection_views={job.idempotency_key: view},
        job_provenance=_job_provenance(
            profile,
            inventory,
            active_policy,
        ),
        jobs=(job,),
        stop_probe=stop_probe,
        final_quality_validator=final_quality_validator,
        monotonic_clock=monotonic_clock,
        observed_at_clock=observed_at_clock,
    )
    state.initialize(inventory)
    return state, inventory, job, key


async def _dispatch(
    state: vnext.VNextAuthenticatedBatchState,
    job: batch.AlignmentJob,
) -> None:
    await state.append_event(
        job=job,
        phase="smoke_exact1",
        state="reserved",
        reason_code="budget_reserved",
        reservation={
            "requests": 1,
            "input_tokens": 1,
            "output_tokens": 0,
            "cost_units": 1,
        },
    )
    await state.append_event(
        job=job,
        phase="smoke_exact1",
        state="dispatching",
        reason_code="provider_dispatch",
    )


def _pending(
    state: vnext.VNextAuthenticatedBatchState,
    inventory: batch.SourceInventory,
    job: batch.AlignmentJob,
    key: bytes,
    *,
    rejected: bool = False,
) -> batch.PendingCommit:
    usage = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    attempts = {
        "wire_attempts": 1,
        "retry_count": 0,
        "retry_reasons": [],
        "response_id": None,
    }
    output_record = (
        None
        if rejected
        else {
            "idempotency_key": job.idempotency_key,
            "usage": usage,
            "attempts": attempts,
        }
    )
    outcome = "rejected" if rejected else "succeeded"
    reason = (
        "vnext_closed_grammar_keys_invalid"
        if rejected
        else "online_candidate_committed"
    )
    payload = {
        "id": _hash_text(f"receipt-{outcome}"),
        "schema_version": vnext.RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "job",
        "idempotency_key": job.idempotency_key,
        "role": state._pending_group_strata.get(
            job.idempotency_key,
            vnext.body_free_stratum(
                job.source,
                state._selection_views[job.idempotency_key],
            ),
        )["role"],
        "validation_contract_role": job.source.role,
        "language": vnext._normalized_language(job.source.language),
        "stratum": vnext.body_free_stratum(
            job.source,
            state._selection_views[job.idempotency_key],
        ),
        "outcome": outcome,
        "reason_code": reason,
        "output_record_sha256": (
            batch._hash_object(output_record) if output_record is not None else None
        ),
        "planner_lineage": batch._planner_lineage_receipt(
            job,
            overlay_identity_sha256=None,
        ),
        "prompt_binding": vnext.prompt_binding(
            job,
            state._selection_views[job.idempotency_key],
            identity_derivative_spec=None,
            inventory=inventory,
            config=state.config,
        ),
        "usage": usage,
        "attempts": attempts,
        "provenance": dict(state._job_provenance),
        "committed_at": batch._iso(),
        "content_retained": False,
        "planner_body_retained": False,
        "raw_token_ids_retained": False,
    }
    receipt = {
        **payload,
        "hmac_sha256": batch._receipt_hmac(payload, key),
    }
    return batch.PendingCommit(
        job=job,
        output_record=output_record,
        receipt=receipt,
        terminal_state="rejected" if rejected else "committed",
        event_reason=reason,
    )


def test_real_state_initializes_checkpoint_tree_outside_repository(
    tmp_path: Path,
) -> None:
    state, _, _, _ = _state(tmp_path / "external-runtime")
    assert batch.REPO_ROOT.absolute() not in state.root.absolute().parents
    assert state.checkpoint_dir.is_dir()
    assert state.checkpoint_dir.is_relative_to(state.root)
    assert state.full_replay_counts == {"startup": 1}


def _candidate(
    ordinal: int,
    *,
    kind: str,
    role: str,
    validation_role: str,
    language: str,
    bundle: str,
    tool_family: str | None = None,
    identity_class: str | None = None,
    review_verdict: str | None = None,
    review_fault_type: str | None = None,
    router_label: str | None = None,
    parent_source_record_id_sha256: str | None = None,
    identity_parent_class: str | None = None,
) -> selector.SelectorCandidate:
    source_record_id_sha256 = _hash_text(f"record-{ordinal}")
    idempotency_key = _hash_text(f"key-{ordinal}")
    task_bundle_sha256 = _hash_text(bundle)
    stable = {
        "idempotency_key": idempotency_key,
        "teacher_request_idempotency_key": idempotency_key,
        "source_record_id_sha256": source_record_id_sha256,
        "source_content_sha256": _hash_text(f"source-content:{ordinal}"),
        "source_serialization_identity_sha256": _hash_text(
            f"source-serialization:{ordinal}"
        ),
        "teacher_request_record_id_sha256": source_record_id_sha256,
        "parent_source_record_id_sha256": (
            parent_source_record_id_sha256 or source_record_id_sha256
        ),
        "task_bundle_sha256": task_bundle_sha256,
        "parent_task_bundle_sha256": task_bundle_sha256,
        "source_semantic_sha256": _hash_text(f"semantic-{ordinal}"),
        "overlay_row_sha256": _hash_text(f"overlay-{ordinal}"),
        "identity_derivative_spec_sha256": (
            _hash_text(f"identity-spec-{ordinal}") if kind == "identity" else None
        ),
        "selection_kind": kind,
        "output_expert_role": role,
        "training_asset": selector.TRAINING_ASSET_BY_OUTPUT_ROLE[role],
        "validation_contract_role": validation_role,
        "language": language,
        "tool_family": tool_family,
        "identity_class": identity_class,
        "identity_parent_class": identity_parent_class,
        "identity_provenance_intent": (
            "air_provenance_invariant" if kind == "identity" else None
        ),
        "review_verdict": review_verdict,
        "review_fault_type": review_fault_type,
        "router_label": router_label,
    }
    return selector.SelectorCandidate(
        candidate_id_sha256=selector.candidate_identity_sha256(stable),
        idempotency_key=idempotency_key,
        teacher_request_idempotency_key=idempotency_key,
        source_record_id_sha256=source_record_id_sha256,
        source_content_sha256=stable["source_content_sha256"],
        source_serialization_identity_sha256=stable[
            "source_serialization_identity_sha256"
        ],
        teacher_request_record_id_sha256=source_record_id_sha256,
        parent_source_record_id_sha256=(
            parent_source_record_id_sha256 or source_record_id_sha256
        ),
        task_bundle_sha256=task_bundle_sha256,
        parent_task_bundle_sha256=task_bundle_sha256,
        source_semantic_sha256=stable["source_semantic_sha256"],
        overlay_row_sha256=stable["overlay_row_sha256"],
        identity_derivative_spec_sha256=stable["identity_derivative_spec_sha256"],
        selection_kind=kind,
        output_expert_role=role,
        training_asset=stable["training_asset"],
        validation_contract_role=validation_role,
        language=language,
        tool_family=tool_family,
        identity_class=identity_class,
        identity_parent_class=identity_parent_class,
        identity_provenance_intent=stable["identity_provenance_intent"],
        review_verdict=review_verdict,
        review_fault_type=review_fault_type,
        router_label=router_label,
    )


def _minimal320_candidates() -> list[selector.SelectorCandidate]:
    result: list[selector.SelectorCandidate] = []
    ordinal = 0
    families = [f"tool.family.{index:02d}" for index in range(10)]
    for family in families:
        for language in selector.LANGUAGES:
            bundle = f"core:{family}:{language}"
            for role in selector.OUTPUT_EXPERT_ROLES:
                ordinal += 1
                result.append(
                    _candidate(
                        ordinal,
                        kind="full_core",
                        role=role,
                        validation_role=role,
                        language=language,
                        bundle=bundle,
                        tool_family=family,
                    )
                )
    for family_rank, family in enumerate(families):
        for language in selector.LANGUAGES:
            verdict = "fail" if language == "en" else "pass"
            fault = (
                selector.FAULT_TYPES[family_rank % len(selector.FAULT_TYPES)]
                if verdict == "fail"
                else None
            )
            bundle = f"depth:{family}:{language}"
            for role in ("tool", "review"):
                ordinal += 1
                result.append(
                    _candidate(
                        ordinal,
                        kind="tool_review_depth",
                        role=role,
                        validation_role=role,
                        language=language,
                        bundle=bundle,
                        tool_family=family,
                        review_verdict=verdict,
                        review_fault_type=fault,
                    )
                )
    identity_bundle_index = 0
    for (
        identity_class,
        language,
    ), quota in selector.IDENTITY_BUNDLE_CELL_QUOTAS.items():
        for _ in range(quota):
            bundle = f"identity:{identity_bundle_index:02d}"
            identity_bundle_index += 1
            for role in selector.OUTPUT_EXPERT_ROLES:
                ordinal += 1
                validation_role = "identity" if role == "tool" else role
                result.append(
                    _candidate(
                        ordinal,
                        kind="identity",
                        role=role,
                        validation_role=validation_role,
                        language=language,
                        bundle=bundle,
                        identity_class=identity_class,
                        parent_source_record_id_sha256=_hash_text(
                            f"identity-parent:{bundle}:{role}"
                        ),
                        identity_parent_class=identity_class,
                    )
                )
    for (label, language), quota in selector.ROUTER_CELL_QUOTAS.items():
        for index in range(quota):
            ordinal += 1
            result.append(
                _candidate(
                    ordinal,
                    kind="router",
                    role="router",
                    validation_role="router",
                    language=language,
                    bundle=f"router:{label}:{language}:{index}",
                    router_label=label,
                )
            )
    assert ordinal == 320
    return result


def _authorize_synthetic_selector(
    monkeypatch: pytest.MonkeyPatch,
    candidates: list[selector.SelectorCandidate]
    | tuple[selector.SelectorCandidate, ...],
) -> None:
    families = sorted(
        {str(item.tool_family) for item in candidates if item.tool_family is not None}
    )
    review_rows = selector._review_availability_rows(
        candidates,
        families,
    )
    assignment = []
    for row in review_rows:
        faults = list(row["fail_fault_types"])
        assignment.append(
            {
                "tool_family": row["tool_family"],
                "language": row["language"],
                "review_verdict": "fail" if faults else "pass",
                "review_fault_type": faults[0] if faults else None,
            }
        )
    identity_bundles: dict[str, list[selector.SelectorCandidate]] = {}
    router = Counter()
    for item in candidates:
        if item.selection_kind == "identity":
            identity_bundles.setdefault(
                item.task_bundle_sha256,
                [],
            ).append(item)
        elif item.selection_kind == "router":
            router[(str(item.router_label), item.language)] += 1
    parent = Counter()
    target = Counter()
    for rows in identity_bundles.values():
        first = rows[0]
        parent[(str(first.identity_parent_class), first.language)] += 1
        target[(str(first.identity_class), first.language)] += 1
    source_availability = {
        "review_cells": review_rows,
        "identity_parent_bundle_cells": [
            {
                "identity_class": identity_class,
                "language": language,
                "count": int(parent[(identity_class, language)]),
            }
            for identity_class in batch.IDENTITY_CLASSES
            for language in selector.LANGUAGES
        ],
        "identity_target_bundle_cells": [
            {
                "identity_class": identity_class,
                "language": language,
                "count": int(target[(identity_class, language)]),
            }
            for identity_class in batch.IDENTITY_CLASSES
            for language in selector.LANGUAGES
        ],
        "router_cells": [
            {
                "router_label": label,
                "language": language,
                "count": int(router[(label, language)]),
            }
            for label in ("humor", "serious", "angry")
            for language in selector.LANGUAGES
        ],
    }
    monkeypatch.setattr(
        selector,
        "EXPECTED_REVIEW_AVAILABILITY_ROOT_SHA256",
        selector._hash(review_rows),
    )
    monkeypatch.setattr(
        selector,
        "EXPECTED_REVIEW_ASSIGNMENT_SHA256",
        selector._hash(assignment),
    )
    monkeypatch.setattr(
        selector,
        "EXPECTED_SOURCE_AVAILABILITY_ROOT_SHA256",
        selector._hash(source_availability),
    )


def test_online_append_uses_held_tip_without_history_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _, job, _ = _state(tmp_path)
    calls = 0
    original = state._read_entries

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(state, "_read_entries", counted)
    asyncio.run(_dispatch(state, job))
    assert calls == 0
    assert state.cursor.sequence == 2
    assert state.full_replay_counts == {"startup": 1}


def test_provider_throughput_updates_at_terminals_and_idle_is_stable() -> None:
    now = [0.0]
    observed = ["2026-07-29T00:00:00Z"]
    accumulator = vnext.ProviderThroughputAccumulator(
        monotonic_clock=lambda: now[0],
        observed_at_clock=lambda: observed[0],
    )
    now[0] = 2.0
    first = accumulator.observe_terminal(
        {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    )
    assert first["provider_input_tokens_per_second"] == 5.0
    assert first["provider_output_tokens_per_second"] == 2.0
    assert first["terminal_jobs"] == 1
    observed[0] = "2026-07-29T00:00:02Z"
    now[0] = 4.0
    second = accumulator.observe_terminal(
        {"input_tokens": 6, "output_tokens": 2, "total_tokens": 8}
    )
    assert second["provider_input_tokens_per_second"] == 4.0
    assert second["provider_output_tokens_per_second"] == 1.5
    assert second["terminal_jobs"] == 2
    now[0] = 59.0
    observed[0] = "2026-07-29T00:00:59Z"
    assert accumulator.snapshot() == second


def test_provider_throughput_missing_usage_is_unknown_fail_closed() -> None:
    now = [0.0]
    accumulator = vnext.ProviderThroughputAccumulator(
        monotonic_clock=lambda: now[0],
        observed_at_clock=lambda: "2026-07-29T00:00:00Z",
    )
    now[0] = 1.0
    telemetry = accumulator.observe_terminal({"input_tokens": 1, "output_tokens": 1})
    assert telemetry["provider_input_tokens_per_second"] == "UNKNOWN"
    assert telemetry["provider_output_tokens_per_second"] == "UNKNOWN"
    assert telemetry["exact"] is False
    assert telemetry["exact_rows"] == 0
    assert telemetry["unknown_rows"] == 1
    assert telemetry["error"] == "provider_usage_missing_or_invalid"


def test_provider_throughput_accepts_provider_total_overhead() -> None:
    now = [0.0]
    accumulator = vnext.ProviderThroughputAccumulator(
        monotonic_clock=lambda: now[0],
        observed_at_clock=lambda: "2026-07-29T00:00:00Z",
    )
    now[0] = 2.0
    telemetry = accumulator.observe_terminal(
        {"input_tokens": 10, "output_tokens": 4, "total_tokens": 17}
    )
    assert telemetry["provider_input_tokens_per_second"] == 5.0
    assert telemetry["provider_output_tokens_per_second"] == 2.0
    assert telemetry["exact"] is True
    assert telemetry["unknown_rows"] == 0


def test_provider_throughput_rejects_provider_usage_under_total() -> None:
    now = [0.0]
    accumulator = vnext.ProviderThroughputAccumulator(
        monotonic_clock=lambda: now[0],
        observed_at_clock=lambda: "2026-07-29T00:00:00Z",
    )
    now[0] = 1.0
    telemetry = accumulator.observe_terminal(
        {"input_tokens": 10, "output_tokens": 4, "total_tokens": 13}
    )
    assert telemetry["provider_input_tokens_per_second"] == "UNKNOWN"
    assert telemetry["provider_output_tokens_per_second"] == "UNKNOWN"
    assert telemetry["exact"] is False
    assert telemetry["unknown_rows"] == 1


def test_provider_terminal_physically_updates_before_commit_without_double_count(
    tmp_path: Path,
) -> None:
    now = [0.0]
    state, inventory, job, key = _state(
        tmp_path,
        monotonic_clock=lambda: now[0],
        observed_at_clock=lambda: "2026-07-29T00:00:02Z",
    )
    asyncio.run(_dispatch(state, job))
    now[0] = 2.0
    state.observe_provider_terminal_usage(
        {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    )
    before_commit = json.loads(state.throughput_path.read_text(encoding="utf-8"))
    assert state.replay().completed == set()
    state.commit_group(
        [_pending(state, inventory, job, key)],
        inventory=inventory,
        phase="smoke_exact1",
        expected_job_keys=[job.idempotency_key],
    )
    telemetry = json.loads(state.throughput_path.read_text(encoding="utf-8"))
    assert telemetry == before_commit
    assert telemetry["provider_input_tokens_per_second"] == 5.5
    assert telemetry["provider_output_tokens_per_second"] == 3.5
    assert telemetry["terminal_jobs"] == 1
    assert telemetry["exact"] is True
    status = state.write_status(
        inventory=inventory,
        jobs=(job,),
        phase_jobs=(job,),
        phase="smoke_exact1",
        state_name="running",
        started_monotonic=None,
    )
    assert status["provider_throughput"] == {
        key: telemetry[key] for key in status["provider_throughput"]
    }
    assert status["provider_throughput"]["observed_at"] == ("2026-07-29T00:00:02Z")
    serialized = json.dumps(telemetry, sort_keys=True)
    assert "synthetic input" not in serialized


def test_execute_job_first_concurrent_provider_terminal_is_physical_before_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    release_second = asyncio.Event()
    status_path = tmp_path / "status.json"
    telemetry_path = tmp_path / "provider-throughput.json"
    batch._atomic_write_json(status_path, {"succeeded": 0, "rejected": 0})

    class TerminalState:
        def __init__(self) -> None:
            self._throughput = vnext.ProviderThroughputAccumulator(
                monotonic_clock=lambda: now[0],
                observed_at_clock=lambda: f"2026-07-29T00:00:0{int(now[0])}Z",
            )
            self._selection_views: dict[str, dict[str, Any]] = {}
            self._job_provenance: dict[str, Any] = {}

        async def reserve_budget(self, *args: Any, **kwargs: Any) -> None:
            return None

        async def append_event(self, *args: Any, **kwargs: Any) -> None:
            return None

        def _observe_stop_probe(self) -> None:
            return None

        def observe_provider_terminal_usage(
            self,
            usage: Mapping[str, Any] | None,
        ) -> None:
            batch._atomic_write_json(
                telemetry_path,
                {
                    **self._throughput.observe_terminal(usage),
                    "controller_run_id": "9" * 32,
                },
            )

    class ImmediateTeacher:
        max_retries = 0
        provider_provenance = {
            "completion": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                },
                "response_id": None,
            },
            "attempts": {
                "wire_attempts": 1,
                "retry_count": 0,
                "retry_reasons": [],
            },
        }

        async def complete(self, **kwargs: Any) -> str:
            now[0] = 2.0
            return "{}"

    class DelayedTeacher:
        max_retries = 0
        provider_provenance = {
            "completion": {
                "usage": {
                    "input_tokens": 6,
                    "output_tokens": 2,
                    "total_tokens": 8,
                },
                "response_id": None,
            },
            "attempts": {
                "wire_attempts": 1,
                "retry_count": 0,
                "retry_reasons": [],
            },
        }

        async def complete(self, **kwargs: Any) -> str:
            await release_second.wait()
            return "{}"

    first_job = _job(_source(role="serious", ordinal=1), ordinal=1)
    second_job = _job(_source(role="humor", ordinal=2), ordinal=2)
    state = TerminalState()
    state._selection_views = {
        first_job.idempotency_key: _view(first_job, tool_family=None),
        second_job.idempotency_key: _view(second_job, tool_family=None),
    }
    runner = object.__new__(vnext.VNextBatchRunner)
    runner.state = state
    runner.teachers = {
        "serious": ImmediateTeacher(),
        "humor": DelayedTeacher(),
    }
    runner._selection_views = state._selection_views
    runner._identity_derivative_specs = {}
    runner.inventory = _inventory()
    runner.config = SimpleNamespace()
    runner._job_receipt = lambda job, **kwargs: {
        "idempotency_key": job.idempotency_key,
        "outcome": kwargs["outcome"],
    }
    monkeypatch.setattr(
        vnext,
        "authenticated_job_prompts",
        lambda *args, **kwargs: ("system", "user"),
    )

    def reject_candidate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise batch.AdapterError("vnext_closed_grammar_keys_invalid")

    monkeypatch.setattr(vnext, "fast_parse_candidate_output", reject_candidate)

    async def exercise() -> None:
        replay = batch.ReplayState()
        first = asyncio.create_task(
            runner._execute_job(
                first_job,
                phase="bulk",
                replay=replay,
                retry_limit=0,
            )
        )
        second = asyncio.create_task(
            runner._execute_job(
                second_job,
                phase="bulk",
                replay=replay,
                retry_limit=0,
            )
        )
        first_pending = await first
        assert first_pending is not None
        assert first_pending.terminal_state == "rejected"
        assert not second.done()
        first_telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        assert first_telemetry["terminal_jobs"] == 1
        assert first_telemetry["provider_input_tokens_per_second"] == 5.0
        assert json.loads(status_path.read_text(encoding="utf-8")) == {
            "rejected": 0,
            "succeeded": 0,
        }
        now[0] = 4.0
        release_second.set()
        second_pending = await second
        assert second_pending is not None
        rolled = json.loads(telemetry_path.read_text(encoding="utf-8"))
        assert rolled["terminal_jobs"] == 2
        assert rolled["provider_input_tokens_per_second"] == 4.0
        assert rolled["provider_output_tokens_per_second"] == 1.5
        assert json.loads(status_path.read_text(encoding="utf-8"))["succeeded"] == 0

    asyncio.run(exercise())


def test_authenticated_checkpoint_restart_keeps_throughput_unknown(
    tmp_path: Path,
) -> None:
    now = [0.0]
    state, inventory, job, key = _state(
        tmp_path,
        monotonic_clock=lambda: now[0],
        observed_at_clock=lambda: "2026-07-29T00:00:02Z",
    )
    asyncio.run(_dispatch(state, job))
    now[0] = 2.0
    state.observe_provider_terminal_usage(
        {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    )
    state.commit_group(
        [_pending(state, inventory, job, key)],
        inventory=inventory,
        phase="smoke_exact1",
        expected_job_keys=[job.idempotency_key],
    )
    with pytest.raises(
        batch.AdapterError,
        match="vnext cross process resume forbidden",
    ):
        _state(
            tmp_path,
            monotonic_clock=lambda: now[0],
            observed_at_clock=lambda: "2026-07-29T00:00:03Z",
        )
    telemetry = json.loads(state.throughput_path.read_text(encoding="utf-8"))
    assert telemetry["provider_input_tokens_per_second"] == "UNKNOWN"
    assert telemetry["provider_output_tokens_per_second"] == "UNKNOWN"
    assert telemetry["error"] == "cross_process_resume_forbidden"
    assert telemetry["exact"] is False


def test_vnext_telemetry_snapshot_is_single_read_body_free_and_preserves_unknown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = [0.0]
    accumulator = vnext.ProviderThroughputAccumulator(
        monotonic_clock=lambda: now[0],
        observed_at_clock=lambda: "2026-07-29T00:00:01Z",
    )
    now[0] = 1.0
    telemetry = accumulator.observe_terminal({"input_tokens": 1, "output_tokens": 1})
    telemetry_path = tmp_path / "vnext_provider_throughput_v1.json"
    batch._atomic_write_json(
        telemetry_path,
        {**telemetry, "controller_run_id": "a" * 32},
    )
    before_bytes = telemetry_path.read_bytes()
    before_stat = telemetry_path.stat()
    assert vnext.main(["--telemetry-snapshot", str(telemetry_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    after_stat = telemetry_path.stat()
    assert telemetry_path.read_bytes() == before_bytes
    assert (after_stat.st_size, after_stat.st_mtime_ns) == (
        before_stat.st_size,
        before_stat.st_mtime_ns,
    )
    assert report["mode"] == "read_only_single_snapshot"
    assert report["live_reload"] is False
    assert report["write_operations"] == 0
    assert report["provider_input_tokens_per_second"] == "UNKNOWN"
    assert report["provider_output_tokens_per_second"] == "UNKNOWN"
    assert report["exact"] is False
    assert report["unknown_rows"] == 1
    assert report["error"] == "provider_usage_missing_or_invalid"
    assert "controller_run_id" not in report
    serialized = json.dumps(report, sort_keys=True)
    assert "teacher_input" not in serialized
    assert "Bearer " not in serialized
    assert "authorization" not in serialized.lower()


def test_public_dry_run_leads_with_physical_consumer_code_blocker() -> None:
    policy = _policy()
    blocked_consumer = {
        **policy.consumer_code_P,
        "status": "blocked_uncommitted_unreviewed",
        "blocker_reason_code": "consumer_code_P_uncommitted_unreviewed",
        "commit_sha1": None,
        "tree_sha1": None,
        "file_inventory_root_sha256": None,
        "implementation_sha256": None,
        "schema_set_root_sha256": None,
        "contract_sha256": None,
        "independently_audited": False,
    }
    report = vnext.public_dry_run(replace(policy, consumer_code_P=blocked_consumer))
    assert report["blockers"] == [
        "consumer_code_P_uncommitted_unreviewed",
        "vnext_transition_evidence_required",
    ]
    assert report["consumer_code_P_ready"] is False
    assert report["consumer_integration_ready"] is False
    assert report["provider_requests"] == 0
    assert report["gpu_requests"] == 0
    assert report["training_authorized"] is False


def test_clean_L23_default_load_and_dry_run_need_no_legacy_consumer_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "clean-L23"
    repo.mkdir()
    _git_test(repo, "init", "-q")
    _git_test(repo, "config", "user.name", "vnext-test")
    _git_test(repo, "config", "user.email", "vnext@example.invalid")

    config_relative = (
        "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.json"
    )
    source_config = json.loads(
        (REPO_ROOT / config_relative).read_text(encoding="utf-8")
    )

    def copy_from_source(relative: str) -> None:
        source = REPO_ROOT / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    modified_paths = {
        relative
        for relative, status in vnext.PRODUCER_L_BASE_DIFF_STATUS
        if status == "M"
    }
    for relative in sorted(modified_paths):
        committed = subprocess.run(
            ["git", "show", f"HEAD^:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(committed)
        base_paths = {
            str(source_config["launch"]["base_profile_path"]),
            str(source_config["teacher_record_schema_path"]),
            *(str(item["path"]) for item in source_config["closed_ancestors"]),
        }
    for relative in sorted(base_paths - modified_paths):
        copy_from_source(relative)
    _git_test(repo, "add", "--", ".")
    _git_test(repo, "commit", "-q", "-m", "audited base B")
    audited_base = _git_test(repo, "rev-parse", "HEAD")

    for relative, _status in vnext.PRODUCER_L_BASE_DIFF_STATUS:
        copy_from_source(relative)
    _git_test(repo, "add", "--", ".")
    _git_test(repo, "commit", "-q", "-m", "producer launch L23")
    producer_L = _git_test(repo, "rev-parse", "HEAD")
    assert _git_test(repo, "status", "--porcelain=v1", "--untracked-files=all") == ""
    _raw, observed = vnext._git_name_status_diff(
        repo_root=repo,
        parent_commit=audited_base,
        child_commit=producer_L,
        reason="test_clean_L23_diff_invalid",
    )
    assert observed == vnext.PRODUCER_L_BASE_DIFF_STATUS

    legacy_paths = (
        "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
        "minimal320_consumer_binding_v1.json",
        "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
        "minimal320_consumer_binding_v1.schema.json",
        "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
        "minimal320_contract_v1.json",
        "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
        "minimal320_contract_v1_source_status.json",
        "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
        "minimal320_metadata_overlay_v1.schema.json",
        "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
        "minimal320_selector_v1.schema.json",
        "src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_alignment_"
        "minimal320_consumer_v1.py",
    )
    assert not any((repo / relative).exists() for relative in legacy_paths)

    monkeypatch.setattr(batch, "REPO_ROOT", repo)
    monkeypatch.setattr(
        vnext,
        "IMPLEMENTATION_PATH",
        repo / "src/anchor_mvp/data/"
        "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
    )
    monkeypatch.setattr(vnext, "DEFAULT_CONFIG_PATH", repo / config_relative)
    original_physical_read = vnext._physical_read
    physical_reads: list[Path] = []

    def tracked_physical_read(path: Path, **kwargs: Any) -> bytes:
        physical_reads.append(path.absolute())
        return original_physical_read(path, **kwargs)

    monkeypatch.setattr(vnext, "_physical_read", tracked_physical_read)
    default_args = vnext._parser().parse_args(["--dry-run"])
    assert default_args.config == repo / config_relative
    policy = vnext.load_vnext_policy(default_args.config)
    assert policy.consumer_code_P == {
        "stage": "consumer_code_P",
        "status": "audited_compatible",
        "blocker_reason_code": None,
        "commit_sha1": "dc3d0da0adca42e52e2e122996e3480f2a285065",
        "tree_sha1": "cad8e848b7bb544e3cb97382df835ca435c73307",
        "file_inventory_root_sha256": (
            "7143287c813cc0843e7d079187b242eee8ff2ffcdb8a9a6dfc3efb579d9a5666"
        ),
        "implementation_sha256": (
            "84eebbd37c53d032ac3a24210117e295dca8a4e32ef2628381cc027a0cc3af23"
        ),
        "schema_set_root_sha256": (
            "3f0f18ff5a9325532980fc68bf0a517bddace5d34d8c7810c77289a024fe4c16"
        ),
        "contract_sha256": (
            "dc1f7783449c6c35b70664426931281a67ca4143e16cf3cb2fbec6b756dae800"
        ),
        "router_record_cells": {
            "humor": {"en": 14, "zh": 13},
            "serious": {"en": 13, "zh": 13},
            "angry": {"en": 13, "zh": 14},
        },
        "independently_audited": True,
        "content_retained": False,
    }
    assert policy.claims["live_authorized"] is True
    assert policy.claims["training_authorized"] is False
    assert policy.claims["formal_training_authorized"] is False
    assert not {(repo / relative).absolute() for relative in legacy_paths}.intersection(
        physical_reads
    )
    report = vnext.public_dry_run(policy)
    assert report["blockers"] == ["vnext_transition_evidence_required"]
    assert report["provider_requests"] == 0
    assert report["gpu_requests"] == 0
    assert report["consumer_code_P_ready"] is True
    assert report["consumer_integration_ready"] is False
    assert report["training_authorized"] is False
    assert report["formal_training_authorized"] is False

    output_root = tmp_path / "must-not-exist" / "run"
    credential_reads = 0

    def forbidden_credential_read() -> bytes:
        nonlocal credential_reads
        credential_reads += 1
        pytest.fail("missing transition evidence must fail before credential read")

    with pytest.raises(
        batch.AdapterError,
        match="vnext transition evidence missing",
    ):
        asyncio.run(
            vnext.execute_vnext_from_process_channel(
                policy,
                tmp_path / "transition-must-not-be-read.json",
                "a" * 64,
                output_root,
                forbidden_credential_read,
                environ={},
            )
        )
    assert credential_reads == 0
    assert not output_root.parent.exists()


def test_online_tip_guard_and_final_full_replay_have_distinct_scope(
    tmp_path: Path,
) -> None:
    state, _, job, _ = _state(tmp_path)
    asyncio.run(_dispatch(state, job))
    first = state.wal_dir / "0000000000000000.json"
    value = json.loads(first.read_text(encoding="utf-8"))
    value["body"]["event"]["reason_code"] = "tampered_old_entry"
    first.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    state.authenticate_authority()
    with pytest.raises(batch.AdapterError):
        state.full_replay_audit(stage="final_freeze")


def test_group_checkpoint_authenticates_stratum_and_stop_latches_after_sidecar(
    tmp_path: Path,
) -> None:
    state, inventory, job, key = _state(tmp_path)
    asyncio.run(_dispatch(state, job))
    state.request_stop_after_group("operator_stop_requested")
    assert state.stop_latch.requested is True
    assert state.stop_latch.effective is False
    with pytest.raises(vnext.StopAfterGroup):
        state.commit_group(
            [_pending(state, inventory, job, key)],
            inventory=inventory,
            phase="smoke_exact1",
        )
    checkpoint_path = next(state.checkpoint_dir.glob("*.json"))
    sidecar_path = checkpoint_path.with_name(f"{checkpoint_path.name}.sha256")
    assert sidecar_path.is_file()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["selector_identity_sha256"] == "f" * 64
    assert checkpoint["selected_set_count"] == 320
    assert checkpoint["counts"] == {
        "requested": 1,
        "completed": 1,
        "accepted": 1,
        "rejected": 0,
        "quarantine": 0,
    }
    audit = state.full_replay_audit(stage="final_freeze")
    assert audit.stage == "final_freeze"
    assert audit.terminal_events == 1
    assert audit.phase_receipts == 0
    assert audit.chain_tip_sha256 == state.cursor.chain_tip_sha256
    assert checkpoint["delta_inventory"][0]["stratum"] == {
        "role": "serious",
        "validation_contract_role": "serious",
        "language": "en",
        "tool_family": "calendar.lookup",
        "identity_class": None,
        "identity_parent_class": None,
        "selection_kind": "full_core",
        "review_verdict": None,
        "review_fault_type": None,
        "router_label": None,
        "source_record_id_sha256": _hash_text(job.source.record_id),
    }
    assert state.stop_latch.effective is True
    assert not state.replay().uncertain
    serialized = json.dumps(checkpoint, sort_keys=True)
    assert "synthetic input" not in serialized
    assert "raw_token_ids" not in serialized or (
        checkpoint["raw_token_ids_retained"] is False
    )


def test_receipt_schema_closes_provenance_lineage_usage_attempts_and_stratum(
    tmp_path: Path,
) -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    state, inventory, job, key = _state(tmp_path)
    receipt = _pending(state, inventory, job, key).receipt
    schema = json.loads(state.policy.receipt_schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(receipt)
    mutations = []
    for field in ("planner_lineage", "usage", "attempts", "stratum"):
        mutated = json.loads(json.dumps(receipt))
        mutated[field]["unexpected"] = False
        mutations.append(mutated)
    mutated = json.loads(json.dumps(receipt))
    mutated["provenance"]["runtime"]["unexpected"] = "0" * 64
    mutations.append(mutated)
    for mutated in mutations:
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(mutated)


def test_final_freeze_reloads_inventories_and_counts_only_validated_accepted(
    tmp_path: Path,
) -> None:
    checkpoint_schema_path = (
        REPO_ROOT / "configs/data/"
        "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_"
        "checkpoint_v1.schema.json"
    )
    policy = replace(
        _policy(),
        checkpoint_schema_sha256=hashlib.sha256(
            checkpoint_schema_path.read_bytes()
        ).hexdigest(),
    )
    state, inventory, job, key = _state(
        tmp_path,
        policy=policy,
        final_quality_validator=lambda *_args, **_kwargs: None,
    )
    asyncio.run(_dispatch(state, job))
    state.commit_group(
        [_pending(state, inventory, job, key)],
        inventory=inventory,
        phase="smoke_exact1",
        expected_job_keys=[job.idempotency_key],
    )
    transition_payload = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2."
            "teacher-alignment-vnext.terminal-transition-recheck.v1"
        ),
        "transition_evidence_sha256": "1" * 64,
        "policy_revalidation_sha256": "2" * 64,
        "old_controller_pid": 123,
        "old_archive_identity_sha256": "3" * 64,
        "old_archive_file_count": 1,
        "controller_count": 0,
        "candidate_controller_count": 1,
        "candidate_launch_process_count": 1,
        "candidate_controller_pid": os.getpid(),
        "controller_inventory_sha256": "4" * 64,
        "all_matching_inventory_sha256": "5" * 64,
        "canonical_root_absent": True,
        "vnext_root_present": True,
        "content_retained": False,
    }
    transition = {
        **transition_payload,
        "recheck_sha256": hashlib.sha256(
            vnext._canonical_bytes(transition_payload)
        ).hexdigest(),
    }
    freeze = state.freeze_handoff(
        inventory=inventory,
        jobs=(job,),
        terminal_transition_recheck=transition,
    )
    assert freeze["counts"] == {
        "requested": 1,
        "completed": 1,
        "accepted": 1,
        "rejected": 0,
        "quarantine": 0,
    }
    assert freeze["output_expert_role_counts"] == {"serious": 1}
    assert freeze["language_counts"] == {"en": 1}
    assert freeze["selection_counts"]["selected"] == 320
    assert freeze["inventory_bindings"]["accepted"]["count"] == 1
    assert freeze["inventory_bindings"]["rejected"]["count"] == 0
    assert freeze["inventory_bindings"]["quarantine"]["count"] == 0
    staging = state.stage_external_attestation(freeze)
    assert (state.freeze_dir / "producer_final_binding_candidate.json").is_file()
    assert staging["producer_final_binding_candidate_sha256"] == (
        vnext._authenticated_file_digest(
            state.freeze_dir / "producer_final_binding_candidate.json",
            stop=state.freeze_dir,
            reason="test_final_binding_candidate_drift",
        )
    )


def test_between_group_kill_switch_uses_prior_durable_checkpoint(
    tmp_path: Path,
) -> None:
    empty, _, _, _ = _state(tmp_path / "empty")
    with pytest.raises(
        batch.AdapterError,
        match="zero group kill switch hard block",
    ):
        empty.stop_at_durable_group_boundary("kill_switch_armed")

    state, inventory, job, key = _state(tmp_path / "durable")
    asyncio.run(_dispatch(state, job))
    state.commit_group(
        [_pending(state, inventory, job, key)],
        inventory=inventory,
        phase="smoke_exact1",
        expected_job_keys=[job.idempotency_key],
    )
    prior = state._prior_checkpoint_sha256
    with pytest.raises(vnext.StopAfterGroup):
        state.stop_at_durable_group_boundary("kill_switch_armed")
    assert state.stop_latch.effective is True
    assert state.stop_latch.effective_checkpoint_sha256 == prior


def test_execute_consumer_code_P_gate_is_zero_output(
    tmp_path: Path,
) -> None:
    output_parent = tmp_path / "must-not-exist"
    policy = replace(
        _policy(),
        output_root_parent=output_parent,
        claims={
            "candidate": True,
            "live_authorized": True,
            "training_authorized": False,
        },
        consumer_code_P={
            "stage": "consumer_code_P",
            "status": "blocked_uncommitted_unreviewed",
            "blocker_reason_code": ("consumer_code_P_uncommitted_unreviewed"),
            "commit_sha1": None,
            "tree_sha1": None,
            "file_inventory_root_sha256": None,
            "implementation_sha256": None,
            "schema_set_root_sha256": None,
            "contract_sha256": None,
            "router_record_cells": {
                "humor": {"en": 14, "zh": 13},
                "serious": {"en": 13, "zh": 13},
                "angry": {"en": 13, "zh": 14},
            },
            "independently_audited": False,
            "content_retained": False,
        },
    )
    with pytest.raises(
        batch.AdapterError,
        match="consumer code P uncommitted unreviewed",
    ):
        asyncio.run(
            vnext.execute_vnext_from_process_channel(
                policy,
                tmp_path / "transition.json",
                "a" * 64,
                output_parent / "run",
                lambda: pytest.fail("credential channel must not be read"),
                environ={},
            )
        )
    assert not output_parent.exists()


def _git_test(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_live_window_P32_materializer_emits_real_teacher_shard_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "clean-worktree"
    runtime = tmp_path / "external-runtime"
    repo.mkdir()
    runtime.mkdir()
    _git_test(repo, "init", "-q")
    _git_test(repo, "config", "user.name", "vnext-test")
    _git_test(repo, "config", "user.email", "vnext@example.invalid")
    (repo / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    (repo / "AUDITED_BASE.txt").write_text("audited\n", encoding="utf-8")
    selector_schema_path = (
        repo / "configs/data/"
        "gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_selector_v2.schema.json"
    )
    teacher_schema_path = (
        repo / "configs/data/"
        "gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json"
    )
    selector_schema_path.parent.mkdir(parents=True)
    selector_schema_path.write_bytes(
        (
            REPO_ROOT / "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_alignment_"
            "minimal320_selector_v2.schema.json"
        ).read_bytes()
    )
    teacher_schema_path.write_bytes(
        (
            REPO_ROOT / "configs/data/"
            "gemma3_chat_unbalanced_v2_teacher_final_record_v2.schema.json"
        ).read_bytes()
    )
    _git_test(repo, "add", "--", ".")
    _git_test(repo, "commit", "-q", "-m", "audited B")
    audited_base = _git_test(repo, "rev-parse", "HEAD")
    (repo / ".gitattributes").write_text("*.json text eol=lf\n", encoding="utf-8")
    (repo / "launch.py").write_text("LAUNCH = 'L'\n", encoding="utf-8")
    _git_test(repo, "add", "--", ".gitattributes", "launch.py")
    _git_test(repo, "commit", "-q", "-m", "producer launch L")
    producer_L = _git_test(repo, "rev-parse", "HEAD")
    producer_L_tree = _git_test(repo, "rev-parse", "HEAD^{tree}")
    l_proof = vnext._verify_audited_base_to_L(
        repo_root=repo,
        audited_base=audited_base,
        producer_L=producer_L,
        expected_path_status=(
            (".gitattributes", "M"),
            ("launch.py", "A"),
        ),
    )
    assert l_proof["producer_L_direct_child_sha1"] == producer_L

    def authenticated(path: Path, raw: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        path.with_name(f"{path.name}.sha256").write_bytes(
            f"{digest}  {path.name}\n".encode("ascii")
        )
        return path

    preimage_dir = runtime / "preimages"
    candidates = sorted(
        _minimal320_candidates(),
        key=lambda item: item.candidate_id_sha256,
    )
    selected_values = [candidate.as_dict() for candidate in candidates]
    selected_raw = b"".join(
        vnext._canonical_bytes(value, newline=True) for value in selected_values
    )
    selected_root = selector._hash(selected_values)
    selected_manifest_value = {
        **_selection_manifest(),
        "selector_identity_sha256": "f" * 64,
        "selected_set_root_sha256": selected_root,
    }
    overlay_rows = authenticated(
        preimage_dir / "authenticated_metadata_overlay.jsonl",
        b'{"overlay":true}\n',
    )
    overlay_manifest = authenticated(
        preimage_dir / "authenticated_metadata_overlay_manifest.json",
        b'{"overlay_manifest":true}\n',
    )
    candidate_rows = authenticated(
        preimage_dir / "candidate_universe.jsonl",
        b'{"candidate":true}\n',
    )
    selected_rows = authenticated(
        preimage_dir / "selected_candidates_320.jsonl",
        selected_raw,
    )
    selection_manifest = authenticated(
        preimage_dir / "selection_manifest.json",
        vnext._canonical_bytes(selected_manifest_value, newline=True),
    )
    selection_preimage = authenticated(
        preimage_dir / "selection_preimage.json",
        b'{"preimage":true}\n',
    )
    preimage_manifest = authenticated(
        preimage_dir / "selector_preimage_manifest.json",
        b'{"preimage_manifest":true}\n',
    )
    overlay = SimpleNamespace(
        directory=preimage_dir,
        rows_path=overlay_rows,
        manifest_path=overlay_manifest,
    )
    preimages = SimpleNamespace(
        directory=preimage_dir,
        candidate_rows_path=candidate_rows,
        selected_rows_path=selected_rows,
        selection_manifest_path=selection_manifest,
        selection_preimage_path=selection_preimage,
        preimage_manifest_path=preimage_manifest,
        manifest={
            "selected_set_root_sha256": selected_root,
        },
    )
    authenticated_selection = selector.Minimal320Selection(
        tuple(candidates),
        selected_manifest_value,
    )
    monkeypatch.setattr(
        selector,
        "authenticate_selector_preimages",
        lambda *args, **kwargs: (tuple(candidates), authenticated_selection),
    )
    checkpoint_dir = runtime / "checkpoints"
    authenticated(checkpoint_dir / "0000.json", b'{"checkpoint":0}\n')
    freeze_dir = runtime / "freeze"
    accepted_candidate = next(
        candidate
        for candidate in candidates
        if candidate.source_record_id_sha256 == _hash_text("record-1")
    )
    source = replace(
        _source(role="humor", ordinal=1),
        task_bundle="core:tool.family.00:en",
    )
    job = batch.AlignmentJob(
        source=source,
        idempotency_key=accepted_candidate.idempotency_key,
        prompt_template_sha256=_hash_text(batch._system_prompt(source.role)),
        output_schema_sha256="e" * 64,
    )
    runtime_record = {
        "idempotency_key": job.idempotency_key,
        "record_id": job.source.record_id,
        "task_bundle": job.source.task_bundle,
        "hidden_reasoning_retained": False,
        "source_binding": {
            "selected_candidate_sha256": (accepted_candidate.candidate_id_sha256),
            "source_record_id_sha256": (accepted_candidate.source_record_id_sha256),
            "source_content_sha256": (accepted_candidate.source_content_sha256),
            "source_serialization_identity_sha256": (
                accepted_candidate.source_serialization_identity_sha256
            ),
            "task_bundle_sha256": accepted_candidate.task_bundle_sha256,
            "training_asset": accepted_candidate.training_asset,
        },
        "output": {
            "kind": "natural_text",
            "value": "consumer-shaped teacher target",
        },
    }
    output_record_sha256 = batch._hash_object(runtime_record)
    hmac_key = b"h" * 32
    receipt_payload = {
        "id": _hash_text("accepted-receipt"),
        "idempotency_key": job.idempotency_key,
        "outcome": "succeeded",
        "output_record_sha256": output_record_sha256,
    }
    runtime_receipt = {
        **receipt_payload,
        "hmac_sha256": batch._receipt_hmac(receipt_payload, hmac_key),
    }
    accepted_row = {
        "idempotency_key": job.idempotency_key,
        "outcome": "accepted",
        "receipt_sha256": batch._hash_object(runtime_receipt),
        "output_record_sha256": output_record_sha256,
    }
    accepted_raw = vnext._canonical_bytes(accepted_row, newline=True)
    quality_rejected_candidate = next(
        candidate
        for candidate in candidates
        if candidate.idempotency_key != accepted_candidate.idempotency_key
    )
    quality_rejected_record = {
        **runtime_record,
        "idempotency_key": quality_rejected_candidate.idempotency_key,
    }
    quality_rejected_record_sha256 = batch._hash_object(quality_rejected_record)
    quality_rejected_receipt_payload = {
        "id": _hash_text("centralized-quality-rejected-receipt"),
        "idempotency_key": quality_rejected_candidate.idempotency_key,
        "outcome": "succeeded",
        "output_record_sha256": quality_rejected_record_sha256,
    }
    quality_rejected_receipt = {
        **quality_rejected_receipt_payload,
        "hmac_sha256": batch._receipt_hmac(
            quality_rejected_receipt_payload,
            hmac_key,
        ),
    }
    quality_rejected_row = {
        "idempotency_key": quality_rejected_candidate.idempotency_key,
        "outcome": "rejected",
        "reason_code": "final_quality_review_faults_invalid",
        "receipt_sha256": batch._hash_object(quality_rejected_receipt),
        "output_record_sha256": quality_rejected_record_sha256,
    }
    rejected_raw = vnext._canonical_bytes(quality_rejected_row, newline=True)
    authenticated(freeze_dir / "accepted_inventory.jsonl", accepted_raw)
    authenticated(freeze_dir / "quarantine_inventory.jsonl", b"")
    authenticated(freeze_dir / "rejected_inventory.jsonl", rejected_raw)
    authenticated(
        freeze_dir / "checkpoint_manifest.json",
        b'{"checkpoint_manifest":true}\n',
    )
    alignment = runtime / "alignment"
    automation = runtime / "automation"
    alignment.mkdir()
    automation.mkdir()
    (alignment / "records.jsonl").write_bytes(
        vnext._canonical_bytes(runtime_record, newline=True)
        + vnext._canonical_bytes(quality_rejected_record, newline=True)
    )
    (alignment / "receipts.jsonl").write_bytes(
        vnext._canonical_bytes(runtime_receipt, newline=True)
        + vnext._canonical_bytes(quality_rejected_receipt, newline=True)
    )
    for path in (
        alignment / "rejections.jsonl",
        automation / "events.jsonl",
        automation / "phase_receipts.jsonl",
    ):
        path.write_bytes(b"")
    (automation / "status.json").write_bytes(b"{}\n")
    (runtime / "dataset.json").write_bytes(b"{}\n")
    policy = SimpleNamespace(
        selector_schema_path=selector_schema_path,
        selector_schema_sha256=hashlib.sha256(
            selector_schema_path.read_bytes()
        ).hexdigest(),
        selector_contract_sha256="6" * 64,
        teacher_record_schema_path=teacher_schema_path,
        teacher_record_schema_sha256=hashlib.sha256(
            teacher_schema_path.read_bytes()
        ).hexdigest(),
    )
    state = SimpleNamespace(
        root=runtime,
        checkpoint_dir=checkpoint_dir,
        freeze_dir=freeze_dir,
        records_path=alignment / "records.jsonl",
        receipts_path=alignment / "receipts.jsonl",
        rejections_path=alignment / "rejections.jsonl",
        events_path=automation / "events.jsonl",
        phase_receipts_path=automation / "phase_receipts.jsonl",
        status_path=automation / "status.json",
        dataset_marker_path=runtime / "dataset.json",
        groups_dir=alignment / "groups",
        policy=policy,
        config=SimpleNamespace(),
        selection_manifest=selected_manifest_value,
        _selection_views={
            str(value["idempotency_key"]): value for value in selected_values
        },
        _jobs_by_key={job.idempotency_key: job},
        _identity_derivative_specs={},
        _final_quality_validator=lambda *args, **kwargs: None,
    )
    launch_L = {
        "stage": "producer_launch_L",
        "commit_sha1": producer_L,
        "tree_sha1": producer_L_tree,
        "global_clean_tree": True,
    }
    freeze_payload = {
        "state": "candidate_pending_external_attestation",
        "selected_set_count": 320,
        "selected_set_root_sha256": selected_root,
        "selector_identity_sha256": "f" * 64,
        "full_replay_counts": {"startup": 1, "final_freeze": 1},
        "checkpoint_manifest_root_sha256": "1" * 64,
        "final_chain_tip_sha256": "2" * 64,
        "final_wal_entries": 0,
        "counts": {
            "requested": 320,
            "completed": 2,
            "accepted": 1,
            "rejected": 1,
            "quarantine": 318,
        },
        "inventory_sha256": {
            "accepted": hashlib.sha256(accepted_raw).hexdigest(),
            "rejected": hashlib.sha256(rejected_raw).hexdigest(),
        },
        "inventory_bindings": {
            "accepted": {
                "sha256": hashlib.sha256(accepted_raw).hexdigest(),
                "inventory_root_sha256": vnext._merkle_root([accepted_row]),
                "count": 1,
                "outcome": "accepted",
            },
            "rejected": {
                "sha256": hashlib.sha256(rejected_raw).hexdigest(),
                "inventory_root_sha256": vnext._merkle_root([quality_rejected_row]),
                "count": 1,
                "outcome": "rejected",
            },
        },
        "accepted_strata_root_sha256": "4" * 64,
    }
    freeze = {
        **freeze_payload,
        "hmac_sha256": batch._receipt_hmac(freeze_payload, hmac_key),
    }
    authenticated(
        freeze_dir / "freeze_attestation.json",
        vnext._canonical_bytes(freeze, newline=True),
    )

    def verify_signed(
        value: Mapping[str, Any],
        *,
        reason: str,
    ) -> dict[str, Any]:
        payload = dict(value)
        observed = payload.pop("hmac_sha256", None)
        if observed != batch._receipt_hmac(payload, hmac_key):
            raise batch.AdapterError(reason)
        return payload

    state._verify_signed_value = verify_signed
    staging = {
        "state": "awaiting_independent_external_attestation",
        "staging_sha256": "3" * 64,
    }
    slots = batch.RuntimeSecretSlots(b"credential", hmac_key)
    lease_checks: list[bool] = []
    release_root = "release"
    p_paths = tuple(
        f"{release_root}/{name}" for name in vnext._PRODUCER_PAYLOAD_P_FILES
    )
    monkeypatch.setattr(batch, "REPO_ROOT", repo)
    receipt = vnext._materialize_producer_payload_P(
        repo_root=repo,
        release_root=release_root,
        producer_payload_P_paths=p_paths,
        runtime_slots=slots,
        lease_authenticate=lambda: lease_checks.append(slots.loaded),
        producer_launch_L=launch_L,
        overlay=overlay,
        preimages=preimages,
        state=state,
        inventory=_inventory(),
        freeze_attestation=freeze,
        staging=staging,
    )
    assert receipt["atomic_directory_publish"] is True
    assert len(receipt["files"]) == 32
    assert lease_checks and all(lease_checks)
    assert (
        tuple(
            sorted(
                path.relative_to(repo).as_posix()
                for path in (repo / "release").iterdir()
            )
        )
        == p_paths
    )
    assert not tuple(repo.glob(".release.staging-*"))
    teacher_rows = vnext._canonical_jsonl_rows(
        (repo / release_root / vnext.TEACHER_TRAINING_SHARD_NAME).read_bytes(),
        reason="test_teacher_shard_invalid",
    )
    training_rows = vnext._canonical_jsonl_rows(
        (repo / release_root / vnext.TRAINING_RECORD_INVENTORY_NAME).read_bytes(),
        reason="test_training_inventory_invalid",
    )
    assert len(teacher_rows) == len(training_rows) == 1
    assert teacher_rows[0]["teacher_target"] == "consumer-shaped teacher target"
    assert (
        training_rows[0]["teacher_record_sha256"]
        == hashlib.sha256(vnext._canonical_bytes(teacher_rows[0])).hexdigest()
    )
    committed_manifest = json.loads(
        (repo / release_root / "committed_shards_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert committed_manifest["schema_version"] == (
        vnext.COMMITTED_SHARDS_SCHEMA_VERSION
    )
    assert committed_manifest["file_count"] == 4
    assert committed_manifest["shard_count"] == 1
    assert committed_manifest["partial_accepted_subset"] is True
    assert committed_manifest["exact_requested_set_complete"] is False
    assert {row["path"] for row in committed_manifest["files"]} == {
        f"{release_root}/{vnext.TEACHER_TRAINING_SHARD_NAME}",
        f"{release_root}/{vnext.TEACHER_TRAINING_SHARD_NAME}.sha256",
        f"{release_root}/{vnext.TRAINING_RECORD_INVENTORY_NAME}",
        f"{release_root}/{vnext.TRAINING_RECORD_INVENTORY_NAME}.sha256",
    }
    slots.close()
    with pytest.raises(batch.AdapterError, match="runtime closed"):
        vnext._materialize_producer_payload_P(
            repo_root=repo,
            release_root=release_root,
            producer_payload_P_paths=p_paths,
            runtime_slots=slots,
            lease_authenticate=lambda: pytest.fail("closed slots must fail first"),
            producer_launch_L=launch_L,
            overlay=overlay,
            preimages=preimages,
            state=state,
            inventory=_inventory(),
            freeze_attestation=freeze,
            staging=staging,
        )
    _git_test(repo, "add", "--", release_root)
    _git_test(repo, "commit", "-q", "-m", "producer payload P")
    producer_P = _git_test(repo, "rev-parse", "HEAD")
    assert _git_test(repo, "rev-list", "--parents", "-n", "1", producer_P).split() == [
        producer_P,
        producer_L,
    ]
    _raw, observed = vnext._git_name_status_diff(
        repo_root=repo,
        parent_commit=producer_L,
        child_commit=producer_P,
        reason="test_P_diff_invalid",
    )
    assert observed == tuple((path, "A") for path in p_paths)


def test_p_runtime_join_cross_binds_posthoc_rejections_and_rejects_orphans() -> None:
    hmac_key = b"j" * 32

    def runtime_pair(label: str) -> tuple[dict[str, Any], dict[str, Any]]:
        key = _hash_text(f"key-{label}")
        record = {
            "idempotency_key": key,
            "record_id": f"record-{label}",
        }
        record_sha256 = batch._hash_object(record)
        payload = {
            "id": _hash_text(f"receipt-{label}"),
            "idempotency_key": key,
            "outcome": "succeeded",
            "output_record_sha256": record_sha256,
        }
        receipt = {
            **payload,
            "hmac_sha256": batch._receipt_hmac(payload, hmac_key),
        }
        return record, receipt

    accepted_record, accepted_receipt = runtime_pair("accepted")
    rejected_record, rejected_receipt = runtime_pair("posthoc-rejected")
    accepted = {
        "idempotency_key": accepted_record["idempotency_key"],
        "outcome": "accepted",
        "receipt_sha256": batch._hash_object(accepted_receipt),
        "output_record_sha256": batch._hash_object(accepted_record),
    }
    posthoc_rejected = {
        "idempotency_key": rejected_record["idempotency_key"],
        "outcome": "rejected",
        "reason_code": "final_quality_review_faults_invalid",
        "receipt_sha256": batch._hash_object(rejected_receipt),
        "output_record_sha256": batch._hash_object(rejected_record),
    }
    online_rejected = {
        "idempotency_key": _hash_text("online-rejected"),
        "outcome": "rejected",
        "reason_code": "vnext_closed_grammar_json_invalid",
        "receipt_sha256": _hash_text("online-rejection-receipt"),
        "output_record_sha256": None,
    }
    vnext._validate_p_runtime_join(
        accepted_rows=[accepted],
        rejected_rows=[posthoc_rejected, online_rejected],
        runtime_records=[accepted_record, rejected_record],
        runtime_receipts=[accepted_receipt, rejected_receipt],
        hmac_key=hmac_key,
    )

    orphan_record, orphan_receipt = runtime_pair("orphan")
    failing_inputs = (
        {
            "runtime_records": [accepted_record, rejected_record, orphan_record],
            "runtime_receipts": [accepted_receipt, rejected_receipt],
            "rejected_rows": [posthoc_rejected, online_rejected],
        },
        {
            "runtime_records": [accepted_record, rejected_record],
            "runtime_receipts": [accepted_receipt],
            "rejected_rows": [posthoc_rejected, online_rejected],
        },
        {
            "runtime_records": [accepted_record, rejected_record],
            "runtime_receipts": [accepted_receipt, rejected_receipt],
            "rejected_rows": [
                {**posthoc_rejected, "output_record_sha256": "0" * 64},
                online_rejected,
            ],
        },
        {
            "runtime_records": [accepted_record, rejected_record, orphan_record],
            "runtime_receipts": [
                accepted_receipt,
                rejected_receipt,
                orphan_receipt,
            ],
            "rejected_rows": [
                posthoc_rejected,
                {
                    **online_rejected,
                    "idempotency_key": orphan_record["idempotency_key"],
                },
            ],
        },
        {
            "runtime_records": [accepted_record, rejected_record],
            "runtime_receipts": [accepted_receipt, rejected_receipt],
            "rejected_rows": [
                {
                    **posthoc_rejected,
                    "reason_code": "vnext_closed_grammar_json_invalid",
                },
                online_rejected,
            ],
        },
    )
    for inputs in failing_inputs:
        with pytest.raises(batch.AdapterError, match="runtime join drift"):
            vnext._validate_p_runtime_join(
                accepted_rows=[accepted],
                hmac_key=hmac_key,
                **inputs,
            )


def _release_repo(
    root: Path,
    *,
    variant: str = "valid",
) -> tuple[Path, str, str, str, str, tuple[str, ...], tuple[str, ...]]:
    root.mkdir()
    _git_test(root, "init", "-q")
    _git_test(root, "config", "user.name", "vnext-test")
    _git_test(root, "config", "user.email", "vnext@example.invalid")
    (root / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    (root / "AUDITED_BASE.txt").write_text("audited\n", encoding="utf-8")
    _git_test(root, "add", "--", ".gitattributes", "AUDITED_BASE.txt")
    _git_test(root, "commit", "-q", "-m", "audited base")
    audited_base = _git_test(root, "rev-parse", "HEAD")
    (root / ".gitattributes").write_text("*.json text eol=lf\n", encoding="utf-8")
    (root / "code.py").write_text("L_CODE = True\n", encoding="utf-8")
    _git_test(root, "add", "--", ".gitattributes", "code.py")
    _git_test(root, "commit", "-q", "-m", "producer launch L")
    producer_L = _git_test(root, "rev-parse", "HEAD")
    p_paths = (
        "release/accepted.jsonl",
        "release/checkpoint.json",
    )
    r_paths = (
        "release/external.json",
        "release/freeze.json",
    )
    (root / "release").mkdir()
    (root / p_paths[0]).write_text("{}\n", encoding="utf-8")
    (root / p_paths[1]).write_text("{}\n", encoding="utf-8")
    if variant == "p_extra":
        (root / "UNREVIEWED_EXTRA").write_text(
            "not reviewed\n",
            encoding="utf-8",
        )
    if variant == "moved_stage":
        (root / r_paths[1]).write_text("{}\n", encoding="utf-8")
    _git_test(root, "add", "--", ".")
    _git_test(root, "commit", "-q", "-m", "producer P")
    producer_P = _git_test(root, "rev-parse", "HEAD")
    if variant == "non_direct":
        (root / "intermediate.txt").write_text(
            "intermediate\n",
            encoding="utf-8",
        )
        _git_test(root, "add", "--", "intermediate.txt")
        _git_test(root, "commit", "-q", "-m", "intermediate")
    (root / r_paths[0]).write_text("{}\n", encoding="utf-8")
    if variant != "omitted":
        (root / r_paths[1]).write_text(
            '{"freeze":true}\n',
            encoding="utf-8",
        )
    if variant == "extra":
        (root / "release/extra.json").write_text("{}\n", encoding="utf-8")
    if variant == "rename":
        (root / p_paths[0]).rename(root / "release/accepted-moved.jsonl")
    _git_test(root, "add", "-A", "--", ".")
    _git_test(root, "commit", "-q", "-m", "producer R")
    producer_R = _git_test(root, "rev-parse", "HEAD")
    if variant == "dirty":
        (root / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    return root, audited_base, producer_L, producer_P, producer_R, p_paths, r_paths


def test_producer_L_to_P_to_R_verifier_derives_exact_git_stages(
    tmp_path: Path,
) -> None:
    (
        repo,
        audited_base,
        producer_L,
        producer_P,
        producer_R,
        p_paths,
        r_paths,
    ) = _release_repo(tmp_path / "valid")
    proof = vnext._verify_producer_L_to_P_to_R(
        repo_root=repo,
        audited_base=audited_base,
        producer_L=producer_L,
        producer_P=producer_P,
        producer_R=producer_R,
        reviewed_paths=("code.py",),
        producer_payload_P_paths=p_paths,
        producer_R_release_paths=r_paths,
        release_root="release",
        producer_L_base_diff_status=(
            (".gitattributes", "M"),
            ("code.py", "A"),
        ),
    )
    assert proof["audited_base"]["commit_sha1"] == audited_base
    assert proof["audited_base"]["exact_global_diff_allowlist"] is True
    assert proof["unique_direct_children"] is True
    assert proof["exact_raw_diff_allowlist"] is True
    assert {
        row["release_stage"] for row in proof["producer_launch_L"]["file_inventory"]
    } == {"producer_launch_L"}
    assert {
        row["release_stage"] for row in proof["producer_payload_P"]["file_inventory"]
    } == {"producer_payload_P"}
    assert {
        row["release_stage"] for row in proof["producer_final_R"]["file_inventory"]
    } == {"producer_final_R"}
    assert proof["release_self_pin_present"] is False


@pytest.mark.parametrize(
    "variant,reason",
    [
        ("extra", "diff allowlist"),
        ("omitted", "diff allowlist"),
        ("moved_stage", "diff allowlist"),
        ("non_direct", "not unique direct child"),
        ("dirty", "dirty tree"),
        ("rename", "diff allowlist"),
        ("p_extra", "diff allowlist"),
    ],
)
def test_producer_L_to_P_to_R_verifier_rejects_release_drift(
    tmp_path: Path,
    variant: str,
    reason: str,
) -> None:
    (
        repo,
        audited_base,
        producer_L,
        producer_P,
        producer_R,
        p_paths,
        r_paths,
    ) = _release_repo(tmp_path / variant, variant=variant)
    with pytest.raises(batch.AdapterError, match=reason):
        vnext._verify_producer_L_to_P_to_R(
            repo_root=repo,
            audited_base=audited_base,
            producer_L=producer_L,
            producer_P=producer_P,
            producer_R=producer_R,
            reviewed_paths=("code.py",),
            producer_payload_P_paths=p_paths,
            producer_R_release_paths=r_paths,
            release_root="release",
            producer_L_base_diff_status=(
                (".gitattributes", "M"),
                ("code.py", "A"),
            ),
        )


def test_producer_L_to_P_to_R_verifier_rejects_pathspec_escape(
    tmp_path: Path,
) -> None:
    (
        repo,
        audited_base,
        producer_L,
        producer_P,
        producer_R,
        p_paths,
        r_paths,
    ) = _release_repo(tmp_path / "pathspec")
    with pytest.raises(
        batch.AdapterError,
        match="exact path set invalid",
    ):
        vnext._verify_producer_L_to_P_to_R(
            repo_root=repo,
            audited_base=audited_base,
            producer_L=producer_L,
            producer_P=producer_P,
            producer_R=producer_R,
            reviewed_paths=("code.py",),
            producer_payload_P_paths=p_paths,
            producer_R_release_paths=r_paths,
            release_root="../release",
            producer_L_base_diff_status=(
                (".gitattributes", "M"),
                ("code.py", "A"),
            ),
        )


def _materializer_repo(
    root: Path,
    *,
    same_reviewer: bool = False,
) -> tuple[
    Path,
    tuple[str, ...],
    tuple[str, ...],
    Path,
    str,
    Path,
    str,
]:
    root.mkdir()
    _git_test(root, "init", "-q")
    _git_test(root, "config", "user.name", "vnext-test")
    _git_test(root, "config", "user.email", "vnext@example.invalid")
    (root / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
    (root / "AUDITED_BASE.txt").write_text("audited\n", encoding="utf-8")
    _git_test(root, "add", "--", ".")
    _git_test(root, "commit", "-q", "-m", "audited B")
    audited_base = _git_test(root, "rev-parse", "HEAD")
    audited_tree = _git_test(root, "rev-parse", "HEAD^{tree}")
    (root / ".gitattributes").write_text("*.json text eol=lf\n", encoding="utf-8")
    (root / "code.py").write_text("LAUNCH = 'L'\n", encoding="utf-8")
    _git_test(root, "add", "--", ".gitattributes", "code.py")
    _git_test(root, "commit", "-q", "-m", "producer launch L")
    producer_L = _git_test(root, "rev-parse", "HEAD")
    producer_L_tree = _git_test(root, "rev-parse", "HEAD^{tree}")
    p_paths = tuple(f"release/{name}" for name in vnext._PRODUCER_PAYLOAD_P_FILES)
    r_paths = (
        "release/external_attestation.json",
        "release/external_attestation.json.sha256",
        "release/freeze_attestation.json",
        "release/freeze_attestation.json.sha256",
    )
    (root / "release").mkdir()
    for path in p_paths:
        (root / path).write_text("{}\n", encoding="utf-8")
    _git_test(root, "add", "--", ".")
    _git_test(root, "commit", "-q", "-m", "producer P")
    producer_P = _git_test(root, "rev-parse", "HEAD")
    producer_tree = _git_test(root, "rev-parse", "HEAD^{tree}")
    raw_l_diff, l_rows = vnext._git_name_status_diff(
        repo_root=root,
        parent_commit=audited_base,
        child_commit=producer_L,
        reason="test_L_diff_invalid",
    )
    raw_p_diff, p_rows = vnext._git_name_status_diff(
        repo_root=root,
        parent_commit=producer_L,
        child_commit=producer_P,
        reason="test_P_diff_invalid",
    )
    assert p_rows == tuple((path, "A") for path in p_paths)
    p_inventory = vnext._git_commit_inventory(
        repo_root=root,
        commit_sha1=producer_P,
        paths=p_paths,
        release_stage="producer_payload_P",
    )
    input_root = root / ".git" / "materializer-inputs"
    input_root.mkdir()
    freeze = {
        "producer_launch_L": {
            "stage": "producer_launch_L",
            "audited_base_commit_sha1": audited_base,
            "audited_base_tree_sha1": audited_tree,
            "audited_base_path_status_root_sha256": hashlib.sha256(
                vnext._canonical_bytes(
                    [{"path": path, "status": status} for path, status in l_rows]
                )
            ).hexdigest(),
            "commit_sha1": producer_L,
            "tree_sha1": producer_L_tree,
            "reviewed_file_inventory_root_sha256": "1" * 64,
            "reviewed_file_count": 20,
            "global_clean_tree": True,
            "procedurally_independent_review_present": False,
        },
        "checkpoint_manifest_root_sha256": "2" * 64,
        "selector_identity_sha256": "3" * 64,
        "selected_set_root_sha256": "4" * 64,
        "accepted_strata_root_sha256": "5" * 64,
        "counts": {
            "requested": 320,
            "completed": 320,
            "accepted": 300,
            "rejected": 20,
            "quarantine": 0,
        },
        "coverage_gaps": [],
    }
    freeze_path = input_root / "freeze-preimage.json"
    freeze_raw = vnext._canonical_bytes(freeze, newline=True)
    freeze_path.write_bytes(freeze_raw)
    freeze_sha = hashlib.sha256(freeze_raw).hexdigest()
    freeze_path.with_name(f"{freeze_path.name}.sha256").write_bytes(
        f"{freeze_sha}  {freeze_path.name}\n".encode("ascii"),
    )
    reviewer_identity = "6" * 64
    external = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
            "external-attestation.v1"
        ),
        "decision": "PASS",
        "audited_base": {
            "commit_sha1": audited_base,
            "tree_sha1": audited_tree,
            "producer_L_raw_diff_sha256": hashlib.sha256(raw_l_diff).hexdigest(),
            "producer_L_path_status_root_sha256": hashlib.sha256(
                vnext._canonical_bytes(
                    [{"path": path, "status": status} for path, status in l_rows]
                )
            ).hexdigest(),
            "producer_L_exact_global_diff_allowlist": True,
        },
        "producer_launch_L": {
            "commit_sha1": producer_L,
            "tree_sha1": producer_L_tree,
            "direct_parent_sha1": audited_base,
            "reviewed_file_inventory_root_sha256": "1" * 64,
            "reviewed_file_count": 20,
            "global_clean_tree": True,
        },
        "producer_payload_P": {
            "commit_sha1": producer_P,
            "tree_sha1": producer_tree,
            "direct_parent_sha1": producer_L,
            "raw_diff_sha256": hashlib.sha256(raw_p_diff).hexdigest(),
            "file_inventory_root_sha256": hashlib.sha256(
                vnext._canonical_bytes(p_inventory)
            ).hexdigest(),
            "file_count": 32,
            "exact_global_diff_allowlist": True,
        },
        "freeze_preimage": {
            "freeze_attestation_sha256": freeze_sha,
            "checkpoint_manifest_root_sha256": "2" * 64,
            "selector_identity_sha256": "3" * 64,
            "selected_set_root_sha256": "4" * 64,
            "accepted_strata_root_sha256": "5" * 64,
            "counts": dict(freeze["counts"]),
            "coverage_gaps": [],
            "coverage_complete": True,
        },
        "reviewer_receipt": {
            "reviewer_identity_sha256": reviewer_identity,
            "implementer_identity_sha256": (
                reviewer_identity if same_reviewer else "9" * 64
            ),
            "reviewer_is_not_implementer": True,
            "procedurally_independent_review": True,
            "review_procedure_sha256": "c" * 64,
            "reviewed_at": "2026-07-29T00:00:00Z",
        },
        "training_authorized": False,
        "formal_training_authorized": False,
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }
    external_path = input_root / "external-review.json"
    external_raw = vnext._canonical_bytes(external, newline=True)
    external_path.write_bytes(external_raw)
    external_sha = hashlib.sha256(external_raw).hexdigest()
    return (
        root,
        p_paths,
        r_paths,
        freeze_path,
        freeze_sha,
        external_path,
        external_sha,
    )


def test_R_materializer_create_once_exact_four_and_no_self_pin(
    tmp_path: Path,
) -> None:
    (
        repo,
        p_paths,
        r_paths,
        freeze_path,
        freeze_sha,
        external_path,
        external_sha,
    ) = _materializer_repo(tmp_path / "create-once")
    receipt = vnext._materialize_producer_R(
        repo_root=repo,
        release_root="release",
        producer_P_release_paths=p_paths,
        producer_R_release_paths=r_paths,
        freeze_preimage_path=freeze_path,
        freeze_preimage_sha256=freeze_sha,
        external_review_receipt_path=external_path,
        external_review_receipt_sha256=external_sha,
    )
    assert receipt["created_once"] is True
    assert receipt["release_self_pin_present"] is False
    assert receipt["R_identity_proof_written_into_R"] is False
    assert {path.name for path in (repo / "release").iterdir()} == {
        Path(path).name for path in (*p_paths, *r_paths)
    }
    with pytest.raises(batch.AdapterError, match="dirty tree|no replace"):
        vnext._materialize_producer_R(
            repo_root=repo,
            release_root="release",
            producer_P_release_paths=p_paths,
            producer_R_release_paths=r_paths,
            freeze_preimage_path=freeze_path,
            freeze_preimage_sha256=freeze_sha,
            external_review_receipt_path=external_path,
            external_review_receipt_sha256=external_sha,
        )


def test_R_materializer_rolls_back_partial_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        repo,
        p_paths,
        r_paths,
        freeze_path,
        freeze_sha,
        external_path,
        external_sha,
    ) = _materializer_repo(tmp_path / "rollback")
    original = vnext._exclusive_bytes
    calls = 0

    def fail_third(path: Path, raw: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected materializer failure")
        original(path, raw)

    monkeypatch.setattr(vnext, "_exclusive_bytes", fail_third)
    with pytest.raises(OSError, match="injected"):
        vnext._materialize_producer_R(
            repo_root=repo,
            release_root="release",
            producer_P_release_paths=p_paths,
            producer_R_release_paths=r_paths,
            freeze_preimage_path=freeze_path,
            freeze_preimage_sha256=freeze_sha,
            external_review_receipt_path=external_path,
            external_review_receipt_sha256=external_sha,
        )
    assert not any((repo / path).exists() for path in r_paths)


def test_R_materializer_rejects_reviewer_equal_to_implementer(
    tmp_path: Path,
) -> None:
    (
        repo,
        p_paths,
        r_paths,
        freeze_path,
        freeze_sha,
        external_path,
        external_sha,
    ) = _materializer_repo(
        tmp_path / "same-reviewer",
        same_reviewer=True,
    )
    with pytest.raises(batch.AdapterError, match="reviewer not independent"):
        vnext._materialize_producer_R(
            repo_root=repo,
            release_root="release",
            producer_P_release_paths=p_paths,
            producer_R_release_paths=r_paths,
            freeze_preimage_path=freeze_path,
            freeze_preimage_sha256=freeze_sha,
            external_review_receipt_path=external_path,
            external_review_receipt_sha256=external_sha,
        )
    assert not any((repo / path).exists() for path in r_paths)


def test_execute_between_group_stop_freezes_before_secret_and_lease_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    output_parent = tmp_path / "runs"
    output_parent.mkdir()
    output_root = output_parent / "run-one"
    profile = batch.load_config(
        REPO_ROOT / "configs/data/"
        "gemma3_chat_five_expert_qonly_unbalanced_v2_"
        "teacher_alignment.bulk_c30.yaml"
    )
    policy = replace(
        _policy(),
        base_profile_path=REPO_ROOT / "unused-profile.yaml",
        base_profile_sha256=profile.physical_sha256,
        output_root_parent=output_parent,
        controller_lease_name=".execute-e2e.lock",
        claims={
            "candidate": True,
            "live_authorized": True,
            "training_authorized": False,
        },
    )

    class Slots:
        loaded = True
        closed = False

        def receipt_hmac_key(self) -> bytes:
            assert not self.closed
            return b"h" * 32

        def close(self) -> None:
            self.closed = True
            events.append("slots.close")

    slots = Slots()

    class TrackingLease(vnext._VNextControllerLease):
        released = False

        def release(self) -> None:
            super().release()
            self.released = True
            events.append("lease.release")

    lease_instances: list[TrackingLease] = []

    class LeaseFactory(TrackingLease):
        def __init__(self, path: Path, run_id: str) -> None:
            super().__init__(path, run_id)
            lease_instances.append(self)

    terminal_calls = 0

    def terminal_recheck(**_: Any) -> dict[str, Any]:
        nonlocal terminal_calls
        terminal_calls += 1
        events.append(f"terminal.{terminal_calls}")
        payload = {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2."
                "teacher-alignment-vnext.terminal-transition-recheck.v1"
            ),
            "transition_evidence_sha256": "a" * 64,
            "policy_revalidation_sha256": "b" * 64,
            "old_controller_pid": 1,
            "old_archive_identity_sha256": "c" * 64,
            "old_archive_file_count": 3,
            "controller_count": 0,
            "candidate_controller_count": 1,
            "candidate_launch_process_count": 2,
            "candidate_controller_pid": os.getpid(),
            "controller_inventory_sha256": "d" * 64,
            "all_matching_inventory_sha256": "e" * 64,
            "canonical_root_absent": True,
            "vnext_root_present": True,
            "content_retained": False,
        }
        return {
            **payload,
            "recheck_sha256": hashlib.sha256(
                vnext._canonical_bytes(payload)
            ).hexdigest(),
        }

    class FakeState:
        def __init__(self) -> None:
            self.checkpoint_dir = output_root / "checkpoints"
            self.checkpoint_dir.mkdir()
            checkpoint = self.checkpoint_dir / "0001.json"
            checkpoint.write_text("{}", encoding="utf-8")
            checkpoint.with_name(f"{checkpoint.name}.sha256").write_text(
                "checkpoint",
                encoding="ascii",
            )
            self.freeze_dir = output_root / "freeze"
            self.stop_latch = vnext.StopAfterGroupLatch(
                requested=True,
                effective=True,
                reason_code="kill_switch_armed",
                requested_at_sequence=1,
                effective_checkpoint_sha256="f" * 64,
            )
            self._replay = batch.ReplayState(
                completed={f"completed-{index}" for index in range(30)},
                request_attempts=30,
                input_tokens=300,
                output_tokens=150,
            )

        def replay(self) -> batch.ReplayState:
            return self._replay

        def freeze_handoff(self, **_: Any) -> dict[str, Any]:
            assert slots.closed is False
            assert lease_instances[0].released is False
            events.append("freeze")
            self.freeze_dir.mkdir()
            (self.freeze_dir / "freeze_attestation.json").write_text(
                "{}",
                encoding="utf-8",
            )
            return {
                "counts": {
                    "requested": 320,
                    "completed": 30,
                    "accepted": 30,
                    "rejected": 0,
                    "quarantine": 290,
                }
            }

        def stage_external_attestation(
            self,
            _freeze: Mapping[str, Any],
        ) -> dict[str, Any]:
            assert slots.closed is False
            assert lease_instances[0].released is False
            events.append("stage")
            (self.freeze_dir / "external_attestation_staging.json").write_text(
                "{}", encoding="utf-8"
            )
            return {"staging_sha256": "1" * 64}

    class FakeRunner:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.jobs = tuple(range(320))
            self.state = FakeState()
            self.provider_requests = 0

        async def run(self, _phase: str) -> batch.RunReport:
            self.provider_requests += 30
            events.append("group.1.requests")
            events.append("kill.between.groups")
            raise vnext.StopAfterGroup("kill_switch_armed")

    monkeypatch.setattr(
        batch.RuntimeSecretSlots,
        "from_process_channel",
        classmethod(lambda _cls, _receiver, **_kwargs: slots),
    )
    monkeypatch.setattr(vnext, "_VNextControllerLease", LeaseFactory)
    monkeypatch.setattr(
        vnext,
        "validate_transition_evidence",
        lambda *_args, **_kwargs: {"evidence_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        vnext,
        "_terminal_transition_recheck",
        terminal_recheck,
    )
    monkeypatch.setattr(
        vnext,
        "_exclusive_vnext_root",
        lambda _policy, path: path.mkdir(),
    )
    monkeypatch.setattr(
        vnext,
        "authenticate_producer_launch_L",
        lambda: {"stage": "producer_launch_L"},
    )
    synthetic_candidates = tuple(_minimal320_candidates())
    _authorize_synthetic_selector(monkeypatch, synthetic_candidates)
    synthetic_selection = selector.select_minimal_320(
        synthetic_candidates,
        contract_sha256=policy.selector_contract_sha256,
        source_identity_sha256=_inventory().source_identity_sha256,
    )
    fake_overlay = SimpleNamespace(
        manifest={
            "identity_derivative_plan": {
                "derivative_specs": [
                    {"derived_teacher_job_idempotency_key": (f"{index:064x}")}
                    for index in range(100)
                ]
            }
        }
    )
    fake_preimages = SimpleNamespace(directory=output_root / "preimages")
    monkeypatch.setattr(
        selector,
        "build_authenticated_metadata_overlay",
        lambda *_args, **_kwargs: fake_overlay,
    )
    monkeypatch.setattr(
        selector,
        "authenticate_metadata_overlay",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        selector,
        "persist_selector_preimages",
        lambda *_args, **_kwargs: fake_preimages,
    )
    monkeypatch.setattr(
        selector,
        "load_selector_preimages",
        lambda _path: fake_preimages,
    )
    monkeypatch.setattr(
        selector,
        "authenticate_selector_preimages",
        lambda *_args, **_kwargs: (
            synthetic_candidates,
            synthetic_selection,
        ),
    )
    monkeypatch.setattr(
        vnext,
        "build_job_provenance",
        lambda **_kwargs: {"synthetic": True},
    )

    def materialize_P(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["runtime_slots"] is slots
        assert slots.closed is False
        assert lease_instances[0].released is False
        events.append("materialize.P32")
        (kwargs["state"].freeze_dir / "producer_payload_P_binding.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return {"binding_sha256": "2" * 64}

    monkeypatch.setattr(
        vnext,
        "_materialize_producer_payload_P",
        materialize_P,
    )
    result = asyncio.run(
        vnext.execute_vnext_from_process_channel(
            policy,
            tmp_path / "transition.json",
            "a" * 64,
            output_root,
            lambda: b"credential",
            profile_loader=lambda _path: profile,
            source_loader=lambda _profile: _inventory(),
            candidate_deriver=lambda _inventory_value, _profile, _overlay: (
                _minimal320_candidates()
            ),
            teacher_builder=lambda _profile, _slots: {},
            runner_factory=FakeRunner,
            environ={},
            hmac_factory=lambda _length: b"h" * 32,
        )
    )
    assert result["safe_stopped_after_complete_group"] is True
    assert result["state"] == "awaiting_independent_external_attestation"
    assert terminal_calls == 2
    assert events == [
        "terminal.1",
        "group.1.requests",
        "kill.between.groups",
        "terminal.2",
        "freeze",
        "stage",
        "materialize.P32",
        "slots.close",
        "lease.release",
    ]
    assert not any(item == "group.2.requests" for item in events)


def test_identity_tool_view_is_not_a_sixth_expert() -> None:
    source = _source(
        role="identity",
        language="zh-CN",
        identity_class="air_attribution",
    )
    job = _job(source)
    view = _view(
        job,
        output_role="tool",
        tool_family=None,
        selection_kind="identity",
    )
    stratum = vnext.body_free_stratum(source, view)
    assert stratum["role"] == "tool"
    assert stratum["validation_contract_role"] == "identity"
    assert stratum["tool_family"] is None
    assert stratum["language"] == "zh"
    with pytest.raises(
        batch.AdapterError,
        match="identity output expert mapping missing",
    ):
        vnext.body_free_stratum(source)


def test_closed_grammar_fast_gate_and_rejection_fuse_are_fail_closed() -> None:
    source = _source(role="serious")
    vnext.fast_validate_closed_grammar(
        source,
        '{"final_answer":"bounded public answer"}',
    )
    with pytest.raises(batch.AdapterError, match="closed grammar keys"):
        vnext.fast_validate_closed_grammar(
            source,
            '{"final_answer":"ok","extra":"forbidden"}',
        )
    fuse = vnext.RejectionRateFuse(
        window=4,
        min_samples=4,
        max_rate=0.25,
    )
    report = fuse.observe([False, True, True, False])
    assert report["latched"] is True


def test_absolute_deadline_kills_noncooperative_exit_without_residue(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "noncooperative_worker.py"
    marker = tmp_path / "wire-attempts.txt"
    worker_source = (
        "\n".join(
            (
                "import asyncio",
                "import json",
                "from pathlib import Path",
                "import sys",
                "frame = json.loads(sys.stdin.buffer.read())",
                "class Client:",
                "    async def __aenter__(self):",
                "        return self",
                "    async def post(self):",
                "        Path(sys.argv[1]).write_text(",
                "            'request\\n', encoding='ascii'",
                "        )",
                "    async def __aexit__(self, *_):",
                "        while True:",
                "            try:",
                "                await asyncio.sleep(60)",
                "            except asyncio.CancelledError:",
                "                continue",
                "async def run():",
                "    async with Client() as client:",
                "        await client.post()",
                "asyncio.run(run())",
            )
        )
        + "\n"
    )
    worker.write_text(
        worker_source,
        encoding="utf-8",
    )
    encoded_worker = base64.b64encode(worker_source.encode("utf-8")).decode("ascii")
    teacher = vnext.AbsoluteDeadlineCompatibleTeacher(
        base_url="https://example.invalid",
        model="synthetic-model",
        protocol="openai",
        fallback_protocol=None,
        api_key_env="",
        credential_provider=lambda: "memory-only",
        stream_openai=False,
        max_retries=2,
        timeout_seconds=1,
        wall_clock_deadline_seconds=1.0,
        isolated_worker_command_factory=lambda: (
            sys.executable,
            "-S",
            "-c",
            f"import base64;exec(base64.b64decode('{encoded_worker}'))",
            str(marker),
        ),
    )

    async def scenario() -> None:
        started = time.monotonic()
        with pytest.raises(batch.ClientDeadlineExceeded):
            await teacher.complete(
                system="synthetic",
                user="synthetic",
                idempotency_key="a" * 64,
            )
        assert time.monotonic() - started <= (teacher.wall_clock_deadline_seconds)
        await asyncio.sleep(0)
        assert teacher.provider_provenance["attempts"]["wire_attempts"] == 1
        assert teacher.active_isolated_process_count == 0
        pid = teacher._last_isolated_pid
        assert isinstance(pid, int)
        try:
            import psutil
        except ImportError:
            psutil = None
        if psutil is not None:
            assert psutil.pid_exists(pid) is False
        assert marker.read_text(encoding="ascii") == "request\n"
        current = asyncio.current_task()
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        ]

    asyncio.run(scenario())


def test_physical_20ms_deadline_rejects_before_request_with_zero_residue() -> None:
    teacher = vnext.AbsoluteDeadlineCompatibleTeacher(
        base_url="https://example.invalid",
        model="synthetic-model",
        protocol="openai",
        fallback_protocol=None,
        api_key_env="",
        credential_provider=lambda: "memory-only",
        stream_openai=False,
        max_retries=2,
        timeout_seconds=1,
        wall_clock_deadline_seconds=0.02,
        isolated_worker_command_factory=lambda: (
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ),
    )

    async def scenario() -> None:
        started = time.monotonic()
        with pytest.raises(batch.ClientDeadlineExceeded):
            await teacher.complete(
                system="synthetic",
                user="synthetic",
                idempotency_key="a" * 64,
            )
        assert time.monotonic() - started <= (teacher.wall_clock_deadline_seconds)
        assert teacher.provider_provenance["attempts"]["wire_attempts"] == 0
        assert teacher.active_isolated_process_count == 0
        assert teacher._last_isolated_pid is None
        current = asyncio.current_task()
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        ]

    asyncio.run(scenario())


def test_minimal320_selector_freezes_exact_quotas_and_remainders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _minimal320_candidates()
    _authorize_synthetic_selector(monkeypatch, candidates)
    selection = selector.select_minimal_320(
        candidates,
        contract_sha256="a" * 64,
        source_identity_sha256="b" * 64,
    )
    assert len(selection.candidates) == 320
    assert selection.manifest["counts"]["selected_by_role"] == {
        "angry": 40,
        "humor": 40,
        "review": 60,
        "router": 80,
        "serious": 40,
        "tool": 60,
    }
    assert selection.manifest["counts"]["selected_by_language"] == {
        "en": 160,
        "zh": 160,
    }
    assert all(item.output_expert_role != "identity" for item in selection.candidates)
    identity_tool = [
        item
        for item in selection.candidates
        if item.selection_kind == "identity" and item.output_expert_role == "tool"
    ]
    assert len(identity_tool) == 20
    assert all(
        item.validation_contract_role == "identity" and item.tool_family is None
        for item in identity_tool
    )
    identity_bundles: dict[str, list[selector.SelectorCandidate]] = {}
    for item in selection.candidates:
        if item.selection_kind == "identity":
            identity_bundles.setdefault(item.task_bundle_sha256, []).append(item)
    assert len(identity_bundles) == 20
    assert all(
        {item.output_expert_role for item in rows} == set(selector.OUTPUT_EXPERT_ROLES)
        and len({(item.identity_class, item.language) for item in rows}) == 1
        for rows in identity_bundles.values()
    )


def test_minimal320_selector_rejects_runtime_remainder_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _minimal320_candidates()
    identity = next(
        item
        for item in candidates
        if item.selection_kind == "identity"
        and item.identity_class == "air_attribution"
    )
    mutated = replace(
        identity,
        identity_class="false_openai_attribution",
    )
    mutated = replace(
        mutated,
        candidate_id_sha256=selector.candidate_identity_sha256(mutated.as_dict()),
    )
    candidates[candidates.index(identity)] = mutated
    _authorize_synthetic_selector(monkeypatch, candidates)
    with pytest.raises(
        batch.AdapterError,
        match="identity availability bundle mixed",
    ):
        selector.select_minimal_320(
            candidates,
            contract_sha256="a" * 64,
            source_identity_sha256="b" * 64,
        )


def test_historical_partial_is_always_quarantined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _minimal320_candidates()
    _authorize_synthetic_selector(monkeypatch, candidates)
    selection = selector.select_minimal_320(
        candidates,
        contract_sha256="a" * 64,
        source_identity_sha256="b" * 64,
    )
    decision = selector.assess_legacy_reuse(
        selection,
        evidence=selector.ReuseClosureEvidence(
            full_wal_chain_and_sequence=True,
            runtime_hmac_receipts=True,
            source_identity_and_heldout=True,
            bundle_membership_and_selector_metadata=True,
            uncertain_zero=True,
        ),
        authenticated_idempotency_keys=[
            item.idempotency_key for item in selection.candidates
        ],
    )
    assert decision["eligible"] is False
    assert decision["decision"] == "fresh_vnext_320_required"
    assert decision["legacy_disposition"] == "historical_quarantine_only"


def test_transition_gate_rejects_dual_controller_and_accepts_closed_evidence(
    tmp_path: Path,
) -> None:
    evidence = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
            "transition-evidence.v1"
        ),
        "controller_count": 1,
        "controller_inventory_scope": (
            "matching_controller_entrypoints_excluding_"
            "authenticated_candidate_process_tree"
        ),
        "controller_inventory_sha256": "f" * 64,
        "old_controller_pid": 2147483000,
        "old_controller_pid_alive": False,
        "old_canonical_archive_state": "archived_entire_root_no_replace",
        "old_archive_identity_sha256": "a" * 64,
        "old_canonical_root": str(tmp_path / "old"),
        "old_archive_root": str(tmp_path / "archive"),
        "canonical_root_absent": True,
        "vnext_root": str(tmp_path / "vnext"),
        "vnext_root_absent": True,
        "legacy_disposition": "historical_quarantine_only",
        "legacy_partial_uncertain_consumable": False,
        "credential_channel": "anonymous_stdin_same_process",
        "runtime_hmac_scope": "same_process_memory_only",
        "checked_at": "2026-07-29T00:00:00+00:00",
    }
    path = tmp_path / "transition.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(batch.AdapterError, match="transition gate blocked"):
        vnext.validate_transition_evidence(path)
    evidence["controller_count"] = 0
    path.write_text(json.dumps(evidence), encoding="utf-8")
    report = vnext.validate_transition_evidence(path)
    assert report["transition_gate_satisfied"] is True
    assert report["legacy_partial_uncertain_consumable"] is False


def test_windows_handle_inventory_accepts_one_launch_tree_and_detects_second(
    tmp_path: Path,
) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows handle inventory test")
    single = vnext._trusted_controller_inventory()
    assert single["candidate_controller_count"] == 1
    assert single["candidate_launch_process_count"] >= 1
    assert single["controller_count"] == 0

    ready = tmp_path / "second-controller.ready"
    marker = "gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1"
    code = (
        "from pathlib import Path; import time; "
        f"marker={marker!r}; "
        f"Path({str(ready)!r}).write_text(marker, encoding='utf-8'); "
        "time.sleep(3)"
    )
    process = subprocess.Popen(
        ["py", "-3", "-c", code],
        cwd=REPO_ROOT,
    )
    try:
        deadline = time.monotonic() + 2
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file()
        duplicate = vnext._trusted_controller_inventory()
        assert duplicate["candidate_controller_count"] == 1
        assert duplicate["controller_count"] >= 1
    finally:
        process.wait(timeout=6)


def test_windows_inventory_error_cannot_be_reported_as_zero_controllers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows handle inventory test")

    def fail() -> list[dict[str, Any]]:
        raise batch.AdapterError("vnext_controller_inventory_image_unreadable")

    monkeypatch.setattr(vnext, "_windows_controller_process_rows", fail)
    with pytest.raises(
        batch.AdapterError,
        match="inventory image unreadable",
    ):
        vnext._trusted_controller_inventory()


def test_dangling_named_entry_is_present_for_no_replace_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_repo = tmp_path / "repo"
    synthetic_repo.mkdir()
    monkeypatch.setattr(batch, "REPO_ROOT", synthetic_repo)
    parent = tmp_path / "external-runs"
    parent.mkdir(parents=True)
    dangling = parent / "run-one"
    try:
        os.symlink(
            parent / "missing-target",
            dangling,
            target_is_directory=True,
        )
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    assert vnext._named_entry_absent(dangling, stop=tmp_path) is False
    assert vnext._named_regular_directory(dangling, stop=tmp_path) is False
    policy = replace(_policy(), output_root_parent=parent)
    with pytest.raises(
        batch.AdapterError,
        match="unique output root invalid",
    ):
        vnext._exclusive_vnext_root(policy, dangling)


def test_windows_junction_is_present_and_not_a_regular_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows junction test")
    synthetic_repo = tmp_path / "repo"
    synthetic_repo.mkdir()
    monkeypatch.setattr(batch, "REPO_ROOT", synthetic_repo)
    parent = tmp_path / "external-runs"
    target = tmp_path / "junction-target"
    parent.mkdir(parents=True)
    target.mkdir()
    junction = parent / "run-junction"
    escaped_junction = str(junction).replace("'", "''")
    escaped_target = str(target).replace("'", "''")
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "New-Item -ItemType Junction "
                f"-Path '{escaped_junction}' "
                f"-Target '{escaped_target}' | Out-Null"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip("junction creation is unavailable")
    assert vnext._named_entry_absent(junction, stop=tmp_path) is False
    assert vnext._named_regular_directory(junction, stop=tmp_path) is False
    policy = replace(_policy(), output_root_parent=parent)
    with pytest.raises(
        batch.AdapterError,
        match="unique output root invalid",
    ):
        vnext._exclusive_vnext_root(policy, junction)


def _terminal_transition_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    vnext.VNextPolicy,
    Path,
    str,
    Path,
    vnext._VNextControllerLease,
    dict[str, Any],
    Path,
]:
    synthetic_repo = tmp_path / "repo"
    synthetic_repo.mkdir()
    monkeypatch.setattr(batch, "REPO_ROOT", synthetic_repo)
    config_dir = synthetic_repo / "configs"
    config_dir.mkdir()
    implementation_path = synthetic_repo / "implementation.py"
    implementation_path.write_text("PINNED = True\n", encoding="utf-8")
    config_schema_path = config_dir / "config.schema.json"
    config_schema_path.write_text(
        json.dumps({"type": "object"}),
        encoding="utf-8",
    )
    transition_schema_path = config_dir / "transition.schema.json"
    transition_schema_path.write_text(
        json.dumps({"type": "object"}),
        encoding="utf-8",
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config = {
        "schema_path": "configs/config.schema.json",
        "schema_sha256": digest(config_schema_path),
        "implementation_path": "implementation.py",
        "implementation_sha256": digest(implementation_path),
        "transition_schema_path": "configs/transition.schema.json",
        "transition_schema_sha256": digest(transition_schema_path),
    }
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_parent = tmp_path / "external-runs"
    output_parent.mkdir(parents=True)
    output_root = output_parent / "run-one"
    output_root.mkdir()
    old_canonical = synthetic_repo / "data" / "old"
    archive_parent = synthetic_repo / "data" / "archive"
    archive_root = archive_parent / "old-run"
    wal = archive_root / "automation" / "controller_wal_v1"
    wal.mkdir(parents=True)
    old_pid = 2147483000
    (wal / "genesis.json").write_text(
        json.dumps({"process_id": old_pid}),
        encoding="utf-8",
    )
    (wal / "process_lock.json").write_text(
        json.dumps({"process_id": old_pid}),
        encoding="utf-8",
    )
    (archive_root / "automation" / "status.json").write_text(
        json.dumps({"state": "stopped"}),
        encoding="utf-8",
    )
    archive_identity, _ = vnext._archived_root_identity(
        archive_root,
        expected_pid=old_pid,
    )
    inventory = {
        "controller_count": 0,
        "candidate_controller_count": 1,
        "candidate_launch_process_count": 2,
        "candidate_controller_pid": os.getpid(),
        "controller_inventory_sha256": "f" * 64,
        "all_matching_inventory_sha256": "e" * 64,
        "content_retained": False,
    }
    evidence = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-vnext."
            "transition-evidence.v1"
        ),
        "controller_count": 0,
        "controller_inventory_scope": (
            "matching_controller_entrypoints_excluding_"
            "authenticated_candidate_process_tree"
        ),
        "controller_inventory_sha256": inventory["controller_inventory_sha256"],
        "old_controller_pid": old_pid,
        "old_controller_pid_alive": False,
        "old_canonical_archive_state": "archived_entire_root_no_replace",
        "old_archive_identity_sha256": archive_identity,
        "old_canonical_root": str(old_canonical),
        "old_archive_root": str(archive_root),
        "canonical_root_absent": True,
        "vnext_root": str(output_root),
        "vnext_root_absent": True,
        "legacy_disposition": "historical_quarantine_only",
        "legacy_partial_uncertain_consumable": False,
        "credential_channel": "anonymous_stdin_same_process",
        "runtime_hmac_scope": "same_process_memory_only",
        "checked_at": "2026-07-29T00:00:00+00:00",
    }
    evidence_path = synthetic_repo / "transition.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    evidence_sha256 = digest(evidence_path)
    base = _policy()
    policy = replace(
        base,
        path=config_path,
        physical_sha256=digest(config_path),
        schema_path=config_schema_path,
        schema_sha256=digest(config_schema_path),
        implementation_path=implementation_path,
        implementation_sha256=digest(implementation_path),
        transition_schema_path=transition_schema_path,
        transition_schema_sha256=digest(transition_schema_path),
        output_root_parent=output_parent,
        old_canonical_root=old_canonical,
        old_archive_parent=archive_parent,
        controller_lease_name=".controller.lock",
    )
    monkeypatch.setattr(vnext, "_pid_alive", lambda _pid: False)
    lease = vnext._VNextControllerLease(
        output_parent / policy.controller_lease_name,
        "1" * 32,
    )
    lease.acquire()
    return (
        policy,
        evidence_path,
        evidence_sha256,
        output_root,
        lease,
        inventory,
        archive_root,
    )


def test_terminal_transition_recheck_binds_pins_inventory_and_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        policy,
        evidence_path,
        evidence_sha256,
        output_root,
        lease,
        inventory,
        archive_root,
    ) = _terminal_transition_fixture(tmp_path, monkeypatch)
    try:
        first = vnext._terminal_transition_recheck(
            policy=policy,
            transition_evidence=evidence_path,
            transition_evidence_sha256=evidence_sha256,
            output_root=output_root,
            lease=lease,
            process_inventory_probe=lambda: inventory,
        )
        assert first["controller_count"] == 0
        assert first["candidate_launch_process_count"] == 2
        (archive_root / "automation" / "status.json").write_text(
            json.dumps({"state": "tampered"}),
            encoding="utf-8",
        )
        with pytest.raises(
            batch.AdapterError,
            match="archive identity drift",
        ):
            vnext._terminal_transition_recheck(
                policy=policy,
                transition_evidence=evidence_path,
                transition_evidence_sha256=evidence_sha256,
                output_root=output_root,
                lease=lease,
                process_inventory_probe=lambda: inventory,
            )
    finally:
        lease.release()


def test_terminal_transition_recheck_rejects_pin_mutation_and_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        policy,
        evidence_path,
        evidence_sha256,
        output_root,
        lease,
        inventory,
        _,
    ) = _terminal_transition_fixture(tmp_path, monkeypatch)
    try:
        policy.implementation_path.write_text(
            "PINNED = False\n",
            encoding="utf-8",
        )
        with pytest.raises(
            batch.AdapterError,
            match="implementation identity drift",
        ):
            vnext._terminal_transition_recheck(
                policy=policy,
                transition_evidence=evidence_path,
                transition_evidence_sha256=evidence_sha256,
                output_root=output_root,
                lease=lease,
                process_inventory_probe=lambda: inventory,
            )
        policy.implementation_path.write_text(
            "PINNED = True\n",
            encoding="utf-8",
        )
        duplicate = {
            **inventory,
            "controller_count": 1,
        }
        with pytest.raises(
            batch.AdapterError,
            match="controller inventory drift",
        ):
            vnext._terminal_transition_recheck(
                policy=policy,
                transition_evidence=evidence_path,
                transition_evidence_sha256=evidence_sha256,
                output_root=output_root,
                lease=lease,
                process_inventory_probe=lambda: duplicate,
            )
    finally:
        lease.release()


def test_policy_transition_schema_relocation_is_not_derived(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        policy,
        evidence_path,
        evidence_sha256,
        output_root,
        lease,
        _,
        _,
    ) = _terminal_transition_fixture(tmp_path, monkeypatch)
    relocated = tmp_path / "transition.schema.json"
    relocated.write_text('{"type":"object"}', encoding="utf-8")
    try:
        with pytest.raises(
            batch.AdapterError,
            match="transition schema path drift",
        ):
            vnext.validate_transition_evidence(
                evidence_path,
                schema_path=relocated,
                expected_sha256=evidence_sha256,
                policy=policy,
                output_root=output_root,
            )
    finally:
        lease.release()


def test_consumer_code_P_gate_rejects_unreviewed_binding() -> None:
    blocked = {
        **_policy().consumer_code_P,
        "status": "blocked_uncommitted_unreviewed",
        "blocker_reason_code": "consumer_code_P_uncommitted_unreviewed",
        "commit_sha1": None,
        "tree_sha1": None,
        "file_inventory_root_sha256": None,
        "implementation_sha256": None,
        "schema_set_root_sha256": None,
        "contract_sha256": None,
        "independently_audited": False,
    }
    with pytest.raises(
        batch.AdapterError,
        match="consumer code P uncommitted unreviewed",
    ):
        vnext._authenticate_consumer_code_P(blocked)


def test_model_free_benchmark_reports_quadratic_to_linear_operation_change() -> None:
    path = (
        REPO_ROOT / "scripts/research/"
        "benchmark_gemma3_teacher_alignment_vnext_wal_v1.py"
    )
    spec = importlib.util.spec_from_file_location("vnext_wal_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_benchmark(20, 1)
    assert report["model_requests"] == 0
    assert report["gpu_requests"] == 0
    assert report["legacy"]["entries_authenticated"] == 190
    assert report["vnext"]["tip_files_authenticated"] == 20
    assert report["legacy"]["total_complexity"] == "O(n^2)"
    assert report["vnext"]["total_complexity"] == "O(n)"
    assert report["real_vnext_path"] == {
        **report["real_vnext_path"],
        "jobs": 20,
        "group_size": 20,
        "groups": 1,
        "append_event_calls": 40,
        "terminal_commit_calls": 20,
        "delta_checkpoint_writes": 1,
        "status_writes": 1,
        "startup_full_replays": 1,
        "online_full_history_replays": 0,
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }
    assert report["real_vnext_path"]["path"].endswith(
        "commit_group->delta_checkpoint->write_status"
    )
    throughput = report["real_vnext_path"]["provider_throughput"]
    assert throughput["source"] == "vnext_body_free_terminal_telemetry"
    assert throughput["exact"] is True
    assert throughput["terminal_jobs"] == 20
    assert throughput["exact_rows"] == 20
    assert throughput["unknown_rows"] == 0
    assert throughput["error"] is None
    assert throughput["window_seconds"] > 0
    assert throughput["provider_input_tokens_per_second"] > 0
    assert throughput["provider_output_tokens_per_second"] > 0
    assert (
        throughput["provider_input_tokens_per_second"]
        / throughput["provider_output_tokens_per_second"]
    ) == pytest.approx(11 / 7, rel=1e-5)
    assert throughput["live_reload"] is False


def test_vnext_python_materialization_stays_lf_with_core_autocrlf_true(
    tmp_path: Path,
) -> None:
    paths = (
        "scripts/data/run_gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
        "scripts/observability/distillation_dashboard.py",
        "scripts/research/benchmark_gemma3_teacher_alignment_vnext_wal_v1.py",
        "src/anchor_mvp/data/"
        "gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2.py",
        "src/anchor_mvp/data/gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
        "tests/test_distillation_dashboard.py",
        "tests/test_gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2.py",
        "tests/test_gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1.py",
    )
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    for path in paths:
        assert attributes.splitlines().count(f"{path} text eol=lf") == 1
    release_root = (
        "data/gemma3_chat_five_expert_qonly_unbalanced_v2_"
        "teacher_alignment_vnext_v1/release_v1"
    )
    release_rule = f"{release_root}/** text eol=lf"
    assert attributes.splitlines().count(release_rule) == 1
    release_payloads = {
        f"{release_root}/{vnext.TEACHER_TRAINING_SHARD_NAME}": (
            b'{"teacher_target":"stable-lf"}\n'
        ),
        f"{release_root}/{vnext.TEACHER_TRAINING_SHARD_NAME}.sha256": (
            b"a" * 64 + b"  teacher_training_shard-00000.jsonl\n"
        ),
        f"{release_root}/{vnext.TRAINING_RECORD_INVENTORY_NAME}": (
            b'{"training_ordinal":0}\n'
        ),
        f"{release_root}/{vnext.TRAINING_RECORD_INVENTORY_NAME}.sha256": (
            b"b" * 64 + b"  training_record_inventory.jsonl\n"
        ),
    }
    all_paths = (*paths, *release_payloads)

    repository = tmp_path / "autocrlf-materialization"
    repository.mkdir()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "--quiet")
    (repository / ".gitattributes").write_text(
        attributes,
        encoding="utf-8",
        newline="\n",
    )
    for path in paths:
        source = REPO_ROOT / path
        destination = repository / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    for path, raw in release_payloads.items():
        destination = repository / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    git("-c", "core.autocrlf=false", "add", "--", ".gitattributes", *all_paths)
    checked = git(
        "-c",
        "core.autocrlf=true",
        "check-attr",
        "text",
        "eol",
        "--",
        *all_paths,
    ).stdout.splitlines()
    for path in all_paths:
        assert f"{path}: text: set" in checked
        assert f"{path}: eol: lf" in checked
        (repository / path).unlink()
    git(
        "-c",
        "core.autocrlf=true",
        "checkout-index",
        "--force",
        "--",
        *all_paths,
    )
    for path in all_paths:
        materialized = (repository / path).read_bytes()
        assert materialized
        assert b"\r\n" not in materialized
        assert b"\r" not in materialized
        if path in release_payloads:
            assert materialized == release_payloads[path]
            assert (
                hashlib.sha256(materialized).hexdigest()
                == hashlib.sha256(release_payloads[path]).hexdigest()
            )
