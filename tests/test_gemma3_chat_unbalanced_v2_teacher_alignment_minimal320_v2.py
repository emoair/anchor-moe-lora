from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import pytest

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_v2 as selector,
)
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_teacher_alignment_vnext_v1 as vnext,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    REPO_ROOT / "configs/data/"
    "gemma3_chat_five_expert_qonly_unbalanced_v2_"
    "teacher_alignment.bulk_c30.yaml"
)
CONTRACT_PATH = (
    REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_contract_v2.json"
)


def _overlay_from_directory(
    directory: Path,
) -> selector.AuthenticatedMetadataOverlay:
    rows_path = directory / "authenticated_metadata_overlay.jsonl"
    manifest_path = directory / "authenticated_metadata_overlay_manifest.json"
    rows = tuple(
        json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return selector.AuthenticatedMetadataOverlay(
        directory=directory,
        rows_path=rows_path,
        manifest_path=manifest_path,
        rows=rows,
        manifest=manifest,
    )


@pytest.fixture(scope="module")
def physical_v2(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    output_root = tmp_path_factory.mktemp("minimal320-v2-physical")
    config = batch.load_config(PROFILE_PATH)
    inventory = batch.load_source_inventory(config)
    contract_sha256 = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    overlay = selector.build_authenticated_metadata_overlay(
        inventory,
        config,
        output_root=output_root,
    )
    candidates = selector.derive_candidates_from_source(
        inventory,
        config,
        overlay,
    )
    selection = selector.select_minimal_320(
        candidates,
        contract_sha256=contract_sha256,
        source_identity_sha256=inventory.source_identity_sha256,
    )
    preimages = selector.persist_selector_preimages(
        candidates,
        selection,
        overlay=overlay,
        inventory=inventory,
        config=config,
        contract_sha256=contract_sha256,
    )
    candidates, selection = selector.authenticate_selector_preimages(
        selector.load_selector_preimages(preimages.directory),
        overlay=overlay,
        inventory=inventory,
        config=config,
        contract_sha256=contract_sha256,
    )
    specs = tuple(
        dict(spec)
        for spec in overlay.manifest["identity_derivative_plan"]["derivative_specs"]
    )
    specs_by_key = {
        str(spec["derived_teacher_job_idempotency_key"]): spec for spec in specs
    }
    views = {
        candidate.idempotency_key: candidate.as_dict()
        for candidate in selection.candidates
    }
    jobs = vnext.derive_selected_vnext_jobs(
        batch.derive_jobs(inventory, config),
        views,
        identity_derivative_specs=specs,
        inventory=inventory,
        config=config,
    )
    return {
        "config": config,
        "inventory": inventory,
        "contract_sha256": contract_sha256,
        "overlay": overlay,
        "candidates": candidates,
        "selection": selection,
        "preimages": preimages,
        "specs": specs,
        "specs_by_key": specs_by_key,
        "views": views,
        "jobs": jobs,
    }


def _identity_jobs(
    physical_v2: Mapping[str, Any],
) -> list[batch.AlignmentJob]:
    views = physical_v2["views"]
    return [
        job
        for job in physical_v2["jobs"]
        if views[job.idempotency_key]["selection_kind"] == "identity"
    ]


def _record(
    job: batch.AlignmentJob,
    view: Mapping[str, Any],
    spec: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "idempotency_key": job.idempotency_key,
        "record_id": job.source.record_id,
        "task_bundle": job.source.task_bundle,
        "semantic": job.source.semantic,
        "role": view["output_expert_role"],
        "language": job.source.language,
        "source_binding": {
            "selected_candidate_sha256": view["candidate_id_sha256"],
            "source_record_id_sha256": view["source_record_id_sha256"],
            "source_content_sha256": view["source_content_sha256"],
            "source_serialization_identity_sha256": view[
                "source_serialization_identity_sha256"
            ],
            "task_bundle_sha256": view["task_bundle_sha256"],
            "training_asset": view["training_asset"],
        },
        "teacher_binding": {
            "prompt_version": spec["v5_prompt_version"],
            "prompt_template_id": job.source.teacher_contract["prompt_template_id"],
            "prompt_template_sha256": job.prompt_template_sha256,
            "output_schema_id": job.source.teacher_contract["output_schema_id"],
            "output_schema_sha256": job.output_schema_sha256,
        },
        "output": dict(output),
    }


def test_real_derivative_inventory_is_exact_20x5_and_balanced(
    physical_v2: Mapping[str, Any],
) -> None:
    jobs = _identity_jobs(physical_v2)
    views = physical_v2["views"]
    specs = physical_v2["specs_by_key"]
    assert len(jobs) == 100
    assert len({job.source.record_id for job in jobs}) == 100
    assert len({job.idempotency_key for job in jobs}) == 100
    assert (
        len(
            {
                views[job.idempotency_key]["parent_source_record_id_sha256"]
                for job in jobs
            }
        )
        == 100
    )
    assert Counter(
        (
            views[job.idempotency_key]["output_expert_role"],
            job.source.role,
        )
        for job in jobs
    ) == Counter(
        {
            ("humor", "humor"): 20,
            ("serious", "serious"): 20,
            ("angry", "angry"): 20,
            ("tool", "identity"): 20,
            ("review", "review"): 20,
        }
    )
    assert Counter(
        (
            views[job.idempotency_key]["identity_class"],
            views[job.idempotency_key]["language"],
        )
        for job in jobs
    ) == Counter(selector.IDENTITY_RECORD_CELL_QUOTAS)
    assert all(
        job.prompt_template_sha256
        == specs[job.idempotency_key]["derived_prompt_contract_sha256"]
        != specs[job.idempotency_key]["parent_base_prompt_template_sha256"]
        and job.output_schema_sha256
        == specs[job.idempotency_key]["derived_output_contract_sha256"]
        for job in jobs
    )


def test_exact_six_identity_cells_reconstruct_authenticated_prompts(
    physical_v2: Mapping[str, Any],
) -> None:
    views = physical_v2["views"]
    specs = physical_v2["specs_by_key"]
    selected: dict[tuple[str, str], batch.AlignmentJob] = {}
    for job in _identity_jobs(physical_v2):
        view = views[job.idempotency_key]
        if view["output_expert_role"] != "tool":
            continue
        selected.setdefault(
            (view["identity_class"], view["language"]),
            job,
        )
    assert set(selected) == {
        (identity_class, language)
        for identity_class in batch.IDENTITY_CLASSES
        for language in ("en", "zh")
    }
    for (identity_class, _language), job in selected.items():
        spec = specs[job.idempotency_key]
        system, user = vnext.authenticated_job_prompts(
            job,
            views[job.idempotency_key],
            identity_derivative_spec=spec,
            inventory=physical_v2["inventory"],
            config=physical_v2["config"],
        )
        assert system == selector._v5_role_system_prompt("identity")
        assert user == batch._user_prompt(job.source)
        assert job.source.role == "identity"
        assert spec["expert_asset_role"] == "tool"
        assert spec["validation_contract_role"] == "identity"
        assert spec["response_contract"] == ("raw_exact_identity_template_no_tool")
        assert spec["tool_invocation_allowed"] is False
        assert (
            spec["target_template"] == (selector.V5_IDENTITY_TEMPLATES[identity_class])
        )


class _CaptureTeacher:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []
        self.provider_provenance = {
            "completion": {
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                },
                "response_id": "response-1",
            },
            "attempts": {
                "wire_attempts": 1,
                "retry_count": 0,
                "retry_reasons": [],
            },
        }

    def set_retry_limit(self, _value: int) -> None:
        return None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        idempotency_key: str,
    ) -> str:
        self.calls.append((system, user, idempotency_key))
        return self.response


class _WireState:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.provider_terminal_usage: list[Mapping[str, Any] | None] = []

    async def reserve_budget(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def append_event(
        self,
        *,
        state: str,
        **_kwargs: Any,
    ) -> None:
        self.events.append(state)

    def _observe_stop_probe(self) -> None:
        return None

    def observe_provider_terminal_usage(
        self,
        usage: Mapping[str, Any] | None,
    ) -> None:
        self.provider_terminal_usage.append(usage)


def test_execute_job_sends_exact_authenticated_wire_bytes_for_six_cells(
    physical_v2: Mapping[str, Any],
) -> None:
    views = physical_v2["views"]
    specs = physical_v2["specs_by_key"]
    selected: dict[tuple[str, str], batch.AlignmentJob] = {}
    for job in _identity_jobs(physical_v2):
        view = views[job.idempotency_key]
        if view["output_expert_role"] == "tool":
            selected.setdefault(
                (view["identity_class"], view["language"]),
                job,
            )
    for job in selected.values():
        spec = specs[job.idempotency_key]
        teacher = _CaptureTeacher(str(spec["target_template"]))
        runner = object.__new__(vnext.VNextBatchRunner)
        runner.state = _WireState()
        runner.teachers = {"identity": teacher}
        runner.inventory = physical_v2["inventory"]
        runner.config = physical_v2["config"]
        runner._selection_views = {job.idempotency_key: views[job.idempotency_key]}
        runner._identity_derivative_specs = {job.idempotency_key: spec}
        runner._output_record = lambda *_args: {
            "source_binding": {},
            "derived_planner_overlay": {"overlay_identity_sha256": "0" * 64},
        }
        runner._job_receipt = lambda *_args, **_kwargs: {}
        pending = asyncio.run(
            runner._execute_job(
                job,
                phase="bulk",
                replay=batch.ReplayState(),
                retry_limit=0,
            )
        )
        assert pending is not None
        expected_system, expected_user = selector.identity_derivative_prompts_from_spec(
            spec,
            inventory=physical_v2["inventory"],
            config=physical_v2["config"],
        )[:2]
        assert teacher.calls == [
            (
                expected_system,
                expected_user,
                job.idempotency_key,
            )
        ]
        assert runner.state.provider_terminal_usage == [
            {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        ]


@pytest.mark.parametrize(
    "field",
    [
        "system_prompt_sha256",
        "user_prompt_sha256",
        "teacher_input_sha256",
        "derived_teacher_job_idempotency_key",
        "derived_prompt_contract_sha256",
        "derived_output_contract_sha256",
    ],
)
def test_derivative_spec_or_contract_mutation_is_rejected(
    physical_v2: Mapping[str, Any],
    field: str,
) -> None:
    job = _identity_jobs(physical_v2)[0]
    spec = dict(physical_v2["specs_by_key"][job.idempotency_key])
    spec[field] = "0" * 64
    with pytest.raises(batch.AdapterError):
        selector.identity_derivative_job_from_spec(
            spec,
            inventory=physical_v2["inventory"],
            config=physical_v2["config"],
        )


def test_tool_asset_accepts_only_exact_template_and_rejects_tool_call(
    physical_v2: Mapping[str, Any],
) -> None:
    views = physical_v2["views"]
    specs = physical_v2["specs_by_key"]
    job = next(
        job
        for job in _identity_jobs(physical_v2)
        if views[job.idempotency_key]["output_expert_role"] == "tool"
    )
    view = views[job.idempotency_key]
    spec = specs[job.idempotency_key]
    template = str(spec["target_template"])
    output = vnext.fast_parse_candidate_output(job.source, template)
    record = _record(job, view, spec, output)
    vnext.final_quality_validate(
        job,
        record,
        view,
        identity_derivative_spec=spec,
        inventory=physical_v2["inventory"],
        config=physical_v2["config"],
    )
    with pytest.raises(batch.AdapterError):
        vnext.fast_parse_candidate_output(
            job.source,
            json.dumps(
                {
                    "final_answer": template,
                    "tool_call": {
                        "name": "identity",
                        "arguments": {},
                    },
                    "evidence_ids": [],
                }
            ),
        )
    with pytest.raises(batch.AdapterError):
        vnext.fast_parse_candidate_output(
            job.source,
            json.dumps(
                {
                    "direct_answer": template,
                    "tool_used": False,
                }
            ),
        )


def test_review_derivative_has_explicit_exact_body_free_audit_contract(
    physical_v2: Mapping[str, Any],
) -> None:
    views = physical_v2["views"]
    specs = physical_v2["specs_by_key"]
    job = next(
        job
        for job in _identity_jobs(physical_v2)
        if views[job.idempotency_key]["output_expert_role"] == "review"
    )
    view = views[job.idempotency_key]
    spec = specs[job.idempotency_key]
    wire = json.dumps(
        {
            "verdict": "fail",
            "faults": ["missing_identity_answer"],
            "correction": spec["target_template"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    output = vnext.fast_parse_candidate_output(job.source, wire)
    record = _record(job, view, spec, output)
    vnext.final_quality_validate(
        job,
        record,
        view,
        identity_derivative_spec=spec,
        inventory=physical_v2["inventory"],
        config=physical_v2["config"],
    )
    wrong = dict(output)
    wrong["value"] = dict(output["value"])
    wrong["value"]["faults"] = ["other_fault"]
    with pytest.raises(batch.AdapterError):
        vnext.final_quality_validate(
            job,
            _record(job, view, spec, wrong),
            view,
            identity_derivative_spec=spec,
            inventory=physical_v2["inventory"],
            config=physical_v2["config"],
        )


@pytest.mark.parametrize("output_role", ["humor", "serious", "angry"])
def test_style_derivative_rejects_appended_attribution(
    physical_v2: Mapping[str, Any],
    output_role: str,
) -> None:
    views = physical_v2["views"]
    specs = physical_v2["specs_by_key"]
    job = next(
        job
        for job in _identity_jobs(physical_v2)
        if views[job.idempotency_key]["output_expert_role"] == output_role
        and views[job.idempotency_key]["identity_class"] == "false_google_attribution"
    )
    view = views[job.idempotency_key]
    spec = specs[job.idempotency_key]
    contradictory = str(spec["target_template"]) + " Google actually trained me."
    output = {
        "kind": "natural_text",
        "value": contradictory,
    }
    with pytest.raises(batch.AdapterError):
        vnext.final_quality_validate(
            job,
            _record(job, view, spec, output),
            view,
            identity_derivative_spec=spec,
            inventory=physical_v2["inventory"],
            config=physical_v2["config"],
        )


def _copied_artifacts(
    physical_v2: Mapping[str, Any],
    tmp_path: Path,
) -> tuple[
    selector.AuthenticatedMetadataOverlay,
    selector.SelectorPreimages,
]:
    target = tmp_path / "minimal320_selector_preimages_v2"
    shutil.copytree(
        physical_v2["preimages"].directory,
        target,
    )
    return _overlay_from_directory(target), selector.load_selector_preimages(target)


@pytest.mark.parametrize("mutation", ["missing", "reorder", "duplicate"])
def test_preimage_missing_reorder_and_duplicate_are_rejected(
    physical_v2: Mapping[str, Any],
    tmp_path: Path,
    mutation: str,
) -> None:
    overlay, preimages = _copied_artifacts(physical_v2, tmp_path)
    rows_path = preimages.candidate_rows_path
    if mutation == "missing":
        rows_path.unlink()
    else:
        lines = rows_path.read_bytes().splitlines(keepends=True)
        if mutation == "reorder":
            lines[0], lines[1] = lines[1], lines[0]
        else:
            lines[1] = lines[0]
        rows_path.write_bytes(b"".join(lines))
    with pytest.raises(batch.AdapterError):
        selector.authenticate_selector_preimages(
            preimages,
            overlay=overlay,
            inventory=physical_v2["inventory"],
            config=physical_v2["config"],
            contract_sha256=physical_v2["contract_sha256"],
        )


def test_preimage_canonical_path_substitution_is_rejected(
    physical_v2: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    overlay, preimages = _copied_artifacts(physical_v2, tmp_path)
    alternate = preimages.directory / "alternate-candidates.jsonl"
    shutil.copy2(preimages.candidate_rows_path, alternate)
    shutil.copy2(
        preimages.candidate_rows_path.with_name(
            f"{preimages.candidate_rows_path.name}.sha256"
        ),
        alternate.with_name(f"{alternate.name}.sha256"),
    )
    substituted = replace(
        preimages,
        candidate_rows_path=alternate,
    )
    with pytest.raises(
        batch.AdapterError,
        match="canonical path drift",
    ):
        selector.authenticate_selector_preimages(
            substituted,
            overlay=overlay,
            inventory=physical_v2["inventory"],
            config=physical_v2["config"],
            contract_sha256=physical_v2["contract_sha256"],
        )


def test_candidate_identity_spoof_is_rejected(
    physical_v2: Mapping[str, Any],
) -> None:
    candidate = physical_v2["candidates"][0]
    value = candidate.as_dict()
    value["candidate_id_sha256"] = "0" * 64
    with pytest.raises(batch.AdapterError):
        selector.candidate_from_mapping(value)
