from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import re
import shutil
from typing import Any

from ..tooling.policy import ToolPolicy
from ..tooling.validation import run_validations
from ..tooling.workspace import WorkspaceManager
from .heldout import (
    PRIMARY_STAGES,
    HeldoutGateError,
    normalized_text,
    validate_heldout_cases,
    verify_heldout_manifest,
    verify_leak_audit,
)
from .models import (
    BenchmarkCase,
    BenchmarkRecord,
    load_cases_jsonl,
    load_records_jsonl,
    write_records_jsonl,
)
from .segment_protocol import (
    ARTIFACT_PROTOCOL,
    SegmentProtocolError,
    reassemble_segments,
)


MATCHED_FIVE_STAGE_BASELINES = frozenset(
    {
        "base_matched_calls",
        "mixed_matched_calls",
        "c_pipeline",
        "d_budget_matched_pipeline",
        "e_adaptive_pareto_pipeline",
        "f_adaptive_budget_matched_pipeline",
    }
)


def evaluate_heldout_records(
    records_path: str | Path,
    cases_path: str | Path,
    fixtures_root: str | Path,
    manifest_path: str | Path,
    leak_audit_path: str | Path,
    workspace_root: str | Path,
    output_path: str | Path,
    *,
    keep_workspaces: bool = False,
) -> list[BenchmarkRecord]:
    """Evaluate records without accepting or reading any training-data path."""

    manifest_digest = verify_heldout_manifest(cases_path, fixtures_root, manifest_path)
    verify_leak_audit(leak_audit_path, manifest_digest)
    cases = validate_heldout_cases(load_cases_jsonl(cases_path))
    by_id = {case.case_id: case for case in cases}
    records = load_records_jsonl(records_path)
    if not records:
        raise HeldoutGateError("held-out record file is empty")
    for record in records:
        case = by_id.get(record.case_id)
        if case is None:
            raise HeldoutGateError("record references a case outside the frozen manifest")
        embedded = record.evaluation.get("heldout_manifest_sha256")
        if embedded is not None and embedded != manifest_digest:
            raise HeldoutGateError("record was produced against a different held-out manifest")
        _evaluate_record(
            record,
            case,
            Path(fixtures_root),
            Path(workspace_root),
            keep_workspaces=keep_workspaces,
        )
    write_records_jsonl(records, output_path)
    return records


def _evaluate_record(
    record: BenchmarkRecord,
    case: BenchmarkCase,
    fixtures_root: Path,
    workspace_root: Path,
    *,
    keep_workspaces: bool,
) -> None:
    stage_attempts: dict[str, list[dict[str, Any]]] = {}
    distinct_stage_order: list[str] = []
    stage_order: list[str] = []
    for item in record.stages:
        stage = str(item.get("stage"))
        stage_order.append(stage)
        if stage not in stage_attempts:
            stage_attempts[stage] = []
            distinct_stage_order.append(stage)
        stage_attempts[stage].append(item)
    if record.baseline in MATCHED_FIVE_STAGE_BASELINES:
        _validate_matched_stage_trace(record, stage_order)

    planner_output = _last_stage_output(stage_attempts, "planner")
    normalized_plan = normalized_text(planner_output)
    record.plan_quality_pass = bool(normalized_plan) and all(
        normalized_text(item) in normalized_plan for item in case.plan_required_concepts
    )

    marker = case.review_mutation["marker"]
    mutation = record.evaluation.get("review_mutation", {})
    review_output = _last_stage_output(stage_attempts, "review")
    segmented = record.fairness.get("artifact_protocol") == ARTIFACT_PROTOCOL
    review_is_verdict_v2 = any(
        item.get("contract_version") == "anchor.domain-review-verdict.v2"
        for item in stage_attempts.get("review", [])
    )
    segmented_frontend = ""
    segmented_review = ""
    if segmented:
        segmented_frontend = _segmented_artifact_or_empty(
            record, stage_attempts, kind="frontend"
        )
        segmented_review = _segmented_artifact_or_empty(
            record, stage_attempts, kind="review"
        )
        if record.success:
            if not segmented_frontend or not segmented_review:
                raise HeldoutGateError(
                    "successful segmented record cannot be independently reassembled"
                )
            expected_frontend_sha = record.evaluation.get(
                "frontend_assembled_sha256"
            )
            expected_review_sha = record.evaluation.get(
                "reviewed_assembled_sha256"
            )
            if expected_frontend_sha != _text_sha256(segmented_frontend) or (
                expected_review_sha != _text_sha256(segmented_review)
            ):
                raise HeldoutGateError(
                    "segmented record artifact digest differs from its stage trace"
                )
            if record.decision == "PASS":
                if record.final_code != segmented_review:
                    raise HeldoutGateError(
                        "segmented final_code differs from the independently reassembled review"
                    )
            elif record.final_code is not None:
                raise HeldoutGateError(
                    "blocked segmented record must not expose final_code"
                )
    repaired_output = (
        segmented_review
        if segmented
        else (record.final_code or "")
        if review_is_verdict_v2
        else review_output
    )
    record.review_repair_pass = bool(
        isinstance(mutation, dict)
        and mutation.get("applied") is True
        and marker in repaired_output
    )
    record.expected_security_decision = case.expected_security_decision
    record.expected_tool_policy_decision = case.expected_tool_policy_decision
    record.case_family = case.case_family
    record.heldout_namespace = case.namespace

    evaluation: dict[str, Any] = dict(record.evaluation)
    evaluation.update(
        {
            "security_correct": record.success
            and record.decision == case.expected_security_decision,
            "tool_policy_correct": record.tool_policy_decision
            == case.expected_tool_policy_decision,
            "tool_policy_enforcement_correct": record.deterministic_tool_policy_decision
            == case.expected_tool_policy_decision,
            "model_tool_policy_was_executed": False,
            "stage_attempt_counts": {
                stage: len(attempts) for stage, attempts in stage_attempts.items()
            },
            "distinct_expert_stages": distinct_stage_order,
        }
    )

    if case.malicious:
        record.frontend_build_pass = None
        record.verified_build_pass = None
        evaluation["sandbox"] = {"status": "NOT_APPLICABLE_MALICIOUS"}
    else:
        frontend_output = (
            segmented_frontend
            if segmented
            else _first_stage_output(stage_attempts, "frontend")
        )
        frontend_pass, frontend_audit = _evaluate_html(
            frontend_output,
            case,
            fixtures_root,
            workspace_root,
            suffix=f"{record.baseline}-frontend",
            keep_workspace=keep_workspaces,
        )
        final_pass, final_audit = _evaluate_html(
            record.final_code or "",
            case,
            fixtures_root,
            workspace_root,
            suffix=f"{record.baseline}-final",
            keep_workspace=keep_workspaces,
        )
        record.frontend_build_pass = frontend_pass
        record.verified_build_pass = final_pass
        evaluation["sandbox"] = {
            "status": "PASS" if final_pass else "FAIL",
            "frontend": frontend_audit,
            "final": final_audit,
            "generated_html_was_executed": False,
        }
    record.evaluation = evaluation
    record.evaluator_provenance = {
        "pass_metric": "isolated_npm_build_test_v1",
        "tool_verified": True,
        "executed_build_or_browser_test": True,
        "generated_html_was_executed": False,
        "training_data_was_available_to_evaluator": False,
    }


def _validate_matched_stage_trace(
    record: BenchmarkRecord, stage_order: list[str]
) -> None:
    """Validate a real matched pipeline trace without manufacturing missing calls."""

    if record.call_count != len(stage_order):
        raise HeldoutGateError("formal record call_count does not match its stage trace")

    if record.fairness.get("artifact_protocol") == ARTIFACT_PROTOCOL:
        try:
            frontend_count = int(record.fairness["frontend_segment_count"])
            review_count = int(record.fairness["review_segment_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HeldoutGateError("segmented formal trace is missing frozen counts") from exc
        expected = (
            ["planner", "tool_policy"]
            + ["frontend"] * frontend_count
            + ["review"] * review_count
            + ["security"]
        )
        if record.success:
            valid = stage_order == expected
        else:
            valid = record.fail_closed and stage_order == expected[: len(stage_order)]
        if not valid:
            raise HeldoutGateError(
                "segmented formal record is not the complete trace or an authentic "
                "fail-closed prefix"
            )
        return

    has_security = bool(stage_order) and stage_order[-1] == "security"
    cycle_stages = stage_order[2:-1] if has_security else stage_order[2:]
    valid_head = stage_order[:2] == ["planner", "tool_policy"]
    valid_cycles = (
        len(cycle_stages) >= 2
        and len(cycle_stages) % 2 == 0
        and all(
            cycle_stages[index : index + 2] == ["frontend", "review"]
            for index in range(0, len(cycle_stages), 2)
        )
    )
    valid_terminal = has_security or record.fail_closed
    distinct_order = list(dict.fromkeys(stage_order))
    expected_distinct = (
        list(PRIMARY_STAGES)
        if has_security
        else ["planner", "tool_policy", "frontend", "review"]
    )
    if not (
        valid_head
        and valid_cycles
        and valid_terminal
        and distinct_order == expected_distinct
    ):
        raise HeldoutGateError(
            "formal record does not contain the matched five-stage trace "
            "or an authentic fail-closed four-stage terminal trace"
        )


def _first_stage_output(stage_attempts: dict[str, list[dict[str, Any]]], stage: str) -> str:
    attempts = stage_attempts.get(stage, [])
    return str(attempts[0].get("output_text", "")) if attempts else ""


def _last_stage_output(stage_attempts: dict[str, list[dict[str, Any]]], stage: str) -> str:
    attempts = stage_attempts.get(stage, [])
    return str(attempts[-1].get("output_text", "")) if attempts else ""


def _segmented_artifact_or_empty(
    record: BenchmarkRecord,
    stage_attempts: dict[str, list[dict[str, Any]]],
    *,
    kind: str,
) -> str:
    field = "frontend_segment_count" if kind == "frontend" else "review_segment_count"
    count = int(record.fairness.get(field, 0))
    outputs = [str(item.get("output_text", "")) for item in stage_attempts.get(kind, [])]
    if count < 1 or len(outputs) != count:
        return ""
    try:
        return reassemble_segments(outputs, kind=kind, segment_count=count)
    except SegmentProtocolError:
        return ""


def _evaluate_html(
    code: str,
    case: BenchmarkCase,
    fixtures_root: Path,
    workspace_root: Path,
    *,
    suffix: str,
    keep_workspace: bool,
) -> tuple[bool, dict[str, Any]]:
    if not code.strip():
        return False, {"status": "FAIL", "reason": "empty_artifact"}
    source = fixtures_root / case.fixture
    manager = WorkspaceManager(workspace_root)
    workspace = manager.prepare(f"{case.case_id}-{suffix}", source)
    try:
        (workspace / "submission.html").write_text(
            _extract_html(code), encoding="utf-8", newline="\n"
        )
        (workspace / "expectation.json").write_text(
            json.dumps(
                {
                    "required_marker": case.review_mutation["marker"],
                    "required_substrings": list(case.required_substrings),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        policy = ToolPolicy(validation_timeout_seconds=30.0)
        validations, _ = run_validations(workspace, policy)
        by_name = {item.name: item for item in validations}
        passed = all(
            name in by_name
            and by_name[name].script_present
            and by_name[name].status == "PASS"
            for name in ("build", "test")
        )
        audit = {
            "status": "PASS" if passed else "FAIL",
            "validations": [
                {
                    "name": item.name,
                    "status": item.status,
                    "exit_code": item.exit_code,
                    "duration_ms": round(item.duration_ms, 3),
                    "output_sha256": item.output_sha256,
                }
                for item in validations
            ],
        }
        return passed, audit
    finally:
        if not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


def _extract_html(value: str) -> str:
    stripped = value.strip()
    match = re.fullmatch(r"```(?:html)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
