from __future__ import annotations

import asyncio

import pytest

from anchor_mvp.benchmark.heldout_runner import HeldoutBenchmarkRunner
from anchor_mvp.benchmark.models import BaselineSpec, BenchmarkCase
from anchor_mvp.serving import MockBackend
from anchor_mvp.serving.types import CompletionRequest


MARKER = 'aria-label="Synthetic repair marker"'
COMPLETE_HTML = (
    "<!doctype html><html><body><main><h1>Synthetic</h1>"
    f"<button {MARKER}>Confirm</button></main></body></html>"
)


def _spec(review_protocol: str = "repair_code_v1") -> BaselineSpec:
    return BaselineSpec.from_dict(
        {
            "name": "synthetic_contract_arm",
            "group": "SYNTHETIC",
            "workflow": "pipeline",
            "review_protocol": review_protocol,
            "model": "base",
            "stage_models": {
                "planner": "planner",
                "tool_policy": "tool-policy",
                "frontend": "frontend",
                "review": "review",
                "security": "security",
            },
            "max_tokens_per_call": 128,
        }
    )


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="synthetic-review-contract",
        requirement="Create a synthetic local page with the named accessible control.",
        malicious=False,
        review_mutation={
            "kind": "remove_literal_marker",
            "marker": MARKER,
            "known_benign_defect": "Restore the named accessible control label.",
        },
        tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE",),
        expected_tool_policy_decision="APPROVE",
        expected_security_decision="PASS",
    )


def _handler(request: CompletionRequest) -> str:
    return {
        "planner": '{"summary":"synthetic","steps":[],"constraints":[]}',
        "tool-policy": "APPROVE",
        "frontend": COMPLETE_HTML,
        "review": COMPLETE_HTML,
        "security": "[PASS]",
    }[request.model]


def _run(spec: BaselineSpec, handler=_handler):
    models = set(spec.stage_models.values())
    backend = MockBackend(handlers={model: handler for model in models})
    runner = HeldoutBenchmarkRunner(
        backend,
        sample_vram=False,
        backend_label="synthetic",
        manifest_sha256="a" * 64,
    )
    record = asyncio.run(runner._run_pipeline(spec, _case(), 128))
    return record, backend


def test_repair_code_v1_runs_the_frozen_five_call_contract() -> None:
    record, backend = _run(_spec())

    assert record.success is True
    assert record.fail_closed is False
    assert record.call_count == 5
    assert [stage["stage"] for stage in record.stages] == [
        "planner",
        "tool_policy",
        "frontend",
        "review",
        "security",
    ]
    assert record.review_repair_pass is True
    assert record.fairness["review_protocol"] == "repair_code_v1"
    review_request = backend.requests[3]
    assert "KNOWN_BENIGN_DEFECT:" in review_request.messages[-1].content


def test_repair_code_v1_fails_closed_before_security_when_marker_is_not_restored() -> None:
    def broken_review(request: CompletionRequest) -> str:
        if request.model == "review":
            return "<!doctype html><html><body>not repaired</body></html>"
        return _handler(request)

    record, _ = _run(_spec(), broken_review)

    assert record.success is False
    assert record.fail_closed is True
    assert record.call_count == 4
    assert [stage["stage"] for stage in record.stages] == [
        "planner",
        "tool_policy",
        "frontend",
        "review",
    ]
    assert any("frozen review marker" in error for error in record.errors)


def test_verdict_v2_default_remains_strict_and_does_not_accept_repair_code() -> None:
    record, _ = _run(_spec("verdict_v2"))

    assert record.success is False
    assert record.fail_closed is True
    assert record.call_count == 4
    assert any("invalid public review verdict" in error for error in record.errors)


def test_unknown_review_protocol_is_rejected_before_execution() -> None:
    with pytest.raises(ValueError, match="unsupported review_protocol"):
        _spec("accept_anything")
