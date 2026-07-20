from __future__ import annotations

import asyncio
import json

import pytest

from anchor_mvp.benchmark.heldout_runner import HeldoutBenchmarkRunner
from anchor_mvp.benchmark.heldout import HeldoutGateError
from anchor_mvp.benchmark.heldout_eval import _validate_matched_stage_trace
from anchor_mvp.benchmark.metrics import compute_a100_indices
from anchor_mvp.benchmark.models import BaselineSpec, BenchmarkCase
from anchor_mvp.benchmark.segment_protocol import (
    ARTIFACT_PROTOCOL,
    PROMPT_BUNDLE_SHA256,
    SEGMENT_CONTRACT_VERSION,
    SegmentContract,
    SegmentProtocolError,
    parse_segment,
    protocol_binding_metadata,
    reassemble_segments,
    split_review_candidate,
    validate_protocol_binding_metadata,
)
from anchor_mvp.serving import MockBackend
from anchor_mvp.serving.types import CompletionRequest


MARKER = 'aria-label="Segment repair marker"'
COMPLETE_TSX = (
    "export default function App(){return (<main><h1>Segmented</h1>"
    f"<button {MARKER}>Confirm</button></main>);}}"
)


def _wrapped(kind: str, index: int, count: int, payload: str) -> str:
    prefix = "anchor-tsx-segment" if kind == "frontend" else "anchor-tsx-review-segment"
    return (
        f"/*<{prefix} {index + 1}/{count}>*/\n"
        f"{payload}\n"
        f"/*</{prefix}>*/"
    )


def _spec() -> BaselineSpec:
    return BaselineSpec.from_dict(
        {
            "name": "synthetic-segmented-arm",
            "group": "SYNTHETIC",
            "workflow": "pipeline",
            "review_protocol": "segmented_repair_v1",
            "artifact_protocol": ARTIFACT_PROTOCOL,
            "segment_contract_version": SEGMENT_CONTRACT_VERSION,
            "frontend_segment_count": 2,
            "review_segment_count": 2,
            "model": "base",
            "stage_models": {
                "planner": "planner",
                "tool_policy": "tool-policy",
                "frontend": "frontend",
                "review": "review",
                "security": "security",
            },
            "max_tokens_per_call": 512,
        }
    )


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="synthetic-segment-contract",
        requirement="Create a local TSX page with one named accessible control.",
        malicious=False,
        review_mutation={
            "kind": "remove_literal_marker",
            "marker": MARKER,
            "known_benign_defect": "Restore the accessible control label.",
        },
        tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE",),
        expected_tool_policy_decision="APPROVE",
        expected_security_decision="PASS",
    )


def _segment_index(request: CompletionRequest) -> int:
    user = request.messages[-1].content
    if user.startswith("GENERATE_TSX_SEGMENT|"):
        return int(json.loads(user.split("|", 1)[1])["segment_index"])
    header = user.splitlines()[0]
    return int(header.split("|", 1)[1].split("/", 1)[0]) - 1


def _handler(request: CompletionRequest) -> str:
    if request.model == "planner":
        return json.dumps(
            {
                "summary": "build a segmented TSX page",
                "constraints": ["single TSX artifact"],
                "steps": [
                    {"id": "P1", "goal": "build", "deliverable": "TSX page"}
                ],
            }
        )
    if request.model == "tool-policy":
        return "APPROVE"
    if request.model == "frontend":
        index = _segment_index(request)
        payloads = split_review_candidate(COMPLETE_TSX, 2)
        return _wrapped("frontend", index, 2, payloads[index])
    if request.model == "review":
        index = _segment_index(request)
        payloads = split_review_candidate(COMPLETE_TSX, 2)
        return _wrapped("review", index, 2, payloads[index])
    if request.model == "security":
        return "[PASS]"
    raise AssertionError(request.model)


def _run(handler=_handler):
    spec = _spec()
    backend = MockBackend(
        handlers={model: handler for model in set(spec.stage_models.values())}
    )
    runner = HeldoutBenchmarkRunner(
        backend,
        sample_vram=False,
        backend_label="synthetic-segmented",
        manifest_sha256="a" * 64,
    )
    return asyncio.run(runner._run_pipeline(spec, _case(), 512)), backend


def test_segment_wrappers_reassemble_losslessly() -> None:
    payloads = ["first\n", "second"]
    outputs = [_wrapped("frontend", index, 2, payload) for index, payload in enumerate(payloads)]

    assert parse_segment(
        outputs[0], kind="frontend", segment_index=0, segment_count=2
    ) == payloads[0]
    assert reassemble_segments(outputs, kind="frontend", segment_count=2) == "first\nsecond"


def test_segmented_runner_uses_five_logical_stages_and_reassembles_code() -> None:
    record, backend = _run()

    assert record.success is True
    assert record.fail_closed is False
    assert record.final_code == COMPLETE_TSX
    assert record.call_count == 7
    assert [stage["stage"] for stage in record.stages] == [
        "planner",
        "tool_policy",
        "frontend",
        "frontend",
        "review",
        "review",
        "security",
    ]
    assert record.fairness["matched_stage_order"] == [
        "planner",
        "tool_policy",
        "frontend",
        "review",
        "security",
    ]
    assert record.fairness["target_artifact_digest_used_as_input"] is False
    assert record.review_repair_pass is None
    assert all(request.max_tokens == 512 for request in backend.requests)
    assert all([message.role for message in request.messages] == ["user"] for request in backend.requests)

    review_requests = [request for request in backend.requests if request.model == "review"]
    assert review_requests
    for request in review_requests:
        prompt = request.messages[0].content
        assert MARKER not in prompt
        assert "Restore the accessible control label." not in prompt
        assert "DEFECT:independently inspect this candidate excerpt" in prompt

    contract = SegmentContract(
        artifact_protocol=ARTIFACT_PROTOCOL,
        contract_version=SEGMENT_CONTRACT_VERSION,
        frontend_segments=2,
        review_segments=2,
    )
    validate_protocol_binding_metadata(
        record.fairness,
        contract,
        max_completion_tokens_per_physical_call=512,
    )
    assert record.fairness["prompt_bundle_sha256"] == PROMPT_BUNDLE_SHA256
    assert record.fairness["segment_contract_sha256"] == protocol_binding_metadata(
        contract, max_completion_tokens_per_physical_call=512
    )["segment_contract_sha256"]


def test_invalid_segment_fails_closed_without_calling_review_or_security() -> None:
    def broken(request: CompletionRequest) -> str:
        if request.model == "frontend" and _segment_index(request) == 1:
            return "plain unwrapped code"
        return _handler(request)

    record, _ = _run(broken)

    assert record.success is False
    assert record.fail_closed is True
    assert record.call_count == 4
    assert [stage["stage"] for stage in record.stages] == [
        "planner",
        "tool_policy",
        "frontend",
        "frontend",
    ]
    assert any("invalid wrapper" in error for error in record.errors)

    _validate_matched_stage_trace(record, [stage["stage"] for stage in record.stages])


def test_segmented_trace_rejects_non_prefix_stage_reordering() -> None:
    record, _ = _run()
    record.baseline = "c_pipeline"
    reordered = [stage["stage"] for stage in record.stages]
    reordered[3], reordered[4] = reordered[4], reordered[3]

    with pytest.raises(HeldoutGateError, match="segmented formal record"):
        _validate_matched_stage_trace(record, reordered)


def test_wrong_declared_position_is_rejected() -> None:
    try:
        parse_segment(
            _wrapped("frontend", 1, 2, "code"),
            kind="frontend",
            segment_index=0,
            segment_count=2,
        )
    except SegmentProtocolError as exc:
        assert "invalid wrapper" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("wrong segment position was accepted")


def test_duplicate_closing_wrapper_is_rejected() -> None:
    output = _wrapped(
        "review",
        0,
        1,
        "payload\n/*</anchor-tsx-review-segment>*/",
    )

    with pytest.raises(SegmentProtocolError, match="duplicate segment wrapper"):
        parse_segment(output, kind="review", segment_index=0, segment_count=1)


def test_review_segment_failure_is_a_strict_fail_closed_prefix() -> None:
    def broken(request: CompletionRequest) -> str:
        if request.model == "review" and _segment_index(request) == 1:
            return _handler(request) + "\n/*</anchor-tsx-review-segment>*/"
        return _handler(request)

    record, _ = _run(broken)

    assert record.success is False
    assert record.fail_closed is True
    assert record.call_count == 6
    assert [stage["stage"] for stage in record.stages] == [
        "planner",
        "tool_policy",
        "frontend",
        "frontend",
        "review",
        "review",
    ]
    assert any("duplicate segment wrapper" in error for error in record.errors)


def test_hidden_marker_is_not_a_runtime_acceptance_gate() -> None:
    without_marker = COMPLETE_TSX.replace(MARKER, "")

    def no_repair(request: CompletionRequest) -> str:
        if request.model == "review":
            index = _segment_index(request)
            payloads = split_review_candidate(without_marker, 2)
            return _wrapped("review", index, 2, payloads[index])
        return _handler(request)

    record, _ = _run(no_repair)

    assert record.success is True
    assert record.fail_closed is False
    assert record.review_repair_pass is None
    assert record.final_code == without_marker


def test_protocol_binding_validation_rejects_hash_tampering() -> None:
    contract = SegmentContract(
        artifact_protocol=ARTIFACT_PROTOCOL,
        contract_version=SEGMENT_CONTRACT_VERSION,
        frontend_segments=10,
        review_segments=10,
    )
    metadata = protocol_binding_metadata(
        contract, max_completion_tokens_per_physical_call=512
    )
    metadata["prompt_bundle_sha256"] = "0" * 64

    with pytest.raises(SegmentProtocolError, match="prompt_bundle_sha256"):
        validate_protocol_binding_metadata(
            metadata,
            contract,
            max_completion_tokens_per_physical_call=512,
        )


def test_a100_indices_fix_native_q4_at_100_without_dividing_by_zero() -> None:
    metrics = {
        "base_matched_calls": {
            "build_pass_at_1": 0.5,
            "mean_latency_ms": 20.0,
            "fpr_valid_security": 0.0,
        },
        "c_pipeline": {
            "build_pass_at_1": 0.75,
            "mean_latency_ms": 30.0,
            "fpr_valid_security": 0.1,
        },
    }

    indexed = compute_a100_indices(metrics)

    assert indexed["base_matched_calls"]["build_pass_at_1"] == 100.0
    assert indexed["c_pipeline"]["build_pass_at_1"] == 150.0
    assert indexed["c_pipeline"]["mean_latency_ms"] == 150.0
    assert indexed["c_pipeline"]["fpr_valid_security"] is None
