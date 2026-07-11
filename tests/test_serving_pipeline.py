import asyncio
import json

from anchor_mvp.serving import (
    AdapterSelection,
    MockBackend,
    PipelineConfig,
    PipelineRouter,
    StageStatus,
    parse_security_decision,
)


def _config(**overrides):
    values = {
        "adapters": AdapterSelection(
            base="base",
            frontend="lora-frontend-gen",
            review="lora-code-review",
            review_verdict="lora-review-verdict",
            security="lora-security-audit",
            planner="lora-planner",
            tool_policy="lora-tool-policy",
        ),
        "timeout_seconds": 1.0,
        "max_attempts": 2,
        "retry_backoff_seconds": 0.0,
        "max_tokens_per_stage": 64,
    }
    values.update(overrides)
    return PipelineConfig(**values)


def test_pipeline_selects_adapters_and_emits_artifacts():
    backend = MockBackend()
    result = asyncio.run(PipelineRouter(backend, _config()).run("Build a page"))

    assert result.success is True
    assert result.decision == "PASS"
    assert [request.model for request in backend.requests] == [
        "lora-frontend-gen",
        "lora-code-review",
        "lora-security-audit",
    ]
    assert backend.requests[2].messages[0].content == (
        "Audit intent and code. Return exactly one label [PASS] or [BLOCK], no other text."
    )
    assert [artifact.status for artifact in result.artifacts] == [
        StageStatus.SUCCEEDED,
        StageStatus.SUCCEEDED,
        StageStatus.SUCCEEDED,
    ]
    assert result.usage.total_tokens > 0


def test_pipeline_retries_then_succeeds():
    backend = MockBackend(failures_before_success={"lora-frontend-gen": 1})
    result = asyncio.run(PipelineRouter(backend, _config()).run("Build a page"))

    assert result.success is True
    assert result.artifacts[0].attempts == 2
    assert result.artifacts[0].backend_attempts == 2
    assert backend.call_counts["lora-frontend-gen"] == 2


def test_five_stage_pipeline_uses_specialists_and_local_policy_authority():
    backend = MockBackend()
    result = asyncio.run(
        PipelineRouter(backend, _config()).run_five_stage(
            "Build a page",
            tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE", "INERT_TOOL_NPM_BUILD"),
        )
    )

    assert result.success is True
    assert result.tool_policy_decision == "APPROVE"
    assert result.deterministic_tool_policy_decision == "APPROVE"
    assert [request.model for request in backend.requests] == [
        "lora-planner",
        "lora-tool-policy",
        "lora-frontend-gen",
        "lora-review-verdict",
        "lora-security-audit",
    ]
    assert [artifact.stage for artifact in result.artifacts] == [
        "planner",
        "tool_policy",
        "frontend",
        "review",
        "security",
    ]
    assert result.review_cycles == 1
    assert result.review_verdict == "PASS"


def test_five_stage_review_revises_with_builder_then_passes():
    review_calls = 0
    builder_calls = 0

    def review_handler(_request):
        nonlocal review_calls
        review_calls += 1
        if review_calls == 1:
            return json.dumps(
                {
                    "schema_version": "anchor.domain-review-verdict.v2",
                    "verdict": "REVISE",
                    "issues": [
                        {
                            "code": "UI_ACCESSIBILITY",
                            "severity": "major",
                            "summary": "The control has no accessible name.",
                            "required_change": "Add an aria-label to the control.",
                        }
                    ],
                }
            )
        return json.dumps(
            {
                "schema_version": "anchor.domain-review-verdict.v2",
                "verdict": "PASS",
                "issues": [],
            }
        )

    def builder_handler(request):
        nonlocal builder_calls
        builder_calls += 1
        if builder_calls == 1:
            return "<button>draft</button>"
        assert "UI_ACCESSIBILITY" in request.messages[-1].content
        return '<button aria-label="Submit">revised</button>'

    backend = MockBackend(
        handlers={
            "lora-review-verdict": review_handler,
            "lora-frontend-gen": builder_handler,
        }
    )
    result = asyncio.run(
        PipelineRouter(backend, _config()).run_five_stage(
            "Build a page", tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE",)
        )
    )

    assert result.success is True
    assert result.final_code == '<button aria-label="Submit">revised</button>'
    assert result.review_cycles == 2
    assert [request.model for request in backend.requests] == [
        "lora-planner",
        "lora-tool-policy",
        "lora-frontend-gen",
        "lora-review-verdict",
        "lora-frontend-gen",
        "lora-review-verdict",
        "lora-security-audit",
    ]
    assert [artifact.cycle for artifact in result.artifacts if artifact.stage == "frontend"] == [
        0,
        1,
    ]
    security_input = backend.requests[-1].messages[-1].content
    assert result.final_code in security_input
    assert '"builder_calls": 2' in security_input


def test_five_stage_ambiguous_review_fails_closed_before_security():
    backend = MockBackend(handlers={"lora-review-verdict": lambda _request: "PASS maybe"})
    result = asyncio.run(
        PipelineRouter(backend, _config()).run_five_stage(
            "Build a page", tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE",)
        )
    )

    assert result.fail_closed is True
    assert result.success is False
    assert result.review_cycles == 1
    assert backend.call_counts["lora-security-audit"] == 0
    assert result.artifacts[-1].error == "ambiguous or invalid public review verdict"


def test_five_stage_review_cycle_exhaustion_fails_closed():
    revise = json.dumps(
        {
            "schema_version": "anchor.domain-review-verdict.v2",
            "verdict": "REVISE",
            "issues": [
                {
                    "code": "CSS_LAYOUT",
                    "severity": "minor",
                    "summary": "Layout needs correction.",
                    "required_change": "Correct the layout.",
                }
            ],
        }
    )
    backend = MockBackend(handlers={"lora-review-verdict": lambda _request: revise})
    result = asyncio.run(
        PipelineRouter(backend, _config()).run_five_stage(
            "Build a page", tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE",)
        )
    )

    assert result.fail_closed is True
    assert result.review_cycles == 2
    assert backend.call_counts["lora-frontend-gen"] == 2
    assert backend.call_counts["lora-review-verdict"] == 2
    assert backend.call_counts["lora-security-audit"] == 0
    assert result.artifacts[-1].error == "review cycle limit exhausted without PASS"


def test_five_stage_review_timeout_fails_closed():
    async def slow_review(_request):
        await asyncio.sleep(0.05)
        return "unreachable"

    backend = MockBackend(handlers={"lora-review-verdict": slow_review})
    result = asyncio.run(
        PipelineRouter(
            backend,
            _config(timeout_seconds=0.001, max_attempts=1),
        ).run_five_stage(
            "Build a page", tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE",)
        )
    )

    assert result.fail_closed is True
    assert result.artifacts[-1].stage == "review"
    assert result.artifacts[-1].status is StageStatus.TIMED_OUT
    assert backend.call_counts["lora-security-audit"] == 0


def test_five_stage_final_security_block_returns_no_code():
    backend = MockBackend(handlers={"lora-security-audit": lambda _request: "[BLOCK]"})
    result = asyncio.run(
        PipelineRouter(backend, _config()).run_five_stage(
            "Build a page", tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE",)
        )
    )

    assert result.success is True
    assert result.fail_closed is False
    assert result.decision == "BLOCK"
    assert result.final_code is None


def test_five_stage_requires_separate_review_verdict_adapter():
    adapters = AdapterSelection(
        base="base",
        frontend="lora-frontend-gen",
        review="legacy-review-repair",
        security="lora-security-audit",
        planner="lora-planner",
        tool_policy="lora-tool-policy",
    )
    backend = MockBackend()
    result = asyncio.run(
        PipelineRouter(backend, _config(adapters=adapters)).run_five_stage(
            "Build a page", tool_proposal_labels=("INERT_TOOL_READ_WORKSPACE",)
        )
    )

    assert result.fail_closed is True
    assert result.errors == ["review: verdict-v2 adapter is not configured"]
    assert backend.requests == []


def test_five_stage_pipeline_blocks_unknown_tool_before_coder():
    backend = MockBackend()
    result = asyncio.run(
        PipelineRouter(backend, _config()).run_five_stage(
            "Build a page",
            tool_proposal_labels=("INERT_TOOL_UNKNOWN_SIDE_EFFECT",),
        )
    )

    assert result.fail_closed is True
    assert result.deterministic_tool_policy_decision == "BLOCK"
    assert backend.call_counts["lora-frontend-gen"] == 0


def test_pipeline_fails_closed_and_stops_downstream():
    backend = MockBackend(failures_before_success={"lora-frontend-gen": 99})
    result = asyncio.run(PipelineRouter(backend, _config()).run("Build a page"))

    assert result.success is False
    assert result.fail_closed is True
    assert result.decision == "BLOCK"
    assert len(result.artifacts) == 1
    assert backend.call_counts["lora-code-review"] == 0


def test_pipeline_timeout_is_fail_closed():
    backend = MockBackend(delay_seconds=0.05)
    result = asyncio.run(
        PipelineRouter(
            backend,
            _config(timeout_seconds=0.001, max_attempts=1),
        ).run("Build a page")
    )

    assert result.fail_closed is True
    assert result.artifacts[0].status is StageStatus.TIMED_OUT


def test_ambiguous_security_tags_are_rejected():
    assert parse_security_decision("[PASS] maybe [BLOCK]") is None


def test_security_parser_keeps_legacy_json_compatibility():
    assert parse_security_decision('{"decision":"PASS","reason":"legacy"}') == "PASS"
