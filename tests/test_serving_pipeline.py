import asyncio

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
        "lora-code-review",
        "lora-security-audit",
    ]
    assert [artifact.stage for artifact in result.artifacts] == [
        "planner",
        "tool_policy",
        "frontend",
        "review",
        "security",
    ]


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
