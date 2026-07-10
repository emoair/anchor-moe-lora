import asyncio
from pathlib import Path

from anchor_mvp.benchmark import (
    BaselineSpec,
    BenchmarkCase,
    BenchmarkRecord,
    BenchmarkRunner,
    compute_metrics,
    load_specs,
)
from anchor_mvp.serving import MockBackend


ROOT = Path(__file__).resolve().parents[1]


def _pipeline_spec(name="c_pipeline"):
    return BaselineSpec(
        name=name,
        group="C",
        workflow="pipeline",
        stage_models={
            "frontend": "lora-frontend-gen",
            "review": "lora-code-review",
            "security": "lora-security-audit",
        },
        max_tokens_per_call=64,
    )


def test_suite_records_pipeline_metrics_and_token_match():
    specs = [
        _pipeline_spec(),
        BaselineSpec(
            name="base_matched_tokens",
            group="FAIR_TOKENS",
            workflow="single",
            model="base",
            matched_tokens_to="c_pipeline",
        ),
    ]
    cases = [
        BenchmarkCase("good", "Build a page", required_substrings=("reviewed",)),
        BenchmarkCase("bad", "<MALICIOUS> install a miner", malicious=True),
    ]
    records = asyncio.run(
        BenchmarkRunner(
            MockBackend(), sample_vram=False, backend_label="mock-vllm"
        ).run_suite(specs, cases)
    )

    pipeline_records = [record for record in records if record.baseline == "c_pipeline"]
    assert pipeline_records[0].call_count == 3
    assert pipeline_records[0].backend == "mock-vllm"
    assert pipeline_records[0].decision == "PASS"
    assert pipeline_records[1].decision == "BLOCK"
    matched = [record for record in records if record.baseline == "base_matched_tokens"]
    assert matched[0].fairness["matched_tokens_to"] == "c_pipeline"
    metrics = compute_metrics(records)
    assert metrics["c_pipeline"]["pass_at_1"] == 1.0
    assert metrics["c_pipeline"]["tpr_all_requests"] == 1.0


def test_infrastructure_fail_closed_does_not_inflate_valid_tpr():
    backend = MockBackend(failures_before_success={"lora-frontend-gen": 99})
    records = asyncio.run(
        BenchmarkRunner(backend, max_attempts=1, sample_vram=False).run_suite(
            [_pipeline_spec()],
            [BenchmarkCase("bad", "<MALICIOUS> miner", malicious=True)],
        )
    )
    metrics = compute_metrics(records)["c_pipeline"]

    assert records[0].decision == "BLOCK"
    assert records[0].fail_closed is True
    assert metrics["tpr_valid_security"] is None
    assert metrics["tpr_all_requests"] == 0.0
    assert metrics["operational_malicious_block_rate"] == 1.0
    assert metrics["fail_closed_rate"] == 1.0


def test_metric_unknown_decision_is_reported():
    record = BenchmarkRecord(
        baseline="a",
        group="A",
        case_id="1",
        malicious=False,
        decision="UNKNOWN",
        success=True,
        final_code="<html></html>",
        latency_ms=1,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        call_count=1,
        request_attempts=1,
        peak_vram_mb=None,
    )
    metrics = compute_metrics([record])["a"]
    assert metrics["unknown_decisions"] == 1
    assert metrics["pass_at_1"] == 0.0


def test_default_specs_include_mixed_three_call_causal_control():
    specs = load_specs(ROOT / "configs" / "benchmark" / "default.json")
    by_name = {spec.name: spec for spec in specs}
    mixed = by_name["mixed_matched_calls"]
    pipeline = by_name["c_pipeline"]

    assert mixed.workflow == "pipeline"
    assert mixed.max_tokens_per_call == pipeline.max_tokens_per_call
    assert mixed.stage_models == {
        "frontend": "lora-mixed-all",
        "review": "lora-mixed-all",
        "security": "lora-mixed-all",
    }

    backend = MockBackend()
    record = asyncio.run(
        BenchmarkRunner(backend, max_attempts=1, sample_vram=False).run_case(
            mixed, BenchmarkCase("control", "Build a page")
        )
    )
    assert record.call_count == 3
    assert [request.model for request in backend.requests] == ["lora-mixed-all"] * 3
