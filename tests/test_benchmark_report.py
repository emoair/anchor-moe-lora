import csv
import json
from anchor_mvp.benchmark import (
    BenchmarkRecord,
    compute_metrics,
    generate_report,
    write_records_jsonl,
)


def _records():
    records = []
    for index, baseline in enumerate(
        ["base_matched_calls", "mixed_matched_calls", "c_pipeline"]
    ):
        records.extend(
            [
                BenchmarkRecord(
                    baseline=baseline,
                    group="FAIR_CALLS" if baseline != "c_pipeline" else "C",
                    case_id=f"{baseline}-benign",
                    malicious=False,
                    decision="PASS",
                    success=True,
                    final_code="<html>ok</html>",
                    latency_ms=10 + index,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    call_count=3,
                    request_attempts=3,
                    peak_vram_mb=None if index == 0 else 7000 + index,
                    required_substrings=("<html>",),
                    backend="mock",
                ),
                BenchmarkRecord(
                    baseline=baseline,
                    group="FAIR_CALLS" if baseline != "c_pipeline" else "C",
                    case_id=f"{baseline}-malicious",
                    malicious=True,
                    decision="BLOCK",
                    success=True,
                    final_code=None,
                    latency_ms=11 + index,
                    prompt_tokens=11,
                    completion_tokens=4,
                    total_tokens=15,
                    call_count=3,
                    request_attempts=3,
                    peak_vram_mb=None if index == 0 else 7001 + index,
                    backend="mock",
                ),
            ]
        )
    return records


def test_report_generates_auditable_markdown_csv_and_svg(tmp_path):
    records = _records()
    records_path = tmp_path / "records.jsonl"
    metrics_path = tmp_path / "metrics.json"
    write_records_jsonl(records, records_path)
    supplied = compute_metrics(records)
    supplied["c_pipeline"]["error_rate"] = 0.5
    metrics_path.write_text(json.dumps(supplied), encoding="utf-8")

    paths = generate_report(records_path, metrics_path, tmp_path / "report")

    summary = paths.summary.read_text(encoding="utf-8")
    svg = paths.chart_svg.read_text(encoding="utf-8")
    with paths.metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert "not true Pass@1" in summary
    assert "OpenCode/tool-verified" in summary
    assert "sha256:" in summary
    assert "Supplied-metrics reconciliation" in summary
    assert "c_pipeline.error_rate" in summary
    assert summary.index("base_matched_calls") < summary.index("mixed_matched_calls")
    assert summary.index("mixed_matched_calls") < summary.index("c_pipeline")
    assert rows[0]["baseline"] == "base_matched_calls"
    assert rows[0]["peak_vram_mb"] == "N/A"
    assert rows[0]["comparison_role"] == "primary causal"
    assert "Structural pass proxy (NOT true Pass@1)" in svg
    assert "Valid-security TPR" in svg
    assert "Operational malicious block rate" in svg
    assert "Peak sampled VRAM (MiB)" in svg
    assert ">N/A<" in svg


def test_record_round_trip_preserves_evaluator_provenance(tmp_path):
    record = _records()[0]
    record.evaluator_provenance = {
        "pass_metric": "future_opencode_build_v1",
        "tool_verified": True,
        "executed_build_or_browser_test": True,
    }
    path = tmp_path / "records.jsonl"
    write_records_jsonl([record], path)

    from anchor_mvp.benchmark import load_records_jsonl

    loaded = load_records_jsonl(path)[0]
    assert loaded.evaluator_provenance == record.evaluator_provenance
    assert loaded.verified_build_pass is None
