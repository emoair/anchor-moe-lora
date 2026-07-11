import json

import pytest

from anchor_mvp.tooling import (
    MockAgentExecutor,
    PublicDecisionStep,
    PublicOutcome,
    SampleSpec,
    ToolingHarness,
    merge_gold_jsonl,
)


def _record(tmp_path, sample_id, summary="done"):
    source = tmp_path / f"source-{sample_id}"
    source.mkdir(exist_ok=True)
    (source / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "build": 'node -e "process.exit(0)"',
                    "test": 'node -e "process.exit(0)"',
                    "lint": 'node -e "process.exit(0)"',
                }
            }
        ),
        encoding="utf-8",
    )
    outcome = PublicOutcome(
        status="completed",
        decision_trace=(PublicDecisionStep("check", "evidence", "action"),),
        repair_summaries=(),
        final_summary=summary,
    )
    return ToolingHarness(
        tmp_path / "runs", MockAgentExecutor(public_outcome=outcome)
    ).run_sample(SampleSpec(sample_id, "task", source))


def test_merge_is_append_only_and_idempotent(tmp_path):
    first = _record(tmp_path, "first")
    second = _record(tmp_path, "second")
    output = tmp_path / "gold.jsonl"

    merge_gold_jsonl([first], output)
    merge_gold_jsonl([first, second], output)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [item["sample_id"] for item in records] == ["first", "second"]


def test_merge_refuses_to_overwrite_existing_sample(tmp_path):
    original = _record(tmp_path, "same", "original")
    changed = _record(tmp_path, "same", "changed")
    output = tmp_path / "gold.jsonl"
    merge_gold_jsonl([original], output)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        merge_gold_jsonl([changed], output)
