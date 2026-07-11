import json
from dataclasses import replace

import pytest

from anchor_mvp.tooling import (
    MockAgentExecutor,
    PublicDecisionStep,
    PublicOutcome,
    SampleSpec,
    ToolingHarness,
    canonical_json,
    merge_gold_jsonl,
    persist_attempts_and_gold,
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


def test_failed_attempt_is_quarantined_and_does_not_reserve_gold_sample_id(tmp_path):
    accepted = _record(tmp_path, "retry")
    failed = replace(
        accepted,
        success=False,
        workspace_id="failed-attempt",
        public_outcome=None,
        error_codes=("public_outcome_missing",),
    )
    attempts = tmp_path / "attempts.jsonl"
    gold = tmp_path / "gold.jsonl"

    assert persist_attempts_and_gold(
        [failed], attempts_path=attempts, gold_path=gold
    ) == ()
    assert not gold.exists()
    accepted_records = persist_attempts_and_gold(
        [accepted], attempts_path=attempts, gold_path=gold
    )

    assert accepted_records == (accepted,)
    attempt_rows = [json.loads(line) for line in attempts.read_text(encoding="utf-8").splitlines()]
    gold_rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    assert [row["success"] for row in attempt_rows] == [False, True]
    assert [row["sample_id"] for row in gold_rows] == ["retry"]


def test_gold_merge_rejects_failed_or_noncompleted_records(tmp_path):
    accepted = _record(tmp_path, "gate")
    failed = replace(accepted, success=False)
    partial = replace(
        accepted,
        public_outcome=replace(accepted.public_outcome, status="partial"),
    )

    with pytest.raises(ValueError, match="non-accepted"):
        merge_gold_jsonl([failed], tmp_path / "failed.jsonl")
    with pytest.raises(ValueError, match="non-accepted"):
        merge_gold_jsonl([partial], tmp_path / "partial.jsonl")


def test_existing_legacy_failure_blocks_gold_until_explicit_migration(tmp_path):
    accepted = _record(tmp_path, "accepted")
    failed = replace(
        accepted,
        sample_id="legacy-failure",
        success=False,
        public_outcome=None,
    )
    output = tmp_path / "legacy-gold.jsonl"
    output.write_text(canonical_json(failed) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="migrate it to the attempt ledger"):
        merge_gold_jsonl([accepted], output)
