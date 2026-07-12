import json
from hashlib import sha256

from anchor_mvp.tooling import (
    MockAgentExecutor,
    PublicDecisionStep,
    PublicOutcome,
    SampleSpec,
    ToolingHarness,
    canonical_json,
    write_attempts_jsonl,
)


OUTCOME = PublicOutcome(
    status="completed",
    decision_trace=(PublicDecisionStep("validation", "offline fixture", "kept patch"),),
    repair_summaries=(),
    final_summary="Offline fixture passed.",
)


def _make_project(path):
    path.mkdir()
    (path / "package.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "private": True,
                "scripts": {
                    "build": 'node -e "process.exit(0)"',
                    "test": 'node -e "process.exit(0)"',
                    "lint": 'node -e "process.exit(0)"',
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "index.js").write_text("export const value = 1;\n", encoding="utf-8")


def test_harness_isolates_sample_runs_validations_and_hashes_changes(tmp_path):
    source = tmp_path / "source"
    _make_project(source)
    harness = ToolingHarness(
        tmp_path / "runs",
        MockAgentExecutor(
            file_updates={"index.js": "export const value = 2;\n"},
            public_outcome=OUTCOME,
        ),
    )

    record = harness.run_sample(
        SampleSpec("sample/one", "Update value", source, ("build", "test", "lint"))
    )

    assert record.success is True
    assert record.workspace_id.startswith("sample-one--")
    assert (source / "index.js").read_text(encoding="utf-8").endswith("1;\n")
    assert [item.status for item in record.validations] == ["PASS", "PASS", "PASS"]
    assert [item.command for item in record.validations] == [
        "npm run build --if-present",
        "npm run test --if-present",
        "npm run lint --if-present",
    ]
    assert record.changed_files[0].path == "index.js"
    assert record.changed_files[0].before_sha256 != record.changed_files[0].after_sha256
    assert all(item.output_sha256 for item in record.validations)
    assert not list((tmp_path / "runs").iterdir())


def test_harness_can_retain_a_task_workspace_for_operator_debugging(tmp_path):
    source = tmp_path / "source"
    _make_project(source)
    runs = tmp_path / "runs"
    record = ToolingHarness(
        runs,
        MockAgentExecutor(public_outcome=OUTCOME),
        retain_workspace=True,
    ).run_sample(SampleSpec("keep/workspace", "Inspect fixture", source))

    retained = runs / record.workspace_id
    assert retained.is_dir()
    assert (retained / "index.js").is_file()


def test_missing_required_script_fails_closed_and_jsonl_is_canonical(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "package.json").write_text(
        json.dumps({"name": "no-build", "scripts": {}}), encoding="utf-8"
    )
    record = ToolingHarness(
        tmp_path / "runs", MockAgentExecutor(public_outcome=OUTCOME)
    ).run_sample(
        SampleSpec("b", "Do nothing", source)
    )

    assert record.success is False
    assert record.validations[0].name == "build"
    assert record.validations[0].status == "SKIP"
    output = write_attempts_jsonl([record], tmp_path / "attempts.jsonl")
    line = output.read_text(encoding="utf-8").strip()
    assert line == canonical_json(record)
    assert "prompt" not in line
    assert "thinking" not in line


def test_change_required_task_fails_when_agent_makes_no_change(tmp_path):
    source = tmp_path / "source"
    _make_project(source)
    record = ToolingHarness(
        tmp_path / "runs", MockAgentExecutor(public_outcome=OUTCOME)
    ).run_sample(
        SampleSpec("no-op", "Change index.js", source, requires_changes=True)
    )

    assert record.success is False
    assert record.changed_files == ()
    assert "no_changes" in record.error_codes


def test_rejected_mock_command_marks_record_failed(tmp_path):
    source = tmp_path / "source"
    _make_project(source)
    executor = MockAgentExecutor(
        commands=("npm install bad-package",), public_outcome=OUTCOME
    )

    record = ToolingHarness(tmp_path / "runs", executor).run_sample(
        SampleSpec("unsafe", "Try unsafe command", source)
    )

    assert record.success is False
    assert record.rejected_events == 1
    assert record.tool_trace[0].command is None
    assert record.tool_trace[0].command_sha256 is not None


def test_public_outcome_is_a_required_success_gate(tmp_path):
    source = tmp_path / "source"
    _make_project(source)

    record = ToolingHarness(tmp_path / "runs", MockAgentExecutor()).run_sample(
        SampleSpec("missing-outcome", "Do nothing", source)
    )

    assert record.success is False
    assert "public_outcome_missing" in record.error_codes


def test_protected_acceptance_files_cannot_be_rewritten_to_self_certify(tmp_path):
    source = tmp_path / "source"
    _make_project(source)
    package = source / "package.json"
    expected = sha256(package.read_bytes()).hexdigest()
    weakened = json.dumps(
        {
            "name": "weakened-fixture",
            "scripts": {
                "build": 'node -e "process.exit(0)"',
                "test": 'node -e "process.exit(0)"',
                "lint": 'node -e "process.exit(0)"',
            },
        }
    )
    record = ToolingHarness(
        tmp_path / "runs",
        MockAgentExecutor(file_updates={"package.json": weakened}, public_outcome=OUTCOME),
    ).run_sample(
        SampleSpec(
            "protected-contract",
            "Implement the task",
            source,
            ("build", "test", "lint"),
            protected_files=(("package.json", expected),),
        )
    )

    assert all(result.status == "PASS" for result in record.validations)
    assert record.success is False
    assert "protected_fixture_modified" in record.error_codes
