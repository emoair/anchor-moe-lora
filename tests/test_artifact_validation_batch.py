from __future__ import annotations

import json
from pathlib import Path

import anchor_mvp.data.artifact_validation as artifact_validation
from anchor_mvp.data.artifact_validation import (
    validate_tsx_fragment,
    validate_tsx_fragments,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "data" / "fixtures" / "tsx-fragment"
VALID = """import { useState } from 'react';
export function Card() {
  const [ready] = useState(true);
  return <main>{ready ? 'Ready' : 'Waiting'}</main>;
}
"""
INVALID = "export function Broken(){return (<div><span>bad</div>)}"


def test_batch_matches_single_artifact_semantics(tmp_path: Path) -> None:
    single_valid = validate_tsx_fragment(
        VALID,
        fixture_root=FIXTURE,
        workspace_root=tmp_path / "single-valid",
        timeout_seconds=15,
    )
    single_invalid = validate_tsx_fragment(
        INVALID,
        fixture_root=FIXTURE,
        workspace_root=tmp_path / "single-invalid",
        timeout_seconds=15,
    )

    batched = validate_tsx_fragments(
        [VALID, INVALID],
        fixture_root=FIXTURE,
        workspace_root=tmp_path / "batch",
        timeout_seconds=15,
    )

    assert [item["passed"] for item in batched] == [
        single_valid["passed"],
        single_invalid["passed"],
    ]
    assert [item["code_sha256"] for item in batched] == [
        single_valid["code_sha256"],
        single_invalid["code_sha256"],
    ]


def test_batch_deduplicates_digest_and_launches_one_validation_run(
    tmp_path: Path, monkeypatch
) -> None:
    original = artifact_validation.run_validations_with_output
    observed_batch_sizes: list[int] = []

    def observe(workspace: Path, policy):
        payload = json.loads(
            (workspace / "submissions.json").read_text(encoding="utf-8")
        )
        observed_batch_sizes.append(len(payload["submissions"]))
        return original(workspace, policy)

    monkeypatch.setattr(
        artifact_validation,
        "run_validations_with_output",
        observe,
    )
    reports = validate_tsx_fragments(
        [VALID, VALID, INVALID, INVALID],
        fixture_root=FIXTURE,
        workspace_root=tmp_path / "workspaces",
        timeout_seconds=15,
    )

    assert observed_batch_sizes == [2]
    assert [item["passed"] for item in reports] == [True, True, False, False]
    assert reports[0] == reports[1]
    assert reports[2] == reports[3]


def test_generated_code_is_inspected_as_data_and_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "generated-code-executed.txt"
    malicious = f"""export function Card() {{
  require('node:fs').writeFileSync('{marker.as_posix()}', 'executed');
  return <main>Generated code remains inert</main>;
}}
"""

    report = validate_tsx_fragments(
        [malicious],
        fixture_root=FIXTURE,
        workspace_root=tmp_path / "workspaces",
        timeout_seconds=15,
    )[0]

    assert report["passed"] is True
    assert not marker.exists()


def test_batch_workspace_is_removed_after_validation(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    validate_tsx_fragments(
        [VALID, INVALID],
        fixture_root=FIXTURE,
        workspace_root=workspace_root,
        timeout_seconds=15,
    )

    assert workspace_root.is_dir()
    assert list(workspace_root.iterdir()) == []


def test_batch_protocol_failure_rejects_every_artifact_and_cleans_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        artifact_validation,
        "run_validations_with_output",
        lambda workspace, policy: ((), (), ()),
    )
    workspace_root = tmp_path / "workspaces"

    reports = validate_tsx_fragments(
        [VALID, INVALID],
        fixture_root=FIXTURE,
        workspace_root=workspace_root,
        timeout_seconds=15,
    )

    assert all(item["passed"] is False for item in reports)
    assert {item["reason"] for item in reports} == {"validator_infrastructure_failure"}
    assert list(workspace_root.iterdir()) == []
