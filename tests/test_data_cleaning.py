from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.data.cleaning import (  # noqa: E402
    build_inert_security_fixture,
    contains_secret_material,
    extract_frontend_payload,
    extract_json_object,
    redact_active_payload_material,
    sanitize_security_seed,
    validate_safe_payload,
)
from anchor_mvp.data.schema import DataValidationError, SeedDemand  # noqa: E402
from anchor_mvp.data.schema import DistilledRecord  # noqa: E402
from anchor_mvp.data.sops import load_sop  # noqa: E402


def test_extracts_json_after_preface_and_fence() -> None:
    value = extract_json_object('Sure! ```json\n{"ok": true}\n```')
    assert value == {"ok": True}


def test_security_seed_replaces_active_material() -> None:
    seed = SeedDemand("seed-1", "Unsafe", "Render <script>alert(1)</script> directly")
    safe = sanitize_security_seed(seed)
    assert "<script" not in safe.request.casefold()
    assert "[DEFENSIVE_ACTIVE_CONTENT_PLACEHOLDER]" in safe.request


@pytest.mark.parametrize(
    "value",
    [
        "ark-SyntheticCredential_0123456789ABCDEF",
        "prefix ark-SyntheticCredential_0123456789ABCDEF suffix",
        {"nested": "ark-SyntheticCredential_0123456789ABCDEF"},
    ],
)
def test_detects_ark_shaped_secret_material(value) -> None:
    assert contains_secret_material(value)


def test_planning_payload_redactor_preserves_structure_and_removes_markers() -> None:
    source = {
        "decision_trace": [
            {
                "check": "avoid active forms",
                "evidence": "Do not emit eval(user_input) or javascript: links.",
                "action": "Use inert text rendering.",
            }
        ],
        "output": {"summary": "Keep the rest of this explanation."},
    }
    redacted, count = redact_active_payload_material(source)
    assert count == 2
    assert redacted["output"] == source["output"]
    assert "eval(" not in str(redacted).casefold()
    assert "javascript:" not in str(redacted).casefold()
    assert str(redacted).count("DEFENSIVE_ACTIVE_CONTENT_PLACEHOLDER") == 2


def test_security_payload_rejects_active_material() -> None:
    with pytest.raises(DataValidationError, match="active payloads"):
        validate_safe_payload(
            "security",
            {"output": {"decision": "BLOCK", "rationale": "javascript: placeholder"}},
        )


@pytest.mark.parametrize(
    "reviewed_code",
    [
        "element.innerHTML = value",
        "<img onerror=placeholder>",
    ],
)
def test_security_reviewed_code_requires_safe_code_or_placeholders(
    reviewed_code: str,
) -> None:
    with pytest.raises(DataValidationError, match="safe code or inert placeholders"):
        validate_safe_payload(
            "security",
            {
                "input": {"reviewed_code": reviewed_code},
                "output": {"decision": "BLOCK", "rationale": "Unsafe reviewed code."},
            },
        )


def test_security_reviewed_code_allows_react_event_props() -> None:
    validate_safe_payload(
        "security",
        {
            "input": {
                "reviewed_code": (
                    "export const Form = () => "
                    "<form onSubmit={submit}><button onClick={save}>Save</button></form>"
                )
            },
            "output": {
                "decision": "PASS",
                "rationale": "JSX event props are not inline HTML.",
            },
        },
    )


@pytest.mark.parametrize("task_type", ["frontend", "review"])
def test_code_tasks_reject_oversized_teacher_output(task_type: str) -> None:
    with pytest.raises(DataValidationError, match="12000-character hard limit"):
        validate_safe_payload(task_type, {"output": {"code": "x" * 12001}})  # type: ignore[arg-type]


def test_frontend_accepts_one_fenced_code_artifact() -> None:
    payload, source = extract_frontend_payload(
        "```tsx\nexport const Card = () => <main />\n```"
    )
    assert source == "fenced_code"
    assert payload["output"]["code"].startswith("export")


def test_hidden_reasoning_fields_are_rejected() -> None:
    payload = {
        "thinking": "private reasoning",
        "decision_trace": [
            {"check": "input", "evidence": "request", "action": "implement"}
        ],
        "output": {"code": "export default 1"},
    }
    with pytest.raises(DataValidationError, match="hidden reasoning"):
        DistilledRecord.from_teacher_payload(
            payload=payload,
            task_type="frontend",
            seed=SeedDemand("seed-1", "title", "request"),
            sop=load_sop(ROOT / "skills" / "frontend.md"),
            teacher_model="mock",
            teacher_base_url="mock://local",
            teacher_protocol="mock",
            generation_params={},
            template_sha256="abc",
        )


@pytest.mark.parametrize(
    "hidden_key",
    [
        "reasoning",
        "reasoning_content",
        "thinking",
        "thinking-details",
        "cot",
        "chain_of_thought",
    ],
)
def test_hidden_reasoning_fields_are_rejected_recursively(hidden_key: str) -> None:
    payload = {
        "decision_trace": [
            {"check": "input", "evidence": "request", "action": "implement"}
        ],
        "output": {
            "language": "tsx",
            "code": "export default 1",
            "metadata": {hidden_key: "x"},
        },
    }
    with pytest.raises(DataValidationError, match="hidden reasoning"):
        DistilledRecord.from_teacher_payload(
            payload=payload,
            task_type="frontend",
            seed=SeedDemand("seed-1", "title", "request"),
            sop=load_sop(ROOT / "skills" / "frontend.md"),
            teacher_model="mock",
            teacher_base_url="mock://local",
            teacher_protocol="mock",
            generation_params={},
            template_sha256="abc",
        )


def test_every_domain_rejects_non_allowlisted_output_keys() -> None:
    from anchor_mvp.data.schema import validate_output

    valid = {
        "plan": {
            "summary": "s",
            "steps": [{"id": "P1", "goal": "g", "deliverable": "d"}],
            "constraints": [],
        },
        "tool_policy": {
            "decision": "APPROVE",
            "rationale": "bounded",
            "proposal_labels": [],
        },
        "frontend": {"language": "tsx", "code": "export default 1"},
        "review": {"language": "tsx", "summary": "fixed", "code": "export default 2"},
        "security": {"decision": "PASS", "rationale": "safe", "findings": []},
    }
    for task_type, output in valid.items():
        with pytest.raises(DataValidationError, match="non-allowlisted"):
            validate_output(task_type, {**output, "extra": "forbidden"})  # type: ignore[arg-type]


def test_inert_security_fixtures_are_balanced_and_have_gold_labels() -> None:
    generated = [
        build_inert_security_fixture("export const Safe = () => <main />", index)
        for index in range(4)
    ]
    assert [output["decision"] for _, output, _ in generated] == [
        "PASS",
        "BLOCK",
        "PASS",
        "BLOCK",
    ]
    assert all(not manifest["active_payload_present"] for _, _, manifest in generated)
    assert all(
        "<script" not in code.casefold() and "javascript:" not in code.casefold()
        for code, _, _ in generated
    )


def _review_record(
    candidate: str, fixed: str = "export const Fixed = () => <main>Ready</main>"
):
    return DistilledRecord.from_teacher_payload(
        payload={
            "decision_trace": [
                {
                    "check": "semantics",
                    "evidence": "candidate uses a div",
                    "action": "use main",
                }
            ],
            "output": {"code": fixed},
        },
        task_type="review",
        seed=SeedDemand("seed-1", "title", "Build a status page"),
        sop=load_sop(ROOT / "skills" / "review.md"),
        teacher_model="mock",
        teacher_base_url="mock://local",
        teacher_protocol="mock",
        generation_params={},
        template_sha256="abc",
        canonical_task_input={
            "candidate_code": candidate,
            "known_benign_defect": "Restore the known semantic defect.",
        },
    )


def test_review_id_is_based_on_real_canonical_input() -> None:
    first = _review_record("export const Bad = () => <div>tiny</div>")
    changed_input = _review_record("export const Bad = () => <span>tiny</span>")
    changed_output = _review_record(
        "export const Bad = () => <div>tiny</div>",
        fixed="export const Fixed = () => <main>Status ready</main>",
    )
    assert first.id != changed_input.id
    assert first.id == changed_output.id
    case_sensitive_input = _review_record("export const BAD = () => <div>tiny</div>")
    assert first.id != case_sensitive_input.id


def test_review_requires_candidate_and_real_repair() -> None:
    with pytest.raises(DataValidationError, match="candidate_code"):
        _review_record("")
    with pytest.raises(DataValidationError, match="must repair"):
        _review_record("export const Same = 1", fixed="export const Same = 1")


def test_review_teacher_input_echo_is_rejected() -> None:
    payload = {
        "input": {"candidate_code": "teacher must not echo this"},
        "decision_trace": [
            {
                "check": "semantic",
                "evidence": "known local mutation",
                "action": "restore",
            }
        ],
        "output": {"code": "export const Fixed = 1"},
    }
    with pytest.raises(DataValidationError, match="must not echo input"):
        DistilledRecord.from_teacher_payload(
            payload=payload,
            task_type="review",
            seed=SeedDemand("seed-1", "title", "Build a status page"),
            sop=load_sop(ROOT / "skills" / "review.md"),
            teacher_model="mock",
            teacher_base_url="mock://local",
            teacher_protocol="mock",
            generation_params={},
            template_sha256="abc",
            canonical_task_input={
                "candidate_code": "local candidate",
                "known_benign_defect": "Restore the known semantic defect.",
            },
        )


def test_review_rejects_active_payload_in_candidate_or_fix() -> None:
    with pytest.raises(DataValidationError, match="active payloads"):
        validate_safe_payload(
            "review",
            {
                "input": {"candidate_code": "const value = 'javascript:unsafe'"},
                "output": {"code": "export const Safe = 1"},
            },
        )
    with pytest.raises(DataValidationError, match="active payloads"):
        validate_safe_payload(
            "review",
            {
                "input": {"candidate_code": "export const Bug = 1"},
                "output": {"code": "eval('[DEFENSIVE_PLACEHOLDER]')"},
            },
        )
