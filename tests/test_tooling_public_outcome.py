import json

from anchor_mvp.tooling.trace import parse_public_outcome


def _event(text):
    return json.dumps({"type": "text", "part": {"text": text}})


def test_extracts_narrow_public_outcome_from_event_stream():
    payload = json.dumps(
        {
            "schema_version": "anchor.public-outcome.v1",
            "status": "completed",
            "decision_trace": [
                {
                    "check": "Build state",
                    "evidence": "npm build exited 0",
                    "action": "Kept the minimal patch",
                }
            ],
            "repair_summaries": ["Corrected a stale state update and reran tests"],
            "final_summary": "Build and tests passed.",
        }
    )

    outcome = parse_public_outcome(_event("discard me") + "\n" + _event(payload))

    assert outcome is not None
    assert outcome.status == "completed"
    assert outcome.decision_trace[0].evidence == "npm build exited 0"


def test_rejects_hidden_reasoning_and_unbounded_trace():
    hidden = json.dumps(
        {
            "schema_version": "anchor.public-outcome.v1",
            "status": "completed",
            "thinking": "private",
            "decision_trace": [{"check": "a", "evidence": "b", "action": "c"}],
            "repair_summaries": [],
            "final_summary": "done",
        }
    )
    too_many = json.dumps(
        {
            "schema_version": "anchor.public-outcome.v1",
            "status": "completed",
            "decision_trace": [
                {"check": "a", "evidence": "b", "action": "c"} for _ in range(9)
            ],
            "repair_summaries": [],
            "final_summary": "done",
        }
    )

    assert parse_public_outcome(_event(hidden)) is None
    assert parse_public_outcome(_event(too_many)) is None


def test_never_extracts_public_outcome_from_reasoning_event():
    payload = json.dumps(
        {
            "schema_version": "anchor.public-outcome.v1",
            "status": "completed",
            "decision_trace": [
                {"check": "private", "evidence": "private", "action": "private"}
            ],
            "repair_summaries": [],
            "final_summary": "must not persist",
        }
    )
    event = json.dumps(
        {"type": "reasoning", "part": {"type": "reasoning", "text": payload}}
    )

    assert parse_public_outcome(event) is None
