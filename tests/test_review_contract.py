import json

import pytest

from anchor_mvp.data.prompts import frontend_revision_prompt, review_verdict_prompt
from anchor_mvp.data.schema import (
    DataValidationError,
    REVIEW_LOOP_DATA_SCHEMA_VERSION,
    validate_frontend_revision_payload,
    validate_review_verdict_payload,
)
from anchor_mvp.review_contract import parse_review_verdict


def _revise_payload():
    return {
        "schema_version": "anchor.domain-review-verdict.v2",
        "verdict": "REVISE",
        "issues": [
            {
                "code": "JS_EVENT_HANDLER",
                "severity": "major",
                "summary": "The button handler is missing.",
                "required_change": "Attach the required click handler.",
            }
        ],
    }


def test_review_contract_accepts_pass_and_revise():
    passed = parse_review_verdict(
        json.dumps(
            {
                "schema_version": "anchor.domain-review-verdict.v2",
                "verdict": "PASS",
                "issues": [],
            }
        )
    )
    revised = parse_review_verdict(json.dumps(_revise_payload()))

    assert passed is not None and passed.verdict == "PASS"
    assert revised is not None and revised.issues[0].code == "JS_EVENT_HANDLER"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": "anchor.domain-review-verdict.v2",
            "verdict": "PASS",
            "issues": [],
            "reasoning": "private",
        },
        {
            "schema_version": "anchor.domain-review-verdict.v2",
            "verdict": "REVISE",
            "issues": [],
        },
        {
            "schema_version": "anchor.domain-review-verdict.v2",
            "verdict": "PASS",
            "issues": _revise_payload()["issues"],
        },
    ],
)
def test_review_contract_rejects_ambiguous_or_hidden_payloads(payload):
    assert parse_review_verdict(json.dumps(payload)) is None


def test_v2_prompts_match_runtime_contract_and_builder_revision():
    system, user = review_verdict_prompt("Build a button", "<button>x</button>", cycle=1, max_cycles=2)
    assert "anchor.domain-review-verdict.v2" in system
    assert "CANDIDATE CODE" in user

    verdict = validate_review_verdict_payload(_revise_payload())
    revision_system, revision_user = frontend_revision_prompt(
        "Build a button", "<button>x</button>", verdict
    )
    assert "complete revised code" in revision_system
    assert "JS_EVENT_HANDLER" in revision_user


def test_frontend_revision_v2_validation_is_versioned_and_requires_change():
    payload = {
        "schema_version": REVIEW_LOOP_DATA_SCHEMA_VERSION,
        "requirement": "Build a button",
        "current_code": "<button>x</button>",
        "review_verdict": _revise_payload(),
        "revised_code": '<button onclick="go()">x</button>',
    }
    validate_frontend_revision_payload(payload)

    payload["revised_code"] = payload["current_code"]
    with pytest.raises(DataValidationError, match="must change"):
        validate_frontend_revision_payload(payload)
