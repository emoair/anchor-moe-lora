"""Versioned public reviewer verdict shared by serving and data generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Mapping


REVIEW_VERDICT_SCHEMA_VERSION = "anchor.domain-review-verdict.v2"
_ISSUE_KEYS = frozenset({"code", "severity", "summary", "required_change"})
_TOP_LEVEL_KEYS = frozenset({"schema_version", "verdict", "issues"})
_SEVERITIES = frozenset({"critical", "major", "minor"})
_HIDDEN_KEY = re.compile(r"(?:reasoning|thinking|chain.?of.?thought|\bcot\b)", re.IGNORECASE)


@dataclass(frozen=True)
class ReviewIssue:
    code: str
    severity: str
    summary: str
    required_change: str


@dataclass(frozen=True)
class ReviewVerdict:
    verdict: str
    issues: tuple[ReviewIssue, ...]
    schema_version: str = REVIEW_VERDICT_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _contains_hidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _HIDDEN_KEY.search(str(key)) is not None or _contains_hidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_hidden_key(item) for item in value)
    return False


def parse_review_verdict(text: str) -> ReviewVerdict | None:
    """Parse an exact public PASS/REVISE contract; ambiguity fails closed."""

    try:
        payload = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_KEYS:
        return None
    if _contains_hidden_key(payload):
        return None
    if payload.get("schema_version") != REVIEW_VERDICT_SCHEMA_VERSION:
        return None
    verdict = str(payload.get("verdict", ""))
    if verdict not in {"PASS", "REVISE"}:
        return None
    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list) or len(raw_issues) > 16:
        return None
    issues: list[ReviewIssue] = []
    for raw in raw_issues:
        if not isinstance(raw, Mapping) or set(raw) != _ISSUE_KEYS:
            return None
        code = str(raw.get("code", "")).strip()
        severity = str(raw.get("severity", "")).strip()
        summary = str(raw.get("summary", "")).strip()
        required_change = str(raw.get("required_change", "")).strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", code):
            return None
        if severity not in _SEVERITIES:
            return None
        if not summary or not required_change:
            return None
        if len(summary) > 500 or len(required_change) > 500:
            return None
        issues.append(ReviewIssue(code, severity, summary, required_change))
    if verdict == "PASS" and issues:
        return None
    if verdict == "REVISE" and not issues:
        return None
    return ReviewVerdict(verdict, tuple(issues))


def revision_issues_json(verdict: ReviewVerdict) -> str:
    if verdict.verdict != "REVISE":
        raise ValueError("revision issues require a REVISE verdict")
    return json.dumps(
        [asdict(issue) for issue in verdict.issues],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
