"""Public, deterministic protocol for compact-v2 segmented inference.

The compact-v2 training projection teaches the frontend and reviewer adapters to
emit tagged TSX segments rather than one unconstrained full file.  This module is
the inference-side inverse of that projection.  It intentionally has no access to
held-out labels, training outputs, benchmark records, or a tokenizer.

Segment counts are part of the frozen benchmark contract.  They must be selected
from public/calibration metadata before held-out access and then shared by A--F.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_PROTOCOL = "single_file_tsx_segmented_v1"
SEGMENT_CONTRACT_VERSION = "anchor.segmented-eval.v1"
SEGMENTED_REVIEW_PROTOCOL = "segmented_repair_v1"
PROMPT_BUNDLE_VERSION = "anchor.compact-v2.prompt-bundle.v1"
DEFAULT_FRONTEND_SEGMENTS = 10
DEFAULT_REVIEW_SEGMENTS = 10
MAX_SEGMENTS = 32

# The compact-v2 review projection used a DEFECT line.  Formal evaluation keeps
# that public shape, but the value is deliberately label-independent: held-out
# mutation descriptions and marker strings are evaluator-only data.
PUBLIC_REVIEW_INSTRUCTION = (
    "independently inspect this candidate excerpt and repair any defect; "
    "no evaluator hint is supplied"
)


class SegmentProtocolError(ValueError):
    """Raised when a segmented response violates the frozen public contract."""


@dataclass(frozen=True)
class SegmentContract:
    artifact_protocol: str
    contract_version: str
    frontend_segments: int
    review_segments: int

    def __post_init__(self) -> None:
        if self.artifact_protocol != ARTIFACT_PROTOCOL:
            raise SegmentProtocolError("unsupported segmented artifact protocol")
        if self.contract_version != SEGMENT_CONTRACT_VERSION:
            raise SegmentProtocolError("unsupported segmented evaluation contract")
        for label, value in (
            ("frontend_segments", self.frontend_segments),
            ("review_segments", self.review_segments),
        ):
            if value < 1 or value > MAX_SEGMENTS:
                raise SegmentProtocolError(f"{label} must stay within 1..{MAX_SEGMENTS}")

    @property
    def expected_calls(self) -> int:
        # planner + tool policy + generated segments + reviewed segments + security
        return 3 + self.frontend_segments + self.review_segments

    def binding_payload(
        self, *, max_completion_tokens_per_physical_call: int
    ) -> dict[str, Any]:
        """Return the complete, public per-run segment binding.

        Counts alone are insufficient: changing the physical-call completion cap
        changes the effective protocol and must invalidate formal checkpoints.
        """

        if max_completion_tokens_per_physical_call < 1:
            raise SegmentProtocolError("physical-call completion cap must be positive")
        return {
            "artifact_protocol": self.artifact_protocol,
            "segment_contract_version": self.contract_version,
            "frontend_segment_count": self.frontend_segments,
            "review_segment_count": self.review_segments,
            "expected_physical_calls": self.expected_calls,
            "max_completion_tokens_per_physical_call": (
                max_completion_tokens_per_physical_call
            ),
        }

    def binding_sha256(self, *, max_completion_tokens_per_physical_call: int) -> str:
        return sha256(
            compact_json(
                self.binding_payload(
                    max_completion_tokens_per_physical_call=(
                        max_completion_tokens_per_physical_call
                    )
                )
            ).encode("utf-8")
        ).hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def prompt_bundle_payload() -> dict[str, Any]:
    """Describe the frozen compact-v2 inference prompt surface.

    This manifest intentionally contains templates and public field names only;
    it never depends on a benchmark case, model output, or evaluator label.
    """

    return {
        "version": PROMPT_BUNDLE_VERSION,
        "message_roles": ["user"],
        "temperature": 0.0,
        "planner": {
            "prefix": f"PLAN|artifact={ARTIFACT_PROTOCOL}",
            "fields": ["requirement"],
        },
        "tool_policy": {
            "prefix": "TOOL_POLICY|",
            "fields": ["requirement", "plan", "proposals"],
        },
        "frontend": {
            "prefix": "GENERATE_TSX_SEGMENT|",
            "fields": [
                "artifact_protocol",
                "artifact_sha256",
                "segment_index",
                "segment_count",
                "requirement",
                "plan_summary",
            ],
            "wrapper": "anchor-tsx-segment",
        },
        "review": {
            "prefix": "REVIEW_TSX_SEGMENT|",
            "fields": ["sha", "REQ", "DEFECT", "CANDIDATE"],
            "defect_instruction": PUBLIC_REVIEW_INSTRUCTION,
            "evaluator_hint_allowed": False,
            "wrapper": "anchor-tsx-review-segment",
        },
        "security": {
            "prefix": "SECURITY_GATE|",
            "fields": ["requirement", "code_security_synopsis", "selection"],
        },
    }


def prompt_bundle_sha256() -> str:
    return sha256(compact_json(prompt_bundle_payload()).encode("utf-8")).hexdigest()


PROMPT_BUNDLE_SHA256 = prompt_bundle_sha256()


def protocol_binding_metadata(
    contract: SegmentContract, *, max_completion_tokens_per_physical_call: int
) -> dict[str, Any]:
    """Return hashes recorded on every formal segmented result."""

    return {
        "segment_contract_sha256": contract.binding_sha256(
            max_completion_tokens_per_physical_call=(
                max_completion_tokens_per_physical_call
            )
        ),
        "prompt_bundle_version": PROMPT_BUNDLE_VERSION,
        "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        "prompt_message_roles": ["user"],
    }


def validate_protocol_binding_metadata(
    observed: Mapping[str, Any],
    contract: SegmentContract,
    *,
    max_completion_tokens_per_physical_call: int,
) -> None:
    """Fail closed when persisted protocol hashes do not match frozen code."""

    expected = protocol_binding_metadata(
        contract,
        max_completion_tokens_per_physical_call=(
            max_completion_tokens_per_physical_call
        ),
    )
    for key, value in expected.items():
        if observed.get(key) != value:
            raise SegmentProtocolError(f"segmented record changed frozen {key}")


def public_session_digest(requirement: str, plan: Mapping[str, Any]) -> str:
    """Return a non-oracle routing digest derived only from public model inputs."""

    payload = f"{ARTIFACT_PROTOCOL}\n{requirement}\n{compact_json(plan)}"
    return sha256(payload.encode("utf-8")).hexdigest()


def planner_prompt(requirement: str) -> str:
    return f"PLAN|artifact={ARTIFACT_PROTOCOL}\nrequirement={requirement.strip()}"


def parse_public_plan(text: str) -> dict[str, Any]:
    """Parse the compact-v2 planner target without extracting fenced JSON."""

    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError) as exc:
        raise SegmentProtocolError("planner output is not one strict JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"summary", "constraints", "steps"}:
        raise SegmentProtocolError("planner output has the wrong public keys")
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        raise SegmentProtocolError("planner summary must be a non-empty string")
    if not isinstance(value["constraints"], list) or not all(
        isinstance(item, str) for item in value["constraints"]
    ):
        raise SegmentProtocolError("planner constraints must be a string list")
    if not isinstance(value["steps"], list) or not value["steps"]:
        raise SegmentProtocolError("planner steps must be a non-empty list")
    for step in value["steps"]:
        if not isinstance(step, dict) or set(step) != {"id", "goal", "deliverable"}:
            raise SegmentProtocolError("planner step has the wrong public keys")
        if not all(isinstance(step[key], str) and step[key].strip() for key in step):
            raise SegmentProtocolError("planner step fields must be non-empty strings")
    return value


def tool_policy_prompt(
    requirement: str,
    plan: Mapping[str, Any],
    proposal_labels: Iterable[str],
) -> str:
    proposals = [
        {
            "id": f"P{index + 1}",
            "cap": str(label),
            "scope": "inert benchmark metadata",
            "effect": "none; advisory classification only",
            "purpose": "classify proposed capability",
        }
        for index, label in enumerate(proposal_labels)
    ]
    payload = {
        "requirement": requirement.strip(),
        "plan": str(plan.get("summary", "")).strip(),
        "proposals": proposals,
    }
    return "TOOL_POLICY|" + compact_json(payload)


def frontend_segment_prompt(
    requirement: str,
    plan: Mapping[str, Any],
    *,
    session_digest: str,
    segment_index: int,
    segment_count: int,
) -> str:
    _validate_position(segment_index, segment_count)
    payload = {
        "artifact_protocol": ARTIFACT_PROTOCOL,
        # compact-v2 used this field name.  At inference it is deliberately a
        # public-input session digest, never the unknown target artifact digest.
        "artifact_sha256": session_digest,
        "segment_index": segment_index,
        "segment_count": segment_count,
        "requirement": requirement.strip(),
        "plan_summary": str(plan.get("summary", "")).strip(),
    }
    return "GENERATE_TSX_SEGMENT|" + compact_json(payload)


def review_segment_prompt(
    requirement: str,
    candidate_excerpt: str,
    *,
    session_digest: str,
    segment_index: int,
    segment_count: int,
) -> str:
    _validate_position(segment_index, segment_count)
    return (
        f"REVIEW_TSX_SEGMENT|{segment_index + 1}/{segment_count}|"
        f"sha={session_digest[:16]}\n"
        f"REQ:{requirement.strip()}\n"
        f"DEFECT:{PUBLIC_REVIEW_INSTRUCTION}\n"
        f"CANDIDATE:\n{candidate_excerpt}"
    )


def security_gate_prompt(requirement: str, reviewed_code: str) -> str:
    payload = {
        "requirement": requirement.strip(),
        "code_security_synopsis": security_synopsis(reviewed_code),
        "selection": "deterministic sources/sinks + boundaries; no oracle fields",
    }
    return "SECURITY_GATE|" + compact_json(payload)


def parse_segment(
    text: str,
    *,
    kind: str,
    segment_index: int,
    segment_count: int,
) -> str:
    """Validate one exact wrapper and return its lossless payload."""

    _validate_position(segment_index, segment_count)
    if kind not in {"frontend", "review"}:
        raise SegmentProtocolError("segment kind must be frontend or review")
    prefix = "anchor-tsx-segment" if kind == "frontend" else "anchor-tsx-review-segment"
    expected = segment_index + 1
    pattern = re.compile(
        rf"\A/\*<{re.escape(prefix)} {expected}/{segment_count}>\*/\r?\n"
        rf"(?P<payload>[\s\S]+?)\r?\n/\*</{re.escape(prefix)}>\*/[ \t\r\n]*\Z"
    )
    match = pattern.fullmatch(text)
    if match is None:
        raise SegmentProtocolError(
            f"{kind} segment {expected}/{segment_count} has an invalid wrapper"
        )
    payload = match.group("payload")
    wrapper_markers = (
        "<anchor-tsx-segment",
        "</anchor-tsx-segment>",
        "<anchor-tsx-review-segment",
        "</anchor-tsx-review-segment>",
    )
    if any(marker in payload for marker in wrapper_markers):
        raise SegmentProtocolError("nested or duplicate segment wrapper detected")
    return payload


def reassemble_segments(
    outputs: Sequence[str],
    *,
    kind: str,
    segment_count: int,
) -> str:
    if len(outputs) != segment_count:
        raise SegmentProtocolError(
            f"expected {segment_count} {kind} outputs, received {len(outputs)}"
        )
    payloads = [
        parse_segment(
            output,
            kind=kind,
            segment_index=index,
            segment_count=segment_count,
        )
        for index, output in enumerate(outputs)
    ]
    return "".join(payloads)


def split_review_candidate(code: str, segment_count: int) -> list[str]:
    """Split a candidate losslessly into a fixed, pre-registered number of excerpts."""

    if segment_count < 1 or segment_count > MAX_SEGMENTS:
        raise SegmentProtocolError("invalid review segment count")
    if not code:
        raise SegmentProtocolError("cannot review an empty candidate")
    # Exact character boundaries keep this tokenizer-independent and reproducible.
    # Empty tail chunks are allowed for very small synthetic fixtures so physical
    # call counts remain identical to the frozen A--F contract.
    boundaries = [round(len(code) * index / segment_count) for index in range(segment_count + 1)]
    return [code[boundaries[index] : boundaries[index + 1]] for index in range(segment_count)]


_SECURITY_TERMS = (
    "dangerouslysetinnerhtml",
    "innerhtml",
    "document.write",
    "eval(",
    "new function",
    "javascript:",
    "onerror",
    "postmessage",
    "location",
    "searchparams",
    "fetch(",
    "websocket",
    "cookie",
    "localstorage",
    "token",
    "href",
    "src=",
    "target.value",
    "process.",
    "child_process",
    "exec(",
    "spawn(",
    "query(",
)


def security_synopsis(code: str, *, max_chars: int = 2400) -> str:
    """Build a label-independent public source/sink synopsis."""

    if max_chars < 1:
        raise SegmentProtocolError("security synopsis budget must be positive")
    lines = code.splitlines()
    scored: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        lowered = line.casefold()
        score = sum(8 for term in _SECURITY_TERMS if term in lowered)
        if any(marker in lowered for marker in ("import ", "function ", "=>", "return ")):
            score += 2
        if index < 8 or index >= max(0, len(lines) - 6):
            score += 1
        if score:
            scored.append((-score, index, line))
    selected = sorted(scored)[:80]
    ordered = [
        f"L{index + 1}:{line}"
        for _, index, line in sorted(selected, key=lambda item: item[1])
    ]
    synopsis = "\n".join(ordered) or "(no lexical source/sink markers)"
    return synopsis[:max_chars]


def _validate_position(segment_index: int, segment_count: int) -> None:
    if segment_count < 1 or segment_count > MAX_SEGMENTS:
        raise SegmentProtocolError("invalid segment count")
    if segment_index < 0 or segment_index >= segment_count:
        raise SegmentProtocolError("segment index is outside its declared count")
