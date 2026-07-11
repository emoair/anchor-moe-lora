"""Schema and validation for distilled Anchor-MoE-LoRA records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Any, Literal, Mapping

from ..review_contract import ReviewVerdict, parse_review_verdict


TaskType = Literal["plan", "tool_policy", "frontend", "review", "security"]
SCHEMA_VERSION = "1.0"
REVIEW_LOOP_DATA_SCHEMA_VERSION = "anchor.review-loop-data.v2"
REVIEW_VERDICT_DATASET = "data_review_verdict_v2.jsonl"
FRONTEND_REVISION_DATASET = "data_frontend_revision_v2.jsonl"
TASK_TYPES: tuple[TaskType, ...] = (
    "plan",
    "tool_policy",
    "frontend",
    "review",
    "security",
)
EXPERT_BY_TASK: dict[TaskType, str] = {
    "plan": "planner",
    "tool_policy": "tool_policy",
    "frontend": "frontend_gen",
    "review": "frontend_review",
    "security": "security_gate",
}

OUTPUT_KEYS_BY_TASK: dict[TaskType, frozenset[str]] = {
    "plan": frozenset({"summary", "steps", "constraints"}),
    "tool_policy": frozenset({"decision", "rationale", "proposal_labels"}),
    "frontend": frozenset({"language", "code"}),
    "review": frozenset({"language", "summary", "code"}),
    "security": frozenset({"decision", "rationale", "findings"}),
}


class DataValidationError(ValueError):
    """Raised when generated data does not satisfy the public schema."""


def validate_review_verdict_payload(payload: Mapping[str, Any]) -> ReviewVerdict:
    """Validate one v2 public reviewer target without changing the legacy v1 schema."""

    reject_hidden_reasoning_keys(payload)
    verdict = parse_review_verdict(json.dumps(dict(payload), ensure_ascii=False))
    if verdict is None:
        raise DataValidationError("invalid public review verdict v2 payload")
    return verdict


def validate_frontend_revision_payload(payload: Mapping[str, Any]) -> None:
    """Validate a builder revision example paired with a public REVISE verdict."""

    reject_hidden_reasoning_keys(payload)
    expected = {"schema_version", "requirement", "current_code", "review_verdict", "revised_code"}
    if set(payload) != expected:
        raise DataValidationError("frontend revision v2 payload has missing or extra keys")
    if payload.get("schema_version") != REVIEW_LOOP_DATA_SCHEMA_VERSION:
        raise DataValidationError("frontend revision v2 schema_version mismatch")
    for key in ("requirement", "current_code", "revised_code"):
        if not isinstance(payload.get(key), str) or not str(payload[key]).strip():
            raise DataValidationError(f"frontend revision v2 requires non-empty {key}")
    raw_verdict = payload.get("review_verdict")
    if not isinstance(raw_verdict, Mapping):
        raise DataValidationError("frontend revision v2 requires review_verdict")
    verdict = validate_review_verdict_payload(raw_verdict)
    if verdict.verdict != "REVISE":
        raise DataValidationError("frontend revision v2 requires a REVISE verdict")
    if normalized_text(str(payload["current_code"])) == normalized_text(
        str(payload["revised_code"])
    ):
        raise DataValidationError("frontend revision v2 must change the current code")


def stable_id(prefix: str, value: str) -> str:
    digest = sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def reject_hidden_reasoning_keys(value: Any, *, path: str = "payload") -> None:
    """Reject hidden-reasoning fields at any nesting depth in teacher data."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if (
                normalized == "cot"
                or "chainofthought" in normalized
                or normalized.startswith("reasoning")
                or normalized.startswith("thinking")
            ):
                raise DataValidationError(
                    f"hidden reasoning field is forbidden at {path}.{key}; use decision_trace"
                )
            reject_hidden_reasoning_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_hidden_reasoning_keys(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class SeedDemand:
    """A website request used to drive one or more expert tasks."""

    seed_id: str
    title: str
    request: str
    category: str = "standard"
    tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SeedDemand":
        request = str(value.get("request", "")).strip()
        title = str(value.get("title", "Untitled request")).strip()
        if not request:
            raise DataValidationError("seed request is empty")
        seed_id = str(value.get("seed_id") or stable_id("seed", normalized_text(request)))
        tags = tuple(str(item).strip() for item in value.get("tags", []) if str(item).strip())
        return cls(
            seed_id=seed_id,
            title=title,
            request=request,
            category=str(value.get("category", "standard")).strip() or "standard",
            tags=tags,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tags"] = list(self.tags)
        return result


@dataclass(frozen=True)
class ExpertSOP:
    """Loaded expert procedure and its provenance."""

    sop_id: str
    task_type: TaskType
    source: str
    content: str
    sha256: str

    def to_public_dict(self) -> dict[str, str]:
        return {
            "sop_id": self.sop_id,
            "task_type": self.task_type,
            "source": self.source,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DecisionStep:
    """Short, auditable work product; explicitly not hidden chain-of-thought."""

    check: str
    evidence: str
    action: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionStep":
        step = cls(
            check=str(value.get("check", "")).strip(),
            evidence=str(value.get("evidence", "")).strip(),
            action=str(value.get("action", "")).strip(),
        )
        if not all((step.check, step.evidence, step.action)):
            raise DataValidationError("decision_trace entries need check, evidence, and action")
        if any(len(item) > 600 for item in (step.check, step.evidence, step.action)):
            raise DataValidationError("decision_trace entry is too long")
        return step


@dataclass(frozen=True)
class DistilledRecord:
    """One validated JSONL training example."""

    id: str
    expert: str
    messages: tuple[dict[str, str], ...]
    input: dict[str, Any]
    provenance: dict[str, Any]
    decision_trace: tuple[DecisionStep, ...]
    output: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["messages"] = list(self.messages)
        result["decision_trace"] = [asdict(step) for step in self.decision_trace]
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_teacher_payload(
        cls,
        *,
        payload: Mapping[str, Any],
        task_type: TaskType,
        seed: SeedDemand,
        sop: ExpertSOP,
        teacher_model: str,
        teacher_base_url: str,
        teacher_protocol: str,
        generation_params: Mapping[str, Any],
        template_sha256: str,
        provider_provenance: Mapping[str, Any] | None = None,
        canonical_task_input: Mapping[str, Any] | None = None,
        provenance_extra: Mapping[str, Any] | None = None,
    ) -> "DistilledRecord":
        if task_type not in TASK_TYPES:
            raise DataValidationError(f"unsupported task type: {task_type}")
        reject_hidden_reasoning_keys(payload)
        raw_trace = payload.get("decision_trace")
        if not isinstance(raw_trace, list) or not raw_trace:
            raise DataValidationError("decision_trace must be a non-empty list")
        trace = tuple(DecisionStep.from_mapping(item) for item in raw_trace if isinstance(item, Mapping))
        if len(trace) != len(raw_trace):
            raise DataValidationError("every decision_trace item must be an object")
        output = payload.get("output")
        if not isinstance(output, Mapping):
            raise DataValidationError("output must be an object")
        clean_output = {str(key): value for key, value in output.items()}
        validate_output(task_type, clean_output)
        clean_input, user_content = canonical_input(
            task_type,
            seed,
            payload,
            canonical_task_input=canonical_task_input,
        )
        if task_type == "review" and normalized_text(str(clean_input["candidate_code"])) == normalized_text(
            str(clean_output["code"])
        ):
            raise DataValidationError("review output.code must repair, not repeat, candidate_code")
        if task_type == "security":
            assistant_content = f"[{clean_output['decision']}]"
        elif task_type == "tool_policy":
            assistant_content = str(clean_output["decision"])
        elif task_type == "plan":
            assistant_content = json.dumps(clean_output, ensure_ascii=False, sort_keys=True)
        else:
            assistant_content = str(clean_output["code"]).strip()
        canonical_user = user_content.replace("\r\n", "\n").replace("\r", "\n").strip()
        identity = f"{EXPERT_BY_TASK[task_type]}\n{sop.sha256}\n{canonical_user}"
        provenance: dict[str, Any] = {
            "seed_id": seed.seed_id,
            "sop": sop.to_public_dict(),
            "teacher": {
                "model": teacher_model,
                "base_url": teacher_base_url,
                "protocol": teacher_protocol,
                "generation_params": dict(generation_params),
            },
            "template_sha256": template_sha256,
            "created_at": utc_now(),
        }
        if provider_provenance:
            provenance["teacher"]["provider"] = dict(provider_provenance)
        if provenance_extra:
            provenance.update({str(key): value for key, value in provenance_extra.items()})
        return cls(
            id=stable_id("record", identity),
            expert=EXPERT_BY_TASK[task_type],
            messages=(
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ),
            input=clean_input,
            provenance=provenance,
            decision_trace=trace,
            output=clean_output,
        )


def validate_output(task_type: TaskType, output: Mapping[str, Any]) -> None:
    unexpected = set(output).difference(OUTPUT_KEYS_BY_TASK[task_type])
    if unexpected:
        raise DataValidationError(
            f"{task_type} output contains non-allowlisted keys: {', '.join(sorted(unexpected))}"
        )
    if task_type == "plan":
        if not isinstance(output.get("summary"), str) or not str(output["summary"]).strip():
            raise DataValidationError("plan output requires a concise summary")
        steps = output.get("steps")
        if not isinstance(steps, list) or not steps:
            raise DataValidationError("plan output requires non-empty steps")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping) or any(
                not isinstance(step.get(key), str) or not str(step[key]).strip()
                for key in ("id", "goal", "deliverable")
            ):
                raise DataValidationError(
                    f"plan output steps[{index}] requires id, goal, and deliverable"
                )
        constraints = output.get("constraints", [])
        if not isinstance(constraints, list) or any(
            not isinstance(item, str) or not item.strip() for item in constraints
        ):
            raise DataValidationError("plan output constraints must be concise strings")
        for index, step in enumerate(steps):
            extras = set(step).difference({"id", "goal", "deliverable"})
            if extras:
                raise DataValidationError(
                    f"plan output steps[{index}] contains non-allowlisted keys: "
                    f"{', '.join(sorted(extras))}"
                )
    elif task_type == "tool_policy":
        decision = output.get("decision")
        if decision not in ("APPROVE", "BLOCK", "ESCALATE"):
            raise DataValidationError(
                "tool_policy output decision must be APPROVE, BLOCK, or ESCALATE"
            )
        if not isinstance(output.get("rationale"), str) or not str(output["rationale"]).strip():
            raise DataValidationError("tool_policy output requires a concise rationale")
        labels = output.get("proposal_labels", [])
        if not isinstance(labels, list) or any(
            not isinstance(item, str) or not item.strip() for item in labels
        ):
            raise DataValidationError("tool_policy proposal_labels must be strings")
    elif task_type in ("frontend", "review"):
        required = "code"
        if not isinstance(output.get(required), str) or not str(output[required]).strip():
            raise DataValidationError(f"{task_type} output requires non-empty code")
    elif task_type == "security":
        decision = output.get("decision")
        if decision not in ("BLOCK", "PASS"):
            raise DataValidationError("security output decision must be BLOCK or PASS")
        if not isinstance(output.get("rationale"), str) or not str(output["rationale"]).strip():
            raise DataValidationError("security output requires a concise rationale")
        findings = output.get("findings", [])
        if not isinstance(findings, list) or any(
            not isinstance(item, str) or not item.strip() for item in findings
        ):
            raise DataValidationError("security output findings must be inert label strings")


def canonical_input(
    task_type: TaskType,
    seed: SeedDemand,
    payload: Mapping[str, Any],
    *,
    canonical_task_input: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate task-specific teacher input and build the actual SFT user turn."""

    if task_type == "plan":
        clean = {"requirement": seed.request}
        return clean, seed.request
    if "input" in payload:
        raise DataValidationError(
            f"{task_type} teacher must not echo input code; pipeline supplies canonical input"
        )
    raw_input = canonical_task_input
    if not isinstance(raw_input, Mapping):
        raise DataValidationError(f"{task_type} payload requires an input object")
    if task_type == "tool_policy":
        plan = raw_input.get("plan")
        proposals = raw_input.get("tool_proposals")
        if not isinstance(plan, Mapping):
            raise DataValidationError("tool_policy input requires the upstream plan")
        if not isinstance(proposals, list) or not proposals or any(
            not isinstance(item, Mapping) for item in proposals
        ):
            raise DataValidationError("tool_policy input requires inert tool_proposals")
        clean = {
            "requirement": seed.request,
            "plan": dict(plan),
            "tool_proposals": [dict(item) for item in proposals],
        }
        user_content = (
            f"REQUIREMENT:\n{seed.request}\n\nPLAN:\n"
            f"{json.dumps(clean['plan'], ensure_ascii=False, sort_keys=True)}\n\n"
            "INERT TOOL PROPOSALS:\n"
            f"{json.dumps(clean['tool_proposals'], ensure_ascii=False, sort_keys=True)}"
        )
        return clean, user_content
    if task_type == "frontend":
        plan = raw_input.get("plan")
        tool_policy = raw_input.get("tool_policy")
        if not isinstance(plan, Mapping):
            raise DataValidationError("frontend input requires the upstream plan")
        if not isinstance(tool_policy, Mapping):
            raise DataValidationError("frontend input requires the upstream tool_policy output")
        clean = {
            "requirement": seed.request,
            "plan": dict(plan),
            "tool_policy": dict(tool_policy),
        }
        user_content = (
            f"REQUIREMENT:\n{seed.request}\n\nAPPROVED IMPLEMENTATION PLAN:\n"
            f"{json.dumps(clean['plan'], ensure_ascii=False, sort_keys=True)}\n\n"
            "TOOL POLICY ADVISORY (not an execution grant):\n"
            f"{json.dumps(clean['tool_policy'], ensure_ascii=False, sort_keys=True)}"
        )
        return clean, user_content
    code_field = "candidate_code" if task_type == "review" else "reviewed_code"
    code = raw_input.get(code_field)
    if not isinstance(code, str) or not code.strip():
        raise DataValidationError(f"{task_type} input requires non-empty {code_field}")
    clean = {"requirement": seed.request, code_field: code.strip()}
    label = "CANDIDATE CODE" if task_type == "review" else "REVIEWED CODE"
    user_content = f"REQUIREMENT:\n{seed.request}\n\n{label}:\n{code.strip()}"
    if task_type == "review":
        defect = raw_input.get("known_benign_defect")
        if not isinstance(defect, str) or not defect.strip():
            raise DataValidationError("review input requires known_benign_defect")
        clean["known_benign_defect"] = defect.strip()
        user_content += f"\n\nKNOWN_BENIGN_DEFECT:\n{defect.strip()}"
    return clean, user_content
