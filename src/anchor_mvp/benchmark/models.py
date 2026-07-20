from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
from typing import Any, Literal


Workflow = Literal["single", "pipeline"]
PromptStyle = Literal["direct", "composite"]
ReviewProtocol = Literal["verdict_v2", "repair_code_v1", "segmented_repair_v1"]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    requirement: str
    malicious: bool = False
    required_substrings: tuple[str, ...] = ()
    split: str = "unspecified"
    namespace: str = ""
    seed_namespace: str = ""
    seed_id: str = ""
    case_family: str = ""
    expected_security_decision: str | None = None
    security_intent_label: str = ""
    review_mutation: dict[str, str] = field(default_factory=dict)
    fixture: str = ""
    plan_required_concepts: tuple[str, ...] = ()
    tool_proposal_labels: tuple[str, ...] = ()
    expected_tool_policy_decision: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(payload["case_id"]),
            requirement=str(payload["requirement"]),
            malicious=bool(payload.get("malicious", False)),
            required_substrings=tuple(payload.get("required_substrings", ())),
            split=str(payload.get("split", "unspecified")),
            namespace=str(payload.get("namespace", "")),
            seed_namespace=str(payload.get("seed_namespace", "")),
            seed_id=str(payload.get("seed_id", "")),
            case_family=str(payload.get("case_family", "")),
            expected_security_decision=(
                str(payload["expected_security_decision"])
                if payload.get("expected_security_decision") is not None
                else None
            ),
            security_intent_label=str(payload.get("security_intent_label", "")),
            review_mutation={
                str(key): str(value)
                for key, value in payload.get("review_mutation", {}).items()
            },
            fixture=str(payload.get("fixture", "")),
            plan_required_concepts=tuple(payload.get("plan_required_concepts", ())),
            tool_proposal_labels=tuple(payload.get("tool_proposal_labels", ())),
            expected_tool_policy_decision=(
                str(payload["expected_tool_policy_decision"])
                if payload.get("expected_tool_policy_decision") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    group: str
    workflow: Workflow
    prompt_style: PromptStyle = "direct"
    review_protocol: ReviewProtocol = "verdict_v2"
    artifact_protocol: str = ""
    segment_contract_version: str = ""
    frontend_segment_count: int = 1
    review_segment_count: int = 1
    model: str | None = None
    base_contract_id: str = ""
    base_source_sha256: str = ""
    q4_artifact_sha256: str | None = None
    stage_models: dict[str, str] = field(default_factory=dict)
    max_tokens_per_call: int = 1024
    matched_tokens_to: str | None = None
    adapter_trainable_parameters: int | None = None
    stage_adapter_ranks: dict[str, int] = field(default_factory=dict)
    allocation_method: str = ""
    selection_split: str = ""
    allocation_frozen: bool = True
    allocation_manifest_sha256: str | None = None
    maximum_stage_rank: int | None = None
    rank_sum_constraint: int | None = None
    parameter_budget_constraint: int | None = None
    status: str = "ready"
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineSpec":
        review_protocol = str(payload.get("review_protocol", "verdict_v2"))
        if review_protocol not in {
            "verdict_v2",
            "repair_code_v1",
            "segmented_repair_v1",
        }:
            raise ValueError(f"unsupported review_protocol: {review_protocol}")
        return cls(
            name=str(payload["name"]),
            group=str(payload["group"]),
            workflow=payload["workflow"],
            prompt_style=payload.get("prompt_style", "direct"),
            review_protocol=review_protocol,
            artifact_protocol=str(payload.get("artifact_protocol", "")),
            segment_contract_version=str(payload.get("segment_contract_version", "")),
            frontend_segment_count=int(payload.get("frontend_segment_count", 1)),
            review_segment_count=int(payload.get("review_segment_count", 1)),
            model=payload.get("model"),
            base_contract_id=str(payload.get("base_contract_id", "")),
            base_source_sha256=str(payload.get("base_source_sha256", "")),
            q4_artifact_sha256=(
                str(payload["q4_artifact_sha256"])
                if payload.get("q4_artifact_sha256") is not None
                else None
            ),
            stage_models={str(k): str(v) for k, v in payload.get("stage_models", {}).items()},
            max_tokens_per_call=int(payload.get("max_tokens_per_call", 1024)),
            matched_tokens_to=payload.get("matched_tokens_to"),
            adapter_trainable_parameters=(
                int(payload["adapter_trainable_parameters"])
                if payload.get("adapter_trainable_parameters") is not None
                else None
            ),
            stage_adapter_ranks={
                str(key): int(value)
                for key, value in payload.get("stage_adapter_ranks", {}).items()
            },
            allocation_method=str(payload.get("allocation_method", "")),
            selection_split=str(payload.get("selection_split", "")),
            allocation_frozen=bool(payload.get("allocation_frozen", True)),
            allocation_manifest_sha256=(
                str(payload["allocation_manifest_sha256"])
                if payload.get("allocation_manifest_sha256") is not None
                else None
            ),
            maximum_stage_rank=(
                int(payload["maximum_stage_rank"])
                if payload.get("maximum_stage_rank") is not None
                else None
            ),
            rank_sum_constraint=(
                int(payload["rank_sum_constraint"])
                if payload.get("rank_sum_constraint") is not None
                else None
            ),
            parameter_budget_constraint=(
                int(payload["parameter_budget_constraint"])
                if payload.get("parameter_budget_constraint") is not None
                else None
            ),
            status=str(payload.get("status", "ready")),
            notes=str(payload.get("notes", "")),
        )


@dataclass
class BenchmarkRecord:
    baseline: str
    group: str
    case_id: str
    malicious: bool
    decision: str
    success: bool
    final_code: str | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    call_count: int
    request_attempts: int
    peak_vram_mb: float | None
    backend: str = "unspecified"
    fail_closed: bool = False
    errors: list[str] = field(default_factory=list)
    required_substrings: tuple[str, ...] = ()
    stages: list[dict[str, Any]] = field(default_factory=list)
    fairness: dict[str, Any] = field(default_factory=dict)
    evaluator_provenance: dict[str, Any] = field(
        default_factory=lambda: {
            "pass_metric": "structural_required_substring_proxy_v1",
            "tool_verified": False,
            "executed_build_or_browser_test": False,
        }
    )
    verified_build_pass: bool | None = None
    frontend_build_pass: bool | None = None
    review_repair_pass: bool | None = None
    expected_security_decision: str | None = None
    case_family: str = ""
    heldout_namespace: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)
    plan_quality_pass: bool | None = None
    tool_policy_decision: str | None = None
    expected_tool_policy_decision: str | None = None
    deterministic_tool_policy_decision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["required_substrings"] = list(self.required_substrings)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkRecord":
        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in allowed}
        values["required_substrings"] = tuple(values.get("required_substrings", ()))
        return cls(**values)


def load_specs(path: str | Path) -> list[BaselineSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BaselineSpec.from_dict(item) for item in payload["baselines"]]


def load_cases_jsonl(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                cases.append(BenchmarkCase.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid case at {path}:{line_number}: {exc}") from exc
    return cases


def write_records_jsonl(records: list[BenchmarkRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_records_jsonl(path: str | Path) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(BenchmarkRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
    return records
