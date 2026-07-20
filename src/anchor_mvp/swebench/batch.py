"""Sequential SWE task-card distillation over the five-stage MVP contract.

The module coordinates already-imported train cards and already-controlled
OpenCode execution exports.  It never downloads benchmark rows or uses an
upstream answer.  The default CLI path only compiles work orders; live teacher
use requires an explicit opt-in and a process-local credential environment
variable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, cast

from ..data.provider import provider_spec, select_provider_model
from ..data.storage import JsonlStore
from ..data.teacher import CompatibleTeacher, Teacher
from .importer import IMPORT_MANIFEST_SCHEMA_VERSION, assert_content_free_manifest
from .partition import file_sha256, iter_jsonl_mappings, permanent_heldout_variant
from .schema import (
    CHAIN_STAGES,
    SWEBenchValidationError,
    TaskCard,
    canonical_json,
    digest_value,
)
from .trajectory import (
    PLANNER_OUTPUT_SCHEMA_VERSION,
    REVIEW_OUTPUT_SCHEMA_VERSION,
    SECURITY_OUTPUT_SCHEMA_VERSION,
    TOOL_POLICY_OUTPUT_SCHEMA_VERSION,
    PlannerOutput,
    ToolPolicyOutput,
    WorkspaceInventory,
    adapt_task_card_trajectory,
    project_task_card_builder_sessions,
)


BATCH_CONFIG_SCHEMA_VERSION = "anchor.swebench-five-stage-batch-config.v1"
WORK_ORDER_SCHEMA_VERSION = "anchor.swebench-five-stage-work-order.v1"
STAGE_RECORD_SCHEMA_VERSION = "anchor.swebench-five-stage-record.v1"
EXECUTION_BUNDLE_SCHEMA_VERSION = "anchor.swebench-execution-bundle.v1"
BATCH_MANIFEST_SCHEMA_VERSION = "anchor.swebench-five-stage-batch-manifest.v1"
REPLAY_RESPONSE_SCHEMA_VERSION = "anchor.swebench-replay-response.v1"

_SAFE_STAGE = frozenset(CHAIN_STAGES)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ORACLE_MARKER = re.compile(
    r"\b(?:FAIL_TO_PASS|PASS_TO_PASS|test_patch|hints_text|gold_patch|"
    r"gold_solution|oracle_text)\b",
    re.IGNORECASE,
)
_FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "patch",
        "testpatch",
        "hint",
        "hints",
        "hintstext",
        "failtopass",
        "passtopass",
        "testname",
        "testnames",
        "tests",
        "testcases",
        "gold",
        "goldpatch",
        "goldsolution",
        "oracle",
        "oracletext",
    }
)


@dataclass(frozen=True)
class BatchConfig:
    cards_jsonl: Path
    import_manifest: Path
    output_dir: Path
    mode: str = "dry-run"
    execution_bundles_jsonl: Path | None = None
    replay_responses_jsonl: Path | None = None
    concurrency: int = 1
    max_cards: int | None = None
    provider: Mapping[str, Any] | None = None

    @classmethod
    def load(cls, project_root: Path, path: Path) -> "BatchConfig":
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise SWEBenchValidationError("SWE batch config root must be an object")
        if raw.get("schema_version") != BATCH_CONFIG_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported SWE batch config schema")
        allowed = {
            "schema_version",
            "mode",
            "cards_jsonl",
            "import_manifest",
            "execution_bundles_jsonl",
            "replay_responses_jsonl",
            "output_dir",
            "concurrency",
            "max_cards",
            "provider",
        }
        if set(raw).difference(allowed):
            raise SWEBenchValidationError("SWE batch config has unexpected fields")
        mode = str(raw.get("mode", "dry-run"))
        if mode not in {"dry-run", "replay", "live"}:
            raise SWEBenchValidationError(
                "SWE batch mode must be dry-run, replay, or live"
            )
        concurrency = raw.get("concurrency", 1)
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency < 1
        ):
            raise SWEBenchValidationError("SWE batch concurrency must be positive")
        max_cards = raw.get("max_cards")
        if max_cards is not None and (
            isinstance(max_cards, bool)
            or not isinstance(max_cards, int)
            or max_cards < 1
        ):
            raise SWEBenchValidationError("max_cards must be positive when configured")
        provider = raw.get("provider")
        if provider is not None and not isinstance(provider, Mapping):
            raise SWEBenchValidationError("provider must be an object")
        if isinstance(provider, Mapping):
            # This also rejects inline api_key/token/secret fields.
            provider_spec(provider)

        def required_path(name: str) -> Path:
            value = raw.get(name)
            if not isinstance(value, str) or not value.strip():
                raise SWEBenchValidationError(f"SWE batch config requires {name}")
            return _project_path(project_root, value, name)

        def optional_path(name: str) -> Path | None:
            value = raw.get(name)
            if value is None:
                return None
            if not isinstance(value, str) or not value.strip():
                raise SWEBenchValidationError(f"{name} must be a non-empty path")
            return _project_path(project_root, value, name)

        config = cls(
            cards_jsonl=required_path("cards_jsonl"),
            import_manifest=required_path("import_manifest"),
            execution_bundles_jsonl=optional_path("execution_bundles_jsonl"),
            replay_responses_jsonl=optional_path("replay_responses_jsonl"),
            output_dir=required_path("output_dir"),
            mode=mode,
            concurrency=concurrency,
            max_cards=max_cards,
            provider=dict(provider) if isinstance(provider, Mapping) else None,
        )
        if mode != "dry-run" and config.execution_bundles_jsonl is None:
            raise SWEBenchValidationError(
                "replay/live mode requires execution_bundles_jsonl"
            )
        if mode == "replay" and config.replay_responses_jsonl is None:
            raise SWEBenchValidationError("replay mode requires replay_responses_jsonl")
        if mode == "live" and config.provider is None:
            raise SWEBenchValidationError("live mode requires provider settings")
        return config


@dataclass(frozen=True)
class BatchResult:
    manifest: Mapping[str, Any]
    chains: tuple[Mapping[str, Any], ...]


class ReplayTeacher:
    """Offline response lookup implementing the shared ``Teacher`` protocol."""

    model = "swe-replay-v1"
    base_url = "replay://local"
    protocol = "replay"

    def __init__(self, responses: Mapping[str, Mapping[str, Any]]) -> None:
        self._responses = dict(responses)

    @property
    def generation_params(self) -> dict[str, Any]:
        return {"deterministic": True, "network": False}

    @property
    def provider_provenance(self) -> dict[str, Any]:
        return {"preset": "replay", "model": self.model, "network": False}

    async def complete(self, *, system: str, user: str) -> str:
        del system
        request = json.loads(user)
        if not isinstance(request, Mapping):
            raise SWEBenchValidationError("replay request must be an object")
        record_id = str(request.get("record_id", ""))
        if record_id not in self._responses:
            raise SWEBenchValidationError(
                f"replay response is missing for record {record_id}"
            )
        await asyncio.sleep(0)
        return canonical_json(self._responses[record_id])


def load_validated_cards(cards_path: Path, manifest_path: Path) -> tuple[TaskCard, ...]:
    """Load only canonical cards bound to a successful train import manifest."""

    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, Mapping):
        raise SWEBenchValidationError("SWE import manifest must be an object")
    if manifest_value.get("schema_version") != IMPORT_MANIFEST_SCHEMA_VERSION:
        raise SWEBenchValidationError("unsupported SWE import manifest schema")
    assert_content_free_manifest(manifest_value)
    source = _mapping(manifest_value.get("source"), "import source")
    partition = _mapping(manifest_value.get("partition"), "import partition")
    license_gate = _mapping(manifest_value.get("license_gate"), "license gate")
    card_summary = _mapping(manifest_value.get("cards"), "card summary")
    if source.get("split") != "train":
        raise SWEBenchValidationError("batch accepts only imported train cards")
    if permanent_heldout_variant(str(source.get("dataset_id", ""))) in {
        "lite",
        "verified",
    }:
        raise SWEBenchValidationError(
            "Lite/Verified/held-out data cannot enter training"
        )
    if partition.get("full_lite_verified_permanent_deny") is not True:
        raise SWEBenchValidationError("import manifest lacks permanent held-out proof")
    variants = partition.get("heldout_variant_row_counts")
    if not isinstance(variants, Mapping) or not {"full", "lite", "verified"}.issubset(
        {str(key).casefold() for key in variants}
    ):
        raise SWEBenchValidationError(
            "import manifest lacks Full/Lite/Verified coverage"
        )
    if (
        license_gate.get("unknown_repository_policy") != "fail_closed"
        or isinstance(license_gate.get("approved_repository_count"), bool)
        or not isinstance(license_gate.get("approved_repository_count"), int)
        or int(license_gate["approved_repository_count"]) < 1
    ):
        raise SWEBenchValidationError(
            "import manifest lacks a fail-closed license gate"
        )
    expected_sha = str(card_summary.get("cards_file_sha256", ""))
    if not _SHA256.fullmatch(expected_sha) or file_sha256(cards_path) != expected_sha:
        raise SWEBenchValidationError("task-card file differs from its import manifest")

    cards = tuple(
        TaskCard.from_mapping(row) for _, row in iter_jsonl_mappings(cards_path)
    )
    expected_count = card_summary.get("card_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(cards)
        or len({card.card_id for card in cards}) != len(cards)
        or len({card.alignment_id for card in cards}) != len(cards)
    ):
        raise SWEBenchValidationError("task-card cardinality differs from its manifest")
    for card in cards:
        if (
            card.source.dataset_id != source.get("dataset_id")
            or card.source.dataset_revision != source.get("dataset_revision")
            or card.source.split != "train"
        ):
            raise SWEBenchValidationError(
                "task card differs from import source identity"
            )
    return cards


def compile_work_orders(cards: Sequence[TaskCard]) -> tuple[dict[str, Any], ...]:
    """Split every card into the agreed ordered five-stage dependency graph."""

    orders: list[dict[str, Any]] = []
    for card in cards:
        previous = card.card_id
        for stage in CHAIN_STAGES:
            revision = 1 if stage in {"domain_builder", "domain_review"} else None
            record_id = stage_record_id(
                card,
                stage,
                revision=revision,
                upstream_record_ids=(previous,),
            )
            orders.append(
                {
                    "schema_version": WORK_ORDER_SCHEMA_VERSION,
                    "record_id": record_id,
                    "stage": stage,
                    "identity": _identity(card),
                    "upstream_record_ids": [previous],
                    "revision": revision,
                    "status": "ready" if stage == "planner" else "blocked",
                    "input_contract": _input_contract(stage),
                    "execution": (
                        "controlled-opencode-sandbox"
                        if stage == "domain_builder"
                        else "teacher-json"
                    ),
                }
            )
            previous = record_id
    return tuple(orders)


def compile_manifest(cards: Sequence[TaskCard]) -> dict[str, Any]:
    orders = compile_work_orders(cards)
    by_stage = {
        stage: sum(order["stage"] == stage for order in orders)
        for stage in CHAIN_STAGES
    }
    return {
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "mode": "dry-run",
        "card_count": len(cards),
        "work_order_count": len(orders),
        "work_orders_by_stage": by_stage,
        "one_five_stage_chain_per_card": len(orders) == len(cards) * len(CHAIN_STAGES),
        "teacher_requests_sent": 0,
        "files_written": 0,
        "oracle_inputs_allowed": False,
    }


def write_work_orders(path: Path, orders: Sequence[Mapping[str, Any]]) -> None:
    """Atomically materialize the content-bearing five-stage question bank."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for order in orders:
                handle.write(canonical_json(order) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def run_batch(
    config: BatchConfig,
    *,
    teacher: Teacher,
) -> BatchResult:
    """Run plan -> approval -> OpenCode -> review -> final gate per card."""

    if config.mode == "dry-run":
        raise SWEBenchValidationError("run_batch is unavailable in dry-run mode")
    cards = load_validated_cards(config.cards_jsonl, config.import_manifest)
    if config.max_cards is not None:
        cards = cards[: config.max_cards]
    bundles = _load_execution_bundles(cast(Path, config.execution_bundles_jsonl))
    missing = [card.card_id for card in cards if card.card_id not in bundles]
    if missing:
        raise SWEBenchValidationError(
            f"execution bundles are missing for {len(missing)} imported cards"
        )

    request_store = JsonlStore(
        config.output_dir / "requests.jsonl", id_field="record_id"
    )
    stage_store = JsonlStore(
        config.output_dir / "stage_records.jsonl", id_field="record_id"
    )
    chain_store = JsonlStore(config.output_dir / "chains.jsonl", id_field="card_id")
    semaphore = asyncio.Semaphore(config.concurrency)

    async def one(card: TaskCard) -> Mapping[str, Any]:
        async with semaphore:
            return await _run_card(
                card,
                bundles[card.card_id],
                teacher=teacher,
                request_store=request_store,
                stage_store=stage_store,
                chain_store=chain_store,
            )

    chains = tuple(await asyncio.gather(*(one(card) for card in cards)))
    manifest = {
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "mode": config.mode,
        "card_count": len(cards),
        "complete_chain_count": len(chains),
        "complete_stage_count": sum(len(chain["stages"]) for chain in chains),
        "all_cards_complete": len(chains) == len(cards),
        "source_cards_sha256": file_sha256(config.cards_jsonl),
        "requests_sha256": file_sha256(request_store.path),
        "stage_records_sha256": file_sha256(stage_store.path),
        "chains_sha256": file_sha256(chain_store.path),
        "oracle_inputs_allowed": False,
    }
    _atomic_json(config.output_dir / "manifest.json", manifest)
    return BatchResult(manifest=manifest, chains=chains)


async def _run_card(
    card: TaskCard,
    bundle: Mapping[str, Any],
    *,
    teacher: Teacher,
    request_store: JsonlStore,
    stage_store: JsonlStore,
    chain_store: JsonlStore,
) -> Mapping[str, Any]:
    _validate_bundle_identity(card, bundle)
    inventory = WorkspaceInventory.from_mapping(
        _mapping(bundle.get("workspace_inventory"), "workspace inventory")
    )
    plan_upstream = (card.card_id,)
    plan_id = stage_record_id(card, "planner", upstream_record_ids=plan_upstream)
    plan_request = _planner_request(card, inventory, plan_id, plan_upstream)
    planner_output = await _teacher_stage(
        card,
        stage="planner",
        record_id=plan_id,
        upstream_record_ids=plan_upstream,
        request=plan_request,
        teacher=teacher,
        request_store=request_store,
        stage_store=stage_store,
    )
    planner = PlannerOutput.from_mapping(planner_output, card)

    policy_upstream = (plan_id,)
    policy_id = stage_record_id(
        card, "tool_policy", upstream_record_ids=policy_upstream
    )
    policy_request = _policy_request(card, planner, policy_id, policy_upstream)
    policy_output = await _teacher_stage(
        card,
        stage="tool_policy",
        record_id=policy_id,
        upstream_record_ids=policy_upstream,
        request=policy_request,
        teacher=teacher,
        request_store=request_store,
        stage_store=stage_store,
    )
    policy = ToolPolicyOutput.from_mapping(
        policy_output,
        card=card,
        planner=planner,
        expected_expert_id="tool-policy",
    )

    candidates = bundle.get("opencode_session_exports")
    if not isinstance(candidates, list) or not candidates:
        raise SWEBenchValidationError("execution bundle has no OpenCode sessions")
    projected = project_task_card_builder_sessions(
        card=card,
        workspace_inventory=inventory.to_dict(),
        planner_output=planner.to_dict(),
        tool_policy_output=policy.to_dict(),
        opencode_session_exports=[
            _mapping(item, "OpenCode session") for item in candidates
        ],
    )
    review_outputs: list[Mapping[str, Any]] = []
    previous = policy_id
    for revision, builder in enumerate(projected, 1):
        builder_upstream = (previous,)
        builder_id = stage_record_id(
            card,
            "domain_builder",
            revision=revision,
            upstream_record_ids=builder_upstream,
        )
        builder_record = _stage_record(
            card,
            stage="domain_builder",
            record_id=builder_id,
            upstream_record_ids=builder_upstream,
            request={
                "execution": "controlled-opencode-sandbox",
                "approved_proposal_ids": sorted(policy.approved_ids),
                "revision": revision,
            },
            output=dict(builder),
            provenance={"executor": "controlled-opencode-export"},
        )
        _append_or_verify(stage_store, builder_record)

        review_upstream = (builder_id,)
        review_id = stage_record_id(
            card,
            "domain_review",
            revision=revision,
            upstream_record_ids=review_upstream,
        )
        review_request = _review_request(
            card,
            builder,
            review_id,
            review_upstream,
            revision=revision,
        )
        review = await _teacher_stage(
            card,
            stage="domain_review",
            record_id=review_id,
            upstream_record_ids=review_upstream,
            request=review_request,
            teacher=teacher,
            request_store=request_store,
            stage_store=stage_store,
        )
        review_outputs.append(review)
        previous = review_id

    security_upstream = (previous,)
    security_id = stage_record_id(
        card, "security", upstream_record_ids=security_upstream
    )
    security_request = _security_request(
        card,
        projected[-1],
        security_id,
        security_upstream,
    )
    security_output = await _teacher_stage(
        card,
        stage="security",
        record_id=security_id,
        upstream_record_ids=security_upstream,
        request=security_request,
        teacher=teacher,
        request_store=request_store,
        stage_store=stage_store,
    )

    chain = adapt_task_card_trajectory(
        card=card,
        workspace_inventory=inventory.to_dict(),
        planner_output=planner.to_dict(),
        tool_policy_output=policy.to_dict(),
        opencode_session_exports=[
            _mapping(item, "OpenCode session") for item in candidates
        ],
        review_outputs=review_outputs,
        security_output=security_output,
        sandbox_audit_bundle=_mapping(
            bundle.get("sandbox_audit_bundle"), "sandbox audit bundle"
        ),
        trusted_sandbox_audit_sha256=str(
            bundle.get("trusted_sandbox_audit_sha256", "")
        ),
    )
    chain = dict(chain)
    _append_or_verify(chain_store, chain)
    return chain


async def _teacher_stage(
    card: TaskCard,
    *,
    stage: str,
    record_id: str,
    upstream_record_ids: Sequence[str],
    request: Mapping[str, Any],
    teacher: Teacher,
    request_store: JsonlStore,
    stage_store: JsonlStore,
) -> Mapping[str, Any]:
    request_record = {
        "schema_version": WORK_ORDER_SCHEMA_VERSION,
        "record_id": record_id,
        "stage": stage,
        "identity": _identity(card),
        "upstream_record_ids": list(upstream_record_ids),
        "request": dict(request),
    }
    _reject_oracle_source_material(request_record)
    _append_or_verify(request_store, request_record)
    existing = {str(row["record_id"]): row for row in stage_store.records}.get(
        record_id
    )
    if existing is not None:
        _verify_stage_record(existing, request_record)
        stored_output = _mapping(existing.get("output"), f"stored {stage} output")
        _validate_teacher_output_shape(stage, stored_output)
        return stored_output
    system = _system_prompt(stage)
    user = canonical_json(request)
    raw_response = await teacher.complete(system=system, user=user)
    output = _parse_teacher_json(str(raw_response), stage)
    _validate_teacher_output_shape(stage, output)
    _reject_oracle_source_material(output)
    provenance = {
        "teacher": {
            "model": str(teacher.model),
            "base_url": str(teacher.base_url),
            "protocol": str(teacher.protocol),
            "generation_params": dict(teacher.generation_params),
            "provider": dict(teacher.provider_provenance),
        }
    }
    record = _stage_record(
        card,
        stage=stage,
        record_id=record_id,
        upstream_record_ids=upstream_record_ids,
        request=request,
        output=output,
        provenance=provenance,
    )
    _append_or_verify(stage_store, record)
    return output


def stage_record_id(
    card: TaskCard,
    stage: str,
    *,
    upstream_record_ids: Sequence[str],
    revision: int | None = None,
) -> str:
    if stage not in _SAFE_STAGE:
        raise SWEBenchValidationError("unknown five-stage work-order stage")
    identity = {
        "schema": STAGE_RECORD_SCHEMA_VERSION,
        "card_id": card.card_id,
        "instance_id": card.source.instance_id,
        "source_fingerprint": card.source_fingerprint,
        "stage": stage,
        "revision": revision,
        "upstream_record_ids": list(upstream_record_ids),
    }
    return f"swe-stage-v1:{digest_value(identity)}"


def _planner_request(
    card: TaskCard,
    inventory: WorkspaceInventory,
    record_id: str,
    upstream: Sequence[str],
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "stage": "planner",
        "identity": _identity(card),
        "upstream_record_ids": list(upstream),
        "public_input": {
            "problem_statement": card.problem_statement,
            "base_workspace_inventory": inventory.to_dict(),
        },
        "required_output_schema": {
            "schema_version": PLANNER_OUTPUT_SCHEMA_VERSION,
            "alignment_id": card.alignment_id,
            "domain_id": card.domain_id,
            "builder_expert_id": card.builder_expert_id,
            "reviewer_expert_id": card.reviewer_expert_id,
            "work_items": ["public implementation step"],
            "tool_proposals": [
                {
                    "proposal_id": "stable-id",
                    "tool": "tool-id",
                    "purpose": "public purpose",
                    "input": {"workspace_bound_argument": "value"},
                }
            ],
        },
    }


def _policy_request(
    card: TaskCard,
    planner: PlannerOutput,
    record_id: str,
    upstream: Sequence[str],
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "stage": "tool_policy",
        "identity": _identity(card),
        "upstream_record_ids": list(upstream),
        "public_input": {
            "problem_statement": card.problem_statement,
            "tool_proposals": [item.to_dict() for item in planner.tool_proposals],
        },
        "required_output_schema": {
            "schema_version": TOOL_POLICY_OUTPUT_SCHEMA_VERSION,
            "alignment_id": card.alignment_id,
            "executed_expert_id": "tool-policy",
            "decisions": [
                {
                    "proposal_id": "one supplied proposal_id",
                    "decision": "APPROVE or DENY",
                    "reason": "public safety rationale",
                }
            ],
        },
    }


def _review_request(
    card: TaskCard,
    builder: Mapping[str, Any],
    record_id: str,
    upstream: Sequence[str],
    *,
    revision: int,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "stage": "domain_review",
        "identity": _identity(card),
        "upstream_record_ids": list(upstream),
        "public_input": {
            "problem_statement": card.problem_statement,
            "generated_diff": builder["output"]["diff"],
            "execution_summary": builder["output"]["execution_summary"],
        },
        "required_output_schema": {
            "schema_version": REVIEW_OUTPUT_SCHEMA_VERSION,
            "alignment_id": card.alignment_id,
            "revision": revision,
            "executed_expert_id": card.reviewer_expert_id,
            "decision": "PASS or REVISE",
            "feedback": ["public, actionable repair request"],
        },
    }


def _security_request(
    card: TaskCard,
    builder: Mapping[str, Any],
    record_id: str,
    upstream: Sequence[str],
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "stage": "security",
        "identity": _identity(card),
        "upstream_record_ids": list(upstream),
        "public_input": {
            "problem_statement": card.problem_statement,
            "generated_diff": builder["output"]["diff"],
            "execution_summary": builder["output"]["execution_summary"],
        },
        "required_output_schema": {
            "schema_version": SECURITY_OUTPUT_SCHEMA_VERSION,
            "alignment_id": card.alignment_id,
            "executed_expert_id": "security-audit",
            "decision": "PASS or BLOCK",
            "findings": ["public security finding"],
        },
    }


def _system_prompt(stage: str) -> str:
    return (
        f"You are the {stage} expert in a staged software repair pipeline. "
        "Return exactly one JSON object matching required_output_schema. Use only "
        "the supplied public_input and upstream records. Do not reveal private "
        "chain-of-thought; emit concise public decisions. Never infer or request "
        "benchmark answer patches, hidden tests, hints, expected test names, gold "
        "solutions, or oracle labels. Do not call tools in this response."
    )


def _stage_record(
    card: TaskCard,
    *,
    stage: str,
    record_id: str,
    upstream_record_ids: Sequence[str],
    request: Mapping[str, Any],
    output: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version": STAGE_RECORD_SCHEMA_VERSION,
        "record_id": record_id,
        "stage": stage,
        "identity": _identity(card),
        "upstream_record_ids": list(upstream_record_ids),
        "request_sha256": digest_value(request),
        "input": dict(request),
        "output": dict(output),
        "provenance": dict(provenance),
    }
    _reject_oracle_source_material(record, allow_generated_diff=True)
    return record


def _identity(card: TaskCard) -> dict[str, str]:
    return {
        "card_id": card.card_id,
        "instance_id": card.source.instance_id,
        "alignment_id": card.alignment_id,
        "source_fingerprint": card.source_fingerprint,
    }


def _input_contract(stage: str) -> list[str]:
    return {
        "planner": ["problem_statement", "base_workspace_inventory"],
        "tool_policy": ["problem_statement", "tool_proposals"],
        "domain_builder": [
            "problem_statement",
            "approved_tool_policy",
            "controlled_workspace",
        ],
        "domain_review": ["problem_statement", "generated_diff", "execution_summary"],
        "security": ["problem_statement", "generated_diff", "execution_summary"],
    }[stage]


def _load_execution_bundles(path: Path) -> dict[str, Mapping[str, Any]]:
    bundles: dict[str, Mapping[str, Any]] = {}
    for _, value in iter_jsonl_mappings(path):
        if value.get("schema_version") != EXECUTION_BUNDLE_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported SWE execution-bundle schema")
        expected = {
            "schema_version",
            "card_id",
            "instance_id",
            "source_fingerprint",
            "workspace_inventory",
            "opencode_session_exports",
            "sandbox_audit_bundle",
            "trusted_sandbox_audit_sha256",
        }
        if set(value) != expected:
            raise SWEBenchValidationError("SWE execution bundle has unexpected fields")
        _reject_oracle_source_material(value, allow_generated_diff=True)
        card_id = str(value.get("card_id", ""))
        if not card_id or card_id in bundles:
            raise SWEBenchValidationError("SWE execution bundle repeats one card_id")
        bundles[card_id] = value
    return bundles


def load_replay_teacher(path: Path) -> ReplayTeacher:
    responses: dict[str, Mapping[str, Any]] = {}
    for _, value in iter_jsonl_mappings(path):
        if (
            set(value) != {"schema_version", "record_id", "response"}
            or value.get("schema_version") != REPLAY_RESPONSE_SCHEMA_VERSION
        ):
            raise SWEBenchValidationError("invalid SWE replay response row")
        record_id = str(value.get("record_id", ""))
        response = _mapping(value.get("response"), "replay response")
        _reject_oracle_source_material(response)
        if not record_id or record_id in responses:
            raise SWEBenchValidationError("SWE replay responses repeat one record_id")
        responses[record_id] = dict(response)
    return ReplayTeacher(responses)


def build_live_teacher(config: Mapping[str, Any]) -> CompatibleTeacher:
    """Construct the shared compatible client; credentials stay in its named env."""

    spec = provider_spec(config)
    selection = select_provider_model(
        spec,
        requested_model=str(config.get("model", "")) or None,
        discover=bool(config.get("discover_models", False)),
        force_model=bool(config.get("force_model", True)),
        model_index=(
            int(config["model_index"])
            if config.get("model_index") is not None
            else None
        ),
        timeout_seconds=float(config.get("discovery_timeout_seconds", 20)),
    )
    return CompatibleTeacher(
        base_url=spec.base_url,
        model=selection.model,
        protocol=spec.protocol,
        fallback_protocol=None,
        fallback_base_url=spec.base_url,
        api_key_env=spec.api_key_env,
        user_agent=str(config.get("user_agent", "anchor-moe-lora/0.1")),
        timeout_seconds=float(config.get("timeout_seconds", 600)),
        max_retries=int(config.get("max_retries", 1)),
        temperature=float(config.get("temperature", 0.2)),
        max_tokens=int(config.get("max_tokens", 16384)),
        thinking_enabled=bool(config.get("thinking_enabled", True)),
        thinking_effort=cast(Any, str(config.get("thinking_effort", "max"))),
        thinking_budget_tokens=int(config.get("thinking_budget_tokens", 4096)),
        stream_openai=bool(config.get("stream_openai", True)),
        stream_options_include_usage=bool(
            config.get("stream_options_include_usage", False)
        ),
        wall_clock_deadline_seconds=float(
            config.get("wall_clock_deadline_seconds", 900)
        ),
        max_requests=int(config.get("max_requests", 4100)),
        max_output_tokens_total=int(config.get("max_output_tokens_total", 12_500_000)),
        provider_preset=spec.preset,
        model_source=selection.model_source,
        discovery_status=selection.discovery.status,
        discovery_model_count=len(selection.discovery.models),
    )


def _validate_bundle_identity(card: TaskCard, value: Mapping[str, Any]) -> None:
    if (
        value.get("card_id") != card.card_id
        or value.get("instance_id") != card.source.instance_id
        or value.get("source_fingerprint") != card.source_fingerprint
    ):
        raise SWEBenchValidationError(
            "execution bundle identity differs from task card"
        )


def _parse_teacher_json(value: str, stage: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SWEBenchValidationError(
            f"{stage} teacher returned non-JSON output"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise SWEBenchValidationError(f"{stage} teacher output must be an object")
    return dict(parsed)


def _validate_teacher_output_shape(stage: str, value: Mapping[str, Any]) -> None:
    """Reject unexpected fields before any teacher bytes become durable."""

    expected = {
        "planner": {
            "schema_version",
            "alignment_id",
            "domain_id",
            "builder_expert_id",
            "reviewer_expert_id",
            "work_items",
            "tool_proposals",
        },
        "tool_policy": {
            "schema_version",
            "alignment_id",
            "executed_expert_id",
            "decisions",
        },
        "domain_review": {
            "schema_version",
            "alignment_id",
            "revision",
            "executed_expert_id",
            "decision",
            "feedback",
        },
        "security": {
            "schema_version",
            "alignment_id",
            "executed_expert_id",
            "decision",
            "findings",
        },
    }
    if stage not in expected or set(value) != expected[stage]:
        raise SWEBenchValidationError(f"{stage} teacher output has unexpected fields")
    nested_name = "tool_proposals" if stage == "planner" else "decisions"
    if stage in {"planner", "tool_policy"}:
        nested = value.get(nested_name)
        nested_fields = (
            {"proposal_id", "tool", "purpose", "input"}
            if stage == "planner"
            else {"proposal_id", "decision", "reason"}
        )
        if not isinstance(nested, list) or any(
            not isinstance(item, Mapping) or set(item) != nested_fields
            for item in nested
        ):
            raise SWEBenchValidationError(
                f"{stage} teacher output has unexpected nested fields"
            )
    public_list = {
        "planner": "work_items",
        "domain_review": "feedback",
        "security": "findings",
    }.get(stage)
    if public_list is not None:
        items = value.get(public_list)
        if not isinstance(items, list) or any(
            not isinstance(item, str) for item in items
        ):
            raise SWEBenchValidationError(
                f"{stage} teacher output {public_list} must be a string list"
            )


def _reject_oracle_source_material(
    value: Any,
    *,
    allow_generated_diff: bool = False,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = re.sub(r"[^a-z0-9]", "", str(raw_key).casefold())
            generated_patch = (
                allow_generated_diff
                and key == "patch"
                and len(path) >= 2
                and path[-2] == "final_diff"
                and path[-1].isdigit()
            )
            key_forbidden = (
                key in _FORBIDDEN_SOURCE_KEYS
                or key.startswith(
                    ("gold", "oracle", "hint", "failtopass", "passtopass")
                )
                or key.endswith(("gold", "oracle", "hint", "patch"))
                or "testname" in key
                or "testcase" in key
            )
            if key_forbidden and not generated_patch:
                raise SWEBenchValidationError(
                    "upstream patch, hidden test, hint, gold, or oracle material is forbidden"
                )
            _reject_oracle_source_material(
                child,
                allow_generated_diff=allow_generated_diff,
                path=(*path, str(raw_key)),
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_oracle_source_material(
                child,
                allow_generated_diff=allow_generated_diff,
                path=(*path, str(index)),
            )
    elif isinstance(value, str) and _ORACLE_MARKER.search(value):
        raise SWEBenchValidationError("oracle metadata marker entered SWE stage data")


def _append_or_verify(store: JsonlStore, value: Mapping[str, Any]) -> None:
    record_id = str(value.get(store.id_field, ""))
    existing = next(
        (row for row in store.records if str(row.get(store.id_field, "")) == record_id),
        None,
    )
    if existing is not None:
        if canonical_json(existing) != canonical_json(dict(value)):
            raise SWEBenchValidationError(
                f"durable record {record_id} disagrees with the current chain"
            )
        return
    store.append(value)


def _verify_stage_record(
    existing: Mapping[str, Any], request_record: Mapping[str, Any]
) -> None:
    if (
        existing.get("record_id") != request_record.get("record_id")
        or existing.get("stage") != request_record.get("stage")
        or existing.get("identity") != request_record.get("identity")
        or existing.get("upstream_record_ids")
        != request_record.get("upstream_record_ids")
        or existing.get("request_sha256")
        != digest_value(_mapping(request_record.get("request"), "stage request"))
    ):
        raise SWEBenchValidationError("stored SWE stage record has stale dependencies")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SWEBenchValidationError(f"{label} must be an object")
    return value


def _project_path(root: Path, value: str, label: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SWEBenchValidationError(
            f"{label} must stay inside the project root"
        ) from exc
    return path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


__all__ = [
    "BATCH_CONFIG_SCHEMA_VERSION",
    "BATCH_MANIFEST_SCHEMA_VERSION",
    "EXECUTION_BUNDLE_SCHEMA_VERSION",
    "REPLAY_RESPONSE_SCHEMA_VERSION",
    "STAGE_RECORD_SCHEMA_VERSION",
    "WORK_ORDER_SCHEMA_VERSION",
    "BatchConfig",
    "BatchResult",
    "ReplayTeacher",
    "build_live_teacher",
    "compile_manifest",
    "compile_work_orders",
    "load_replay_teacher",
    "load_validated_cards",
    "run_batch",
    "stage_record_id",
    "write_work_orders",
]
