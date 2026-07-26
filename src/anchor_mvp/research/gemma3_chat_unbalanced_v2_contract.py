"""Cross-field validators for the unbalanced Chat v2 contract proposal.

The JSON Schemas intentionally remain adapter-arm neutral.  Relationships that
JSON Schema Draft 2020-12 cannot express (set inclusion, digest recomputation,
and route complements) are enforced here.  This module performs no I/O and
does not materialize records.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEGMENT_ID_RE = re.compile(r"^chat-segment-v1:[0-9a-f]{64}$")
STYLE_ROLES = ("humor", "serious", "angry_style")
ROUTER_RELATIONS = {
    "humor": ("lighthearted", "lighten_without_trivializing"),
    "serious": ("calm_precise", "prioritize_precision_and_calm"),
    "angry_style": ("firm_direct", "acknowledge_anger_with_firm_clarity"),
}
ORDERED_SEGMENT_DOMAIN = "anchor.ordered-segment-ids.v1"
ALLOWED_INTERSECTION_DOMAIN = "anchor.allowed-context-intersection.v1"
ROUTER_SEMANTIC_DOMAIN = "anchor.gemma3-chat-emotion-route-semantic.v1"
EVIDENCE_SET_DOMAIN = "anchor.route-evidence-set.v1"
REVIEW_DEPENDENCY_DOMAIN = "anchor.review-tool-dependency.v2"
TASK_TEMPLATE_PAIR_DOMAIN = "anchor.task-template-pair.v1"


class ChatUnbalancedV2ContractError(ValueError):
    """Raised when one proposal record violates a cross-field invariant."""


def _canonical_bytes(value: Any) -> bytes:
    """Encode the contract's restricted RFC 8785-compatible JSON subset."""

    def walk(node: Any, path: str) -> None:
        if node is None or isinstance(node, (bool, str, int)):
            return
        if isinstance(node, float):
            raise ChatUnbalancedV2ContractError(
                f"{path}: floating-point values are forbidden in hash preimages"
            )
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
            return
        if isinstance(node, dict):
            for key, item in node.items():
                if not isinstance(key, str) or not key.isascii():
                    raise ChatUnbalancedV2ContractError(
                        f"{path}: canonical object keys must be ASCII strings"
                    )
                walk(item, f"{path}.{key}")
            return
        raise ChatUnbalancedV2ContractError(
            f"{path}: unsupported canonical value type {type(node).__name__}"
        )

    walk(value, "$")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(domain_tag: str, value: Any) -> str:
    if not isinstance(domain_tag, str) or not domain_tag.isascii() or not domain_tag:
        raise ChatUnbalancedV2ContractError("domain tag must be non-empty ASCII")
    return hashlib.sha256(
        domain_tag.encode("ascii") + b"\x00" + _canonical_bytes(value)
    ).hexdigest()


def content_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise ChatUnbalancedV2ContractError("content must be text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ChatUnbalancedV2ContractError(f"{path}: expected object")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChatUnbalancedV2ContractError(f"{path}: expected non-empty text")
    return value


def _sha256(value: Any, path: str) -> str:
    text = _text(value, path)
    if SHA256_RE.fullmatch(text) is None:
        raise ChatUnbalancedV2ContractError(f"{path}: invalid sha256")
    return text


def _segment_ids(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ChatUnbalancedV2ContractError(f"{path}: expected non-empty array")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _text(item, f"{path}[{index}]")
        if SEGMENT_ID_RE.fullmatch(text) is None:
            raise ChatUnbalancedV2ContractError(f"{path}[{index}]: invalid segment id")
        result.append(text)
    if len(result) != len(set(result)):
        raise ChatUnbalancedV2ContractError(f"{path}: duplicate segment id")
    return tuple(result)


def validate_attention_specialization(value: Any) -> None:
    attention = _mapping(value, "attention_specialization")
    ordered = _segment_ids(
        attention.get("ordered_segment_ids"),
        "attention_specialization.ordered_segment_ids",
    )
    expected_ordered_digest = canonical_sha256(ORDERED_SEGMENT_DOMAIN, list(ordered))
    if (
        _sha256(
            attention.get("ordered_segment_ids_sha256"),
            "attention_specialization.ordered_segment_ids_sha256",
        )
        != expected_ordered_digest
    ):
        raise ChatUnbalancedV2ContractError("ordered segment digest mismatch")
    _sha256(
        attention.get("prefix_lineage_sha256"),
        "attention_specialization.prefix_lineage_sha256",
    )

    intersection = _mapping(
        attention.get("allowed_context_intersection"),
        "attention_specialization.allowed_context_intersection",
    )
    allowed = _segment_ids(
        intersection.get("segment_ids"),
        "attention_specialization.allowed_context_intersection.segment_ids",
    )
    if allowed != ordered:
        raise ChatUnbalancedV2ContractError(
            "allowed context intersection must equal the authenticated ordered prefix"
        )
    expected_allowed_digest = canonical_sha256(
        ALLOWED_INTERSECTION_DOMAIN, list(allowed)
    )
    if (
        _sha256(
            intersection.get("segment_ids_sha256"),
            "attention_specialization.allowed_context_intersection.segment_ids_sha256",
        )
        != expected_allowed_digest
    ):
        raise ChatUnbalancedV2ContractError("allowed intersection digest mismatch")

    salient = set(
        _segment_ids(
            attention.get("salient_segment_ids"),
            "attention_specialization.salient_segment_ids",
        )
    )
    distractors = set(
        _segment_ids(
            attention.get("distractor_segment_ids"),
            "attention_specialization.distractor_segment_ids",
        )
    )
    allowed_set = set(allowed)
    if salient & distractors:
        raise ChatUnbalancedV2ContractError(
            "salient and distractor segment sets overlap"
        )
    if not salient <= allowed_set or not distractors <= allowed_set:
        raise ChatUnbalancedV2ContractError(
            "salient/distractor segment lies outside allowed intersection"
        )
    if salient | distractors != allowed_set:
        raise ChatUnbalancedV2ContractError(
            "salient and distractor sets must partition allowed intersection"
        )

    raw_refs = attention.get("route_relevant_evidence_refs")
    if (
        isinstance(raw_refs, (str, bytes))
        or not isinstance(raw_refs, Sequence)
        or not raw_refs
    ):
        raise ChatUnbalancedV2ContractError(
            "route_relevant_evidence_refs must be non-empty"
        )
    ref_segments: list[str] = []
    for index, raw_ref in enumerate(raw_refs):
        ref = _mapping(
            raw_ref,
            f"attention_specialization.route_relevant_evidence_refs[{index}]",
        )
        segment_id = _text(
            ref.get("segment_id"),
            f"attention_specialization.route_relevant_evidence_refs[{index}].segment_id",
        )
        if SEGMENT_ID_RE.fullmatch(segment_id) is None:
            raise ChatUnbalancedV2ContractError("evidence ref has invalid segment id")
        _sha256(
            ref.get("content_sha256"),
            f"attention_specialization.route_relevant_evidence_refs[{index}].content_sha256",
        )
        ref_segments.append(segment_id)
    if not set(ref_segments) <= salient:
        raise ChatUnbalancedV2ContractError(
            "route-relevant evidence must resolve only to salient segments"
        )

    tail = _mapping(
        attention.get("private_tail_contract"),
        "attention_specialization.private_tail_contract",
    )
    exact_scope = tail.get("exact_reuse_scope")
    if exact_scope != "identical_token_order_positions_rope_and_prefix_lineage_only":
        raise ChatUnbalancedV2ContractError("exact reuse scope changed")
    if tail.get("cross_expert_reuse_allowed") is not False and (
        tail.get("cross_expert_private_tail_reuse") is not False
    ):
        raise ChatUnbalancedV2ContractError("private tail reuse must be disabled")
    if tail.get("full_generation_kv_shared_claimed") is True or (
        tail.get("ordinary_in_stack_full_layer_kv_shared_claimed") is True
    ):
        raise ChatUnbalancedV2ContractError("forbidden KV-sharing claim enabled")


def validate_router_record(value: Any) -> None:
    record = _mapping(value, "router_record")
    descriptor = _mapping(record.get("semantic_descriptor"), "semantic_descriptor")
    expected_semantic = canonical_sha256(ROUTER_SEMANTIC_DOMAIN, descriptor)
    if _sha256(record.get("task_semantic_sha256"), "task_semantic_sha256") != (
        expected_semantic
    ):
        raise ChatUnbalancedV2ContractError("router semantic identity mismatch")

    target = _text(record.get("target_expert_id"), "target_expert_id")
    if target not in STYLE_ROLES:
        raise ChatUnbalancedV2ContractError("unknown emotion-router target")
    allowed = record.get("allowed_expert_ids")
    forbidden = record.get("forbidden_expert_ids")
    if allowed != [target]:
        raise ChatUnbalancedV2ContractError("allowed route must contain only target")
    expected_forbidden = [role for role in STYLE_ROLES if role != target]
    if forbidden != expected_forbidden:
        raise ChatUnbalancedV2ContractError("forbidden route complement changed")

    requested_tone, relation = ROUTER_RELATIONS[target]
    if descriptor.get("requested_tone") != requested_tone:
        raise ChatUnbalancedV2ContractError("requested tone/target route mismatch")
    if descriptor.get("target_expert_relation") != relation:
        raise ChatUnbalancedV2ContractError("semantic relation/target route mismatch")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ChatUnbalancedV2ContractError("router messages must be [system,user]")
    user = _mapping(messages[1], "messages[1]")
    if user.get("role") != "user":
        raise ChatUnbalancedV2ContractError("router final input message must be user")
    if content_sha256(_text(user.get("content"), "messages[1].content")) != _sha256(
        record.get("user_message_sha256"), "user_message_sha256"
    ):
        raise ChatUnbalancedV2ContractError("router user message digest mismatch")

    validate_attention_specialization(record.get("attention_specialization"))
    attention = _mapping(
        record.get("attention_specialization"), "attention_specialization"
    )
    salient = set(attention["salient_segment_ids"])
    evidence = _mapping(record.get("user_emotion_evidence"), "user_emotion_evidence")
    evidence_segments = _segment_ids(
        evidence.get("evidence_segment_ids"),
        "user_emotion_evidence.evidence_segment_ids",
    )
    if not set(evidence_segments) <= salient:
        raise ChatUnbalancedV2ContractError(
            "emotion evidence must resolve only to salient segments"
        )
    expected_evidence_digest = canonical_sha256(
        EVIDENCE_SET_DOMAIN, list(evidence_segments)
    )
    if (
        _sha256(
            evidence.get("evidence_set_sha256"),
            "user_emotion_evidence.evidence_set_sha256",
        )
        != expected_evidence_digest
    ):
        raise ChatUnbalancedV2ContractError("emotion evidence digest mismatch")

    topology = _mapping(record.get("route_topology"), "route_topology")
    forbidden_true = (
        "return_to_router",
        "router_rewrite",
        "router_vote",
        "router_aggregate",
        "review_in_main_path",
    )
    if any(topology.get(field) is not False for field in forbidden_true):
        raise ChatUnbalancedV2ContractError("router returned to a global controller")
    if topology.get("router_exits_after_selection") is not True:
        raise ChatUnbalancedV2ContractError("router must exit after selection")


def validate_generation_record(value: Any) -> None:
    record = _mapping(value, "generation_record")
    role = _text(record.get("role"), "role")
    bundle_sha = _sha256(record.get("task_bundle_sha256"), "task_bundle_sha256")
    expected_record_id = f"gemma3-chat-unbalanced-v2:{bundle_sha}:{role}"
    if record.get("record_id") != expected_record_id:
        raise ChatUnbalancedV2ContractError("generation record id mismatch")

    source_user = _mapping(record.get("source_user"), "source_user")
    if content_sha256(_text(source_user.get("content"), "source_user.content")) != (
        _sha256(source_user.get("content_sha256"), "source_user.content_sha256")
    ):
        raise ChatUnbalancedV2ContractError("source user digest mismatch")
    target = _mapping(record.get("target"), "target")
    if content_sha256(_text(target.get("assistant_text"), "target.assistant_text")) != (
        _sha256(target.get("output_sha256"), "target.output_sha256")
    ):
        raise ChatUnbalancedV2ContractError("target digest mismatch")

    route = _mapping(record.get("route_control"), "route_control")
    if route.get("selected_role") != role:
        raise ChatUnbalancedV2ContractError(
            "route selected role differs from record role"
        )
    if route.get("return_to_router") is not False:
        raise ChatUnbalancedV2ContractError("generation route returned to router")
    if route.get("posthoc_aggregation") is not False:
        raise ChatUnbalancedV2ContractError("post-hoc aggregation entered main path")

    validate_attention_specialization(record.get("attention_specialization"))

    review = _mapping(record.get("review_dependency"), "review_dependency")
    if role == "review_audit":
        expected_tool_id = f"gemma3-chat-unbalanced-v2:{bundle_sha}:tool_call"
        if review.get("tool_record_id") != expected_tool_id:
            raise ChatUnbalancedV2ContractError("review dependency crosses bundle")
        preimage = {
            "tool_record_id": expected_tool_id,
            "tool_target_sha256": _sha256(
                review.get("tool_target_sha256"),
                "review_dependency.tool_target_sha256",
            ),
            "candidate_projection_sha256": _sha256(
                review.get("candidate_projection_sha256"),
                "review_dependency.candidate_projection_sha256",
            ),
        }
        if _sha256(
            review.get("dependency_set_sha256"),
            "review_dependency.dependency_set_sha256",
        ) != canonical_sha256(REVIEW_DEPENDENCY_DOMAIN, preimage):
            raise ChatUnbalancedV2ContractError("review dependency digest mismatch")
        verdict = review.get("verdict")
        if verdict == "pass":
            if review.get("fault_type") is not None:
                raise ChatUnbalancedV2ContractError("passing review has fault")
            if review.get("corrected_output_sha256") is not None:
                raise ChatUnbalancedV2ContractError("passing review has correction")
        elif verdict == "fail":
            _text(review.get("fault_type"), "review_dependency.fault_type")
            _sha256(
                review.get("corrected_output_sha256"),
                "review_dependency.corrected_output_sha256",
            )
        else:
            raise ChatUnbalancedV2ContractError("review verdict is invalid")


def validate_router_semantic_inventory(records: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    count = 0
    for record in records:
        validate_router_record(record)
        semantic_id = _sha256(
            record.get("task_semantic_sha256"), "task_semantic_sha256"
        )
        if semantic_id in seen:
            raise ChatUnbalancedV2ContractError("duplicate router semantic identity")
        seen.add(semantic_id)
        count += 1
    if count != 100:
        raise ChatUnbalancedV2ContractError("router inventory must contain 100 rows")


def validate_comparison_identity(value: Any) -> None:
    identity = _mapping(value, "comparison_identity")
    if identity.get("role") != "tool_call":
        raise ChatUnbalancedV2ContractError("comparison identity is not tool_call")
    pair_preimage = {
        "task_semantic_sha256": _sha256(
            identity.get("task_semantic_sha256"), "task_semantic_sha256"
        ),
        "template_family_sha256": _sha256(
            identity.get("template_family_sha256"), "template_family_sha256"
        ),
    }
    expected_pair = canonical_sha256(TASK_TEMPLATE_PAIR_DOMAIN, pair_preimage)
    if (
        _sha256(identity.get("task_template_pair_sha256"), "task_template_pair_sha256")
        != expected_pair
    ):
        raise ChatUnbalancedV2ContractError("task-template pair digest mismatch")
    forbidden_body_keys = {
        "content",
        "prompt",
        "answer",
        "messages",
        "token_ids",
        "adapter_arm",
    }
    if forbidden_body_keys & set(identity):
        raise ChatUnbalancedV2ContractError(
            "comparison identity contains body/arm field"
        )
