"""Content-free task-bank coverage and seed-level near-duplicate gates."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
import unicodedata
from typing import Any, Mapping, Sequence

from .task_cards import TASK_CARD_AXES, CardAssignment, TaskCardCatalog


NEAR_DUPLICATE_POLICY_ID = "anchor-requirement-near-duplicate-v1"
TASK_CARD_SCHEMA_ID = "anchor.task-card-coverage-matrix.v2"
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_CJK = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$")


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def normalized_requirement_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    result: list[str] = []
    for match in _TOKEN.finditer(normalized):
        token = match.group(0)
        result.extend(token if _CJK.fullmatch(token) else (token,))
    return tuple(result)


def _ngrams(tokens: Sequence[str], size: int) -> frozenset[tuple[str, ...]]:
    if len(tokens) < size:
        return frozenset()
    return frozenset(
        tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    )


def _containment(left: frozenset[Any], right: frozenset[Any]) -> float:
    denominator = min(len(left), len(right))
    return len(left.intersection(right)) / denominator if denominator else 0.0


def _requirements_near_duplicate(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    ngram_size: int,
    unigram_threshold: float,
) -> bool:
    if len(left) < 8 or len(right) < 8:
        return left == right
    unigrams_left = frozenset(left)
    unigrams_right = frozenset(right)
    if _containment(unigrams_left, unigrams_right) < unigram_threshold:
        return False
    return _containment(_ngrams(left, ngram_size), _ngrams(right, ngram_size)) >= 0.75


def detect_near_duplicate_seeds(
    candidates: Sequence[Mapping[str, Any]],
    catalog: TaskCardCatalog,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Select stable representatives and return content-free loser evidence."""

    prepared: list[tuple[int, str, tuple[str, ...], str]] = []
    for item in candidates:
        seed_id = str(item.get("seed_id", "")).strip()
        requirement = item.get("requirement")
        raw_index = item.get("seed_index")
        seed_index = (
            raw_index
            if isinstance(raw_index, int) and not isinstance(raw_index, bool)
            else 2**63 - 1
        )
        if not seed_id or not isinstance(requirement, str):
            continue
        tokens = normalized_requirement_tokens(requirement)
        normalized_hash = _canonical_sha(tokens)
        prepared.append((seed_index, seed_id, tokens, normalized_hash))
    prepared.sort(key=lambda item: (item[0], item[1]))

    representatives: list[tuple[str, tuple[str, ...], str]] = []
    losers: dict[str, dict[str, str]] = {}
    for _seed_index, seed_id, tokens, normalized_hash in prepared:
        matched: tuple[str, tuple[str, ...], str] | None = None
        for representative in representatives:
            if _requirements_near_duplicate(
                representative[1],
                tokens,
                ngram_size=catalog.near_duplicate_ngram_size,
                unigram_threshold=catalog.near_duplicate_threshold,
            ):
                matched = representative
                break
        if matched is None:
            representatives.append((seed_id, tokens, normalized_hash))
            continue
        losers[seed_id] = {
            "representative_seed_id": matched[0],
            "normalized_requirement_sha256": normalized_hash,
            "representative_requirement_sha256": matched[2],
        }

    policy = {
        "policy_id": NEAR_DUPLICATE_POLICY_ID,
        "ngram_size": catalog.near_duplicate_ngram_size,
        "unigram_containment_threshold": catalog.near_duplicate_threshold,
        "ngram_containment_threshold": 0.75,
        "short_requirement_policy": "normalized_exact_under_8_tokens",
    }
    candidate_ids = [item[1] for item in prepared]
    negative_ids = sorted(losers)
    normalized_bindings = sorted((item[1], item[3]) for item in prepared)
    report = {
        "policy_id": NEAR_DUPLICATE_POLICY_ID,
        "policy_sha256": _canonical_sha(policy),
        "candidate_seed_count": len(prepared),
        "representative_seed_count": len(prepared) - len(losers),
        "negative_seed_count": len(losers),
        "candidate_seed_ids_sha256": _canonical_sha(candidate_ids),
        "negative_seed_ids_sha256": _canonical_sha(negative_ids),
        "normalized_requirements_sha256": _canonical_sha(normalized_bindings),
        "passed": True,
    }
    return losers, report


def evaluate_task_card_coverage(
    assignments: Sequence[tuple[str, CardAssignment]],
    catalog: TaskCardCatalog,
    *,
    minimum_complete_chain_count: int,
    task_bank_count: int,
    stage_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Evaluate hard one-card/one-chain cardinality and nine-axis coverage.

    Legacy collected rows participate in the hard cardinality proof but never
    acquire inferred axis labels.  Held-out source rows are forbidden from the
    training bank entirely.
    """

    complete_chain_count = len(assignments)
    card_ids = [assignment.card_id for _seed_id, assignment in assignments]
    alignments = [
        assignment.alignment_for_seed(seed_id) for seed_id, assignment in assignments
    ]
    unique_card_ids = sorted(set(card_ids))
    unique_alignment_ids = sorted(set(alignments))
    card_count = len(unique_card_ids)
    unique_alignment_id_count = len(unique_alignment_ids)
    stage_cardinality_equal = all(
        isinstance(count, int)
        and not isinstance(count, bool)
        and count == complete_chain_count
        for count in stage_counts.values()
    ) and set(stage_counts) == {
        "plan",
        "tool_policy",
        "frontend",
        "review",
        "security",
    }
    cardinality_equal = (
        card_count
        == unique_alignment_id_count
        == complete_chain_count
        == task_bank_count
        and stage_cardinality_equal
    )

    heldout = [
        assignment
        for _seed_id, assignment in assignments
        if assignment.source_kind == "swebench_heldout"
    ]
    canonical = [
        assignment
        for _seed_id, assignment in assignments
        if not assignment.legacy
        and assignment.source_kind != "swebench_heldout"
        and assignment.axes is not None
        and assignment.template_id is not None
    ]
    legacy_count = sum(assignment.legacy for _seed_id, assignment in assignments)
    template_counts = Counter(assignment.template_id for assignment in canonical)
    axis_counts: dict[str, Counter[str]] = {
        axis: Counter(assignment.axes[axis] for assignment in canonical)
        for axis in TASK_CARD_AXES
    }

    enforce_coverage = len(canonical) >= len(catalog.cards)
    required_per_template = (
        len(canonical) // len(catalog.cards) if enforce_coverage else 0
    )
    template_shortfalls = [
        {
            "template_id": template.template_id,
            "count": template_counts[template.template_id],
        }
        for template in catalog.cards
        if template_counts[template.template_id] < required_per_template
    ]
    axis_shortfalls: list[dict[str, Any]] = []
    if enforce_coverage:
        for axis in TASK_CARD_AXES:
            required = len(canonical) // len(catalog.values[axis])
            for value in catalog.values[axis]:
                if axis_counts[axis][value] < required:
                    axis_shortfalls.append(
                        {
                            "axis_id": axis,
                            "value_id": value,
                            "count": axis_counts[axis][value],
                        }
                    )

    cells = [
        {
            "row_id": template.template_id,
            "column_id": task,
            "count": template_counts[template.template_id],
        }
        for template in catalog.cards
        for task in ("plan", "tool_policy", "frontend", "review", "security")
    ]
    axes = [
        {"axis_id": axis, "value_id": value, "count": axis_counts[axis][value]}
        for axis in TASK_CARD_AXES
        for value in catalog.values[axis]
    ]
    coverage_passed = not enforce_coverage or not (
        template_shortfalls or axis_shortfalls
    )
    return {
        "catalog_id": TASK_CARD_SCHEMA_ID,
        "catalog_sha256": catalog.sha256,
        "complete_chain_count": complete_chain_count,
        "minimum_complete_chain_count": minimum_complete_chain_count,
        "task_bank_count": task_bank_count,
        "card_count": card_count,
        "unique_alignment_id_count": unique_alignment_id_count,
        "card_ids_sha256": _canonical_sha(unique_card_ids),
        "alignment_ids_sha256": _canonical_sha(unique_alignment_ids),
        "stage_counts": dict(sorted(stage_counts.items())),
        "stage_cardinality_equal": stage_cardinality_equal,
        "cardinality_equal": cardinality_equal,
        "cardinality_passed": cardinality_equal,
        "canonical_chain_count": len(canonical),
        "legacy_chain_count": legacy_count,
        "heldout_chain_count": len(heldout),
        "cell_count": len(cells),
        "cells": cells,
        "cells_sha256": _canonical_sha(cells),
        "axis_value_count": len(axes),
        "axes": axes,
        "axes_sha256": _canonical_sha(axes),
        "required_per_template": required_per_template,
        "template_shortfall_count": len(template_shortfalls),
        "template_shortfalls_sha256": _canonical_sha(template_shortfalls),
        "axis_shortfall_count": len(axis_shortfalls),
        "axis_shortfalls_sha256": _canonical_sha(axis_shortfalls),
        "coverage_enforced": enforce_coverage,
        "coverage_passed": coverage_passed,
        "mode": "canonical" if canonical else "legacy_only",
        "passed": cardinality_equal and coverage_passed and not heldout,
    }
