"""Deterministic task-card templates and content-bound card identities.

The catalog contains *sampling templates*, not final task cards.  A final card
is materialized only after a canonical requirement (or an immutable external
source instance) exists.  This distinction prevents a rotating set of briefs
from being counted as a large task bank merely by appending slot numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

import yaml

from .schema import SeedDemand, stable_id


TASK_CARD_SCHEMA = "anchor.task-card-catalog.v1"
TASK_CARD_AXES = (
    "domain",
    "interaction",
    "layout",
    "edge_case",
    "complexity",
    "accessibility_risk",
    "tool_posture",
    "review_defect",
    "security_class",
)
DEFAULT_TASK_CARD_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "data" / "task_cards.v1.yaml"
)
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_LEGACY_VARIANT = re.compile(r"^variant-(\d{2})$")
_OLD_SLOT_CARD = re.compile(r"^.+-slot-\d{8,}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_KINDS = frozenset({"self_synthetic", "swe_smith", "swebench_heldout"})


def canonical_requirement(value: str) -> str:
    """Return the stable text used for self-synthetic card identity."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


@dataclass(frozen=True)
class TaskCardTemplate:
    """One orthogonal sampling mould from the versioned catalog."""

    template_id: str
    brief: str
    axes: Mapping[str, str]
    source_kind: str = "self_synthetic"
    source_digest: str | None = None

    @property
    def card_id(self) -> str:
        """Compatibility alias for old prompt callers; never a final card id."""

        return self.template_id

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            f"task-card-template:{self.template_id}",
            f"source-kind:{self.source_kind}",
            *(f"{axis}:{self.axes[axis]}" for axis in TASK_CARD_AXES),
        )


# Keep the pre-existing public import while making its meaning explicit.
TaskCard = TaskCardTemplate


@dataclass(frozen=True)
class CardAssignment:
    """Pipeline-owned binding between one accepted seed and one final card."""

    card_id: str
    template_id: str | None
    tags: tuple[str, ...]
    axes: Mapping[str, str] | None
    legacy: bool
    catalog_sha256: str | None
    seed_index: int | None
    source_kind: str
    source_digest: str | None

    def alignment_for_seed(self, seed_id: str) -> str:
        return stable_id("alignment", f"{seed_id}\n{self.card_id}")

    def provenance(self, seed_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "card_id": self.card_id,
            "card_tags": list(self.tags),
            "alignment_id": self.alignment_for_seed(seed_id),
            "task_card_legacy": self.legacy,
            "source_kind": self.source_kind,
        }
        if self.template_id is not None:
            result["template_id"] = self.template_id
        if self.seed_index is not None:
            result["seed_index"] = self.seed_index
        if self.catalog_sha256 is not None:
            result["task_card_catalog_sha256"] = self.catalog_sha256
        if self.source_digest is not None:
            result["source_digest"] = self.source_digest
        return result


@dataclass(frozen=True)
class TaskCardCatalog:
    source: Path
    cards: tuple[TaskCardTemplate, ...]
    values: Mapping[str, tuple[str, ...]]
    sha256: str
    near_duplicate_ngram_size: int
    near_duplicate_threshold: float

    def template_for_index(self, index: int) -> TaskCardTemplate:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("task-card index must be a non-negative integer")
        return self.cards[index % len(self.cards)]

    def card_for_index(self, index: int) -> TaskCardTemplate:
        """Compatibility alias: indices select templates, never final cards."""

        return self.template_for_index(index)

    def template_by_id(self, template_id: str) -> TaskCardTemplate | None:
        return next(
            (card for card in self.cards if card.template_id == template_id), None
        )

    def card_by_id(self, card_id: str) -> None:
        """Final cards are content-bound and cannot be recovered from the catalog."""

        del card_id
        return None


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _slug(value: object, *, label: str) -> str:
    text = str(value).strip()
    if not _SLUG.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase slug")
    return text


def load_task_card_catalog(path: str | Path | None = None) -> TaskCardCatalog:
    source = Path(path or DEFAULT_TASK_CARD_CONFIG).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != TASK_CARD_SCHEMA:
        raise ValueError("task-card catalog schema is invalid")
    raw_axes = raw.get("axes")
    if not isinstance(raw_axes, Mapping) or tuple(raw_axes) != TASK_CARD_AXES:
        raise ValueError("task-card catalog must declare the canonical axis order")
    values: dict[str, tuple[str, ...]] = {}
    for axis in TASK_CARD_AXES:
        options = raw_axes.get(axis)
        if not isinstance(options, list) or not options:
            raise ValueError(f"task-card axis {axis} must have values")
        normalized = tuple(
            _slug(item, label=f"task-card axis {axis}") for item in options
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"task-card axis {axis} contains duplicate values")
        values[axis] = normalized

    raw_cards = raw.get("cards")
    if not isinstance(raw_cards, list) or len(raw_cards) < 2:
        raise ValueError("task-card catalog requires at least two templates")
    cards: list[TaskCardTemplate] = []
    seen_ids: set[str] = set()
    allowed = {"card_id", "brief", "axes", "source_kind", "source_digest"}
    for position, item in enumerate(raw_cards):
        if (
            not isinstance(item, Mapping)
            or not set(item).issubset(allowed)
            or not {"card_id", "brief", "axes"}.issubset(item)
        ):
            raise ValueError(f"task-card cards[{position}] schema is invalid")
        template_id = _slug(
            item["card_id"], label=f"task-card cards[{position}].card_id"
        )
        if template_id in seen_ids:
            raise ValueError("task-card template identifiers must be unique")
        brief = str(item["brief"]).strip()
        if not brief or len(brief) > 600:
            raise ValueError(f"task-card cards[{position}].brief is invalid")
        raw_card_axes = item["axes"]
        if not isinstance(raw_card_axes, Mapping) or set(raw_card_axes) != set(
            TASK_CARD_AXES
        ):
            raise ValueError(f"task-card cards[{position}] axes are incomplete")
        card_axes = {
            axis: _slug(
                raw_card_axes[axis],
                label=f"task-card cards[{position}].axes.{axis}",
            )
            for axis in TASK_CARD_AXES
        }
        for axis, option in card_axes.items():
            if option not in values[axis]:
                raise ValueError(
                    f"task-card cards[{position}].axes.{axis} is undeclared"
                )
        source_kind = str(item.get("source_kind", "self_synthetic")).strip()
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"task-card cards[{position}].source_kind is invalid")
        raw_source_digest = item.get("source_digest")
        source_digest = (
            str(raw_source_digest).strip().casefold()
            if raw_source_digest is not None
            else None
        )
        if source_kind != "self_synthetic" and not (
            isinstance(source_digest, str) and _SHA256.fullmatch(source_digest)
        ):
            raise ValueError(
                f"task-card cards[{position}] external source digest is invalid"
            )
        if source_kind == "self_synthetic" and source_digest is not None:
            raise ValueError(
                f"task-card cards[{position}] synthetic template cannot claim a source digest"
            )
        seen_ids.add(template_id)
        cards.append(
            TaskCardTemplate(
                template_id=template_id,
                brief=brief,
                axes=card_axes,
                source_kind=source_kind,
                source_digest=source_digest,
            )
        )

    # Every configured cycle is marginally balanced.  Global-index slices are
    # deterministic and complete cycles differ by at most one per axis value.
    for axis in TASK_CARD_AXES:
        counts = [
            sum(card.axes[axis] == value for card in cards) for value in values[axis]
        ]
        if max(counts) - min(counts) > 1:
            raise ValueError(f"task-card axis {axis} is not cycle-balanced")

    coverage = raw.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("task-card catalog coverage settings are required")
    ngram_size = coverage.get("near_duplicate_ngram_size")
    threshold = coverage.get("near_duplicate_threshold")
    if (
        isinstance(ngram_size, bool)
        or not isinstance(ngram_size, int)
        or ngram_size < 2
    ):
        raise ValueError("near_duplicate_ngram_size must be an integer >= 2")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 < float(threshold) <= 1
    ):
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    canonical = {
        "schema_version": TASK_CARD_SCHEMA,
        "axes": values,
        "cards": [
            {
                "card_id": card.template_id,
                "brief": card.brief,
                "axes": card.axes,
                "source_kind": card.source_kind,
                "source_digest": card.source_digest,
            }
            for card in cards
        ],
        "coverage": {
            "near_duplicate_ngram_size": ngram_size,
            "near_duplicate_threshold": float(threshold),
        },
    }
    return TaskCardCatalog(
        source=source,
        cards=tuple(cards),
        values=values,
        sha256=_canonical_digest(canonical),
        near_duplicate_ngram_size=ngram_size,
        near_duplicate_threshold=float(threshold),
    )


def assignment_for_requirement(
    requirement: str,
    template: TaskCardTemplate,
    catalog: TaskCardCatalog,
    *,
    seed_index: int,
) -> CardAssignment:
    """Materialize one final, content-bound card after teacher acceptance."""

    if template.source_kind == "self_synthetic":
        identity = canonical_requirement(requirement)
        if not identity:
            raise ValueError("cannot materialize a task card from an empty requirement")
        card_id = stable_id("card", f"self_synthetic\n{identity}")
    else:
        if template.source_digest is None:
            raise ValueError("external task card requires an immutable source digest")
        card_id = stable_id("card", f"{template.source_kind}\n{template.source_digest}")
    tags = (
        f"task-card:{card_id}",
        f"task-card-template:{template.template_id}",
        f"source-kind:{template.source_kind}",
        *(f"{axis}:{template.axes[axis]}" for axis in TASK_CARD_AXES),
        f"task-card-catalog:{catalog.sha256}",
        f"seed-index:{seed_index}",
    )
    return CardAssignment(
        card_id=card_id,
        template_id=template.template_id,
        tags=tags,
        axes=template.axes,
        legacy=False,
        catalog_sha256=catalog.sha256,
        seed_index=seed_index,
        source_kind=template.source_kind,
        source_digest=template.source_digest,
    )


def assignment_for_card(
    card: TaskCardTemplate,
    catalog: TaskCardCatalog,
    *,
    seed_index: int,
    requirement: str | None = None,
) -> CardAssignment:
    """Compatibility wrapper; final identity requires accepted content."""

    if requirement is None:
        raise ValueError("final task-card assignment requires canonical requirement")
    return assignment_for_requirement(requirement, card, catalog, seed_index=seed_index)


def _legacy_assignment(seed: SeedDemand) -> CardAssignment:
    variants = sorted(tag for tag in seed.tags if _LEGACY_VARIANT.fullmatch(tag))
    legacy_card_id = stable_id(
        "legacy-card", f"{seed.seed_id}\n{canonical_requirement(seed.request)}"
    )
    return CardAssignment(
        card_id=legacy_card_id,
        template_id=None,
        tags=(variants[0],) if variants else (),
        axes=None,
        legacy=True,
        catalog_sha256=None,
        seed_index=seed.seed_index,
        source_kind="legacy_collected",
        source_digest=None,
    )


def assignment_for_seed(seed: SeedDemand, catalog: TaskCardCatalog) -> CardAssignment:
    """Resolve a new canonical card or conservatively map any old accepted seed.

    Old variant-only rows and the short-lived ``-slot-XXXXXXXX`` rows are both
    mapped to a unique ``legacy_collected`` card derived from the accepted seed.
    They retain no fabricated nine-axis labels.
    """

    if seed.template_id is None or seed.source_kind is None:
        return _legacy_assignment(seed)
    if seed.card_id is None or seed.seed_index is None:
        raise ValueError("canonical seed task-card binding is incomplete")
    template = catalog.template_by_id(seed.template_id)
    if template is None:
        raise ValueError("seed task-card template is not in the active catalog")
    if seed.source_kind != template.source_kind:
        raise ValueError("seed task-card source kind does not match its template")
    if seed.source_digest != template.source_digest:
        raise ValueError("seed task-card source digest does not match its template")
    assignment = assignment_for_requirement(
        seed.request, template, catalog, seed_index=seed.seed_index
    )
    if seed.card_id != assignment.card_id or tuple(seed.tags) != assignment.tags:
        raise ValueError("seed task-card id/tags do not match accepted content")
    return assignment


def is_old_slot_card(card_id: str | None) -> bool:
    return isinstance(card_id, str) and bool(_OLD_SLOT_CARD.fullmatch(card_id))


def axis_from_tags(tags: Sequence[object], axis: str) -> str | None:
    prefix = f"{axis}:"
    matches = [str(tag)[len(prefix) :] for tag in tags if str(tag).startswith(prefix)]
    return matches[0] if len(matches) == 1 and matches[0] else None
