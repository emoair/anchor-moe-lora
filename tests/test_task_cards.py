from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
import re
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.data.coverage import (  # noqa: E402
    detect_near_duplicate_seeds,
    evaluate_task_card_coverage,
)
from anchor_mvp.data.pipeline import DistillationPipeline  # noqa: E402
from anchor_mvp.data.schema import SeedDemand  # noqa: E402
from anchor_mvp.data.task_cards import (  # noqa: E402
    TASK_CARD_AXES,
    assignment_for_requirement,
    assignment_for_seed,
    load_task_card_catalog,
)
from anchor_mvp.data.teacher import MockTeacher  # noqa: E402


def test_catalog_is_nine_axis_balanced_and_indices_select_templates() -> None:
    catalog = load_task_card_catalog()

    assert tuple(catalog.values) == TASK_CARD_AXES
    assert len(catalog.cards) == 16
    assert catalog.template_for_index(0) == catalog.template_for_index(16)
    assert "-slot-" not in catalog.template_for_index(16).template_id
    for axis, values in catalog.values.items():
        counts = [
            sum(card.axes[axis] == value for card in catalog.cards) for value in values
        ]
        assert max(counts) - min(counts) <= 1


def test_final_card_identity_is_requirement_bound_not_slot_bound() -> None:
    catalog = load_task_card_catalog()
    template = catalog.template_for_index(0)

    first = assignment_for_requirement(
        "Build a local alert console with one filter.",
        template,
        catalog,
        seed_index=0,
    )
    same = assignment_for_requirement(
        "  BUILD a local alert console with one filter. ",
        template,
        catalog,
        seed_index=0,
    )
    different = assignment_for_requirement(
        "Build a local incident console with one acknowledgement action.",
        template,
        catalog,
        seed_index=16,
    )

    assert first.card_id == same.card_id
    assert first.card_id != different.card_id
    assert first.template_id == different.template_id == template.template_id
    assert first.alignment_for_seed("seed-a") != first.alignment_for_seed("seed-b")


def test_old_slot_and_variant_seeds_become_unique_unlabelled_legacy_cards() -> None:
    catalog = load_task_card_catalog()
    seeds = [
        SeedDemand(
            seed_id=f"old-{index}",
            title="old",
            request=f"Accepted historical requirement {index}",
            tags=("variant-00", "domain:operations"),
            card_id=f"card-00-operations-slot-{index:08d}",
            seed_index=index,
        )
        for index in range(2)
    ]
    assignments = [assignment_for_seed(seed, catalog) for seed in seeds]

    assert len({item.card_id for item in assignments}) == 2
    assert all(item.legacy and item.axes is None for item in assignments)
    assert all(item.source_kind == "legacy_collected" for item in assignments)
    assert all(item.tags == ("variant-00",) for item in assignments)


def test_external_sources_require_and_preserve_immutable_digest(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (ROOT / "configs" / "data" / "task_cards.v1.yaml").read_text(encoding="utf-8")
    )
    digest = "a" * 64
    raw["cards"][0]["source_kind"] = "swebench_heldout"
    raw["cards"][0]["source_digest"] = digest
    config = tmp_path / "cards.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    catalog = load_task_card_catalog(config)
    assignment = assignment_for_requirement(
        "Public issue statement only; no patch or test oracle.",
        catalog.template_for_index(0),
        catalog,
        seed_index=0,
    )

    assert assignment.source_kind == "swebench_heldout"
    assert assignment.source_digest == digest
    assert assignment.provenance("seed-heldout")["source_digest"] == digest
    assert "patch" not in assignment.provenance("seed-heldout")
    report = evaluate_task_card_coverage(
        [("seed-heldout", assignment)],
        catalog,
        minimum_complete_chain_count=1,
        task_bank_count=1,
        stage_counts={
            task: 1
            for task in ("plan", "tool_policy", "frontend", "review", "security")
        },
    )
    assert report["heldout_chain_count"] == 1
    assert report["canonical_chain_count"] == 0
    assert report["passed"] is False


def test_cardinality_is_hard_but_legacy_axis_coverage_is_not_fabricated() -> None:
    catalog = load_task_card_catalog()
    assignments = []
    for index in range(2):
        seed = SeedDemand(
            seed_id=f"legacy-{index}",
            title="legacy",
            request=f"Unique accepted legacy request {index}",
        )
        assignments.append((seed.seed_id, assignment_for_seed(seed, catalog)))
    report = evaluate_task_card_coverage(
        assignments,
        catalog,
        minimum_complete_chain_count=2,
        task_bank_count=2,
        stage_counts={
            task: 2
            for task in ("plan", "tool_policy", "frontend", "review", "security")
        },
    )

    assert report["card_count"] == report["unique_alignment_id_count"] == 2
    assert report["cardinality_equal"] is True
    assert report["mode"] == "legacy_only"
    assert report["canonical_chain_count"] == 0
    assert report["coverage_enforced"] is False
    assert report["passed"] is True


def test_near_duplicate_gate_chooses_stable_seed_level_representative() -> None:
    catalog = load_task_card_catalog()
    candidates = [
        {
            "seed_id": "later",
            "seed_index": 9,
            "requirement": "Build a local accessible inventory table with search filters and clear empty states",
        },
        {
            "seed_id": "first",
            "seed_index": 1,
            "requirement": "Build a local accessible inventory table with search filters and clear empty state",
        },
    ]

    losers, report = detect_near_duplicate_seeds(candidates, catalog)

    assert losers["later"]["representative_seed_id"] == "first"
    assert report["negative_seed_count"] == 1
    assert all(item["requirement"] not in str(report) for item in candidates)


class _RetrySameSlotTeacher(MockTeacher):
    def __init__(self) -> None:
        self.calls: Counter[int] = Counter()

    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: seed" not in user:
            return await super().complete(system=system, user=user)
        match = re.search(r"^SEED_INDEX:\s*(\d+)$", user, flags=re.MULTILINE)
        assert match is not None
        index = int(match.group(1))
        self.calls[index] += 1
        if self.calls[index] == 1:
            duplicate_user = re.sub(
                r"^SEED_INDEX:\s*\d+$",
                "SEED_INDEX: 0",
                user,
                flags=re.MULTILINE,
            )
            return await super().complete(system=system, user=duplicate_user)
        return await super().complete(system=system, user=user)


def test_seed_retry_keeps_same_schedule_slots(tmp_path: Path) -> None:
    bootstrap = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=1,
    )
    asyncio.run(bootstrap.generate_seeds(1))
    teacher = _RetrySameSlotTeacher()
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=3,
    )

    seeds = asyncio.run(pipeline.generate_seeds(2))

    assert teacher.calls == Counter({1: 2})
    assert [seed.seed_index for seed in seeds] == [0, 1]
    assert len({seed.card_id for seed in seeds}) == 2
