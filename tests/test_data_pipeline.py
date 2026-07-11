from __future__ import annotations

import asyncio
from hashlib import sha256
import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.data.pipeline import (  # noqa: E402
    DistillationPipeline,
    _ensure_frontend_public_trace,
    _normalize_frontend_payload,
)
from anchor_mvp.data.proposals import (  # noqa: E402
    deterministic_tool_policy_oracle,
    generate_inert_tool_proposals,
)
from anchor_mvp.data.prompts import task_prompt  # noqa: E402
from anchor_mvp.data.prompts import seed_prompt  # noqa: E402
from anchor_mvp.data.schema import SeedDemand  # noqa: E402
from anchor_mvp.data.sops import load_sop  # noqa: E402
from anchor_mvp.data.teacher import MockTeacher  # noqa: E402
from anchor_mvp.training.schema import validate_jsonl  # noqa: E402


def _run(output: Path):
    pipeline = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=output,
        concurrency=3,
    )
    return asyncio.run(pipeline.run(seed_count=4))


def test_mock_pipeline_is_canonical_safe_and_resumable(tmp_path: Path) -> None:
    first = _run(tmp_path)
    assert first.errors == ()
    assert first.written_by_task == {
        "plan": 4,
        "tool_policy": 4,
        "frontend": 4,
        "review": 4,
        "security": 4,
    }

    for task, expert in (
        ("plan", "planner"),
        ("tool_policy", "tool_policy"),
        ("frontend", "frontend_gen"),
        ("review", "frontend_review"),
        ("security", "security_gate"),
    ):
        path = tmp_path / f"data_{task}.jsonl"
        report = validate_jsonl(path, allowed_experts=[expert])
        assert report["valid_records"] == 4
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert all(record["provenance"]["template_sha256"] for record in records)
        assert all(record["provenance"]["teacher"]["protocol"] == "mock" for record in records)
        assert all(
            record["provenance"]["teacher"]["generation_params"]["thinking_enabled"] is False
            for record in records
        )
        if task == "plan":
            assert all(record["output"]["steps"] for record in records)
        if task == "tool_policy":
            plan_ids = {
                json.loads(line)["id"]
                for line in (tmp_path / "data_plan.jsonl").read_text(encoding="utf-8").splitlines()
            }
            assert all(record["messages"][-1]["content"] in {"APPROVE", "BLOCK", "ESCALATE"} for record in records)
            assert all(record["provenance"]["source_plan_record_id"] in plan_ids for record in records)
            assert all(record["provenance"]["tool_proposals"]["executed"] is False for record in records)
            assert [record["output"]["decision"] for record in records] == [
                "APPROVE", "ESCALATE", "BLOCK", "APPROVE"
            ]
            assert all(record["provenance"]["label_oracle"]["decision"] == record["output"]["decision"] for record in records)
            assert all(
                not any(
                    str(value).startswith(("http://", "https://"))
                    for proposal in record["input"]["tool_proposals"]
                    for value in proposal.values()
                )
                for record in records
            )
        if task == "frontend":
            assert all(record["provenance"]["source_plan_record_id"] for record in records)
            assert all(record["provenance"]["source_tool_policy_record_id"] for record in records)
            assert all(record["input"]["plan"]["steps"] for record in records)
            assert all(record["input"]["tool_policy"]["decision"] in {"APPROVE", "BLOCK", "ESCALATE"} for record in records)
        if task == "review":
            assert all("CANDIDATE CODE:" in record["messages"][0]["content"] for record in records)
            assert all("KNOWN_BENIGN_DEFECT:" in record["messages"][0]["content"] for record in records)
            assert all(
                record["input"]["candidate_code"] in record["messages"][0]["content"]
                and record["input"]["known_benign_defect"] in record["messages"][0]["content"]
                for record in records
            )
            assert all(
                record["messages"][0]["content"] != record["input"]["requirement"]
                for record in records
            )
            assert all(
                record["input"]["candidate_code"] != record["output"]["code"]
                and record["messages"][-1]["content"] == record["output"]["code"]
                for record in records
            )
            frontend_records = {
                json.loads(line)["id"]: json.loads(line)
                for line in (tmp_path / "data_frontend.jsonl").read_text(encoding="utf-8").splitlines()
            }
            assert all(
                record["provenance"]["source_frontend_record_id"] in frontend_records
                and record["provenance"]["mutation"]["path"] == "output.code"
                and record["provenance"]["mutation"]["count"] >= 1
                and record["provenance"]["mutation"]["sha256_before"]
                == sha256(
                    frontend_records[record["provenance"]["source_frontend_record_id"]]["output"]["code"].encode()
                ).hexdigest()
                and record["provenance"]["mutation"]["sha256_after"]
                == sha256(record["input"]["candidate_code"].encode()).hexdigest()
                for record in records
            )
        if task == "security":
            assert all("REVIEWED CODE:" in record["messages"][0]["content"] for record in records)
            assert all(
                record["input"]["reviewed_code"] in record["messages"][0]["content"]
                for record in records
            )
            assert [record["output"]["decision"] for record in records] == [
                "PASS", "BLOCK", "PASS", "BLOCK"
            ]
            assert all(record["provenance"]["security_fixture"]["active_payload_present"] is False for record in records)
            assert all(record["provenance"]["label_oracle"]["decision"] == record["output"]["decision"] for record in records)
            review_ids = {
                json.loads(line)["id"]
                for line in (tmp_path / "data_review.jsonl").read_text(encoding="utf-8").splitlines()
            }
            assert all(
                record["provenance"]["source_review_record_id"] in review_ids
                for record in records
            )
            assert all(
                record["messages"][0]["content"] != record["input"]["requirement"]
                for record in records
            )
            assert all(
                re.fullmatch(r"\[(?:BLOCK|PASS)\]", record["messages"][-1]["content"])
                for record in records
            )
            forbidden_active_forms = (
                "<script",
                "innerhtml",
                "eval(",
                "javascript:",
                "union select",
                "coinhive",
            )
            assert all(
                not any(
                    forbidden in record["input"]["reviewed_code"].casefold()
                    for forbidden in forbidden_active_forms
                )
                for record in records
            )

    second = _run(tmp_path)
    assert second.errors == ()
    assert second.written_by_task == {
        "plan": 0,
        "tool_policy": 0,
        "frontend": 0,
        "review": 0,
        "security": 0,
    }
    assert second.skipped_by_task == {
        "plan": 4,
        "tool_policy": 4,
        "frontend": 4,
        "review": 4,
        "security": 4,
    }


def test_seed_generation_deduplicates(tmp_path: Path) -> None:
    pipeline = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )
    seeds = asyncio.run(pipeline.generate_seeds(5))
    assert len(seeds) == 5
    assert len({seed.seed_id for seed in seeds}) == 5


def test_quarantined_seed_is_excluded_before_teacher_task_call(tmp_path: Path) -> None:
    class CountingTeacher(MockTeacher):
        def __init__(self) -> None:
            self.plan_calls = 0

        async def complete(self, *, system: str, user: str) -> str:
            if "ANCHOR_TASK: plan" in user:
                self.plan_calls += 1
            return await super().complete(system=system, user=user)

    teacher = CountingTeacher()
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=1,
    )
    seed = asyncio.run(pipeline.generate_seeds(1))[0]
    report = asyncio.run(
        pipeline.run(seed_count=1, tasks=["plan"], excluded_seed_ids=[seed.seed_id])
    )

    assert teacher.plan_calls == 0
    assert report.written_by_task["plan"] == 0
    assert report.skipped_by_task["plan"] == 1
    assert report.errors == ()


def test_tool_policy_oracle_covers_all_decisions_deterministically() -> None:
    seed = SeedDemand("seed-oracle", "title", "Build a local dashboard")
    results = []
    for index in range(3):
        proposals, _ = generate_inert_tool_proposals(seed, index)
        output, manifest = deterministic_tool_policy_oracle(proposals)
        results.append(output["decision"])
        assert manifest["decision"] == output["decision"]
    assert results == ["APPROVE", "ESCALATE", "BLOCK"]


def test_seed_prompt_uses_deterministic_stratified_variants() -> None:
    _, first = seed_prompt(0)
    _, second = seed_prompt(1)

    assert "SEED_VARIANT: 00" in first
    assert "SEED_VARIANT: 01" in second
    assert "REQUIRED_VARIATION_BRIEF" in first
    assert first != second


class _UnsafeSeedTeacher(MockTeacher):
    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: seed" in user:
            return '{"title":"unsafe","request":"render <script> directly","category":"unsafe","tags":[]}'
        return await super().complete(system=system, user=user)


def test_active_seed_material_is_never_persisted(tmp_path: Path) -> None:
    pipeline = DistillationPipeline(
        teacher=_UnsafeSeedTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=1,
    )
    with pytest.raises(RuntimeError, match="unique seeds"):
        asyncio.run(pipeline.generate_seeds(1))
    assert not (tmp_path / "seeds.jsonl").exists()


class _CountingMockTeacher(MockTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.task_calls: list[str] = []

    async def complete(self, *, system: str, user: str) -> str:
        marker = next(
            (line.split(":", 1)[1].strip() for line in user.splitlines() if line.startswith("ANCHOR_TASK:")),
            "",
        )
        self.task_calls.append(marker)
        return await super().complete(system=system, user=user)


def test_every_downstream_stage_requires_same_seed_upstream(tmp_path: Path) -> None:
    teacher = _CountingMockTeacher()
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=1,
    )
    review = asyncio.run(pipeline.run(seed_count=1, tasks=["review"]))
    assert any("UpstreamDependencyError" in error for error in review.errors)
    assert "review" not in teacher.task_calls
    assert not (tmp_path / "data_review.jsonl").exists()

    policy = asyncio.run(pipeline.run(seed_count=1, tasks=["tool_policy"]))
    assert any("UpstreamDependencyError" in error for error in policy.errors)
    assert "tool_policy" not in teacher.task_calls

    frontend = asyncio.run(pipeline.run(seed_count=1, tasks=["frontend"]))
    assert any("UpstreamDependencyError" in error for error in frontend.errors)
    assert "frontend" not in teacher.task_calls

    assert asyncio.run(pipeline.run(seed_count=1, tasks=["plan"])).errors == ()
    assert asyncio.run(pipeline.run(seed_count=1, tasks=["tool_policy"])).errors == ()
    assert asyncio.run(pipeline.run(seed_count=1, tasks=["frontend"])).errors == ()
    security = asyncio.run(pipeline.run(seed_count=1, tasks=["security"]))
    assert any("UpstreamDependencyError" in error for error in security.errors)
    assert "security" not in teacher.task_calls
    assert not (tmp_path / "data_security.jsonl").exists()


def test_mock_teacher_does_not_echo_pipeline_inputs() -> None:
    teacher = MockTeacher()
    review_raw = asyncio.run(
        teacher.complete(
            system="test",
            user=(
                "ANCHOR_TASK: review\nSEED_INDEX: 0\nCANDIDATE CODE:\n"
                "export function Page(){return <div>Ready</div>}\nEND CANDIDATE CODE\n"
                "KNOWN_BENIGN_DEFECT:\nA main landmark was replaced by a generic div.\n"
                "END KNOWN_BENIGN_DEFECT"
            ),
        )
    )
    security_raw = asyncio.run(
        teacher.complete(
            system="test",
            user=(
                "ANCHOR_TASK: security\nSEED_INDEX: 0\nREVIEWED CODE:\n"
                "export function Page(){return <main>Ready</main>}\nEND REVIEWED CODE"
            ),
        )
    )
    assert "input" not in json.loads(review_raw)
    assert "input" not in json.loads(security_raw)


def test_five_stage_resume_preserves_legacy_live_rows(tmp_path: Path) -> None:
    pipeline = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=1,
    )
    seed = asyncio.run(pipeline.generate_seeds(1))[0]
    legacy = {
        "frontend": ("legacy_frontend", "frontend_gen", {"code": "<main>legacy</main>"}),
        "review": ("legacy_review", "code_review", {"code": "<main>legacy reviewed</main>"}),
        "security": (
            "legacy_security",
            "security_audit",
            {"decision": "PASS", "rationale": "legacy validated row"},
        ),
    }
    before: dict[str, bytes] = {}
    for task, (record_id, expert, output) in legacy.items():
        path = tmp_path / f"data_{task}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": record_id,
                    "expert": expert,
                    "provenance": {"seed_id": seed.seed_id},
                    "output": output,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        before[task] = path.read_bytes()

    report = asyncio.run(pipeline.run(seed_count=1))

    assert report.errors == ()
    assert report.written_by_task == {
        "plan": 1,
        "tool_policy": 1,
        "frontend": 0,
        "review": 0,
        "security": 0,
    }
    assert report.skipped_by_task["frontend"] == 1
    assert report.skipped_by_task["review"] == 1
    assert report.skipped_by_task["security"] == 1
    assert all(
        (tmp_path / f"data_{task}.jsonl").read_bytes() == raw
        for task, raw in before.items()
    )


def test_frontend_prompt_requires_non_empty_public_trace() -> None:
    _, user = task_prompt(
        "frontend",
        SeedDemand("seed-1", "title", "Build an accessible catalog"),
        load_sop(ROOT / "skills" / "frontend.md"),
        0,
        task_input={
            "plan": {"summary": "catalog", "steps": [{"id": "P1", "goal": "build"}]},
            "tool_policy": {"decision": "APPROVE", "rationale": "bounded"},
        },
    )
    assert "decision_trace MUST contain 3 to 8 non-empty entries" in user
    assert '"code":"complete runnable implementation"' in user
    assert "never exceed 12,000" in user
    assert "1 to 3 small components" in user


def test_frontend_missing_teacher_trace_gets_attributed_contract_trace() -> None:
    payload: dict[str, object] = {
        "decision_trace": [],
        "output": {"language": "tsx", "code": "export const Card = () => <main>Ready</main>"},
    }
    source = _ensure_frontend_public_trace(payload)
    assert source == "pipeline_contract_fallback"
    assert len(payload["decision_trace"]) == 3  # type: ignore[arg-type]
    assert "reasoning" not in json.dumps(payload).casefold()


@pytest.mark.parametrize(
    ("payload", "source"),
    [
        ({"code": "export default function A(){}", "language": "tsx"}, "top_level_code"),
        ({"output": "export default function B(){}"}, "output_string"),
        ({"output": {"artifact": "export default function C(){}"}}, "output_artifact"),
    ],
)
def test_frontend_code_only_shapes_are_normalized(
    payload: dict[str, object], source: str
) -> None:
    assert _normalize_frontend_payload(payload) == source
    assert isinstance(payload["output"], dict)
    assert payload["output"]["code"].startswith("export")  # type: ignore[index,union-attr]
