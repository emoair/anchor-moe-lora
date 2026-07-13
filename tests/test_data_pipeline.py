from __future__ import annotations

import asyncio
from contextvars import ContextVar
from hashlib import sha256
import json
import re
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anchor_mvp.data.teacher as teacher_module  # noqa: E402
from anchor_mvp.data.pipeline import (  # noqa: E402
    DistillationPipeline,
    _ensure_frontend_public_trace,
    _normalize_frontend_payload,
)
from anchor_mvp.data.cleaning import validate_safe_payload  # noqa: E402
from anchor_mvp.data.proposals import (  # noqa: E402
    deterministic_tool_policy_oracle,
    generate_inert_tool_proposals,
)
from anchor_mvp.data.prompts import task_prompt  # noqa: E402
from anchor_mvp.data.prompts import seed_prompt  # noqa: E402
from anchor_mvp.data.schema import SeedDemand  # noqa: E402
from anchor_mvp.data.sops import load_sop  # noqa: E402
from anchor_mvp.data.teacher import (  # noqa: E402
    CompatibleTeacher,
    MockTeacher,
    ProviderQuotaExhausted,
    RateLimitError,
)
from anchor_mvp.training.schema import validate_jsonl  # noqa: E402


def _run(output: Path):
    pipeline = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=output,
        concurrency=3,
    )
    return asyncio.run(pipeline.run(seed_count=4))


def test_credential_like_teacher_payload_is_a_hard_reject() -> None:
    with pytest.raises(ValueError, match="credential-like"):
        validate_safe_payload(
            "plan",
            {"output": {"summary": "api_key=sk-do-not-write-this-value-123456"}},
        )


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
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert all(record["provenance"]["template_sha256"] for record in records)
        assert all(
            record["provenance"]["teacher"]["protocol"] == "mock" for record in records
        )
        assert all(
            record["provenance"]["teacher"]["generation_params"]["thinking_enabled"]
            is False
            for record in records
        )
        if task == "plan":
            assert all(record["output"]["steps"] for record in records)
        if task == "tool_policy":
            plan_ids = {
                json.loads(line)["id"]
                for line in (tmp_path / "data_plan.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            }
            assert all(
                record["messages"][-1]["content"] in {"APPROVE", "BLOCK", "ESCALATE"}
                for record in records
            )
            assert all(
                record["provenance"]["source_plan_record_id"] in plan_ids
                for record in records
            )
            assert all(
                record["provenance"]["tool_proposals"]["executed"] is False
                for record in records
            )
            assert [record["output"]["decision"] for record in records] == [
                "APPROVE",
                "ESCALATE",
                "BLOCK",
                "APPROVE",
            ]
            assert all(
                record["provenance"]["label_oracle"]["decision"]
                == record["output"]["decision"]
                for record in records
            )
            assert all(
                record["provenance"]["teacher_observed_decision"]
                in {"APPROVE", "BLOCK", "ESCALATE"}
                for record in records
            )
            assert all(
                not any(
                    str(value).startswith(("http://", "https://"))
                    for proposal in record["input"]["tool_proposals"]
                    for value in proposal.values()
                )
                for record in records
            )
        if task == "frontend":
            assert all(
                record["provenance"]["source_plan_record_id"] for record in records
            )
            assert all(
                record["provenance"]["source_tool_policy_record_id"]
                for record in records
            )
            assert all(record["input"]["plan"]["steps"] for record in records)
            assert all(
                record["input"]["tool_policy"]["decision"]
                in {"APPROVE", "BLOCK", "ESCALATE"}
                for record in records
            )
        if task == "review":
            assert all(
                "CANDIDATE CODE:" in record["messages"][0]["content"]
                for record in records
            )
            assert all(
                "KNOWN_BENIGN_DEFECT:" in record["messages"][0]["content"]
                for record in records
            )
            assert all(
                record["input"]["candidate_code"] in record["messages"][0]["content"]
                and record["input"]["known_benign_defect"]
                in record["messages"][0]["content"]
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
                for line in (tmp_path / "data_frontend.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            }
            assert all(
                record["provenance"]["source_frontend_record_id"] in frontend_records
                and record["provenance"]["mutation"]["path"] == "output.code"
                and record["provenance"]["mutation"]["count"] >= 1
                and record["provenance"]["mutation"]["sha256_before"]
                == sha256(
                    frontend_records[record["provenance"]["source_frontend_record_id"]][
                        "output"
                    ]["code"].encode()
                ).hexdigest()
                and record["provenance"]["mutation"]["sha256_after"]
                == sha256(record["input"]["candidate_code"].encode()).hexdigest()
                for record in records
            )
        if task == "security":
            assert all(
                "REVIEWED CODE:" in record["messages"][0]["content"]
                for record in records
            )
            assert all(
                record["input"]["reviewed_code"] in record["messages"][0]["content"]
                for record in records
            )
            assert [record["output"]["decision"] for record in records] == [
                "PASS",
                "BLOCK",
                "PASS",
                "BLOCK",
            ]
            assert all(
                record["provenance"]["security_fixture"]["active_payload_present"]
                is False
                for record in records
            )
            assert all(
                record["provenance"]["label_oracle"]["decision"]
                == record["output"]["decision"]
                for record in records
            )
            assert all(
                record["provenance"]["teacher_observed_decision"] in {"PASS", "BLOCK"}
                for record in records
            )
            review_ids = {
                json.loads(line)["id"]
                for line in (tmp_path / "data_review.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
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


def test_request_local_retries_keep_one_seed_alignment_and_write_one_record(
    tmp_path: Path, monkeypatch
) -> None:
    seed = SeedDemand(
        "seed-retry-1",
        "Retry fixture",
        "Build an accessible catalog with stable empty states.",
    )
    (tmp_path / "seeds.jsonl").write_text(
        json.dumps(seed.to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    teacher = CompatibleTeacher(max_retries=2, fallback_protocol=None)
    calls: list[tuple[str, str]] = []

    def fake_request(protocol, base_url, system, user, max_tokens):
        calls.append((system, user))
        if len(calls) <= 2:
            raise URLError("transient fixture interruption")
        return teacher_module._CompletionText(
            json.dumps(
                {
                    "decision_trace": [
                        {
                            "check": "Requirement decomposition",
                            "evidence": "The public request defines the required UI states.",
                            "action": "Produce one bounded implementation plan.",
                        }
                    ],
                    "output": {
                        "summary": "Build the requested accessible catalog.",
                        "steps": [
                            {
                                "id": "P1",
                                "goal": "Define semantic structure",
                                "deliverable": "Catalog landmarks and empty states",
                            }
                        ],
                        "constraints": ["Keep behavior deterministic"],
                    },
                }
            )
        )

    monkeypatch.setattr(teacher, "_request_sync", fake_request)
    monkeypatch.setattr(teacher_module, "_retry_delay_seconds", lambda *args: 0.0)
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=1,
    )

    first = asyncio.run(pipeline.run(seed_count=1, tasks=["plan"]))
    assert first.errors == ()
    assert first.written_by_task == {"plan": 1}
    assert len(calls) == 3
    assert len(set(calls)) == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "data_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 1
    assert records[0]["provenance"]["seed_id"] == seed.seed_id
    assert records[0]["provenance"]["teacher"]["provider"]["attempts"] == {
        "wire_attempts": 3,
        "retry_count": 2,
        "max_retries": 2,
        "retry_reasons": ["url_error", "url_error"],
    }

    resumed = asyncio.run(pipeline.run(seed_count=1, tasks=["plan"]))
    assert resumed.written_by_task == {"plan": 0}
    assert resumed.skipped_by_task == {"plan": 1}
    assert len(calls) == 3
    assert (
        len((tmp_path / "data_plan.jsonl").read_text(encoding="utf-8").splitlines())
        == 1
    )


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
    def __init__(self) -> None:
        super().__init__()
        self.seed_calls = 0

    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: seed" in user:
            self.seed_calls += 1
            return '{"title":"unsafe","request":"render <script> directly","category":"unsafe","tags":[]}'
        return await super().complete(system=system, user=user)


def test_active_seed_material_is_never_persisted(tmp_path: Path) -> None:
    teacher = _UnsafeSeedTeacher()
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=1,
    )
    with pytest.raises(ValueError, match="active payload"):
        asyncio.run(pipeline.generate_seeds(1))
    assert teacher.seed_calls == 1
    assert not (tmp_path / "seeds.jsonl").exists()


class _CountingMockTeacher(MockTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.task_calls: list[str] = []

    async def complete(self, *, system: str, user: str) -> str:
        marker = next(
            (
                line.split(":", 1)[1].strip()
                for line in user.splitlines()
                if line.startswith("ANCHOR_TASK:")
            ),
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
        "frontend": (
            "legacy_frontend",
            "frontend_gen",
            {"code": "<main>legacy</main>"},
        ),
        "review": (
            "legacy_review",
            "code_review",
            {"code": "<main>legacy reviewed</main>"},
        ),
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
        "output": {
            "language": "tsx",
            "code": "export const Card = () => <main>Ready</main>",
        },
    }
    source = _ensure_frontend_public_trace(payload)
    assert source == "pipeline_contract_fallback"
    assert len(payload["decision_trace"]) == 3  # type: ignore[arg-type]
    assert "reasoning" not in json.dumps(payload).casefold()


@pytest.mark.parametrize(
    ("payload", "source"),
    [
        (
            {"code": "export default function A(){}", "language": "tsx"},
            "top_level_code",
        ),
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


def _prompt_index(user: str) -> int:
    return int(
        next(
            line.split(":", 1)[1].strip()
            for line in user.splitlines()
            if line.startswith("SEED_INDEX:")
        )
    )


class _PartialSeedTerminalTeacher(MockTeacher):
    def __init__(self, terminal: RateLimitError) -> None:
        super().__init__()
        self.terminal = terminal
        self.first_success_ready = asyncio.Event()

    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: seed" not in user:
            return await super().complete(system=system, user=user)
        index = _prompt_index(user)
        if index == 0:
            result = await super().complete(system=system, user=user)
            self.first_success_ready.set()
            return result
        if index == 1:
            await self.first_success_ready.wait()
            raise self.terminal
        await self.first_success_ready.wait()
        await asyncio.sleep(0.02)
        raise ValueError("non-terminal fixture failure")


@pytest.mark.parametrize(
    ("terminal", "expected_type", "retry_after"),
    [
        (RateLimitError(7), RateLimitError, 7),
        (ProviderQuotaExhausted(11), ProviderQuotaExhausted, 11),
    ],
)
def test_seed_generation_persists_completed_rows_before_terminal_and_resumes(
    tmp_path: Path,
    terminal: RateLimitError,
    expected_type: type[RateLimitError],
    retry_after: float,
) -> None:
    interrupted = DistillationPipeline(
        teacher=_PartialSeedTerminalTeacher(terminal),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=3,
    )

    with pytest.raises(expected_type) as captured:
        asyncio.run(interrupted.generate_seeds(3))

    assert captured.value.retry_after_seconds == retry_after
    partial = [
        json.loads(line)
        for line in (tmp_path / "seeds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(partial) == 1
    assert "variant 0" in partial[0]["request"]

    resumed = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )
    seeds = asyncio.run(resumed.generate_seeds(3))
    assert len(seeds) == 3
    assert len({seed.seed_id for seed in seeds}) == 3
    assert seeds[0].seed_id == partial[0]["seed_id"]
    assert len((tmp_path / "seeds.jsonl").read_text(encoding="utf-8").splitlines()) == 3


class _IncrementalPlanTeacher(MockTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.two_started = asyncio.Event()
        self.first_returned = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: plan" not in user:
            return await super().complete(system=system, user=user)
        index = _prompt_index(user)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active >= 2:
            self.two_started.set()
        try:
            if index == 0:
                await self.two_started.wait()
                result = await super().complete(system=system, user=user)
                self.first_returned.set()
                return result
            await self.release.wait()
            return await super().complete(system=system, user=user)
        finally:
            self.active -= 1


def test_task_distillation_appends_each_completion_and_honors_concurrency(
    tmp_path: Path,
) -> None:
    seed_pipeline = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=3,
    )
    asyncio.run(seed_pipeline.generate_seeds(3))
    teacher = _IncrementalPlanTeacher()
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )

    async def exercise() -> None:
        running = asyncio.create_task(pipeline.run(seed_count=3, tasks=["plan"]))
        await asyncio.wait_for(teacher.first_returned.wait(), timeout=1)
        path = tmp_path / "data_plan.jsonl"
        for _ in range(100):
            if (
                path.exists()
                and len(path.read_text(encoding="utf-8").splitlines()) == 1
            ):
                break
            await asyncio.sleep(0.001)
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1
        assert not running.done()
        assert teacher.max_active == 2
        teacher.release.set()
        report = await asyncio.wait_for(running, timeout=1)
        assert report.written_by_task == {"plan": 3}

    asyncio.run(exercise())
    assert (
        len((tmp_path / "data_plan.jsonl").read_text(encoding="utf-8").splitlines())
        == 3
    )


class _PartialPlanQuotaTeacher(MockTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.first_success_ready = asyncio.Event()
        self.frontend_calls = 0

    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: frontend" in user:
            self.frontend_calls += 1
        if "ANCHOR_TASK: plan" not in user:
            return await super().complete(system=system, user=user)
        index = _prompt_index(user)
        if index == 0:
            result = await super().complete(system=system, user=user)
            self.first_success_ready.set()
            return result
        if index == 1:
            await self.first_success_ready.wait()
            raise ProviderQuotaExhausted(13)
        await self.first_success_ready.wait()
        await asyncio.sleep(0.02)
        raise ValueError("non-terminal fixture failure")


def test_task_quota_keeps_partial_record_reports_deterministically_and_resumes(
    tmp_path: Path,
) -> None:
    initial = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=3,
    )
    seeds = asyncio.run(initial.generate_seeds(3))
    teacher = _PartialPlanQuotaTeacher()
    interrupted = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=3,
    )

    report = asyncio.run(interrupted.run(seed_count=3, tasks=["plan", "frontend"]))

    assert report.written_by_task == {"plan": 1, "frontend": 0}
    assert report.skipped_by_task == {"plan": 0, "frontend": 0}
    assert teacher.frontend_calls == 0
    assert report.rate_limited is True
    assert report.provider_quota_exhausted is True
    assert report.retry_after_seconds == 13
    assert report.errors == (
        f"plan:{seeds[1].seed_id}: ProviderQuotaExhausted: "
        "provider explicitly reported quota exhausted",
        f"plan:{seeds[2].seed_id}: ValueError: non-terminal fixture failure",
    )
    assert (
        len((tmp_path / "data_plan.jsonl").read_text(encoding="utf-8").splitlines())
        == 1
    )

    resumed = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )
    resumed_report = asyncio.run(resumed.run(seed_count=3, tasks=["plan"]))
    assert resumed_report.errors == ()
    assert resumed_report.written_by_task == {"plan": 2}
    assert resumed_report.skipped_by_task == {"plan": 1}
    assert (
        len((tmp_path / "data_plan.jsonl").read_text(encoding="utf-8").splitlines())
        == 3
    )


class _OutOfOrderFailureTeacher(MockTeacher):
    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: plan" not in user:
            return await super().complete(system=system, user=user)
        index = _prompt_index(user)
        await asyncio.sleep(0.01 if index == 0 else 0)
        raise ValueError(f"failure-{index}")


def test_task_error_report_order_is_seed_deterministic(tmp_path: Path) -> None:
    initial = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )
    seeds = asyncio.run(initial.generate_seeds(2))
    pipeline = DistillationPipeline(
        teacher=_OutOfOrderFailureTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )

    report = asyncio.run(pipeline.run(seed_count=2, tasks=["plan"]))

    assert report.errors == (
        f"plan:{seeds[0].seed_id}: ValueError: failure-0",
        f"plan:{seeds[1].seed_id}: ValueError: failure-1",
    )


class _InFlightSeedQuotaTeacher(MockTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.inflight_started = threading.Event()
        self.release_inflight = threading.Event()
        self.inflight_returned = threading.Event()
        self.started_indices: list[int] = []

    def _slow_seed(self) -> str:
        self.inflight_started.set()
        if not self.release_inflight.wait(timeout=1):
            raise AssertionError("quota branch did not release in-flight request")
        time.sleep(0.02)
        self.inflight_returned.set()
        return json.dumps(
            {
                "title": "Durable in-flight seed",
                "request": "Build an accessible local task board with clear empty states.",
                "category": "standard",
                "tags": ["task-board"],
            }
        )

    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: seed" not in user:
            return await super().complete(system=system, user=user)
        index = _prompt_index(user)
        self.started_indices.append(index)
        if index == 0:
            return await asyncio.to_thread(self._slow_seed)
        while not self.inflight_started.is_set():
            await asyncio.sleep(0)
        self.release_inflight.set()
        raise ProviderQuotaExhausted(17)


def test_seed_quota_waits_for_uncancellable_to_thread_and_persists_success(
    tmp_path: Path,
) -> None:
    teacher = _InFlightSeedQuotaTeacher()
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )

    with pytest.raises(ProviderQuotaExhausted):
        asyncio.run(pipeline.generate_seeds(5))

    assert teacher.inflight_returned.is_set()
    assert sorted(teacher.started_indices) == [0, 1]
    persisted = (tmp_path / "seeds.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(persisted) == 1
    assert json.loads(persisted[0])["title"] == "Durable in-flight seed"


class _InFlightPlanQuotaTeacher(MockTeacher):
    def __init__(self) -> None:
        super().__init__()
        self.inflight_started = threading.Event()
        self.release_inflight = threading.Event()

    def _slow_wire(self) -> None:
        self.inflight_started.set()
        if not self.release_inflight.wait(timeout=1):
            raise AssertionError("quota branch did not release in-flight request")
        time.sleep(0.02)

    async def complete(self, *, system: str, user: str) -> str:
        if "ANCHOR_TASK: plan" not in user:
            return await super().complete(system=system, user=user)
        index = _prompt_index(user)
        if index == 0:
            await asyncio.to_thread(self._slow_wire)
            return await super().complete(system=system, user=user)
        while not self.inflight_started.is_set():
            await asyncio.sleep(0)
        self.release_inflight.set()
        raise ProviderQuotaExhausted(19)


def test_task_quota_waits_for_uncancellable_to_thread_and_persists_success(
    tmp_path: Path,
) -> None:
    initial = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )
    asyncio.run(initial.generate_seeds(2))
    pipeline = DistillationPipeline(
        teacher=_InFlightPlanQuotaTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )

    report = asyncio.run(pipeline.run(seed_count=2, tasks=["plan"]))

    assert report.provider_quota_exhausted is True
    assert report.written_by_task == {"plan": 1}
    assert (
        len((tmp_path / "data_plan.jsonl").read_text(encoding="utf-8").splitlines())
        == 1
    )


def test_seed_index_offset_makes_parallel_shard_prompts_disjoint(
    tmp_path: Path,
) -> None:
    class CapturingTeacher(MockTeacher):
        def __init__(self) -> None:
            super().__init__()
            self.seed_users: list[str] = []

        async def complete(self, *, system: str, user: str) -> str:
            if "ANCHOR_TASK: seed" in user:
                self.seed_users.append(user)
            return await super().complete(system=system, user=user)

    teacher = CapturingTeacher()
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path / "shard",
        concurrency=2,
        seed_index_offset=100_000,
    )

    assert len(asyncio.run(pipeline.generate_seeds(2))) == 2
    assert sorted(_prompt_index(prompt) for prompt in teacher.seed_users) == [
        100_000,
        100_001,
    ]


def test_seed_index_offset_cannot_be_negative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed_index_offset"):
        DistillationPipeline(
            teacher=MockTeacher(),
            sop_dir=ROOT / "skills",
            output_dir=tmp_path,
            seed_index_offset=-1,
        )


class _BarrierRouteTeacher(MockTeacher):
    def __init__(self) -> None:
        super().__init__()
        self._route_context: ContextVar[dict[str, str] | None] = ContextVar(
            "test_pipeline_route", default=None
        )
        self.first_route_ready = asyncio.Event()
        self.second_route_ready = asyncio.Event()

    @property
    def provider_provenance(self) -> dict[str, object]:
        route = self._route_context.get()
        if route is None:
            return super().provider_provenance
        return {
            "preset": "barrier-fixture",
            "base_url": route["base_url"],
            "protocol": route["protocol"],
            "model": self.model,
            "model_source": "fixture",
            "request_marker": route["request_marker"],
            "discovery": {"status": "skipped", "model_count": 0},
        }

    async def complete(self, *, system: str, user: str) -> str:
        result = await super().complete(system=system, user=user)
        if "ANCHOR_TASK: plan" not in user:
            return result
        index = _prompt_index(user)
        route = {
            "base_url": f"https://route-{index}.example/v1",
            "protocol": "openai" if index == 0 else "openai_responses",
            "request_marker": f"route-{index}",
        }
        self._route_context.set(route)
        if index == 0:
            self.first_route_ready.set()
            await self.second_route_ready.wait()
        else:
            await self.first_route_ready.wait()
            # Deliberately mutate the shared defaults before request zero is
            # allowed to return. Its ContextVar must remain route zero.
            self.base_url = route["base_url"]
            self.protocol = route["protocol"]
            self.second_route_ready.set()
        return result


def test_record_route_uses_one_request_local_snapshot_under_barrier(
    tmp_path: Path,
) -> None:
    seed_pipeline = DistillationPipeline(
        teacher=MockTeacher(),
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )
    asyncio.run(seed_pipeline.generate_seeds(2))
    teacher = _BarrierRouteTeacher()
    pipeline = DistillationPipeline(
        teacher=teacher,
        sop_dir=ROOT / "skills",
        output_dir=tmp_path,
        concurrency=2,
    )

    report = asyncio.run(pipeline.run(seed_count=2, tasks=["plan"]))

    assert report.errors == ()
    records = [
        json.loads(line)
        for line in (tmp_path / "data_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    observed = set()
    for record in records:
        top = record["provenance"]["teacher"]
        provider = top["provider"]
        assert top["base_url"] == provider["base_url"]
        assert top["protocol"] == provider["protocol"]
        observed.add(
            (
                provider["request_marker"],
                provider["base_url"],
                provider["protocol"],
            )
        )
    assert observed == {
        ("route-0", "https://route-0.example/v1", "openai"),
        (
            "route-1",
            "https://route-1.example/v1",
            "openai_responses",
        ),
    }
