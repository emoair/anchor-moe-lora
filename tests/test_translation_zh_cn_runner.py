from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from anchor_mvp.data import translation_zh_cn as runner
from anchor_mvp.data.translation_qa import SOURCE_FILES, prepare_translation_shards


class _PrefixTranslator:
    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def translate_batch(self, items: Mapping[str, str]) -> Mapping[str, str]:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return {key: f"中译：{value}" for key, value in items.items()}
        finally:
            self.active -= 1


class _FailingThenPrefixTranslator:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def translate_batch(self, items: Mapping[str, str]) -> Mapping[str, str]:
        self.calls += 1
        if self.calls <= self.failures:
            raise runner.TranslationRunError("provider returned invalid translation JSON")
        return {key: "\u4e2d\u8bd1\uff1a" + value for key, value in items.items()}


class _CoordinatedCacheTranslator:
    def __init__(self) -> None:
        self.shared_started = asyncio.Event()
        self.release_shared = asyncio.Event()
        self.slow_started = asyncio.Event()
        self.release_slow = asyncio.Event()

    async def translate_batch(self, items: Mapping[str, str]) -> Mapping[str, str]:
        values = set(items.values())
        if values == {"Shared contract text."}:
            self.shared_started.set()
            await self.release_shared.wait()
        else:
            assert values == {"Slow independent text."}
            self.slow_started.set()
            await self.release_slow.wait()
        return {key: "\u4e2d\u8bd1\uff1a" + value for key, value in items.items()}


class _MalformedThenPrefixTranslator:
    def __init__(self) -> None:
        self.requests: list[set[str]] = []

    async def translate_batch(self, items: Mapping[str, str]) -> Mapping[str, str]:
        self.requests.append(set(items.values()))
        translated = {
            key: "\u4e2d\u8bd1\uff1a" + value for key, value in items.items()
        }
        if len(self.requests) == 1:
            translated[sorted(translated)[-1]] = ""
        return translated


def _base(expert: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "id": f"record-{expert}::compact-v2",
        "expert": expert,
        "provenance": {
            "generator": "anchor_compact_v2_offline_projection",
            "source_id": f"source-{expert}",
        },
        "decision_trace": [
            {
                "check": "Check the bounded contract.",
                "evidence": "The source row is deterministic.",
                "action": "Retain the public semantics.",
            }
        ],
    }


def _record(expert: str) -> dict[str, Any]:
    code = "export const App = () => <main>Safe</main>;"
    record = _base(expert)
    if expert == "planner":
        output = {
            "summary": "Build a bounded component.",
            "constraints": ["Keep https://example.test/api unchanged."],
            "steps": [
                {"id": "P1", "goal": "Build the view.", "deliverable": "One file."}
            ],
        }
        record.update(
            {
                "input": {
                    "requirement": "Create <main> and run `npm test` before delivery."
                },
                "output": output,
                "messages": [
                    {"role": "user", "content": "old planner prompt"},
                    {
                        "role": "assistant",
                        "content": json.dumps(output, ensure_ascii=False, sort_keys=True),
                    },
                ],
            }
        )
    elif expert == "tool_policy":
        output = {
            "decision": "APPROVE",
            "rationale": "The local read is bounded.",
            "proposal_labels": ["INERT_READ_ONLY_WORKSPACE"],
        }
        record.update(
            {
                "input": {
                    "requirement": "Read one local file.",
                    "plan": "Inspect the workspace safely.",
                    "proposals": [
                        {
                            "id": "proposal_1",
                            "cap": "workspace.read_text",
                            "scope": "workspace-root",
                            "effect": "none",
                            "purpose": "inspect project conventions",
                        }
                    ],
                },
                "output": output,
                "messages": [
                    {"role": "user", "content": "old tool prompt"},
                    {"role": "assistant", "content": "APPROVE"},
                ],
            }
        )
    elif expert == "frontend_gen":
        record.update(
            {
                "input": {
                    "artifact_protocol": "single_file_tsx_segmented_v1",
                    "artifact_sha256": "a" * 64,
                    "segment_index": 0,
                    "segment_count": 1,
                    "requirement": "Render an accessible status view.",
                    "plan_summary": "Build one local component.",
                },
                "output": {"language": "tsx", "code": code},
                "compact_v2": {
                    "lossless_reconstruction": True,
                    "payload_sha256": "b" * 64,
                },
                "messages": [
                    {"role": "user", "content": "old frontend prompt"},
                    {"role": "assistant", "content": code},
                ],
            }
        )
    elif expert == "frontend_review":
        output = {
            "language": "tsx",
            "code": code,
            "summary": "The corrected component is accessible.",
        }
        record.update(
            {
                "input": {
                    "artifact_protocol": "single_file_tsx_segmented_v1",
                    "corrected_artifact_sha256_prefix": "c" * 16,
                    "segment_index": 0,
                    "segment_count": 1,
                    "requirement": "Review the status view.",
                    "known_benign_defect": "Restore the main landmark.",
                    "candidate_excerpt": code,
                },
                "output": output,
                "compact_v2": {
                    "review_protocol": "aligned_excerpt_to_corrected_segment_v1",
                    "payload_sha256": "d" * 64,
                },
                "messages": [
                    {"role": "user", "content": "old review prompt"},
                    {"role": "assistant", "content": code},
                ],
            }
        )
    elif expert == "security_gate":
        output = {
            "decision": "PASS",
            "findings": ["No unsafe flow was found."],
            "rationale": "All display content is inert.",
        }
        synopsis = "L1:export const App = () => <main>Safe</main>;"
        record.update(
            {
                "input": {
                    "requirement": "Audit the local component.",
                    "code_security_synopsis": synopsis,
                    "selection": "deterministic sources/sinks + boundaries; no oracle fields",
                },
                "output": output,
                "messages": [
                    {"role": "user", "content": "old security prompt"},
                    {"role": "assistant", "content": "[PASS]"},
                ],
            }
        )
    else:  # pragma: no cover
        raise AssertionError(expert)
    return record


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value)


def test_protector_round_trips_code_commands_urls_and_identifiers() -> None:
    source = (
        "Use https://example.test/api with <main>, `npm test`, textContent, React, "
        "HTML, [PASS], [Unit A placeholder], and --dry-run.\n"
        "python -m pytest tests/test_one.py"
    )
    protected = runner.protect_natural_text(source)

    assert "https://example.test/api" not in protected.provider_text
    assert "`npm test`" not in protected.provider_text
    assert "python -m pytest" not in protected.provider_text
    assert "React" not in protected.provider_text
    assert "[Unit A placeholder]" not in protected.provider_text
    restored = protected.restore("中译：" + protected.provider_text)
    assert restored == "中译：" + source


def test_compact_records_translate_only_natural_language_and_rebuild_messages() -> None:
    async def exercise() -> None:
        translator = _PrefixTranslator()
        cache = runner.TranslationCache(translator)
        for expert in (
            "planner",
            "tool_policy",
            "frontend_gen",
            "frontend_review",
            "security_gate",
        ):
            source = _record(expert)
            target = await runner.translate_compact_record(source, cache)

            assert target["id"] == source["id"] + "::zh-CN"
            assert target["provenance"] == source["provenance"]
            assert target.get("compact_v2") == source.get("compact_v2")
            assert _shape(target) == _shape(source)
            assert "中译" in target["input"]["requirement"]
            if "code" in source["output"]:
                assert target["output"]["code"] == source["output"]["code"]
                assert target["messages"][-1]["content"] == source["output"]["code"]
            if expert == "frontend_review":
                assert (
                    target["input"]["candidate_excerpt"]
                    == source["input"]["candidate_excerpt"]
                )
                assert target["messages"][0]["content"].startswith(
                    "REVIEW_TSX_SEGMENT|1/1|sha="
                )
            if expert == "security_gate":
                assert (
                    target["input"]["code_security_synopsis"]
                    == source["input"]["code_security_synopsis"]
                )
                assert target["messages"][-1]["content"] == "[PASS]"
            if expert == "planner":
                assert target["messages"][0]["content"].startswith(
                    "PLAN|artifact=single_file_tsx_segmented_v1"
                )
                assert json.loads(target["messages"][-1]["content"]) == target["output"]
            if expert == "tool_policy":
                assert target["output"]["decision"] == "APPROVE"
                assert target["output"]["proposal_labels"] == [
                    "INERT_READ_ONLY_WORKSPACE"
                ]
                assert target["input"]["proposals"][0]["cap"] == "workspace.read_text"

    asyncio.run(exercise())


def test_translation_cache_deduplicates_concurrent_exact_text() -> None:
    async def exercise() -> None:
        translator = _PrefixTranslator()
        cache = runner.TranslationCache(translator)
        first, second = await asyncio.gather(
            cache.resolve_many(["Build one view."]),
            cache.resolve_many(["Build one view."]),
        )
        assert first == second == {"Build one view.": "中译：Build one view."}
        assert translator.calls == 1

    asyncio.run(exercise())


def test_cached_boilerplate_does_not_serialize_independent_batches() -> None:
    async def exercise() -> None:
        translator = _PrefixTranslator()
        cache = runner.TranslationCache(translator)
        cache.seed("Shared contract text.", "中译：Shared contract text.")
        await asyncio.gather(
            cache.resolve_many(["Shared contract text.", "Build first view."]),
            cache.resolve_many(["Shared contract text.", "Build second view."]),
        )
        assert translator.calls == 2
        assert translator.max_active == 2

    asyncio.run(exercise())


def test_concurrent_cache_invalidation_cannot_remove_a_row_local_snapshot() -> None:
    async def exercise() -> None:
        translator = _CoordinatedCacheTranslator()
        cache = runner.TranslationCache(translator)
        first = asyncio.create_task(cache.resolve_many(["Shared contract text."]))
        await translator.shared_started.wait()
        second = asyncio.create_task(
            cache.resolve_many(
                ["Shared contract text.", "Slow independent text."]
            )
        )
        translator.release_shared.set()
        await translator.slow_started.wait()

        # The second resolver has snapshotted the shared value and released its
        # lock while awaiting the slow value. Invalidating the global entry must
        # not cause a KeyError or contaminate either row-local result.
        await cache.invalidate(
            {"Shared contract text.": "\u4e2d\u8bd1\uff1aShared contract text."}
        )
        translator.release_slow.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result["Shared contract text."].startswith("\u4e2d\u8bd1")
        assert second_result["Shared contract text."] == first_result[
            "Shared contract text."
        ]
        assert second_result["Slow independent text."].startswith("\u4e2d\u8bd1")

    asyncio.run(exercise())


def test_malformed_batch_does_not_partially_commit_translation_cache() -> None:
    async def exercise() -> None:
        values = ["First natural sentence.", "Second natural sentence."]
        translator = _MalformedThenPrefixTranslator()
        cache = runner.TranslationCache(translator)

        try:
            await cache.resolve_many(values)
        except runner.TranslationRunError as error:
            assert "empty translation" in str(error)
        else:  # pragma: no cover
            raise AssertionError("malformed batch was accepted")

        resolved = await cache.resolve_many(values)
        assert all(resolved[value].startswith("\u4e2d\u8bd1") for value in values)
        assert translator.requests == [set(values), set(values)]

    asyncio.run(exercise())


def test_conflicting_resume_seed_is_hash_reported_and_refreshed() -> None:
    async def exercise() -> None:
        source = "Repeat this natural sentence."
        first_target = "\u7b2c\u4e00\u79cd\u8bd1\u6587\u3002"
        second_target = "\u7b2c\u4e8c\u79cd\u8bd1\u6587\u3002"
        translator = _PrefixTranslator()
        cache = runner.TranslationCache(translator)
        cache.seed(source, first_target, origin_record_sha256="1" * 64)
        cache.seed(source, second_target, origin_record_sha256="2" * 64)

        conflicts = cache.seed_conflicts()
        serialized = json.dumps(conflicts, ensure_ascii=False, sort_keys=True)
        assert len(conflicts) == 1
        assert conflicts[0]["source_text_sha256"] == hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()
        assert source not in serialized
        assert first_target not in serialized
        assert second_target not in serialized
        assert "1" * 64 in serialized and "2" * 64 in serialized

        # No historical variant is selected silently. The next use refreshes
        # this exact text from the provider and can then be shared normally.
        resolved = await cache.resolve_many([source])
        assert translator.calls == 1
        assert resolved[source] not in {first_target, second_target}

    asyncio.run(exercise())


def test_resume_journal_ignores_only_a_torn_final_append(tmp_path: Path) -> None:
    good = {"source_path": "a", "source_line": 1}
    path = tmp_path / "part-000.jsonl"
    path.write_bytes(
        runner.canonical_json_bytes(good)
        + b"\n"
        + b'{"source_path":"torn"'
    )

    assert runner._read_jsonl_journal(path) == [good]


def test_cli_accepts_only_an_environment_variable_name_for_credentials() -> None:
    parser = runner.build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "api_key_env" in destinations
    assert "api_key" not in destinations


def test_background_launcher_prompts_securely_and_clears_process_secret() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "data"
        / "start_translation_zh_cn_background.ps1"
    ).read_text(encoding="utf-8")

    assert "[switch]$PromptForApiKey" in launcher
    assert "Read-Host" in launcher and "-AsSecureString" in launcher
    assert "SecureStringToBSTR" in launcher
    assert "ZeroFreeBSTR" in launcher
    assert "SetEnvironmentVariable($ApiKeyEnv, $credential, 'Process')" in launcher
    assert "SetEnvironmentVariable($ApiKeyEnv, $null, 'Process')" in launcher
    assert "'--api-key-env', $ApiKeyEnv" in launcher
    assert "ConvertFrom-SecureString" not in launcher


def test_runner_rejects_an_actual_heldout_path_before_inventory_read(
    tmp_path: Path,
) -> None:
    source = tmp_path / "heldout-cases"
    source.mkdir()
    registry = source / "manifest.registry-formal-v2.json"
    registry.write_text("not source data", encoding="utf-8")
    config = runner.PartRunConfig(
        source_dir=source,
        registry_path=registry,
        shard_dir=tmp_path / "shards",
        journal_dir=tmp_path / "journals",
        part_index=0,
    )

    try:
        config.validate_paths()
    except runner.TranslationRunError as error:
        assert "heldout/benchmark" in str(error)
    else:  # pragma: no cover
        raise AssertionError("heldout path was accepted")


def test_part_runner_checkpoints_publishes_and_resumes_without_network(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "candidate_dataset"
    source_dir.mkdir()
    registry_files = []
    for filename, expert in SOURCE_FILES:
        record = _record(expert)
        record["messages"] = runner.rebuild_compact_messages(record)
        content = runner.canonical_json_bytes(record) + b"\n"
        path = source_dir / filename
        path.write_bytes(content)
        registry_files.append(
            {
                "expert": expert,
                "path": f"artifacts/test/candidate_dataset/{filename}",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    snapshot = "e" * 64
    registry = source_dir / "manifest.registry-formal-v2.json"
    registry.write_text(
        json.dumps(
                {
                    "schema_version": "anchor.compact-mvp-v2b-dataset-snapshot.v1",
                    "snapshot_sha256": snapshot,
                    "artifact_protocol": "single_file_tsx_segmented_v1",
                    "source_manifest": {
                        "path": "artifacts/test/candidate_dataset/manifest.compact-v2.json",
                        "sha256": "f" * 64,
                    },
                    "files": registry_files,
                "heldout_content_read": False,
                "benchmark_record_content_read": False,
            }
        ),
        encoding="utf-8",
    )
    shard_dir = tmp_path / "translation" / "shards"
    prepare_translation_shards(
        source_dir=source_dir,
        registry_path=registry,
        shard_dir=shard_dir,
        expected_snapshot_sha256=snapshot,
    )
    config = runner.PartRunConfig(
        source_dir=source_dir,
        registry_path=registry,
        shard_dir=shard_dir,
        journal_dir=tmp_path / "translation" / "journals",
        part_index=0,
        concurrency=2,
        expected_snapshot_sha256=snapshot,
        progress_every=1,
    )

    first_translator = _PrefixTranslator()
    first = asyncio.run(runner.run_translation_part(config, first_translator))
    assert first["status"] == "complete"
    assert first["translated_this_run"] == 2
    assert config.journal_path.is_file()
    assert config.conflict_report_path.is_file()
    conflict_report = json.loads(config.conflict_report_path.read_text("utf-8"))
    assert conflict_report["conflict_count"] == 0
    assert conflict_report["contains_source_or_target_text"] is False
    envelopes = [
        json.loads(line)
        for line in config.part_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(envelopes) == 2
    assert all(set(item) == runner.ENVELOPE_FIELDS for item in envelopes)
    assert all(
        item["translated_record"]["id"] == item["source_id"] + "::zh-CN"
        for item in envelopes
    )

    second_translator = _PrefixTranslator()
    second = asyncio.run(runner.run_translation_part(config, second_translator))
    assert second["translated_this_run"] == 0
    assert second["resumed_records"] == 2
    assert second_translator.calls == 0


def _runner_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    source_dir = tmp_path / "candidate_dataset"
    source_dir.mkdir()
    registry_files = []
    for filename, expert in SOURCE_FILES:
        record = _record(expert)
        record["messages"] = runner.rebuild_compact_messages(record)
        content = runner.canonical_json_bytes(record) + b"\n"
        path = source_dir / filename
        path.write_bytes(content)
        registry_files.append(
            {
                "expert": expert,
                "path": f"artifacts/test/candidate_dataset/{filename}",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    snapshot = "e" * 64
    registry = source_dir / "manifest.registry-formal-v2.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "anchor.compact-mvp-v2b-dataset-snapshot.v1",
                "snapshot_sha256": snapshot,
                "artifact_protocol": "single_file_tsx_segmented_v1",
                "source_manifest": {
                    "path": "artifacts/test/candidate_dataset/manifest.compact-v2.json",
                    "sha256": "f" * 64,
                },
                "files": registry_files,
                "heldout_content_read": False,
                "benchmark_record_content_read": False,
            }
        ),
        encoding="utf-8",
    )
    shard_dir = tmp_path / "translation" / "shards"
    prepare_translation_shards(
        source_dir=source_dir,
        registry_path=registry,
        shard_dir=shard_dir,
        expected_snapshot_sha256=snapshot,
    )
    return source_dir, registry, shard_dir, snapshot


def test_row_rejection_does_not_abort_later_rows_and_resume_keeps_only_gold(
    tmp_path: Path,
) -> None:
    source_dir, registry, shard_dir, snapshot = _runner_fixture(tmp_path)
    config = runner.PartRunConfig(
        source_dir=source_dir,
        registry_path=registry,
        shard_dir=shard_dir,
        journal_dir=tmp_path / "translation" / "journals",
        part_index=0,
        concurrency=1,
        expected_snapshot_sha256=snapshot,
        progress_every=1,
        row_max_retries=0,
    )

    first_error = None
    try:
        asyncio.run(
            runner.run_translation_part(
                config, _FailingThenPrefixTranslator(failures=1)
            )
        )
    except runner.TranslationRunError as error:
        first_error = error
    assert first_error is not None
    assert "1 rejected row(s)" in str(first_error)

    good_rows = runner._read_jsonl_journal(config.journal_path)
    rejected_rows = runner._read_jsonl_journal(config.rejection_journal_path)
    assert len(good_rows) == 1
    assert len(rejected_rows) == 1
    assert "translated_record" in good_rows[0]
    assert "translated_record" not in rejected_rows[0]
    assert "prompt" not in rejected_rows[0]
    assert "response" not in rejected_rows[0]
    assert rejected_rows[0]["error_code"] == "invalid_translation_json"

    resumed = asyncio.run(
        runner.run_translation_part(config, _PrefixTranslator())
    )
    assert resumed["status"] == "complete"
    assert resumed["resumed_records"] == 1
    assert resumed["translated_this_run"] == 1
    assert resumed["rejected_this_run"] == 0
    published = [
        json.loads(line)
        for line in config.part_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(published) == 2
    assert all("translated_record" in row for row in published)


def test_row_semantic_retry_can_recover_without_writing_rejection(
    tmp_path: Path,
) -> None:
    source_dir, registry, shard_dir, snapshot = _runner_fixture(tmp_path)
    config = runner.PartRunConfig(
        source_dir=source_dir,
        registry_path=registry,
        shard_dir=shard_dir,
        journal_dir=tmp_path / "translation" / "journals",
        part_index=0,
        concurrency=1,
        expected_snapshot_sha256=snapshot,
        row_max_retries=1,
    )
    translator = _FailingThenPrefixTranslator(failures=1)

    report = asyncio.run(runner.run_translation_part(config, translator))

    assert report["status"] == "complete"
    assert report["row_retries_this_run"] == 1
    assert report["rejected_this_run"] == 0
    assert translator.calls == 3
    assert not config.rejection_journal_path.exists()


def test_strict_audit_retry_invalidates_cached_bad_translation(
    tmp_path: Path, monkeypatch
) -> None:
    source_dir, registry, shard_dir, snapshot = _runner_fixture(tmp_path)
    config = runner.PartRunConfig(
        source_dir=source_dir,
        registry_path=registry,
        shard_dir=shard_dir,
        journal_dir=tmp_path / "translation" / "journals",
        part_index=1,
        concurrency=1,
        expected_snapshot_sha256=snapshot,
        row_max_retries=1,
    )
    translator = _PrefixTranslator()
    original_audit = runner._audit_translated_record
    translated_audits = 0

    def fail_first_translated_audit(row, target) -> None:
        nonlocal translated_audits
        if translator.calls > 0 and str(target.get("id", "")).endswith("::zh-CN"):
            translated_audits += 1
            if translated_audits == 1:
                raise runner.TranslationAuditError(
                    "code/URL/protocol tokens changed"
                )
        original_audit(row, target)

    monkeypatch.setattr(runner, "_audit_translated_record", fail_first_translated_audit)

    report = asyncio.run(runner.run_translation_part(config, translator))

    assert report["status"] == "complete"
    assert report["row_retries_this_run"] == 1
    assert translator.calls == 2
    assert translated_audits == 2
