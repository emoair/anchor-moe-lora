from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from anchor_mvp.data.translation_qa import (
    COMPACT_ARTIFACT_PROTOCOL,
    REGISTRY_SCHEMA,
    SHARD_NAMES,
    SOURCE_FILES,
    TARGET_ID_SUFFIX,
    TranslationAuditError,
    canonical_json_bytes,
    merge_translation_shards,
    prepare_translation_shards,
)


def _compact_user(record: dict[str, Any]) -> str:
    inputs = record["input"]
    expert = record["expert"]
    compact = json.dumps(
        inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if expert == "planner":
        return (
            f"PLAN|artifact={COMPACT_ARTIFACT_PROTOCOL}\n"
            f"requirement={inputs['requirement']}"
        )
    if expert == "tool_policy":
        return "TOOL_POLICY|" + compact
    if expert == "frontend_gen":
        return "GENERATE_TSX_SEGMENT|" + compact
    if expert == "frontend_review":
        return (
            f"REVIEW_TSX_SEGMENT|{inputs['segment_index'] + 1}/"
            f"{inputs['segment_count']}|"
            f"sha={inputs['corrected_artifact_sha256_prefix']}\n"
            f"REQ:{inputs['requirement']}\n"
            f"DEFECT:{inputs['known_benign_defect']}\n"
            f"CANDIDATE:\n{inputs['candidate_excerpt']}"
        )
    if expert == "security_gate":
        return "SECURITY_GATE|" + compact
    raise AssertionError(expert)


def _record(expert: str) -> dict[str, Any]:
    code = "export const App = () => <main>Safe</main>;"
    common = {
        "schema_version": "1.0",
        "id": f"source-{expert}",
        "expert": expert,
        "provenance": {
            "generator": "fixture",
            "source_id": f"lineage-{expert}",
        },
        "decision_trace": [
            {
                "check": "Check the bounded contract.",
                "evidence": "The fixture is deterministic.",
                "action": "Accept the record.",
            }
        ],
    }
    if expert == "planner":
        protected = (
            "Use https://example.test/api and keep [PASS], <h1>, `npm test`, "
            "and this block:\n```js\nconst x = 1;\n```"
        )
        output = {
            "summary": "Build a bounded component.",
            "constraints": ["Keep the implementation local."],
            "steps": [
                {"id": "P1", "goal": "Build the view.", "deliverable": "One file."}
            ],
        }
        record = {
            **common,
            "input": {"requirement": protected},
            "output": output,
            "messages": [
                {"role": "user", "content": "placeholder"},
                {
                    "role": "assistant",
                    "content": json.dumps(output, ensure_ascii=False, sort_keys=True),
                },
            ],
        }
        record["messages"][0]["content"] = _compact_user(record)
        return record
    if expert == "tool_policy":
        output = {
            "decision": "APPROVE",
            "rationale": "The local read is bounded.",
            "proposal_labels": ["APPROVE"],
        }
        record = {
            **common,
            "input": {"requirement": "Read one local file."},
            "output": output,
            "messages": [
                {"role": "user", "content": "placeholder"},
                {"role": "assistant", "content": "APPROVE"},
            ],
        }
        record["messages"][0]["content"] = _compact_user(record)
        return record
    if expert == "frontend_gen":
        record = {
            **common,
            "input": {
                "artifact_protocol": "single_file_tsx_segmented_v1",
                "segment_index": 0,
                "segment_count": 1,
                "requirement": "Render a safe status view.",
            },
            "output": {"language": "tsx", "code": code},
            "compact_v2": {
                "lossless_reconstruction": True,
                "payload_sha256": "1" * 64,
            },
            "messages": [
                {"role": "user", "content": "placeholder"},
                {"role": "assistant", "content": code},
            ],
        }
        record["messages"][0]["content"] = _compact_user(record)
        return record
    if expert == "frontend_review":
        output = {
            "language": "tsx",
            "code": code,
            "summary": "The corrected component is accessible.",
        }
        record = {
            **common,
            "input": {
                "artifact_protocol": "single_file_tsx_segmented_v1",
                "corrected_artifact_sha256_prefix": "3" * 12,
                "segment_index": 0,
                "segment_count": 1,
                "requirement": "Review the status view.",
                "known_benign_defect": "Restore the semantic heading.",
                "candidate_excerpt": code,
            },
            "output": output,
            "compact_v2": {
                "review_protocol": "aligned_excerpt_to_corrected_segment_v1",
                "payload_sha256": "2" * 64,
            },
            "messages": [
                {"role": "user", "content": "placeholder"},
                {"role": "assistant", "content": code},
            ],
        }
        record["messages"][0]["content"] = _compact_user(record)
        return record
    if expert == "security_gate":
        record = {
            **common,
            "input": {
                "requirement": "Audit the local component.",
                "code_security_synopsis": "FILE App.tsx\nSINK textContent",
            },
            "output": {
                "decision": "PASS",
                "findings": ["No unsafe flow was found."],
                "rationale": "All content is inert.",
            },
            "messages": [
                {"role": "user", "content": "placeholder"},
                {"role": "assistant", "content": "[PASS]"},
            ],
        }
        record["messages"][0]["content"] = _compact_user(record)
        return record
    raise AssertionError(expert)


def _translated(source: dict[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(source)
    target["id"] = source["id"] + TARGET_ID_SUFFIX
    target["decision_trace"] = [
        {
            "check": "检查受限契约。",
            "evidence": "该夹具是确定性的。",
            "action": "接受此记录。",
        }
    ]
    expert = source["expert"]
    if expert == "planner":
        protected = (
            "使用 https://example.test/api，并保留 [PASS]、<h1>、`npm test` "
            "以及此代码块：\n```js\nconst x = 1;\n```"
        )
        target["input"]["requirement"] = protected
        target["output"] = {
            "summary": "构建一个边界明确的组件。",
            "constraints": ["实现必须保持在本地。"],
            "steps": [
                {"id": "P1", "goal": "构建视图。", "deliverable": "一个文件。"}
            ],
        }
        target["messages"][0]["content"] = protected
        target["messages"][-1]["content"] = json.dumps(
            target["output"], ensure_ascii=False, sort_keys=True
        )
    elif expert == "tool_policy":
        target["input"]["requirement"] = "读取一个本地文件。"
        target["output"]["rationale"] = "此本地读取操作边界明确。"
        target["messages"][0]["content"] = "读取一个本地文件。"
    elif expert == "frontend_gen":
        target["input"]["requirement"] = "渲染一个安全的状态视图。"
        target["messages"][0]["content"] = "渲染一个安全的状态视图。"
    elif expert == "frontend_review":
        target["input"]["requirement"] = "审查该状态视图。"
        target["input"]["known_benign_defect"] = "恢复语义化标题。"
        target["output"]["summary"] = "修正后的组件具备可访问性。"
        target["messages"][0]["content"] = "审查该状态视图。"
    elif expert == "security_gate":
        target["input"]["requirement"] = "审计该本地组件。"
        target["output"]["findings"] = ["未发现不安全的数据流。"]
        target["output"]["rationale"] = "所有内容均为惰性数据。"
        target["messages"][0]["content"] = "审计该本地组件。"
    else:  # pragma: no cover
        raise AssertionError(expert)
    target["messages"][0]["content"] = _compact_user(target)
    return target


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(value) + b"\n" for value in values))


def _source_fixture(tmp_path: Path) -> tuple[Path, Path, str, dict[str, dict]]:
    source_dir = tmp_path / "candidate_dataset"
    source_dir.mkdir()
    records: dict[str, dict] = {}
    manifest_files = []
    for filename, expert in SOURCE_FILES:
        record = _record(expert)
        records[record["id"]] = record
        path = source_dir / filename
        _write_jsonl(path, [record])
        manifest_files.append(
            {
                "expert": expert,
                "path": f"artifacts/test/candidate_dataset/{filename}",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    snapshot = "a" * 64
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "snapshot_sha256": snapshot,
        "artifact_protocol": COMPACT_ARTIFACT_PROTOCOL,
        "source_manifest": {
            "path": "artifacts/test/candidate_dataset/manifest.compact-v2.json",
            "sha256": "b" * 64,
        },
        "files": manifest_files,
        "heldout_content_read": False,
        "benchmark_record_content_read": False,
    }
    registry_path = source_dir / "manifest.registry-formal-v2.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return source_dir, registry_path, snapshot, records


def _ready_shards(
    tmp_path: Path, source_dir: Path, registry_path: Path, snapshot: str
) -> Path:
    shard_dir = tmp_path / "shards"
    prepare_translation_shards(
        source_dir=source_dir,
        registry_path=registry_path,
        shard_dir=shard_dir,
        expected_snapshot_sha256=snapshot,
    )
    for name in SHARD_NAMES:
        path = shard_dir / name
        envelopes = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        for envelope in envelopes:
            envelope["translated_record"] = _translated(
                envelope["translated_record"]
                | {"id": envelope["source_id"]}
            )
        _write_jsonl(path, envelopes)
    return shard_dir


def _merge(
    source_dir: Path,
    registry_path: Path,
    snapshot: str,
    shard_dir: Path,
    output_dir: Path,
):
    return merge_translation_shards(
        source_dir=source_dir,
        registry_path=registry_path,
        shard_paths=[shard_dir / name for name in SHARD_NAMES],
        output_dir=output_dir,
        expected_snapshot_sha256=snapshot,
    )


def _mutate_translated_record(shard_dir: Path, expert: str, mutate) -> None:
    for name in SHARD_NAMES:
        path = shard_dir / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for envelope in rows:
            record = envelope["translated_record"]
            if record["expert"] == expert:
                mutate(record)
                _write_jsonl(path, rows)
                return
    raise AssertionError(f"missing expert {expert}")


def test_prepare_and_merge_publish_one_to_one_bilingual_snapshot(
    tmp_path: Path,
) -> None:
    source_dir, registry_path, snapshot, _records = _source_fixture(tmp_path)
    shard_dir = _ready_shards(tmp_path, source_dir, registry_path, snapshot)
    output_dir = tmp_path / "candidate_dataset_zh_cn"

    manifest = _merge(
        source_dir, registry_path, snapshot, shard_dir, output_dir
    )

    assert manifest["counts"] == {
        "source_records": 5,
        "translated_records": 5,
        "bilingual_records": 5,
        "shards": 4,
    }
    assert manifest["quality"]["one_to_one"] is True
    assert manifest["quality"]["empty_or_untranslated_records"] == 0
    assert manifest["exclusions"]["heldout_content_read"] is False
    assert {path.name for path in output_dir.iterdir()} == {
        *(name for name, _expert in SOURCE_FILES),
        "bilingual_snapshot.jsonl",
        "manifest.translation-zh-CN.json",
        "manifest.translation-zh-CN.json.sha256",
    }
    bilingual = [
        json.loads(line)
        for line in (output_dir / "bilingual_snapshot.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(bilingual) == 5
    assert all(row["target_id"] == row["source_id"] + TARGET_ID_SUFFIX for row in bilingual)
    planner = json.loads(
        (output_dir / "data_plan.jsonl").read_text(encoding="utf-8")
    )
    assert planner["provenance"]["source_id"] == "lineage-planner"
    assert "构建" in planner["output"]["summary"]


def test_merge_rejects_changed_code_url_or_protocol_tokens(tmp_path: Path) -> None:
    source_dir, registry_path, snapshot, _records = _source_fixture(tmp_path)
    shard_dir = _ready_shards(tmp_path, source_dir, registry_path, snapshot)
    shard = shard_dir / "part-000.jsonl"
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    planner = rows[0]["translated_record"]
    planner["input"]["requirement"] = planner["input"]["requirement"].replace(
        "https://example.test/api", "https://evil.test/api"
    )
    _write_jsonl(shard, rows)

    with pytest.raises(TranslationAuditError, match="code/URL/protocol tokens changed"):
        _merge(source_dir, registry_path, snapshot, shard_dir, tmp_path / "output")


def test_merge_rejects_changed_protected_code_field(tmp_path: Path) -> None:
    source_dir, registry_path, snapshot, _records = _source_fixture(tmp_path)
    shard_dir = _ready_shards(tmp_path, source_dir, registry_path, snapshot)
    _mutate_translated_record(
        shard_dir,
        "frontend_gen",
        lambda record: record["output"].__setitem__(
            "code", "export const App = () => <main>Changed</main>;"
        ),
    )

    with pytest.raises(TranslationAuditError, match="protected field changed"):
        _merge(source_dir, registry_path, snapshot, shard_dir, tmp_path / "output")


def test_merge_rejects_noncanonical_translated_message_binding(tmp_path: Path) -> None:
    source_dir, registry_path, snapshot, _records = _source_fixture(tmp_path)
    shard_dir = _ready_shards(tmp_path, source_dir, registry_path, snapshot)
    _mutate_translated_record(
        shard_dir,
        "tool_policy",
        lambda record: record["messages"][0].__setitem__(
            "content", "TOOL_POLICY|与结构化输入不一致的中文提示"
        ),
    )

    with pytest.raises(
        TranslationAuditError,
        match="user message is not canonical for translated input",
    ):
        _merge(source_dir, registry_path, snapshot, shard_dir, tmp_path / "output")


def test_merge_rejects_duplicate_and_missing_source_rows(tmp_path: Path) -> None:
    source_dir, registry_path, snapshot, _records = _source_fixture(tmp_path)
    shard_dir = _ready_shards(tmp_path, source_dir, registry_path, snapshot)
    first = (shard_dir / "part-000.jsonl").read_text(encoding="utf-8").splitlines()[0]
    with (shard_dir / "part-001.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")

    with pytest.raises(TranslationAuditError, match="duplicate source locator"):
        _merge(source_dir, registry_path, snapshot, shard_dir, tmp_path / "output")


def test_merge_rejects_duplicate_translated_training_payloads(tmp_path: Path) -> None:
    source_dir, registry_path, snapshot, _records = _source_fixture(tmp_path)
    plan_path = source_dir / "data_plan.jsonl"
    first = _record("planner")
    second = copy.deepcopy(first)
    second["id"] = "source-planner-duplicate"
    second["provenance"]["source_id"] = "lineage-planner-duplicate"
    _write_jsonl(plan_path, [first, second])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    plan_entry = next(
        entry for entry in registry["files"] if entry["expert"] == "planner"
    )
    plan_entry["bytes"] = plan_path.stat().st_size
    plan_entry["sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    shard_dir = _ready_shards(tmp_path, source_dir, registry_path, snapshot)

    with pytest.raises(TranslationAuditError, match="duplicate translated training payload"):
        _merge(source_dir, registry_path, snapshot, shard_dir, tmp_path / "output")


def test_merge_rejects_empty_or_wholly_untranslated_record(tmp_path: Path) -> None:
    source_dir, registry_path, snapshot, records = _source_fixture(tmp_path)
    shard_dir = _ready_shards(tmp_path, source_dir, registry_path, snapshot)
    shard = shard_dir / "part-000.jsonl"
    rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    source_id = rows[0]["source_id"]
    untranslated = copy.deepcopy(records[source_id])
    untranslated["id"] = source_id + TARGET_ID_SUFFIX
    rows[0]["translated_record"] = untranslated
    _write_jsonl(shard, rows)

    with pytest.raises(TranslationAuditError, match="untranslated record"):
        _merge(source_dir, registry_path, snapshot, shard_dir, tmp_path / "output")


def test_prepare_refuses_heldout_path_before_reading_extra_content(
    tmp_path: Path,
) -> None:
    source_dir, registry_path, snapshot, _records = _source_fixture(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["files"][0]["path"] = "artifacts/test/heldout/data_plan.jsonl"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(TranslationAuditError, match="heldout/benchmark path"):
        prepare_translation_shards(
            source_dir=source_dir,
            registry_path=registry_path,
            shard_dir=tmp_path / "shards",
            expected_snapshot_sha256=snapshot,
        )


def test_prepare_refuses_source_file_hash_drift(tmp_path: Path) -> None:
    source_dir, registry_path, snapshot, _records = _source_fixture(tmp_path)
    with (source_dir / "data_plan.jsonl").open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(TranslationAuditError, match="byte size drifted"):
        prepare_translation_shards(
            source_dir=source_dir,
            registry_path=registry_path,
            shard_dir=tmp_path / "shards",
            expected_snapshot_sha256=snapshot,
        )
