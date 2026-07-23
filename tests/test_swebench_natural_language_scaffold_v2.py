from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator
import pytest

from anchor_mvp.swebench import natural_language_scaffold_v2 as module
from anchor_mvp.swebench import taskboard_projector as projector
from anchor_mvp.swebench.natural_language_scaffold_v2 import (
    CAPABILITY_LABELS,
    EXPERTS,
    NaturalLanguageScaffoldV2Error,
    ScaffoldV2Config,
    audit_bundle_profiles,
    audit_training_view,
    freeze_bundle_profiles,
    materialize_training_view,
)
from anchor_mvp.swebench.schema import canonical_json
from tests.test_swebench_taskboard_projector import (
    CACHE_IDENTITY_FIELDS,
    CONFIG as PROJECTOR_CONFIG,
    HIERARCHICAL_TASK_KV_SUMMARY,
    _gold_rows,
    _task_bank_row,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/swebench_natural_language_scaffold_v2.yaml"
BUILD_CLI = ROOT / "scripts/data/build_swebench_natural_language_scaffold_v2.py"
AUDIT_CLI = ROOT / "scripts/data/audit_swebench_natural_language_scaffold_v2.py"
SCHEMAS = tuple(
    ROOT / "configs/research" / filename
    for filename in module.SCHEMA_FILENAMES.values()
)
STAGES = tuple(projector.STAGES)
STAGE_TO_EXPERT = dict(projector.STAGE_EXPERTS)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


def _sha_value(value: object) -> str:
    return _sha_text(canonical_json(value))


def _canonical_line(value: object) -> bytes:
    return canonical_json(value).encode("utf-8") + b"\n"


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(_canonical_line(row) for row in rows)
    path.write_bytes(data)
    return {
        "path": path.as_posix(),
        "sha256": _sha_bytes(data),
        "bytes": len(data),
        "records": len(rows),
    }


def _relative_binding(
    root: Path, relative: str, rows: list[Mapping[str, Any]]
) -> dict[str, Any]:
    binding = _write_jsonl(root / relative, rows)
    binding["path"] = relative
    return binding


def _projector_bundle(
    number: int,
    split: str,
    cfg: projector.TaskBoardProjectorConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_id = "swe-full-v1:" + _sha_text(f"v2-task:{split}:{number}")
    instance_id = f"body-free-fixture-{split}-{number}"
    language = "en" if number % 2 == 0 else "zh-CN"
    task_bank = _task_bank_row(
        task_id=task_id,
        instance_id=instance_id,
        partition="train" if split == "train" else "validation",
        language=language,
        secret=False,
    )
    gold = _gold_rows(
        split=split,
        task_id=task_id,
        instance_id=instance_id,
        task_number=number,
        mutate=lambda _split, stage, record: record["messages"][-1].update(
            {
                "content": (
                    str(record["messages"][-1]["content"])
                    + f"\nCURRENT_TARGET_SENTINEL_{split}_{number}_{stage}"
                )
            }
        ),
    )
    records = {stage: gold[STAGE_TO_EXPERT[stage]][0] for stage in STAGES}
    bundle_sha = _sha_text(f"v2-bundle:{split}:{number}")
    bundle = {
        "task_id": task_id,
        "task_bundle_sha256": bundle_sha,
        "task_bank": task_bank,
        "records": records,
    }
    rows = list(
        projector._sidecar_records(
            partition=split,
            bundle=bundle,
            config=cfg,
            snapshot_sha256=_sha_text("snapshot"),
            manifest_sha256=_sha_text("snapshot-manifest"),
            source_file_sha256={
                expert: _sha_text(f"source:{expert}") for expert in EXPERTS
            },
        )
    )
    return bundle, rows


def _build_projector(
    tmp_path: Path,
    *,
    train_bundles: int,
    calibration_bundles: int,
) -> tuple[Path, str, list[dict[str, Any]]]:
    root = tmp_path / "projector"
    root.mkdir()
    cfg = projector.TaskBoardProjectorConfig.load(PROJECTOR_CONFIG)
    clean_train: list[dict[str, Any]] = []
    noisy_train: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    bundle_metadata: list[dict[str, Any]] = []
    for split, count, offset in (
        ("train", train_bundles, 0),
        ("calibration", calibration_bundles, train_bundles),
    ):
        for local in range(count):
            bundle, rows = _projector_bundle(offset + local, split, cfg)
            bundle_metadata.append(
                {
                    "task_bundle_sha256": bundle["task_bundle_sha256"],
                    "task_id": bundle["task_id"],
                    "language": (
                        "en"
                        if bundle["task_bank"]["bilingual"]["requested_locale"]
                        == "en-US"
                        else "zh-CN"
                    ),
                    "source_split": split,
                }
            )
            if split == "train":
                clean_train.extend(row for row in rows if row["variant"] == "clean")
                noisy_train.extend(row for row in rows if row["variant"] == "noisy")
            else:
                calibration.extend(rows)
    partitions = (
        ("train/clean.jsonl", "train", "clean", clean_train),
        ("train/noisy.jsonl", "train", "noisy", noisy_train),
        ("calibration/clean.jsonl", "calibration", "clean", calibration),
    )
    file_bindings = [
        {
            **_relative_binding(root, relative, rows),
            "split": split,
            "variant": variant,
        }
        for relative, split, variant, rows in partitions
    ]
    all_rows = clean_train + noisy_train + calibration
    unique_segments: dict[str, str] = {}
    segment_refs = 0
    for row in all_rows:
        for segment in row["segment_plan"]["segments"]:
            segment_refs += 1
            unique_segments[str(segment["segment_id"])] = str(segment["cache_scope"])
    by_scope = Counter(unique_segments.values())
    by_split = Counter(str(row["split"]) for row in all_rows)
    by_variant = Counter(str(row["variant"]) for row in all_rows)
    by_stage = Counter(str(row["stage"]) for row in all_rows)
    by_expert = Counter(str(row["expert"]) for row in all_rows)
    by_language = Counter(str(row["training_record"]["language"]) for row in all_rows)
    manifest = {
        "schema_version": "anchor.swebench-taskboard-projector-manifest.v2",
        "input": {
            "snapshot_schema_version": "anchor.training-snapshot.v2",
            "snapshot_sha256": _sha_text("snapshot"),
            "snapshot_manifest_path": "manifest.json",
            "snapshot_manifest_sha256": _sha_text("snapshot-manifest"),
            "snapshot_sha256_sidecar_path": "manifest.json.sha256",
            "snapshot_sha256_sidecar_sha256": _sha_text("snapshot-sidecar"),
            "splits": ["train", "calibration"],
        },
        "producer": {
            "name": "anchor.swebench-taskboard-projector",
            "projector_version": "anchor.swebench-taskboard-projector.v2",
            "config_sha256": cfg.sha256,
            "sidecar_schema_sha256": cfg.sidecar_schema_sha256,
            "segment_plan_schema_version": "anchor.hierarchical-task-kv-segment-plan.v1",
            "segment_plan_schema_sha256": cfg.segment_plan_schema_sha256,
            "manifest_schema_sha256": cfg.manifest_schema_sha256,
            "record_schema_version": "anchor.query-specialization.v1",
        },
        "files": file_bindings,
        "counts": {
            "total": len(all_rows),
            "unique_task_bundles": len(bundle_metadata),
            "task_ids_sha256": _sha_value(
                sorted(item["task_id"] for item in bundle_metadata)
            ),
            "segment_references": segment_refs,
            "unique_segments": len(unique_segments),
            "unique_segments_by_cache_scope": {
                "task_shared_prefix": by_scope["task_shared_prefix"],
                "downstream_task_shared_immutable": by_scope[
                    "downstream_task_shared_immutable"
                ],
                "expert_private_delta": by_scope["expert_private_delta"],
            },
            "by_split": {
                "train": by_split["train"],
                "calibration": by_split["calibration"],
            },
            "by_variant": {
                "clean": by_variant["clean"],
                "noisy": by_variant["noisy"],
            },
            "by_stage": {stage: by_stage[stage] for stage in STAGES},
            "by_expert": {expert: by_expert[expert] for expert in EXPERTS},
            "by_language": {
                "en": by_language["en"],
                "zh-CN": by_language["zh-CN"],
            },
        },
        "hierarchical_task_kv": {
            **HIERARCHICAL_TASK_KV_SUMMARY,
            "cache_identity_required_exact_match_fields": CACHE_IDENTITY_FIELDS,
        },
        "split_group_key": "task_bundle_sha256",
        "task_id_cross_binding_key": "training_record.task_board.task_id",
        "all_five_role_views_same_split": True,
        "canonical_gold_written": False,
        "provider_requests": 0,
        "heldout_content_read": False,
        "heldout_content_emitted": False,
        "split_preserved": True,
        "augmentation_applied_after_split": True,
        "claim_scope": "research_proxy_only",
    }
    manifest_data = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode(
            "utf-8"
        )
        + b"\n"
    )
    (root / "manifest.json").write_bytes(manifest_data)
    manifest_sha = _sha_bytes(manifest_data)
    (root / "manifest.json.sha256").write_bytes(
        f"{manifest_sha}  manifest.json\n".encode("ascii")
    )
    return root, manifest_sha, bundle_metadata


def _write_descriptors(
    tmp_path: Path,
    bundle_metadata: list[dict[str, Any]],
    *,
    mutate: Callable[[int, dict[str, Any]], None] | None = None,
) -> tuple[Path, str]:
    path = tmp_path / "bundle-profile-descriptors.jsonl"
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(bundle_metadata):
        source_split = source["source_split"]
        row = {
            "schema_version": (
                "anchor.frozen-prefix-qreader-bundle-profile-descriptor.v2"
            ),
            "task_bundle_sha256": source["task_bundle_sha256"],
            "task_id": source["task_id"],
            "task_id_sha256": _sha_text(source["task_id"]),
            "task_semantic_sha256": _sha_text(f"semantic:{index}"),
            "language": source["language"],
            "information_flow_stratum": module.STRATA[index % len(module.STRATA)],
            "source_split": source_split,
            "output_split": "train" if source_split == "train" else "eval_proxy",
            "problem_profile_sha256": _sha_text(f"problem-profile:{index}"),
            "capability_labels": [CAPABILITY_LABELS[index % len(CAPABILITY_LABELS)]],
        }
        if mutate is not None:
            mutate(index, row)
        rows.append(row)
    data = b"".join(_canonical_line(row) for row in rows)
    path.write_bytes(data)
    sha = _sha_bytes(data)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{sha}  {path.name}\n".encode("ascii")
    )
    return path, sha


def _resign_partition(
    root: Path,
    relative: str,
    data: bytes,
) -> str:
    (root / relative).write_bytes(data)
    manifest = _read_json(root / "manifest.json")
    binding = next(item for item in manifest["files"] if item["path"] == relative)
    binding.update(
        {
            "sha256": _sha_bytes(data),
            "bytes": len(data),
            "records": sum(1 for line in data.splitlines() if line.strip()),
        }
    )
    manifest_data = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    (root / "manifest.json").write_bytes(manifest_data)
    manifest_sha = _sha_bytes(manifest_data)
    (root / "manifest.json.sha256").write_bytes(
        f"{manifest_sha}  manifest.json\n".encode("ascii")
    )
    return manifest_sha


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _freeze_and_materialize(
    tmp_path: Path,
    *,
    train_bundles: int = 1,
    calibration_bundles: int = 1,
) -> tuple[Path, str, Path, str, Path, str]:
    projector_root, projector_sha, metadata = _build_projector(
        tmp_path,
        train_bundles=train_bundles,
        calibration_bundles=calibration_bundles,
    )
    descriptor, descriptor_sha = _write_descriptors(tmp_path, metadata)
    profile_root = tmp_path / "bundle-profile"
    profile_result = freeze_bundle_profiles(
        CONFIG,
        projector_root,
        projector_sha,
        descriptor,
        descriptor_sha,
        profile_root,
    )
    output = tmp_path / "training-view"
    view_result = materialize_training_view(
        CONFIG,
        projector_root,
        projector_sha,
        profile_root,
        str(profile_result["manifest_sha256"]),
        output,
    )
    return (
        projector_root,
        projector_sha,
        profile_root,
        str(profile_result["manifest_sha256"]),
        output,
        str(view_result["manifest_sha256"]),
    )


def test_all_schemas_and_yaml_pass_real_draft_2020_12() -> None:
    config = ScaffoldV2Config.load(CONFIG)
    assert config.raw["schema_version"] == module.CONFIG_SCHEMA
    for path in SCHEMAS:
        schema = _read_json(path)
        Draft202012Validator.check_schema(schema)
    Draft202012Validator(_read_json(SCHEMAS[0])).validate(config.raw)


@pytest.mark.parametrize(
    ("train_bundles", "calibration_bundles"),
    [(1, 1), (3, 2)],
)
def test_freezes_explicit_bundle_profiles_for_dynamic_bundle_counts(
    tmp_path: Path,
    train_bundles: int,
    calibration_bundles: int,
) -> None:
    projector_root, projector_sha, metadata = _build_projector(
        tmp_path,
        train_bundles=train_bundles,
        calibration_bundles=calibration_bundles,
    )
    descriptor, descriptor_sha = _write_descriptors(tmp_path, metadata)
    result = freeze_bundle_profiles(
        CONFIG,
        projector_root,
        projector_sha,
        descriptor,
        descriptor_sha,
        tmp_path / "profiles",
    )
    assert result["records"] == train_bundles + calibration_bundles
    manifest = _read_json(tmp_path / "profiles/manifest.json")
    assert manifest["counts"]["bundles"] == result["records"]
    assert set(manifest["counts"]["by_capability_label"]) == set(CAPABILITY_LABELS)
    assert manifest["training_authorized"] is False
    assert manifest["formal_training_authorized"] is False
    audit = audit_bundle_profiles(
        CONFIG, tmp_path / "profiles", str(result["manifest_sha256"])
    )
    assert audit["records"] == result["records"]
    assert audit["provider_requests"] == 0


def test_materializes_deterministic_25_records_and_real_schema_validation(
    tmp_path: Path,
) -> None:
    projector_root, projector_sha, metadata = _build_projector(
        tmp_path, train_bundles=3, calibration_bundles=2
    )
    descriptor, descriptor_sha = _write_descriptors(tmp_path, metadata)
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    freeze_a = freeze_bundle_profiles(
        CONFIG,
        projector_root,
        projector_sha,
        descriptor,
        descriptor_sha,
        profile_a,
    )
    freeze_b = freeze_bundle_profiles(
        CONFIG,
        projector_root,
        projector_sha,
        descriptor,
        descriptor_sha,
        profile_b,
    )
    assert _tree(profile_a) == _tree(profile_b)
    output_a = tmp_path / "view-a"
    output_b = tmp_path / "view-b"
    result_a = materialize_training_view(
        CONFIG,
        projector_root,
        projector_sha,
        profile_a,
        str(freeze_a["manifest_sha256"]),
        output_a,
    )
    result_b = materialize_training_view(
        CONFIG,
        projector_root,
        projector_sha,
        profile_b,
        str(freeze_b["manifest_sha256"]),
        output_b,
    )
    assert result_a["records"] == result_b["records"] == 25
    assert result_a["pairs"] == 0
    assert _tree(output_a) == _tree(output_b)
    validator = Draft202012Validator(
        _read_json(
            ROOT
            / "configs/research/swebench_natural_language_scaffold_v2_record.schema.json"
        )
    )
    rows = _read_jsonl(output_a / "train.jsonl") + _read_jsonl(
        output_a / "eval_proxy.jsonl"
    )
    assert len(rows) == 25
    assert len({row["task_bundle_sha256"] for row in rows}) == 5
    for row in rows:
        validator.validate(row)
        assert row["scaffold_variant"] == "concise_rationale_plus_json"
        assert row["pair_id"] is None
        assert row["adapter_control"] == {
            "primary": "q_only",
            "diagnostic_overlays": ["o_only", "q_plus_o"],
            "wide_lora_inherited": False,
        }
        assert row["training_view"]["routing_json_sha256"] == _sha_value(
            row["training_view"]["routing_json"]
        )
        assert "wide_lora" not in row["adapter_control"]["diagnostic_overlays"]


def test_current_stage_commit_is_only_in_assistant_target_and_hash_bound(
    tmp_path: Path,
) -> None:
    projector_root, _, _, _, output, _ = _freeze_and_materialize(tmp_path)
    source_targets: dict[tuple[str, str], str] = {}
    source_forbidden: dict[tuple[str, str], set[str]] = {}
    for relative in ("train/noisy.jsonl", "calibration/clean.jsonl"):
        for row in _read_jsonl(projector_root / relative):
            source_targets[(row["task_bundle_sha256"], row["stage"])] = row[
                "training_record"
            ]["target"]["answer"]
            source_forbidden[(row["task_bundle_sha256"], row["stage"])] = set(
                row["training_record"]["attention_targets"]["forbidden_block_ids"]
            )
    for row in _read_jsonl(output / "train.jsonl") + _read_jsonl(
        output / "eval_proxy.jsonl"
    ):
        target = source_targets[(row["task_bundle_sha256"], row["stage"])]
        training = row["training_view"]
        assistant_target = training["assistant_target"]
        assert assistant_target["stage_commit"]["text"] == target
        assert assistant_target["text_sha256"] == _sha_text(assistant_target["text"])
        assert _sha_text(target) == row["source_binding"]["target_sha256"]
        assert target not in training["shared_prefix_text"]
        assert target not in training["scaffold_text"]
        assert target not in training["request2_input_text"]
        trace = assistant_target["tool_trace"]
        if row["expert"] == "frontend_gen":
            assert trace["status"] == "authenticated_source"
            assert (
                trace["binding_sha256"] == row["source_binding"]["source_gold_sha256"]
            )
            assert trace["events"]
            assert trace["evidence_sha256"] == _sha_value(trace["events"])
            for event in trace["events"]:
                assert (
                    event["source_block_id"]
                    in source_forbidden[(row["task_bundle_sha256"], row["stage"])]
                )
                assert event["kind"] in {"tool_call", "tool_result", "test_result"}
                assert event["content_sha256"] == _sha_text(event["text"])
                assert event["text"] not in training["shared_prefix_text"]
                assert event["text"] not in training["request2_input_text"]
        else:
            assert trace == {
                "status": "not_applicable",
                "source": "not_applicable",
                "binding_sha256": None,
                "evidence_sha256": None,
                "events": [],
            }


def test_1000_record_path_is_streamed_without_path_bulk_jsonl_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projector_root, projector_sha, metadata = _build_projector(
        tmp_path, train_bundles=160, calibration_bundles=40
    )
    descriptor, descriptor_sha = _write_descriptors(tmp_path, metadata)
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def guarded_read_bytes(path: Path) -> bytes:
        if path.suffix == ".jsonl":
            raise AssertionError("bulk JSONL read forbidden")
        return original_read_bytes(path)

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.suffix == ".jsonl":
            raise AssertionError("bulk JSONL read forbidden")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    profile = tmp_path / "profile-200"
    freeze = freeze_bundle_profiles(
        CONFIG,
        projector_root,
        projector_sha,
        descriptor,
        descriptor_sha,
        profile,
    )
    assert freeze["records"] == 200
    output = tmp_path / "view-1000"
    result = materialize_training_view(
        CONFIG,
        projector_root,
        projector_sha,
        profile,
        str(freeze["manifest_sha256"]),
        output,
    )
    assert result["records"] == 1000
    assert result["unique_task_bundles"] == 200
    manifest = _read_json(output / "manifest.json")
    assert manifest["counts"]["by_split"] == {"train": 800, "eval_proxy": 200}
    assert all(value == 200 for value in manifest["counts"]["by_role"].values())
    assert manifest["counts"]["pair_count"] == 0
    assert manifest["safety"]["provider_requests"] == 0
    assert manifest["safety"]["model_loads"] == 0
    assert manifest["safety"]["gpu_requests"] == 0
    assert manifest["safety"]["network_requests"] == 0


def test_rejects_bad_role_and_split_and_duplicate_semantic(
    tmp_path: Path,
) -> None:
    projector_root, projector_sha, metadata = _build_projector(
        tmp_path, train_bundles=2, calibration_bundles=1
    )
    bad_descriptor, bad_sha = _write_descriptors(
        tmp_path,
        metadata,
        mutate=lambda index, row: (
            row.update({"output_split": "eval_proxy"}) if index == 0 else None
        ),
    )
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            projector_sha,
            bad_descriptor,
            bad_sha,
            tmp_path / "bad-split",
        )
    assert exc.value.code == "scaffold_v2_descriptor_cross_binding_invalid"

    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    duplicate_descriptor, duplicate_sha = _write_descriptors(
        duplicate_dir,
        metadata,
        mutate=lambda index, row: (
            row.update({"task_semantic_sha256": _sha_text("same")})
            if index in {0, 1}
            else None
        ),
    )
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            projector_sha,
            duplicate_descriptor,
            duplicate_sha,
            tmp_path / "bad-semantic",
        )
    assert exc.value.code == "scaffold_v2_descriptor_semantic_identity_invalid"

    # A malformed role is rejected by the real published projector schema.
    path = projector_root / "train/noisy.jsonl"
    rows = _read_jsonl(path)
    rows[0]["expert"] = "frontend_review"
    data = b"".join(_canonical_line(row) for row in rows)
    path.write_bytes(data)
    manifest = _read_json(projector_root / "manifest.json")
    binding = next(
        item for item in manifest["files"] if item["path"] == "train/noisy.jsonl"
    )
    binding.update(
        {"sha256": _sha_bytes(data), "bytes": len(data), "records": len(rows)}
    )
    manifest_data = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode(
            "utf-8"
        )
        + b"\n"
    )
    (projector_root / "manifest.json").write_bytes(manifest_data)
    bad_projector_sha = _sha_bytes(manifest_data)
    (projector_root / "manifest.json.sha256").write_bytes(
        f"{bad_projector_sha}  manifest.json\n".encode("ascii")
    )
    good_dir = tmp_path / "good-descriptor"
    good_dir.mkdir()
    descriptor, descriptor_sha = _write_descriptors(good_dir, metadata)
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            bad_projector_sha,
            descriptor,
            descriptor_sha,
            tmp_path / "bad-role",
        )
    assert exc.value.code in {
        "scaffold_v2_projector_partition_invalid",
        "scaffold_v2_projector_record_invalid",
        "scaffold_v2_segment_plan_invalid",
    }


def test_rejects_dot_path_and_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projector_root, projector_sha, metadata = _build_projector(
        tmp_path, train_bundles=1, calibration_bundles=1
    )
    descriptor, descriptor_sha = _write_descriptors(tmp_path, metadata)
    manifest = _read_json(projector_root / "manifest.json")
    manifest["files"][0]["path"] = "./train/clean.jsonl"
    data = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode(
            "utf-8"
        )
        + b"\n"
    )
    (projector_root / "manifest.json").write_bytes(data)
    bad_sha = _sha_bytes(data)
    (projector_root / "manifest.json.sha256").write_bytes(
        f"{bad_sha}  manifest.json\n".encode("ascii")
    )
    with pytest.raises(NaturalLanguageScaffoldV2Error):
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            bad_sha,
            descriptor,
            descriptor_sha,
            tmp_path / "dot-path",
        )

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    projector_root, projector_sha, metadata = _build_projector(
        clean_root, train_bundles=1, calibration_bundles=1
    )
    descriptor, descriptor_sha = _write_descriptors(clean_root, metadata)
    original = module._verify_stream_unchanged
    mutated = False

    def mutate_before_terminal(seal: module.StreamSeal, code: str) -> None:
        nonlocal mutated
        if not mutated and seal.path == descriptor:
            mutated = True
            seal.path.write_bytes(seal.path.read_bytes() + b" ")
        original(seal, code)

    monkeypatch.setattr(module, "_verify_stream_unchanged", mutate_before_terminal)
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            projector_sha,
            descriptor,
            descriptor_sha,
            tmp_path / "toctou",
        )
    assert exc.value.code == "scaffold_v2_descriptor_changed_during_read"


def test_rejects_absolute_input_symlink(tmp_path: Path) -> None:
    projector_root, projector_sha, metadata = _build_projector(
        tmp_path, train_bundles=1, calibration_bundles=1
    )
    descriptor, descriptor_sha = _write_descriptors(tmp_path, metadata)
    link = tmp_path / "descriptor-link.jsonl"
    try:
        link.symlink_to(descriptor)
    except OSError:
        pytest.skip("symlink capability unavailable")
    link.with_name(link.name + ".sha256").write_bytes(
        f"{descriptor_sha}  {link.name}\n".encode("ascii")
    )
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            projector_sha,
            link,
            descriptor_sha,
            tmp_path / "symlink",
        )
    assert exc.value.code == "scaffold_v2_descriptor_invalid"


def test_parse_error_cli_never_echoes_malformed_body(
    tmp_path: Path,
) -> None:
    projector_root, projector_sha, metadata = _build_projector(
        tmp_path, train_bundles=1, calibration_bundles=1
    )
    descriptor, _ = _write_descriptors(tmp_path, metadata)
    secret = "DO_NOT_ECHO_MALFORMED_BODY_93cc"
    data = descriptor.read_bytes() + f'{{"bad":"{secret}"\n'.encode()
    descriptor.write_bytes(data)
    descriptor_sha = _sha_bytes(data)
    descriptor.with_name(descriptor.name + ".sha256").write_bytes(
        f"{descriptor_sha}  {descriptor.name}\n".encode("ascii")
    )
    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_CLI),
            "freeze-bundle-profile",
            "--config",
            str(CONFIG),
            "--projector-dir",
            str(projector_root),
            "--projector-manifest-sha256",
            projector_sha,
            "--descriptor-jsonl",
            str(descriptor),
            "--descriptor-sha256",
            descriptor_sha,
            "--output-dir",
            str(tmp_path / "parse-error"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "scaffold_v2_descriptor_invalid"
    assert secret not in result.stderr
    assert secret not in result.stdout


def test_rejects_duplicate_keys_nonfinite_json_and_duplicate_yaml(
    tmp_path: Path,
) -> None:
    # Projector manifest: a duplicate top-level key must fail before schema use.
    manifest_case = tmp_path / "manifest-case"
    manifest_case.mkdir()
    projector_root, _projector_sha, metadata = _build_projector(
        manifest_case, train_bundles=1, calibration_bundles=1
    )
    descriptor, descriptor_sha = _write_descriptors(manifest_case, metadata)
    original_manifest = (projector_root / "manifest.json").read_bytes()
    duplicate_manifest = (
        b'{"schema_version":"duplicate",' + original_manifest.lstrip()[1:]
    )
    (projector_root / "manifest.json").write_bytes(duplicate_manifest)
    duplicate_manifest_sha = _sha_bytes(duplicate_manifest)
    (projector_root / "manifest.json.sha256").write_bytes(
        f"{duplicate_manifest_sha}  manifest.json\n".encode("ascii")
    )
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            duplicate_manifest_sha,
            descriptor,
            descriptor_sha,
            manifest_case / "output",
        )
    assert exc.value.code == "scaffold_v2_projector_manifest_invalid"

    # Explicit descriptor: non-finite JSON is never accepted as a scalar.
    descriptor_case = tmp_path / "descriptor-case"
    descriptor_case.mkdir()
    projector_root, projector_sha, metadata = _build_projector(
        descriptor_case, train_bundles=1, calibration_bundles=1
    )
    descriptor, _ = _write_descriptors(descriptor_case, metadata)
    first = json.loads(descriptor.read_text(encoding="utf-8").splitlines()[0])
    needle = (f'"task_semantic_sha256":"{first["task_semantic_sha256"]}"').encode(
        "utf-8"
    )
    invalid_descriptor = descriptor.read_bytes().replace(
        needle,
        b'"task_semantic_sha256":NaN',
        1,
    )
    descriptor.write_bytes(invalid_descriptor)
    invalid_descriptor_sha = _sha_bytes(invalid_descriptor)
    descriptor.with_name(descriptor.name + ".sha256").write_bytes(
        f"{invalid_descriptor_sha}  {descriptor.name}\n".encode("ascii")
    )
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            projector_sha,
            descriptor,
            invalid_descriptor_sha,
            descriptor_case / "output",
        )
    assert exc.value.code == "scaffold_v2_descriptor_invalid"

    # Projector JSONL: duplicate keys cannot be normalized away by json.loads.
    projector_case = tmp_path / "projector-case"
    projector_case.mkdir()
    projector_root, _projector_sha, metadata = _build_projector(
        projector_case, train_bundles=1, calibration_bundles=1
    )
    descriptor, descriptor_sha = _write_descriptors(projector_case, metadata)
    partition = projector_root / "train/clean.jsonl"
    lines = partition.read_bytes().splitlines(keepends=True)
    lines[0] = b'{"schema_version":"duplicate",' + lines[0][1:]
    duplicate_projector_sha = _resign_partition(
        projector_root,
        "train/clean.jsonl",
        b"".join(lines),
    )
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            duplicate_projector_sha,
            descriptor,
            descriptor_sha,
            projector_case / "output",
        )
    assert exc.value.code == "scaffold_v2_projector_partition_invalid"

    # YAML config: duplicate mappings are rejected before schema validation.
    duplicate_config = tmp_path / "duplicate-config.yaml"
    duplicate_config.write_bytes(
        CONFIG.read_bytes()
        + b"\nschema_version: anchor.frozen-prefix-qreader-training-view-config.v2\n"
    )
    snapshot = module._read_small(duplicate_config, "scaffold_v2_config_invalid")
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        module._yaml_snapshot(snapshot, "scaffold_v2_config_invalid")
    assert exc.value.code == "scaffold_v2_config_invalid"


def test_audit_rejects_nonfinite_output_jsonl(
    tmp_path: Path,
) -> None:
    _, _, _, _, output, _manifest_sha = _freeze_and_materialize(tmp_path)
    train = output / "train.jsonl"
    invalid = train.read_bytes().replace(b'"pair_id":null', b'"pair_id":NaN', 1)
    manifest_sha = _resign_partition(output, "train.jsonl", invalid)
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        audit_training_view(CONFIG, output, manifest_sha)
    assert exc.value.code == "scaffold_v2_audit_partition_invalid"


def test_post_publish_terminal_drift_removes_only_new_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = module._publish_directory

    freeze_case = tmp_path / "freeze-drift"
    freeze_case.mkdir()
    projector_root, projector_sha, metadata = _build_projector(
        freeze_case, train_bundles=1, calibration_bundles=1
    )
    descriptor, descriptor_sha = _write_descriptors(freeze_case, metadata)
    frozen_output = freeze_case / "profile"

    def publish_then_mutate_source(
        temporary: Path,
        output: Path,
        parent_identity: tuple[int, int, int, int],
    ) -> None:
        original_publish(temporary, output, parent_identity)
        descriptor.write_bytes(descriptor.read_bytes() + b" ")

    monkeypatch.setattr(module, "_publish_directory", publish_then_mutate_source)
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        freeze_bundle_profiles(
            CONFIG,
            projector_root,
            projector_sha,
            descriptor,
            descriptor_sha,
            frozen_output,
        )
    assert exc.value.code == "scaffold_v2_descriptor_changed_after_publish"
    assert not frozen_output.exists()

    materialize_case = tmp_path / "materialize-drift"
    materialize_case.mkdir()
    projector_root, projector_sha, metadata = _build_projector(
        materialize_case, train_bundles=1, calibration_bundles=1
    )
    descriptor, descriptor_sha = _write_descriptors(materialize_case, metadata)
    profile = materialize_case / "profile"
    monkeypatch.setattr(module, "_publish_directory", original_publish)
    profile_result = freeze_bundle_profiles(
        CONFIG,
        projector_root,
        projector_sha,
        descriptor,
        descriptor_sha,
        profile,
    )
    training_output = materialize_case / "training-view"

    def publish_then_mutate_output(
        temporary: Path,
        output: Path,
        parent_identity: tuple[int, int, int, int],
    ) -> None:
        original_publish(temporary, output, parent_identity)
        path = output / "train.jsonl"
        path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(module, "_publish_directory", publish_then_mutate_output)
    with pytest.raises(NaturalLanguageScaffoldV2Error) as exc:
        materialize_training_view(
            CONFIG,
            projector_root,
            projector_sha,
            profile,
            str(profile_result["manifest_sha256"]),
            training_output,
        )
    assert exc.value.code == "scaffold_v2_output_post_publish_invalid"
    assert not training_output.exists()


def test_audit_cli_and_function_are_zero_request_and_fail_closed(
    tmp_path: Path,
) -> None:
    _, _, _, _, output, manifest_sha = _freeze_and_materialize(tmp_path)
    result = audit_training_view(CONFIG, output, manifest_sha)
    assert result == {
        "status": "authenticated_research_proxy_only",
        "manifest_sha256": manifest_sha,
        "records": 10,
        "provider_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "network_requests": 0,
        "training_authorized": False,
        "formal_training_authorized": False,
    }
    cli = subprocess.run(
        [
            sys.executable,
            str(AUDIT_CLI),
            "--config",
            str(CONFIG),
            "--artifact-dir",
            str(output),
            "--manifest-sha256",
            manifest_sha,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert cli.returncode == 0
    payload = json.loads(cli.stdout)
    assert payload["records"] == 10
    assert payload["provider_requests"] == 0
    assert payload["training_authorized"] is False
    (output / "manifest.json.sha256").write_text(
        f"{'0' * 64}  manifest.json\n", encoding="ascii", newline="\n"
    )
    failed = subprocess.run(
        [
            sys.executable,
            str(AUDIT_CLI),
            "--config",
            str(CONFIG),
            "--artifact-dir",
            str(output),
            "--manifest-sha256",
            manifest_sha,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode == 2
    assert failed.stderr.strip() == "scaffold_v2_audit_sidecar_invalid"
