from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

import pytest

from anchor_mvp.swebench import long_context_preflight as preflight_module
from anchor_mvp.swebench.long_context_preflight import (
    LongContextPreflightError,
    SyntheticFixtureTokenCounter,
    build_long_context_token_inventory,
)
from anchor_mvp.swebench.taskboard_projector import (
    STAGES,
    STAGE_EXPERTS,
    project_taskboards,
)
from tests.test_swebench_taskboard_projector import (
    CONFIG as PROJECTOR_CONFIG,
    _build_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/swebench_long_context_preflight_v1.yaml"
RECORD_SCHEMA = (
    ROOT / "configs/research/swebench_long_context_preflight_sidecar.schema.json"
)
MANIFEST_SCHEMA = (
    ROOT / "configs/research/swebench_long_context_preflight_manifest.schema.json"
)
CLI = ROOT / "scripts/data/preflight_swebench_long_context.py"
FIXED_FILES = (
    "train/clean.jsonl",
    "train/noisy.jsonl",
    "calibration/clean.jsonl",
)
FORBIDDEN_SENTINEL = "[PASS]"


@dataclass(frozen=True)
class ProjectedFixture:
    root: Path
    manifest_sha256: str


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _project_fixture(
    tmp_path: Path,
) -> ProjectedFixture:
    snapshot = _build_snapshot(tmp_path)
    projector_root = tmp_path / "projector"
    result = project_taskboards(
        PROJECTOR_CONFIG,
        snapshot.root,
        snapshot.manifest_sha256,
        projector_root,
    )
    return ProjectedFixture(
        root=projector_root,
        manifest_sha256=str(result["manifest_sha256"]),
    )


def _build_inventory(
    fixture: ProjectedFixture,
    output: Path,
    *,
    counter: Any | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    return build_long_context_token_inventory(
        CONFIG,
        fixture.root,
        expected_sha256 or fixture.manifest_sha256,
        output,
        counter=counter or SyntheticFixtureTokenCounter(),
    )


def _resign_projector_file(
    fixture: ProjectedFixture,
    relative: str,
    mutate: Callable[[list[dict[str, Any]]], None],
) -> ProjectedFixture:
    path = fixture.root / relative
    rows = _read_jsonl(path)
    mutate(rows)
    data = b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.write_bytes(data)
    manifest_path = fixture.root / "manifest.json"
    manifest = _read_json(manifest_path)
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    entry.update(
        {
            "sha256": _sha256(data),
            "bytes": len(data),
            "records": len(rows),
        }
    )
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = _sha256(manifest_bytes)
    (fixture.root / "manifest.json.sha256").write_bytes(
        f"{manifest_sha256}  manifest.json\n".encode("ascii")
    )
    return ProjectedFixture(fixture.root, manifest_sha256)


def _resign_projector_manifest(
    fixture: ProjectedFixture,
    mutate: Callable[[dict[str, Any]], None],
) -> ProjectedFixture:
    manifest_path = fixture.root / "manifest.json"
    manifest = _read_json(manifest_path)
    mutate(manifest)
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = _sha256(manifest_bytes)
    (fixture.root / "manifest.json.sha256").write_bytes(
        f"{manifest_sha256}  manifest.json\n".encode("ascii")
    )
    return ProjectedFixture(fixture.root, manifest_sha256)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _contains_denied_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in preflight_module._DENIED_OUTPUT_KEYS
            or _contains_denied_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_denied_key(item) for item in value)
    return False


def _output_rows(output: Path) -> list[dict[str, Any]]:
    return [row for relative in FIXED_FILES for row in _read_jsonl(output / relative)]


def test_synthetic_inventory_happy_deterministic_and_body_free(
    tmp_path: Path,
) -> None:
    fixture = _project_fixture(tmp_path)
    first = tmp_path / "inventory-first"
    second = tmp_path / "inventory-second"
    result = _build_inventory(fixture, first)
    second_result = _build_inventory(fixture, second)

    assert result["records"] == second_result["records"] == 15
    assert result["provider_requests"] == second_result["provider_requests"] == 0
    assert _tree_bytes(first) == _tree_bytes(second)
    assert set(_tree_bytes(first)) == {
        *FIXED_FILES,
        "manifest.json",
        "manifest.json.sha256",
    }

    manifest_bytes = (first / "manifest.json").read_bytes()
    manifest_sha256 = _sha256(manifest_bytes)
    assert (first / "manifest.json.sha256").read_bytes() == (
        f"{manifest_sha256}  manifest.json\n".encode("ascii")
    )
    manifest = json.loads(manifest_bytes)
    manifest_schema = _read_json(MANIFEST_SCHEMA)
    record_schema = _read_json(RECORD_SCHEMA)
    assert set(manifest) == set(manifest_schema["required"])
    assert manifest["counts"]["total"] == 15
    assert manifest["provider_requests"] == 0
    assert manifest["manifest_sha256_sidecar_required"] is True
    assert manifest["status"] == "synthetic_fixture_inventory_ready"
    assert manifest["inventory_mode"] == "synthetic_fixture"
    assert manifest["target_model_tokenizer_match"] == "not_applicable"
    assert manifest["bucket_basis"] == "bound_tokenizer_total_tokens"
    assert manifest["capability_validated"] is False
    assert manifest["claim_scope"] == "synthetic_fixture_contract_only"
    assert manifest["tokenizer_binding"]["backend"] == "explicit_synthetic_tokenizer"
    assert (
        manifest["tokenizer_binding"]["tokenizer_label_source"]
        == "caller_supplied_and_hash_bound"
    )
    assert manifest["tokenizer_binding"]["exact_token_counts"] is True
    assert manifest["tokenizer_binding"]["synthetic_fixture_only"] is True
    source_manifest = _read_json(fixture.root / "manifest.json")
    assert (
        manifest["counts"]["task_ids_sha256"]
        == source_manifest["counts"]["task_ids_sha256"]
    )
    assert [item["path"] for item in manifest["files"]] == list(FIXED_FILES)
    assert [item["records"] for item in manifest["files"]] == [5, 5, 5]

    rows = _output_rows(first)
    assert len(rows) == 15
    assert all(set(row) == set(record_schema["required"]) for row in rows)
    assert all(
        row["record_id"].startswith("long-context-token-inventory-v1:")
        and len(row["record_id"]) == len("long-context-token-inventory-v1:") + 64
        for row in rows
    )
    assert not _contains_denied_key(manifest)
    assert all(not _contains_denied_key(row) for row in rows)
    assert FORBIDDEN_SENTINEL.encode("utf-8") not in b"".join(
        _tree_bytes(first).values()
    )
    assert all(row["provider_requests"] == 0 for row in rows)
    assert all(
        row["input_tokens"] + row["reserved_output_tokens"] == row["total_tokens"]
        for row in rows
    )
    assert all(
        row["shared_prefix_input_tokens"] + row["private_delta_input_tokens"]
        == row["input_tokens"]
        for row in rows
    )

    expected_roles = {(stage, STAGE_EXPERTS[stage]) for stage in STAGES}
    groups: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    bundle_splits: dict[str, set[str]] = {}
    for row in rows:
        group = (row["split"], row["variant"], row["task_bundle_sha256"])
        groups.setdefault(group, set()).add((row["stage"], row["expert"]))
        bundle_splits.setdefault(row["task_bundle_sha256"], set()).add(row["split"])
    assert len(groups) == 3
    assert all(roles == expected_roles for roles in groups.values())
    assert all(len(splits) == 1 for splits in bundle_splits.values())


def test_schemas_close_synthetic_mode_stage_expert_and_published_ranges() -> None:
    manifest_schema = _read_json(MANIFEST_SCHEMA)
    record_schema = _read_json(RECORD_SCHEMA)

    backend_conditions: dict[str, dict[str, Any]] = {}
    for condition in manifest_schema["allOf"]:
        tokenizer = (
            condition.get("if", {}).get("properties", {}).get("tokenizer_binding", {})
        )
        backend = tokenizer.get("properties", {}).get("backend", {}).get("const")
        if backend is not None:
            backend_conditions[str(backend)] = {
                key: value["const"]
                for key, value in condition["then"]["properties"].items()
            }
    assert backend_conditions["explicit_synthetic_tokenizer"] == {
        "status": "synthetic_fixture_inventory_ready",
        "inventory_mode": "synthetic_fixture",
        "target_model_tokenizer_match": "not_applicable",
        "claim_scope": "synthetic_fixture_contract_only",
    }
    synthetic_manifest_condition = next(
        condition
        for condition in manifest_schema["allOf"]
        if condition.get("if", {})
        .get("properties", {})
        .get("tokenizer_binding", {})
        .get("properties", {})
        .get("backend", {})
        .get("const")
        == "explicit_synthetic_tokenizer"
    )
    assert synthetic_manifest_condition["if"]["required"] == ["tokenizer_binding"]
    assert synthetic_manifest_condition["if"]["properties"]["tokenizer_binding"][
        "required"
    ] == ["backend"]
    tokenizer_schema = manifest_schema["$defs"]["tokenizer_binding"]
    assert tokenizer_schema["properties"]["tokenizer_label_source"] == {
        "const": "caller_supplied_and_hash_bound"
    }
    synthetic_binding_condition = next(
        condition
        for condition in tokenizer_schema["allOf"]
        if condition["if"]["properties"]["backend"].get("const")
        == "explicit_synthetic_tokenizer"
    )
    assert synthetic_binding_condition["then"]["properties"] == {
        "synthetic_fixture_only": {"const": True}
    }

    stage_experts: dict[str, str] = {}
    for condition in record_schema["allOf"]:
        stage = (
            condition.get("if", {}).get("properties", {}).get("stage", {}).get("const")
        )
        if stage is not None:
            stage_experts[str(stage)] = condition["then"]["properties"]["expert"][
                "const"
            ]
    assert stage_experts == dict(STAGE_EXPERTS)

    published_buckets = record_schema["$defs"]["bucket"]["enum"]
    published_gates = record_schema["$defs"]["gate"]["enum"]
    assert published_buckets == [
        name for name, _limit, _gate in preflight_module.BUCKETS
    ]
    assert "gt_1m" not in published_buckets
    assert published_gates == [
        "measurement_candidate",
        "capability_only",
        "research_only_blocked",
    ]
    assert "reject" not in published_gates


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (8_191, ("le_8k", "measurement_candidate")),
        (8_192, ("le_8k", "measurement_candidate")),
        (8_193, ("le_16k", "measurement_candidate")),
        (262_144, ("le_256k", "capability_only")),
        (1_048_576, ("le_1m", "research_only_blocked")),
    ],
)
def test_bucket_boundaries(tokens: int, expected: tuple[str, str]) -> None:
    assert preflight_module._bucket(tokens) == expected


def test_bucket_rejects_over_one_million() -> None:
    with pytest.raises(
        LongContextPreflightError, match="long_context_over_1m_rejected"
    ):
        preflight_module._bucket(1_048_577)


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("manifest", "long_context_projector_manifest_invalid"),
        ("sidecar", "long_context_projector_manifest_sidecar_invalid"),
        ("file", "long_context_projector_file_invalid"),
    ],
)
def test_rejects_wrong_projector_hash_bindings(
    tmp_path: Path,
    kind: str,
    error: str,
) -> None:
    fixture = _project_fixture(tmp_path)
    expected = fixture.manifest_sha256
    if kind == "manifest":
        expected = "0" * 64
    elif kind == "sidecar":
        (fixture.root / "manifest.json.sha256").write_bytes(
            f"{'0' * 64}  manifest.json\n".encode("ascii")
        )
    else:
        path = fixture.root / "train/clean.jsonl"
        path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(LongContextPreflightError, match=error):
        _build_inventory(
            fixture,
            tmp_path / "inventory",
            expected_sha256=expected,
        )
    assert not (tmp_path / "inventory").exists()


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("heldout_content_read", True),
        ("heldout_content_emitted", True),
        ("claim_scope", "heldout_or_formal_evidence"),
    ],
)
def test_rejects_projector_manifest_heldout_or_claim_scope_forgery(
    tmp_path: Path,
    field: str,
    forged_value: object,
) -> None:
    fixture = _project_fixture(tmp_path)

    def forge_manifest(manifest: dict[str, Any]) -> None:
        manifest[field] = forged_value

    fixture = _resign_projector_manifest(fixture, forge_manifest)
    output = tmp_path / "inventory"
    with pytest.raises(
        LongContextPreflightError,
        match="long_context_projector_manifest_invalid",
    ):
        _build_inventory(fixture, output)
    assert not output.exists()


def test_rejects_symlinked_projector_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _project_fixture(tmp_path)
    partition = fixture.root / "train/clean.jsonl"
    target = partition.with_name("clean.real.jsonl")
    partition.replace(target)
    try:
        partition.symlink_to(target.name)
    except OSError:
        target.replace(partition)
        is_symlink = Path.is_symlink

        def report_partition_symlink(path: Path) -> bool:
            return path == partition or is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", report_partition_symlink)

    with pytest.raises(
        LongContextPreflightError, match="long_context_projector_file_invalid"
    ):
        _build_inventory(fixture, tmp_path / "inventory")


def test_rejects_symlinked_output_parent_when_supported(tmp_path: Path) -> None:
    fixture = _project_fixture(tmp_path)
    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(
        LongContextPreflightError,
        match="long_context_output_exists_or_overlaps_input",
    ):
        _build_inventory(fixture, linked_parent / "inventory")
    assert not (real_parent / "inventory").exists()


def test_rejects_output_nested_in_projector_artifact(tmp_path: Path) -> None:
    fixture = _project_fixture(tmp_path)
    output = fixture.root / "nested-inventory"
    with pytest.raises(
        LongContextPreflightError,
        match="long_context_output_exists_or_overlaps_input",
    ):
        _build_inventory(fixture, output)
    assert not output.exists()


def test_rejects_forbidden_segment_selection_after_resigning(
    tmp_path: Path,
) -> None:
    fixture = _project_fixture(tmp_path)

    def select_forbidden(rows: list[dict[str, Any]]) -> None:
        row = rows[0]
        forbidden = row["training_record"]["attention_targets"]["forbidden_block_ids"][
            0
        ]
        blocks = row["training_record"]["task_board"]["blocks"]
        segment = row["segment_plan"]["segments"][0]
        segment["source_block_id"] = forbidden
        segment["causal_order"] = next(
            index for index, block in enumerate(blocks) if block["id"] == forbidden
        )

    fixture = _resign_projector_file(
        fixture,
        "train/clean.jsonl",
        select_forbidden,
    )
    with pytest.raises(
        LongContextPreflightError,
        match="long_context_(segment_order_invalid|forbidden_selection)",
    ):
        _build_inventory(fixture, tmp_path / "inventory")


@pytest.mark.parametrize("forgery", ["source", "content", "visibility"])
def test_rejects_noisy_overlay_forgery_after_resigning(
    tmp_path: Path,
    forgery: str,
) -> None:
    fixture = _project_fixture(tmp_path)

    def forge_overlay(rows: list[dict[str, Any]]) -> None:
        row = rows[0]
        inner = row["training_record"]
        augmentation = row["augmentation"]
        if forgery == "source":
            augmentation["source_block_ids"][0] = inner["attention_targets"][
                "forbidden_block_ids"
            ][0]
            return
        overlay_id = augmentation["overlay_block_ids"][0]
        overlay = next(
            block
            for block in inner["task_board"]["blocks"]
            if block["id"] == overlay_id
        )
        if forgery == "content":
            overlay["content"] += " [forged]"
            return
        other_expert = next(
            expert for expert in STAGE_EXPERTS.values() if expert != row["expert"]
        )
        overlay["visible_to"] = [other_expert]

    fixture = _resign_projector_file(
        fixture,
        "train/noisy.jsonl",
        forge_overlay,
    )
    output = tmp_path / "inventory"
    with pytest.raises(
        LongContextPreflightError,
        match="long_context_augmentation_pair_invalid",
    ):
        _build_inventory(fixture, output)
    assert not output.exists()


def test_rejects_bundle_cross_split_after_resigning(tmp_path: Path) -> None:
    fixture = _project_fixture(tmp_path)
    train_bundle = _read_jsonl(fixture.root / "train/clean.jsonl")[0][
        "task_bundle_sha256"
    ]

    def cross_split(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            row["task_bundle_sha256"] = train_bundle
            row["segment_plan"]["bindings"]["task_bundle_sha256"] = train_bundle

    fixture = _resign_projector_file(
        fixture,
        "calibration/clean.jsonl",
        cross_split,
    )
    with pytest.raises(
        LongContextPreflightError, match="long_context_bundle_split_invalid"
    ):
        _build_inventory(fixture, tmp_path / "inventory")


def test_rejects_missing_role_after_resigning(tmp_path: Path) -> None:
    fixture = _project_fixture(tmp_path)
    fixture = _resign_projector_file(
        fixture,
        "calibration/clean.jsonl",
        lambda rows: rows.pop(),
    )
    with pytest.raises(
        LongContextPreflightError, match="long_context_role_group_invalid"
    ):
        _build_inventory(fixture, tmp_path / "inventory")


def test_counter_receives_only_selected_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _project_fixture(tmp_path)
    expected_calls: set[tuple[str, ...]] = set()
    selected_contents: set[str] = set()
    for relative in FIXED_FILES:
        for row in _read_jsonl(fixture.root / relative):
            by_id = {
                block["id"]: block["content"]
                for block in row["training_record"]["task_board"]["blocks"]
            }
            segments = row["segment_plan"]["segments"]
            ordered = tuple(by_id[item["source_block_id"]] for item in segments)
            expected_calls.add(ordered)
            selected_contents.update(ordered)
            shared = tuple(
                by_id[item["source_block_id"]]
                for item in segments
                if item["cache_scope"] != "expert_private_delta"
            )
            if len(shared) != len(ordered):
                expected_calls.add(shared)

    counter = SyntheticFixtureTokenCounter()
    calls: list[tuple[str, ...]] = []
    verify_called = False
    original_count = counter.count
    original_verify = counter.verify_unchanged

    def observe_count(ordered_segments: tuple[str, ...]) -> int:
        calls.append(tuple(ordered_segments))
        return original_count(ordered_segments)

    def observe_verify() -> None:
        nonlocal verify_called
        verify_called = True
        original_verify()

    monkeypatch.setattr(counter, "count", observe_count)
    monkeypatch.setattr(counter, "verify_unchanged", observe_verify)
    _build_inventory(fixture, tmp_path / "inventory", counter=counter)
    assert verify_called is True
    assert len(calls) == 20
    assert all(call in expected_calls for call in calls)
    assert all(item in selected_contents for call in calls for item in call)
    assert all(FORBIDDEN_SENTINEL not in item for call in calls for item in call)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("tokenizer_assets_sha256", "invalid"),
        ("tokenizer_label_source", "unbound_or_self_reported"),
    ],
)
def test_rejects_invalid_counter_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    forged_value: str,
) -> None:
    fixture = _project_fixture(tmp_path)
    counter = SyntheticFixtureTokenCounter()
    metadata = {**counter.metadata, field: forged_value}
    monkeypatch.setattr(counter, "_metadata", metadata)
    monkeypatch.setattr(
        counter,
        "_binding_sha256",
        preflight_module._sha256_value(metadata),
    )
    with pytest.raises(
        LongContextPreflightError,
        match="long_context_tokenizer_binding_invalid",
    ):
        _build_inventory(
            fixture,
            tmp_path / "inventory",
            counter=counter,
        )
    assert not (tmp_path / "inventory").exists()


def test_rejects_counter_outside_closed_implementations(tmp_path: Path) -> None:
    fixture = _project_fixture(tmp_path)
    with pytest.raises(
        LongContextPreflightError,
        match="long_context_tokenizer_binding_invalid",
    ):
        _build_inventory(fixture, tmp_path / "inventory", counter=object())
    assert not (tmp_path / "inventory").exists()


def test_local_tokenizer_assets_reject_model_weights_before_loading(
    tmp_path: Path,
) -> None:
    tokenizer_root = tmp_path / "tokenizer-with-weights"
    tokenizer_root.mkdir()
    (tokenizer_root / "model.safetensors").write_bytes(b"not-a-real-weight")

    with pytest.raises(
        LongContextPreflightError,
        match="long_context_tokenizer_assets_invalid",
    ):
        preflight_module.LocalTransformersTokenCounter(
            tokenizer_root,
            tokenizer_id="fixture-tokenizer",
            tokenizer_revision="fixture-revision",
            max_asset_bytes=1024,
        )


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=30,
    )
    assert completed.returncode == 0
    assert "--projector-manifest-sha256" in completed.stdout
    assert "--synthetic-fixture-tokenizer" in completed.stdout
    assert "Traceback" not in completed.stderr
