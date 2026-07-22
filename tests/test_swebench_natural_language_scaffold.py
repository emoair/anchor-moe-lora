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
import yaml

from anchor_mvp.swebench import natural_language_scaffold as scaffold_module
from anchor_mvp.swebench.natural_language_scaffold import (
    NaturalLanguageScaffoldConfig,
    NaturalLanguageScaffoldError,
    build_natural_language_scaffold,
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
CONFIG = ROOT / "configs/research/swebench_natural_language_scaffold_v1.yaml"
RECORD_SCHEMA = (
    ROOT / "configs/research/swebench_natural_language_scaffold_sidecar.schema.json"
)
MANIFEST_SCHEMA = (
    ROOT / "configs/research/swebench_natural_language_scaffold_manifest.schema.json"
)
SMOKE_SCHEMA = (
    ROOT
    / "configs/research/swebench_natural_language_scaffold_smoke_contract.schema.json"
)
SMOKE_CONFIG = (
    ROOT / "configs/research/swebench_natural_language_scaffold_smoke_v1.yaml"
)
BUILD_CLI = ROOT / "scripts/data/build_swebench_natural_language_scaffold.py"
AUDIT_CLI = ROOT / "scripts/data/audit_swebench_natural_language_scaffold_fixture.py"
SOURCE_FILES = (
    "train/clean.jsonl",
    "train/noisy.jsonl",
    "calibration/clean.jsonl",
)
FIXED_FILES = (
    "train/json_only.jsonl",
    "train/concise_rationale_plus_json.jsonl",
    "calibration/json_only.jsonl",
    "calibration/concise_rationale_plus_json.jsonl",
)
VARIANTS = {"json_only", "concise_rationale_plus_json"}
FORBIDDEN_SENTINEL = "[PASS]"
TOKEN_POSITION_KEYS = {
    "activation_token_ids",
    "boundary_token_index",
    "invocation_token_ids",
    "invocation_token_index",
    "position_ids",
    "token_ids",
    "token_index",
    "token_indices",
    "token_offset",
    "token_offsets",
}


@dataclass(frozen=True)
class ProjectedFixture:
    root: Path
    manifest_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert all(isinstance(value, dict) for value in values)
    return values


def _project_fixture(tmp_path: Path) -> ProjectedFixture:
    snapshot = _build_snapshot(tmp_path)
    output = tmp_path / "projector"
    result = project_taskboards(
        PROJECTOR_CONFIG,
        snapshot.root,
        snapshot.manifest_sha256,
        output,
    )
    return ProjectedFixture(output, str(result["manifest_sha256"]))


def _build(
    fixture: ProjectedFixture,
    output: Path,
    *,
    config: NaturalLanguageScaffoldConfig | Path = CONFIG,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    return build_natural_language_scaffold(
        config,
        fixture.root,
        expected_manifest_sha256 or fixture.manifest_sha256,
        output,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _rows(output: Path) -> list[dict[str, Any]]:
    return [row for relative in FIXED_FILES for row in _read_jsonl(output / relative)]


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(str(key) for key in value),
            *(key for item in value.values() for key in _walk_keys(item)),
        }
    if isinstance(value, list):
        return {key for item in value for key in _walk_keys(item)}
    return set()


def _resign_projector_file(
    fixture: ProjectedFixture,
    relative: str,
    mutate: Callable[[list[dict[str, Any]]], None],
) -> ProjectedFixture:
    path = fixture.root / relative
    rows = _read_jsonl(path)
    mutate(rows)
    data = b"".join(_canonical(row) + b"\n" for row in rows)
    path.write_bytes(data)

    manifest_path = fixture.root / "manifest.json"
    manifest = _read_json(manifest_path)
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    entry.update({"sha256": _sha256(data), "bytes": len(data), "records": len(rows)})
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = _sha256(manifest_bytes)
    (fixture.root / "manifest.json.sha256").write_bytes(
        f"{manifest_sha256}  manifest.json\n".encode("ascii")
    )
    return ProjectedFixture(fixture.root, manifest_sha256)


def _denied_source_bodies(fixture: ProjectedFixture) -> set[bytes]:
    denied: set[bytes] = set()
    for relative in SOURCE_FILES:
        for row in _read_jsonl(fixture.root / relative):
            inner = row["training_record"]
            by_id = {
                block["id"]: block["content"] for block in inner["task_board"]["blocks"]
            }
            for block_id in inner["attention_targets"]["forbidden_block_ids"]:
                denied.add(by_id[block_id].encode("utf-8"))
            denied.add(inner["target"]["answer"].encode("utf-8"))
    return denied


def _pair_normal_form(row: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {
        "record_id",
        "scaffold_variant",
        "concise_rationale_summary",
        "scaffold_text",
        "scaffold_text_sha256",
    }
    return {key: value for key, value in row.items() if key not in ignored}


def _is_closed_boundary_error(error: NaturalLanguageScaffoldError) -> bool:
    """The producer deliberately preserves authenticated projector error codes."""

    return error.code.startswith(("natural_language_scaffold_", "long_context_"))


def test_builds_deterministic_authenticated_body_free_twenty_record_fixture(
    tmp_path: Path,
) -> None:
    fixture = _project_fixture(tmp_path)
    first = tmp_path / "scaffold-first"
    second = tmp_path / "scaffold-second"
    result = _build(fixture, first)
    second_result = _build(fixture, second)

    assert result["records"] == second_result["records"] == 20
    assert result["pairs"] == second_result["pairs"] == 10
    assert result["provider_requests"] == second_result["provider_requests"] == 0
    assert _tree_bytes(first) == _tree_bytes(second)
    assert set(_tree_bytes(first)) == {
        *FIXED_FILES,
        "manifest.json",
        "manifest.json.sha256",
    }

    manifest_bytes = (first / "manifest.json").read_bytes()
    manifest_sha256 = _sha256(manifest_bytes)
    assert result["manifest_sha256"] == manifest_sha256
    assert (first / "manifest.json.sha256").read_bytes() == (
        f"{manifest_sha256}  manifest.json\n".encode("ascii")
    )
    manifest = json.loads(manifest_bytes)
    manifest_schema = _read_json(MANIFEST_SCHEMA)
    assert set(manifest) == set(manifest_schema["required"])
    nested_schema_names = {
        "input": "input",
        "producer": "producer",
        "architecture_contract": "architecture_contract",
        "visibility_contract": "visibility_contract",
        "route_activation_contract": "route_activation_contract",
        "cache_contract": "cache_contract",
        "adapter_control_contract": "adapter_control_contract",
        "serialization_contract": "serialization_contract",
        "counts": "counts",
        "smoke_contract": "smoke_binding",
    }
    for manifest_key, definition_name in nested_schema_names.items():
        assert set(manifest[manifest_key]) == set(
            manifest_schema["$defs"][definition_name]["required"]
        )
    assert all(
        set(item) == set(manifest_schema["$defs"]["file"]["required"])
        for item in manifest["files"]
    )
    implementation_sha256 = manifest["producer"]["implementation_sha256"]
    assert len(implementation_sha256) == 64
    assert all(char in "0123456789abcdef" for char in implementation_sha256)
    assert implementation_sha256 == _sha256(Path(scaffold_module.__file__).read_bytes())
    assert manifest["provider_requests"] == 0
    assert manifest["model_loads"] == 0
    assert manifest["gpu_requests"] == 0
    assert manifest["network_requests"] == 0
    assert manifest["canonical_gold_written"] is False
    assert manifest["heldout_written"] is False
    assert manifest["quality_validated"] is False
    assert manifest["execution_authorized"] is False
    assert manifest["counts"]["total"] == 20
    assert manifest["counts"]["pairs"] == 10
    assert [item["path"] for item in manifest["files"]] == list(FIXED_FILES)
    assert [item["records"] for item in manifest["files"]] == [5, 5, 5, 5]

    rows = _rows(first)
    assert len(rows) == 20
    assert all(row["provider_requests"] == 0 for row in rows)
    artifact_bytes = b"".join(_tree_bytes(first).values())
    assert FORBIDDEN_SENTINEL.encode("utf-8") not in artifact_bytes
    for denied_body in _denied_source_bodies(fixture):
        assert denied_body not in artifact_bytes
    assert not (TOKEN_POSITION_KEYS & _walk_keys(manifest))
    assert all(not (TOKEN_POSITION_KEYS & _walk_keys(row)) for row in rows)


def test_record_schema_and_scaffold_serialization_are_closed(tmp_path: Path) -> None:
    fixture = _project_fixture(tmp_path)
    output = tmp_path / "scaffold"
    _build(fixture, output)
    schema = _read_json(RECORD_SCHEMA)

    for row in _rows(output):
        expected_keys = set(schema["required"])
        if row["scaffold_variant"] == "concise_rationale_plus_json":
            expected_keys.add("concise_rationale_summary")
            assert 0 < len(row["concise_rationale_summary"].encode("utf-8")) <= 512
        else:
            assert "concise_rationale_summary" not in row
        assert set(row) == expected_keys
        assert row["schema_version"] == "anchor.natural-language-scaffold.v1"
        assert row["evaluation_status"] == "not_evaluated"
        assert row["quality_validated"] is False
        assert row["execution_authorized"] is False
        assert row["training_outcome_claimed"] is False
        assert row["adapter_control_labels"] == ["q_only", "q_plus_o", "wide_lora"]

        route = row["routing_json"]
        assert row["routing_json_sha256"] == _sha256(_canonical(route))
        assert route["role"] == row["stage"]
        assert route["expert"] == row["expert"]
        allowed = route["allowed_segment_refs"]
        evidence = route["evidence_segment_refs"]
        assert allowed
        assert all(ref in allowed for ref in evidence)
        assert all(
            set(ref)
            == {
                "segment_id",
                "source_block_id",
                "content_sha256",
                "causal_order",
                "cache_scope",
            }
            for ref in allowed
        )
        assert all(ref["cache_scope"] != "expert_private_delta" for ref in evidence)

        trigger = row["expert_trigger"]
        assert trigger["expert"] == row["expert"]
        assert trigger["trigger_text_sha256"] == _sha256(
            trigger["trigger_text"].encode("utf-8")
        )
        payload = {
            "routing_json": route,
            "tool_calls": row["tool_calls"],
            "tool_results": row["tool_results"],
            "expert_trigger": trigger,
        }
        payload_text = _canonical(payload).decode("utf-8")
        assert row["canonical_json_payload_sha256"] == _sha256(_canonical(payload))
        if row["scaffold_variant"] == "json_only":
            assert row["scaffold_text"] == payload_text
        else:
            assert row["scaffold_text"] == (
                row["concise_rationale_summary"] + "\n" + payload_text
            )
        assert row["scaffold_text_sha256"] == _sha256(
            row["scaffold_text"].encode("utf-8")
        )


def test_pair_bundle_split_noise_language_and_five_role_invariants(
    tmp_path: Path,
) -> None:
    fixture = _project_fixture(tmp_path)
    output = tmp_path / "scaffold"
    _build(fixture, output)
    rows = _rows(output)
    pairs: dict[str, list[dict[str, Any]]] = {}
    bundle_splits: dict[str, set[str]] = {}
    role_groups: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
        bundle_splits.setdefault(row["task_bundle_sha256"], set()).add(row["split"])
        role_groups.setdefault((row["split"], row["task_bundle_sha256"]), set()).add(
            (row["stage"], row["expert"])
        )
        assert row["source_variant"] == (
            "noisy" if row["split"] == "train" else "clean"
        )
        assert row["language"] == ("en" if row["split"] == "train" else "zh-CN")

    assert len(pairs) == 10
    for paired in pairs.values():
        assert len(paired) == 2
        assert {row["scaffold_variant"] for row in paired} == VARIANTS
        assert _pair_normal_form(paired[0]) == _pair_normal_form(paired[1])
    assert len(bundle_splits) == 2
    assert all(len(splits) == 1 for splits in bundle_splits.values())
    assert len(role_groups) == 2
    expected_roles = {(stage, STAGE_EXPERTS[stage]) for stage in STAGES}
    assert all(roles == expected_roles for roles in role_groups.values())


def test_route_boundary_is_two_request_commit_then_frozen_base_reencode(
    tmp_path: Path,
) -> None:
    fixture = _project_fixture(tmp_path)
    output = tmp_path / "scaffold"
    _build(fixture, output)

    for row in _rows(output):
        route = row["route_boundary"]
        assert route["semantics"] == "explicit_two_request_commit_boundary"
        assert (
            route["planner_request_phase"] == "rationale_route_and_sentinel_candidate"
        )
        assert route["validation_required"] is True
        assert route["commit_required"] is True
        assert route["commit_promotes_text_only"] is True
        assert route["planner_private_tail_kv_transfer_allowed"] is False
        assert route["committed_scaffold_reencode_required"] is True
        assert route["committed_scaffold_reencode_producer"] == "frozen_base"
        assert route["committed_scaffold_reencode_adapter_state"] == "off"
        assert route["expert_request_phase"] == "next_request"
        assert route["expert_request_requires_committed_scaffold_as_input"] is True
        assert route["token_boundary_status"] == "tokenizer_binding_required"

        alora = row["alora_invocation"]
        assert alora["optional"] is True
        assert alora["activation_semantics"] == "next_request_input_activation_only"
        assert alora["invocation_scan_scope"] == "new_request_input_tokens_only"
        assert alora["same_request_activation_allowed"] is False
        assert alora["mid_request_generated_activation_allowed"] is False
        assert alora["mid_request_generated_trigger_switch_claimed"] is False
        assert alora["explicit_commit_required"] is True
        assert alora["adapter_available"] is False
        assert alora["adapter_loaded"] is False
        assert alora["activation_executed"] is False
        assert alora["cross_attention_q_reader_claimed"] is False
        assert alora["physical_shared_kv_claimed"] is False
        assert alora["trigger_text"] == row["expert_trigger"]["trigger_text"]
        assert (
            alora["trigger_text_sha256"] == row["expert_trigger"]["trigger_text_sha256"]
        )


def test_cache_contract_never_hands_planner_private_kv_to_expert(
    tmp_path: Path,
) -> None:
    fixture = _project_fixture(tmp_path)
    output = tmp_path / "scaffold"
    _build(fixture, output)
    for row in _rows(output):
        cache = row["cache_metadata"]
        assert cache["prefix_lineage_sha256"] == row["terminal_prefix_lineage_sha256"]
        assert cache["adapter_state_on_prefix"] == "off"
        assert cache["adapter_state_after_boundary"] == "expert_only"
        assert cache["private_tail_kv_required"] is True
        assert cache["full_generation_kv_shared_claimed"] is False
        assert cache["exact_reuse_scope"] == "identical_ordered_prefix_lineage_only"
        assert cache["cache_identity_status"] == "identity_unbound"
        assert cache["exact_cache_reuse_enabled"] is False
        assert cache["reuse_savings_tokens"] == 0
        assert cache["planner_private_tail_kv_reused_by_expert"] is False
        assert cache["physical_kv_tensor_emitted"] is False
        assert cache["committed_scaffold_reencode_executed"] is False
        assert cache["downstream_immutable_segment_emitted"] is False


def test_config_and_smoke_contract_are_non_authorizing_and_model_free() -> None:
    config, inventory = NaturalLanguageScaffoldConfig.load(CONFIG)
    assert config is not None
    assert inventory
    assert all(path.is_absolute() for path in inventory)
    smoke = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    assert (
        smoke["schema_version"] == "anchor.natural-language-scaffold-smoke-contract.v1"
    )
    assert smoke["model_artifact"]["basename"] == ("qwen2.5-1.5b-instruct-q4_k_m.gguf")
    assert smoke["model_artifact"]["format"] == "gguf"
    assert smoke["model_artifact"]["quantization"] == "q4_k_m"
    assert smoke["model_artifact"]["trainable_weights"] is False
    assert smoke["model_artifact"]["training_use_allowed"] is False
    assert smoke["current_execution"]["model_loaded"] is False
    assert smoke["current_execution"]["provider_requests"] == 0
    assert smoke["current_execution"]["network_requests"] == 0
    capability = smoke["runtime_capability"]
    assert capability["cross_attention_q_reader_implemented"] is False
    assert capability["physical_shared_kv_implemented"] is False
    assert capability["zero_copy_kv_claimed"] is False
    assert capability["activation_semantics"] == "next_request_input_activation_only"
    assert capability["mid_request_generated_trigger_switch_claimed"] is False
    assert smoke["prohibited_claims"]["quality_validated"] is False
    assert smoke["prohibited_claims"]["gguf_is_trainable_weights"] is False
    assert _read_json(SMOKE_SCHEMA)["additionalProperties"] is False


@pytest.mark.parametrize(
    ("kind", "relative"),
    [
        ("expected", ""),
        ("manifest_sidecar", "manifest.json.sha256"),
        ("partition", "train/noisy.jsonl"),
    ],
)
def test_rejects_wrong_projector_hash_bindings(
    tmp_path: Path,
    kind: str,
    relative: str,
) -> None:
    fixture = _project_fixture(tmp_path)
    expected = fixture.manifest_sha256
    if kind == "expected":
        expected = "0" * 64
    elif kind == "manifest_sidecar":
        (fixture.root / relative).write_bytes(
            f"{'0' * 64}  manifest.json\n".encode("ascii")
        )
    else:
        path = fixture.root / relative
        path.write_bytes(path.read_bytes() + b" ")

    output = tmp_path / "scaffold"
    with pytest.raises(NaturalLanguageScaffoldError) as caught:
        _build(fixture, output, expected_manifest_sha256=expected)
    assert _is_closed_boundary_error(caught.value)
    assert not output.exists()


def test_rejects_forbidden_segment_selection_even_when_partition_is_resigned(
    tmp_path: Path,
) -> None:
    fixture = _project_fixture(tmp_path)

    def mutate(rows: list[dict[str, Any]]) -> None:
        row = rows[0]
        inner = row["training_record"]
        forbidden_id = inner["attention_targets"]["forbidden_block_ids"][0]
        blocks = inner["task_board"]["blocks"]
        segment = row["segment_plan"]["segments"][0]
        segment["source_block_id"] = forbidden_id
        segment["causal_order"] = next(
            index for index, block in enumerate(blocks) if block["id"] == forbidden_id
        )

    fixture = _resign_projector_file(fixture, "train/noisy.jsonl", mutate)
    output = tmp_path / "scaffold"
    with pytest.raises(NaturalLanguageScaffoldError) as caught:
        _build(fixture, output)
    assert _is_closed_boundary_error(caught.value)
    assert not output.exists()


def test_rejects_cross_split_bundle_even_when_partition_is_resigned(
    tmp_path: Path,
) -> None:
    fixture = _project_fixture(tmp_path)
    train_bundle = _read_jsonl(fixture.root / "train/noisy.jsonl")[0][
        "task_bundle_sha256"
    ]

    def mutate(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            row["task_bundle_sha256"] = train_bundle
            row["segment_plan"]["bindings"]["task_bundle_sha256"] = train_bundle

    fixture = _resign_projector_file(fixture, "calibration/clean.jsonl", mutate)
    output = tmp_path / "scaffold"
    with pytest.raises(NaturalLanguageScaffoldError) as caught:
        _build(fixture, output)
    assert _is_closed_boundary_error(caught.value)
    assert not output.exists()


def test_rejects_symlinked_source_partition_when_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _project_fixture(tmp_path)
    partition = fixture.root / "train/noisy.jsonl"
    target = partition.with_name("noisy.real.jsonl")
    partition.replace(target)
    try:
        partition.symlink_to(target.name)
    except (NotImplementedError, OSError):
        target.replace(partition)
        original_is_symlink = Path.is_symlink

        def report_partition_symlink(path: Path) -> bool:
            return path == partition or original_is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", report_partition_symlink)

    output = tmp_path / "scaffold"
    with pytest.raises(NaturalLanguageScaffoldError) as caught:
        _build(fixture, output)
    assert _is_closed_boundary_error(caught.value)
    assert not output.exists()


def test_rejects_symlinked_output_parent_when_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _project_fixture(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except (NotImplementedError, OSError):
        linked_parent.mkdir()
        original_is_symlink = Path.is_symlink

        def report_output_parent_symlink(path: Path) -> bool:
            return path == linked_parent or original_is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", report_output_parent_symlink)

    output = linked_parent / "scaffold"
    with pytest.raises(NaturalLanguageScaffoldError) as caught:
        _build(fixture, output)
    assert _is_closed_boundary_error(caught.value)
    assert not (real_parent / "scaffold").exists()
    assert not output.exists()


def test_rejects_output_inside_projector_artifact(tmp_path: Path) -> None:
    fixture = _project_fixture(tmp_path)
    output = fixture.root / "nested-scaffold"
    with pytest.raises(NaturalLanguageScaffoldError) as caught:
        _build(fixture, output)
    assert _is_closed_boundary_error(caught.value)
    assert not output.exists()


def test_toctou_change_before_publish_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _project_fixture(tmp_path)
    output = tmp_path / "scaffold"
    original = scaffold_module._lc._verify_inventory_unchanged
    mutated = False

    def mutate_then_verify(inventory: Mapping[Path, Any]) -> None:
        nonlocal mutated
        source = next(
            (
                path
                for path in inventory
                if path.name == "noisy.jsonl" and fixture.root in path.parents
            ),
            None,
        )
        if source is not None and not mutated:
            mutated = True
            source.write_bytes(source.read_bytes() + b" ")
        original(inventory)

    monkeypatch.setattr(
        scaffold_module._lc,
        "_verify_inventory_unchanged",
        mutate_then_verify,
    )
    with pytest.raises(Exception) as caught:
        _build(fixture, output)
    assert mutated is True
    code = getattr(caught.value, "code", str(caught.value))
    assert "input_changed" in code
    assert not output.exists()


def test_audit_cli_accepts_fixture_and_rejects_tampered_file(tmp_path: Path) -> None:
    fixture = _project_fixture(tmp_path)
    output = tmp_path / "scaffold"
    result = _build(fixture, output)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    command = [
        sys.executable,
        str(AUDIT_CLI),
        "--config",
        str(CONFIG),
        "--artifact-dir",
        str(output),
        "--manifest-sha256",
        result["manifest_sha256"],
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["audit_passed"] is True
    assert "Traceback" not in completed.stderr

    path = output / "train/json_only.jsonl"
    path.write_bytes(path.read_bytes() + b" ")
    tampered = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert tampered.returncode == 2
    assert "natural_language_scaffold_audit_file_hash_invalid" in tampered.stderr
    assert "Traceback" not in tampered.stderr


@pytest.mark.parametrize(
    ("cli", "expected_flag"),
    [
        (BUILD_CLI, "--projector-manifest-sha256"),
        (AUDIT_CLI, "--manifest-sha256"),
    ],
)
def test_cli_help(cli: Path, expected_flag: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(cli), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=30,
    )
    assert completed.returncode == 0
    assert expected_flag in completed.stdout
    assert "Traceback" not in completed.stderr


def test_published_schemas_are_closed_and_manifest_requires_zero_requests() -> None:
    record_schema = _read_json(RECORD_SCHEMA)
    manifest_schema = _read_json(MANIFEST_SCHEMA)
    smoke_schema = _read_json(SMOKE_SCHEMA)
    assert record_schema["additionalProperties"] is False
    assert manifest_schema["additionalProperties"] is False
    assert smoke_schema["additionalProperties"] is False
    assert manifest_schema["properties"]["provider_requests"] == {"const": 0}
    assert manifest_schema["properties"]["model_loads"] == {"const": 0}
    assert manifest_schema["properties"]["gpu_requests"] == {"const": 0}
    assert manifest_schema["properties"]["network_requests"] == {"const": 0}
