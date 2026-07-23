from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from anchor_mvp.research.query_specialization import (
    QuerySpecializationError,
    block_attention_auxiliary_loss,
    build_training_view,
    canonical_query_training_record,
    canonical_taskboard_sidecar,
    dataset_summary,
    load_taskboard_sidecar_dataset,
    lora_target_modules,
    parse_query_training_record,
    parse_taskboard_sidecar,
    query_training_record_sha256,
    taskboard_sidecar_sha256,
    validate_paired_records,
    validate_source_task_partition,
    validate_taskboard_sidecar_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECTOR_FIXTURE = ROOT / "fixtures" / "research" / "taskboard_projector"
QUERY_SPECIALIZATION_SCRIPT = (
    ROOT / "scripts" / "research" / "train_query_specialization_mvp.py"
)


def _load_query_specialization_script():
    module_name = "query_specialization_formal_semantics_test"
    spec = importlib.util.spec_from_file_location(
        module_name, QUERY_SPECIALIZATION_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_release_authorization_keeps_research_proxy_distinct_from_formal() -> None:
    module = _load_query_specialization_script()

    fields = module._release_authorization_fields(
        {
            "research_proxy_training_authorized": True,
            "formal_training_authorized": False,
        }
    )

    assert fields == {
        "research_proxy_training_authorized": True,
        "formal_training_authorized": False,
    }


def test_release_authorization_rejects_formal_promotion() -> None:
    module = _load_query_specialization_script()

    with pytest.raises(
        QuerySpecializationError,
        match="research_proxy_only release cannot authorize formal training",
    ):
        module._release_authorization_fields(
            {
                "research_proxy_training_authorized": True,
                "formal_training_authorized": True,
            }
        )


def test_dry_run_reports_proxy_and_formal_authority_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_query_specialization_script()
    config = ROOT / "configs" / "research" / "query_specialization_mvp.yaml"

    assert module.main(["--config", str(config), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]

    assert plan["research_proxy_training_authorized"] is False
    assert plan["formal_training_authorized"] is False
    assert (
        plan["release_lock_validation"]["research_proxy_training_authorized"] is False
    )
    assert plan["release_lock_validation"]["formal_training_authorized"] is False


def _copy_projector_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "taskboard_projector"
    shutil.copytree(PROJECTOR_FIXTURE, destination)
    return destination


def _rewrite_manifest(fixture: Path, manifest: dict) -> None:
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    (fixture / "manifest.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (fixture / "manifest.json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="utf-8", newline="\n"
    )


def _partition_with_replaced_answer(payload: bytes, replacement: str) -> bytes:
    lines = payload.decode("utf-8").splitlines()
    first = json.loads(lines[0])
    first["training_record"]["target"]["answer"] = replacement
    lines[0] = json.dumps(
        first,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _open_from_bytes(
    payload: bytes, mode: str, encoding: str | None
) -> io.BytesIO | io.StringIO:
    if "b" in mode:
        return io.BytesIO(payload)
    return io.StringIO(payload.decode(encoding or "utf-8"))


def _raw_record(*, variant: str = "clean", record_id: str | None = None) -> dict:
    distractors = ["b-noise"] if variant == "noisy" else []
    blocks = [
        {
            "id": "b-required",
            "kind": "requirement",
            "content": "Build a calculator.",
            "commit_state": "committed",
            "visible_to": ["all"],
        },
        {
            "id": "b-forbidden",
            "kind": "review",
            "content": "FORBIDDEN-CONTENT-MUST-NOT-RENDER",
            "commit_state": "verified",
            "visible_to": ["all"],
        },
    ]
    if variant == "noisy":
        blocks.append(
            {
                "id": "b-noise",
                "kind": "history",
                "content": "Unrelated legacy attempt.",
                "commit_state": "candidate",
                "visible_to": ["all"],
            }
        )
    return {
        "schema_version": "anchor.query-specialization.v1",
        "id": record_id or f"record-{variant}",
        "pair_id": "pair-1",
        "variant": variant,
        "language": "zh-CN",
        "split": "train",
        "role": "planner",
        "task_board": {
            "task_id": "task-1",
            "generation": 1,
            "blocks": blocks,
        },
        "attention_targets": {
            "relevant_block_ids": ["b-required"],
            "distractor_block_ids": distractors,
            "forbidden_block_ids": ["b-forbidden"],
        },
        "target": {
            "action": "PLAN",
            "answer": "Delegate implementation to the builder.",
            "selected_block_ids": ["b-required"],
        },
    }


def test_valid_record_parses_into_typed_contract() -> None:
    record = parse_query_training_record(_raw_record(), source="fixture")

    assert record.record_id == "record-clean"
    assert record.pair_id == "pair-1"
    assert record.role == "planner"
    assert record.task_id == "task-1"
    assert record.generation == 1
    assert record.targets.relevant == ("b-required",)
    assert record.targets.forbidden == ("b-forbidden",)
    assert tuple(block.block_id for block in record.blocks) == (
        "b-required",
        "b-forbidden",
    )


def test_unknown_attention_block_is_rejected() -> None:
    raw = _raw_record()
    raw["attention_targets"]["relevant_block_ids"] = ["missing-block"]
    raw["target"]["selected_block_ids"] = ["missing-block"]

    with pytest.raises(QuerySpecializationError, match="references unknown blocks"):
        parse_query_training_record(raw)


def test_attention_target_sets_must_be_pairwise_disjoint() -> None:
    raw = _raw_record(variant="noisy")
    raw["attention_targets"]["distractor_block_ids"] = ["b-required"]

    with pytest.raises(QuerySpecializationError, match="pairwise disjoint"):
        parse_query_training_record(raw)


def test_forbidden_blocks_are_hard_excluded_from_training_view() -> None:
    view = build_training_view(parse_query_training_record(_raw_record()))

    assert "b-required" in view.visible_block_ids
    assert "b-forbidden" not in view.visible_block_ids
    assert "FORBIDDEN-CONTENT-MUST-NOT-RENDER" not in view.prompt
    assert view.relevant_mask == (True,)
    assert view.distractor_mask == (False,)
    assert json.loads(view.prompt)["role"] == "planner"


def test_clean_and_noisy_records_form_one_counterfactual_pair() -> None:
    clean = parse_query_training_record(_raw_record())
    noisy = parse_query_training_record(
        _raw_record(variant="noisy", record_id="record-noisy")
    )

    assert validate_paired_records([clean, noisy]) == {"pairs": 1, "records": 2}
    assert clean.target_output == noisy.target_output
    assert noisy.targets.distractors == ("b-noise",)


@pytest.mark.parametrize(
    "records, message",
    [
        (lambda clean, noisy: [clean], "exactly one clean and one noisy"),
        (
            lambda clean, noisy: [clean, copy.copy(clean)],
            "duplicate query-specialization record id",
        ),
    ],
)
def test_malformed_counterfactual_pairs_are_rejected(records, message: str) -> None:
    clean = parse_query_training_record(_raw_record())
    noisy = parse_query_training_record(
        _raw_record(variant="noisy", record_id="record-noisy")
    )

    with pytest.raises(QuerySpecializationError, match=message):
        validate_paired_records(records(clean, noisy))


def test_unknown_target_field_is_rejected() -> None:
    raw = _raw_record()
    raw["target"]["untrusted_teacher_note"] = "must not enter the target"

    with pytest.raises(QuerySpecializationError, match="unknown fields"):
        parse_query_training_record(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update({"unexpected": 1}),
        lambda raw: raw["task_board"].update({"unexpected": 1}),
        lambda raw: raw["task_board"]["blocks"][0].update({"unexpected": 1}),
        lambda raw: raw["attention_targets"].update({"unexpected": 1}),
    ],
)
def test_unknown_fields_are_rejected_at_every_object_level(mutate) -> None:
    raw = _raw_record()
    mutate(raw)

    with pytest.raises(QuerySpecializationError, match="unknown fields"):
        parse_query_training_record(raw)


@pytest.mark.parametrize(
    "remove",
    [
        lambda raw: raw["task_board"]["blocks"][0].pop("visible_to"),
        lambda raw: raw["attention_targets"].pop("distractor_block_ids"),
        lambda raw: raw["attention_targets"].pop("forbidden_block_ids"),
    ],
)
def test_schema_required_lists_cannot_be_omitted(remove) -> None:
    raw = _raw_record()
    remove(raw)

    with pytest.raises(QuerySpecializationError, match="must be a list"):
        parse_query_training_record(raw)


def test_target_selection_must_exactly_match_relevant_blocks() -> None:
    raw = _raw_record()
    raw["target"]["selected_block_ids"] = ["b-forbidden"]

    with pytest.raises(QuerySpecializationError, match="must exactly match"):
        parse_query_training_record(raw)


def test_lora_profiles_are_explicit_and_unknown_profiles_fail_closed() -> None:
    assert lora_target_modules("q_only") == ("q_proj",)
    assert lora_target_modules("q_o") == ("q_proj", "o_proj")
    assert lora_target_modules("q_o_mlp") == (
        "q_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    with pytest.raises(QuerySpecializationError, match="unknown LoRA profile"):
        lora_target_modules("q_k_v_everything")


def test_block_attention_auxiliary_loss_runs_on_cpu_and_has_gradient() -> None:
    torch = pytest.importorskip("torch")
    attention = torch.tensor(
        [[[[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    relevant = torch.tensor([[True, False, False]])
    distractor = torch.tensor([[False, True, False]])

    loss, metrics = block_attention_auxiliary_loss(
        attention,
        relevant,
        distractor,
        distractor_weight=0.25,
    )

    expected = -math.log(0.8) + 0.25 * 0.1
    assert loss.item() == pytest.approx(expected)
    assert metrics["relevant_mass"].item() == pytest.approx(0.8)
    assert metrics["distractor_mass"].item() == pytest.approx(0.1)
    loss.backward()
    assert attention.grad is not None
    assert bool(torch.isfinite(attention.grad).all())


def test_attention_loss_supervises_every_relevant_block_uniformly() -> None:
    torch = pytest.importorskip("torch")
    balanced = torch.tensor([[0.4, 0.4, 0.1, 0.1]], requires_grad=True)
    collapsed = torch.tensor([[0.79, 0.01, 0.1, 0.1]], requires_grad=True)
    relevant = torch.tensor([[True, True, False, False]])
    distractor = torch.tensor([[False, False, True, False]])

    balanced_loss, balanced_metrics = block_attention_auxiliary_loss(
        balanced, relevant, distractor, distractor_weight=0.25
    )
    collapsed_loss, collapsed_metrics = block_attention_auxiliary_loss(
        collapsed, relevant, distractor, distractor_weight=0.25
    )

    assert balanced_metrics["relevant_mass"].item() == pytest.approx(0.8)
    assert collapsed_metrics["relevant_mass"].item() == pytest.approx(0.8)
    assert balanced_metrics["distractor_mass"].item() == pytest.approx(0.1)
    assert collapsed_loss.item() > balanced_loss.item()
    assert balanced_loss.item() == pytest.approx(-math.log(0.4) + 0.25 * 0.1)


@pytest.mark.parametrize(
    "attention,relevant,distractor,message",
    [
        ([0.5, 0.5], [False, False], [True, False], "relevant blocks"),
        ([float("nan"), 1.0], [True, False], [False, True], "finite"),
        ([-0.1, 1.1], [True, False], [False, True], "non-negative"),
        ([0.2, 0.2], [True, False], [False, True], "sum to one"),
        ([0.5, 0.5], [True, False], [True, False], "disjoint"),
    ],
)
def test_attention_loss_rejects_invalid_probabilities_and_masks(
    attention, relevant, distractor, message: str
) -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(QuerySpecializationError, match=message):
        block_attention_auxiliary_loss(
            torch.tensor([attention]),
            torch.tensor([relevant]),
            torch.tensor([distractor]),
        )


def test_attention_loss_requires_positive_epsilon() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(QuerySpecializationError, match="epsilon must be positive"):
        block_attention_auxiliary_loss(
            torch.tensor([[1.0]]),
            torch.tensor([[True]]),
            torch.tensor([[False]]),
            epsilon=0.0,
        )


def test_official_projector_fixture_loads_with_fixed_three_partition_contract() -> None:
    sidecars, manifest, summary = load_taskboard_sidecar_dataset(PROJECTOR_FIXTURE)

    assert len(sidecars) == 15
    assert manifest["canonical_gold_written"] is False
    assert summary["source_tasks_by_split"] == {"calibration": 1, "train": 1}
    assert summary["by_split"] == {"calibration": 5, "train": 10}
    assert summary["by_variant"] == {"clean": 10, "noisy": 5}
    assert summary["train_pairs"] == 5
    assert summary["segment_references"] == manifest["counts"]["segment_references"]
    assert summary["unique_segments"] == manifest["counts"]["unique_segments"]
    assert (
        summary["unique_segments_by_cache_scope"]
        == manifest["counts"]["unique_segments_by_cache_scope"]
    )
    assert summary["segment_plan_cross_bindings_validated"] is True
    assert summary["by_expert"] == {
        "frontend_gen": 3,
        "frontend_review": 3,
        "planner": 3,
        "security_gate": 3,
        "tool_policy": 3,
    }
    validation_summary = validate_source_task_partition(sidecars)
    assert {
        key: value
        for key, value in summary.items()
        if key not in {"manifest_sha256", "authenticated_file_sha256"}
    } == validation_summary
    assert (
        summary["authenticated_file_sha256"]["manifest.json"]
        == summary["manifest_sha256"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "split_group_key",
        "task_id_cross_binding_key",
        "all_five_role_views_same_split",
    ],
)
def test_manifest_requires_task_bundle_grouping_fields(
    tmp_path: Path, field: str
) -> None:
    fixture = _copy_projector_fixture(tmp_path)
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    del manifest[field]
    _rewrite_manifest(fixture, manifest)

    with pytest.raises(QuerySpecializationError, match="missing fields"):
        load_taskboard_sidecar_dataset(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split_group_key", "source_gold_record_id"),
        ("task_id_cross_binding_key", "training_record.id"),
        ("all_five_role_views_same_split", False),
    ],
)
def test_manifest_rejects_changed_task_bundle_grouping_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    fixture = _copy_projector_fixture(tmp_path)
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    manifest[field] = value
    _rewrite_manifest(fixture, manifest)

    with pytest.raises(QuerySpecializationError, match="grouping contract changed"):
        load_taskboard_sidecar_dataset(fixture)


def test_manifest_requires_producer_manifest_schema_hash(tmp_path: Path) -> None:
    fixture = _copy_projector_fixture(tmp_path)
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    del manifest["producer"]["manifest_schema_sha256"]
    _rewrite_manifest(fixture, manifest)

    with pytest.raises(QuerySpecializationError, match="missing fields"):
        load_taskboard_sidecar_dataset(fixture)


def test_expected_manifest_schema_hash_is_bound_fail_closed(tmp_path: Path) -> None:
    fixture = _copy_projector_fixture(tmp_path)
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    manifest["producer"]["manifest_schema_sha256"] = "0" * 64
    _rewrite_manifest(fixture, manifest)
    expected = hashlib.sha256(
        (
            ROOT / "configs/research/taskboard_projector_manifest.schema.json"
        ).read_bytes()
    ).hexdigest()

    with pytest.raises(QuerySpecializationError, match="manifest schema hash mismatch"):
        load_taskboard_sidecar_dataset(
            fixture,
            expected_manifest_schema_sha256=expected,
        )


def test_partition_hash_and_parser_share_one_bytes_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _copy_projector_fixture(tmp_path)
    partition_path = (fixture / "train" / "clean.jsonl").resolve()
    authenticated = partition_path.read_bytes()
    unauthenticated_answer = "UNAUTHENTICATED-PARTITION-SNAPSHOT"
    replaced = _partition_with_replaced_answer(authenticated, unauthenticated_answer)
    original_open = Path.open
    target_open_count = 0

    def racing_open(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ):
        nonlocal target_open_count
        if self.resolve() == partition_path:
            target_open_count += 1
            payload = authenticated if target_open_count == 1 else replaced
            return _open_from_bytes(payload, mode, encoding)
        return original_open(
            self,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "open", racing_open)
    try:
        sidecars, _, _ = load_taskboard_sidecar_dataset(fixture)
    except QuerySpecializationError:
        # A loader that deliberately double-checks the path may reject the
        # detected change, but it must never return records parsed from it.
        assert target_open_count >= 2
        return

    assert target_open_count == 1
    assert all(
        json.loads(sidecar.training_record.target_output)["answer"]
        != unauthenticated_answer
        for sidecar in sidecars
    )


def test_manifest_parser_and_sha_sidecar_share_one_bytes_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _copy_projector_fixture(tmp_path)
    manifest_path = (fixture / "manifest.json").resolve()
    authenticated = manifest_path.read_bytes()
    authenticated_manifest = json.loads(authenticated)
    replaced_manifest = copy.deepcopy(authenticated_manifest)
    replaced_manifest["input"]["snapshot_manifest_path"] = "untrusted.json"
    replaced = json.dumps(
        replaced_manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    original_open = Path.open
    target_open_count = 0

    def racing_open(
        self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ):
        nonlocal target_open_count
        if self.resolve() == manifest_path:
            target_open_count += 1
            # The dangerous ordering for the old loader is to parse replaced
            # bytes first, then authenticate the original path contents.
            payload = replaced if target_open_count == 1 else authenticated
            return _open_from_bytes(payload, mode, encoding)
        return original_open(
            self,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "open", racing_open)
    try:
        _, manifest, _ = load_taskboard_sidecar_dataset(fixture)
    except QuerySpecializationError as exc:
        assert "manifest SHA-256 sidecar" in str(exc)
        return

    assert target_open_count == 1
    assert (
        manifest["input"]["snapshot_manifest_path"]
        == authenticated_manifest["input"]["snapshot_manifest_path"]
    )


def test_manifest_sha256_sidecar_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _copy_projector_fixture(tmp_path)
    (fixture / "manifest.json.sha256").write_text(
        f"{'0' * 64}  manifest.json\n", encoding="utf-8"
    )

    with pytest.raises(QuerySpecializationError, match="manifest SHA-256 sidecar"):
        load_taskboard_sidecar_dataset(fixture)


def test_manifest_sha256_sidecar_is_mandatory(tmp_path: Path) -> None:
    fixture = _copy_projector_fixture(tmp_path)
    (fixture / "manifest.json.sha256").unlink()

    with pytest.raises(QuerySpecializationError) as exc_info:
        load_taskboard_sidecar_dataset(fixture)
    assert "SHA-256 sidecar" in str(exc_info.value)


@pytest.mark.parametrize(
    "declaration_template",
    [
        "{sha}  other.json\n",
        "{sha}\n",
        "{sha} manifest.json\n",
    ],
    ids=["wrong-filename", "bare-hash", "one-space-separator"],
)
def test_manifest_sha256_sidecar_rejects_nonstandard_declarations(
    tmp_path: Path, declaration_template: str
) -> None:
    fixture = _copy_projector_fixture(tmp_path)
    manifest_path = fixture / "manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (fixture / "manifest.json.sha256").write_text(
        declaration_template.format(sha=digest), encoding="utf-8"
    )

    with pytest.raises(QuerySpecializationError, match="manifest SHA-256 sidecar"):
        load_taskboard_sidecar_dataset(fixture)


def test_outer_sidecar_keeps_provenance_outside_inner_record() -> None:
    sidecars, _, _ = load_taskboard_sidecar_dataset(PROJECTOR_FIXTURE)
    sidecar = sidecars[0]
    inner = canonical_query_training_record(sidecar.training_record)
    outer = canonical_taskboard_sidecar(sidecar)

    assert "provenance" not in inner
    assert outer["training_record"] == inner
    assert outer["id"] == inner["id"]
    assert query_training_record_sha256(
        sidecar.training_record
    ) != taskboard_sidecar_sha256(sidecar)


def test_sidecar_parser_rejects_wrapper_inner_and_augmentation_mismatches() -> None:
    raw = json.loads(
        (PROJECTOR_FIXTURE / "train" / "clean.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    mismatch = copy.deepcopy(raw)
    mismatch["training_record"]["id"] = "different-id"
    with pytest.raises(QuerySpecializationError, match="wrapper/training_record"):
        parse_taskboard_sidecar(mismatch)

    augmented_clean = copy.deepcopy(raw)
    augmented_clean["augmentation"]["overlay_block_ids"] = ["fake-overlay"]
    with pytest.raises(QuerySpecializationError, match="clean sidecars"):
        parse_taskboard_sidecar(augmented_clean)

    wrong_stage = copy.deepcopy(raw)
    wrong_stage["expert"] = "security_gate"
    with pytest.raises(QuerySpecializationError, match="must map to expert"):
        parse_taskboard_sidecar(wrong_stage)


def test_v1_sidecar_and_manifest_are_rejected_with_explicit_version_policy(
    tmp_path: Path,
) -> None:
    raw = json.loads(
        (PROJECTOR_FIXTURE / "train" / "clean.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    raw["schema_version"] = "anchor.swebench-taskboard-sidecar.v1"
    with pytest.raises(
        QuerySpecializationError, match="unsupported sidecar schema_version"
    ):
        parse_taskboard_sidecar(raw)

    fixture = _copy_projector_fixture(tmp_path)
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "anchor.swebench-taskboard-projector-manifest.v1"
    _rewrite_manifest(fixture, manifest)
    with pytest.raises(
        QuerySpecializationError,
        match="unsupported projector manifest schema_version",
    ):
        load_taskboard_sidecar_dataset(fixture)


@pytest.mark.parametrize(
    "field",
    [
        "task_bundle_sha256",
        "task_id",
        "base_task_board_sha256",
        "config_sha256",
        "sidecar_schema_sha256",
        "segment_plan_schema_sha256",
        "source_gold_sha256",
        "source_gold_file_sha256",
        "source_snapshot_sha256",
        "source_snapshot_manifest_sha256",
        "split",
        "stage",
        "expert",
        "variant",
    ],
)
def test_segment_plan_bindings_are_strictly_cross_bound(field: str) -> None:
    raw = json.loads(
        (PROJECTOR_FIXTURE / "train" / "clean.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    binding = raw["segment_plan"]["bindings"][field]
    raw["segment_plan"]["bindings"][field] = (
        "0" * 64 if isinstance(binding, str) and len(binding) == 64 else "drift"
    )
    with pytest.raises(QuerySpecializationError, match="bindings do not match"):
        parse_taskboard_sidecar(raw)


def test_segment_plan_rejects_cache_claims_for_unbound_identity() -> None:
    raw = json.loads(
        (PROJECTOR_FIXTURE / "train" / "clean.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    raw["segment_plan"]["cache_compatibility"]["cache_reuse_allowed"] = True
    with pytest.raises(QuerySpecializationError, match="cache_compatibility contract"):
        parse_taskboard_sidecar(raw)


def test_segment_plan_recomputes_content_scope_visibility_and_lineage() -> None:
    raw = json.loads(
        (PROJECTOR_FIXTURE / "train" / "clean.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    for field, value in (
        ("content_sha256", "0" * 64),
        ("cache_scope", "expert_private_delta"),
        ("visibility", ["planner"]),
        ("prefix_lineage_sha256", "0" * 64),
    ):
        tampered = copy.deepcopy(raw)
        tampered["segment_plan"]["segments"][0][field] = value
        with pytest.raises(QuerySpecializationError, match="binding changed"):
            parse_taskboard_sidecar(tampered)


def test_segment_plan_never_contains_forbidden_or_current_target_payload() -> None:
    raw = json.loads(
        (PROJECTOR_FIXTURE / "train" / "clean.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    sidecar = parse_taskboard_sidecar(raw)
    source_ids = {
        segment["source_block_id"] for segment in sidecar.segment_plan["segments"]
    }
    forbidden = set(sidecar.training_record.targets.forbidden)
    assert source_ids.isdisjoint(forbidden)
    assert json.loads(sidecar.training_record.target_output)["answer"] not in (
        sidecar.segment_plan_json
    )


def test_sidecar_dataset_prevents_bundle_split_leakage_and_language_mix() -> None:
    sidecars, _, _ = load_taskboard_sidecar_dataset(PROJECTOR_FIXTURE)
    train = next(sidecar for sidecar in sidecars if sidecar.split == "train")
    calibration_index = next(
        index
        for index, sidecar in enumerate(sidecars)
        if sidecar.split == "calibration"
    )
    leaked = list(sidecars)
    leaked[calibration_index] = replace(
        leaked[calibration_index], task_bundle_sha256=train.task_bundle_sha256
    )
    with pytest.raises(
        QuerySpecializationError,
        match="segment-plan bindings do not match|task bundle hash crosses",
    ):
        validate_taskboard_sidecar_dataset(leaked)

    mixed = list(sidecars)
    changed_language = "zh-CN" if mixed[0].training_record.language == "en" else "en"
    mixed[0] = replace(
        mixed[0],
        training_record=replace(mixed[0].training_record, language=changed_language),
    )
    with pytest.raises(QuerySpecializationError, match="mixes role languages"):
        validate_taskboard_sidecar_dataset(mixed)


def test_five_role_task_bundle_is_the_split_group_not_source_gold_record_id() -> None:
    sidecars, _, _ = load_taskboard_sidecar_dataset(PROJECTOR_FIXTURE)
    train_clean = [
        sidecar
        for sidecar in sidecars
        if sidecar.split == "train" and sidecar.variant == "clean"
    ]

    assert len(train_clean) == 5
    assert len({sidecar.task_bundle_sha256 for sidecar in train_clean}) == 1
    assert len({sidecar.source_gold_record_id for sidecar in train_clean}) == 5

    moved = train_clean[0]
    split_across_roles = [
        replace(
            sidecar,
            split="calibration",
            training_record=replace(sidecar.training_record, split="calibration"),
        )
        if sidecar.record_id == moved.record_id
        else sidecar
        for sidecar in sidecars
    ]
    with pytest.raises(
        QuerySpecializationError,
        match="segment-plan bindings do not match|task bundle hash crosses",
    ):
        validate_taskboard_sidecar_dataset(split_across_roles)


def test_dataset_summary_reports_content_free_coverage() -> None:
    records = [
        parse_query_training_record(_raw_record()),
        parse_query_training_record(
            _raw_record(variant="noisy", record_id="record-noisy")
        ),
    ]

    summary = dataset_summary(records)

    assert summary == {
        "schema_version": "anchor.query-specialization.v1",
        "records": 2,
        "pairs": 1,
        "roles": {"planner": 2},
        "block_kinds": {"requirement": 2, "review": 2, "history": 1},
        "relevant_labels": 2,
        "distractor_labels": 1,
        "forbidden_labels": 2,
    }
