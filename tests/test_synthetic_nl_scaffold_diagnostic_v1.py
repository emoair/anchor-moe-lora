from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil

from jsonschema import Draft202012Validator
import pytest
import yaml

from anchor_mvp.research import synthetic_nl_scaffold_diagnostic_v1 as diagnostic


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / diagnostic.CONFIG_PATH
CONFIG_SCHEMA = ROOT / diagnostic.CONFIG_SCHEMA_PATH
ARTIFACT = ROOT / diagnostic.CANONICAL_FIXTURE_PATH
RECORD_SCHEMA = ROOT / diagnostic.RECORD_SCHEMA_PATH
MANIFEST_SCHEMA = ROOT / diagnostic.MANIFEST_SCHEMA_PATH
GRAMMAR = ROOT / diagnostic.CLOSED_GRAMMAR_PATH


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path = ARTIFACT) -> dict[str, object]:
    return json.loads((root / "manifest.json").read_bytes())


def _records(root: Path = ARTIFACT) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative in diagnostic.PARTITION_PATHS:
        records.extend(
            json.loads(line)
            for line in (root / relative).read_text("utf-8").splitlines()
        )
    return records


def test_canonical_fixture_audits_and_remains_diagnostic_only() -> None:
    manifest = diagnostic.audit_dataset(ROOT, CONFIG, ARTIFACT)
    assert manifest["status"] == "dataset_proxy_ready_training_not_authorized"
    assert manifest["counts"] == {
        "records": 100,
        "pairs": 50,
        "source_bundles": 10,
        "roles": 5,
        "languages": 2,
        "variants": 2,
        "train_records": 80,
        "eval_proxy_records": 20,
    }
    assert manifest["claims"] == {
        "dataset_proxy_ready": True,
        "diagnostic_only": True,
        "formal": False,
        "training_authorized": False,
        "quality_validated": False,
        "eval_proxy_is_heldout": False,
        "physical_kv_reuse_claimed": False,
        "numeric_equivalence_claimed": False,
    }
    assert manifest["audit"] == {
        "record_schema_validated": True,
        "manifest_schema_validated": True,
        "bundle_split_disjoint": True,
        "pair_invariants_validated": True,
        "forbidden_content_excluded": True,
        "mandatory_sidecar": True,
        "protected_body_reads": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "real_tool_executions": 0,
    }


def test_manifest_sidecar_and_all_physical_hashes_are_exact() -> None:
    manifest_bytes = (ARTIFACT / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    assert (ARTIFACT / "manifest.json.sha256").read_bytes() == (
        f"{manifest_sha}  manifest.json\n".encode("ascii")
    )
    manifest = _manifest()
    producer = manifest["producer"]
    assert isinstance(producer, dict)
    for field in (
        "config",
        "config_schema",
        "closed_grammar",
        "closed_grammar_schema",
        "record_schema",
        "manifest_schema",
        "implementation",
    ):
        binding = producer[field]
        assert isinstance(binding, dict)
        path = ROOT / str(binding["path"])
        assert binding["bytes"] == path.stat().st_size
        assert binding["sha256"] == _sha(path)
    for partition in manifest["partitions"]:
        assert isinstance(partition, dict)
        path = ARTIFACT / str(partition["path"])
        assert partition["bytes"] == path.stat().st_size
        assert partition["sha256"] == _sha(path)
        assert partition["records"] == len(path.read_bytes().splitlines())


def test_record_pair_role_language_variant_and_split_balance() -> None:
    records = _records()
    assert len(records) == 100
    assert len({record["record_id"] for record in records}) == 100
    pairs: dict[str, list[dict[str, object]]] = defaultdict(list)
    bundles: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        pairs[str(record["pair_id"])].append(record)
        bundles[str(record["source_bundle_id"])].append(record)
    assert len(pairs) == 50
    assert len(bundles) == 10
    for pair in pairs.values():
        assert len(pair) == 2
        assert {record["variant"] for record in pair} == set(diagnostic.VARIANTS)
        assert pair[0]["input"] == pair[1]["input"]
        assert pair[0]["forbidden_segment_ids"] == pair[1]["forbidden_segment_ids"]
        assert (
            pair[0]["target"]["canonical_json_sha256"]
            == (pair[1]["target"]["canonical_json_sha256"])
        )
    for bundle in bundles.values():
        assert len(bundle) == 10
        assert len({record["split"] for record in bundle}) == 1
        assert Counter(record["role"] for record in bundle) == Counter(
            {role: 2 for role in diagnostic.ROLES}
        )
    matrix = Counter(
        (record["split"], record["language"], record["role"], record["variant"])
        for record in records
    )
    for language in diagnostic.LANGUAGES:
        for role in diagnostic.ROLES:
            for variant in diagnostic.VARIANTS:
                assert matrix[("train", language, role, variant)] == 4
                assert matrix[("eval_proxy", language, role, variant)] == 1


def test_real_materializer_uses_only_short_committed_summaries() -> None:
    config = yaml.safe_load(CONFIG.read_text("utf-8"))
    grammar = json.loads(GRAMMAR.read_bytes())
    for bundle in config["bundles"]:
        bundle_id, _ = diagnostic._bundle_identity(bundle)
        segments = diagnostic._make_segments(bundle, bundle_id, grammar)
        for stage, role in enumerate(diagnostic.ROLES):
            view = diagnostic.build_training_view(bundle, role, segments, grammar)
            prompt = view["materialized_prompt"]
            for previous in segments[:stage]:
                assert previous["segment_ref"] in prompt
                assert previous["commit_summary"] in prompt
                assert previous["segment_id"] not in prompt
                assert previous["content"] not in prompt
            for forbidden in segments[stage:]:
                assert forbidden["segment_ref"] not in prompt
                assert forbidden["segment_id"] not in prompt
                assert forbidden["content"] not in prompt


def test_grammar_seed_read_set_and_protected_boundaries_are_explicit() -> None:
    manifest = _manifest()
    generation = manifest["generation_contract"]
    assert generation == {
        "seed_id": "anchor.synthetic-nl-scaffold-diagnostic.seed.v1",
        "seed_sha256": hashlib.sha256(
            b"anchor-synthetic-nl-scaffold-diagnostic-generation-seed-v1"
        ).hexdigest(),
        "source_namespace": "anchor.synthetic-nl-scaffold-diagnostic.v1",
        "augmentation": "none",
        "split_before_augmentation": True,
    }
    read_set = manifest["read_set"]
    assert read_set["scope"] == "declared_semantic_generation_inputs_only"
    assert len(read_set["ordered_artifacts"]) == 7
    assert read_set["protected_source_paths_read"] == 0
    protected = manifest["protected_inventory_status"]
    assert protected["consumes_protected_inventories"] is False
    assert set(protected["statuses"]) == {
        "swebench_source",
        "gold_partition",
        "partial_gold_export",
        "heldout",
        "legacy_heldout_cases",
        "synthetic_scaffold",
    }
    assert set(protected["statuses"].values()) == {"unavailable_not_read"}
    assert manifest["source_disjoint_boundary"] == {
        "source_namespace": "anchor.synthetic-nl-scaffold-diagnostic.v1",
        "zero_intersection_claimed": False,
        "source_disjoint_attestation_emitted": False,
        "formal_source_disjoint_proven": False,
        "status": "unavailable_without_protected_inventory_identities",
    }


def test_all_three_ablation_arms_share_the_same_inventory() -> None:
    manifest = _manifest()
    inventories = manifest["inventories"]["arm_record_inventory_sha256"]
    assert set(inventories) == set(diagnostic.ABLATION_LABELS)
    assert len(set(inventories.values())) == 1
    records = _records()
    expected_content_inventory = diagnostic._inventory_sha256(
        "anchor.synthetic-nl-scaffold-record-content-inventory.v1",
        [
            hashlib.sha256(diagnostic._canonical_json_bytes(record)).hexdigest()
            for record in records
        ],
    )
    assert manifest["inventories"]["record_content_sha256"] == (
        expected_content_inventory
    )
    assert set(inventories.values()) == {expected_content_inventory}
    assert manifest["ablation_contract"] == {
        "labels": list(diagnostic.ABLATION_LABELS),
        "same_record_inventory_for_all_arms": True,
        "assignment_location": "diagnostic_run_manifest_only",
        "producer_selects_winner": False,
        "target_modules_bound_by_dataset": False,
    }


def test_token_lengths_are_explicitly_unbound_in_dataset() -> None:
    contract = _manifest()["token_length_contract"]
    assert contract == {
        "status": "tokenizer_unbound",
        "token_counts_emitted": False,
        "truncation_allowed": False,
        "run_preflight_must_bind_tokenizer_and_full_chat_plus_target_lengths": True,
        "diagnostic_target_max_tokens": 1024,
        "diagnostic_preferred_p95_tokens": 768,
    }
    serialized = json.dumps(_manifest(), sort_keys=True)
    assert "observed_token" not in serialized


def test_deterministic_rebuild_is_byte_identical(tmp_path: Path) -> None:
    rebuilt = tmp_path / "rebuilt"
    diagnostic.build_dataset(ROOT, CONFIG, rebuilt)
    expected = {
        path.relative_to(ARTIFACT).as_posix(): path.read_bytes()
        for path in ARTIFACT.rglob("*")
        if path.is_file()
    }
    observed = {
        path.relative_to(rebuilt).as_posix(): path.read_bytes()
        for path in rebuilt.rglob("*")
        if path.is_file()
    }
    assert observed == expected


def test_existing_output_is_never_replaced(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticError,
        match="synthetic_output_already_exists",
    ):
        diagnostic.build_dataset(ROOT, CONFIG, output)
    assert sentinel.read_text("utf-8") == "keep"


def test_publish_race_is_no_replace_and_preserves_racer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "raced"
    original = diagnostic._rename_directory_no_replace

    def race_then_rename(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "racer.txt").write_text("keep", encoding="utf-8")
        original(source, destination)

    monkeypatch.setattr(diagnostic, "_rename_directory_no_replace", race_then_rename)
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticError,
        match="synthetic_output_already_exists",
    ):
        diagnostic.build_dataset(ROOT, CONFIG, output)
    assert (output / "racer.txt").read_text("utf-8") == "keep"


def test_temp_mutation_after_prepublication_audit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "mutated"
    original = diagnostic._capture_artifact_snapshots
    captures = 0

    def capture_then_mutate(artifact: Path):
        nonlocal captures
        result = original(artifact)
        captures += 1
        if captures == 2:
            partition = artifact / diagnostic.PARTITION_PATHS[0]
            partition.write_bytes(partition.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(diagnostic, "_capture_artifact_snapshots", capture_then_mutate)
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticError,
        match="changed_before_publish",
    ):
        diagnostic.build_dataset(ROOT, CONFIG, output)
    assert not output.exists()


def test_config_schema_and_runtime_reject_route_contract_drift() -> None:
    config = yaml.safe_load(CONFIG.read_text("utf-8"))
    schema = json.loads(CONFIG_SCHEMA.read_bytes())
    config["route_contract"]["real_tool_execution"] = True
    assert not Draft202012Validator(schema).is_valid(config)
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticError,
        match="synthetic_route_contract_drift",
    ):
        diagnostic._validate_config(config)


def test_duplicate_yaml_and_nonfinite_or_duplicate_json_are_rejected(
    tmp_path: Path,
) -> None:
    duplicate_config = tmp_path / "duplicate.yaml"
    duplicate_config.write_bytes(
        CONFIG.read_bytes()
        + b"\nclaim_scope: synthetic_diagnostic_only_no_formal_or_training_authority\n"
    )
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticError,
        match="synthetic_config_invalid",
    ):
        diagnostic._load_contract_snapshots(ROOT, duplicate_config)
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(diagnostic.SyntheticScaffoldDiagnosticError):
            diagnostic._strict_json(raw, "synthetic_test_json_invalid")


@pytest.mark.parametrize("mode", ["missing", "tampered", "wrong_name"])
def test_manifest_sidecar_is_mandatory_and_strict(tmp_path: Path, mode: str) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    sidecar = copied / "manifest.json.sha256"
    if mode == "missing":
        sidecar.unlink()
    elif mode == "tampered":
        sidecar.write_text(f"{'0' * 64}  manifest.json\n", encoding="ascii")
    else:
        value = sidecar.read_text("ascii").replace("manifest.json", "other.json")
        sidecar.write_text(value, encoding="ascii")
    with pytest.raises(diagnostic.SyntheticScaffoldDiagnosticError):
        diagnostic.audit_dataset(ROOT, CONFIG, copied)


def test_partition_tamper_is_fail_closed(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    partition = copied / diagnostic.PARTITION_PATHS[0]
    partition.write_bytes(partition.read_bytes() + b" \n")
    with pytest.raises(diagnostic.SyntheticScaffoldDiagnosticError):
        diagnostic.audit_dataset(ROOT, CONFIG, copied)


def test_extra_directory_is_rejected_by_exact_layout(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    (copied / "unexpected-empty-directory").mkdir()
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticError,
        match="synthetic_artifact_layout_invalid",
    ):
        diagnostic.audit_dataset(ROOT, CONFIG, copied)


def test_dangling_symlink_is_rejected_by_exact_layout(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    dangling = copied / "dangling-link"
    try:
        dangling.symlink_to(copied / "does-not-exist")
    except OSError:
        pytest.skip("host does not permit symlink creation")
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticError,
        match="synthetic_artifact_reparse_entry",
    ):
        diagnostic.audit_dataset(ROOT, CONFIG, copied)


def test_audit_open_set_excludes_protected_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    original = Path.open

    def traced_open(path: Path, *args: object, **kwargs: object):
        opened.append(path.absolute().as_posix().lower())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", traced_open)
    diagnostic.audit_dataset(ROOT, CONFIG, ARTIFACT)
    for value in opened:
        assert "data/automated_v3_shards" not in value
        assert "artifacts/benchmark/heldout" not in value
        assert "configs/training/heldout_cases" not in value
        assert "datasets/public/swebench" not in value


def test_full_audit_detects_mid_evaluation_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    original = diagnostic._expected_materialization
    changed = False

    def mutate_after_snapshot(*args: object, **kwargs: object):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            partition = copied / diagnostic.PARTITION_PATHS[0]
            replacement = partition.with_name("replacement.jsonl")
            replacement.write_bytes(partition.read_bytes() + b"\n")
            os.replace(replacement, partition)
        return result

    monkeypatch.setattr(diagnostic, "_expected_materialization", mutate_after_snapshot)
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticError,
        match="changed_during_audit",
    ):
        diagnostic.audit_dataset(ROOT, CONFIG, copied)


@pytest.mark.parametrize(
    ("document", "mutation"),
    [
        ("record", lambda value: value["claims"].__setitem__("formal", True)),
        (
            "record",
            lambda value: value["synthetic_source"].__setitem__("unexpected", True),
        ),
        (
            "record",
            lambda value: value["target"].__setitem__(
                "concise_rationale_summary", "not valid for json_only"
            ),
        ),
        (
            "manifest",
            lambda value: value["claims"].__setitem__("training_authorized", True),
        ),
        (
            "manifest",
            lambda value: value["source_disjoint_boundary"].__setitem__(
                "zero_intersection_claimed", True
            ),
        ),
        (
            "manifest",
            lambda value: value["partitions"][0].__setitem__(
                "path", "train/concise_rationale_plus_json.jsonl"
            ),
        ),
        (
            "manifest",
            lambda value: value["read_set"]["ordered_artifacts"].__setitem__(
                slice(0, 2),
                list(reversed(value["read_set"]["ordered_artifacts"][:2])),
            ),
        ),
    ],
)
def test_closed_schemas_reject_promotions_and_unknown_fields(
    document: str, mutation: object
) -> None:
    if document == "record":
        value = _records()[0]
        schema = json.loads(RECORD_SCHEMA.read_bytes())
    else:
        value = _manifest()
        schema = json.loads(MANIFEST_SCHEMA.read_bytes())
    mutation(value)
    assert not Draft202012Validator(schema).is_valid(value)


def test_artifact_layout_encoding_size_and_no_protected_paths() -> None:
    expected = {"manifest.json", "manifest.json.sha256", *diagnostic.PARTITION_PATHS}
    files = {
        path.relative_to(ARTIFACT).as_posix()
        for path in ARTIFACT.rglob("*")
        if path.is_file()
    }
    assert files == expected
    for relative in expected:
        raw = (ARTIFACT / relative).read_bytes()
        assert len(raw) < 50_000_000
        raw.decode("utf-8")
        assert b"\r" not in raw
    contract_text = "\n".join(
        [
            CONFIG.read_text("utf-8"),
            GRAMMAR.read_text("utf-8"),
            (ARTIFACT / "manifest.json").read_text("utf-8"),
        ]
    ).lower()
    for forbidden in (
        "data/automated_v3_shards",
        "artifacts/benchmark/heldout",
        "configs/training/heldout_cases",
        "datasets/public/swebench",
    ):
        assert forbidden not in contract_text
