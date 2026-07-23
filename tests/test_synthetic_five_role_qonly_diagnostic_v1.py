from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest

from anchor_mvp.research import (
    synthetic_five_role_qonly_diagnostic_v1 as diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / diagnostic.CONFIG_PATH
CONFIG_SCHEMA = ROOT / diagnostic.CONFIG_SCHEMA_PATH
CATALOG = ROOT / diagnostic.CATALOG_PATH
CATALOG_SCHEMA = ROOT / diagnostic.CATALOG_SCHEMA_PATH
GRAMMAR = ROOT / diagnostic.CLOSED_GRAMMAR_PATH
RECORD_SCHEMA = ROOT / diagnostic.RECORD_SCHEMA_PATH
MANIFEST_SCHEMA = ROOT / diagnostic.MANIFEST_SCHEMA_PATH
ARTIFACT = ROOT / diagnostic.CANONICAL_FIXTURE_PATH


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path = ARTIFACT) -> dict[str, object]:
    return json.loads((root / "manifest.json").read_bytes())


def _catalog() -> dict[str, object]:
    return json.loads(CATALOG.read_bytes())


def _records(root: Path = ARTIFACT) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in diagnostic.PARTITION_PATHS:
        rows.extend(
            json.loads(line)
            for line in (root / relative).read_text("utf-8").splitlines()
        )
    return rows


def test_canonical_fixture_audits_and_is_diagnostic_only() -> None:
    manifest = diagnostic.audit_dataset(ROOT, CONFIG, ARTIFACT)
    assert manifest["status"] == "dataset_proxy_ready_training_not_authorized"
    assert manifest["counts"] == diagnostic.EXPECTED_COUNTS
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
    assert manifest["audit"]["protected_body_reads"] == 0
    assert manifest["audit"]["provider_requests"] == 0
    assert manifest["audit"]["network_requests"] == 0
    assert manifest["audit"]["model_loads"] == 0
    assert manifest["audit"]["gpu_requests"] == 0


def test_physical_hashes_sidecar_and_schema_instances_are_exact() -> None:
    manifest_raw = (ARTIFACT / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    assert (ARTIFACT / "manifest.json.sha256").read_bytes() == (
        f"{manifest_sha}  manifest.json\n".encode("ascii")
    )
    manifest = _manifest()
    for field in (
        "config",
        "config_schema",
        "closed_grammar",
        "closed_grammar_schema",
        "bundle_catalog",
        "bundle_catalog_schema",
        "record_schema",
        "manifest_schema",
        "implementation",
        "base_security_implementation",
    ):
        binding = manifest["producer"][field]
        path = ROOT / binding["path"]
        assert binding["bytes"] == path.stat().st_size
        assert binding["sha256"] == _sha(path)
    Draft202012Validator(json.loads(MANIFEST_SCHEMA.read_bytes())).validate(manifest)


def test_exact_1000_rows_and_per_role_split_balance() -> None:
    records = _records()
    assert len(records) == 1000
    assert len({row["record_id"] for row in records}) == 1000
    assert Counter(row["role"] for row in records) == Counter(
        {role: 200 for role in diagnostic.ROLES}
    )
    assert Counter(row["split"] for row in records) == Counter(
        {"train": 800, "eval_proxy": 200}
    )
    for role in diagnostic.ROLES:
        assert Counter(
            row["split"] for row in records if row["role"] == role
        ) == Counter({"train": 160, "eval_proxy": 40})


def test_200_bundles_have_five_roles_and_never_cross_split() -> None:
    bundles: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in _records():
        bundles[record["task_bundle_sha256"]].append(record)
    assert len(bundles) == 200
    for rows in bundles.values():
        assert len(rows) == 5
        assert {row["role"] for row in rows} == set(diagnostic.ROLES)
        assert len({row["split"] for row in rows}) == 1
        assert len({row["task_semantic_sha256"] for row in rows}) == 1
        assert len({row["inner_task_id"] for row in rows}) == 1
        assert len({row["chain_root_sha256"] for row in rows}) == 1


def test_each_language_stratum_role_cell_is_16_train_4_eval() -> None:
    records = _records()
    strata = {row["stratum"] for row in records}
    assert len(strata) == 5
    matrix = Counter(
        (row["language"], row["stratum"], row["role"], row["split"]) for row in records
    )
    for language in diagnostic.LANGUAGES:
        for stratum in strata:
            for role in diagnostic.ROLES:
                assert matrix[(language, stratum, role, "train")] == 16
                assert matrix[(language, stratum, role, "eval_proxy")] == 4


def test_semantic_inventory_is_200_unique_untranslated_tasks_per_role() -> None:
    records = _records()
    by_role = {
        role: {row["task_semantic_sha256"] for row in records if row["role"] == role}
        for role in diagnostic.ROLES
    }
    assert {len(values) for values in by_role.values()} == {200}
    assert len({tuple(sorted(values)) for values in by_role.values()}) == 1
    by_language = {
        language: {
            row["task_semantic_sha256"]
            for row in records
            if row["language"] == language
        }
        for language in diagnostic.LANGUAGES
    }
    assert len(by_language["en"]) == 100
    assert len(by_language["zh-CN"]) == 100
    assert by_language["en"].isdisjoint(by_language["zh-CN"])
    assert _manifest()["semantic_identity_contract"]["translation_pair_count"] == 0


def test_real_materializer_routes_role_specific_evidence_and_hides_targets() -> None:
    catalog = _catalog()
    grammar = json.loads(GRAMMAR.read_bytes())
    for bundle in catalog["bundles"]:
        bundle_id, _ = diagnostic._bundle_identity(bundle)
        segments = diagnostic._make_segments(bundle, bundle_id, grammar)
        for stage, role in enumerate(diagnostic.ROLES):
            spec = bundle["role_specs"][role]
            route = segments[stage]["route"]
            assert route["constraints"] == spec["constraints"]
            assert route["evidence_intent"] == spec["evidence_intent"]
            assert route["goal"] == spec["goal"]
            view = diagnostic.build_training_view(bundle, role, segments, grammar)
            prompt = view["materialized_prompt"]
            for previous in segments[:stage]:
                assert previous["segment_ref"] in prompt
                assert previous["commit_summary"] in prompt
                assert previous["segment_id"] not in prompt
                assert previous["content"] not in prompt
                assert previous["segment_id"] not in diagnostic._canonical_json(view)
                assert previous["content_sha256"] not in diagnostic._canonical_json(
                    view
                )
                assert previous[
                    "commit_summary_sha256"
                ] not in diagnostic._canonical_json(view)
            for forbidden in segments[stage:]:
                assert forbidden["segment_ref"] not in prompt
                assert forbidden["segment_id"] not in prompt
                assert forbidden["content"] not in prompt
                assert forbidden["commit_summary"] not in prompt
                serialized = diagnostic._canonical_json(view)
                assert forbidden["content_sha256"] not in serialized
                assert forbidden["commit_summary_sha256"] not in serialized


def test_rows_are_arm_neutral_and_controls_are_manifest_only() -> None:
    serialized = "\n".join(diagnostic._canonical_json(row) for row in _records())
    for forbidden in ("q_only", "o_only", "q_plus_o", "wide_lora"):
        assert forbidden not in serialized
    manifest = _manifest()
    assert manifest["ablation_contract"]["primary_label"] == "q_only"
    assert manifest["ablation_contract"]["diagnostic_control_labels"] == [
        "o_only",
        "q_plus_o",
    ]
    assert manifest["ablation_contract"]["control_arm_rows_materialized"] is False
    assert manifest["ablation_contract"]["legacy_wide_lora_control_inherited"] is False
    inventories = manifest["inventories"]["arm_record_inventory_sha256"]
    assert set(inventories) == {"q_only", "o_only", "q_plus_o"}
    assert len(set(inventories.values())) == 1


def test_capability_metadata_is_auditable_without_guessing_quotas() -> None:
    coverage = _manifest()["capability_coverage"]
    assert coverage["adds_rows_views_or_arms"] is False
    assert coverage["source"] == "catalog_explicit_labels_only"
    for capability in ("simple_tool_search", "micro_coding"):
        assert coverage[capability] == {
            "status": "unavailable_no_explicit_catalog_label",
            "task_bundle_count": None,
            "quota_claimed": False,
        }


def test_bundle_hash_excludes_role_view_arm_and_noise() -> None:
    bundle = _catalog()["bundles"][0]
    original = diagnostic._bundle_identity(bundle)
    mutated = deepcopy(bundle)
    mutated["role_specs"]["planner"]["goal"] += " role-only mutation"
    mutated["role_specs"]["security_gate"]["tool_action"] += " overlay-only mutation"
    assert diagnostic._bundle_identity(mutated) == original
    assert set(diagnostic._bundle_payload(bundle)) == {
        "domain",
        "source_namespace",
        "bundle_key",
        "language",
        "archetype",
        "stratum",
        "task_text",
        "constraints",
        "acceptance_hints",
        "task_semantic_sha256",
    }


def test_catalog_semantic_tamper_and_collision_fail_closed() -> None:
    catalog = _catalog()
    tampered = deepcopy(catalog)
    tampered["bundles"][0]["semantic_identity"]["intent"] += "_tampered"
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticV2Error,
        match="semantic_hash_mismatch",
    ):
        diagnostic._validate_catalog(tampered)
    collided = deepcopy(catalog)
    collided["bundles"][1]["semantic_identity"] = deepcopy(
        collided["bundles"][0]["semantic_identity"]
    )
    collided["bundles"][1]["task_semantic_sha256"] = diagnostic._task_semantic_sha256(
        collided["bundles"][1]
    )
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticV2Error,
        match="semantic_identity_collision",
    ):
        diagnostic._validate_catalog(collided)


@pytest.mark.parametrize(
    "salt",
    [
        "en",
        "planner",
        "anchor_synthetic_five_role_qonly_diagnostic_v1",
        "en_prefix_evidence_selection_00_accessible_checkout",
    ],
)
def test_semantic_identity_value_salt_is_rejected(salt: str) -> None:
    candidate = deepcopy(_catalog())
    candidate["bundles"][0]["semantic_identity"]["intent"] = f"semantic_probe::{salt}"
    candidate["bundles"][0]["task_semantic_sha256"] = diagnostic._task_semantic_sha256(
        candidate["bundles"][0]
    )
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticV2Error,
        match="semantic_salt_embedded",
    ):
        diagnostic._validate_catalog(candidate)


def test_future_commit_summary_injection_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots, config, grammar, catalog, record_schema, _ = (
        diagnostic._load_contract_snapshots(ROOT, CONFIG)
    )
    original = diagnostic.build_training_view

    def inject_current_summary(
        bundle: dict[str, object],
        role: str,
        segments: list[dict[str, object]],
        closed_grammar: dict[str, object],
    ) -> dict[str, object]:
        view = original(bundle, role, segments, closed_grammar)
        stage = diagnostic.ROLES.index(role)
        view["materialized_prompt"] += segments[stage]["commit_summary"] + "\n"
        view["input_sha256"] = diagnostic._canonical_sha256(
            {key: value for key, value in view.items() if key != "input_sha256"}
        )
        return view

    monkeypatch.setattr(diagnostic, "build_training_view", inject_current_summary)
    records, _, _ = diagnostic._generate_records(config, grammar, snapshots, catalog)
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticV2Error,
        match="forbidden_content_leaked",
    ):
        diagnostic._validate_records(config, grammar, catalog, records, record_schema)


def test_forbidden_hash_only_injection_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots, config, grammar, catalog, record_schema, _ = (
        diagnostic._load_contract_snapshots(ROOT, CONFIG)
    )
    original = diagnostic.build_training_view

    def inject_current_hash(
        bundle: dict[str, object],
        role: str,
        segments: list[dict[str, object]],
        closed_grammar: dict[str, object],
    ) -> dict[str, object]:
        view = original(bundle, role, segments, closed_grammar)
        stage = diagnostic.ROLES.index(role)
        view["materialized_prompt"] += segments[stage]["content_sha256"] + "\n"
        view["input_sha256"] = diagnostic._canonical_sha256(
            {key: value for key, value in view.items() if key != "input_sha256"}
        )
        return view

    monkeypatch.setattr(diagnostic, "build_training_view", inject_current_hash)
    records, _, _ = diagnostic._generate_records(config, grammar, snapshots, catalog)
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticV2Error,
        match="forbidden_content_leaked",
    ):
        diagnostic._validate_records(config, grammar, catalog, records, record_schema)


def test_segment_visibility_and_forbidden_union_tamper_is_rejected() -> None:
    snapshots, config, grammar, catalog, record_schema, _ = (
        diagnostic._load_contract_snapshots(ROOT, CONFIG)
    )
    records, _, _ = diagnostic._generate_records(config, grammar, snapshots, catalog)
    record = records[0]
    current = int(record["stage_index"])
    record["board_segment_inventory"][current]["visibility"] = "previous_committed"
    record["forbidden_segment_ids"].remove(
        record["board_segment_inventory"][current]["segment_id"]
    )
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticV2Error,
        match="segment_inventory_mismatch",
    ):
        diagnostic._validate_records(config, grammar, catalog, records, record_schema)


def test_catalog_schema_rejects_split_fields_unknown_roles_and_old_versions() -> None:
    schema = json.loads(CATALOG_SCHEMA.read_bytes())
    validator = Draft202012Validator(schema)
    base = _catalog()
    for mutate in (
        lambda value: value["bundles"][0].__setitem__("split", "train"),
        lambda value: value["bundles"][0]["role_specs"].pop("security_gate"),
        lambda value: value.__setitem__("schema_version", "legacy"),
    ):
        candidate = deepcopy(base)
        mutate(candidate)
        assert not validator.is_valid(candidate)


def test_record_schema_rejects_arm_fields_and_formal_promotion() -> None:
    schema = json.loads(RECORD_SCHEMA.read_bytes())
    validator = Draft202012Validator(schema)
    base = _records()[0]
    for mutate in (
        lambda value: value.__setitem__("arm", "q_only"),
        lambda value: value["claims"].__setitem__("formal", True),
        lambda value: value["ablation"].__setitem__(
            "row_duplicated_for_controls", True
        ),
    ):
        candidate = deepcopy(base)
        mutate(candidate)
        assert not validator.is_valid(candidate)


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
        sidecar.write_text(
            sidecar.read_text("ascii").replace("manifest.json", "other.json"),
            encoding="ascii",
        )
    with pytest.raises(diagnostic.SyntheticScaffoldDiagnosticV2Error):
        diagnostic.audit_dataset(ROOT, CONFIG, copied)


def test_partition_tamper_and_mid_audit_replacement_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    partition = copied / diagnostic.PARTITION_PATHS[0]
    partition.write_bytes(partition.read_bytes() + b" \n")
    with pytest.raises(diagnostic.SyntheticScaffoldDiagnosticV2Error):
        diagnostic.audit_dataset(ROOT, CONFIG, copied)

    copied = tmp_path / "raced"
    shutil.copytree(ARTIFACT, copied)
    original = diagnostic._expected_materialization
    changed = False

    def mutate_after_snapshot(*args: object, **kwargs: object):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            current = copied / diagnostic.PARTITION_PATHS[0]
            replacement = current.with_name("replacement.jsonl")
            replacement.write_bytes(current.read_bytes() + b"\n")
            os.replace(replacement, current)
        return result

    monkeypatch.setattr(diagnostic, "_expected_materialization", mutate_after_snapshot)
    with pytest.raises(
        diagnostic.SyntheticScaffoldDiagnosticV2Error,
        match="changed_during_audit",
    ):
        diagnostic.audit_dataset(ROOT, CONFIG, copied)


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
            CATALOG.read_text("utf-8"),
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


def test_authorship_is_not_confused_with_zero_request_build_counters() -> None:
    authorship = _manifest()["authorship"]
    assert authorship == {
        "catalog_authored_with_openai_codex_gpt_5_6_sol_assistance": True,
        "catalog_content_is_not_reported_as_zero_model_authored": True,
        "deterministic_build_provider_requests": 0,
        "acknowledgement": "OpenAI GPT-5.6-sol assisted dataset authorship.",
    }


def test_cli_bootstraps_src_without_pythonpath() -> None:
    script = (
        ROOT
        / "scripts"
        / "research"
        / "build_synthetic_five_role_qonly_diagnostic_v1.py"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_result.returncode == 0
    assert "{build,audit}" in help_result.stdout
    audit_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "audit",
            "--repo-root",
            ".",
            "--config",
            diagnostic.CONFIG_PATH,
            "--artifact",
            diagnostic.CANONICAL_FIXTURE_PATH,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert audit_result.returncode == 0, audit_result.stderr
