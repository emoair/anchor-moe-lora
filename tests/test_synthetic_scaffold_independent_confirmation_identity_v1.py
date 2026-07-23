from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
import pytest

import anchor_mvp.swebench.synthetic_scaffold_independent_confirmation_identity as identity


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(identity.CONFIG_PATH)
SOURCE_RELATIVE_PATHS = (
    Path("configs/research/synthetic_nl_scaffold_diagnostic_v1.yaml"),
    Path("fixtures/research/synthetic_nl_scaffold_diagnostic_v1/manifest.json"),
    Path("fixtures/research/synthetic_nl_scaffold_diagnostic_v1/manifest.json.sha256"),
    Path("configs/research/qwen_multiseed_independent_bundle_plan_v1.yaml"),
)
PRODUCER_RELATIVE_PATHS = (
    CONFIG_PATH,
    Path(
        "configs/research/"
        "synthetic_scaffold_independent_confirmation_identity_v1_config.schema.json"
    ),
    Path(
        "configs/research/"
        "synthetic_scaffold_independent_confirmation_identity_v1_record.schema.json"
    ),
    Path(
        "configs/research/"
        "synthetic_scaffold_independent_confirmation_identity_v1_proof.schema.json"
    ),
    Path(
        "configs/research/"
        "synthetic_scaffold_independent_confirmation_identity_v1_manifest.schema.json"
    ),
    Path(
        "src/anchor_mvp/swebench/"
        "synthetic_scaffold_independent_confirmation_identity.py"
    ),
)
JSONL_PARTITIONS = tuple(
    relative for relative in identity.PARTITION_PATHS if relative.endswith(".jsonl")
)
PROOF_PARTITIONS = tuple(
    relative for relative in identity.PARTITION_PATHS if relative.endswith(".json")
)
BODY_KEYS = {
    "answer",
    "assistant",
    "canonical_routing_json",
    "content",
    "input_ids",
    "output",
    "preview",
    "prompt",
    "rationale",
    "task_text",
    "token_ids",
    "tool_calls",
    "tool_results",
}


def _source_root() -> Path:
    if all((ROOT / relative).is_file() for relative in SOURCE_RELATIVE_PATHS):
        return ROOT
    sibling = ROOT.parent / "anchor-moe-lora-neural-swarm"
    assert all((sibling / relative).is_file() for relative in SOURCE_RELATIVE_PATHS)
    return sibling


SOURCE_ROOT = _source_root()


def _canonical_json(value: object, *, newline: bool = True) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sidecar(data: bytes) -> bytes:
    return f"{hashlib.sha256(data).hexdigest()}  manifest.json\n".encode("ascii")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    result = [json.loads(line) for line in raw.splitlines()]
    assert all(isinstance(item, dict) for item in result)
    return result


def _copy_artifact(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "artifact"
    shutil.copytree(source, destination)
    return destination


def _copy_producer_root(tmp_path: Path) -> Path:
    destination = tmp_path / "producer"
    destination.mkdir()
    for relative in PRODUCER_RELATIVE_PATHS:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return destination


def _load_producer_module(repo_root: Path) -> Any:
    implementation = repo_root.resolve(strict=True) / identity.IMPLEMENTATION_PATH
    module_name = (
        "_anchor_identity_copy_"
        + hashlib.sha256(str(implementation).encode("utf-8")).hexdigest()
    )
    spec = importlib.util.spec_from_file_location(module_name, implementation)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _audit(
    artifact: Path,
    *,
    repo_root: Path = ROOT,
    config_path: str | Path = identity.CONFIG_PATH,
) -> Mapping[str, Any]:
    module = identity
    resolved_repo = Path(repo_root).resolve(strict=True)
    if resolved_repo != ROOT.resolve(strict=True):
        module = _load_producer_module(resolved_repo)
    return module.audit_identity_fixture(repo_root, SOURCE_ROOT, config_path, artifact)


def _assert_error(code: str, callback: Any) -> None:
    with pytest.raises(RuntimeError) as raised:
        callback()
    assert code in str(raised.value)


def _iter_records(artifact: Path, relatives: Iterable[str]) -> list[dict[str, Any]]:
    return [
        record for relative in relatives for record in _load_jsonl(artifact / relative)
    ]


def _walk_keys(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


@pytest.fixture(scope="session")
def built_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Mapping[str, Any]]:
    output = tmp_path_factory.mktemp("identity-v1") / "fixture"
    manifest = identity.build_identity_fixture(
        ROOT, SOURCE_ROOT, identity.CONFIG_PATH, output
    )
    return output, manifest


def test_real_build_and_real_audit_are_identical(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, built = built_artifact
    audited = _audit(artifact)
    assert audited == built
    assert audited["status"] == (
        "metadata_identity_assets_ready_execution_and_training_blocked"
    )
    assert audited["counts"] == {
        "discovery_views": 10,
        "discovery_unique_semantics": 5,
        "independent_confirmation_bundles": 60,
        "secondary_factorial_bundles": 60,
        "metadata_bundle_rows": 130,
        "records_if_both_tracks_materialized": 1200,
        "languages": 2,
        "strata": 5,
        "roles": 5,
        "variants": 2,
        "protected_body_reads": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
    }


def test_all_published_instances_pass_real_draft202012(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, manifest = built_artifact
    record_schema = _load_json(ROOT / PRODUCER_RELATIVE_PATHS[2])
    proof_schema = _load_json(ROOT / PRODUCER_RELATIVE_PATHS[3])
    manifest_schema = _load_json(ROOT / PRODUCER_RELATIVE_PATHS[4])
    for schema in (record_schema, proof_schema, manifest_schema):
        Draft202012Validator.check_schema(schema)
    record_validator = Draft202012Validator(record_schema)
    validated = 0
    for record in _iter_records(artifact, JSONL_PARTITIONS):
        record_validator.validate(record)
        validated += 1
    assert validated > 130
    proof_validator = Draft202012Validator(proof_schema)
    for relative in PROOF_PARTITIONS:
        proof_validator.validate(_load_json(artifact / relative))
    Draft202012Validator(manifest_schema).validate(manifest)


def test_discovery_bridge_is_ten_views_five_bilingual_semantics(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, _ = built_artifact
    records = _load_jsonl(artifact / "discovery/views.jsonl")
    assert len(records) == 10
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["identities"]["task_semantic_sha256"]].append(record)
    assert len(groups) == 5
    for views in groups.values():
        assert {item["language"] for item in views} == {"en", "zh-CN"}
        assert len({item["semantic_group_sha256"] for item in views}) == 1
        assert (
            len({item["identities"]["template_family_sha256"] for item in views}) == 1
        )
        assert (
            len({item["identities"]["task_template_pair_sha256"] for item in views})
            == 1
        )
        assert all(
            item["curated_bridge_not_automatic_translation_proof"] for item in views
        )


def test_historical_discovery_semantic_split_intersection_is_exactly_two(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, _ = built_artifact
    records = _load_jsonl(artifact / "discovery/views.jsonl")
    train = {
        item["identities"]["task_semantic_sha256"]
        for item in records
        if item["split"] == "train"
    }
    evaluation = {
        item["identities"]["task_semantic_sha256"]
        for item in records
        if item["split"] == "eval_proxy"
    }
    assert len(train) == 5
    assert len(evaluation) == 2
    assert len(train & evaluation) == 2
    proof = _load_json(artifact / "proofs/discovery_vs_independent.json")
    assert proof["historical_discovery_semantic_split"]["intersection"][
        "ids"
    ] == sorted(train & evaluation)
    assert (
        proof["historical_discovery_semantic_split"]["bundle_generalization_supported"]
        is False
    )


def test_independent_confirmation_counts_cells_and_unique_tasks(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, _ = built_artifact
    records = _load_jsonl(artifact / "independent_confirmation/bundles.jsonl")
    assert len(records) == 60
    assert Counter(item["split"] for item in records) == Counter(
        {"train": 40, "eval_proxy": 20}
    )
    assert len({item["task_bundle_sha256"] for item in records}) == 60
    assert len({item["identities"]["task_semantic_sha256"] for item in records}) == 60
    assert all(
        item["cross_language_translation_pair_sha256"] is None for item in records
    )
    cells: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for item in records:
        cells[(item["language"], item["stratum"])][item["split"]] += 1
    assert len(cells) == 10
    assert set(tuple(sorted(value.items())) for value in cells.values()) == {
        (("eval_proxy", 2), ("train", 4))
    }


def test_independent_confirmation_common_domain_zero_overlap(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, _ = built_artifact
    proof = _load_json(artifact / "proofs/discovery_vs_independent.json")
    assert proof["domains"] == {
        "task": "anchor.controlled-factorial-task-identity.v1",
        "template": "anchor.controlled-factorial-template-identity.v1",
        "pair": "anchor.controlled-factorial-task-template-pair.v1",
        "namespace_language_source_fields_excluded": True,
    }
    for kind in ("task", "template", "pair"):
        intersection = proof["intersections"][kind]
        assert intersection["count"] == 0
        assert intersection["ids"] == []
        assert intersection["zero_overlap"] is True
    assert proof["claims"]["independent_confirmation_executed"] is False


def test_secondary_factorial_counts_quotas_cells_and_match_keys(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, _ = built_artifact
    records = _load_jsonl(artifact / "secondary_factorial/bundles.jsonl")
    assert len(records) == 60
    assert Counter(item["factor"] for item in records) == Counter(
        {
            "old_task_new_template": 20,
            "new_task_old_template": 20,
            "new_task_new_template": 20,
        }
    )
    assert Counter(
        item["factor"] for item in records if item["split"] == "train"
    ) == Counter(
        {
            "old_task_new_template": 13,
            "new_task_old_template": 14,
            "new_task_new_template": 13,
        }
    )
    assert Counter(
        item["factor"] for item in records if item["split"] == "eval_proxy"
    ) == Counter(
        {
            "old_task_new_template": 7,
            "new_task_old_template": 6,
            "new_task_new_template": 7,
        }
    )
    cells: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    matches: dict[str, set[str]] = defaultdict(set)
    for item in records:
        cells[(item["language"], item["stratum"])][item["split"]] += 1
        matches[item["factorial_match_key_sha256"]].add(item["factor"])
    assert len(cells) == 10
    assert all(
        value == Counter({"train": 4, "eval_proxy": 2}) for value in cells.values()
    )
    assert len(matches) == 20
    assert all(value == set(identity.FACTORS) for value in matches.values())


def test_secondary_factorial_truth_table_is_recomputed(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, _ = built_artifact
    discovery = _load_jsonl(artifact / "discovery/views.jsonl")
    records = _load_jsonl(artifact / "secondary_factorial/bundles.jsonl")
    old_tasks = {item["identities"]["task_semantic_sha256"] for item in discovery}
    old_templates = {item["identities"]["template_family_sha256"] for item in discovery}
    old_pairs = {item["identities"]["task_template_pair_sha256"] for item in discovery}
    expected = {
        "old_task_new_template": (True, False),
        "new_task_old_template": (False, True),
        "new_task_new_template": (False, False),
    }
    for record in records:
        task = record["identities"]["task_semantic_sha256"]
        template = record["identities"]["template_family_sha256"]
        pair = record["identities"]["task_template_pair_sha256"]
        task_old, template_old = expected[record["factor"]]
        assert (task in old_tasks, template in old_templates) == (
            task_old,
            template_old,
        )
        assert pair not in old_pairs
        assert record["membership"]["task_in_discovery"] is task_old
        assert record["membership"]["template_in_discovery"] is template_old
        assert record["membership"]["pair_in_discovery"] is False
        assert record["may_satisfy_independent_confirmation"] is False
    proof = _load_json(artifact / "proofs/secondary_factorial.json")
    assert all(item["observed_pass"] for item in proof["truth_table"])
    assert proof["global_discovery_intersections"]["pair"]["count"] == 0
    assert proof["claims"]["may_satisfy_independent_confirmation"] is False


def test_every_pair_and_blueprint_alias_recomputes_from_authenticated_leaves(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, _ = built_artifact
    config = _load_json(ROOT / CONFIG_PATH)
    records = _iter_records(
        artifact,
        (
            "discovery/views.jsonl",
            "independent_confirmation/bundles.jsonl",
            "secondary_factorial/bundles.jsonl",
        ),
    )
    assert len(records) == 130
    for record in records:
        leaves = record["identities"]
        assert leaves["source_task_blueprint_sha256"] == leaves["task_semantic_sha256"]
        assert leaves["task_template_pair_sha256"] == identity._pair_identity(
            leaves["task_semantic_sha256"],
            leaves["template_family_sha256"],
            config,
        )


def test_outputs_are_body_free_and_source_read_set_is_metadata_only(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, manifest = built_artifact
    for relative in identity.PARTITION_PATHS:
        values = (
            _load_jsonl(artifact / relative)
            if relative.endswith(".jsonl")
            else [_load_json(artifact / relative)]
        )
        for value in values:
            assert BODY_KEYS.isdisjoint(_walk_keys(value))
    read_set = manifest["read_set"]
    assert read_set["jsonl_inputs_read"] == 0
    assert read_set["protected_body_reads"] == 0
    source_paths = [item["path"] for item in read_set["source_metadata_artifacts"]]
    consumer_paths = [item["path"] for item in read_set["consumer_contract_artifacts"]]
    assert source_paths == [
        relative.as_posix() for relative in SOURCE_RELATIVE_PATHS[:3]
    ]
    assert consumer_paths == [SOURCE_RELATIVE_PATHS[3].as_posix()]
    assert all(
        not value.endswith(".jsonl") for value in [*source_paths, *consumer_paths]
    )
    assert all(
        "gold" not in value.lower() and "heldout" not in value.lower()
        for value in source_paths
    )


def test_manifest_sidecar_and_partition_hashes_counts_are_physical(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    artifact, manifest = built_artifact
    raw_manifest = (artifact / "manifest.json").read_bytes()
    assert (artifact / "manifest.json.sha256").read_bytes() == _sidecar(raw_manifest)
    assert raw_manifest == _canonical_json(manifest)
    assert len(manifest["partitions"]) == 8
    for entry in manifest["partitions"]:
        raw = (artifact / entry["path"]).read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        expected_records = (
            len(raw.splitlines()) if entry["path"].endswith(".jsonl") else 1
        )
        assert entry["records"] == expected_records


def test_all_authorization_and_execution_claims_stay_false(
    built_artifact: tuple[Path, Mapping[str, Any]],
) -> None:
    _, manifest = built_artifact
    claims = manifest["claims"]
    assert set(claims) == {
        "metadata_identity_promotes_execution",
        "descriptor_zero_overlap_proves_real_world_semantic_disjointness",
        "automatic_translation_equivalence_proven",
        "records_materialized",
        "independent_confirmation_executed",
        "controlled_factorial_executed",
        "multi_seed_validated",
        "bundle_generalization_validated",
        "quality_validated",
        "eval_proxy_is_heldout",
        "training_authorized",
        "formal_training_authorized",
        "formal",
    }
    assert all(value is False for value in claims.values())


def test_deterministic_rebuild_is_byte_identical(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    first, first_manifest = built_artifact
    second = tmp_path / "second"
    second_manifest = identity.build_identity_fixture(
        ROOT, SOURCE_ROOT, identity.CONFIG_PATH, second
    )
    assert second_manifest == first_manifest
    assert {
        relative: (first / relative).read_bytes()
        for relative in identity.ARTIFACT_PATHS
    } == {
        relative: (second / relative).read_bytes()
        for relative in identity.ARTIFACT_PATHS
    }


def test_extra_file_is_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    (artifact / "unexpected.json").write_bytes(b"{}\n")
    _assert_error("identity_artifact_layout_invalid", lambda: _audit(artifact))


def test_missing_partition_is_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    (artifact / "inventories/template_family_ids.jsonl").unlink()
    _assert_error("identity_artifact_path_invalid", lambda: _audit(artifact))


@pytest.mark.parametrize(
    "bad_sidecar",
    [
        b"",
        b"0" * 64 + b"  manifest.json\n",
        b"0" * 64 + b" manifest.json\n",
        b"0" * 64 + b"  manifest.json\r\n",
    ],
)
def test_manifest_sidecar_variants_are_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]],
    tmp_path: Path,
    bad_sidecar: bytes,
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    (artifact / "manifest.json.sha256").write_bytes(bad_sidecar)
    _assert_error("identity_manifest_sidecar_invalid", lambda: _audit(artifact))


def test_pair_leaf_tamper_is_rejected_after_real_schema_validation(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    path = artifact / "independent_confirmation/bundles.jsonl"
    records = _load_jsonl(path)
    records[0]["identities"]["task_template_pair_sha256"] = "0" * 64
    path.write_bytes(b"".join(_canonical_json(item) for item in records))
    _assert_error(
        "identity_partition_materialization_mismatch", lambda: _audit(artifact)
    )


def test_duplicate_json_key_in_jsonl_is_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    path = artifact / "discovery/views.jsonl"
    raw = path.read_bytes().replace(
        b'{"curated_bridge_not_automatic_translation_proof"',
        b'{"track":"discovery_bridge","curated_bridge_not_automatic_translation_proof"',
        1,
    )
    path.write_bytes(raw)
    _assert_error("identity_duplicate_json_key", lambda: _audit(artifact))


def test_crlf_jsonl_is_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    path = artifact / "secondary_factorial/bundles.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    _assert_error("identity_partition_jsonl_invalid", lambda: _audit(artifact))


def test_claim_promotion_with_rehashed_manifest_is_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    path = artifact / "manifest.json"
    manifest = _load_json(path)
    manifest["claims"]["training_authorized"] = True
    raw = _canonical_json(manifest)
    path.write_bytes(raw)
    (artifact / "manifest.json.sha256").write_bytes(_sidecar(raw))
    _assert_error(
        "identity_manifest_schema_validation_failed", lambda: _audit(artifact)
    )


def test_proof_claim_promotion_is_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    path = artifact / "proofs/secondary_factorial.json"
    proof = _load_json(path)
    proof["claims"]["controlled_factorial_executed"] = True
    path.write_bytes(_canonical_json(proof))
    _assert_error("identity_proof_schema_validation_failed", lambda: _audit(artifact))


def test_manifest_partition_digest_tamper_with_rehashed_sidecar_is_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    path = artifact / "manifest.json"
    manifest = _load_json(path)
    manifest["partitions"][0]["sha256"] = "0" * 64
    raw = _canonical_json(manifest)
    path.write_bytes(raw)
    (artifact / "manifest.json.sha256").write_bytes(_sidecar(raw))
    _assert_error(
        "identity_manifest_materialization_mismatch", lambda: _audit(artifact)
    )


def test_output_no_replace_is_enforced(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()
    _assert_error(
        "identity_output_already_exists",
        lambda: identity.build_identity_fixture(
            ROOT, SOURCE_ROOT, identity.CONFIG_PATH, output
        ),
    )


def test_forbidden_source_jsonl_read_set_is_rejected_without_opening_it() -> None:
    config = _load_json(ROOT / CONFIG_PATH)
    config["source_binding"]["source_metadata_allowed_reads"][-1] = (
        "fixtures/research/synthetic_nl_scaffold_diagnostic_v1/train/json_only.jsonl"
    )
    _assert_error(
        "identity_source_read_set_not_exact",
        lambda: identity._load_source(SOURCE_ROOT.resolve(strict=True), config),
    )


def test_duplicate_config_key_is_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    producer = _copy_producer_root(tmp_path)
    path = producer / CONFIG_PATH
    raw = path.read_bytes()
    assert raw.startswith(b"{")
    raw = b'{"schema_version":"duplicate",' + raw[1:]
    path.write_bytes(raw)
    _assert_error(
        "identity_duplicate_json_key",
        lambda: _audit(built_artifact[0], repo_root=producer, config_path=CONFIG_PATH),
    )


def test_crlf_config_is_rejected(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    producer = _copy_producer_root(tmp_path)
    path = producer / CONFIG_PATH
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    _assert_error(
        "identity_config_json_invalid",
        lambda: _audit(built_artifact[0], repo_root=producer, config_path=CONFIG_PATH),
    )


def test_forbidden_descriptor_key_is_rejected_before_hashing() -> None:
    config = _load_json(ROOT / CONFIG_PATH)
    descriptor = copy.deepcopy(
        config["discovery_bridge"]["semantic_groups"][0]["descriptor"]
    )
    descriptor["typed_parameters"]["language"] = "en"
    _assert_error(
        "identity_descriptor_forbidden_key",
        lambda: identity._task_identity(descriptor, config),
    )


def test_artifact_symlink_is_rejected_when_supported(
    built_artifact: tuple[Path, Mapping[str, Any]], tmp_path: Path
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    path = artifact / "inventories/task_semantic_ids.jsonl"
    external = tmp_path / "external.jsonl"
    shutil.copyfile(path, external)
    path.unlink()
    try:
        path.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    _assert_error("identity_artifact_path_invalid", lambda: _audit(artifact))


def test_final_artifact_toctou_recheck_detects_post_validation_change(
    built_artifact: tuple[Path, Mapping[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    original = identity._expected_materialization
    changed = False

    def mutate_after_recomputation(*args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            path = artifact / "manifest.json"
            path.write_bytes(path.read_bytes() + b" ")
            changed = True
        return result

    monkeypatch.setattr(
        identity, "_expected_materialization", mutate_after_recomputation
    )
    _assert_error("identity_artifact_toctou_detected", lambda: _audit(artifact))
    assert changed is True


def test_noncanonical_config_path_is_rejected_even_when_it_is_inside_repo() -> None:
    _assert_error(
        "identity_noncanonical_config_forbidden",
        lambda: identity._resolve_config(
            ROOT.resolve(strict=True), ROOT / identity.CONFIG_SCHEMA_PATH
        ),
    )


def test_synchronized_config_and_schema_claim_promotion_cannot_replace_trust_root(
    tmp_path: Path,
) -> None:
    producer = _copy_producer_root(tmp_path)
    module = _load_producer_module(producer)
    config_path = producer / CONFIG_PATH
    schema_path = producer / identity.CONFIG_SCHEMA_PATH
    config = _load_json(config_path)
    schema = _load_json(schema_path)
    config["claims"]["training_authorized"] = True
    schema["$defs"]["claims"]["properties"]["training_authorized"]["const"] = True
    config_path.write_bytes(_canonical_json(config))
    schema_path.write_bytes(_canonical_json(schema))

    _assert_error(
        "identity_pinned_contract_sha256_mismatch",
        lambda: module._load_contract(producer.resolve(strict=True), config_path),
    )


def test_running_implementation_bytes_must_match_import_snapshot(
    tmp_path: Path,
) -> None:
    producer = _copy_producer_root(tmp_path)
    module = _load_producer_module(producer)
    implementation = producer / identity.IMPLEMENTATION_PATH
    implementation.write_bytes(implementation.read_bytes() + b"\n# post-import drift\n")
    _assert_error(
        "identity_running_implementation_mismatch",
        lambda: module._load_contract(
            producer.resolve(strict=True), producer / CONFIG_PATH
        ),
    )


def test_descriptor_value_namespace_salt_is_rejected_even_if_catalog_admits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_json(ROOT / CONFIG_PATH)
    descriptor = copy.deepcopy(
        config["discovery_bridge"]["semantic_groups"][0]["descriptor"]
    )
    allowed, catalog_sha256 = identity._descriptor_atom_catalog(config)
    malicious_atom = "source_bundle_escape"
    monkeypatch.setattr(
        identity,
        "_descriptor_atom_catalog",
        lambda _config: (frozenset({*allowed, malicious_atom}), catalog_sha256),
    )
    descriptor["typed_parameters"]["state_scope"] = malicious_atom
    _assert_error(
        "identity_descriptor_namespace_or_salt_value",
        lambda: identity._task_identity(descriptor, config),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("descriptor_atom_catalog_count", identity.DESCRIPTOR_ATOM_CATALOG_COUNT + 1),
        ("descriptor_atom_catalog_sha256", "0" * 64),
    ],
)
def test_descriptor_closed_atom_catalog_drift_is_rejected(
    field: str, value: object
) -> None:
    config = _load_json(ROOT / CONFIG_PATH)
    config["identity_contract"][field] = value
    descriptor = copy.deepcopy(
        config["discovery_bridge"]["semantic_groups"][0]["descriptor"]
    )
    _assert_error(
        "identity_descriptor_atom_catalog_drift",
        lambda: identity._task_identity(descriptor, config),
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (
            "independent_confirmation",
            "split_algorithm",
            "sha256_unreviewed_split_drift",
        ),
        ("independent_confirmation", "split_domain", "unreviewed.split.domain"),
        ("secondary_controlled_factorial", "quota_rotation", 1),
        (
            "secondary_controlled_factorial",
            "within_factor_pair_assignment_domain",
            "unreviewed.quota.domain",
        ),
    ],
)
def test_split_and_quota_contract_drift_is_rejected(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    producer = _copy_producer_root(tmp_path)
    module = _load_producer_module(producer)
    config_path = producer / CONFIG_PATH
    config = _load_json(config_path)
    config[section][field] = value
    config_path.write_bytes(_canonical_json(config))
    _assert_error(
        "identity_pinned_contract_sha256_mismatch",
        lambda: module._load_contract(producer.resolve(strict=True), config_path),
    )


def _fake_git_authentication_inputs(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    source_root = tmp_path / "source"
    git_dir = source_root / ".git"
    (git_dir / "info").mkdir(parents=True)
    binding: dict[str, Any] = {
        "source_commit": "1" * 40,
        "source_parent_commit": "2" * 40,
        "source_tree": "3" * 40,
        "consumer_plan_commit": "4" * 40,
        "consumer_plan_parent_commit": "5" * 40,
        "consumer_plan_tree": "6" * 40,
        "config": {"path": "source-config.yaml"},
        "manifest": {"path": "source-manifest.json"},
        "manifest_sidecar": {"path": "source-manifest.json.sha256"},
        "consumer_plan_config": {"path": "consumer-plan.yaml"},
    }
    snapshots = {
        "config": SimpleNamespace(data=b"source-config\n"),
        "manifest": SimpleNamespace(data=b"source-manifest\n"),
        "manifest_sidecar": SimpleNamespace(data=b"source-sidecar\n"),
        "consumer_plan_config": SimpleNamespace(data=b"consumer-plan\n"),
    }
    return source_root.resolve(strict=True), binding, snapshots, git_dir


def _install_fake_git_authentication(
    monkeypatch: pytest.MonkeyPatch,
    source_root: Path,
    binding: Mapping[str, Any],
    snapshots: Mapping[str, Any],
    git_dir: Path,
    attack: str,
    tmp_path: Path,
) -> None:
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    source_header = (
        f"tree {binding['source_tree']}\n"
        f"parent {binding['source_parent_commit']}\n"
        "author producer <producer@example.invalid> 0 +0000\n\nsource\n"
    ).encode("ascii")
    plan_header = (
        f"tree {binding['consumer_plan_tree']}\n"
        f"parent {binding['consumer_plan_parent_commit']}\n"
        "author consumer <consumer@example.invalid> 0 +0000\n\nplan\n"
    ).encode("ascii")

    def fake_git(_source_root: Path, arguments: Any, *, check: bool = True) -> bytes:
        del check
        args = tuple(arguments)
        if args == ("rev-parse", "--show-toplevel"):
            root = other_root if attack == "top_level" else source_root
            return f"{root}\n".encode("utf-8")
        if args == ("rev-parse", "--absolute-git-dir"):
            return f"{git_dir}\n".encode("utf-8")
        if args == ("for-each-ref", "--format=%(refname)", "refs/replace/"):
            return b"refs/replace/hostile\n" if attack == "replace" else b""
        if args == ("cat-file", "-p", binding["source_commit"]):
            if attack == "source_commit_shape":
                return source_header.replace(
                    str(binding["source_parent_commit"]).encode("ascii"), b"7" * 40
                )
            return source_header
        if args == ("cat-file", "-p", binding["consumer_plan_commit"]):
            if attack == "plan_commit_shape":
                return plan_header.replace(
                    str(binding["consumer_plan_parent_commit"]).encode("ascii"),
                    b"8" * 40,
                )
            return plan_header
        if len(args) == 3 and args[:2] == ("cat-file", "blob"):
            object_name = str(args[2])
            commit, relative = object_name.split(":", 1)
            if commit == binding["source_commit"]:
                role = next(
                    role
                    for role in ("config", "manifest", "manifest_sidecar")
                    if binding[role]["path"] == relative
                )
                if attack == "source_blob" and role == "config":
                    return b"hostile-source-blob\n"
                return snapshots[role].data
            if commit == binding["consumer_plan_commit"]:
                assert relative == binding["consumer_plan_config"]["path"]
                if attack == "plan_blob":
                    return b"hostile-plan-blob\n"
                return snapshots["consumer_plan_config"].data
        raise AssertionError(f"unexpected fake git arguments: {args!r}")

    monkeypatch.setattr(identity, "_git", fake_git)
    returncode = 1 if attack == "ancestor" else 0
    monkeypatch.setattr(
        identity.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout=b"", stderr=b""
        ),
    )


@pytest.mark.parametrize(
    ("attack", "error_code"),
    [
        ("top_level", "identity_source_git_top_level_mismatch"),
        ("replace", "identity_source_git_replace_refs_forbidden"),
        ("source_commit_shape", "identity_source_git_commit_shape_invalid"),
        ("plan_commit_shape", "identity_consumer_plan_git_commit_shape_invalid"),
        ("source_blob", "identity_source_git_blob_mismatch"),
        ("plan_blob", "identity_consumer_plan_git_blob_mismatch"),
        ("ancestor", "identity_source_not_ancestor_of_consumer_plan"),
    ],
)
def test_git_authentication_attacks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    error_code: str,
) -> None:
    source_root, binding, snapshots, git_dir = _fake_git_authentication_inputs(tmp_path)
    _install_fake_git_authentication(
        monkeypatch,
        source_root,
        binding,
        snapshots,
        git_dir,
        attack,
        tmp_path,
    )
    _assert_error(
        error_code,
        lambda: identity._authenticate_source_git(source_root, binding, snapshots),
    )


def test_git_graft_file_is_rejected_before_object_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, binding, snapshots, git_dir = _fake_git_authentication_inputs(tmp_path)
    (git_dir / "info" / "grafts").write_bytes(b"1" * 40 + b"\n")
    _install_fake_git_authentication(
        monkeypatch,
        source_root,
        binding,
        snapshots,
        git_dir,
        "graft",
        tmp_path,
    )
    _assert_error(
        "identity_source_git_grafts_forbidden",
        lambda: identity._authenticate_source_git(source_root, binding, snapshots),
    )


def test_contract_schema_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    producer = _copy_producer_root(tmp_path)
    module = _load_producer_module(producer)
    schema = producer / identity.CONFIG_SCHEMA_PATH
    external = tmp_path / "external-config-schema.json"
    shutil.copyfile(schema, external)
    schema.unlink()
    try:
        schema.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    _assert_error(
        "identity_input_path_invalid",
        lambda: module._load_contract(
            producer.resolve(strict=True), producer / CONFIG_PATH
        ),
    )


def test_source_metadata_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    relative = SOURCE_RELATIVE_PATHS[0]
    target = source / relative
    target.parent.mkdir(parents=True)
    external = tmp_path / "external-source-metadata.yaml"
    shutil.copyfile(SOURCE_ROOT / relative, external)
    try:
        target.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink capability unavailable: {exc}")
    _assert_error(
        "identity_input_path_invalid",
        lambda: identity._capture_declared(
            source.resolve(strict=True), relative.as_posix()
        ),
    )


def test_second_git_authentication_window_is_followed_by_final_toctou_recheck(
    built_artifact: tuple[Path, Mapping[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _copy_artifact(built_artifact[0], tmp_path)
    original = identity._authenticate_source_git
    calls = 0

    def authenticate_then_mutate(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        original(*args, **kwargs)
        calls += 1
        if calls == 2:
            path = artifact / "manifest.json"
            path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(identity, "_authenticate_source_git", authenticate_then_mutate)
    _assert_error("identity_artifact_final_toctou_detected", lambda: _audit(artifact))
    assert calls == 2


def test_publish_race_after_prepublication_audit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "publish-race"
    original = identity.audit_identity_fixture
    calls = 0

    def audit_then_claim_destination(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            output.mkdir()
        return result

    monkeypatch.setattr(
        identity, "audit_identity_fixture", audit_then_claim_destination
    )
    _assert_error(
        "identity_output_publish_race",
        lambda: identity.build_identity_fixture(
            ROOT, SOURCE_ROOT, identity.CONFIG_PATH, output
        ),
    )
    assert calls == 1
