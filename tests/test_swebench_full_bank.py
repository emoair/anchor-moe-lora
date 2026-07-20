from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from anchor_mvp.swebench.full_bank import (
    FIFTY_MIB,
    FullBankConfig,
    _validate_opencode_bundle,
    build_full_bank,
    preflight_full_bank,
    refresh_hash_only_manifest_from_public,
)
from anchor_mvp.swebench.schema import (
    SWEBenchValidationError,
    clean_problem_statement,
)
from anchor_mvp.tooling.tool_contract import EXECUTION_TOOL_CONTRACT_V3_VERSION


REVISION = "7" * 40


def _config(root: Path) -> dict[str, object]:
    return {
        "schema_version": "anchor.swebench-full-bank-config.v2",
        "source": {
            "dataset_id": "SWE-bench/SWE-bench",
            "dataset_revision": REVISION,
            "split": "train",
            "parquet": "artifacts/source/train.parquet",
            "parquet_sha256": "0" * 64,
            "expected_rows": 4,
        },
        "output_dir": "artifacts/full-bank",
        "public_manifest": "configs/data/manifests/full-bank.hash-only.json",
        "split": {
            "algorithm": "per_repo_sha256_rank_v1",
            "validation_numerator": 1,
            "validation_denominator": 2,
        },
        "bilingual": {
            "locales": ["en-US", "zh-CN"],
            "assignment": "global_sha256_rank_alternating_v1",
            "require_localized_text_before_live": True,
        },
        "publication": {
            "shard_rows": 1,
            "max_file_bytes": FIFTY_MIB,
            "raw_source_publishable": False,
            "audited_export_dir": "datasets/public/swebench-full-bank-v1",
        },
        "gates": {
            "launch": {
                "opencode_bundle_manifest": "artifacts/tooling/opencode/bundle.json",
                "ccswitch_route_manifest": "artifacts/tooling/ccswitch/route.json",
            },
            "training": {
                "gold_manifest": "artifacts/gates/gold.json",
                "zh_cn_localization_manifest": "artifacts/gates/zh.json",
                "real_tool_results_manifest": "artifacts/gates/tool-results.json",
            },
            "publication": {
                "mit_attribution_file": (
                    "datasets/public/swebench-full-bank-v1/ATTRIBUTION.md"
                ),
            },
        },
        "providers": {
            "broad": {
                "provider": "custom-openai-responses",
                "protocol": "openai_responses",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "TEACHER_API_KEY",
                "model_id": "broad-model",
                "user_agent": "claude-code/test",
                "reasoning_effort": "max",
                "discover_models": False,
                "force_manual_model": True,
                "discovery_failure_policy": "require_manual_model",
            },
            "frontend": {
                "provider": "custom-openai-responses",
                "protocol": "openai_responses",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "TEACHER_API_KEY",
                "model_id": "frontend-model",
                "user_agent": "claude-code/test",
                "reasoning_effort": "max",
                "discover_models": False,
                "force_manual_model": True,
                "discovery_failure_policy": "require_manual_model",
            },
        },
        "stage_routes": {
            "planner": {
                "default_provider": "broad",
                "frontend_provider": "broad",
                "execution": "teacher-json",
            },
            "tool_policy": {
                "default_provider": "broad",
                "frontend_provider": "broad",
                "execution": "teacher-json",
            },
            "domain_builder": {
                "default_provider": "broad",
                "frontend_provider": "frontend",
                "execution": "controlled-opencode-sandbox",
            },
            "domain_review": {
                "default_provider": "broad",
                "frontend_provider": "frontend",
                "execution": "teacher-json",
            },
            "security": {
                "default_provider": "broad",
                "frontend_provider": "broad",
                "execution": "teacher-json",
            },
        },
        "classification": {"frontend_keywords": ["css", "frontend", "ui"]},
        "formal_profile": {
            "enabled": True,
            "required_reasoning_effort": "max",
            "required_provider_aliases": ["broad", "frontend"],
            "require_real_sandbox": True,
            "capture_tool_calls": True,
            "capture_tool_results": True,
            "capture_hidden_chain_of_thought": False,
            "capture_public_reasoning_summary": True,
        },
    }


def _write_config(root: Path, value: dict[str, object]) -> Path:
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_parquet(root: Path, value: dict[str, object]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [
        {
            "repo": f"org/repo-{index // 2}",
            "instance_id": f"issue-{index}",
            "base_commit": f"{index + 1:040x}",
            "problem_statement": f"Repair public behavior {index}.",
            "patch": "excluded",
            "test_patch": "excluded",
            "hints_text": "excluded",
            "FAIL_TO_PASS": "excluded",
            "PASS_TO_PASS": "excluded",
        }
        for index in range(4)
    ]
    source = root / "artifacts/source/train.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), source)
    value["source"]["parquet_sha256"] = _sha256(source)  # type: ignore[index]


def _write_launch_attestations(root: Path) -> None:
    opencode = root / "artifacts/tooling/opencode"
    opencode.mkdir(parents=True)
    platforms: dict[str, object] = {}
    for target, binary_name in (
        ("windows-x64", "windows-x64/opencode-anchor.exe"),
        ("linux-x64", "linux-x64/opencode-anchor"),
    ):
        binary = opencode / binary_name
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(f"{target}-binary".encode())
        platform_manifest = opencode / f"{target}.manifest.json"
        platform_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "anchor.patched-opencode.platform.v1",
                    "target": target,
                }
            ),
            encoding="utf-8",
        )
        platforms[target] = {
            "manifest": platform_manifest.name,
            "manifest_sha256": _sha256(platform_manifest),
            "binary": {
                "path": binary.relative_to(opencode).as_posix(),
                "sha256": _sha256(binary),
            },
        }
    (opencode / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": "anchor.patched-opencode.bundle.v1",
                "source": {
                    "tool_contract_version": EXECUTION_TOOL_CONTRACT_V3_VERSION,
                    "tool_contract": {
                        "version": EXECUTION_TOOL_CONTRACT_V3_VERSION
                    },
                },
                "platforms": platforms,
            }
        ),
        encoding="utf-8",
    )

    patch = root / "patches/ccswitch.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("pinned patch", encoding="utf-8")
    binary = root / "artifacts/tooling/ccswitch/ccswitch.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"ccswitch-binary")
    route = binary.parent / "route.json"
    route.write_text(
        json.dumps(
            {
                "schema_version": "anchor.ccswitch-route-manifest.v1",
                "ready": True,
                "secret_persisted": False,
                "route": {
                    "app_type": "anchor-opencode",
                    "base_url": "http://127.0.0.1:15731/anchor/v1",
                    "content_free_health_status": True,
                },
                "patch": {
                    "path": patch.relative_to(root).as_posix(),
                    "sha256": _sha256(patch),
                },
                "binary": {
                    "path": binary.relative_to(root).as_posix(),
                    "sha256": _sha256(binary),
                },
                "verified_tests": [{"name": "offline", "status": "passed"}],
            }
        ),
        encoding="utf-8",
    )


def _write_training_gates(root: Path, value: dict[str, object]) -> None:
    source = value["source"]  # type: ignore[index]
    common_source = {
        "dataset_id": source["dataset_id"],
        "dataset_revision": source["dataset_revision"],
        "split": "train",
        "parquet_sha256": source["parquet_sha256"],
    }
    gate_paths = value["gates"]["training"]  # type: ignore[index]
    definitions = {
        "gold_manifest": (
            "anchor.swebench-gold-manifest.v1",
            {"gold_records": True},
        ),
        "zh_cn_localization_manifest": (
            "anchor.swebench-zh-cn-localization-manifest.v1",
            {"locale": "zh-CN"},
        ),
        "real_tool_results_manifest": (
            "anchor.swebench-real-tool-results-manifest.v1",
            {"real_tool_results": True},
        ),
    }
    for name, (schema, extra) in definitions.items():
        payload = root / f"artifacts/runtime/{name}.jsonl"
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text('{"record":"content stays local"}\n', encoding="utf-8")
        manifest_path = root / gate_paths[name]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": schema,
                    "source": common_source,
                    "complete": True,
                    "train_only": True,
                    "contains_heldout": False,
                    "record_count": 1,
                    "files": [
                        {
                            "path": payload.relative_to(root).as_posix(),
                            "sha256": _sha256(payload),
                            "records": 1,
                        }
                    ],
                    **extra,
                }
            ),
            encoding="utf-8",
        )


def test_formal_profile_rejects_any_non_max_teacher(tmp_path: Path) -> None:
    value = _config(tmp_path)
    value["providers"]["frontend"]["reasoning_effort"] = "high"  # type: ignore[index]
    path = _write_config(tmp_path, value)
    with pytest.raises(SWEBenchValidationError, match="reasoning_effort=max"):
        FullBankConfig.load(tmp_path, path)


def test_provider_schema_allows_discovery_with_manual_failure_policy(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path)
    provider = value["providers"]["frontend"]  # type: ignore[index]
    provider["model_id"] = None
    provider["discover_models"] = True
    provider["force_manual_model"] = False
    path = _write_config(tmp_path, value)
    config = FullBankConfig.load(tmp_path, path)
    assert config.providers["frontend"].model_id is None
    assert config.providers["frontend"].discovery_failure_policy == "require_manual_model"


def test_inline_key_and_oversized_publication_fail_closed(tmp_path: Path) -> None:
    value = _config(tmp_path)
    value["providers"]["broad"]["api_key"] = "secret"  # type: ignore[index]
    path = _write_config(tmp_path, value)
    with pytest.raises(SWEBenchValidationError, match="unexpected fields"):
        FullBankConfig.load(tmp_path, path)

    value = _config(tmp_path)
    value["publication"]["max_file_bytes"] = FIFTY_MIB + 1  # type: ignore[index]
    path = _write_config(tmp_path, value)
    with pytest.raises(SWEBenchValidationError, match="50 MiB"):
        FullBankConfig.load(tmp_path, path)


def test_official_long_train_issue_fits_card_contract() -> None:
    assert len(clean_problem_statement("x" * 256_368)) == 256_368
    with pytest.raises(SWEBenchValidationError, match="300000"):
        clean_problem_statement("x" * 300_001)


def test_reboot_preflight_uses_only_source_and_component_attestations(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path)
    _write_source_parquet(tmp_path, value)
    _write_launch_attestations(tmp_path)
    config = FullBankConfig.load(tmp_path, _write_config(tmp_path, value))

    report = preflight_full_bank(config)

    assert report["offline"] is True
    assert report["provider_requests"] == 0
    assert report["launch_ready"] is True
    assert report["training_ready"] is False
    assert report["publication_ready"] is False
    assert report["launch_gate_requires_runtime_outputs"] is False
    assert set(report["gates"]["launch"]["checks"]) == {
        "source_train_parquet",
        "formal_route_contract",
        "opencode_bundle_manifest",
        "ccswitch_route_manifest",
    }
    assert not (
        set(report["gates"]["launch"]["checks"])
        & set(report["gates"]["training"]["checks"])
    )


def test_opencode_launch_gate_accepts_v3_and_rejects_legacy_v2(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path)
    _write_launch_attestations(tmp_path)
    bundle_path = tmp_path / "artifacts/tooling/opencode/bundle.json"
    config = FullBankConfig.load(tmp_path, _write_config(tmp_path, value))

    accepted = _validate_opencode_bundle(config, bundle_path)

    assert accepted["errors"] == []
    assert accepted["validated"] is True

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["source"]["tool_contract_version"] = (
        "anchor.execution-tool-contract.v2"
    )
    bundle["source"]["tool_contract"]["version"] = (
        "anchor.execution-tool-contract.v2"
    )
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    rejected = _validate_opencode_bundle(config, bundle_path)

    assert rejected["validated"] is False
    assert "tool_contract_mismatch" in rejected["errors"]


def test_training_gate_is_independent_of_current_launch_components(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path)
    _write_source_parquet(tmp_path, value)
    _write_training_gates(tmp_path, value)
    config = FullBankConfig.load(tmp_path, _write_config(tmp_path, value))

    report = preflight_full_bank(config)

    assert report["launch_ready"] is False
    assert report["training_ready"] is True
    assert report["publication_ready"] is False
    assert report["gates"]["training"]["missing"] == []
    assert report["gates"]["training"]["invalid"] == []


def test_small_train_parquet_builds_all_candidates_but_stays_blocked(
    tmp_path: Path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    ark_credential = "ark-" + "00000000-0000-4000-8000-000000000000-demo"
    spark_identifier = "spark-" + "00000000-0000-4000-8000-000000000000"
    rows = [
        {
            "repo": "org/frontend",
            "instance_id": f"frontend-{index}",
            "base_commit": f"{index + 1:040x}",
            "problem_statement": (
                "Repair the CSS UI without changing the public API. "
                "Example sk-abcdefghijklmnopqrstuvwxyz must be scrubbed.\n"
                if index == 0
                else (
                    "Repair the parser regression. Scrub "
                    f"{ark_credential}, but retain {spark_identifier}.\n"
                )
            ),
            "patch": "must never be projected",
            "test_patch": "must never be projected",
            "hints_text": "must never be projected",
            "FAIL_TO_PASS": "must never be projected",
            "PASS_TO_PASS": "must never be projected",
        }
        for index in range(2)
    ] + [
        {
            "repo": "org/backend",
            "instance_id": f"backend-{index}",
            "base_commit": f"{index + 11:040x}",
            "problem_statement": "Repair transaction ordering.\n",
            "patch": "must never be projected",
            "test_patch": "must never be projected",
            "hints_text": "must never be projected",
            "FAIL_TO_PASS": "must never be projected",
            "PASS_TO_PASS": "must never be projected",
        }
        for index in range(2)
    ]
    source = tmp_path / "artifacts" / "source" / "train.parquet"
    source.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), source)
    value = _config(tmp_path)
    from anchor_mvp.swebench.partition import file_sha256

    value["source"]["parquet_sha256"] = file_sha256(source)  # type: ignore[index]
    config = FullBankConfig.load(tmp_path, _write_config(tmp_path, value))
    result = build_full_bank(config)
    manifest = result.manifest
    assert manifest["source"]["row_count"] == 4
    assert manifest["source"]["oracle_fields_projected"] is False
    assert manifest["derived_split"]["train_count"] == 2
    assert manifest["derived_split"]["validation_count"] == 2
    assert manifest["bilingual"]["counts"] == {"en-US": 2, "zh-CN": 2}
    assert manifest["routing"]["work_order_count"] == 20
    assert manifest["launch_ready"] is False
    assert manifest["training_ready"] is False
    assert manifest["publication_ready"] is True
    assert manifest["launch_gate_requires_runtime_outputs"] is False
    assert manifest["sandbox_results_claimed"] is False
    assert all(
        item["bytes"] < FIFTY_MIB
        for item in manifest["publication"]["local_staging_files"]
    )
    public_manifest = tmp_path / "configs/data/manifests/full-bank.hash-only.json"
    public = json.loads(public_manifest.read_text(encoding="utf-8"))
    assert public["source"]["row_count"] == 4
    manifest_text = public_manifest.read_text(encoding="utf-8")
    assert "Repair the CSS UI" not in manifest_text
    assert "Repair transaction ordering" not in manifest_text

    export = tmp_path / "datasets/public/swebench-full-bank-v1"
    exported_manifest = json.loads((export / "manifest.json").read_text("utf-8"))
    assert exported_manifest["publication_ready"] is True
    assert exported_manifest["source_split"] == "train"
    assert exported_manifest["attribution"]["upstream_license"] == "MIT"
    assert exported_manifest["safety"]["credential_redaction_count"] == 2
    source_shards = sorted(export.glob("source-metadata.train*.jsonl"))
    source_records = [
        json.loads(line)
        for shard in source_shards
        for line in shard.read_text(encoding="utf-8").splitlines()
    ]
    assert len(source_records) == 4
    assert all(
        set(row) == {"repo", "instance_id", "base_commit", "problem_statement"}
        for row in source_records
    )
    exported_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in export.rglob("*")
        if path.is_file()
    )
    for forbidden in (
        "must never be projected",
        '"test_patch"',
        '"hints_text"',
        '"FAIL_TO_PASS"',
        '"PASS_TO_PASS"',
        '"api_key"',
        '"api_key_env"',
        "sk-abcdefghijklmnopqrstuvwxyz",
        ark_credential,
    ):
        assert forbidden not in exported_text
    assert "[REDACTED_CREDENTIAL]" in exported_text
    assert spark_identifier in exported_text
    assert all(
        path.stat().st_size < FIFTY_MIB
        for path in export.rglob("*")
        if path.is_file()
    )


def test_checked_in_formal_profile_locks_both_teachers_to_max() -> None:
    root = Path(__file__).resolve().parents[1]
    config = FullBankConfig.load(
        root, root / "configs/data/swebench_full_bank.formal.yaml"
    )
    assert config.expected_rows == 19_008
    assert config.dataset_revision == "7074ef12ea2a6f70a228943c1336553333c22786"
    assert config.providers["glm52_max"].reasoning_effort == "max"
    assert config.providers["kimi_k3_max"].reasoning_effort == "max"
    assert config.providers["kimi_k3_max"].model_id == "kimi-k3"
    assert config.stage_routes["domain_builder"].frontend_provider == "kimi_k3_max"


def _built_refresh_fixture(tmp_path: Path) -> FullBankConfig:
    value = _config(tmp_path)
    _write_source_parquet(tmp_path, value)
    config = FullBankConfig.load(tmp_path, _write_config(tmp_path, value))
    build_full_bank(config)
    return config


def test_hash_only_refresh_copies_exact_public_payload_inventory(
    tmp_path: Path,
) -> None:
    config = _built_refresh_fixture(tmp_path)
    public_path = config.audited_export_dir / "manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))

    result = refresh_hash_only_manifest_from_public(config)

    publication = result.manifest["publication"]
    assert publication["public_manifest_sha256"] == _sha256(public_path)
    assert publication["payload_file_count"] == 15
    assert publication["public_file_count"] == 16
    assert publication["payload_inventory"] == sorted(
        public["files"], key=lambda item: item["path"]
    )
    assert publication["counts"] == public["counts"]
    assert result.manifest["publication_ready"] is public["publication_ready"]


def test_hash_only_refresh_rejects_payload_tamper(tmp_path: Path) -> None:
    config = _built_refresh_fixture(tmp_path)
    public = json.loads(
        (config.audited_export_dir / "manifest.json").read_text(encoding="utf-8")
    )
    payload = config.audited_export_dir / public["files"][0]["path"]
    payload.write_bytes(payload.read_bytes() + b"tamper")

    with pytest.raises(SWEBenchValidationError, match="payload binding mismatch"):
        refresh_hash_only_manifest_from_public(config)


def test_hash_only_refresh_rejects_count_mismatch(tmp_path: Path) -> None:
    config = _built_refresh_fixture(tmp_path)
    public_path = config.audited_export_dir / "manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public["counts"]["tasks"] -= 1
    public_path.write_text(json.dumps(public), encoding="utf-8")

    with pytest.raises(SWEBenchValidationError, match="counts are invalid"):
        refresh_hash_only_manifest_from_public(config)


def test_hash_only_refresh_rejects_path_mismatch(tmp_path: Path) -> None:
    config = _built_refresh_fixture(tmp_path)
    public_path = config.audited_export_dir / "manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public["files"][0]["path"] = "../escape.jsonl"
    public_path.write_text(json.dumps(public), encoding="utf-8")

    with pytest.raises(SWEBenchValidationError, match="inventory item is invalid"):
        refresh_hash_only_manifest_from_public(config)


def test_hash_only_refresh_rejects_record_count_mismatch(tmp_path: Path) -> None:
    config = _built_refresh_fixture(tmp_path)
    public_path = config.audited_export_dir / "manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public["files"][0]["records"] += 1
    public_path.write_text(json.dumps(public), encoding="utf-8")

    with pytest.raises(SWEBenchValidationError, match="record counts are invalid"):
        refresh_hash_only_manifest_from_public(config)


def test_publication_audit_rejects_a_forbidden_structured_field(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path)
    _write_source_parquet(tmp_path, value)
    config = FullBankConfig.load(tmp_path, _write_config(tmp_path, value))
    build_full_bank(config)
    shard = next(
        (tmp_path / "datasets/public/swebench-full-bank-v1").glob(
            "source-metadata.train*.jsonl"
        )
    )
    rows = [json.loads(line) for line in shard.read_text("utf-8").splitlines()]
    rows[0]["test_patch"] = "must remain private"
    shard.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = preflight_full_bank(config)

    publication = report["gates"]["publication"]["checks"]["audited_export"]
    assert report["publication_ready"] is False
    assert any(error.startswith("forbidden_field:") for error in publication["errors"])


def test_gitignore_public_dataset_allowlist_is_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    required = {
        "datasets/*",
        "!datasets/public/",
        "datasets/public/*",
        "!datasets/public/swebench-full-bank-v1/",
        "datasets/public/swebench-full-bank-v1/*",
        "!datasets/public/swebench-full-bank-v1/ATTRIBUTION.md",
        "!datasets/public/swebench-full-bank-v1/manifest.json",
        "!datasets/public/swebench-full-bank-v1/allowlists/train.json",
        (
            "!datasets/public/swebench-full-bank-v1/allowlists/"
            "validation-from-train.json"
        ),
    }
    assert required.issubset(lines)
    assert "!datasets/**" not in lines
    assert "!datasets/public/**" not in lines


def test_full_bank_guides_are_bilingual_and_describe_split_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    english = (root / "docs/swebench_full_bank.md").read_text(encoding="utf-8")
    chinese = (root / "docs/swebench_full_bank.zh-CN.md").read_text(
        encoding="utf-8"
    )
    for document in (english, chinese):
        assert "launch_ready" in document
        assert "training_ready" in document
        assert "publication_ready" in document
        assert "--preflight" in document
        assert "Get-NetAdapter -Physical" in document
        assert "SPDX-License-Identifier: MIT" in document
        assert "52,428,800" in document
    assert "运行后产物，绝不是启动输入" in chinese
