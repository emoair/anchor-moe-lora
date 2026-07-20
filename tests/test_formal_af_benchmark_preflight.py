from __future__ import annotations

import asyncio
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anchor_mvp.benchmark.formal_af_preflight import (
    EXPECTED_PARAMETERS,
    EXPECTED_RANKS,
    FormalAFPreflightError,
    preflight,
    validate_contract,
)
from anchor_mvp.benchmark.heldout import HeldoutGateError
from anchor_mvp.benchmark.heldout_cli import (
    _enforce_formal_af_preflight,
    _require_heldout_authorization,
    _run_formal_live,
    _verify_local_serial_endpoints,
    _verify_model_catalog,
    build_parser,
)
from anchor_mvp.benchmark.heldout_eval import (
    MATCHED_FIVE_STAGE_BASELINES,
    _evaluate_record,
)
from anchor_mvp.benchmark.models import BenchmarkCase, BenchmarkRecord
from anchor_mvp.benchmark.segment_protocol import (
    PROMPT_BUNDLE_SHA256,
    PROMPT_BUNDLE_VERSION,
    prompt_bundle_payload,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark" / "formal_partial_v1_af.json"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _synthetic_project(tmp_path: Path, *, omit: str | None = None) -> Path:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_path = tmp_path / "formal.json"

    heldout_manifest = tmp_path / "metadata" / "heldout-manifest.json"
    leak_audit = tmp_path / "metadata" / "leak-audit.json"
    _write_json(heldout_manifest, {"metadata_only": True})
    _write_json(leak_audit, {"status": "PASS", "content_emitted": False})
    config["heldout_binding"].update(
        {
            "manifest_path": heldout_manifest.relative_to(tmp_path).as_posix(),
            "manifest_sha256": _sha(heldout_manifest),
            "leak_audit_path": leak_audit.relative_to(tmp_path).as_posix(),
            "leak_audit_sha256": _sha(leak_audit),
        }
    )

    for arm in config["baselines"]:
        group = arm["registry_group"]
        if group not in {"E", "F"}:
            continue
        allocation = tmp_path / "allocations" / f"{group}.json"
        _write_json(allocation, {"group": group, "frozen": True})
        arm["allocation_manifest_path"] = allocation.relative_to(tmp_path).as_posix()
        arm["allocation_manifest_sha256"] = _sha(allocation)

    run_path = tmp_path / config["run_manifest_path"]
    groups = {}
    for group in "ABCDEF":
        registry_path = run_path.parent / group / "group_registry.json"
        groups[group] = {
            "registry_path": registry_path.relative_to(tmp_path).as_posix(),
            "artifact_root": None if group == "A" else f"artifacts/{group}",
        }
        if group == omit:
            continue
        arm = next(item for item in config["baselines"] if item["registry_group"] == group)
        artifacts = sorted(set(arm["stage_adapter_artifacts"].values()))
        adapter_records = []
        for artifact in artifacts:
            adapter_model = tmp_path / "artifacts" / group / artifact / "adapter_model.safetensors"
            adapter_model.parent.mkdir(parents=True, exist_ok=True)
            adapter_model.write_bytes(b"synthetic")
            adapter_records.append(
                {
                    "artifact_name": artifact,
                    "adapter_sha256": sha256(artifact.encode()).hexdigest(),
                    "final_files": {
                        "adapter_model": {
                            "path": adapter_model.relative_to(tmp_path).as_posix(),
                            "sha256": _sha(adapter_model),
                        }
                    },
                }
            )
        registry = {
            "schema_version": "anchor.af-group-registry.v1",
            "run_id": config["run_id"],
            "group": group,
            "status": "registered" if group == "A" else "completed",
            "adapters": adapter_records,
            "adapter_summary": {
                "ranks": EXPECTED_RANKS[group],
                "trainable_parameter_total": EXPECTED_PARAMETERS[group],
            },
        }
        _write_json(registry_path, registry)
    _write_json(
        run_path,
        {
            "schema_version": "anchor.af-run-manifest.v1",
            "run_id": config["run_id"],
            "base_artifact": {
                "format": config["base_binding"]["format"],
                "manifest_sha256": config["base_binding"]["q4_artifact_sha256"],
                "source_weight_sha256": config["base_binding"]["base_source_sha256"],
                "weight_set_sha256": config["base_binding"]["weight_set_sha256"],
            },
            "dataset_snapshot": config["dataset_binding"],
            "groups": groups,
        },
    )
    _write_json(config_path, config)
    return config_path


def _segmented_synthetic_project(tmp_path: Path) -> Path:
    config_path = _synthetic_project(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = "compact-mvp-v2b-formal-v2-test"
    config["run_id"] = run_id

    token_contract = {
        "stages": ["planner", "tool_policy", "frontend", "review", "security"],
        "review_protocol": "segmented_repair_v1",
        "artifact_protocol": "single_file_tsx_segmented_v1",
        "segment_contract_version": "anchor.segmented-eval.v1",
        "frontend_segment_count": 10,
        "review_segment_count": 10,
        "expected_physical_calls": 23,
        "max_tokens_per_call": 512,
        "sampling": {"temperature": 0.0, "top_p": 1.0},
    }
    config["token_contract"] = token_contract
    for arm in config["baselines"]:
        arm.update(
            {
                "review_protocol": "segmented_repair_v1",
                "artifact_protocol": "single_file_tsx_segmented_v1",
                "segment_contract_version": "anchor.segmented-eval.v1",
                "frontend_segment_count": 10,
                "review_segment_count": 10,
                "max_tokens_per_call": 512,
            }
        )

    public = tmp_path / "public"
    segment_contract = public / "segment-contract.json"
    _write_json(
        segment_contract,
        {
            "schema_version": "anchor.segmented-eval-contract.v1",
            "artifact_protocol": "single_file_tsx_segmented_v1",
            "segment_contract_version": "anchor.segmented-eval.v1",
            "review_protocol": "segmented_repair_v1",
            "frontend_segment_count": 10,
            "review_segment_count": 10,
            "expected_physical_calls": 23,
            "max_completion_tokens_per_physical_call": 512,
        },
    )
    prompt_implementation = public / "segment_protocol.py"
    prompt_implementation.write_text("# synthetic public prompt module\n", encoding="utf-8")
    prompt_bundle = public / "prompt-bundle.json"
    _write_json(
        prompt_bundle,
        {
            "schema_version": "anchor.formal-v2-prompt-bundle.v1",
            "prompt_bundle_version": PROMPT_BUNDLE_VERSION,
            "canonical_prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
            "implementation": {
                "path": prompt_implementation.relative_to(tmp_path).as_posix(),
                "sha256": _sha(prompt_implementation),
            },
            "payload": prompt_bundle_payload(),
        },
    )
    processor_manifest = public / "processor.json"
    processor_tree_sha = sha256(b"synthetic-processor-tree").hexdigest()
    _write_json(
        processor_manifest,
        {
            "schema_version": "anchor.formal-af-processor.v1",
            "tree_sha256": processor_tree_sha,
        },
    )
    config["segment_contract_binding"] = {
        "path": segment_contract.relative_to(tmp_path).as_posix(),
        "sha256": _sha(segment_contract),
    }
    config["prompt_bundle_binding"] = {
        "path": prompt_bundle.relative_to(tmp_path).as_posix(),
        "sha256": _sha(prompt_bundle),
    }
    config["processor_binding"] = {
        "manifest_path": processor_manifest.relative_to(tmp_path).as_posix(),
        "manifest_sha256": _sha(processor_manifest),
        "tree_sha256": processor_tree_sha,
    }

    run_path = tmp_path / config["run_manifest_path"]
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["run_id"] = run_id
    _write_json(run_path, run)
    for group in "ABCDEF":
        registry_path = tmp_path / run["groups"][group]["registry_path"]
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry["run_id"] = run_id
        _write_json(registry_path, registry)
    _write_json(config_path, config)
    return config_path


def test_formal_contract_freezes_all_six_arms_and_fair_comparisons() -> None:
    config = validate_contract(CONFIG)

    assert config["run_id"] == "formal-partial-v1-forced-20260715-v1"
    assert [arm["registry_group"] for arm in config["baselines"]] == list("ABCDEF")
    assert {arm["max_tokens_per_call"] for arm in config["baselines"]} == {1024}
    assert config["metrics_plan"]["index_baseline"] == "A"
    assert config["metrics_plan"]["index_value"] == 100
    assert config["metrics_plan"]["equal_budget_comparison"] == ["B", "D", "F"]
    assert config["metrics_plan"]["capacity_comparison"] == ["C", "E"]


def test_e_and_f_stay_disclosed_as_calibration_pending() -> None:
    config = validate_contract(CONFIG)
    by_group = {arm["registry_group"]: arm for arm in config["baselines"]}

    assert by_group["E"]["stage_adapter_ranks"] == {
        "planner": 8,
        "tool_policy": 4,
        "frontend": 16,
        "review": 12,
        "security": 4,
    }
    assert by_group["F"]["stage_adapter_ranks"] == {
        "planner": 4,
        "tool_policy": 1,
        "frontend": 6,
        "review": 4,
        "security": 1,
    }
    assert by_group["E"]["calibration_status"] == "calibration_pending"
    assert by_group["F"]["calibration_status"] == "calibration_pending"
    assert "not a measured Pareto optimum" in by_group["E"]["notes"]


def test_llama_cpp_is_not_silently_treated_as_same_frozen_backend() -> None:
    config = validate_contract(CONFIG)
    audit = config["backend_audit"]["llama_cpp"]

    assert audit["eligible_for_this_frozen_contract"] is False
    assert audit["worktree_clean"] is False
    assert audit["local_modified_files"] == ["src/models/gemma4.cpp"]
    assert any("bitsandbytes NF4" in item for item in audit["blocking_gaps"])
    assert any("PEFT safetensors" in item for item in audit["blocking_gaps"])
    assert any("per-request lora field" in item for item in audit["blocking_gaps"])


def test_vllm_formal_backend_is_frozen_to_one_runtime_lora() -> None:
    config = validate_contract(CONFIG)
    serial = config["backend_audit"]["vllm_serial_runtime_lora"]

    assert serial["eligible"] is True
    assert serial["catalog_gate"] == "base_model_only_before_heldout"
    assert serial["maximum_active_loras"] == 1
    assert serial["maximum_cpu_loras"] == 1
    assert serial["allow_static_lora_modules"] is False
    assert serial["require_localhost_admin"] is True
    assert serial["server_project_root_transport"] == "explicit_absolute_posix"
    assert serial["server_launcher"] == (
        "scripts/serve/start_formal_af_serial_vllm.ps1"
    )
    assert serial["benchmark_launcher"] == (
        "scripts/benchmark/run_formal_partial_v1_af_serial.ps1"
    )


def test_static_contract_rejects_a_changed_f_budget(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    arm_f = next(arm for arm in config["baselines"] if arm["registry_group"] == "F")
    arm_f["stage_adapter_ranks"]["security"] = 2
    changed = tmp_path / "changed.json"
    _write_json(changed, config)

    with pytest.raises(FormalAFPreflightError, match="sum exactly|rank allocation"):
        validate_contract(changed)


def test_preflight_blocks_before_hashing_when_one_registry_is_missing(
    tmp_path: Path,
) -> None:
    config = _synthetic_project(tmp_path, omit="F")

    with pytest.raises(FormalAFPreflightError) as caught:
        preflight(config, tmp_path)

    assert caught.value.code == "training_incomplete"
    assert caught.value.details == {"missing_registry_groups": ["F"]}


def test_ready_preflight_never_opens_heldout_case_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_project(tmp_path)
    seen_groups: list[str] = []

    def fake_verify_registry(project_root, run_root, *, group):
        del project_root, run_root
        seen_groups.append(group)
        return {"registry_sha256": sha256(group.encode()).hexdigest()}

    monkeypatch.setattr(
        "anchor_mvp.benchmark.formal_af_preflight.verify_registry",
        fake_verify_registry,
    )
    result = preflight(config, tmp_path)

    assert seen_groups == list("ABCDEF")
    assert result["status"] == "ready"
    assert result["execution_authorized"] is True
    assert result["heldout_case_content_read"] is False
    assert result["review_protocol"] == "repair_code_v1"
    assert result["repair_code_review_calls"] == 1
    assert result["runtime_bindings"]["A"]["planner"]["adapter_dir"] is None
    assert result["runtime_bindings"]["F"]["frontend"] == {
        "model_id": "fpv1-f-frontend-r6",
        "adapter_artifact": "frontend_gen-r6",
        "adapter_dir": "artifacts/F/frontend_gen-r6",
        "adapter_sha256": sha256(b"frontend_gen-r6").hexdigest(),
    }
    assert result["comparison_plan"] == {
        "A_index": 100,
        "equal_budget": ["B", "D", "F"],
        "capacity": ["C", "E"],
        "E_calibration_status": "calibration_pending",
    }
    assert result["serial_runtime_contract"] == {
        "base_model_id": "gemma4-12b-base-q4",
        "maximum_active_loras": 1,
        "maximum_cpu_loras": 1,
        "catalog_gate": "base_model_only_before_heldout",
        "allow_static_lora_modules": False,
        "require_localhost_admin": True,
        "server_project_root_transport": "explicit_absolute_posix",
    }


def test_segmented_formal_v2_binds_public_execution_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _segmented_synthetic_project(tmp_path)
    monkeypatch.setattr(
        "anchor_mvp.benchmark.formal_af_preflight.verify_registry",
        lambda _root, _run_root, *, group: {
            "registry_sha256": sha256(group.encode()).hexdigest()
        },
    )

    result = preflight(config_path, tmp_path)

    assert result["run_id"] == "compact-mvp-v2b-formal-v2-test"
    assert result["review_protocol"] == "segmented_repair_v1"
    assert result["frontend_segment_count"] == 10
    assert result["review_segment_count"] == 10
    assert result["expected_physical_calls"] == 23
    assert result["per_stage_token_cap"] == 512
    assert len(result["runtime_bindings_sha256"]) == 64
    assert len(result["execution_contract_sha256"]) == 64
    frozen = result["execution_contract"]
    assert frozen["segment_contract_sha256"] == result["segment_contract_sha256"]
    assert frozen["prompt_bundle_sha256"] == result["prompt_bundle_sha256"]
    assert frozen["processor_manifest_sha256"] == result["processor_manifest_sha256"]
    assert frozen["processor_tree_sha256"] == result["processor_tree_sha256"]
    assert frozen["runtime_bindings_sha256"] == result["runtime_bindings_sha256"]
    assert frozen["sampling"] == {"temperature": 0.0, "top_p": 1.0}


def test_segmented_formal_v2_rejects_prompt_bundle_drift(tmp_path: Path) -> None:
    config_path = _segmented_synthetic_project(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompt = tmp_path / config["prompt_bundle_binding"]["path"]
    _write_json(prompt, {"schema_version": "drift"})

    with pytest.raises(FormalAFPreflightError) as caught:
        preflight(config_path, tmp_path)

    assert caught.value.code == "prompt_bundle_changed"


def test_live_cli_fails_before_heldout_open_when_registry_gate_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*args, **kwargs):
        del args, kwargs
        raise FormalAFPreflightError("training_incomplete", "registry missing")

    monkeypatch.setattr(
        "anchor_mvp.benchmark.heldout_cli.formal_af_preflight", blocked
    )
    args = SimpleNamespace(specs=str(CONFIG), project_root=str(ROOT))

    with pytest.raises(HeldoutGateError, match="training_incomplete"):
        _enforce_formal_af_preflight(args)


def test_formal_run_requires_explicit_heldout_authorization() -> None:
    argv = [
        "formal-run",
        "--cases",
        "heldout-cases-not-opened.jsonl",
        "--fixtures-root",
        "fixtures-not-opened",
        "--manifest",
        "manifest-not-opened.json",
        "--leak-audit",
        "audit-not-opened.json",
        "--specs",
        str(CONFIG),
        "--output-dir",
        "output-not-created",
    ]
    blocked = build_parser().parse_args(argv)
    with pytest.raises(HeldoutGateError, match="explicit held-out access"):
        _require_heldout_authorization(blocked)

    authorized = build_parser().parse_args([*argv, "--authorize-heldout-access"])
    _require_heldout_authorization(authorized)
    assert authorized.resume is False

    resumed = build_parser().parse_args(
        [*argv, "--authorize-heldout-access", "--resume"]
    )
    assert resumed.resume is True

    serial = build_parser().parse_args(
        [
            *argv,
            "--authorize-heldout-access",
            "--serial-runtime-lora",
            "--admin-base-url",
            "http://127.0.0.1:8000",
            "--server-project-root",
            "/mnt/d/LLM/anchor-moe-lora",
        ]
    )
    assert serial.serial_runtime_lora is True
    assert serial.server_project_root == "/mnt/d/LLM/anchor-moe-lora"


def test_all_formal_arms_require_the_same_five_stage_trace() -> None:
    assert MATCHED_FIVE_STAGE_BASELINES == {
        "base_matched_calls",
        "mixed_matched_calls",
        "c_pipeline",
        "d_budget_matched_pipeline",
        "e_adaptive_pareto_pipeline",
        "f_adaptive_budget_matched_pipeline",
    }


@pytest.mark.parametrize(
    "baseline",
    [
        "d_budget_matched_pipeline",
        "e_adaptive_pareto_pipeline",
        "f_adaptive_budget_matched_pipeline",
    ],
)
def test_d_e_f_invalid_trace_fails_before_sandbox_access(
    baseline: str, tmp_path: Path
) -> None:
    record = BenchmarkRecord(
        baseline=baseline,
        group=baseline[0].upper(),
        case_id="synthetic-no-heldout",
        malicious=False,
        decision="PASS",
        success=True,
        final_code=None,
        latency_ms=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        call_count=1,
        request_attempts=1,
        peak_vram_mb=None,
        stages=[{"stage": "planner"}],
    )
    case = BenchmarkCase(case_id=record.case_id, requirement="synthetic")

    with pytest.raises(HeldoutGateError, match="matched five-stage trace"):
        _evaluate_record(
            record,
            case,
            tmp_path / "fixtures-not-opened",
            tmp_path / "workspaces-not-created",
            keep_workspaces=False,
        )


def test_model_catalog_gate_reports_missing_ids_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def read(self) -> bytes:
            return b'{"data":[{"id":"base"},{"id":"adapter-a"}]}'

    monkeypatch.setattr(
        "anchor_mvp.benchmark.heldout_cli.urllib.request.urlopen",
        lambda request, timeout: Response(),
    )

    with pytest.raises(HeldoutGateError, match="adapter-b"):
        _verify_model_catalog(
            "http://127.0.0.1:8000/v1",
            None,
            {"base", "adapter-a", "adapter-b"},
            1.0,
        )


def test_serial_endpoints_must_share_one_loopback_origin() -> None:
    _verify_local_serial_endpoints(
        "http://127.0.0.1:8000/v1", "http://127.0.0.1:8000"
    )
    with pytest.raises(HeldoutGateError, match="local|invalid"):
        _verify_local_serial_endpoints(
            "http://127.0.0.1:8000/v1", "https://example.invalid:8000"
        )
    with pytest.raises(HeldoutGateError, match="share one local origin"):
        _verify_local_serial_endpoints(
            "http://127.0.0.1:8000/v1", "http://127.0.0.1:8001"
        )


def test_frozen_formal_contract_rejects_static_mode_before_case_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "anchor_mvp.benchmark.heldout_cli._enforce_formal_af_preflight",
        lambda *args, **kwargs: {
            "serial_runtime_contract": {"allow_static_lora_modules": False}
        },
    )
    args = SimpleNamespace(
        authorize_heldout_access=True,
        output_dir=str(tmp_path / "not-created"),
        serial_runtime_lora=False,
    )

    with pytest.raises(HeldoutGateError, match="requires explicit"):
        asyncio.run(_run_formal_live(args))
    assert not Path(args.output_dir).exists()
