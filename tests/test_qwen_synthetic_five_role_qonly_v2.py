from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from anchor_mvp.training import qwen_synthetic_five_role_qonly_v2 as consumer
from anchor_mvp.training.config import ConfigError, _expand_env


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / consumer.CONFIG_PATH
PREFLIGHT_PS1 = (
    REPO_ROOT
    / "scripts"
    / "research"
    / "run_synthetic_five_role_qonly_v2_preflight.ps1"
)
PREPARE_SCRIPT = (
    REPO_ROOT / "scripts" / "research" / "prepare_synthetic_five_role_qonly_v2.py"
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        rendered = f"U:{messages[0]['content']}\nA:"
        if len(messages) == 2:
            rendered += messages[1]["content"] + "<eos>"
        else:
            assert add_generation_prompt is True
        return list(rendered.encode("utf-8"))


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text("utf-8"))
    assert isinstance(value, dict)
    result = _expand_env(value)
    result["_config_path"] = str(CONFIG)
    return result


def test_config_is_additive_blocked_qonly_and_low_memory() -> None:
    config = _config()
    consumer.validate_config(config)
    assert config["dataset"]["replaces_v1"] is False
    assert config["lora"]["target_modules"] == ["q_proj"]
    assert config["lora"]["rank"] == 4
    assert config["training"]["micro_batch_size"] == 1
    assert config["training"]["sequence_length"] == 512
    assert config["training"]["use_cache"] is False
    assert config["precision"] == {
        "compute_dtype": "bfloat16",
        "tf32": True,
        "float32_matmul_precision": "high",
    }
    assert config["kv_runtime_boundary"] == {
        "shared_prefix_adapter_mode": "off",
        "shared_prefix_read_only": True,
        "expert_activation": "q_proj_only",
        "expert_private_tail_append_only": True,
        "private_tail_includes_post_activation_prompt_and_generated_tokens": True,
        "private_tail_cross_expert_reuse": False,
        "committed_text_reencoded_into_next_shared_context": True,
        "full_generation_kv_shared": False,
        "ordinary_in_stack_q_lora_exact_kv_sharing": False,
        "runtime_private_tail_materialized": False,
        "execution_authorized": False,
    }
    assert config["claims"]["training_authorized"] is False
    assert config["claims"]["formal_training_authorized"] is False
    assert config["claims"]["two_track_600_record_contract_satisfied"] is False
    assert config["claims"]["runtime_private_tail_materialized"] is False
    assert config["claims"]["execution_authorized"] is False
    assert config["claims"]["dataset_proxy_ready"] is True
    assert config["claims"]["records_materialized"] is True


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("lora", "target_modules", ["q_proj", "o_proj"]),
        ("lora", "target_modules", ["o_proj"]),
        ("lora", "rank", 8),
        ("training", "micro_batch_size", 2),
        ("training", "sequence_length", 1024),
        ("training", "use_cache", True),
        ("precision", "tf32", False),
        ("kv_runtime_boundary", "shared_prefix_adapter_mode", "on"),
        ("kv_runtime_boundary", "private_tail_cross_expert_reuse", True),
        ("kv_runtime_boundary", "full_generation_kv_shared", True),
        (
            "kv_runtime_boundary",
            "ordinary_in_stack_q_lora_exact_kv_sharing",
            True,
        ),
        ("kv_runtime_boundary", "runtime_private_tail_materialized", True),
        ("kv_runtime_boundary", "execution_authorized", True),
        ("controls", "admitted_to_primary_runner", True),
        ("controls", "duplicate_dataset_rows", True),
        ("claims", "dataset_proxy_ready", False),
        ("claims", "records_materialized", False),
        ("claims", "training_authorized", True),
        ("dataset", "replaces_v1", True),
    ],
)
def test_primary_scope_and_authority_drift_fail_closed(
    section: str, field: str, value: object
) -> None:
    config = deepcopy(_config())
    config[section][field] = value
    with pytest.raises(ConfigError):
        consumer.validate_config(config)


def test_role_is_required_and_pending_fixture_identity_fails_closed() -> None:
    config = _config()
    with pytest.raises(ConfigError, match="explicit_role_required"):
        consumer.load_role_dataset(config, "mixed")
    config = deepcopy(config)
    config["dataset"]["expected_manifest_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="fixture_identity_pending"):
        consumer.load_role_dataset(config, "planner")


def test_partition_parser_error_is_redacted_before_cli_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "SAMPLE_BODY_SENTINEL_MUST_NOT_ESCAPE"

    def leaking_parser(_raw: bytes, _label: str) -> list[dict[str, object]]:
        raise ConfigError(f"duplicate JSON key: {sentinel}")

    monkeypatch.setattr(consumer.base, "_strict_jsonl", leaking_parser)
    with pytest.raises(
        ConfigError,
        match="^five_role_partition_jsonl_invalid_without_record_content$",
    ) as captured:
        consumer.load_role_dataset(_config(), "planner")
    assert sentinel not in str(captured.value)


def test_cli_help_and_dataset_only_output_are_actionable_without_record_body(
    capsys: pytest.CaptureFixture[str],
) -> None:
    help_text = consumer._parser().format_help()
    assert "--role" in help_text
    assert "--tokenizer-only" in help_text
    normalized_help = " ".join(help_text.split())
    assert "never loads a model, CUDA, provider, or training runtime" in normalized_help
    assert consumer.main(["--role", "planner"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed_dataset_only_dry_run_training_blocked"
    assert result["dataset"]["global_records_authenticated_before_role_filter"] == 1000
    assert result["dataset"]["role_train_records"] == 160
    assert result["dataset"]["role_eval_proxy_records"] == 40
    assert result["claims"]["training_executed"] is False


def test_powershell_wrapper_resolves_explicit_python_and_rejects_missing() -> None:
    valid = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PREFLIGHT_PS1),
            "-CheckPythonOnly",
            "-PythonExecutable",
            sys.executable,
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert valid.returncode == 0
    assert "Python interpreter resolved" in valid.stdout
    missing = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PREFLIGHT_PS1),
            "-CheckPythonOnly",
            "-PythonExecutable",
            str(REPO_ROOT / "missing-python.exe"),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert missing.returncode != 0
    combined = missing.stdout + missing.stderr
    assert "conda activate anchor-mvp" in combined
    assert "显式 Python 路径不可用" in combined


def test_direct_src_layout_cli_help_needs_no_editable_install() -> None:
    result = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0
    assert "--role" in result.stdout
    assert "never loads a model, CUDA" in result.stdout.replace("\n", " ")
    dataset_only = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), "--role", "planner"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert dataset_only.returncode == 0
    receipt = json.loads(dataset_only.stdout)
    assert receipt["status"] == "passed_dataset_only_dry_run_training_blocked"
    assert receipt["dataset"]["global_records_authenticated_before_role_filter"] == 1000
    assert receipt["claims"]["model_loaded"] is False


def test_serial_plan_is_five_isolated_qonly_adapters() -> None:
    config = _config()
    plan = consumer.build_serial_launcher_plan(config)
    assert plan["status"] == "declarative_plan_only_execution_blocked"
    assert [job["role"] for job in plan["jobs"]] == list(consumer.ROLES)
    assert len({job["adapter_output"] for job in plan["jobs"]}) == 5
    for job in plan["jobs"]:
        assert job["role"] in Path(job["adapter_output"]).parts
        assert job["preflight_argv"][-4:] == [
            "--role",
            job["role"],
            "--tokenizer-only",
            "--publish-preflight",
        ]
        assert job["training_argv"] is None
        assert job["training_execution_supported"] is False
        assert job["lora"] == {
            "target_modules": ["q_proj"],
            "rank": 4,
            "alpha": 8,
        }
        assert job["training"] == {
            "optimizer_steps": 160,
            "micro_batch_size": 1,
            "sequence_length": 512,
            "use_cache": False,
        }
        assert job["kv_runtime_boundary"] == config["kv_runtime_boundary"]
    assert plan["controls"] == {
        "labels": ["o_only", "q_plus_o"],
        "execution_overlay_only": True,
        "duplicated_rows": False,
        "admitted_to_primary_runner": False,
    }
    assert plan["claims"]["training_executed"] is False
    assert plan["claims"]["gpu_requested"] is False
    assert plan["claims"]["runtime_private_tail_materialized"] is False
    assert plan["claims"]["execution_authorized"] is False


def test_tokenizer_preflight_consumes_only_selected_role_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = consumer.RoleExample(
        record_id="record",
        role_view_id="view",
        task_bundle_id="bundle",
        task_bundle_sha256="a" * 64,
        task_semantic_sha256="b" * 64,
        inner_task_id="inner",
        chain_root_sha256="c" * 64,
        role="planner",
        canonical_stage="planner",
        split="train",
        language="en",
        stratum=consumer.STRATA[0],
        prompt="prompt",
        target='{"ok":true}',
    )
    eval_example = consumer.RoleExample(
        **{
            **example.__dict__,
            "record_id": "eval-record",
            "role_view_id": "eval-view",
            "split": "eval_proxy",
        }
    )
    dataset = consumer.RoleDataset(
        role="planner",
        manifest_sha256="d" * 64,
        partition_sha256={"train": "e" * 64, "eval": "f" * 64},
        global_records_authenticated=1000,
        global_task_bundles_authenticated=200,
        global_task_semantics_authenticated=200,
        train=(example,),
        eval_proxy=(eval_example,),
    )
    calls = 0

    def load_once(_config: object, role: str) -> consumer.RoleDataset:
        nonlocal calls
        calls += 1
        assert role == "planner"
        return dataset

    monkeypatch.setattr(consumer, "load_role_dataset", load_once)
    result = consumer.tokenizer_only_preflight(_config(), "planner", FakeTokenizer())
    assert calls == 1
    assert result["token_lengths"]["records"] == 2
    assert result["claims"]["tokenizer_loaded"] is True
    assert result["claims"]["model_loaded"] is False
    assert result["claims"]["training_executed"] is False


def test_preflight_receipt_is_role_isolated_and_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(consumer.qdiag, "_project_root_from_module", lambda: tmp_path)
    value = {"role": "planner", "status": "passed"}
    path = consumer._publish_receipt(_config(), "planner", value)
    assert "planner" in path.parts
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.with_name("preflight.json.sha256").read_text("ascii") == (
        f"{digest}  preflight.json\n"
    )
    with pytest.raises(ConfigError, match="preflight_output_exists"):
        consumer._publish_receipt(_config(), "planner", value)


def test_causal_filter_rejects_current_or_future_target_in_prompt() -> None:
    records = _minimal_bundle_records()
    records[0]["input"]["materialized_prompt"] += records[0]["target"][
        "serialized_assistant_output"
    ]
    with pytest.raises(ConfigError, match="forbidden_content_reached_prompt"):
        consumer._validate_causal_materialization(records)


def test_causal_filter_rejects_current_or_future_summary_in_prompt() -> None:
    records = _minimal_bundle_records()
    records[0]["input"]["materialized_prompt"] += records[0]["target"][
        "concise_rationale_summary"
    ]
    with pytest.raises(ConfigError, match="forbidden_content_reached_prompt"):
        consumer._validate_causal_materialization(records)


def test_causal_filter_rejects_forbidden_id_and_context_mismatch() -> None:
    records = _minimal_bundle_records()
    records[2]["input"]["materialized_prompt"] += records[2]["forbidden_segment_ids"][0]
    with pytest.raises(ConfigError, match="forbidden_content_reached_prompt"):
        consumer._validate_causal_materialization(records)
    records = _minimal_bundle_records()
    records[2]["input"]["allowed_context_segments"] = []
    with pytest.raises(ConfigError, match="allowed_context_mismatch"):
        consumer._validate_causal_materialization(records)


def test_causal_filter_rejects_segment_metadata_and_forbidden_set_tamper() -> None:
    records = _minimal_bundle_records()
    records[1]["board_segment_inventory"][0]["content_sha256"] = "f" * 64
    with pytest.raises(ConfigError, match="segment_inventory_cross_binding_invalid"):
        consumer._validate_causal_materialization(records)
    records = _minimal_bundle_records()
    for record in records:
        record["board_segment_inventory"][0]["content_sha256"] = "f" * 64
    with pytest.raises(ConfigError, match="target_segment_cross_binding_invalid"):
        consumer._validate_causal_materialization(records)
    records = _minimal_bundle_records()
    records[1]["board_segment_inventory"][0]["role"] = "security_gate"
    with pytest.raises(ConfigError, match="segment_slot_binding_invalid"):
        consumer._validate_causal_materialization(records)
    records = _minimal_bundle_records()
    records[1]["board_segment_inventory"][1]["visibility"] = "future_target"
    with pytest.raises(ConfigError, match="forbidden_visibility_invalid"):
        consumer._validate_causal_materialization(records)
    records = _minimal_bundle_records()
    records[1]["forbidden_segment_ids"] = records[1]["forbidden_segment_ids"][1:]
    with pytest.raises(ConfigError, match="forbidden_inventory_mismatch"):
        consumer._validate_causal_materialization(records)


def test_global_contract_authenticates_all_1000_before_role_selection() -> None:
    records, manifest = _global_records()
    examples = consumer._validate_global_records(records, manifest)
    assert len(examples) == 1000
    assert len({item.task_bundle_sha256 for item in examples}) == 200
    assert len({item.task_semantic_sha256 for item in examples}) == 200
    assert all(
        sum(item.role == role and item.split == "train" for item in examples) == 160
        for role in consumer.ROLES
    )
    assert all(
        sum(item.role == role and item.split == "eval_proxy" for item in examples) == 40
        for role in consumer.ROLES
    )


def test_semantic_inventory_drift_and_language_overlap_fail_closed() -> None:
    records, manifest = _global_records()
    records[0]["task_semantic_sha256"] = "f" * 64
    with pytest.raises(ConfigError, match="task_semantic_role_inventory_invalid"):
        consumer._validate_global_records(records, manifest)
    records, manifest = _global_records()
    first_en_bundle = next(
        item["task_bundle_sha256"] for item in records if item["language"] == "en"
    )
    first_zh_bundle = next(
        item["task_bundle_sha256"] for item in records if item["language"] == "zh-CN"
    )
    en_semantic = next(
        item["task_semantic_sha256"]
        for item in records
        if item["task_bundle_sha256"] == first_en_bundle
    )
    second_en_bundle = next(
        item["task_bundle_sha256"]
        for item in records
        if item["language"] == "en" and item["task_bundle_sha256"] != first_en_bundle
    )
    for item in records:
        if item["task_bundle_sha256"] == first_zh_bundle:
            item["task_semantic_sha256"] = en_semantic
        elif item["task_bundle_sha256"] == second_en_bundle:
            item["task_semantic_sha256"] = _digest("new-semantic")
    with pytest.raises(ConfigError, match="language_semantic_overlap_invalid"):
        consumer._validate_global_records(records, manifest)


def test_bundle_split_and_cell_quota_drift_fail_closed() -> None:
    records, manifest = _global_records()
    bundle = next(
        item["task_bundle_sha256"] for item in records if item["split"] == "train"
    )
    for item in records:
        if item["task_bundle_sha256"] == bundle:
            item["split"] = "eval_proxy"
    with pytest.raises(
        ConfigError,
        match="bundle_split_count_invalid|language_stratum_role_quota_invalid",
    ):
        consumer._validate_global_records(records, manifest)


@pytest.mark.parametrize(
    ("legacy_field", "legacy_value"),
    [
        ("pair_id", "legacy-pair"),
        ("pair", {"id": "legacy-pair"}),
        ("variant", "concise_rationale_plus_json"),
    ],
)
def test_legacy_pair_and_variant_fallbacks_are_rejected(
    legacy_field: str, legacy_value: object
) -> None:
    records, _manifest = _global_records()
    record = records[0]
    record[legacy_field] = legacy_value
    with pytest.raises(ConfigError, match="legacy_pair_or_variant_semantics_forbidden"):
        consumer._strict_record(record)
    record.pop(legacy_field)
    role_view_id = record.pop("role_view_id")
    with pytest.raises(ConfigError, match="training_view_invalid"):
        consumer._strict_record(record)
    record["role_view_id"] = role_view_id
    view = record.pop("view")
    with pytest.raises(ConfigError, match="record_boundary_invalid"):
        consumer._strict_record(record)
    record["view"] = view


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _global_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    for language in consumer.LANGUAGES:
        for stratum in consumer.STRATA:
            for bundle_index in range(20):
                bundle_key = f"{language}:{stratum}:{bundle_index}"
                bundle_sha = _digest(f"bundle:{bundle_key}")
                semantic_sha = _digest(f"semantic:{bundle_key}")
                split = "eval_proxy" if bundle_index < 4 else "train"
                inventory = [
                    {
                        "segment_id": f"segment-{bundle_sha}-{index}",
                        "segment_ref": f"S{index}",
                        "role": consumer.ROLES[index],
                        "canonical_stage": consumer.ROLE_CANONICAL_STAGE[
                            consumer.ROLES[index]
                        ],
                        "stage_index": index,
                        "content_sha256": _digest(
                            json.dumps(
                                {"bundle": bundle_sha, "stage": index},
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                    }
                    for index in range(5)
                ]
                for stage, role in enumerate(consumer.ROLES):
                    route = {"bundle": bundle_sha, "stage": stage}
                    canonical_route = json.dumps(
                        route,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    summary = f"commit-summary-{bundle_sha}-{stage}"
                    serialized = f"{summary}\n{canonical_route}"
                    board = []
                    for item in inventory:
                        entry = dict(item)
                        entry["visibility"] = (
                            "previous_committed"
                            if item["stage_index"] < stage
                            else "current_target"
                            if item["stage_index"] == stage
                            else "future_target"
                        )
                        board.append(entry)
                    records.append(
                        {
                            "record_id": f"record-{bundle_sha}-{role}",
                            "role_view_id": f"view-{bundle_sha}-{role}",
                            "task_bundle_id": f"bundle-{bundle_sha}",
                            "task_bundle_sha256": bundle_sha,
                            "task_semantic_sha256": semantic_sha,
                            "inner_task_id": f"inner-{semantic_sha}",
                            "chain_root_sha256": _digest(f"chain:{bundle_sha}"),
                            "split": split,
                            "language": language,
                            "stratum": stratum,
                            "role": role,
                            "canonical_stage": consumer.ROLE_CANONICAL_STAGE[role],
                            "stage_index": stage,
                            "view": "concise_rationale_plus_json",
                            "input": {
                                "allowed_context_segments": [
                                    {
                                        "segment_ref": inventory[index]["segment_ref"],
                                        "role": inventory[index]["role"],
                                        "canonical_stage": inventory[index][
                                            "canonical_stage"
                                        ],
                                        "stage_index": index,
                                        "committed_summary": (
                                            f"commit-summary-{bundle_sha}-{index}"
                                        ),
                                    }
                                    for index in range(stage)
                                ],
                                "materialized_prompt": (
                                    f"task={bundle_key};role={role};"
                                    + ";".join(
                                        f"S{index};commit-summary-{bundle_sha}-{index}"
                                        for index in range(stage)
                                    )
                                ),
                            },
                            "board_segment_inventory": board,
                            "forbidden_segment_ids": [
                                inventory[index]["segment_id"]
                                for index in range(stage, 5)
                            ],
                            "target": {
                                "serialized_assistant_output": serialized,
                                "concise_rationale_summary": summary,
                                "canonical_routing_json": route,
                                "canonical_json_sha256": _digest(canonical_route),
                                "output_sha256": _digest(serialized),
                            },
                            "claims": {
                                "diagnostic_only": True,
                                "training_authorized": False,
                                "formal": False,
                            },
                            "audit": {
                                "protected_body_reads": 0,
                                "provider_requests": 0,
                                "network_requests": 0,
                                "model_loads": 0,
                                "gpu_requests": 0,
                                "real_tool_executions": 0,
                            },
                        }
                    )
    manifest = {
        "split_contract": {
            "group_key": "task_bundle_sha256",
            "eval_proxy_is_heldout": False,
        },
        "generation_contract": {
            "source_namespace": "anchor.synthetic-five-role-qonly-diagnostic.v1"
        },
        "semantic_identity_contract": {
            "unique_task_semantics": 200,
            "each_role_covers_same_200_semantics": True,
            "en_zh_intersection_count": 0,
            "translation_pair_count": 0,
        },
        "ablation_contract": {
            "primary_label": "q_only",
            "q_only_is_only_primary": True,
            "diagnostic_control_labels": ["o_only", "q_plus_o"],
            "control_arms_are_execution_overlays_only": True,
            "control_arm_rows_materialized": False,
        },
        "compatibility_boundary": {
            "pair_count": 0,
            "replaces_100_v1": False,
            "satisfies_independent_600_materialization": False,
            "satisfies_factorial_600_materialization": False,
            "variants_per_role": 1,
        },
        "counts": {"records": 1000, "task_bundles": 200},
        "claims": {
            "diagnostic_only": True,
            "training_authorized": False,
            "formal": False,
            "eval_proxy_is_heldout": False,
        },
    }
    return records, manifest


def _minimal_bundle_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    inventory = [
        {
            "segment_id": f"segment-{index}",
            "segment_ref": f"S{index}",
            "role": consumer.ROLES[index],
            "canonical_stage": consumer.ROLE_CANONICAL_STAGE[consumer.ROLES[index]],
            "stage_index": index,
            "content_sha256": _digest(
                json.dumps(
                    {"stage": index, "value": consumer.ROLES[index]},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        }
        for index in range(5)
    ]
    for stage, role in enumerate(consumer.ROLES):
        route = {"stage": stage, "value": role}
        canonical_route = json.dumps(
            route,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        summary = f"commit-summary-{stage}"
        serialized = f"{summary}\n{canonical_route}"
        board = []
        for item in inventory:
            entry = dict(item)
            entry["visibility"] = (
                "previous_committed"
                if item["stage_index"] < stage
                else "current_target"
                if item["stage_index"] == stage
                else "future_target"
            )
            board.append(entry)
        records.append(
            {
                "role": role,
                "stage_index": stage,
                "task_bundle_sha256": "a" * 64,
                "input": {
                    "allowed_context_segments": [
                        {
                            "segment_ref": f"S{index}",
                            "role": consumer.ROLES[index],
                            "canonical_stage": consumer.ROLE_CANONICAL_STAGE[
                                consumer.ROLES[index]
                            ],
                            "stage_index": index,
                            "committed_summary": f"commit-summary-{index}",
                        }
                        for index in range(stage)
                    ],
                    "materialized_prompt": (
                        f"task role={role} "
                        + " ".join(
                            f"S{index} commit-summary-{index}" for index in range(stage)
                        )
                    ),
                },
                "board_segment_inventory": board,
                "forbidden_segment_ids": [
                    f"segment-{index}" for index in range(stage, 5)
                ],
                "target": {
                    "serialized_assistant_output": serialized,
                    "concise_rationale_summary": summary,
                    "canonical_routing_json": route,
                    "canonical_json_sha256": _digest(canonical_route),
                    "output_sha256": _digest(serialized),
                },
            }
        )
    return records
