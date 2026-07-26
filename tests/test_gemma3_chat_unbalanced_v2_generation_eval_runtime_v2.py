from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from anchor_mvp.research import (
    gemma3_chat_unbalanced_v2_generation_eval_runtime_v2 as runtime,
)


def _config() -> dict[str, Any]:
    return runtime.load_config()


class FakeSource:
    def __init__(self, config: dict[str, Any]) -> None:
        historical = config["historical_source"]
        self.consumer_receipt_sha256 = config["consumer"]["receipt_sha256"]
        self.source_binding_sha256 = config["consumer"]["source_binding_sha256"]
        self.logical_identity = {
            "eval_proxy_partition_sha256": "0" * 64,
            "identity_probe_inventory_sha256": "3" * 64,
            "logical_all_record_inventory_sha256": "4" * 64,
            "serialization_inventory_sha256": "1" * 64,
            "logical_dataset_sha256": "2" * 64,
            "logical_record_order_sha256": "5" * 64,
            "logical_target_inventory_sha256": "6" * 64,
            "logical_task_bundle_inventory_sha256": "7" * 64,
            "logical_train_partition_sha256": "8" * 64,
            "logical_train_record_inventory_sha256": "9" * 64,
        }
        asset_names = [
            *(str(item["payload_path"]) for item in config["assets"].values()),
            *(f"synthetic/filler-{index:02d}.jsonl" for index in range(18)),
        ]
        self.identity = {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-immutable-git-source.v2"
            ),
            "candidate_commit": historical["candidate_commit"],
            "candidate_tree": historical["candidate_tree"],
            "release_commit": historical["release_commit"],
            "release_tree": historical["release_tree"],
            "read_set_files": 88,
            "payload_files": 23,
            "physical_files": 46,
            "final_files": 46,
            "historical_release_attested": True,
            "current_live_remote_reasserted": False,
            "artifact_prefix": historical["payload_root"],
            "consumer_binding_physical_sha256": config["consumer"]["binding_sha256"],
            "consumer_binding_canonical_sha256": "0" * 64,
            "consumer_preflight_receipt_sha256": config["consumer"]["receipt_sha256"],
            "consumer_source_binding_sha256": config["consumer"][
                "source_binding_sha256"
            ],
            "tree_digest_sha256": "3" * 64,
            "read_set_digest_sha256": "a" * 64,
            "producer_inventory_sha256": "b" * 64,
            "read_set_blob_inventory_sha256": "4" * 64,
            "payload_blob_inventory_sha256": "5" * 64,
            "attestation_blob": {
                "commit": historical["release_commit"],
                "blob_oid": "6" * 40,
                "sha256": "7" * 64,
                "bytes": 1,
            },
            "asset_blobs": {
                name: {
                    "commit": historical["release_commit"],
                    "tree": historical["release_tree"],
                    "path": f"{historical['payload_root']}/{name}",
                    "blob_oid": f"{index + 1:040x}",
                    "sha256": f"{index + 1:064x}",
                    "bytes": index + 1,
                }
                for index, name in enumerate(asset_names)
            },
            "current_head": "7b1f4e95a64d40cf724df436f555781336e9eb7a",
        }

    def public_identity(self) -> dict[str, Any]:
        return deepcopy(self.identity)

    def iter_jsonl(self, payload_path: str) -> Iterator[dict[str, Any]]:
        del payload_path
        return iter(())

    def terminal_recheck(self) -> None:
        return None


def _write_adapter_receipt(
    tmp_path: Path,
    config: dict[str, Any],
    *,
    tokenizer_policy_sha256: str | None = None,
    teacher_schema_version: str = (
        "anchor.gemma3-chat-unbalanced-v2-authenticated-teacher-final.v2"
    ),
) -> tuple[Path, dict[str, Any]]:
    adapters = {}
    phase_receipts = []
    phase_base = {}
    phase_q8 = {}
    for arm in runtime.TRAINED_ARMS:
        adapter = tmp_path / "adapters" / arm
        adapter.mkdir(parents=True)
        profile = runtime.multiarm.ARM_PROFILE[arm]
        targets = {
            "q_only": ["q_proj"],
            "o_only": ["o_proj"],
            "q_plus_micro_o": ["o_proj", "q_proj"],
        }[profile]
        rank = {"q_only": 1024, "o_only": 64, "q_plus_micro_o": 1024}[profile]
        alpha = {"q_only": 2048, "o_only": 128, "q_plus_micro_o": 2048}[profile]
        config_bytes = runtime._canonical_json(
            {
                "alpha_pattern": (
                    {"o_proj": 128} if profile == "q_plus_micro_o" else {}
                ),
                "bias": "none",
                "lora_alpha": alpha,
                "lora_dropout": 0.0,
                "peft_type": "LORA",
                "r": rank,
                "rank_pattern": ({"o_proj": 64} if profile == "q_plus_micro_o" else {}),
                "target_modules": targets,
                "task_type": "CAUSAL_LM",
            }
        )
        model_bytes = f"synthetic-{arm}\n".encode()
        (adapter / "adapter_config.json").write_bytes(config_bytes)
        (adapter / "adapter_model.safetensors").write_bytes(model_bytes)
        config_sha = runtime._sha256(config_bytes)
        model_sha = runtime._sha256(model_bytes)
        config_pin = {
            "path": "adapter_config.json",
            "sha256": config_sha,
            "bytes": len(config_bytes),
        }
        model_pin = {
            "path": "adapter_model.safetensors",
            "sha256": model_sha,
            "bytes": len(model_bytes),
        }
        artifact_sha = runtime._sha256(runtime._canonical_json([config_pin, model_pin]))
        tensor_sha = runtime._sha256(f"tensor:{arm}".encode())
        adapters[arm] = {
            "path": f"adapters/{arm}",
            "adapter_profile": profile,
            "trainable_parameters": (
                runtime.multiarm.PROFILE_PARAMETER_COUNTS[profile]
            ),
            "adapter_config": config_pin,
            "adapter_model": model_pin,
            "artifact_tree_sha256": artifact_sha,
            "tensor_inventory_sha256": tensor_sha,
        }
        phase_base[arm] = "d" * 64
        phase_q8[arm] = "e" * 64
        phase_receipts.append(
            {
                "arm": arm,
                "phase": "full",
                "status": "passed",
                "target_steps": runtime.multiarm.FULL_STEPS[arm],
                "path": f"phases/{arm}/receipt.json",
                "sha256": runtime._sha256(f"phase:{arm}".encode()),
                "bytes": 123,
                "q8_scb_sha256": phase_q8[arm],
                "base_parameter_sha256": phase_base[arm],
                "tensor_inventory_sha256": tensor_sha,
            }
        )
    base_files = {
        "model.safetensors": "8" * 64,
        "config.json": "9" * 64,
        "tokenizer.model": "a" * 64,
        "tokenizer_config.json": "b" * 64,
        "EXPORT_MANIFEST.json": "c" * 64,
    }
    receipt = {
        "schema_version": runtime.ADAPTER_RECEIPT_VERSION,
        "status": "passed",
        "run_id": "synthetic-run",
        "mode": "full",
        "config": {
            "path": (
                "configs/training/"
                "gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.yaml"
            ),
            "sha256": "f" * 64,
            "implementation": {
                "files": [
                    {
                        "path": f"synthetic/runtime-file-{index:02d}.txt",
                        "sha256": f"{index + 1:064x}",
                        "bytes": index + 1,
                    }
                    for index in range(14)
                ],
                "set_sha256": "e" * 64,
            },
        },
        "source": runtime._training_source_identity(FakeSource(config).identity),
        "teacher_final": {
            "schema_version": teacher_schema_version,
            "binding_schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-binding.v2"
            ),
            "record_schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-teacher-final-record.v2"
            ),
            "binding_sha256": "1" * 64,
            "binding_schema_sha256": "0" * 64,
            "manifest_sha256": "2" * 64,
            "record_schema_sha256": "3" * 64,
            "release_receipt_sha256": "4" * 64,
            "release_attestation_sha256": "5" * 64,
            "shard_inventory_sha256": "6" * 64,
            "records": 3520,
            "main_records": 3440,
            "router_records": 80,
            "train_identity_bundles": 20,
            "train_identity_records": 100,
            "independent_identity_eval_bundles_not_in_optimizer": 10,
            "independent_identity_eval_probes_not_in_optimizer": 50,
            "accepted": 3520,
            "rejected": 0,
            "uncertain": 0,
            "producer_original_target_fallback": False,
            "public_identity_sha256": "7" * 64,
        },
        "tokenizer": {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-authenticated-tokenizer.v2"
            ),
            "tokenizer_model_sha256": "2" * 64,
            "tokenizer_config_sha256": "3" * 64,
            "model_config_sha256": "4" * 64,
            "export_manifest_sha256": "5" * 64,
            "chat_template_policy_sha256": (
                tokenizer_policy_sha256
                or config["tokenizer"]["chat_template_policy_sha256"]
            ),
            "combined_identity_sha256": "6" * 64,
            "processor_source": "authenticated_model_proto_bytes",
            "second_model_file_path_read": False,
            "hf_auto_tokenizer_used": False,
            "runtime_special_token_overlay": True,
        },
        "model": {
            "base_files": base_files,
            "base_file_inventory_sha256": runtime._sha256(
                runtime._canonical_json(base_files)
            ),
            "phase_base_hashes": phase_base,
            "phase_q8_scb_hashes": phase_q8,
            "all_phase_base_hashes_equal": True,
            "all_phase_q8_scb_hashes_equal": True,
            "frozen_q8_base": True,
            "base_o_proj_always_frozen": True,
        },
        "phase_order": list(runtime.TRAINED_ARMS),
        "phase_receipts": phase_receipts,
        "adapters": adapters,
        "training_lineage": {
            "path": "training_lineage.json",
            "sha256": "7" * 64,
            "sidecar_sha256": "8" * 64,
        },
        "launcher": {
            "lease_sha256": "1" * 64,
            "attestation_sha256": "2" * 64,
            "launcher_pid": 42,
            "expected_gpu_uuid": "GPU-synthetic",
            "canonical_lock_path": "runs/formal-v3-training.lock",
            "lock_exclusive": True,
            "external_compute_process_policy": "observe_only",
            "smoke_gate": {
                "receipt_sha256": "3" * 64,
                "run_id": "synthetic-smoke",
                "all_nine_arms_passed": True,
                "smoke_checkpoint_consumed": False,
            },
        },
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "release": False,
            "sample_bodies_included": False,
            "raw_token_ids_included": False,
            "producer_original_targets_used": False,
            "physical_kv_validated": False,
            "full_generation_kv_shared": False,
        },
    }
    raw = runtime._canonical_json(receipt)
    path = tmp_path / "run_receipt.json"
    path.write_bytes(raw)
    path.with_name(path.name + ".sha256").write_bytes(
        f"{runtime._sha256(raw)}  {path.name}\n".encode()
    )
    config["adapter_run"]["receipt_sha256"] = runtime._sha256(raw)
    return path, receipt


def test_default_validate_is_body_model_gpu_and_network_free() -> None:
    config = _config()
    status = runtime.build_status(config)
    assert status["status"] == "blocked_waiting_for_bound_adapter_run_receipt"
    assert status["body_reads"] == 0
    assert status["raw_token_ids_read"] == 0
    assert status["model_loads"] == 0
    assert status["gpu_requests"] == 0
    assert status["network_requests"] == 0
    assert runtime.main(["--validate"]) == 0


def test_branch_advanced_does_not_replace_historical_p_r_authority() -> None:
    config = _config()
    source = FakeSource(config)
    identity, source_sha, logical = runtime._validate_runtime_source(config, source)
    assert identity["current_head"].startswith("7b1")
    assert (
        identity["candidate_commit"] == config["historical_source"]["candidate_commit"]
    )
    assert source_sha == config["consumer"]["source_binding_sha256"]
    assert logical["serialization_inventory_sha256"] == "1" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_commit", "0" * 40),
        ("candidate_tree", "0" * 40),
        ("release_tree", "0" * 40),
        ("payload_blob_inventory_sha256", None),
    ],
)
def test_historical_commit_tree_or_blob_inventory_drift_is_rejected(
    field: str,
    value: object,
) -> None:
    config = _config()
    source = FakeSource(config)
    if value is None:
        del source.identity[field]
    else:
        source.identity[field] = value
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_source_identity_drift",
    ):
        runtime._validate_runtime_source(config, source)


def test_adapter_receipt_and_all_nine_adapter_files_are_authenticated(
    tmp_path: Path,
) -> None:
    config = _config()
    path, receipt = _write_adapter_receipt(tmp_path, config)
    observed, observed_sha, adapters = runtime.authenticate_adapter_run(
        config,
        receipt_path=path,
        source_identity=FakeSource(config).identity,
    )
    assert observed == receipt
    assert observed_sha == config["adapter_run"]["receipt_sha256"]
    assert set(adapters) == set(runtime.TRAINED_ARMS)

    model = (
        tmp_path / "adapters" / runtime.TRAINED_ARMS[0] / "adapter_model.safetensors"
    )
    model.write_bytes(b"drift\n")
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_adapter_identity_drift",
    ):
        runtime.authenticate_adapter_run(
            config,
            receipt_path=path,
            source_identity=FakeSource(config).identity,
        )


def test_adapter_receipt_tokenizer_identity_drift_is_rejected_before_backend_start(
    tmp_path: Path,
) -> None:
    config = _config()
    path, _ = _write_adapter_receipt(
        tmp_path,
        config,
        tokenizer_policy_sha256="f" * 64,
    )
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_adapter_receipt_invalid",
    ):
        runtime.authenticate_adapter_run(
            config,
            receipt_path=path,
            source_identity=FakeSource(config).identity,
        )


def test_adapter_receipt_rejects_closed_teacher_v1_handoff(
    tmp_path: Path,
) -> None:
    config = _config()
    path, _ = _write_adapter_receipt(
        tmp_path,
        config,
        teacher_schema_version=(
            "anchor.gemma3-chat-unbalanced-v2-authenticated-teacher-final.v1"
        ),
    )
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_adapter_receipt_invalid",
    ):
        runtime.authenticate_adapter_run(
            config,
            receipt_path=path,
            source_identity=FakeSource(config).identity,
        )


def test_cross_arm_fairness_detects_seed_or_order_drift() -> None:
    baseline = {
        "asset": "tool",
        "records": 2,
        "record_order_sha256": "1" * 64,
        "serialization_sha256": "2" * 64,
        "sampling_sha256": "3" * 64,
        "seed": 1337,
        "max_new_tokens": 256,
    }
    runtime.validate_fairness_locks(
        {"tool": [{**baseline, "arm": "a"}, {**baseline, "arm": "b"}]}
    )
    drift = {**baseline, "arm": "b", "seed": 1338}
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_fairness_lock_drift",
    ):
        runtime.validate_fairness_locks({"tool": [{**baseline, "arm": "a"}, drift]})


def test_prompt_serialization_uses_authenticated_main_and_router_views() -> None:
    system = "system"
    user = "user"
    main_prompt = f"{system}\n\n{user}"
    visible = (
        f"<start_of_turn>user\n{main_prompt}<end_of_turn><eos>\n<start_of_turn>model\n"
    )
    main = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "gemma_serialization": {"prompt_sha256": runtime._sha256(visible.encode())},
    }
    assert runtime._record_prompt(main) == main_prompt

    router_prompt = f"System instructions:\n{system}\n\nUser request:\n{user}"
    router = {
        "namespace": "gemma3_chat_emotion_router_qonly_v1",
        "messages": main["messages"],
        "serialization": {"prompt_sha256": runtime._sha256(router_prompt.encode())},
    }
    assert runtime._record_prompt(router) == router_prompt
    router["serialization"]["prompt_sha256"] = "0" * 64
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_record_contract_invalid",
    ):
        runtime._record_prompt(router)


def test_scalar_scorer_covers_structure_grounding_planner_identity_and_safety() -> None:
    expected = {
        "tool_name": "lookup",
        "arguments": {"city": "Paris"},
        "task_class": "fact",
        "topology": ["lookup", "answer"],
    }
    record = {
        "messages": [{"role": "user", "content": "lookup Paris"}],
        "target": json.dumps(expected),
        "role": "tool_call",
        "selected_expert": "fact",
    }
    generated = {**expected, "identity": "trained by Air"}
    result = runtime.GenerationResult(
        text=json.dumps(generated),
        token_count=12,
        eos_terminated=True,
        latency_ms=1.25,
        active_adapter="tool_q_only",
    )
    score = runtime.score_generation(record, result)
    assert score["tool_name_exact"] is True
    assert score["tool_arguments_exact"] is True
    assert score["tool_arguments_grounded"] is True
    assert score["planner_label_exact"] is True
    assert score["planner_topology_exact"] is True
    assert score["identity_air_attribution"] is True
    assert score["identity_provider_misattribution_absent"] is True
    assert score["refusal_absent"] is True
    assert score["out_of_bounds_absent"] is True
    assert score["eos_terminated"] is True


@pytest.mark.parametrize(
    ("text", "air_attribution", "provider_absent"),
    [
        ("The chair is fair.", False, True),
        (
            "I was not trained by Google; I was trained by Air.",
            True,
            True,
        ),
        ("I was trained by Google, not Air.", False, False),
        ("我不是由 OpenAI 训练的；我是由 Air 训练的。", True, True),
        ("我由谷歌训练，不是由 Air 训练。", False, False),
    ],
)
def test_identity_claims_require_air_and_allow_negated_provider_mentions(
    text: str,
    air_attribution: bool,
    provider_absent: bool,
) -> None:
    result = runtime.GenerationResult(
        text=text,
        token_count=8,
        eos_terminated=True,
        latency_ms=1.0,
        active_adapter=None,
    )
    score = runtime.score_generation(
        {"prompt": "Who trained you?", "target": "I was trained by Air."},
        result,
    )
    assert score["identity_air_attribution"] is air_attribution
    assert score["identity_provider_misattribution_absent"] is provider_absent


def test_receipt_leakage_is_rejected_and_atomic_publish_is_create_once(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_body_leak_detected",
    ):
        runtime._assert_body_free({"metrics": {"completion": "synthetic"}})

    output = tmp_path / "receipt.json"
    digest = runtime._atomic_json(
        output,
        {"status": "synthetic", "aggregate": {"count": 1}},
    )
    assert len(digest) == 64
    before = output.read_bytes()
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_atomic_destination_exists",
    ):
        runtime._atomic_json(
            output,
            {"status": "replacement", "aggregate": {"count": 2}},
        )
    assert output.read_bytes() == before


def test_receipt_schema_requires_complete_runtime_identity() -> None:
    config = _config()
    source = FakeSource(config)
    base_files = {
        "model.safetensors": "1" * 64,
        "config.json": "2" * 64,
        "tokenizer.model": "3" * 64,
        "tokenizer_config.json": "4" * 64,
        "EXPORT_MANIFEST.json": "5" * 64,
    }
    context = runtime.AuthenticatedContext(
        config_sha256=config["_config_sha256"],
        consumer_receipt_sha256=config["consumer"]["receipt_sha256"],
        source_binding_sha256=config["consumer"]["source_binding_sha256"],
        source_identity=source.identity,
        logical_identity=source.logical_identity,
        adapter_run_receipt_sha256="6" * 64,
        adapter_inventory_sha256="7" * 64,
        base_model={
            "path": str(runtime.ROOT),
            "files": base_files,
            "inventory_sha256": runtime._sha256(runtime._canonical_json(base_files)),
        },
        base_model_object_identities={
            name: (1, index + 1, 0x8000)
            for index, name in enumerate(runtime.BASE_MODEL_FILES)
        },
        base_parameter_identity_sha256="8" * 64,
        q8_scb_inventory_sha256="9" * 64,
        adapters={},
        tokenizer={
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-authenticated-tokenizer.v2"
            ),
            "tokenizer_model_sha256": "a" * 64,
            "tokenizer_config_sha256": "b" * 64,
            "model_config_sha256": "c" * 64,
            "export_manifest_sha256": "d" * 64,
            "chat_template_policy_sha256": config["tokenizer"][
                "chat_template_policy_sha256"
            ],
            "combined_identity_sha256": "e" * 64,
            "processor_source": "authenticated_model_proto_bytes",
            "second_model_file_path_read": False,
            "hf_auto_tokenizer_used": False,
            "runtime_special_token_overlay": True,
            "authority": config["tokenizer"]["authority"],
            "runtime_special_token_ids": dict(
                config["tokenizer"]["runtime_special_token_overlay"]
            ),
        },
        tokenizer_capability=object(),
    )
    receipt = runtime._receipt(
        config,
        context,
        run_id="complete-identity",
        status="failed_closed",
        metrics=None,
        fairness=None,
        runtime={"gpu_attempted": False},
        failure_code="eval_record_contract_invalid",
    )
    assert not tuple(runtime._schema(runtime.RECEIPT_SCHEMA_PATH).iter_errors(receipt))

    incomplete = deepcopy(receipt)
    del incomplete["identity"]["base_parameter_identity_sha256"]
    assert tuple(runtime._schema(runtime.RECEIPT_SCHEMA_PATH).iter_errors(incomplete))


def test_identity_bound_release_preserves_same_content_replacement(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "gpu.lock"
    lock.write_bytes(b"same-content\n")
    original_identity = runtime._owned_path_identity(
        lock,
        require_directory=False,
        code="eval_gpu_lock_exists",
    )
    lock.unlink()
    lock.write_bytes(b"same-content\n")
    assert runtime._release_owned_file(lock, original_identity) is False
    assert lock.read_bytes() == b"same-content\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows handle disposition contract")
def test_release_has_no_path_unlink_fallback_after_disposition_entry_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = tmp_path / "owned.tmp"
    displaced = tmp_path / "owned.tmp.displaced"
    payload = b"same-content-new-inode\n"
    owned.write_bytes(payload)
    identity = runtime._owned_path_identity(
        owned,
        require_directory=False,
        code="eval_atomic_destination_exists",
    )

    def replace_at_disposition_entry(descriptor: int) -> bool:
        os.close(descriptor)
        owned.rename(displaced)
        owned.write_bytes(payload)
        return False

    monkeypatch.setattr(
        runtime,
        "_windows_set_delete_disposition",
        replace_at_disposition_entry,
    )
    assert runtime._release_owned_file(owned, identity) is False
    assert owned.read_bytes() == payload
    assert displaced.read_bytes() == payload
    assert runtime._owned_path_identity(
        owned,
        require_directory=False,
        code="eval_atomic_destination_exists",
    ) != runtime._owned_path_identity(
        displaced,
        require_directory=False,
        code="eval_atomic_destination_exists",
    )


@pytest.mark.parametrize(
    ("reader", "error_code"),
    (
        ("sha256", "eval_adapter_identity_drift"),
        ("small", "eval_adapter_identity_drift"),
        ("json", "eval_config_invalid"),
    ),
)
def test_all_stable_readers_reject_lstat_to_open_same_content_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
    error_code: str,
) -> None:
    target = tmp_path / "identity.json"
    displaced = tmp_path / "identity.json.displaced"
    payload = b'{"status":"same-content"}\n'
    target.write_bytes(payload)
    original_open = Path.open
    raced = False

    def racing_open(
        self: Path,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal raced
        if self == target and mode == "rb" and not raced:
            raced = True
            target.rename(displaced)
            target.write_bytes(payload)
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(runtime.GenerationEvalRuntimeError, match=error_code):
        if reader == "sha256":
            runtime._file_sha256(target, expected_bytes=len(payload))
        elif reader == "small":
            runtime._stable_small_file(
                target,
                expected_sha256=runtime._sha256(payload),
                expected_bytes=len(payload),
                code=error_code,
            )
        else:
            runtime._read_json(target)

    assert raced is True
    assert target.read_bytes() == payload
    assert displaced.read_bytes() == payload


def test_load_config_reuses_authenticated_bytes_without_raw_path_reread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"unexpected naked read_bytes: {self}")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    config = runtime.load_config()
    assert runtime._is_sha(config["_config_sha256"])


@pytest.mark.skipif(os.name != "nt", reason="Windows read-lease contract")
def test_windows_read_lease_rejects_auth_a_load_b_and_blocks_mutation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.safetensors"
    displaced = tmp_path / "model.safetensors.displaced"
    replacement = tmp_path / "replacement.safetensors"
    payload = b"immutable-model-bytes\n"
    target.write_bytes(payload)
    original_identity = runtime._owned_path_identity(
        target,
        require_directory=False,
        code="eval_base_identity_drift",
    )
    target.rename(displaced)
    target.write_bytes(payload)
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_base_identity_drift",
    ):
        runtime._acquire_windows_read_lease(
            target,
            expected_sha256=runtime._sha256(payload),
            expected_bytes=len(payload),
            expected_identity=original_identity,
            code="eval_base_identity_drift",
        )

    current_identity = runtime._owned_path_identity(
        target,
        require_directory=False,
        code="eval_base_identity_drift",
    )
    lease = runtime._acquire_windows_read_lease(
        target,
        expected_sha256=runtime._sha256(payload),
        expected_bytes=len(payload),
        expected_identity=current_identity,
        code="eval_base_identity_drift",
    )
    try:
        replacement.write_bytes(payload)
        with pytest.raises(OSError):
            os.replace(replacement, target)
        with pytest.raises(OSError):
            target.open("wb")
        runtime._verify_windows_read_lease(lease)
        assert target.read_bytes() == payload
    finally:
        assert lease.close() is True


def test_runtime_input_lease_inventory_is_exactly_five_plus_nine_pairs() -> None:
    assert len(runtime.TOKENIZER_LEASE_FILES) == 4
    assert set(runtime.TOKENIZER_LEASE_FILES) == (
        set(runtime.BASE_MODEL_FILES) - {"model.safetensors"}
    )
    assert len(runtime.RUNTIME_INPUT_LEASE_KEYS) == 23
    assert runtime.RUNTIME_INPUT_LEASE_KEYS[:5] == tuple(
        f"base:{name}" for name in runtime.BASE_MODEL_FILES
    )
    assert set(runtime.RUNTIME_INPUT_LEASE_KEYS[5:]) == {
        key
        for arm in runtime.TRAINED_ARMS
        for key in (f"adapter-config:{arm}", f"adapter-model:{arm}")
    }


@pytest.mark.skipif(os.name != "nt", reason="Windows read-lease contract")
def test_real_backend_closes_leases_on_start_error_and_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"lease-lifecycle\n"

    def acquire(name: str) -> runtime._WindowsReadLease:
        path = tmp_path / name
        path.write_bytes(payload)
        identity = runtime._owned_path_identity(
            path,
            require_directory=False,
            code="eval_base_identity_drift",
        )
        return runtime._acquire_windows_read_lease(
            path,
            expected_sha256=runtime._sha256(payload),
            expected_bytes=len(payload),
            expected_identity=identity,
            code="eval_base_identity_drift",
        )

    start_lease = acquire("start-error.bin")
    backend = runtime.RealGemmaBackend()
    backend.file_leases["synthetic"] = start_lease
    with pytest.raises(AttributeError):
        backend.start(object())
    assert start_lease.closed is True
    assert not backend.file_leases

    finish_lease = acquire("finish.bin")
    backend.file_leases["synthetic"] = finish_lease
    monkeypatch.setattr(backend, "_finish", lambda: {"verified": True})
    assert backend.finish() == {"verified": True}
    assert finish_lease.closed is True
    assert not backend.file_leases


def test_run_directory_claim_is_no_replace_and_never_cleans_existing(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "run-1"
    identity = runtime._claim_run_directory(destination)
    sentinel = destination / "owned-by-first-run.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_atomic_destination_exists",
    ):
        runtime._claim_run_directory(destination)
    assert (
        runtime._owned_path_identity(
            destination,
            require_directory=True,
            code="eval_atomic_destination_exists",
        )
        == identity
    )
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_gpu_lock_never_unlinks_same_content_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / "formal.lock"
    displaced = tmp_path / "formal.lock.displaced"
    original_release = runtime._release_owned_file
    expected_identity: tuple[int, int, int] | None = None

    def replace_before_release(
        path: Path,
        identity: tuple[int, int, int],
    ) -> bool:
        nonlocal expected_identity
        expected_identity = identity
        raw = path.read_bytes()
        path.rename(displaced)
        path.write_bytes(raw)
        return original_release(path, identity)

    monkeypatch.setattr(runtime, "_release_owned_file", replace_before_release)
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_gpu_lock_exists",
    ):
        with runtime.gpu_lock(lock, "identity-bound-run") as receipt:
            assert receipt["run_id"] == "identity-bound-run"
            assert expected_identity is None

    assert expected_identity == runtime._owned_path_identity(
        displaced,
        require_directory=False,
        code="eval_gpu_lock_exists",
    )
    assert lock.read_bytes() == displaced.read_bytes()


def test_atomic_publish_loses_race_without_overwriting_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "receipt.json"
    attacker = b'{"status":"winner"}\n'
    original_link = os.link

    def racing_link(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        if Path(destination) == output:
            output.write_bytes(attacker)
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(runtime.os, "link", racing_link)
    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_atomic_destination_exists",
    ):
        runtime._atomic_json(output, {"status": "loser"})

    assert output.read_bytes() == attacker
    assert not output.with_name("receipt.json.sha256").exists()
    assert not tuple(tmp_path.glob("*.tmp"))


def test_atomic_publish_rejects_replaced_expected_parent(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "claimed-run"
    displaced = tmp_path / "claimed-run.displaced"
    identity = runtime._claim_run_directory(directory)
    directory.rename(displaced)
    directory.mkdir()
    sentinel = directory / "replacement-owner.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(
        runtime.GenerationEvalRuntimeError,
        match="eval_atomic_destination_exists",
    ):
        runtime._atomic_json(
            directory / "receipt.json",
            {"status": "must-not-publish"},
            expected_parent_identity=identity,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (directory / "receipt.json").exists()
    assert displaced.is_dir()


def test_execute_failure_preserves_claimed_directory_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    sentinel = b"preserved failure evidence\n"

    class CloseTrackingBackend:
        closed = False

        def close(self) -> bool:
            self.closed = True
            return True

    backend = CloseTrackingBackend()

    monkeypatch.setattr(
        runtime,
        "authenticate_inputs",
        lambda *args, **kwargs: (object(), object()),
    )

    def fail_after_progress(
        config: dict[str, Any],
        context: object,
        source: object,
        backend: object,
        *,
        progress_directory: Path,
        progress_directory_identity: tuple[int, int, int],
    ) -> None:
        del config, context, source, backend
        assert (
            runtime._owned_path_identity(
                progress_directory,
                require_directory=True,
                code="eval_atomic_destination_exists",
            )
            == progress_directory_identity
        )
        (progress_directory / "evidence.bin").write_bytes(sentinel)
        raise runtime.GenerationEvalRuntimeError("eval_record_contract_invalid")

    monkeypatch.setattr(runtime, "evaluate", fail_after_progress)
    monkeypatch.setattr(
        runtime,
        "_receipt",
        lambda *args, **kwargs: {
            "status": "failed_closed",
            "failure_code": kwargs["failure_code"],
        },
    )

    @contextmanager
    def fake_lock(path: Path, run_id: str) -> Iterator[dict[str, Any]]:
        del path
        yield {"run_id": run_id}

    monkeypatch.setattr(runtime, "gpu_lock", fake_lock)
    receipt = runtime.execute(
        config,
        source_repo=tmp_path,
        adapter_receipt_path=tmp_path / "adapter-receipt.json",
        run_id="preserve-failure",
        backend=backend,
        output_root=tmp_path / "runs",
    )

    assert json.loads(receipt.read_text(encoding="utf-8")) == {
        "failure_code": "eval_record_contract_invalid",
        "status": "failed_closed",
    }
    assert (receipt.parent / "progress" / "evidence.bin").read_bytes() == sentinel
    assert backend.closed is True
