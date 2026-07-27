from __future__ import annotations

import ast
import copy
import hashlib
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2 as runtime,
)
from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1 as contract,
)


SHA = "a" * 64


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _private_snapshot_fixture(
    path: Path,
) -> tuple[
    dict[str, tuple[int, str]],
    dict[str, runtime.FileObjectIdentity],
]:
    path.mkdir()
    contract: dict[str, tuple[int, str]] = {}
    identities: dict[str, runtime.FileObjectIdentity] = {}
    for index, name in enumerate(runtime.q8_runtime.MODEL_FILES):
        raw = f"model-{index}".encode("ascii")
        target = path / name
        target.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        contract[name] = (len(raw), digest)
        identities[name] = runtime._capture_file_object_identity(
            target,
            expected_bytes=len(raw),
            expected_sha256=digest,
            code="test_private_snapshot_identity",
        )
    return contract, identities


def _source_identity() -> dict[str, object]:
    return {
        "schema_version": "anchor.gemma3-chat-unbalanced-v2-immutable-git-source.v2",
        "candidate_commit": "c5080249aa103d6cac09f0a55e51982b68524480",
        "release_commit": "304be2b86f82aad7ddade80fae04528bf2801977",
        "release_tree": "9bf154d481623b696dbc8a4bdf276967710b7214",
        "artifact_prefix": (
            "fixtures/research/"
            "gemma3_chat_five_expert_qonly_unbalanced_distilled_v2_sharded_v3"
        ),
        "historical_release_attested": True,
        "current_live_remote_reasserted": False,
        "consumer_preflight_receipt_sha256": SHA,
        "consumer_source_binding_sha256": SHA,
        "consumer_binding_physical_sha256": SHA,
        "consumer_binding_canonical_sha256": SHA,
        "tree_digest_sha256": SHA,
        "read_set_digest_sha256": SHA,
        "producer_inventory_sha256": SHA,
        "read_set_blob_oids_sha256": SHA,
        "payload_blob_oids_sha256": SHA,
        "read_set_files": 88,
        "final_files": 46,
    }


def _teacher_identity() -> dict[str, object]:
    return {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-authenticated-teacher-final.v2"
        ),
        "binding_schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-binding.v2"
        ),
        "record_schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-teacher-final-record.v2"
        ),
        "binding_sha256": SHA,
        "binding_schema_sha256": SHA,
        "manifest_sha256": SHA,
        "record_schema_sha256": SHA,
        "release_receipt_sha256": SHA,
        "release_attestation_sha256": SHA,
        "shard_inventory_sha256": SHA,
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
        "public_identity_sha256": SHA,
    }


def _tokenizer_identity() -> dict[str, object]:
    return {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-authenticated-tokenizer.v2"
        ),
        "tokenizer_model_sha256": SHA,
        "tokenizer_config_sha256": SHA,
        "model_config_sha256": SHA,
        "export_manifest_sha256": SHA,
        "chat_template_policy_sha256": SHA,
        "combined_identity_sha256": SHA,
        "processor_source": "authenticated_model_proto_bytes",
        "second_model_file_path_read": False,
        "hf_auto_tokenizer_used": False,
        "runtime_special_token_overlay": True,
    }


def _base_files() -> dict[str, str]:
    return {
        "model.safetensors": SHA,
        "config.json": SHA,
        "tokenizer.model": SHA,
        "tokenizer_config.json": SHA,
        "EXPORT_MANIFEST.json": SHA,
    }


def _phase_receipt() -> dict[str, object]:
    return {
        "schema_version": runtime.PHASE_RECEIPT_VERSION,
        "status": "passed",
        "run_id": "unit-smoke",
        "arm": "humor_q_only",
        "phase": "smoke",
        "source_asset": "humor",
        "adapter_profile": "q_only",
        "optimizer_steps": 2,
        "target_steps": 2,
        "source": _source_identity(),
        "teacher_final": _teacher_identity(),
        "tokenizer": _tokenizer_identity(),
        "model": {
            "base_files": _base_files(),
            "base_parameter_hash_before": SHA,
            "base_parameter_hash_after": SHA,
            "base_parameter_hash_equal": True,
            "q8_scb_sha256_before": SHA,
            "q8_scb_sha256_after": SHA,
            "q8_scb_hash_equal": True,
            "q8_modules": {
                "linear8lt_base_modules": 182,
                "int8_weight_tensors": 182,
            },
            "base_preparation": {
                "frozen_parameters": 999885952,
                "int8_parameters": 697761792,
                "preserved_bf16_parameters": 301989888,
                "norm_parameters_fp32": 134272,
            },
            "runtime_packages": {
                "torch": "2.5.1+cu121",
                "transformers": "5.13.0",
                "peft": "0.19.1",
                "safetensors": "0.6.2",
                "sentencepiece": "0.2.1",
                "bitsandbytes": "0.48.2",
            },
        },
        "tensor_inventory": {
            "sha256": SHA,
            "trainable_tensors": 52,
            "trainable_parameters": 57933824,
            "initial_adapter_sha256": SHA,
            "final_adapter_sha256": "b" * 64,
            "base_o_proj_frozen": True,
        },
        "optimizer": {
            "backend": "bitsandbytes.optim.AdamW8bit",
            "class": "bitsandbytes.optim.AdamW8bit",
            "package_version": "0.48.2",
            "optimizer_state_bits": 8,
            "compatibility_optim_bits_argument": 32,
            "min_8bit_size": 4096,
            "percentile_clipping": 100,
            "block_wise": True,
            "is_paged": False,
            "amsgrad": False,
            "parameter_tensors": 52,
            "state_tensors": 104,
            "state_elements": 115867648,
            "state_dtype": "uint8",
            "state_device": "cuda",
            "group_order": ["Q"],
            "q_learning_rate": 0.000002,
            "o_learning_rate": 0.0000002,
            "o_lr_exactly_one_tenth_of_q": True,
            "uint8_state": True,
        },
        "loss": {
            "finite": True,
            "first": 2.0,
            "last": 1.9,
            "minimum": 1.9,
            "maximum": 2.0,
            "count": 2,
        },
        "gradient": {
            "tensors": 52,
            "finite": 52,
            "nonzero": 52,
            "q_nonzero": 52,
            "o_nonzero": 0,
            "A_nonzero": 26,
            "B_nonzero": 26,
            "all_steps_finite": True,
            "factor_union_nonzero": {"A": 52, "B": 52, "Q": 104, "O": 0},
        },
        "adapter_effect": {
            "view": "first_teacher_aligned_train_record_boundary_v2",
            "record_id_sha256": SHA,
            "serialization_identity_sha256": SHA,
            "forward_prefix_tokens": 32,
            "future_target_suffix_forwarded": False,
            "sample_body_included": False,
            "raw_token_ids_included": False,
            "finite": True,
            "max_abs": 0.1,
            "mean_abs": 0.01,
            "vocabulary_logits": 262144,
        },
        "memory": {
            "torch_peak_allocated_bytes": 1,
            "torch_peak_reserved_bytes": 1,
            "torch_peak_allocated_max_mib": 23962,
            "torch_peak_reserved_max_mib": 23962,
            "comparison": "less_than_or_equal",
            "driver_observation_only": True,
            "driver_memory_used_peak_mib": 1000,
            "driver_sample_count": 4,
            "driver_samples_sha256": SHA,
            "external_compute_process_policy": "observe_only",
        },
        "progress": {
            "schema_version": "anchor.multiarm-progress-evidence-identity.v1",
            "step_count": 2,
            "target_steps": 2,
            "inventory_sha256": SHA,
            "step_file_sha256s": [SHA, "b" * 64],
            "marker_file_sha256s": [SHA, "b" * 64],
            "final_step_file_sha256": "b" * 64,
            "lr_trace_contract_sha256": SHA,
            "observed_lr_trace_sha256": SHA,
            "expected_lr_trace_sha256": SHA,
        },
        "adapter": None,
        "claims": {
            "diagnostic_only": True,
            "proxy_only": True,
            "formal": False,
            "sample_body_included": False,
            "raw_token_ids_included": False,
            "producer_original_target_used": False,
            "physical_kv_validated": False,
            "full_generation_kv_shared": False,
        },
    }


def _run_receipt() -> dict[str, object]:
    phase_summaries = []
    for arm in contract.TRAINED_ARMS:
        phase_summaries.append(
            {
                "arm": arm,
                "phase": "smoke",
                "status": "passed",
                "target_steps": 2,
                "path": f"phases/{arm}/receipt.json",
                "sha256": SHA,
                "bytes": 100,
                "q8_scb_sha256": SHA,
                "base_parameter_sha256": SHA,
                "tensor_inventory_sha256": SHA,
            }
        )
    hashes = {arm: SHA for arm in contract.TRAINED_ARMS}
    return {
        "schema_version": runtime.RUN_RECEIPT_VERSION,
        "status": "passed",
        "run_id": "unit-smoke",
        "mode": "smoke_only",
        "config": {
            "path": runtime.CONFIG_PATH.as_posix(),
            "sha256": SHA,
            "implementation": runtime._implementation_identity(),
        },
        "source": _source_identity(),
        "teacher_final": _teacher_identity(),
        "tokenizer": _tokenizer_identity(),
        "model": {
            "base_files": _base_files(),
            "base_file_inventory_sha256": SHA,
            "phase_base_hashes": hashes,
            "phase_q8_scb_hashes": hashes,
            "all_phase_base_hashes_equal": True,
            "all_phase_q8_scb_hashes_equal": True,
            "frozen_q8_base": True,
            "base_o_proj_always_frozen": True,
        },
        "phase_order": list(contract.TRAINED_ARMS),
        "phase_receipts": phase_summaries,
        "adapters": {},
        "training_lineage": {
            "path": "training_lineage.json",
            "sha256": SHA,
            "sidecar_sha256": SHA,
        },
        "launcher": {
            "lease_sha256": SHA,
            "attestation_sha256": SHA,
            "launcher_pid": 1,
            "expected_gpu_uuid": "GPU-unit",
            "canonical_lock_path": "runs/formal-v3-training.lock",
            "lock_exclusive": True,
            "external_compute_process_policy": "observe_only",
            "smoke_gate": None,
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


def _training_prompts() -> list[runtime.SourcePrompt]:
    values: list[runtime.SourcePrompt] = []
    for bundle_index in range(20):
        bundle = _sha(f"identity-bundle-{bundle_index}")
        for role in runtime.MAIN_ROLES:
            record = f"identity-{bundle_index}-{role}"
            values.append(
                runtime.SourcePrompt(
                    record_id_sha256=_sha(record),
                    source_asset=role,
                    prompt="not persisted",
                    source_content_sha256=_sha(record + "-content"),
                    source_serialization_identity_sha256=_sha(
                        record + "-serialization"
                    ),
                    task_bundle_sha256=bundle,
                    identity_training_record=True,
                )
            )
    existing = {role: 20 for role in runtime.MAIN_ROLES}
    serial = 0
    for asset, total in runtime.EXPECTED_ASSET_COUNTS.items():
        for _ in range(total - existing.get(asset, 0)):
            record = f"ordinary-{asset}-{serial}"
            values.append(
                runtime.SourcePrompt(
                    record_id_sha256=_sha(record),
                    source_asset=asset,
                    prompt="not persisted",
                    source_content_sha256=_sha(record + "-content"),
                    source_serialization_identity_sha256=_sha(
                        record + "-serialization"
                    ),
                    task_bundle_sha256=_sha(record + "-bundle"),
                    identity_training_record=False,
                )
            )
            serial += 1
    return values


@pytest.mark.parametrize(
    "wrong_status",
    ["released", "pending_exact_producer_v2_handoff", "wrong"],
)
def test_wrong_teacher_schema_state_blocks_gpu_before_source_or_output(
    monkeypatch: pytest.MonkeyPatch,
    wrong_status: str,
) -> None:
    config, config_sha = runtime.load_config()
    config = copy.deepcopy(config)
    config["teacher_final"]["binding_schema_release_status"] = wrong_status
    config["teacher_final"]["record_schema_release_status"] = wrong_status
    called = False

    def forbidden_source(_value: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("source auth must not run while v2 hashes are pending")

    monkeypatch.setattr(
        runtime.source_runtime,
        "authenticate_producer_source",
        forbidden_source,
    )
    with pytest.raises(
        runtime.MultiArmRuntimeError,
        match="teacher_v3_physical_handoff_invalid",
    ):
        runtime.execute_runtime(
            config,
            config_sha256=config_sha,
            mode="smoke_only",
            run_id="pending-v2",
            producer_repository=None,
            teacher_binding="missing.json",
            teacher_binding_sha256=SHA,
            lease_path="missing-lease.json",
            lease_sha256=SHA,
            attestation_path="missing-attestation.json",
            attestation_sha256=SHA,
        )
    assert called is False


def test_validate_distinguishes_bound_schemas_from_teacher_final_readiness() -> None:
    config, config_sha = runtime.load_config()
    producer, binding_sha, record_sha = runtime._released_teacher_schema_identities(
        config
    )
    result = runtime._validate_only(config, config_sha)

    assert binding_sha == config["teacher_final"]["binding_schema_sha256"]
    assert record_sha == config["teacher_final"]["record_schema_sha256"]
    assert producer.commit == runtime.ROLLOVER_PRODUCER_COMMIT
    assert producer.parent == runtime.ROLLOVER_PRODUCER_PARENT
    assert producer.tree == runtime.ROLLOVER_PRODUCER_TREE
    assert dict(producer.physical_identity_sha256) == {
        name: str(spec["sha256"])
        for name, spec in runtime.ROLLOVER_PHYSICAL_IDENTITY_SPECS.items()
    }
    assert result["rollover_v3_config_sha256"] == (
        "6664bee43d144d07673ac3457020727180bb047f52bedb05b09aa2cd69e37ae5"
    )
    assert result["rollover_v3_execute_implementation_sha256"] == (
        "a80c3c85239c033412ced8e95328c9189e4207ecde99a150f6f3e91e8c920086"
    )
    assert result["rollover_v3_release_implementation_sha256"] == (
        "75143aaf366bbb9db0e0d81d7763d0b901d045d4e2c45c45deb078d91e0a75fc"
    )
    assert result["teacher_schema_physical_bytes_bound"] is True
    assert result["teacher_final_physical_preflight_required"] is True
    assert result["gpu_execution_ready"] is True
    assert result["gpu_execution_blocker"] is None


def test_explicit_teacher_preflight_reaches_given_binding_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = runtime.load_config()
    observed: dict[str, object] = {}

    class FakeTeacher:
        def public_identity(self) -> dict[str, object]:
            return _teacher_identity()

    def fake_authenticate(
        binding_path: str | Path,
        *,
        expected_binding_sha256: str,
        producer: runtime.AuthenticatedRolloverProducer,
        expected_record_schema_sha256: str,
    ) -> tuple[FakeTeacher, dict[str, object]]:
        observed.update(
            {
                "binding_path": binding_path,
                "binding_sha256": expected_binding_sha256,
                "binding_schema_sha256": producer.schema_sha256["binding"],
                "record_schema_sha256": expected_record_schema_sha256,
            }
        )
        return FakeTeacher(), {}

    monkeypatch.setattr(runtime, "authenticate_teacher_final", fake_authenticate)
    monkeypatch.setattr(runtime, "terminal_recheck_teacher", lambda _teacher: None)
    result = runtime.preflight_teacher_final(
        config,
        binding_path="given-teacher-binding.json",
        binding_sha256=SHA,
    )

    assert observed == {
        "binding_path": "given-teacher-binding.json",
        "binding_sha256": SHA,
        "binding_schema_sha256": (config["teacher_final"]["binding_schema_sha256"]),
        "record_schema_sha256": config["teacher_final"]["record_schema_sha256"],
    }
    assert result["status"] == "passed"
    assert (
        result[
            "all_binding_manifest_receipt_attestation_shard_identities_authenticated"
        ]
        is True
    )


def test_v1_teacher_rejected_and_exact_v2_record_schema() -> None:
    validator = runtime._load_schema(
        runtime._project_root() / runtime.TEACHER_RECORD_SCHEMA_PATH
    )
    target = "Air aligned response"
    record = {
        "schema_version": runtime.TEACHER_RECORD_VERSION,
        "namespace": "gemma3_chat_unbalanced_v2_teacher_alignment_v2",
        "record_id_sha256": SHA,
        "source_content_sha256": SHA,
        "source_serialization_identity_sha256": SHA,
        "task_bundle_sha256": SHA,
        "source_asset": "humor",
        "decision": "accepted",
        "teacher_target": target,
        "teacher_target_sha256": _sha(target),
    }
    validator.validate(record)
    assert runtime._teacher_target(record).target == target

    v1 = dict(record)
    v1["schema_version"] = (
        "anchor.gemma3-chat-unbalanced-v2-teacher-alignment-record." + "v" + "1"
    )
    with pytest.raises(ValidationError):
        validator.validate(v1)
    with pytest.raises(runtime.MultiArmRuntimeError):
        runtime._teacher_target(v1)

    widened = dict(record)
    widened["target_format"] = "assistant_text"
    with pytest.raises(ValidationError):
        validator.validate(widened)


def test_training_inventory_recomputes_20_bundles_100_records_and_excludes_eval() -> (
    None
):
    prompts = _training_prompts()
    runtime._validate_source_prompt_inventory(prompts, router_seen=100)
    assert runtime.IDENTITY_PROBE_RECORDS not in {
        *runtime.TRAIN_SHARDS,
        runtime.ROUTER_RECORDS,
    }
    broken = list(prompts)
    broken[0] = runtime.SourcePrompt(
        **{
            **broken[0].__dict__,
            "task_bundle_sha256": _sha("wrong-bundle"),
        }
    )
    with pytest.raises(
        runtime.MultiArmRuntimeError,
        match="source_prompt_inventory_drift",
    ):
        runtime._validate_source_prompt_inventory(broken, router_seen=100)


def test_v2_teacher_join_is_exact_bijection() -> None:
    prompts = _training_prompts()
    targets = {
        prompt.record_id_sha256: runtime.TeacherTarget(
            record_id_sha256=prompt.record_id_sha256,
            source_asset=prompt.source_asset,
            source_content_sha256=prompt.source_content_sha256,
            source_serialization_identity_sha256=(
                prompt.source_serialization_identity_sha256
            ),
            task_bundle_sha256=prompt.task_bundle_sha256,
            target="{}"
            if prompt.source_asset
            in {
                "tool_call",
                "review_audit",
                "planner_router",
            }
            else "Air aligned response",
            target_sha256=_sha(
                "{}"
                if prompt.source_asset
                in {
                    "tool_call",
                    "review_audit",
                    "planner_router",
                }
                else "Air aligned response"
            ),
        )
        for prompt in prompts
    }
    assets, _projection = runtime.join_teacher_targets(prompts, targets)
    assert {key: len(value) for key, value in assets.items()} == (
        runtime.EXPECTED_ASSET_COUNTS
    )
    mismatched = dict(targets)
    first = prompts[0]
    mismatched[first.record_id_sha256] = runtime.TeacherTarget(
        **{
            **mismatched[first.record_id_sha256].__dict__,
            "source_content_sha256": _sha("drift"),
        }
    )
    with pytest.raises(
        runtime.MultiArmRuntimeError,
        match="teacher_join_identity_mismatch",
    ):
        runtime.join_teacher_targets(prompts, mismatched)


def test_nine_arm_profiles_and_independent_q_o_learning_rates() -> None:
    assert len(contract.TRAINED_ARMS) == 9
    for arm in contract.TRAINED_ARMS:
        q_lr, o_lr = runtime._active_learning_rates(
            arm,
            step=1,
            total_steps=2,
        )
        profile = contract.ARM_PROFILE[arm]
        assert (q_lr > 0) is (profile in {"q_only", "q_plus_micro_o"})
        assert (o_lr > 0) is (profile in {"o_only", "q_plus_micro_o"})
        if profile == "q_plus_micro_o":
            assert o_lr * 10 == q_lr

    class Capture:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    q_only = runtime._peft_config_for_arm("tool_q_only", Capture).kwargs
    o_only = runtime._peft_config_for_arm("planner_o_only", Capture).kwargs
    q_o = runtime._peft_config_for_arm("tool_q_plus_micro_o", Capture).kwargs
    assert q_only["r"] == 1024
    assert o_only["r"] == 64
    assert q_o["r"] == 1024
    assert q_o["rank_pattern"] == {"o_proj": 64}


def test_directory_publish_is_create_once_and_preserves_competitor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "owned.txt").write_text("owned", encoding="utf-8")
    identity = runtime._directory_identity(source, code="test")
    competitor = tmp_path / "published"
    competitor.mkdir()
    (competitor / "competitor.txt").write_text("competitor", encoding="utf-8")
    with pytest.raises(runtime.MultiArmRuntimeError):
        runtime._move_directory_create_once(
            source,
            competitor,
            owned_identity=identity,
            code="publish_collision",
        )
    assert (source / "owned.txt").read_text(encoding="utf-8") == "owned"
    assert (competitor / "competitor.txt").read_text(encoding="utf-8") == "competitor"

    destination = tmp_path / "published-fresh"
    runtime._move_directory_create_once(
        source,
        destination,
        owned_identity=identity,
        code="publish_failed",
    )
    assert not source.exists()
    assert (destination / "owned.txt").read_text(encoding="utf-8") == "owned"

    replaced_source = tmp_path / "replaced-source"
    replaced_source.mkdir()
    replaced_identity = runtime._directory_identity(replaced_source, code="test")
    preserved = tmp_path / "preserved-original"
    replaced_source.rename(preserved)
    replaced_source.mkdir()
    (replaced_source / "competitor.txt").write_text("competitor", encoding="utf-8")
    with pytest.raises(
        runtime.MultiArmRuntimeError,
        match="source_changed",
    ):
        runtime._move_directory_create_once(
            replaced_source,
            tmp_path / "must-not-publish",
            owned_identity=replaced_identity,
            code="owned_identity",
        )
    assert preserved.is_dir()
    assert (replaced_source / "competitor.txt").is_file()


def test_handle_bound_publish_blocks_source_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows handle-bound rename is the supported runtime")
    source = tmp_path / "source"
    source.mkdir()
    (source / "owned.txt").write_text("owned", encoding="utf-8")
    identity = runtime._directory_identity(source, code="test")
    original_open = runtime._windows_open_directory_fd
    attempted = False

    def open_with_swap_attempt(
        path: Path,
        *,
        delete_access: bool,
        code: str,
    ) -> int:
        nonlocal attempted
        fd = original_open(
            path,
            delete_access=delete_access,
            code=code,
        )
        if path == source and delete_access and not attempted:
            attempted = True
            with pytest.raises(OSError):
                source.rename(tmp_path / "attacker-preserved")
        return fd

    monkeypatch.setattr(
        runtime,
        "_windows_open_directory_fd",
        open_with_swap_attempt,
    )
    destination = tmp_path / "published"
    runtime._move_directory_create_once(
        source,
        destination,
        owned_identity=identity,
        code="handle_publish",
    )
    assert attempted is True
    assert (destination / "owned.txt").read_text(encoding="utf-8") == "owned"
    assert not (tmp_path / "attacker-preserved").exists()


def test_canonical_write_pins_owned_parent_against_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows parent handle guard is the supported runtime")
    parent = tmp_path / "owned-parent"
    parent.mkdir()
    identity = runtime._directory_identity(parent, code="test")
    original_open = runtime._windows_open_directory_fd
    attempted = False

    def open_with_swap_attempt(
        path: Path,
        *,
        delete_access: bool,
        code: str,
    ) -> int:
        nonlocal attempted
        fd = original_open(
            path,
            delete_access=delete_access,
            code=code,
        )
        if path == parent and not attempted:
            attempted = True
            with pytest.raises(OSError):
                parent.rename(tmp_path / "attacker-parent")
        return fd

    monkeypatch.setattr(
        runtime,
        "_windows_open_directory_fd",
        open_with_swap_attempt,
    )
    digest, _sidecar_digest = runtime._write_canonical(
        parent / "receipt.json",
        {"status": "passed"},
        parent_identity=identity,
    )
    assert attempted is True
    assert digest == _sha('{"status":"passed"}\n')
    assert not (tmp_path / "attacker-parent").exists()


def test_q8_private_snapshot_is_authenticated_and_retained(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows handle retention is the supported runtime")
    snapshot = tmp_path / "private-model"
    contract, identities = _private_snapshot_fixture(snapshot)
    identity = runtime._directory_identity(snapshot, code="test")
    runtime._retain_private_model_snapshot_by_handle(
        snapshot,
        owned_identity=identity,
        contract_value=contract,
        file_identities=identities,
    )
    assert snapshot.is_dir()
    assert all((snapshot / name).is_file() for name in runtime.q8_runtime.MODEL_FILES)

    preserved = tmp_path / "preserved"
    snapshot.rename(preserved)
    snapshot.mkdir()
    for index, name in enumerate(runtime.q8_runtime.MODEL_FILES):
        (snapshot / name).write_bytes(f"attacker-{index}".encode("ascii"))
    with pytest.raises(runtime.MultiArmRuntimeError, match="identity_drift"):
        runtime._retain_private_model_snapshot_by_handle(
            snapshot,
            owned_identity=identity,
            contract_value=contract,
            file_identities=identities,
        )
    assert preserved.is_dir()
    assert snapshot.is_dir()
    assert all((preserved / name).is_file() for name in runtime.q8_runtime.MODEL_FILES)
    assert all((snapshot / name).is_file() for name in runtime.q8_runtime.MODEL_FILES)


def test_q8_private_snapshot_same_content_inode_swap_preserves_both_objects(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows handle retention is the supported runtime")
    snapshot = tmp_path / "private-model"
    contract, identities = _private_snapshot_fixture(snapshot)
    directory_identity = runtime._directory_identity(snapshot, code="test")
    name = runtime.q8_runtime.MODEL_FILES[0]
    replacement_path = snapshot / name
    raw = replacement_path.read_bytes()
    bypass = tmp_path / "bypass-evidence"
    bypass.mkdir()
    original_path = bypass / name
    replacement_path.rename(original_path)
    replacement_path.write_bytes(raw)
    assert replacement_path.stat().st_ino != identities[name].inode

    with pytest.raises(
        runtime.MultiArmRuntimeError,
        match="runtime_private_snapshot_file_identity_drift",
    ):
        runtime._retain_private_model_snapshot_by_handle(
            snapshot,
            owned_identity=directory_identity,
            contract_value=contract,
            file_identities=identities,
        )

    assert snapshot.is_dir()
    assert replacement_path.read_bytes() == raw
    assert original_path.read_bytes() == raw
    assert len(list(snapshot.iterdir())) == len(runtime.q8_runtime.MODEL_FILES)


def test_q8_private_snapshot_collision_preserves_all_evidence(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows handle retention is the supported runtime")
    snapshot = tmp_path / "private-model"
    contract, identities = _private_snapshot_fixture(snapshot)
    directory_identity = runtime._directory_identity(snapshot, code="test")
    collision = snapshot / "unexpected-collision"
    collision.write_bytes(b"preserve")

    with pytest.raises(
        runtime.MultiArmRuntimeError,
        match="runtime_private_snapshot_inventory_drift",
    ):
        runtime._retain_private_model_snapshot_by_handle(
            snapshot,
            owned_identity=directory_identity,
            contract_value=contract,
            file_identities=identities,
        )

    assert collision.read_bytes() == b"preserve"
    assert all((snapshot / name).is_file() for name in runtime.q8_runtime.MODEL_FILES)


def test_q8_private_snapshot_late_collision_is_zero_delete_and_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows handle retention is the supported runtime")
    snapshot = tmp_path / "private-model"
    contract, identities = _private_snapshot_fixture(snapshot)
    directory_identity = runtime._directory_identity(snapshot, code="test")
    collision = snapshot / "late-collision"
    original_open_file = runtime._windows_open_file_fd
    attempted = False

    def open_after_late_collision(path: Path, *, code: str) -> int:
        nonlocal attempted
        if not attempted:
            attempted = True
            collision.write_bytes(b"preserve")
        return original_open_file(path, code=code)

    monkeypatch.setattr(
        runtime,
        "_windows_open_file_fd",
        open_after_late_collision,
    )
    with pytest.raises(
        runtime.MultiArmRuntimeError,
        match="runtime_private_snapshot_inventory_changed_before_retention",
    ):
        runtime._retain_private_model_snapshot_by_handle(
            snapshot,
            owned_identity=directory_identity,
            contract_value=contract,
            file_identities=identities,
        )

    assert attempted is True
    assert collision.read_bytes() == b"preserve"
    assert snapshot.is_dir()
    assert all((snapshot / name).is_file() for name in runtime.q8_runtime.MODEL_FILES)


def test_parent_directory_swap_and_reparse_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)
    target = parent / "data.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    chain = runtime._directory_chain(root, [target], code="chain")
    moved = root / "moved"
    parent.rename(moved)
    parent.mkdir()
    with pytest.raises(runtime.MultiArmRuntimeError, match="changed"):
        runtime._recheck_directory_chain(chain, code="changed")

    link = root / "link"
    try:
        link.symlink_to(moved, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink capability unavailable")
    with pytest.raises(runtime.MultiArmRuntimeError):
        runtime._directory_chain(root, [link], code="reparse")


def test_phase_run_and_failure_receipts_are_strict_draft202012() -> None:
    phase_validator = runtime._load_schema(
        runtime._project_root() / runtime.PHASE_SCHEMA_PATH
    )
    run_validator = runtime._load_schema(
        runtime._project_root() / runtime.RUN_SCHEMA_PATH
    )
    failure_validator = runtime._load_schema(
        runtime._project_root() / runtime.FAILURE_SCHEMA_PATH
    )
    phase = _phase_receipt()
    run = _run_receipt()
    failure = {
        "schema_version": runtime.FAILURE_RECEIPT_VERSION,
        "status": "failed",
        "run_id": "unit-failure",
        "mode": "smoke_only",
        "error_type": "MultiArmRuntimeError",
        "error_code": "teacher_v3_physical_handoff_invalid",
        "last_completed_arm": None,
        "source_authenticated": False,
        "teacher_final_authenticated": False,
        "sample_body_included": False,
        "raw_token_ids_included": False,
        "automatic_retry": False,
    }
    phase_validator.validate(phase)
    run_validator.validate(run)
    failure_validator.validate(failure)

    full_run = copy.deepcopy(run)
    full_run["run_id"] = "unit-full"
    full_run["mode"] = "full"
    for arm, summary in zip(
        contract.TRAINED_ARMS,
        full_run["phase_receipts"],
        strict=True,
    ):
        summary["phase"] = "full"
        summary["target_steps"] = contract.FULL_STEPS[arm]
    full_run["adapters"] = {
        arm: {
            "path": f"adapters/{arm}",
            "adapter_profile": contract.ARM_PROFILE[arm],
            "trainable_parameters": 1,
            "adapter_config": {
                "path": f"adapters/{arm}/adapter_config.json",
                "sha256": SHA,
                "bytes": 1,
            },
            "adapter_model": {
                "path": f"adapters/{arm}/adapter_model.safetensors",
                "sha256": SHA,
                "bytes": 1,
            },
            "artifact_tree_sha256": SHA,
            "tensor_inventory_sha256": SHA,
        }
        for arm in contract.TRAINED_ARMS
    }
    full_run["launcher"]["smoke_gate"] = {
        "receipt_sha256": SHA,
        "run_id": "unit-smoke",
        "all_nine_arms_passed": True,
        "smoke_checkpoint_consumed": False,
    }
    run_validator.validate(full_run)

    bad_phase = copy.deepcopy(phase)
    bad_phase["model"]["q8_modules"]["unexpected"] = True
    with pytest.raises(ValidationError):
        phase_validator.validate(bad_phase)
    bad_run = copy.deepcopy(run)
    bad_run["teacher_final"]["unexpected"] = True
    with pytest.raises(ValidationError):
        run_validator.validate(bad_run)
    bad_failure = dict(failure)
    bad_failure["sample"] = "forbidden"
    with pytest.raises(ValidationError):
        failure_validator.validate(bad_failure)
    incomplete_full = copy.deepcopy(full_run)
    incomplete_full["adapters"].pop("planner_q_plus_micro_o")
    with pytest.raises(ValidationError):
        run_validator.validate(incomplete_full)


def test_implementation_identity_binds_all_runtime_contract_files() -> None:
    identity = runtime._implementation_identity()
    paths = {item["path"] for item in identity["files"]}
    assert runtime.CONFIG_PATH.as_posix() in paths
    assert runtime.CONFIG_SCHEMA_PATH.as_posix() in paths
    assert runtime.TEACHER_BINDING_SCHEMA_PATH.as_posix() in paths
    assert runtime.TEACHER_RECORD_SCHEMA_PATH.as_posix() in paths
    assert runtime.PHASE_SCHEMA_PATH.as_posix() in paths
    assert runtime.RUN_SCHEMA_PATH.as_posix() in paths
    assert runtime.FAILURE_SCHEMA_PATH.as_posix() in paths
    assert runtime.IMPLEMENTATION_PATH.as_posix() in paths
    assert runtime.SHARED_SOURCE_PATH.as_posix() in paths
    assert runtime.TEST_PATH.as_posix() in paths
    assert len(paths) == len(identity["files"])


def test_launcher_validates_released_teacher_before_gpu_or_lock_access() -> None:
    launcher = (
        runtime._project_root()
        / "scripts/research/run_gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.ps1"
    ).read_text(encoding="utf-8")
    validation_arguments = launcher.index("$ValidationArguments =")
    validation = launcher.index("$ValidationRaw =")
    ready_gate = launcher.index("$Validation.gpu_execution_ready")
    teacher_preflight = launcher.index("$TeacherPreflightRaw =")
    teacher_physical_gate = launcher.index(
        "$TeacherPreflight.all_binding_manifest_receipt_attestation_shard_identities_authenticated"
    )
    helper_import = launcher.index(". $HelperPath")
    lock_probe = launcher.index("if ([IO.File]::Exists($CanonicalLockPath))")
    gpu_probe = launcher.index("$PreLockSamples =")
    lock_acquire = launcher.index("$LockStream = [IO.FileStream]::new(")
    assert (
        validation
        < ready_gate
        < teacher_preflight
        < teacher_physical_gate
        < helper_import
        < lock_probe
        < gpu_probe
        < lock_acquire
    )
    assert "--validate" in launcher[validation_arguments:validation]
    assert "GPU execution remains fail-closed" in launcher[ready_gate:helper_import]


def test_secure_wrapper_is_v3_bound_and_prompts_only_after_preflight() -> None:
    wrapper = (
        runtime._project_root() / "scripts/research/"
        "run_gemma3_chat_unbalanced_v2_glm_rollover_secure_session.ps1"
    ).read_text(encoding="utf-8")

    assert (
        "anchor_mvp.data.gemma3_chat_unbalanced_v2_consumer_identity_rollover_v3"
        in wrapper
    )
    assert "D:\\LLM\\anchor-moe-lora-gemma3-chat-rollover-v3" in wrapper
    assert "D:\\LLM\\anchor-moe-lora-consumer-preflight-clean-6240" in wrapper
    assert "consumer_identity_rollover_v2" not in wrapper
    assert '$credentialEnvironmentVariableName = "glm5.2key"' in wrapper
    assert (
        wrapper.index("$preflight = Get-ControllerPreflight")
        < wrapper.index(
            "$credentialText = [System.Environment]::GetEnvironmentVariable"
        )
        < wrapper.index("$credential = Read-Host")
    )
    assert "$startInfo.EnvironmentVariables.Remove(" in wrapper
    assert "$consumerRepositoryEnvironmentVariableName" in wrapper
    assert (
        "credential_environment_variable = $credentialEnvironmentVariableName"
        in wrapper
    )
    assert wrapper.count("-AsSecureString") == 1
    assert "--credential-stdin" in wrapper
    assert "--implementer-id" in wrapper
    assert "--reviewer-id" in wrapper


def test_invalid_teacher_final_stops_powershell_before_gpu_and_lock() -> None:
    if os.name != "nt":
        pytest.skip("PowerShell launcher gate is Windows-specific")
    project = runtime._project_root()
    owned_parent = project / ".tmp"
    owned_parent.mkdir(exist_ok=True)
    root = owned_parent / f"runtime-v2-bad-teacher-{uuid.uuid4().hex}"
    root.mkdir(exist_ok=False)
    marker = root / "nvidia-smi-called.txt"
    config_path = root / "released-config.yaml"
    binding_path = root / "bad-binding.json"
    sidecar_path = root / "bad-binding.json.sha256"
    shim_path = root / "nvidia-smi.cmd"
    lock_path = project / "runs/formal-v3-training.lock"
    lock_before = (
        None
        if not lock_path.exists()
        else (
            int(lock_path.stat().st_dev),
            int(lock_path.stat().st_ino),
            int(lock_path.stat().st_size),
            int(lock_path.stat().st_mtime_ns),
        )
    )
    try:
        raw_config = runtime.yaml.safe_load(
            (project / runtime.CONFIG_PATH).read_text(encoding="utf-8")
        )
        released = copy.deepcopy(raw_config)
        config_path.write_text(
            runtime.yaml.safe_dump(
                released,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
            newline="\n",
        )
        binding_raw = runtime._canonical_json({})
        binding_sha = hashlib.sha256(binding_raw).hexdigest()
        binding_path.write_bytes(binding_raw)
        sidecar_path.write_text(
            f"{binding_sha}  {binding_path.name}\n",
            encoding="ascii",
            newline="\n",
        )
        shim_path.write_text(
            f'@echo off\r\n(type nul) > "{marker}"\r\nexit /b 99\r\n',
            encoding="ascii",
        )
        environment = dict(os.environ)
        environment["PATH"] = f"{root}{os.pathsep}{environment['PATH']}"
        powershell = (
            Path(os.environ["SystemRoot"])
            / "System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        result = subprocess.run(
            [
                str(powershell),
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(
                    project / "scripts/research/"
                    "run_gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.ps1"
                ),
                "-Mode",
                "SmokeOnly",
                "-TeacherFinalBinding",
                str(binding_path),
                "-TeacherFinalBindingSha256",
                binding_sha,
                "-ExpectedGpuUuid",
                "GPU-test",
                "-PythonExecutable",
                sys.executable,
                "-ConfigPath",
                str(config_path.relative_to(project)),
            ],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert "Teacher FINAL physical preflight failed" in output, output
        assert not marker.exists()
        lock_after = (
            None
            if not lock_path.exists()
            else (
                int(lock_path.stat().st_dev),
                int(lock_path.stat().st_ino),
                int(lock_path.stat().st_size),
                int(lock_path.stat().st_mtime_ns),
            )
        )
        assert lock_after == lock_before
    finally:
        for path in (config_path, binding_path, sidecar_path, shim_path, marker):
            if path.is_file():
                path.unlink()
        root.rmdir()


def test_five_teacher_runtime_schemas_are_git_lf_text() -> None:
    paths = [
        runtime.TEACHER_BINDING_SCHEMA_PATH,
        runtime.TEACHER_RECORD_SCHEMA_PATH,
        runtime.PHASE_SCHEMA_PATH,
        runtime.RUN_SCHEMA_PATH,
        runtime.FAILURE_SCHEMA_PATH,
    ]
    result = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            *(path.as_posix() for path in paths),
        ],
        cwd=runtime._project_root(),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    lines = result.stdout.splitlines()
    assert len(lines) == 10
    for path in paths:
        prefix = path.as_posix()
        assert f"{prefix}: text: set" in lines
        assert f"{prefix}: eol: lf" in lines


def test_runtime_is_python310_syntax_and_uses_compatible_utc() -> None:
    source = (runtime._project_root() / runtime.IMPLEMENTATION_PATH).read_text(
        encoding="utf-8"
    )
    ast.parse(
        source, filename=runtime.IMPLEMENTATION_PATH.as_posix(), feature_version=(3, 10)
    )
    assert "from datetime import UTC" not in source
    assert "datetime.now(UTC)" not in source
    assert "timezone.utc" in source


def test_no_v1_teacher_fallback_or_unsafe_recursive_cleanup() -> None:
    paths = [
        runtime._project_root() / runtime.CONFIG_PATH,
        runtime._project_root() / runtime.CONFIG_SCHEMA_PATH,
        runtime._project_root() / runtime.TEACHER_BINDING_SCHEMA_PATH,
        runtime._project_root() / runtime.TEACHER_RECORD_SCHEMA_PATH,
        runtime._project_root() / runtime.IMPLEMENTATION_PATH,
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "teacher_alignment_binding." + "v" + "1" not in combined
    assert "teacher_alignment_record." + "v" + "1" not in combined
    assert "teacher_alignment_" + "v" + "1" not in combined
    implementation = (runtime._project_root() / runtime.IMPLEMENTATION_PATH).read_text(
        encoding="utf-8"
    )
    assert "shutil.rmtree" not in implementation
    assert "MoveFileExW" not in implementation
    assert "os.replace(staging, published)" not in implementation
    assert "q8_runtime._private_model_snapshot(" not in implementation
    assert "FileDispositionInfo" not in implementation
    assert "_windows_set_handle_delete" not in implementation
    assert "_cleanup_private_model_snapshot_by_handle" not in implementation
    assert implementation.count("_retain_private_model_snapshot_by_handle(") == 2
    assert "with _private_model_snapshot(config, run_id)" in implementation
