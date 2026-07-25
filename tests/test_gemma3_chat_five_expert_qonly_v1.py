from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from jsonschema import Draft202012Validator

from anchor_mvp.training import gemma3_chat_five_expert_qonly_v1 as consumer
from anchor_mvp.training.config import ConfigError


ROOT = Path(__file__).resolve().parents[1]
FAKE_RUNTIME_PACKAGE_VERSIONS = {
    "torch": "2.7.1",
    "transformers": "4.53.2",
    "peft": "0.16.0",
    "safetensors": "0.5.3",
    "sentencepiece": "0.2.0",
    "bitsandbytes": "0.48.2",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _manifest() -> dict[str, object]:
    return {
        "schema_version": consumer.MANIFEST_VERSION,
        "producer": {
            "namespace": "anchor.gemma3-chat-five-expert-qonly.v1",
            "config_path": consumer.PRODUCER_CONFIG_PATH,
            "config_sha256": "1" * 64,
            "implementation_path": consumer.PRODUCER_IMPLEMENTATION_PATH,
            "implementation_sha256": "2" * 64,
        },
        "schemas": {
            "record_path": consumer.RECORD_SCHEMA_PATH,
            "record_sha256": "3" * 64,
            "manifest_path": consumer.MANIFEST_SCHEMA_PATH,
            "manifest_sha256": "4" * 64,
        },
        "partitions": [
            {
                "split": "train",
                "path": consumer.PARTITION_PATHS["train"],
                "records": 800,
                "bytes": 800,
                "sha256": "5" * 64,
            },
            {
                "split": "eval_proxy",
                "path": consumer.PARTITION_PATHS["eval_proxy"],
                "records": 200,
                "bytes": 200,
                "sha256": "6" * 64,
            },
        ],
        "counts": {
            "records": 1000,
            "task_bundles": 200,
            "task_semantics": 200,
            "records_per_bundle": 5,
            "train_bundles": 160,
            "eval_proxy_bundles": 40,
            "train_records": 800,
            "eval_proxy_records": 200,
            "train_records_per_role": 160,
            "eval_proxy_records_per_role": 40,
        },
        "roles": {
            "ordered": list(consumer.ROLES),
            "role_index": consumer.ROLE_INDEX,
        },
        "split_contract": {
            "group_key": "task_bundle_sha256",
            "bundle_disjoint": True,
            "semantic_disjoint": True,
            "eval_proxy_is_heldout": False,
        },
        "bundle_contract": {
            "all_five_roles_per_bundle": True,
            "shared_user_message": True,
            "shared_user_emotion": True,
            "shared_router_label": True,
            "user_emotion_is_expert": False,
            "router_label_is_expert": False,
        },
        "semantic_identity_contract": {
            "descriptor_algorithm": (
                "canonical_json({task_kind,operation,operands,constraints,"
                "expected_relation})_utf8_sha256_v1"
            ),
            "descriptor_keys": [
                "task_kind",
                "operation",
                "operands",
                "constraints",
                "expected_relation",
            ],
            "unique_task_semantics": 200,
            "task_bundle_count": 200,
            "en_unique_task_semantics": 100,
            "zh_cn_unique_task_semantics": 100,
            "en_zh_semantic_intersection_count": 0,
            "translation_pair_count": 0,
            "index_or_language_or_namespace_salt_used": False,
            "template_count": 20,
            "train_eval_template_intersection_count": 14,
            "train_eval_semantic_intersection_count": 0,
            "template_disjoint_claimed": False,
            "eval_proxy_scope": "seen_template_parameter_interpolation",
        },
        "review_contract": {
            "pass_bundles": 100,
            "fail_bundles": 100,
            "fault_counts": {
                "format": 25,
                "grounding": 25,
                "routing": 25,
                "style": 25,
            },
            "parent_target_hashes_bound": True,
            "committed_projection_hash_bound": True,
            "mutation_hash_bound": True,
        },
        "serialization_contract": {
            "version": "anchor.gemma3-it-chat-sft-serialization.v1",
            "model_architecture": "Gemma3ForCausalLM",
            "chat_template_policy_sha256": (
                "0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6"
            ),
            "token_ids_in_records": False,
        },
        "causal_contract": {
            "version": "anchor.chat-five-expert-causal-filter.v1",
            "assistant_turns_forbidden_in_input": True,
            "own_target_forbidden_in_input": True,
            "unapproved_sibling_targets_forbidden_in_input": True,
            "review_dependencies_digest_bound": True,
            "tool_calls_reference_only": True,
            "real_tool_executions": 0,
        },
        "claims": {
            "distilled_chat_sft": True,
            "dataset_materialized": True,
            "training_authorized": False,
            "formal": False,
            "eval_proxy_is_heldout": False,
        },
        "audit": {
            "protected_body_reads": 0,
            "heldout_reads": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "real_tool_executions": 0,
            "single_bytes_snapshot": True,
            "final_snapshot_recheck": True,
            "atomic_publish": True,
        },
    }


def _bundle(bundle_index: int) -> list[dict[str, object]]:
    split = "train" if bundle_index < 160 else "eval_proxy"
    language = "zh-CN" if bundle_index % 2 == 0 else "en"
    user_text = f"synthetic user request for bundle {bundle_index:03d}"
    user_sha = _digest(user_text)
    bundle_sha = _digest(f"bundle:{bundle_index}")
    semantic_sha = _digest(f"semantic:{bundle_index}")
    record_ids = {role: f"record-{bundle_index:03d}-{role}" for role in consumer.ROLES}
    emotion = {
        "taxonomy_version": "anchor.user-emotion.v1",
        "label": "neutral",
        "confidence": 0.75,
        "evidence_message_sha256": user_sha,
    }
    router = {
        "policy_version": "anchor.chat-five-expert-router.v1",
        "label": consumer.ROLES[bundle_index % len(consumer.ROLES)],
        "confidence": 0.8,
        "user_message_sha256": user_sha,
    }
    persona = bundle_index < 5
    persona_core = "我是由Air训练的测试模型。"
    targets = {
        role: (
            f"synthetic target {role}: {persona_core}"
            if persona
            else f"synthetic target {role} for bundle {bundle_index:03d}"
        )
        for role in consumer.ROLES[:3]
    }
    call_id = f"synthetic-call-{bundle_index:03d}"
    tool_response = (
        {
            "mode": "direct_no_tool_identity",
            "tool_decision": {
                "action": "no_tool_required",
                "rationale": "synthetic persona contract",
            },
            "tool_calls": [],
            "tool_results": [],
            "grounded_final": persona_core,
        }
        if persona
        else {
            "mode": "synthetic_local_call",
            "tool_call": {
                "name": "weather.lookup",
                "call_id": call_id,
                "arguments": {"bundle": bundle_index},
            },
            "tool_result": {
                "call_id": call_id,
                "result": {"temperature_c": 20},
            },
            "grounded_final": f"synthetic weather answer {bundle_index:03d}",
        }
    )
    targets["tool_call"] = _canonical(
        {
            "emotion_router": {
                "label": emotion["label"],
                "concise_routing_rationale": "synthetic routing rationale",
            },
            "routing": {
                "selected_expert": router["label"],
                "reason": f"bundle_shared_route_to_{router['label']}",
            },
            "counterfactual_view_role": "tool_call",
            "expert_response": tool_response,
        }
    ).decode("utf-8")
    target_hashes = {role: _digest(text) for role, text in targets.items()}
    dependencies = [
        {
            "record_id": record_ids[role],
            "role": role,
            "target_sha256": target_hashes[role],
        }
        for role in consumer.ROLES[:4]
    ]
    dependency_set_sha256 = _digest(
        _canonical(
            {
                "domain": "anchor.chat-review-dependency-set.v1",
                "dependencies": dependencies,
            }
        )
    )
    fault_type = None
    fault_role = None
    if bundle_index % 2:
        fault_type = ("format", "grounding", "routing", "style")[
            (bundle_index // 2) % 4
        ]
        fault_role = (
            "tool_call"
            if fault_type == "grounding"
            else consumer.ROLES[:4][(bundle_index // 8) % len(consumer.ROLES[:4])]
        )
    views: dict[str, dict[str, object]] = {
        role: {
            "mode": "direct_response",
            "route_claim": role,
            "style_claim": role,
            "text_present": True,
        }
        for role in consumer.ROLES[:3]
    }
    views["tool_call"] = (
        {
            "mode": "direct_no_tool_identity",
            "route_claim": "tool_call",
            "grounded_final_present": True,
        }
        if persona
        else {
            "mode": "synthetic_local_call",
            "route_claim": "tool_call",
            "tool_name": "weather.lookup",
            "arguments_present": True,
            "result_present": True,
            "grounded_final_present": True,
        }
    )
    if fault_type == "format":
        views[str(fault_role)] = {"mode": "invalid_committed_projection"}
    elif fault_type == "grounding":
        views["tool_call"] = {
            "mode": tool_response["mode"],
            "tool_name": (None if persona else tool_response["tool_call"]["name"]),
            "arguments_present": True,
            "result_present": False,
            "grounded_final_present": False,
        }
    elif fault_type == "routing":
        role = str(fault_role)
        views[role]["route_claim"] = "serious" if role != "serious" else "humor"
    elif fault_type == "style":
        views[str(fault_role)]["style_claim"] = "serious"
    mutation_sha256 = _digest(
        _canonical(
            {
                "domain": "anchor.chat-review-projection-mutation.v1",
                "fault_type": fault_type,
                "fault_role": fault_role,
                "parent_target_sha256": {
                    role: target_hashes[role] for role in consumer.ROLES[:4]
                },
            }
        )
    )
    projection = {
        "dependency_set_sha256": dependency_set_sha256,
        "fault_type": fault_type,
        "fault_role": fault_role,
        "views": views,
    }
    projection_text = _canonical(projection).decode("utf-8")
    projection_sha256 = _digest(projection_text)
    review_target = {
        "mode": "four_parent_review",
        "verdict": "pass" if fault_type is None else "fail",
        "checks": [
            {
                "role": role,
                "status": "fail" if role == fault_role else "pass",
                "issue": fault_type if role == fault_role else None,
            }
            for role in consumer.ROLES[:4]
        ],
        "summary": f"synthetic review {bundle_index:03d}",
        "parent_count": 4,
        "dependency_set_sha256": dependency_set_sha256,
        "committed_projection_sha256": projection_sha256,
        "mutation_sha256": mutation_sha256,
        "fault_type": fault_type,
        "fault_role": fault_role,
        "correction_required": fault_type is not None,
    }
    targets["review_audit"] = _canonical(review_target).decode("utf-8")
    target_hashes["review_audit"] = _digest(targets["review_audit"])
    records: list[dict[str, object]] = []
    for role in consumer.ROLES:
        system_text = f"synthetic role instruction: {role}"
        role_dependencies: list[dict[str, str]] = []
        if role == "review_audit":
            role_dependencies = copy.deepcopy(dependencies)
            system_text += (
                "\n[COMMITTED_FROZEN_BASE_REENCODED_PARENT_OUTPUTS]\n" + projection_text
            )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        messages_sha = _digest(_canonical(messages))
        if role == "tool_call" and not persona:
            tool_call = tool_response["tool_call"]
            tool_result = tool_response["tool_result"]
            call_sha = _digest(_canonical(tool_call))
            result_sha = _digest(_canonical(tool_result))
            schemas = [
                {
                    "ref_id": f"synthetic-tool-schema:{tool_call['name']}",
                    "sha256": call_sha,
                }
            ]
            evidence = [
                {
                    "ref_id": f"synthetic-tool-result:{tool_call['call_id']}",
                    "sha256": result_sha,
                }
            ]
            grounding = {
                "required": True,
                "tool_schema_refs": schemas,
                "evidence_refs": evidence,
                "grounding_sha256": _digest(
                    _canonical(
                        {
                            "domain": ("anchor.chat-synthetic-local-tool-grounding.v1"),
                            "tool_call_sha256": call_sha,
                            "tool_result_sha256": result_sha,
                            "grounded_final_sha256": _digest(
                                tool_response["grounded_final"]
                            ),
                        }
                    )
                ),
                "real_tool_executions": 0,
            }
        else:
            grounding = {
                "required": False,
                "tool_schema_refs": [],
                "evidence_refs": [],
                "grounding_sha256": None,
                "real_tool_executions": 0,
            }
        review = {
            "required": role == "review_audit",
            "dependencies": role_dependencies,
            "dependency_set_sha256": (
                dependency_set_sha256 if role == "review_audit" else None
            ),
            "committed_projection_sha256": (
                projection_sha256 if role == "review_audit" else None
            ),
            "mutation_sha256": (mutation_sha256 if role == "review_audit" else None),
            "fault_type": fault_type if role == "review_audit" else None,
        }
        allowed = [item["target_sha256"] for item in role_dependencies]
        forbidden = [
            target_hashes[candidate]
            for candidate in consumer.ROLES
            if target_hashes[candidate] not in set(allowed)
        ]
        causal_body = {
            "contract_version": "anchor.chat-five-expert-causal-filter.v1",
            "input_messages_sha256": messages_sha,
            "own_target_sha256": target_hashes[role],
            "allowed_dependency_target_sha256": allowed,
            "no_post_target_messages": True,
        }
        causal = {
            **causal_body,
            "proof_sha256": _digest(
                _canonical(
                    {
                        "domain": "anchor.chat-five-expert-causal-proof.v1",
                        "proof": causal_body,
                    }
                )
            ),
        }
        prompt = f"[SYSTEM_INSTRUCTION]\n{system_text}\n[USER_MESSAGE]\n{user_text}"
        training_identity = {
            "contract_version": "anchor.gemma3-it-chat-sft-serialization.v1",
            "messages": messages,
            "serialized_prompt": prompt,
            "assistant_text": targets[role],
        }
        records.append(
            {
                "schema_version": consumer.RECORD_VERSION,
                "record_id": record_ids[role],
                "task_bundle_sha256": bundle_sha,
                "task_semantic_sha256": semantic_sha,
                "split": split,
                "language": language,
                "role": role,
                "role_index": consumer.ROLE_INDEX[role],
                "user_emotion": copy.deepcopy(emotion),
                "router": copy.deepcopy(router),
                "messages": messages,
                "target": {
                    "assistant_text": targets[role],
                    "format": (
                        "tool_call_json"
                        if role == "tool_call"
                        else "review_audit_json"
                        if role == "review_audit"
                        else "chat_text"
                    ),
                    "output_sha256": target_hashes[role],
                },
                "gemma_serialization": {
                    "contract_version": ("anchor.gemma3-it-chat-sft-serialization.v1"),
                    "chat_template_policy_sha256": (
                        "0ffb2e2597428da4ee5d727697bcb29a"
                        "116489e220580c7bd1cb5bd8f2e076b6"
                    ),
                    "input_messages_sha256": messages_sha,
                    "serialized_prompt_sha256": _digest(prompt),
                    "serialized_target_sha256": target_hashes[role],
                    "serialized_training_example_sha256": _digest(
                        _canonical(training_identity)
                    ),
                    "assistant_prefix_included": True,
                    "add_generation_prompt": True,
                    "token_ids_included": False,
                },
                "tool_grounding": grounding,
                "review_dependency": review,
                "content_leak_proof": {
                    "method": "exact_target_and_digest_scan_v1",
                    "forbidden_target_sha256": forbidden,
                    "exact_target_text_absent": True,
                    "target_digest_literal_absent": True,
                },
                "causal_proof": causal,
                "claims": {
                    "distilled_chat_sft": True,
                    "user_emotion_is_expert": False,
                    "router_label_is_expert": False,
                    "training_authorized": False,
                    "formal": False,
                    "eval_proxy_is_heldout": False,
                },
                "audit": {
                    "protected_body_reads": 0,
                    "heldout_reads": 0,
                    "provider_requests": 1,
                    "network_requests": 1,
                    "model_loads": 0,
                    "gpu_requests": 0,
                    "real_tool_executions": 0,
                },
            }
        )
    return records


@pytest.fixture(scope="module")
def global_fixture() -> tuple[list[dict[str, object]], dict[str, object]]:
    records = [
        record for bundle_index in range(200) for record in _bundle(bundle_index)
    ]
    return records, _manifest()


def _rebind_messages(record: dict[str, object]) -> None:
    messages = record["messages"]
    assert isinstance(messages, list)
    messages_sha = _digest(_canonical(messages))
    assert len(messages) == 2
    prompt = (
        f"[SYSTEM_INSTRUCTION]\n{messages[0]['content']}"
        f"\n[USER_MESSAGE]\n{messages[1]['content']}"
    )
    target = record["target"]
    assert isinstance(target, dict)
    target_text = str(target["assistant_text"])
    serialization = record["gemma_serialization"]
    assert isinstance(serialization, dict)
    serialization["input_messages_sha256"] = messages_sha
    serialization["serialized_prompt_sha256"] = _digest(prompt)
    serialization["serialized_training_example_sha256"] = _digest(
        _canonical(
            {
                "contract_version": ("anchor.gemma3-it-chat-sft-serialization.v1"),
                "messages": messages,
                "serialized_prompt": prompt,
                "assistant_text": target_text,
            }
        )
    )
    causal = record["causal_proof"]
    assert isinstance(causal, dict)
    causal["input_messages_sha256"] = messages_sha
    body = {key: value for key, value in causal.items() if key != "proof_sha256"}
    causal["proof_sha256"] = _digest(
        _canonical(
            {
                "domain": "anchor.chat-five-expert-causal-proof.v1",
                "proof": body,
            }
        )
    )


def _pending_config() -> dict[str, object]:
    config = copy.deepcopy(consumer.load_config())
    config["identity_state"] = "pending_producer_handoff"
    config["claim_scope"] = (
        "diagnostic_training_engine_waiting_for_authenticated_distilled_dataset"
    )
    config["producer"]["config_sha256"] = "pending"
    config["producer"]["implementation_sha256"] = "pending"
    config["dataset"]["record_schema_sha256"] = "pending"
    config["dataset"]["manifest_schema_sha256"] = "pending"
    config["dataset"]["manifest_sha256"] = "pending"
    config["runner"]["status"] = "engine_ready_waiting_for_authenticated_producer"
    config["claims"]["dataset_materialized"] = False
    config["claims"]["producer_identity_bound"] = False
    consumer.validate_config(config)
    return config


def test_config_locks_full_effective_rank_and_final_identities() -> None:
    config = consumer.load_config()
    assert config["lora"] == {
        "profile": "q_only",
        "factorization": "full_rank_q",
        "target_modules": ["q_proj"],
        "rank": 1024,
        "alpha": 2048,
        "dropout": 0.0,
        "bias": "none",
        "expected_trainable_parameters_per_expert": 57_933_824,
        "expected_five_expert_aggregate_parameters": 289_669_120,
        "effective_rank_ceiling": 1024,
        "low_rank_claimed": False,
        "parameter_efficient_claimed": False,
        "o_proj_allowed": False,
    }
    assert config["training"]["smoke_steps"] == 2
    assert config["training"]["full_steps_per_role"] == 160
    assert config["training"]["role_execution"] == "strictly_serial"
    assert config["training"]["sequence_length"] == 768
    assert config["training"]["truncation"] is False
    assert config["training"]["micro_batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["optimizer"] == "adamw8bit"
    assert config["training"]["optimizer_state_bits"] == 8
    assert config["training"]["learning_rate"] == 0.000002
    assert config["training"]["warmup_steps"] == 8
    assert config["training"]["weight_decay"] == 0.01
    assert config["training"]["max_grad_norm"] == 0.5
    assert config["quantization"] == consumer.Q8_BASE_CONTRACT
    assert consumer.pending_producer_identities(config) == ()
    assert config["identity_state"] == "authenticated_producer_handoff"
    assert config["producer"]["config_sha256"] == (
        "2e93ab6b53f68814a4ae017643ddc4c7113716e871b2fa4ef4346358d2269cee"
    )
    assert config["producer"]["implementation_sha256"] == (
        "98eeac4150a5c51dd422bfe6163e34ce26d691038b497343c20af0a19552fb6f"
    )
    assert config["producer"]["closed_grammar_sha256"] == (
        "3a76e9cb7eb0e451a435e5d6ef4dd8db3624b5674a7a546f3b709e33339a7d92"
    )
    assert config["dataset"]["record_schema_sha256"] == (
        "ba1606f07d250eec7f01bffceaa507edd4f596f83018806df9130d32ecf39d9b"
    )
    assert config["dataset"]["manifest_schema_sha256"] == (
        "fda04d77c494cd5d87ea19aade7db24717c6dffad6880078df19787d519f0b8b"
    )
    assert config["dataset"]["manifest_sha256"] == (
        "e71fb5f239d7d1abb94ff50d5db2d57cfd101ccfad3c2c4e34cc128a7e91317b"
    )
    assert config["dataset"]["token_inventory_sha256"] == (
        "d5c149264e7bb2e4c2256bda72422bd1fdf41822b012c29d006035edabd8f2f6"
    )
    assert config["dataset"]["build_receipt_sha256"] == (
        "013d55017c82d3d407477eca673f8990da02e2bbb037109ea3b3feb7f98dfc82"
    )
    assert config["dataset"]["partition_sha256"] == {
        "train/chat.jsonl": (
            "26fd3f5c30d402ab684d67977b8e790470c85f2c33436047d1cdb237c00379dc"
        ),
        "eval_proxy/chat.jsonl": (
            "2e1434229482b21ee908cfee8f389523c6730691ade177f47948ad8350821a86"
        ),
    }


def test_checked_in_final_dataset_passes_model_free_preflight_without_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in consumer.EXTERNAL_LOCK_ENV.values():
        monkeypatch.delenv(name, raising=False)
    config = consumer.load_config()
    preflight = consumer.build_preflight(config)
    execute = consumer.build_execute_block(config)
    assert preflight["status"] == (
        "passed_model_free_authenticated_dataset_ready_for_explicit_execute"
    )
    assert preflight["pending_producer_identities"] == []
    assert preflight["dataset"]["manifest_sha256"] == (
        "e71fb5f239d7d1abb94ff50d5db2d57cfd101ccfad3c2c4e34cc128a7e91317b"
    )
    assert preflight["dataset"]["partition_sha256"] == {
        "train/chat.jsonl": (
            "26fd3f5c30d402ab684d67977b8e790470c85f2c33436047d1cdb237c00379dc"
        ),
        "eval_proxy/chat.jsonl": (
            "2e1434229482b21ee908cfee8f389523c6730691ade177f47948ad8350821a86"
        ),
    }
    assert preflight["serial_plan"]["status"] == (
        "authenticated_dataset_plan_only_external_lock_required"
    )
    assert preflight["serial_plan"]["execution"]["model_loaded"] is False
    assert preflight["serial_plan"]["execution"]["gpu_requested"] is False
    assert preflight["audit"] == {
        "protected_body_reads": 0,
        "heldout_reads": 0,
        "dataset_body_reads": 3,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "training_runs": 0,
    }
    assert execute["status"] == "blocked_waiting_for_external_gpu_lock"
    assert execute["reason"] == "chat_external_gpu_lock_required"
    assert execute["audit"]["model_loads"] == 0
    assert execute["audit"]["gpu_requests"] == 0
    assert execute["audit"]["training_runs"] == 0


def test_final_dataset_bad_hash_and_sidecar_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = consumer.load_config()
    bad_hash = copy.deepcopy(config)
    bad_hash["dataset"]["manifest_sha256"] = "f" * 64
    with pytest.raises(
        ConfigError,
        match="chat_authenticated_source_identity_mismatch",
    ):
        consumer.build_preflight(bad_hash)

    original_read = consumer.snapshot_io._read_snapshot
    drifted_sidecar = b"f" * 64 + b"  manifest.json\n"

    def read_with_bad_sidecar(path: Path, *, max_bytes: int) -> object:
        snapshot = original_read(path, max_bytes=max_bytes)
        if Path(path).name == "manifest.json.sha256":
            return replace(
                snapshot,
                data=drifted_sidecar,
                sha256=_digest(drifted_sidecar),
            )
        return snapshot

    bad_sidecar = copy.deepcopy(config)
    bad_sidecar["dataset"]["manifest_sidecar_sha256"] = _digest(drifted_sidecar)
    monkeypatch.setattr(
        consumer.snapshot_io,
        "_read_snapshot",
        read_with_bad_sidecar,
    )
    with pytest.raises(ConfigError, match="chat_manifest_sidecar_invalid"):
        consumer.build_preflight(bad_sidecar)


def test_pending_identity_blocks_before_any_dataset_path_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _pending_config()

    def unexpected_path_read(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError(
            "dataset path must not be touched while identity is pending"
        )

    monkeypatch.setattr(consumer, "_repo_path", unexpected_path_read)
    with pytest.raises(ConfigError, match="chat_producer_identity_pending"):
        consumer.load_role_dataset(config, "humor")


def test_partial_or_fake_producer_identity_fails_closed() -> None:
    config = _pending_config()
    partial = copy.deepcopy(config)
    partial["producer"]["config_sha256"] = "a" * 64
    with pytest.raises(ConfigError, match="chat_claims_must_remain_blocked"):
        consumer.validate_config(partial)
    fake = copy.deepcopy(config)
    fake["producer"]["config_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="chat_producer_config_sha_invalid"):
        consumer.validate_config(fake)


def test_fully_bound_future_state_is_declaratively_accepted() -> None:
    config = consumer.load_config()
    bound = copy.deepcopy(config)
    bound["identity_state"] = "authenticated_producer_handoff"
    bound["producer"]["config_sha256"] = "a" * 64
    bound["producer"]["implementation_sha256"] = "b" * 64
    bound["dataset"]["record_schema_sha256"] = "c" * 64
    bound["dataset"]["manifest_schema_sha256"] = "d" * 64
    bound["dataset"]["manifest_sha256"] = "e" * 64
    bound["claims"]["dataset_materialized"] = True
    bound["claims"]["producer_identity_bound"] = True
    consumer.validate_config(bound)


def test_closed_schemas_accept_the_synthetic_contract(
    global_fixture: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    records, manifest = global_fixture
    record_schema = json.loads(
        (ROOT / consumer.RECORD_SCHEMA_PATH).read_text(encoding="utf-8")
    )
    manifest_schema = json.loads(
        (ROOT / consumer.MANIFEST_SCHEMA_PATH).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(record_schema)
    Draft202012Validator.check_schema(manifest_schema)
    Draft202012Validator(record_schema).validate(records[0])
    Draft202012Validator(manifest_schema).validate(manifest)
    drifted = copy.deepcopy(records[0])
    drifted["sixth_expert"] = "emotion"
    assert list(Draft202012Validator(record_schema).iter_errors(drifted))


def test_global_contract_accepts_200_bundles_and_1000_rows(
    global_fixture: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    records, manifest = global_fixture
    examples = consumer._validate_global_records(records, manifest)
    assert len(examples) == 1000
    assert len({item.task_bundle_sha256 for item in examples}) == 200
    assert all(
        sum(item.role == role and item.split == "train" for item in examples) == 160
        for role in consumer.ROLES
    )
    assert all(
        sum(item.role == role and item.split == "eval_proxy" for item in examples) == 40
        for role in consumer.ROLES
    )


def test_emotion_and_router_are_shared_metadata_not_a_sixth_expert(
    global_fixture: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    source, manifest = global_fixture
    records = copy.deepcopy(source)
    assert set(consumer.ROLES) == {
        "humor",
        "serious",
        "angry_style",
        "tool_call",
        "review_audit",
    }
    records[0]["role"] = "user_emotion"
    with pytest.raises(ConfigError, match="chat_record_identity_invalid"):
        consumer._strict_record(records[0])


def test_bundle_split_and_shared_router_drift_fail_closed(
    global_fixture: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    source, manifest = global_fixture
    records = copy.deepcopy(source[25:30])
    records[0]["split"] = "eval_proxy"
    with pytest.raises(ConfigError, match="chat_bundle_cross_binding_invalid"):
        consumer._validate_bundle_records(
            tuple(consumer._strict_record(item) for item in records)
        )
    records = copy.deepcopy(source[:5])
    router = records[0]["router"]
    assert isinstance(router, dict)
    router["label"] = "review_audit"
    with pytest.raises(ConfigError, match="chat_bundle_cross_binding_invalid"):
        consumer._validate_bundle_records(
            tuple(consumer._strict_record(item) for item in records)
        )


def test_assistant_input_and_forbidden_target_leak_fail_closed(
    global_fixture: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    source, manifest = global_fixture
    records = copy.deepcopy(source[:5])
    messages = records[0]["messages"]
    assert isinstance(messages, list)
    messages[0]["role"] = "assistant"
    with pytest.raises(
        ConfigError, match="chat_assistant_or_invalid_input_message_forbidden"
    ):
        consumer._strict_record(records[0])
    records = copy.deepcopy(source[:5])
    messages = records[0]["messages"]
    target = records[0]["target"]
    assert isinstance(messages, list)
    assert isinstance(target, dict)
    messages[0]["content"] += f"; leaked={target['assistant_text']}"
    _rebind_messages(records[0])
    with pytest.raises(
        ConfigError, match="chat_forbidden_target_content_reached_input"
    ):
        consumer._validate_bundle_records(
            tuple(consumer._strict_record(item) for item in records)
        )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "one"}],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "user", "content": "extra"},
        ],
    ],
)
def test_model_free_consumer_rejects_noncanonical_message_cardinality(
    messages: list[dict[str, str]],
) -> None:
    with pytest.raises(
        ConfigError,
        match="chat_messages_cardinality_invalid",
    ):
        consumer._strict_messages(messages)


@pytest.mark.parametrize(
    ("role", "target", "error"),
    [
        (
            "humor",
            '{"answer":"JSON envelope disguised as chat"}',
            "chat_natural_language_target_must_not_be_json_envelope",
        ),
        (
            "tool_call",
            "plain text tool answer",
            "chat_structured_target_must_be_strict_json_object",
        ),
        (
            "review_audit",
            '{"verdict":"pass","verdict":"fail"}',
            "chat_structured_target_must_be_strict_json_object",
        ),
        (
            "review_audit",
            '{"score":NaN}',
            "chat_structured_target_must_be_strict_json_object",
        ),
        (
            "review_audit",
            '{"score":1e400}',
            "chat_structured_target_must_be_strict_json_object",
        ),
    ],
)
def test_target_shapes_separate_natural_chat_from_strict_json(
    role: str,
    target: str,
    error: str,
) -> None:
    with pytest.raises(ConfigError, match=error):
        consumer._strict_target_shape(role, target)


def test_tool_and_review_dependencies_are_digest_bound(
    global_fixture: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    source, manifest = global_fixture
    records = copy.deepcopy(source[25:30])
    tool = next(item for item in records if item["role"] == "tool_call")
    grounding = tool["tool_grounding"]
    assert isinstance(grounding, dict)
    grounding["grounding_sha256"] = "f" * 64
    with pytest.raises(
        ConfigError,
        match="chat_tool_grounding_payload_cross_binding_invalid",
    ):
        consumer._validate_bundle_records(
            tuple(consumer._strict_record(item) for item in records)
        )
    records = copy.deepcopy(source[:5])
    review = next(item for item in records if item["role"] == "review_audit")
    dependency = review["review_dependency"]
    assert isinstance(dependency, dict)
    dependencies = dependency["dependencies"]
    assert isinstance(dependencies, list)
    dependencies[0]["record_id"] = "missing-record"
    dependency["dependency_set_sha256"] = _digest(
        _canonical(
            {
                "domain": "anchor.chat-review-dependency-set.v1",
                "dependencies": dependencies,
            }
        )
    )
    with pytest.raises(
        ConfigError, match="chat_review_dependency_cross_binding_invalid"
    ):
        consumer._validate_bundle_records(
            tuple(consumer._strict_record(item) for item in records)
        )


def test_review_projection_mutation_and_target_are_cross_bound(
    global_fixture: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    source, _manifest_value = global_fixture
    examples = tuple(consumer._strict_record(item) for item in source[:5])
    review = next(item for item in examples if item.role == "review_audit")
    review_index = examples.index(review)

    dependency = dict(review.review_dependency)
    dependency["mutation_sha256"] = "f" * 64
    drifted = list(examples)
    drifted[review_index] = replace(review, review_dependency=dependency)
    with pytest.raises(
        ConfigError,
        match="chat_review_projection_cross_binding_invalid",
    ):
        consumer._validate_bundle_records(tuple(drifted))

    marker = "[COMMITTED_FROZEN_BASE_REENCODED_PARENT_OUTPUTS]\n"
    system = review.messages[0][1]
    prefix, projection_text = system.split(marker, 1)
    projection = json.loads(projection_text)
    projection["views"]["humor"]["text_present"] = False
    drifted = list(examples)
    drifted[review_index] = replace(
        review,
        messages=(
            ("system", prefix + marker + _canonical(projection).decode("utf-8")),
            review.messages[1],
        ),
    )
    with pytest.raises(
        ConfigError,
        match="chat_review_projection_cross_binding_invalid",
    ):
        consumer._validate_bundle_records(tuple(drifted))

    target = json.loads(review.target)
    target["verdict"] = "fail"
    drifted = list(examples)
    drifted[review_index] = replace(
        review,
        target=_canonical(target).decode("utf-8"),
    )
    with pytest.raises(
        ConfigError,
        match="chat_review_target_cross_binding_invalid",
    ):
        consumer._validate_bundle_records(tuple(drifted))


def test_causal_inventory_serialization_and_global_review_quota_fail_closed(
    global_fixture: tuple[list[dict[str, object]], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, manifest = global_fixture
    record = copy.deepcopy(source[0])
    causal = record["causal_proof"]
    assert isinstance(causal, dict)
    body = {key: value for key, value in causal.items() if key != "proof_sha256"}
    causal["proof_sha256"] = _digest(_canonical(body))
    with pytest.raises(ConfigError, match="chat_causal_proof_binding_invalid"):
        consumer._strict_record(record)

    record = copy.deepcopy(source[0])
    serialization = record["gemma_serialization"]
    assert isinstance(serialization, dict)
    serialization["serialized_prompt_sha256"] = "f" * 64
    with pytest.raises(
        ConfigError,
        match="chat_gemma_serialization_identity_invalid",
    ):
        consumer._strict_record(record)

    records = copy.deepcopy(source[:5])
    leak = records[0]["content_leak_proof"]
    assert isinstance(leak, dict)
    forbidden = leak["forbidden_target_sha256"]
    assert isinstance(forbidden, list)
    forbidden[0], forbidden[1] = forbidden[1], forbidden[0]
    with pytest.raises(
        ConfigError,
        match="chat_causal_inventory_cross_binding_invalid",
    ):
        consumer._validate_bundle_records(
            tuple(consumer._strict_record(item) for item in records)
        )

    monkeypatch.setattr(consumer, "_validate_bundle_records", lambda _items: None)
    with pytest.raises(ConfigError, match="chat_review_global_quota_invalid"):
        consumer._validate_global_records(source, manifest)


def test_pending_preflight_and_execute_gate_are_content_free_and_blocked() -> None:
    config = _pending_config()
    preflight = consumer.build_preflight(config)
    execute = consumer.build_execute_block(config)
    assert preflight["status"] == "blocked_waiting_for_authenticated_producer"
    assert execute["status"] == "blocked_waiting_for_authenticated_producer"
    assert execute["reason"] == "chat_producer_identity_pending"
    assert preflight["claims"]["training_execution_supported"] is True
    assert execute["claims"]["training_execution_supported"] is True
    assert preflight["audit"] == {
        "protected_body_reads": 0,
        "heldout_reads": 0,
        "dataset_body_reads": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "training_runs": 0,
    }
    assert execute["audit"] == preflight["audit"]


def test_atomic_receipt_publish_is_no_replace(tmp_path: Path) -> None:
    config = consumer.load_config()
    value = consumer.build_execute_block(config)
    output = tmp_path / "receipt-run"
    receipt = consumer._atomic_publish_receipt_directory(
        output,
        filename="execute_block_receipt.json",
        value=value,
    )
    raw = receipt.read_bytes()
    digest = _digest(raw)
    assert receipt.with_name(receipt.name + ".sha256").read_bytes() == (
        f"{digest}  {receipt.name}\n".encode("ascii")
    )
    with pytest.raises(FileExistsError):
        consumer._atomic_publish_receipt_directory(
            output,
            filename="execute_block_receipt.json",
            value=value,
        )


def _materialize_fake_training_artifacts(
    root: Path,
    *,
    run_id: str,
) -> list[dict[str, object]]:
    roles: list[dict[str, object]] = []
    for role in consumer.ROLES:
        role_root = root / role
        adapter_root = role_root / "adapter"
        adapter_root.mkdir(parents=True)
        adapter_payloads = {
            "adapter_config.json": _canonical(
                {"role": role, "format": "synthetic-test-only"}
            )
            + b"\n",
            "adapter_model.safetensors": f"fake:{role}".encode("ascii"),
        }
        for filename, payload in adapter_payloads.items():
            (adapter_root / filename).write_bytes(payload)
        adapter_hashes = {
            filename: _digest(payload) for filename, payload in adapter_payloads.items()
        }
        receipt_hashes: dict[str, str] = {}
        for phase, steps in (("smoke", 2), ("full", 160)):
            receipt = {
                "schema_version": consumer.PHASE_RECEIPT_VERSION,
                "status": "passed_diagnostic_training_only",
                "run_id": run_id,
                "role": role,
                "phase": phase,
                "optimizer_steps": steps,
                "runtime_package_versions": dict(FAKE_RUNTIME_PACKAGE_VERSIONS),
                "adapter_artifact_sha256": (
                    adapter_hashes if phase == "full" else None
                ),
            }
            receipt_hashes[phase] = consumer._atomic_write_json(
                role_root / f"{phase}_receipt.json",
                receipt,
            )
        roles.append(
            {
                "role": role,
                "smoke_receipt_sha256": receipt_hashes["smoke"],
                "full_receipt_sha256": receipt_hashes["full"],
                "adapter_artifact_sha256": adapter_hashes,
            }
        )
    return roles


def test_training_artifact_snapshots_detect_pre_publish_rewrite(
    tmp_path: Path,
) -> None:
    run_id = "artifact-toctou-test"
    roles = _materialize_fake_training_artifacts(tmp_path, run_id=run_id)
    snapshots, versions = consumer._capture_training_artifacts(
        tmp_path,
        roles=roles,
        run_id=run_id,
    )
    assert len(snapshots) == 30
    assert versions == FAKE_RUNTIME_PACKAGE_VERSIONS
    consumer._assert_success_root_inventory(
        tmp_path,
        run_receipt_name="run_receipt.json",
        receipt_present=False,
    )
    extra = tmp_path / "unbound-extra.txt"
    extra.write_text("must be rejected", encoding="utf-8")
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_success_root_artifact_inventory_invalid",
    ):
        consumer._assert_success_root_inventory(
            tmp_path,
            run_receipt_name="run_receipt.json",
            receipt_present=False,
        )
    extra.unlink()
    progress = tmp_path / "training_progress.json"
    progress.write_text("ephemeral", encoding="utf-8")
    consumer._remove_training_progress_for_publish(tmp_path)
    assert not progress.exists()
    target = tmp_path / "serious/adapter/adapter_model.safetensors"
    target.write_bytes(b"concurrent rewrite")
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match=("chat_training_artifact_(?:snapshot_mismatch|identity_changed)"),
    ):
        snapshots["serious/adapter/adapter_model.safetensors"].assert_unchanged()
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_training_artifact_snapshot_mismatch",
    ):
        consumer._capture_training_artifacts(
            tmp_path,
            roles=roles,
            run_id=run_id,
        )


def test_failure_receipt_has_independent_atomic_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = consumer.load_config()
    monkeypatch.setattr(
        consumer,
        "_output_path",
        lambda _value: tmp_path / "runs",
    )
    receipt = consumer._publish_failure_receipt(
        config,
        run_id="fake-failure",
        error_code="synthetic_failure",
        role="humor",
        phase="smoke",
        failed_staging=None,
    )
    assert receipt is not None
    value = json.loads(receipt.read_text("utf-8"))
    assert value["schema_version"] == consumer.FAILURE_RECEIPT_VERSION
    assert value["sample_content_included"] is False
    assert value["automatic_retry"] is False
    assert receipt.with_name("failure_receipt.json.sha256").is_file()
    assert (
        consumer._publish_failure_receipt(
            config,
            run_id="fake-failure",
            error_code="second_failure",
            role=None,
            phase=None,
            failed_staging=None,
        )
        is None
    )


def test_one_click_launcher_exposes_clear_modes_and_receipt_root() -> None:
    launcher = (
        ROOT / "scripts/research/run_gemma3_chat_five_expert_qonly_v1.ps1"
    ).read_text(encoding="utf-8-sig")
    implementation = (ROOT / consumer.IMPLEMENTATION_PATH).read_text(encoding="utf-8")
    assert "[switch]$Preflight" in launcher
    assert "[switch]$Execute" in launcher
    assert '"--preflight"' in launcher
    assert "--dry-run" in launcher
    assert '"--execute"' in launcher
    assert "ANCHOR_GEMMA_GPU_UUID" in launcher
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "Invoke-AnchorGemmaQ8PythonAndWait" in launcher
    assert "[IO.FileMode]::CreateNew" in launcher
    assert "[IO.FileAccess]::ReadWrite" in launcher
    assert "[IO.FileShare]::None" in launcher
    assert "[IO.FileOptions]::DeleteOnClose" in launcher
    assert "ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_RECEIPT" in launcher
    assert "ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_SHA256" in launcher
    assert "ANCHOR_CHAT_EXTERNAL_LOCK_NONCE" in launcher
    assert "ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID" in launcher
    assert "$env:ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID = [string]$PID" in launcher
    assert launcher.index(
        "$env:ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID = [string]$PID"
    ) > launcher.rindex("Publish-ChatLockOwnerReceipt")
    assert "os.getppid(" not in implementation
    assert "Publish-ChatLockOwnerReceipt" in launcher
    assert "Assert-ChatLockStreamDigest" in launcher
    assert "runs/formal-v3-training.lock" in launcher
    assert "runs/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1/" in launcher
    for relative in consumer.EXECUTION_DEPENDENCY_PATHS:
        assert f'"{relative}"' in launcher


def test_code_snapshots_bind_every_runtime_dependency() -> None:
    snapshots = consumer._code_snapshots()
    config = consumer.load_config()
    assert set(snapshots) == {
        "implementation",
        "runner_script",
        "launcher",
        "config",
        *(f"dependency:{relative}" for relative in consumer.EXECUTION_DEPENDENCY_PATHS),
    }
    assert snapshots["config"].sha256 == config["_config_sha256"]
    assert (
        snapshots[f"dependency:{consumer.TOKENIZER_POLICY_PATH}"].sha256
        == config["model"]["tokenizer_policy_sha256"]
    )


def _external_lock_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
) -> tuple[dict[str, object], Path, Path, dict[str, object], bytes, str]:
    config = copy.deepcopy(consumer.load_config())
    config["gpu_policy"]["expected_gpu_uuid"] = "GPU-" + "a" * 32
    monkeypatch.setattr(
        consumer.qdiag,
        "_project_root_from_module",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        consumer.qdiag,
        "_assert_physical_path",
        lambda path, **_kwargs: Path(path),
    )
    monkeypatch.setattr(
        consumer,
        "_stream_sha256",
        lambda *_args, **_kwargs: "b" * 64,
    )
    owner = consumer._expected_external_lock_owner(
        config,
        run_id=run_id,
        nonce="c" * 64,
        launcher_pid=4321,
    )
    raw = consumer._canonical_json(owner) + b"\n"
    digest = _digest(raw)
    receipt = tmp_path.joinpath(
        *Path(consumer._external_lock_receipt_relative(run_id)).parts
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(raw)
    receipt.with_name("lock_owner.json.sha256").write_bytes(
        f"{digest}  lock_owner.json\n".encode("ascii")
    )
    lock_path = tmp_path.joinpath(
        *Path(str(config["gpu_policy"]["canonical_lock"])).parts
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(raw)
    return config, lock_path, receipt, owner, raw, digest


def test_execute_lock_refuses_all_missing_external_lease_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(consumer.load_config())
    config["gpu_policy"]["expected_gpu_uuid"] = "GPU-" + "9" * 32
    for variable in consumer.EXTERNAL_LOCK_ENV.values():
        monkeypatch.delenv(variable, raising=False)

    def unexpected_root() -> Path:
        raise AssertionError("missing external lease crossed the filesystem gate")

    monkeypatch.setattr(
        consumer.qdiag,
        "_project_root_from_module",
        unexpected_root,
    )
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_external_gpu_lock_required",
    ):
        with consumer._canonical_gpu_lock(config, run_id="missing-external-lock"):
            raise AssertionError("missing external lease was accepted")


def test_execute_lock_refuses_missing_dedicated_launcher_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(consumer.load_config())
    config["gpu_policy"]["expected_gpu_uuid"] = "GPU-" + "9" * 32
    monkeypatch.setenv(consumer.EXTERNAL_LOCK_ENV["receipt"], "unused")
    monkeypatch.setenv(consumer.EXTERNAL_LOCK_ENV["sha256"], "a" * 64)
    monkeypatch.setenv(consumer.EXTERNAL_LOCK_ENV["nonce"], "b" * 64)
    monkeypatch.delenv(
        consumer.EXTERNAL_LOCK_ENV["launcher_pid"],
        raising=False,
    )

    def unexpected_root() -> Path:
        raise AssertionError("missing launcher PID crossed the filesystem gate")

    monkeypatch.setattr(
        consumer.qdiag,
        "_project_root_from_module",
        unexpected_root,
    )
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_external_gpu_lock_identity_invalid",
    ):
        with consumer._canonical_gpu_lock(
            config,
            run_id="missing-launcher-pid",
        ):
            raise AssertionError("missing launcher PID was accepted")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0",
        "00",
        "-1",
        "+1",
        "1.0",
        "not-a-pid",
        " 1",
        "1 ",
        "4294967296",
    ],
)
def test_external_lock_launcher_pid_channel_is_strict(value: str) -> None:
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_external_gpu_lock_identity_invalid",
    ):
        consumer._validated_external_lock_launcher_pid(value)


def test_external_lock_rejects_an_ordinary_spoof_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, lock_path, receipt, owner, _raw, digest = _external_lock_fixture(
        tmp_path,
        monkeypatch,
        run_id="ordinary-file-rejected",
    )
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match=("chat_external_gpu_lock_(?:not_exclusive|requires_windows)"),
    ):
        consumer._load_external_lock_owner(
            config,
            run_id="ordinary-file-rejected",
            lock_path=lock_path,
            receipt_value=str(receipt),
            sha256_value=digest,
            nonce="c" * 64,
            launcher_pid_value=str(owner["launcher_pid"]),
        )


def test_external_lock_rejects_wrong_receipt_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, lock_path, receipt, owner, _raw, _digest_value = _external_lock_fixture(
        tmp_path,
        monkeypatch,
        run_id="wrong-receipt-digest",
    )
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_external_gpu_lock_receipt_digest_invalid",
    ):
        consumer._load_external_lock_owner(
            config,
            run_id="wrong-receipt-digest",
            lock_path=lock_path,
            receipt_value=str(receipt),
            sha256_value="d" * 64,
            nonce="c" * 64,
            launcher_pid_value=str(owner["launcher_pid"]),
        )


def test_external_lock_rejects_owner_field_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, lock_path, receipt, owner, _raw, _digest_value = _external_lock_fixture(
        tmp_path,
        monkeypatch,
        run_id="owner-field-drift",
    )
    owner["concurrency"] = 2
    raw = consumer._canonical_json(owner) + b"\n"
    digest = _digest(raw)
    receipt.write_bytes(raw)
    receipt.with_name("lock_owner.json.sha256").write_bytes(
        f"{digest}  lock_owner.json\n".encode("ascii")
    )
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_external_gpu_lock_owner_drift",
    ):
        consumer._load_external_lock_owner(
            config,
            run_id="owner-field-drift",
            lock_path=lock_path,
            receipt_value=str(receipt),
            sha256_value=digest,
            nonce="c" * 64,
            launcher_pid_value=str(owner["launcher_pid"]),
        )


def test_external_lock_rejects_launcher_pid_channel_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, lock_path, receipt, owner, _raw, digest = _external_lock_fixture(
        tmp_path,
        monkeypatch,
        run_id="launcher-pid-channel-drift",
    )
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_external_gpu_lock_owner_drift",
    ):
        consumer._load_external_lock_owner(
            config,
            run_id="launcher-pid-channel-drift",
            lock_path=lock_path,
            receipt_value=str(receipt),
            sha256_value=digest,
            nonce="c" * 64,
            launcher_pid_value=str(int(owner["launcher_pid"]) + 1),
        )


def test_external_lock_rejects_receipt_launcher_pid_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, lock_path, receipt, owner, _raw, _digest_value = _external_lock_fixture(
        tmp_path,
        monkeypatch,
        run_id="receipt-launcher-pid-mismatch",
    )
    expected_launcher_pid = int(owner["launcher_pid"])
    owner["launcher_pid"] = expected_launcher_pid + 1
    raw = consumer._canonical_json(owner) + b"\n"
    digest = _digest(raw)
    receipt.write_bytes(raw)
    receipt.with_name("lock_owner.json.sha256").write_bytes(
        f"{digest}  lock_owner.json\n".encode("ascii")
    )
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_external_gpu_lock_owner_drift",
    ):
        consumer._load_external_lock_owner(
            config,
            run_id="receipt-launcher-pid-mismatch",
            lock_path=lock_path,
            receipt_value=str(receipt),
            sha256_value=digest,
            nonce="c" * 64,
            launcher_pid_value=str(expected_launcher_pid),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows FileShare.None integration")
def test_external_lock_broker_reparent_fileshare_none_integration() -> None:
    import ctypes
    import msvcrt

    config = copy.deepcopy(consumer.load_config())
    expected_uuid = "GPU-" + "d" * 32
    config["gpu_policy"]["expected_gpu_uuid"] = expected_uuid
    run_id = f"lock-parent-child-{uuid.uuid4().hex}"
    nonce = "e" * 64
    owner = consumer._expected_external_lock_owner(
        config,
        run_id=run_id,
        nonce=nonce,
        launcher_pid=os.getpid(),
    )
    raw = consumer._canonical_json(owner) + b"\n"
    digest = _digest(raw)
    receipt = ROOT.joinpath(
        *Path(consumer._external_lock_receipt_relative(run_id)).parts
    )
    receipt_directory = receipt.parent
    lock_path = ROOT.joinpath(*Path(str(config["gpu_policy"]["canonical_lock"])).parts)
    conflicts = [
        ROOT.joinpath(*Path(str(relative)).parts)
        for relative in config["gpu_policy"]["conflicting_handoff_locks"]
    ]
    if lock_path.exists() or any(path.exists() for path in conflicts):
        pytest.skip("canonical or conflicting GPU lock is already active")
    receipt_directory.mkdir(parents=True, exist_ok=False)
    receipt.write_bytes(raw)
    receipt.with_name("lock_owner.json.sha256").write_bytes(
        f"{digest}  lock_owner.json\n".encode("ascii")
    )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(lock_path),
        0x80000000 | 0x40000000,
        0,
        None,
        1,
        0x00000080 | 0x80000000 | 0x04000000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        shutil.rmtree(receipt_directory)
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDWR)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
        child = (
            "import os;"
            "from anchor_mvp.training import "
            "gemma3_chat_five_expert_qonly_v1 as c;"
            "launcher_pid=int(os.environ["
            "'ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID']);"
            "assert os.getppid()!=launcher_pid,(os.getppid(),launcher_pid);"
            "cfg=c.load_config();"
            "ctx=c._canonical_gpu_lock(cfg,run_id=os.environ['LOCK_RUN_ID']);"
            "owner=ctx.__enter__();"
            "assert owner['launcher_pid']==launcher_pid;"
            "assert owner['nonce']==os.environ['ANCHOR_CHAT_EXTERNAL_LOCK_NONCE'];"
            "ctx.__exit__(None,None,None)"
        )
        broker = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
        if broker is None:
            pytest.skip("PowerShell broker is unavailable")
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": str(ROOT / "src"),
                "BROKER_CHILD_CODE": child,
                "BROKER_PYTHON": sys.executable,
                "ANCHOR_GEMMA_GPU_UUID": expected_uuid,
                "ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_RECEIPT": str(receipt),
                "ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_SHA256": digest,
                "ANCHOR_CHAT_EXTERNAL_LOCK_NONCE": nonce,
                "ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID": str(os.getpid()),
                "LOCK_RUN_ID": run_id,
            }
        )
        result = subprocess.run(
            [
                broker,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                ("& $env:BROKER_PYTHON -c $env:BROKER_CHILD_CODE; exit $LASTEXITCODE"),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    finally:
        os.close(descriptor)
        shutil.rmtree(receipt_directory)


class _ShapeParameter:
    def __init__(self, shape: tuple[int, ...], *, trainable: bool = True) -> None:
        self.shape = shape
        self.requires_grad = trainable

    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result


class _GradientModel:
    def __init__(
        self,
        *,
        dead: set[tuple[int, str]],
        first_step: bool,
    ) -> None:
        self.entries: list[tuple[str, torch.nn.Parameter]] = []
        for layer in range(26):
            for side in ("A", "B"):
                parameter = torch.nn.Parameter(torch.ones(1))
                nonzero = (layer, side) not in dead and (side == "B" or not first_step)
                parameter.grad = torch.ones(1) if nonzero else torch.zeros(1)
                self.entries.append(
                    (
                        "base_model.model.model.layers."
                        f"{layer}.self_attn.q_proj.lora_{side}.default.weight",
                        parameter,
                    )
                )

    def named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return self.entries


def test_gradient_gate_accepts_expected_first_and_second_step_coverage() -> None:
    first = consumer._assert_finite_and_gradient_scope(
        _GradientModel(dead=set(), first_step=True),
        torch,
        completed_steps=1,
    )
    second = consumer._assert_finite_and_gradient_scope(
        _GradientModel(dead=set(), first_step=False),
        torch,
        completed_steps=2,
    )
    assert first == {"A_nonzero": 0, "B_nonzero": 26}
    assert second == {"A_nonzero": 26, "B_nonzero": 26}


@pytest.mark.parametrize(
    ("dead", "completed_steps", "first_step"),
    [
        ({(5, "B")}, 1, True),
        ({(5, "A")}, 2, False),
        ({(2, "A"), (7, "A"), (12, "B")}, 2, False),
    ],
)
def test_gradient_gate_rejects_single_or_multiple_dead_layers(
    dead: set[tuple[int, str]],
    completed_steps: int,
    first_step: bool,
) -> None:
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_q_lora_gradient_coverage_failed",
    ):
        consumer._assert_finite_and_gradient_scope(
            _GradientModel(dead=dead, first_step=first_step),
            torch,
            completed_steps=completed_steps,
        )


class _ScopeModel:
    def __init__(self, *, include_escape: bool = False) -> None:
        entries: list[tuple[str, _ShapeParameter]] = []
        for layer in range(26):
            prefix = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
            entries.extend(
                [
                    (
                        f"{prefix}.lora_A.default.weight",
                        _ShapeParameter((1024, 1152)),
                    ),
                    (
                        f"{prefix}.lora_B.default.weight",
                        _ShapeParameter((1024, 1024)),
                    ),
                ]
            )
        if include_escape:
            entries.append(
                (
                    "base_model.model.model.layers.0.self_attn.o_proj.weight",
                    _ShapeParameter((1,)),
                )
            )
        self.entries = entries

    def named_parameters(self) -> list[tuple[str, _ShapeParameter]]:
        return list(self.entries)


def test_rank1024_q_only_scope_is_exactly_57933824() -> None:
    names, parameters = consumer._validate_trainable_scope(_ScopeModel())
    assert len(names) == 52
    assert parameters == 57_933_824
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_unexpected_trainable_tensor_outside_q_proj",
    ):
        consumer._validate_trainable_scope(_ScopeModel(include_escape=True))


class _OptimizerParameter:
    def __init__(self, elements: int) -> None:
        self.elements = elements

    def numel(self) -> int:
        return self.elements


class _FakeStateTensor:
    dtype = torch.uint8
    device = SimpleNamespace(type="cuda")

    def __init__(self, elements: int) -> None:
        self.elements = elements

    def numel(self) -> int:
        return self.elements


def test_adamw8bit_state_is_empirically_uint8_cuda() -> None:
    parameters = [
        item
        for layer in range(26)
        for item in (
            _OptimizerParameter(1024 * 1152),
            _OptimizerParameter(1024 * 1024),
        )
    ]
    optimizer = SimpleNamespace(
        state={
            parameter: {
                "state1": _FakeStateTensor(parameter.numel()),
                "state2": _FakeStateTensor(parameter.numel()),
            }
            for parameter in parameters
        }
    )
    report = consumer._validate_adamw8bit_state(
        optimizer,
        parameters,
        torch=torch,
        bitsandbytes_version="0.48.2",
    )
    assert report["state_dtype"] == "uint8"
    assert report["state_device"] == "cuda"
    assert report["state_tensors"] == 104
    assert report["state_elements"] == 2 * 57_933_824
    first = parameters[0]
    optimizer.state[first]["state1"].dtype = torch.float32
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_adamw8bit_state_not_uint8_cuda",
    ):
        consumer._validate_adamw8bit_state(
            optimizer,
            parameters,
            torch=torch,
            bitsandbytes_version="0.48.2",
        )


def test_warmup8_reaches_exact_two_e_minus_six() -> None:
    training = consumer.load_config()["training"]
    observed = [consumer._warmup_learning_rate(training, step) for step in range(1, 11)]
    assert observed[0] == pytest.approx(0.000002 / 8)
    assert observed[7] == pytest.approx(0.000002)
    assert observed[8:] == pytest.approx([0.000002, 0.000002])


def test_runtime_package_version_inventory_is_exact_and_bnb_locked() -> None:
    assert (
        consumer._validate_runtime_package_versions(FAKE_RUNTIME_PACKAGE_VERSIONS)
        == FAKE_RUNTIME_PACKAGE_VERSIONS
    )
    missing = dict(FAKE_RUNTIME_PACKAGE_VERSIONS)
    missing.pop("safetensors")
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_runtime_package_version_inventory_invalid",
    ):
        consumer._validate_runtime_package_versions(missing)
    drifted = dict(FAKE_RUNTIME_PACKAGE_VERSIONS)
    drifted["bitsandbytes"] = "0.49.0"
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_bitsandbytes_version_drift",
    ):
        consumer._validate_runtime_package_versions(drifted)


def _phase_receipt(role: str, phase: str) -> dict[str, object]:
    return {
        "schema_version": consumer.PHASE_RECEIPT_VERSION,
        "status": "passed_diagnostic_training_only",
        "run_id": "fake-serial-run",
        "role": role,
        "phase": phase,
        "optimizer_steps": 2 if phase == "smoke" else 160,
        "fresh_base": True,
        "fresh_adapter": True,
        "resume": False,
        "smoke_checkpoint_consumed": False,
        "initial_adapter_sha256": "a" * 64,
        "base_hash_before": "b" * 64,
        "q8_scb_sha256_before": "c" * 64,
        "runtime_package_versions": dict(FAKE_RUNTIME_PACKAGE_VERSIONS),
        "train_record_order_sha256": ("d" * 64 if phase == "smoke" else "e" * 64),
        "adapter_artifact_sha256": (
            None
            if phase == "smoke"
            else {
                "adapter_config.json": "f" * 64,
                "adapter_model.safetensors": "1" * 64,
            }
        ),
    }


def test_serial_orchestrator_calls_smoke_then_fresh_full_for_each_role(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_phase(
        _config: object,
        *,
        role: str,
        phase: str,
        steps: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append((role, phase, steps))
        return _phase_receipt(role, phase)

    datasets = {
        role: consumer.SerializedRoleDataset(role, (), ()) for role in consumer.ROLES
    }
    (tmp_path / "staging").mkdir()
    roles = consumer._run_serial_phases(
        consumer.load_config(),
        datasets=datasets,
        model_snapshot=tmp_path / "unused-model",
        staging=tmp_path / "staging",
        run_id="fake-serial-run",
        phase_executor=fake_phase,
    )
    assert calls == [
        item
        for role in consumer.ROLES
        for item in ((role, "smoke", 2), (role, "full", 160))
    ]
    assert [item["role"] for item in roles] == list(consumer.ROLES)


def test_pending_execute_blocks_before_dataset_lock_or_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _pending_config()

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("pending execute crossed the pre-dataset gate")

    monkeypatch.setattr(consumer, "_load_authenticated_dataset", unexpected)
    monkeypatch.setattr(consumer, "_canonical_gpu_lock", unexpected)
    monkeypatch.setattr(
        consumer,
        "_publish_failure_receipt",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_producer_identity_pending",
    ):
        consumer.execute(config, run_id="pending-execute-test")


def test_bound_python_execute_requires_external_fileshare_lease_before_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(consumer.load_config())
    config["identity_state"] = "authenticated_producer_handoff"
    config["producer"]["config_sha256"] = "a" * 64
    config["producer"]["implementation_sha256"] = "b" * 64
    config["dataset"]["record_schema_sha256"] = "c" * 64
    config["dataset"]["manifest_schema_sha256"] = "d" * 64
    config["dataset"]["manifest_sha256"] = "e" * 64
    config["claims"]["dataset_materialized"] = True
    config["claims"]["producer_identity_bound"] = True
    config["gpu_policy"]["expected_gpu_uuid"] = "GPU-" + "8" * 32
    consumer.validate_config(config)

    class _Snapshot:
        def __init__(self, sha256: str) -> None:
            self.sha256 = sha256

        def assert_unchanged(self) -> None:
            return None

    snapshots = {
        "implementation": _Snapshot("1" * 64),
        "runner_script": _Snapshot("2" * 64),
        "launcher": _Snapshot("3" * 64),
        "config": _Snapshot(config["_config_sha256"]),
        **{
            f"dependency:{relative}": _Snapshot(
                config["model"]["tokenizer_policy_sha256"]
                if relative == consumer.TOKENIZER_POLICY_PATH
                else "4" * 64
            )
            for relative in consumer.EXECUTION_DEPENDENCY_PATHS
        },
    }
    authenticated = consumer.AuthenticatedDataset(
        manifest_sha256="e" * 64,
        partition_sha256={
            "train/chat.jsonl": "5" * 64,
            "eval_proxy/chat.jsonl": "6" * 64,
        },
        records=(),
    )
    monkeypatch.setattr(
        consumer,
        "_load_authenticated_dataset",
        lambda _config: authenticated,
    )
    monkeypatch.setattr(consumer, "_code_snapshots", lambda: snapshots)
    monkeypatch.setattr(
        consumer,
        "_output_path",
        lambda value: (
            tmp_path / "artifacts" if "artifacts/" in str(value) else tmp_path / "runs"
        ),
    )

    def unexpected_model(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("missing external lease reached model snapshot")

    monkeypatch.setattr(consumer, "_private_model_snapshot", unexpected_model)
    for variable in consumer.EXTERNAL_LOCK_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="chat_external_gpu_lock_required",
    ):
        consumer.execute(config, run_id="bound-missing-external-lock")
    assert not list((tmp_path / "artifacts").glob(".*.tmp-*"))


@pytest.mark.parametrize("boundary", ["allocated", "reserved", "physical"])
def test_chat_peak_gate_is_strictly_below_10gib(boundary: str) -> None:
    config = consumer.load_config()
    allocated = 10240 * consumer._MIB - 1
    reserved = 10240 * consumer._MIB - 1
    physical = 10239
    if boundary == "allocated":
        allocated += 1
    elif boundary == "reserved":
        reserved += 1
    else:
        physical += 1
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="strict_memory_gate_exceeded",
    ):
        consumer._strict_memory_gate(
            config,
            peak_allocated_bytes=allocated,
            peak_reserved_bytes=reserved,
            monitor_samples=[{"memory_used_mib": physical}],
        )


class _EffectContext:
    def __init__(self, model: "_EffectModel") -> None:
        self.model = model

    def __enter__(self) -> None:
        self.model.enabled = False

    def __exit__(self, *_args: object) -> None:
        self.model.enabled = True


class _EffectModel:
    def __init__(self) -> None:
        self.enabled = True

    def eval(self) -> None:
        return None

    def disable_adapter(self) -> _EffectContext:
        return _EffectContext(self)

    def __call__(self, **_kwargs: object) -> SimpleNamespace:
        logits = torch.tensor([[[1.0, 2.0, 3.0]]])
        if self.enabled:
            logits = logits + torch.tensor([[[0.0, 0.25, 0.0]]])
        return SimpleNamespace(logits=logits)


def test_adapter_effect_gate_requires_enabled_disabled_delta() -> None:
    batch = {
        "input_ids": torch.tensor([[2, 3, 4]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    report = consumer._enabled_vs_disabled_next_token_effect(
        _EffectModel(),
        batch,
        2,
        torch=torch,
    )
    assert report["finite"] is True
    assert report["max_abs"] == pytest.approx(0.25)


def test_fully_bound_fake_execute_publishes_independent_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(consumer.load_config())
    config["identity_state"] = "authenticated_producer_handoff"
    config["producer"]["config_sha256"] = "a" * 64
    config["producer"]["implementation_sha256"] = "b" * 64
    config["dataset"]["record_schema_sha256"] = "c" * 64
    config["dataset"]["manifest_schema_sha256"] = "d" * 64
    config["dataset"]["manifest_sha256"] = "e" * 64
    config["claims"]["dataset_materialized"] = True
    config["claims"]["producer_identity_bound"] = True
    consumer.validate_config(config)

    authenticated = consumer.AuthenticatedDataset(
        manifest_sha256="e" * 64,
        partition_sha256={
            "train/chat.jsonl": "1" * 64,
            "eval_proxy/chat.jsonl": "2" * 64,
        },
        records=(),
    )

    class _Snapshot:
        def __init__(self, sha256: str) -> None:
            self.sha256 = sha256

        def assert_unchanged(self) -> None:
            return None

    snapshots = {
        "implementation": _Snapshot("3" * 64),
        "runner_script": _Snapshot("4" * 64),
        "launcher": _Snapshot("5" * 64),
        "config": _Snapshot(config["_config_sha256"]),
        **{
            f"dependency:{relative}": _Snapshot(
                config["model"]["tokenizer_policy_sha256"]
                if relative == consumer.TOKENIZER_POLICY_PATH
                else "6" * 64
            )
            for relative in consumer.EXECUTION_DEPENDENCY_PATHS
        },
    }

    @contextmanager
    def fake_lock(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        yield {
            "schema_version": consumer.LOCK_OWNER_VERSION,
            "content_sha256": "6" * 64,
        }

    @contextmanager
    def fake_model_snapshot(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        yield tmp_path / "fake-model", {name: "7" * 64 for name in consumer.MODEL_FILES}

    fake_datasets = {
        role: consumer.SerializedRoleDataset(role, (), ()) for role in consumer.ROLES
    }
    fake_roles = [
        {
            "role": role,
            "smoke_receipt_sha256": "8" * 64,
            "full_receipt_sha256": "9" * 64,
            "adapter_artifact_sha256": {
                "adapter_config.json": "a" * 64,
                "adapter_model.safetensors": "b" * 64,
            },
        }
        for role in consumer.ROLES
    ]
    monkeypatch.setattr(
        consumer,
        "_load_authenticated_dataset",
        lambda _config: authenticated,
    )
    monkeypatch.setattr(consumer, "_code_snapshots", lambda: snapshots)
    monkeypatch.setattr(consumer, "_canonical_gpu_lock", fake_lock)
    monkeypatch.setattr(
        consumer,
        "_private_model_snapshot",
        fake_model_snapshot,
    )
    monkeypatch.setattr(
        consumer.gemma_binding,
        "load_sentencepiece",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        consumer,
        "_serialize_all_role_datasets",
        lambda _processor, _authenticated: fake_datasets,
    )
    monkeypatch.setattr(
        consumer,
        "_run_serial_phases",
        lambda *_args, **_kwargs: fake_roles,
    )
    monkeypatch.setattr(
        consumer,
        "_capture_training_artifacts",
        lambda *_args, **_kwargs: ({}, dict(FAKE_RUNTIME_PACKAGE_VERSIONS)),
    )
    monkeypatch.setattr(
        consumer,
        "_remove_training_progress_for_publish",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        consumer,
        "_assert_success_root_inventory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        consumer,
        "_output_path",
        lambda value: (
            tmp_path / "artifacts" if "artifacts/" in str(value) else tmp_path / "runs"
        ),
    )
    result = consumer.execute(config, run_id="fake-bound-execute")
    assert result["schema_version"] == consumer.RUN_RECEIPT_VERSION
    assert result["status"] == "passed_diagnostic_training_only"
    assert result["identity"]["dataset_manifest_sha256"] == "e" * 64
    assert result["identity"]["partition_sha256"] == {
        "train/chat.jsonl": "1" * 64,
        "eval_proxy/chat.jsonl": "2" * 64,
    }
    assert result["identity"]["execution_dependency_sha256"] == {
        relative: (
            config["model"]["tokenizer_policy_sha256"]
            if relative == consumer.TOKENIZER_POLICY_PATH
            else "6" * 64
        )
        for relative in consumer.EXECUTION_DEPENDENCY_PATHS
    }
    assert (
        result["execution"]["runtime_package_versions"] == FAKE_RUNTIME_PACKAGE_VERSIONS
    )
    receipt = tmp_path / "artifacts/fake-bound-execute/run_receipt.json"
    assert receipt.is_file()
    assert receipt.with_name("run_receipt.json.sha256").is_file()
    assert not list((tmp_path / "artifacts").glob(".*.tmp-*"))

    capture_calls = 0

    def fail_after_publish(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, str]]:
        nonlocal capture_calls
        capture_calls += 1
        if capture_calls == 2:
            raise consumer.ChatFiveExpertError(
                "synthetic_post_publish_artifact_rewrite"
            )
        return {}, dict(FAKE_RUNTIME_PACKAGE_VERSIONS)

    monkeypatch.setattr(
        consumer,
        "_capture_training_artifacts",
        fail_after_publish,
    )
    with pytest.raises(
        consumer.ChatFiveExpertError,
        match="synthetic_post_publish_artifact_rewrite",
    ):
        consumer.execute(config, run_id="fake-post-publish-drift")
    assert not (tmp_path / "artifacts/fake-post-publish-drift").exists()
    assert (
        len(list((tmp_path / "artifacts").glob(".failed-fake-post-publish-drift-*")))
        == 1
    )
