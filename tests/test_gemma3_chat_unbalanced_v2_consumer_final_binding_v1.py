from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest

from anchor_mvp.training import gemma3_chat_unbalanced_v2_consumer_v1 as consumer
from anchor_mvp.training import (
    gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1 as multiarm,
)


ROOT = Path(__file__).resolve().parents[1]
BINDING = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_consumer_binding_sharded_v1.json"
)
RECEIPT = (
    ROOT
    / "fixtures"
    / "research"
    / "gemma3_chat_unbalanced_v2_consumer_preflight_sharded_v1"
    / "receipt.json"
)
RECEIPT_SIDECAR = RECEIPT.with_name("receipt.json.sha256")
SCHEMA = (
    ROOT
    / "configs"
    / "research"
    / "gemma3_chat_unbalanced_v2_consumer_binding_v1.schema.json"
)

BINDING_SHA256 = "9f18c4106d9ed74ecfa132e74628642ce76f9111125a4b3fdd18e0d003dc57c0"
RECEIPT_SHA256 = "a976754d84f48c01b9ff509c48f7fb7b26c32eabfa0b10d6f221661114a1cdb2"
RECEIPT_SIDECAR_SHA256 = (
    "aebd6c53788214c8fcf7783f64a60f4ffc97343c040d5422e7ca79a8fd7afbc3"
)
SOURCE_BINDING_SHA256 = (
    "012f78dafe0081c722fd978c612ecd6c845c8daa62fc289cf0991fc6818e7d6f"
)
DERIVED_ASSETS_SHA256 = (
    "85badacb6e96ab6d384da587b01f11a7f486d6b54015497af7a096bf9b2b5996"
)

PRODUCER_GIT = {
    "candidate_commit": "c5080249aa103d6cac09f0a55e51982b68524480",
    "release_commit": "304be2b86f82aad7ddade80fae04528bf2801977",
    "release_tree": "9bf154d481623b696dbc8a4bdf276967710b7214",
    "upstream_commit": "304be2b86f82aad7ddade80fae04528bf2801977",
    "live_remote_commit": "304be2b86f82aad7ddade80fae04528bf2801977",
    "clean_worktree": True,
    "tags_at_head": 0,
}
LOGICAL_IDENTITY = {
    "logical_dataset_sha256": (
        "c63fd9c46aff129e0840ed32482ed1a856a66a2b89b48b1316d54bf378685033"
    ),
    "logical_train_partition_sha256": (
        "62c0f64b6d2f5a0901169480c946d554cc02175138862b3a210d06e44fcbddd3"
    ),
    "eval_proxy_partition_sha256": (
        "42798778a2c1398393bb0b6588cfd5d1cf5f3fc67ae008e9e20aea0e33bae176"
    ),
    "logical_record_order_sha256": (
        "6419cdac69cb3da05a49eac94a58ab73a47ae11ae65445f29de7b518aceda76c"
    ),
    "logical_target_inventory_sha256": (
        "92372de0b2837b05fbe670fc677b9eee08cf751a379e3fe65d1aee4804493c46"
    ),
    "logical_train_record_inventory_sha256": (
        "cf395044642a3333d31f47926b590011f0cb7922631af204ea7c16b5a46fe8aa"
    ),
    "logical_all_record_inventory_sha256": (
        "5da71c978eacdbb4dbcc466b6a227ede7ddb6b0191cc70164f5aa91747444515"
    ),
    "logical_task_bundle_inventory_sha256": (
        "137f2e37c75ae2ce248439308c66d0c2e80f92fc75600d32acb337f11d713334"
    ),
    "serialization_inventory_sha256": (
        "4b7a35067ab4a0acdbda3095055f20a21b8c24b64f12a70d07aaa9a48afa0f1d"
    ),
    "identity_probe_inventory_sha256": (
        "5740338f0dd5800cbcb76b7ef0671fd26fd4f096b5fea05a21f11eb3e99e56c5"
    ),
}
ASSET_RECORDS = {
    "humor": 240,
    "serious": 240,
    "angry_style": 240,
    "review_audit": 1360,
    "tool_call": 1360,
    "planner_router": 80,
    "planner_router_eval": 20,
    "tool_comparison_eval": 400,
    "planner_comparison_eval": 240,
    "identity_probe": 50,
}
ASSET_NAMESPACES = {
    "humor": "train/chat#role=humor",
    "serious": "train/chat#role=serious",
    "angry_style": "train/chat#role=angry_style",
    "review_audit": "train/chat#role=review_audit",
    "tool_call": "train/chat#role=tool_call",
    "planner_router": "components/router#split=train",
    "planner_router_eval": "components/router#split=eval_proxy",
    "tool_comparison_eval": "components/tool_eval#split=all",
    "planner_comparison_eval": "components/planner_eval#split=all",
    "identity_probe": "identity_eval/probe_inventory#split=all",
}
TRAINING_NAMES = frozenset(
    {
        "humor",
        "serious",
        "angry_style",
        "review_audit",
        "tool_call",
        "planner_router",
    }
)
MAIN_CHAT_NAMES = frozenset(
    {"humor", "serious", "angry_style", "review_audit", "tool_call"}
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _domain_sha256(domain: str, value: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + _canonical_json(value)
    ).hexdigest()


def _independent_assets(
    binding: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "identity_derivation_schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-consumer-derived-asset-identity.v1"
        ),
        "physical_receipt_sha256": RECEIPT_SHA256,
        "artifact_version": receipt["artifact_version"],
        "tree_digest_domain": receipt["tree_digest_domain"],
        "tree_digest_sha256": receipt["tree_digest_sha256"],
        "logical_identity": receipt["logical_identity"],
    }
    shard_schema = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2-consumer-derived-shard-inventory.v1"
        ),
        "identity_origin": ("consumer_derived_from_authenticated_sharded_v1_receipt"),
        "fields": [
            "physical_receipt_sha256",
            "artifact_version",
            "tree_digest_sha256",
            "logical_identity",
            "asset_name",
            "asset_namespace",
            "records",
            "training_eligible",
        ],
    }
    shard_schema_sha = _domain_sha256(
        "anchor.consumer-derived-shard-inventory-schema.v1",
        shard_schema,
    )
    main_chat_shard_sha = _domain_sha256(
        "anchor.consumer-derived-main-chat-shard-inventory.v1",
        {
            **common,
            "training_shards": receipt["training_shards"],
            "physical_training_shard_anchors": binding["training_shards"]["shards"],
        },
    )

    def derive(name: str) -> dict[str, Any]:
        training_eligible = name in TRAINING_NAMES
        preimage = {
            **common,
            "asset_name": name,
            "asset_namespace": ASSET_NAMESPACES[name],
            "records": ASSET_RECORDS[name],
            "training_eligible": training_eligible,
        }
        asset = {
            "records": ASSET_RECORDS[name],
            "asset_namespace": ASSET_NAMESPACES[name],
            "identity_origin": (
                "consumer_derived_from_authenticated_sharded_v1_receipt"
            ),
            "identity_derivation_schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-consumer-derived-asset-identity.v1"
            ),
            "identity_preimage_sha256": _domain_sha256(
                "anchor.consumer-derived-asset-preimage.v1",
                preimage,
            ),
            "logical_dataset_sha256": _domain_sha256(
                "anchor.consumer-derived-logical-dataset.v1",
                preimage,
            ),
            "shard_inventory_schema_version": (
                "anchor.gemma3-chat-unbalanced-v2-consumer-derived-shard-inventory.v1"
            ),
            "shard_inventory_schema_sha256": shard_schema_sha,
            "shard_inventory_sha256": (
                main_chat_shard_sha
                if name in MAIN_CHAT_NAMES
                else _domain_sha256(
                    "anchor.consumer-derived-tree-asset-shard-inventory.v1",
                    preimage,
                )
            ),
            "shard_count": 2 if name in MAIN_CHAT_NAMES else 1,
            "training_eligible": training_eligible,
            "record_order_sha256": _domain_sha256(
                "anchor.consumer-derived-record-order.v1",
                preimage,
            ),
            "target_projection_sha256": _domain_sha256(
                "anchor.consumer-derived-target-projection.v1",
                preimage,
            ),
        }
        if training_eligible:
            asset["shard_order_contract"] = (
                "continuous_bundle_boundary_v1"
                if name in MAIN_CHAT_NAMES
                else "consumer_derived_tree_asset_namespace_v1"
            )
        return asset

    training = {name: derive(name) for name in sorted(TRAINING_NAMES)}
    evaluation = {
        name: derive(name) for name in sorted(set(ASSET_RECORDS) - TRAINING_NAMES)
    }
    return training, evaluation


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _all_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def test_final_binding_and_receipt_are_body_free_and_exact() -> None:
    assert _file_sha256(BINDING) == BINDING_SHA256
    assert _file_sha256(RECEIPT) == RECEIPT_SHA256
    assert _file_sha256(RECEIPT_SIDECAR) == RECEIPT_SIDECAR_SHA256
    assert RECEIPT_SIDECAR.read_bytes() == (
        f"{RECEIPT_SHA256}  receipt.json\n".encode("ascii")
    )

    schema = _load_json(SCHEMA)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    binding = _load_json(BINDING)
    receipt = _load_json(RECEIPT)
    validator.validate(binding)
    validator.validate(receipt)
    assert consumer.validate_binding_document(binding) == binding

    assert binding["schema_version"] == consumer.SHARDED_BINDING_VERSION
    assert receipt["schema_version"] == consumer.SHARDED_RECEIPT_VERSION
    assert binding["producer_git"] == PRODUCER_GIT
    assert receipt["producer_git"] == PRODUCER_GIT
    assert binding["tree_digest_domain"] == consumer.PRODUCER_TREE_DIGEST_DOMAIN
    assert receipt["tree_digest_domain"] == consumer.PRODUCER_TREE_DIGEST_DOMAIN
    assert binding["tree_digest_sha256"] == (
        "25add529328255f6b36f9929be1851f42abd25deda7a7f69733ac8348cad387c"
    )
    assert (
        receipt["binding_contract_sha256"]
        == hashlib.sha256(consumer._canonical_json(binding)).hexdigest()
    )
    assert len(binding["payload_files"]) == 23
    assert binding["read_set"]["count"] == len(binding["read_set"]["files"]) == 88
    assert receipt["file_counts"] == {"payload": 23, "sidecar": 23, "total": 46}
    assert (
        binding["logical_identity"] == receipt["logical_identity"] == LOGICAL_IDENTITY
    )
    assert binding["training_shards"]["ordered_concat_sha256"] == (
        "62c0f64b6d2f5a0901169480c946d554cc02175138862b3a210d06e44fcbddd3"
    )
    assert binding["training_shards"]["boundary_commitment_sha256"] == (
        "6af9bc4cd79c9a8bb211a7147b91eb7c0c07e3adf7026becc1c913045716d089"
    )
    assert [
        (item["records"], item["task_bundles"])
        for item in binding["training_shards"]["shards"]
    ] == [(1770, 564), (1670, 796)]
    assert receipt["release"]["artifact_final_identity"] is True
    assert receipt["release"]["producer_training_authorized"] is False
    assert receipt["claims"]["sample_bodies_parsed"] is False
    assert receipt["claims"]["raw_token_ids_parsed"] is False

    forbidden_body_keys = {
        "messages",
        "prompt",
        "completion",
        "answer",
        "target",
        "input_ids",
        "token_ids",
        "gold",
        "heldout",
    }
    assert not (_all_keys(binding) | _all_keys(receipt)) & forbidden_body_keys
    text = BINDING.read_text("utf-8") + RECEIPT.read_text("utf-8")
    lowered = text.lower()
    assert "\r" not in text
    assert all(marker not in lowered for marker in ("ark-", "api_key", "d:\\", "c:\\"))


def test_multiarm_loader_independently_derives_exact_ten_assets() -> None:
    binding = _load_json(BINDING)
    receipt = _load_json(RECEIPT)
    normalized, receipt_sha, source_sha = multiarm.load_consumer_preflight_receipt(
        RECEIPT
    )
    expected_training, expected_evaluation = _independent_assets(binding, receipt)
    assert receipt_sha == RECEIPT_SHA256
    assert source_sha == SOURCE_BINDING_SHA256
    assert normalized["training_assets"] == expected_training
    assert normalized["evaluation_assets"] == expected_evaluation
    assert len(normalized["training_assets"]) == 6
    assert len(normalized["evaluation_assets"]) == 4
    assert set(normalized["training_assets"]) | set(
        normalized["evaluation_assets"]
    ) == set(ASSET_RECORDS)
    assert (
        _canonical_sha256(
            {
                "training": normalized["training_assets"],
                "evaluation": normalized["evaluation_assets"],
            }
        )
        == DERIVED_ASSETS_SHA256
    )


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "old_receipt_v1",
            lambda value: value.update(
                {
                    "schema_version": (
                        "anchor.gemma3-chat-unbalanced-v2-consumer-preflight-receipt.v1"
                    )
                }
            ),
        ),
        (
            "old_44_file_shape",
            lambda value: value.update(
                {"file_counts": {"payload": 22, "sidecar": 22, "total": 44}}
            ),
        ),
        (
            "superseded_sharded_v2",
            lambda value: value.update(
                {
                    "artifact_version": (
                        "anchor.gemma3-chat-five-expert-qonly-unbalanced-v2-sharded.v2"
                    )
                }
            ),
        ),
    ],
)
def test_multiarm_rejects_superseded_consumer_identities(
    label: str,
    mutate: Any,
) -> None:
    del label
    receipt = deepcopy(_load_json(RECEIPT))
    mutate(receipt)
    with pytest.raises(multiarm.MultiArmContractError):
        multiarm._validate_source_receipt(receipt)
