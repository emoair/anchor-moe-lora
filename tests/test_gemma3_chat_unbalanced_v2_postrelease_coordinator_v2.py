from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from jsonschema.exceptions import ValidationError

from anchor_mvp.research import (
    gemma3_chat_unbalanced_v2_postrelease_coordinator_v2 as coordinator,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _receipt() -> dict[str, object]:
    config, config_sha = coordinator.load_config()
    return coordinator.build_validate_only_receipt(
        config,
        config_sha256=config_sha,
        binding_sha256=SHA_A,
    )


def test_validate_only_freezes_body_free_order_and_binding() -> None:
    receipt = _receipt()

    assert receipt["status"] == "validated_stub_no_execution"
    assert receipt["binding"]["producer_commit"] == (
        "878e27dbc41e8dafc43b6462279a814f53d232b5"
    )
    assert receipt["binding"]["producer_tree"] == (
        "566e84d30bc934def4a52b48f3fd65b2639665ae"
    )
    assert receipt["next_stage"] == "smoke"
    assert [item["name"] for item in receipt["stages"]] == list(coordinator.STAGE_ORDER)
    assert {item["binding_sha256"] for item in receipt["stages"]} == {SHA_A}
    assert receipt["claims"] == {
        "body_free": True,
        "execution_performed": False,
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "sample_bodies_read": False,
        "raw_token_ids_read": False,
        "secret_material_read": False,
        "fallback_used": False,
        "formal": False,
    }


def test_binding_sha_is_the_only_idempotency_identity() -> None:
    config, _ = coordinator.load_config()
    first = coordinator.binding_idempotency_key(config, SHA_A)
    assert coordinator.binding_idempotency_key(config, SHA_A) == first
    assert coordinator.binding_idempotency_key(config, SHA_B) != first


@pytest.mark.parametrize("value", ["", "A" * 64, "a" * 63, "not-a-sha"])
def test_invalid_binding_sha_fails_closed(value: str) -> None:
    config, _ = coordinator.load_config()
    with pytest.raises(
        coordinator.PostreleaseCoordinatorError,
        match="postrelease_binding_sha256_invalid",
    ):
        coordinator.binding_idempotency_key(config, value)


def test_stage_reordering_is_rejected_before_execution() -> None:
    config, config_sha = coordinator.load_config()
    drifted = copy.deepcopy(config)
    drifted["stages"][0], drifted["stages"][1] = (
        drifted["stages"][1],
        drifted["stages"][0],
    )
    with pytest.raises(coordinator.PostreleaseCoordinatorError):
        coordinator.build_validate_only_receipt(
            drifted,
            config_sha256=config_sha,
            binding_sha256=SHA_A,
        )


def test_non_validate_operation_remains_unimplemented() -> None:
    assert coordinator.main(["--binding-sha256", SHA_A]) == 2


@pytest.mark.parametrize("isolated", [False, True], ids=["ordinary", "python-I"])
def test_repository_runner_bootstraps_src_from_any_cwd(
    tmp_path: Path,
    isolated: bool,
) -> None:
    runner = (
        coordinator.ROOT / "scripts/research/"
        "run_gemma3_chat_unbalanced_v2_postrelease_coordinator_v2.py"
    )
    command = [sys.executable]
    if isolated:
        command.append("-I")
    command.extend(
        [
            str(runner),
            "--validate-only",
            "--binding-sha256",
            SHA_A,
        ]
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(result.stdout)
    assert value["status"] == "validated_stub_no_execution"
    assert value["binding"]["sha256"] == SHA_A


def test_aggregate_schema_rejects_duplicate_stage() -> None:
    receipt = _receipt()
    receipt["stages"][1] = copy.deepcopy(receipt["stages"][0])
    validator = coordinator._schema(
        coordinator.ROOT / coordinator.RECEIPT_SCHEMA_PATH,
        code="test_receipt_schema",
    )
    with pytest.raises(ValidationError):
        validator.validate(receipt)


def test_aggregate_schema_rejects_reverse_stage_order() -> None:
    receipt = _receipt()
    receipt["stages"] = list(reversed(receipt["stages"]))
    validator = coordinator._schema(
        coordinator.ROOT / coordinator.RECEIPT_SCHEMA_PATH,
        code="test_receipt_schema",
    )
    with pytest.raises(ValidationError):
        validator.validate(receipt)
