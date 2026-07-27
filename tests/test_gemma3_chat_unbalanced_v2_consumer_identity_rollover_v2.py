from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2 as rollover,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_rollover_contract_and_three_teacher_consumers_are_6240() -> None:
    contract = _json(rollover.CONFIG_PATH)
    schema = _json(rollover.SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)
    assert contract["repository"]["commit"] == (
        "6240ae111182104f22f08e1a569deae866c6e210"
    )
    assert len(contract["exact_files"]) == 17
    assert {
        contract["teacher_chain"]["finalizer"]["commit"],
        contract["teacher_chain"]["final_manifest"]["commit"],
        contract["teacher_chain"]["external_binding"]["consumer_commit"],
    } == {"6240ae111182104f22f08e1a569deae866c6e210"}


def test_additive_finalizer_manifest_and_binding_chain_is_6240() -> None:
    finalizer = _json(
        batch.REPO_ROOT / "configs/data/"
        "gemma3_chat_unbalanced_v2_teacher_finalizer_consumer_rollover_v2.json"
    )
    manifest_schema = _json(
        batch.REPO_ROOT / "configs/data/"
        "gemma3_chat_unbalanced_v2_teacher_final_manifest_"
        "consumer_rollover_v2.schema.json"
    )
    binding_schema = _json(
        batch.REPO_ROOT / "configs/data/"
        "gemma3_chat_unbalanced_v2_teacher_alignment_binding_"
        "consumer_rollover_v2.schema.json"
    )
    commit = "6240ae111182104f22f08e1a569deae866c6e210"
    assert finalizer["consumer"]["commit"] == commit
    assert manifest_schema["properties"]["consumer"]["const"]["commit"] == commit
    assert (
        binding_schema["properties"]["consumer_prerequisite"]["const"][
            "consumer_commit"
        ]
        == commit
    )


def test_old_chain_files_remain_git_identical() -> None:
    expected = {
        "configs/data/gemma3_chat_unbalanced_v2_teacher_finalizer_v1.json": (
            "cff032da9c68af660aad3369a4a46ee694e691e14456ae3ce29179f9f9f4bf9b"
        ),
        "configs/data/gemma3_chat_unbalanced_v2_teacher_final_manifest_v2.schema.json": (
            "3c2cdb0ac404b74797425448de05a8d536d922d57671ab79eadcd730b04d99cb"
        ),
        "configs/data/gemma3_chat_unbalanced_v2_teacher_alignment_binding_v2.schema.json": (
            "41020aa6103a52f13919d164be679da110b6260b2ba27ad1c866bcb305dc3a87"
        ),
    }
    for relative, digest in expected.items():
        raw = (batch.REPO_ROOT / relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest


def test_real_model_free_preflight_has_only_memory_secret_blockers() -> None:
    result = rollover.preflight(validate_source=True)
    assert result["blockers"] == [
        "controller_credential_slot_unloaded",
        "runtime_hmac_slot_unloaded",
    ]
    assert result["consumer_commit"] == ("6240ae111182104f22f08e1a569deae866c6e210")
    assert result["consumer_exact_files"] == 17
    assert result["teacher_chain_consumer_consistent"] is True
    assert result["network_requests"] == 0
    assert result["model_requests"] == 0
    assert result["gpu_requests"] == 0


def test_git_executable_is_authenticated_without_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "")
    rollover.validate_consumer_identity_rollover()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "C:/Windows/System32/cmd.exe"),
        ("sha256", "0" * 64),
    ],
)
def test_git_executable_substitution_is_schema_rejected(
    field: str,
    value: str,
) -> None:
    contract = _json(rollover.CONFIG_PATH)
    schema = _json(rollover.SCHEMA_PATH)
    mutated = copy.deepcopy(contract)
    mutated["repository"]["git_executable"][field] = value
    with pytest.raises(batch.AdapterError, match="contract schema rejected"):
        rollover._validate(
            schema,
            mutated,
            reason="consumer_rollover_contract_schema_rejected",
        )


def test_replace_refs_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    config, _, _ = rollover._load_contract()
    repo = rollover._repo_path(config)
    executable = rollover._authenticated_git_executable(config)
    real_git = rollover._git

    def replaced(
        executable_path: Path,
        repository: Path,
        *arguments: str,
        text: bool = True,
    ) -> str | bytes:
        if arguments[-1:] == ("refs/replace",):
            return "refs/replace/forbidden\n"
        return real_git(
            executable_path,
            repository,
            *arguments,
            text=text,
        )

    monkeypatch.setattr(rollover, "_git", replaced)
    with pytest.raises(batch.AdapterError, match="git rewrite forbidden"):
        rollover._exact_git_identity(executable, repo, config)


def test_old_60e_consumer_commit_is_execution_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_raw, schema_raw = rollover._load_contract()
    mutated = copy.deepcopy(config)
    mutated["repository"]["commit"] = "60e88aa7a6c2dcdd8a3cd9f496f84d3a1ea76bb2"
    monkeypatch.setattr(
        rollover,
        "_load_contract",
        lambda: (mutated, config_raw, schema_raw),
    )
    with pytest.raises(batch.AdapterError) as captured:
        rollover.validate_consumer_identity_rollover()
    assert captured.value.reason_code == "consumer_rollover_git_identity_drift"


def test_consumer_head_drift_is_execution_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_git = rollover._git

    def drifted_head(
        executable: Path,
        repository: Path,
        *arguments: str,
        text: bool = True,
    ) -> str | bytes:
        if arguments == ("rev-parse", "HEAD"):
            return f"{'0' * 40}\n"
        return real_git(executable, repository, *arguments, text=text)

    monkeypatch.setattr(rollover, "_git", drifted_head)
    with pytest.raises(batch.AdapterError) as captured:
        rollover.validate_consumer_identity_rollover()
    assert captured.value.reason_code == "consumer_rollover_git_identity_drift"


def test_consumer_upstream_drift_is_execution_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_git = rollover._git

    def drifted_upstream(
        executable: Path,
        repository: Path,
        *arguments: str,
        text: bool = True,
    ) -> str | bytes:
        if arguments == (
            "rev-parse",
            "refs/remotes/origin/research/gemma3-chat-unbalanced-v2-training",
        ):
            return f"{'f' * 40}\n"
        return real_git(executable, repository, *arguments, text=text)

    monkeypatch.setattr(rollover, "_git", drifted_upstream)
    with pytest.raises(batch.AdapterError) as captured:
        rollover.validate_consumer_identity_rollover()
    assert captured.value.reason_code == "consumer_rollover_git_identity_drift"


def test_exact17_physical_blob_drift_is_execution_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_snapshot = rollover._snapshot

    def drifted_snapshot(path: Path, *, reason: str) -> bytes:
        raw = real_snapshot(path, reason=reason)
        if path.name == ".gitattributes":
            return raw + b"x"
        return raw

    monkeypatch.setattr(rollover, "_snapshot", drifted_snapshot)
    with pytest.raises(batch.AdapterError) as captured:
        rollover.validate_consumer_identity_rollover()
    assert captured.value.reason_code == "consumer_rollover_git_blob_worktree_drift"
