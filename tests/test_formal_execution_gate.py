from __future__ import annotations

import copy
from pathlib import Path

import pytest

from anchor_mvp.training import formal_execution_gate as gate
from anchor_mvp.training import runtime


def _decision(
    *,
    schema_version: str = gate.FORMAL_DECISION_SCHEMA,
    status: str = "blocked_formal_authorization_inputs_unavailable",
    training_authorized: bool = False,
    formal_training_authorized: bool = False,
    formal: bool = False,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": schema_version,
        "status": status,
        "training_authorized": training_authorized,
        "formal_training_authorized": formal_training_authorized,
        "formal": formal,
        "authenticated_inputs": {"formal_v3": "complete"},
    }
    value["decision_sha256"] = gate._canonical_sha256(value)
    return value


def test_formal_identity_and_protected_paths_are_recognized() -> None:
    assert gate.is_formal_v3_config({"experiment": "anchor-moe-lora-formal-v3-C"})
    assert gate.is_formal_v3_config(
        {"paths": {"adapter_dir": "artifacts/formal_v3/C/adapters"}}
    )
    assert gate.is_formal_v3_config(
        {"paths": {"adapter_dir": "ARTIFACTS/FORMAL_V3/C/adapters"}}
    )
    assert gate.is_formal_v3_config(
        {"paths": {"adapter_dir": "artifacts/formal_v3./C/adapters"}}
    )
    assert gate.is_formal_v3_config(
        {"paths": {"adapter_dir": "artifacts/formal_v3 /C/adapters"}}
    )
    assert gate.is_formal_v3_config(
        {"paths": {"adapter_dir": "artifacts/formal_v3:stream"}}
    )
    assert gate.is_formal_v3_config(
        {
            "scale_gate": {
                "dataset_snapshot": {
                    "manifest": "artifacts/formal_v3/dataset/manifest.json"
                }
            }
        }
    )
    assert gate.is_formal_v3_config(
        {
            "experiment": "renamed-diagnostic",
            "active_adapter": {
                "datasets": ["artifacts/formal_v3/dataset/planner.jsonl"]
            },
        }
    )
    assert not gate.is_formal_v3_config(
        {
            "experiment": "anchor-moe-lora-diagnostic",
            "paths": {"adapter_dir": "artifacts/diagnostic/adapters"},
        }
    )


def test_nonformal_config_never_evaluates_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> dict[str, object]:
        pytest.fail("non-formal execution evaluated the formal overlay")

    monkeypatch.setattr(gate, "_evaluate_authenticated_overlay", forbidden)
    gate.require_formal_execution_authorization(
        {"experiment": "anchor-moe-lora-diagnostic"}
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: value.__setitem__(
                "schema_version", "anchor.formal-authorization-consumer-decision.v1"
            ),
            "schema is not eligible",
        ),
        (
            lambda value: value.__setitem__("status", "ready"),
            "v1 is blocked-only",
        ),
        (
            lambda value: value.__setitem__("formal_training_authorized", True),
            "v1 is blocked-only",
        ),
    ],
)
def test_semantically_ineligible_decisions_fail_closed(mutation, error: str) -> None:
    value = _decision()
    mutation(value)
    value.pop("decision_sha256")
    value["decision_sha256"] = gate._canonical_sha256(value)
    with pytest.raises(gate.FormalExecutionGateError, match=error):
        gate._reject_v1_formal_decision(value)


def test_decision_digest_is_canonical_and_mandatory() -> None:
    value = _decision()
    changed = copy.deepcopy(value)
    changed["authenticated_inputs"] = {"formal_v3": "drifted"}
    with pytest.raises(gate.FormalExecutionGateError, match="digest mismatch"):
        gate._reject_v1_formal_decision(changed)

    missing = dict(value)
    missing.pop("decision_sha256")
    with pytest.raises(gate.FormalExecutionGateError, match="invalid decision_sha256"):
        gate._reject_v1_formal_decision(missing)


def test_valid_blocked_v1_decision_cannot_authorize_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_evaluate_authenticated_overlay", _decision)
    with pytest.raises(
        gate.FormalExecutionGateError,
        match="v1 is blocked-only and cannot authorize execution",
    ):
        gate.require_formal_execution_authorization(
            {"experiment": "anchor-moe-lora-formal-v3-C"}
        )


@pytest.mark.parametrize("alias", ["formal_v3.", "formal_v3 "])
def test_win32_trailing_aliases_cannot_hide_formal_runtime_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    monkeypatch.setattr(gate, "_evaluate_authenticated_overlay", _decision)
    aliased_output = tmp_path / "artifacts" / alias / "planner"

    with pytest.raises(
        gate.FormalExecutionGateError,
        match="v1 is blocked-only and cannot authorize execution",
    ):
        gate.require_formal_execution_authorization(
            {"experiment": "renamed-diagnostic"},
            output_dir=aliased_output,
        )


def test_ntfs_ads_cannot_hide_formal_manifest_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_evaluate_authenticated_overlay", _decision)
    aliased_manifest = tmp_path / "artifacts" / "formal_v3:stream"

    with pytest.raises(
        gate.FormalExecutionGateError,
        match="v1 is blocked-only and cannot authorize execution",
    ):
        gate.require_formal_execution_authorization(
            {"experiment": "renamed-diagnostic"},
            manifest_path=aliased_manifest,
        )


def test_runtime_gate_precedes_progress_output_and_training_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "_evaluate_authenticated_overlay", _decision)

    def forbidden_progress(*_args, **_kwargs):
        pytest.fail("formal runtime created a progress reporter before authorization")

    def forbidden_backend(*_args, **_kwargs):
        pytest.fail("formal runtime entered the training backend before authorization")

    monkeypatch.setattr(runtime, "TrainingProgress", forbidden_progress)
    monkeypatch.setattr(runtime, "_train_adapter_impl", forbidden_backend)
    output_dir = tmp_path / "formal-output"
    with pytest.raises(
        gate.FormalExecutionGateError,
        match="v1 is blocked-only and cannot authorize execution",
    ):
        runtime.train_adapter(
            {"experiment": "anchor-moe-lora-formal-v3-C"},
            dataset_paths=[tmp_path / "must-not-be-read.jsonl"],
            output_dir=output_dir,
            allow_model_download=False,
            manifest={},
        )
    assert not output_dir.exists()


def test_deepest_training_backend_rechecks_gate_before_reporter_or_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_evaluate_authenticated_overlay", _decision)

    class ForbiddenReporter:
        def emit(self, *_args, **_kwargs) -> None:
            pytest.fail("deep backend touched the reporter before authorization")

    output_dir = tmp_path / "deep-formal-output"
    with pytest.raises(gate.FormalExecutionGateError):
        runtime._train_adapter_impl(
            {"experiment": "anchor-moe-lora-formal-v3-C"},
            dataset_paths=[tmp_path / "must-not-be-read.jsonl"],
            output_dir=output_dir,
            allow_model_download=False,
            manifest={},
            reporter=ForbiddenReporter(),
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("protected_kind", ["dataset", "output"])
def test_runtime_paths_cannot_be_hidden_by_renaming_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_kind: str,
) -> None:
    monkeypatch.setattr(gate, "_evaluate_authenticated_overlay", _decision)

    def forbidden_progress(*_args, **_kwargs):
        pytest.fail("renamed formal runtime created a progress reporter")

    monkeypatch.setattr(runtime, "TrainingProgress", forbidden_progress)
    protected = tmp_path / "artifacts" / "formal_v3"
    dataset_paths = (
        [protected / "dataset" / "planner.jsonl"]
        if protected_kind == "dataset"
        else [tmp_path / "diagnostic.jsonl"]
    )
    output_dir = (
        protected / "adapters" / "planner"
        if protected_kind == "output"
        else tmp_path / "diagnostic-output"
    )

    with pytest.raises(
        gate.FormalExecutionGateError,
        match="v1 is blocked-only and cannot authorize execution",
    ):
        runtime.train_adapter(
            {"experiment": "renamed-diagnostic"},
            dataset_paths=dataset_paths,
            output_dir=output_dir,
            allow_model_download=False,
            manifest={},
        )
    assert not output_dir.exists()
