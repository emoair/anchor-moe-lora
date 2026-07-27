from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data import gemma3_chat_unbalanced_v2_live_controller as controller


def _inventory(suffix: str = "1") -> batch.SourceInventory:
    digest = suffix * 64
    return batch.SourceInventory(
        records=(),
        manifest_sha256=digest,
        approval_sha256=digest,
        partition_set_sha256=digest,
        readonly_test_identity_sha256=digest,
        source_identity_sha256=digest,
        role_counts={},
        language_counts={},
        identity_counts={},
    )


def _source_record() -> batch.SourceRecord:
    return batch.SourceRecord(
        record_id="record-1",
        task_bundle="bundle-1",
        semantic="semantic-1",
        role="serious",
        language="en",
        teacher_input="synthetic input",
        gemma_serialization_identity_sha256="a" * 64,
        teacher_contract={
            "identity_class": None,
            "prompt_template_id": "template-1",
            "output_schema_id": "natural-text-v1",
            "allowed_tools": [],
            "allowed_evidence_ids": [],
            "router_options": [],
        },
        guards={},
        partition_kind="train",
        line_number=1,
        source_line_sha256="b" * 64,
        content_identity_sha256="c" * 64,
    )


def _job() -> batch.AlignmentJob:
    return batch.AlignmentJob(
        source=_source_record(),
        idempotency_key="d" * 64,
        prompt_template_sha256="e" * 64,
        output_schema_sha256="f" * 64,
    )


def _runtime_slots(key: bytes = b"k" * 32) -> batch.RuntimeSecretSlots:
    return batch.RuntimeSecretSlots.from_process_channel(
        lambda: b"fake-controller-credential",
        hmac_factory=lambda _: key,
    )


def _state(
    tmp_path: Path,
    slots: batch.RuntimeSecretSlots,
) -> tuple[controller.AuthenticatedBatchState, batch.SourceInventory]:
    profile = batch.load_config(
        controller.DEFAULT_CONFIG_PATH.parent
        / "gemma3_chat_five_expert_qonly_unbalanced_v2_"
        "teacher_alignment.smoke_exact1.yaml"
    )
    output = dict(profile.output)
    output["root"] = str(tmp_path / "data" / batch.DATASET_KIND / "shard-0001")
    profile = replace(profile, output=output)
    inventory = _inventory()
    binding = {
        "controller_config_sha256": "1" * 64,
        "controller_implementation_sha256": "2" * 64,
        "teacher_implementation_sha256": "4" * 64,
        "source_identity_sha256": inventory.source_identity_sha256,
        "manifest_sha256": inventory.manifest_sha256,
        "protected_test_identity_sha256": (inventory.readonly_test_identity_sha256),
        "controller_run_id": "3" * 32,
        "accepted_profiles": {profile.profile_id: profile.physical_sha256},
    }
    state = controller.AuthenticatedBatchState(
        profile,
        hmac_key=slots.receipt_hmac_key(),
        controller_binding=binding,
    )
    state.initialize(inventory)
    return state, inventory


async def _append_dispatch(
    state: controller.AuthenticatedBatchState,
    job: batch.AlignmentJob,
) -> None:
    await state.append_event(
        job=job,
        phase="smoke_exact1",
        state="reserved",
        reason_code="budget_reserved",
        reservation={
            "requests": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "cost_units": 1,
        },
    )
    await state.append_event(
        job=job,
        phase="smoke_exact1",
        state="dispatching",
        reason_code="provider_dispatch",
    )


async def _append_rate_limit(
    state: controller.AuthenticatedBatchState,
    job: batch.AlignmentJob,
) -> None:
    await _append_dispatch(state, job)
    await state.append_event(
        job=job,
        phase="bulk",
        state="retryable",
        reason_code="provider_rate_limit",
        attempts={
            "wire_attempts": 1,
            "retry_count": 0,
            "retry_reasons": [],
        },
    )


def _bound_for_tmp(
    tmp_path: Path,
) -> tuple[
    controller.BoundProfiles,
    dict[str, batch.AdapterConfig],
]:
    canonical = controller.load_controller_config()
    common = dict(canonical.common_identity)
    common["output_root"] = str(tmp_path / "data" / batch.DATASET_KIND / "run")
    altered = replace(canonical, common_identity=common)
    values: dict[str, batch.AdapterConfig] = {}
    for spec in altered.profiles:
        value = batch.load_config(spec.path)
        output = dict(value.output)
        output["root"] = common["output_root"]
        values[spec.profile_id] = replace(value, output=output)

    def load(path: Path) -> batch.AdapterConfig:
        spec = next(item for item in altered.profiles if item.path == path)
        return values[spec.profile_id]

    return controller.bind_profiles(altered, profile_loader=load), values


def _report(profile_id: str, phase: str) -> batch.RunReport:
    total = {
        "smoke_exact1": 1,
        "bounded_small_c1": 15,
        "bulk_c30": batch.EXPECTED_TOTAL_JOBS,
        "bulk_c16": batch.EXPECTED_TOTAL_JOBS,
    }[profile_id]
    requests = 1 if profile_id == "smoke_exact1" else 15
    if profile_id.startswith("bulk"):
        requests = batch.EXPECTED_TOTAL_JOBS
    return batch.RunReport(
        phase=phase,
        total=total,
        queued=0,
        succeeded=total,
        rejected=0,
        retried=0,
        requests=requests,
        input_tokens=0,
        output_tokens=0,
        state="complete",
    )


def test_same_runtime_slots_cover_all_three_stages_and_are_closed(
    tmp_path: Path,
) -> None:
    bound, values = _bound_for_tmp(tmp_path)
    slots = _runtime_slots()
    seen: list[int] = []

    async def execute(
        profile: batch.AdapterConfig,
        inventory: batch.SourceInventory,
        runtime_slots: batch.RuntimeSecretSlots,
        phase: str,
        environ: dict[str, str],
        binding: dict[str, Any],
    ) -> batch.RunReport:
        del inventory, environ, binding
        seen.append(id(runtime_slots))
        return _report(profile.profile_id, phase)

    result = asyncio.run(
        controller.SingleProcessController(
            bound,
            slots,
            profile_loader=lambda path: next(
                values[item.profile_id]
                for item in bound.controller.profiles
                if item.path == path
            ),
            source_loader=lambda _: _inventory(),
            heldout_validator=lambda _: {},
            stage_executor=execute,
            consumer_validator=lambda _: None,
            environ={},
        ).run()
    )
    assert result["state"] == "complete"
    assert result["selected_bulk_profile"] == "bulk_c30"
    assert len(seen) == 3 and len(set(seen)) == 1
    assert slots.loaded is False
    assert "fake-controller-credential" not in json.dumps(result)
    assert result["runtime_hmac_public"] is False


def test_stage_failure_stops_later_stages_and_closes_slots(tmp_path: Path) -> None:
    bound, values = _bound_for_tmp(tmp_path)
    slots = _runtime_slots()
    called: list[str] = []

    async def execute(
        profile: batch.AdapterConfig,
        *_: Any,
    ) -> batch.RunReport:
        called.append(profile.profile_id)
        if profile.profile_id == "bounded_small_c1":
            raise batch.AdapterError("synthetic_stage_failure")
        return _report(profile.profile_id, profile.limits.allowed_phases[0])

    with pytest.raises(batch.AdapterError, match="synthetic stage failure"):
        asyncio.run(
            controller.SingleProcessController(
                bound,
                slots,
                profile_loader=lambda path: next(
                    values[item.profile_id]
                    for item in bound.controller.profiles
                    if item.path == path
                ),
                source_loader=lambda _: _inventory(),
                heldout_validator=lambda _: {},
                stage_executor=execute,
                consumer_validator=lambda _: None,
                environ={},
            ).run()
        )
    assert called == ["smoke_exact1", "bounded_small_c1"]
    assert slots.loaded is False


def test_c16_never_runs_without_authenticated_rate_limit(
    tmp_path: Path,
) -> None:
    bound, values = _bound_for_tmp(tmp_path)
    slots = _runtime_slots()
    called: list[str] = []

    async def execute(
        profile: batch.AdapterConfig,
        *_: Any,
    ) -> batch.RunReport:
        called.append(profile.profile_id)
        if profile.profile_id == "bulk_c30":
            raise batch.RateLimitError()
        return _report(profile.profile_id, profile.limits.allowed_phases[0])

    def no_trigger(*_: Any) -> str:
        raise batch.AdapterError("bulk_c16_authenticated_rate_limit_missing")

    with pytest.raises(
        batch.AdapterError, match="bulk c16 authenticated rate limit missing"
    ):
        asyncio.run(
            controller.SingleProcessController(
                bound,
                slots,
                profile_loader=lambda path: next(
                    values[item.profile_id]
                    for item in bound.controller.profiles
                    if item.path == path
                ),
                source_loader=lambda _: _inventory(),
                heldout_validator=lambda _: {},
                stage_executor=execute,
                fallback_verifier=no_trigger,
                consumer_validator=lambda _: None,
                environ={},
            ).run()
        )
    assert called == ["smoke_exact1", "bounded_small_c1", "bulk_c30"]


def test_source_identity_drift_stops_before_next_stage(tmp_path: Path) -> None:
    bound, values = _bound_for_tmp(tmp_path)
    slots = _runtime_slots()
    reads = 0

    def source(_: batch.AdapterConfig) -> batch.SourceInventory:
        nonlocal reads
        reads += 1
        return _inventory("1" if reads == 1 else "2")

    async def execute(
        profile: batch.AdapterConfig,
        *_: Any,
    ) -> batch.RunReport:
        return _report(profile.profile_id, profile.limits.allowed_phases[0])

    with pytest.raises(batch.AdapterError, match="controller source identity drift"):
        asyncio.run(
            controller.SingleProcessController(
                bound,
                slots,
                profile_loader=lambda path: next(
                    values[item.profile_id]
                    for item in bound.controller.profiles
                    if item.path == path
                ),
                source_loader=source,
                heldout_validator=lambda _: {},
                stage_executor=execute,
                consumer_validator=lambda _: None,
                environ={},
            ).run()
        )
    assert slots.loaded is False


def test_nonempty_output_root_is_not_cross_process_resume(tmp_path: Path) -> None:
    bound, values = _bound_for_tmp(tmp_path)
    root = values["smoke_exact1"].output_root
    root.mkdir(parents=True)
    (root / "partial").write_text("x", encoding="utf-8")
    slots = _runtime_slots()
    with pytest.raises(
        batch.AdapterError, match="controller existing output not resumable"
    ):
        asyncio.run(
            controller.SingleProcessController(
                bound,
                slots,
                consumer_validator=lambda _: None,
                environ={},
            ).run()
        )
    assert slots.loaded is False


def test_immutable_wal_detects_event_mutation_and_hmac_rotation(
    tmp_path: Path,
) -> None:
    slots = _runtime_slots()
    state, _ = _state(tmp_path, slots)
    asyncio.run(_append_dispatch(state, _job()))
    state.authenticate_authority()

    other_slots = _runtime_slots(b"z" * 32)
    other = controller.AuthenticatedBatchState(
        state.config,
        hmac_key=other_slots.receipt_hmac_key(),
        controller_binding=state.controller_binding,
    )
    with pytest.raises(batch.AdapterError):
        other.initialize(_inventory())

    raw = state.events_path.read_bytes()
    state.events_path.write_bytes(
        raw.replace(b"provider_dispatch", b"provider_dispatcH")
    )
    with pytest.raises(batch.AdapterError):
        state.authenticate_authority()
    slots.close()
    other_slots.close()


def test_wal_entry_directly_binds_teacher_implementation(tmp_path: Path) -> None:
    slots = _runtime_slots()
    state, _ = _state(tmp_path, slots)
    asyncio.run(_append_dispatch(state, _job()))
    entry_path = state._entry_paths()[0]
    value = json.loads(entry_path.read_text(encoding="utf-8"))
    assert (
        value["teacher_implementation_sha256"]
        == state.controller_binding["teacher_implementation_sha256"]
    )
    value["teacher_implementation_sha256"] = "0" * 64
    resigned = state._signed_value(
        {name: item for name, item in value.items() if name != "hmac_sha256"}
    )
    entry_path.write_text(
        json.dumps(resigned, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(batch.AdapterError) as captured:
        state.authenticate_authority()
    assert captured.value.reason_code == "controller_wal_entry_identity_drift"
    slots.close()


def test_group_prepare_terminal_cross_binding_rejects_group_mutation(
    tmp_path: Path,
) -> None:
    slots = _runtime_slots()
    state, inventory = _state(tmp_path, slots)
    job = _job()
    asyncio.run(_append_dispatch(state, job))
    receipt_payload = {
        "id": "a" * 64,
        "idempotency_key": job.idempotency_key,
        "outcome": "succeeded",
        "reason_code": "validated",
        "attempts": {"wire_attempts": 1},
        "binding": state._binding(inventory),
    }
    receipt = {
        **receipt_payload,
        "hmac_sha256": batch._receipt_hmac(receipt_payload, slots.receipt_hmac_key()),
    }
    state.commit_group(
        [
            batch.PendingCommit(
                job=job,
                output_record={"idempotency_key": job.idempotency_key},
                receipt=receipt,
                terminal_state="committed",
                event_reason="validated",
            )
        ],
        inventory=inventory,
        phase="smoke_exact1",
    )
    state.authenticate_authority()
    group_path = next(state.groups_dir.glob("*.json"))
    value = json.loads(group_path.read_text(encoding="utf-8"))
    value["group_sha256"] = "0" * 64
    group_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(batch.AdapterError):
        state.authenticate_authority()
    slots.close()


def test_regular_file_tty_env_and_argv_secret_paths_are_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    regular = tmp_path / "credential.txt"
    regular.write_bytes(b"not-a-real-key")
    with regular.open("rb") as handle, pytest.raises(batch.AdapterError):
        controller._validate_anonymous_os_channel(handle)
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "rb", closefd=False) as reader:
            controller._validate_anonymous_os_channel(reader)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    with pytest.raises(batch.AdapterError):
        controller._reject_environment_credentials(
            {"ARK_CODING_API_KEY": "not-inspected"}
        )
    controller._reject_argv_credentials(["--execute", "--credential-stdin"])
    marker = "ark" + "-" + "this-must-never-appear-in-output"
    assert controller.main([f"--api-key={marker}", "--dry-run"]) == 2
    assert marker not in capsys.readouterr().out


def test_signed_multi_event_cooldown_ignores_status_and_detects_wal_tamper(
    tmp_path: Path,
) -> None:
    slots = _runtime_slots()
    c30 = batch.load_config(
        controller.DEFAULT_CONFIG_PATH.parent
        / "gemma3_chat_five_expert_qonly_unbalanced_v2_"
        "teacher_alignment.bulk_c30.yaml"
    )
    c16 = batch.load_config(
        controller.DEFAULT_CONFIG_PATH.parent
        / "gemma3_chat_five_expert_qonly_unbalanced_v2_"
        "teacher_alignment.bulk_c16.yaml"
    )
    output = dict(c30.output)
    output["root"] = str(tmp_path / "data" / batch.DATASET_KIND / "shard-0001")
    c30 = replace(c30, output=output)
    c16 = replace(c16, output=output)
    inventory = _inventory()
    binding = {
        "controller_config_sha256": "1" * 64,
        "controller_implementation_sha256": "2" * 64,
        "teacher_implementation_sha256": "4" * 64,
        "source_identity_sha256": inventory.source_identity_sha256,
        "manifest_sha256": inventory.manifest_sha256,
        "protected_test_identity_sha256": (inventory.readonly_test_identity_sha256),
        "controller_run_id": "3" * 32,
        "accepted_profiles": {
            c30.profile_id: c30.physical_sha256,
            c16.profile_id: c16.physical_sha256,
        },
    }
    state = controller.AuthenticatedBatchState(
        c30,
        hmac_key=slots.receipt_hmac_key(),
        controller_binding=binding,
    )
    state.initialize(inventory)
    second = replace(_job(), idempotency_key="9" * 64)
    asyncio.run(_append_rate_limit(state, _job()))
    asyncio.run(_append_rate_limit(state, second))
    lease = state.record_fallback_cooldown(
        fallback_profile=c16,
        retry_after_seconds=1,
        now_seconds=100,
    )
    assert lease["retryable_event_count"] == 2
    state.status_path.write_text('{"cooldown_until":"attacker-controlled"}\n')
    state.complete_fallback_cooldown(
        lease,
        now_seconds=float(lease["not_before_unix_seconds"]),
    )
    assert (
        state.verify_completed_fallback_cooldown(fallback_profile=c16)
        == "provider_rate_limit"
    )
    cooldown_path = state._entry_paths()[-2]
    cooldown_path.write_bytes(
        cooldown_path.read_bytes().replace(
            b'"retryable_event_count":2',
            b'"retryable_event_count":3',
        )
    )
    with pytest.raises(batch.AdapterError):
        state.authenticate_authority()
    slots.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics")
def test_alignment_junction_swap_is_rejected_before_external_write(
    tmp_path: Path,
) -> None:
    slots = _runtime_slots()
    state, _ = _state(tmp_path, slots)
    outside = tmp_path / "outside"
    outside.mkdir()
    backup = state.alignment_dir.with_name("alignment.backup")
    state.alignment_dir.rename(backup)
    result = subprocess.run(
        [
            "cmd",
            "/c",
            "mklink",
            "/J",
            str(state.alignment_dir),
            str(outside),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("junction creation unavailable")
    with pytest.raises(batch.AdapterError):
        asyncio.run(_append_dispatch(state, _job()))
    assert list(outside.iterdir()) == []
    slots.close()


def test_main_unknown_exception_is_body_free(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "provider-secret-response-body-must-not-appear"

    def fail(*_: Any, **__: Any) -> controller.ControllerConfig:
        raise RuntimeError(marker)

    monkeypatch.setattr(controller, "load_controller_config", fail)
    assert controller.main(["--dry-run"]) == 2
    captured = capsys.readouterr()
    assert marker not in captured.out
    assert marker not in captured.err
    assert "controller_internal_failure" in captured.out


def test_controller_implementation_drift_blocks_before_consumer_or_git(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_read = controller._stable_read
    calls = {"consumer": 0, "git": 0}

    def drift(path: Path, *, reason: str) -> bytes:
        raw = original_read(path, reason=reason)
        if path.resolve() == controller.CONTROLLER_IMPLEMENTATION_PATH:
            return raw + b"\n# authenticated-drift-test\n"
        return raw

    def consumer_call(*_: Any, **__: Any) -> None:
        calls["consumer"] += 1

    def git_call(*_: Any, **__: Any) -> Any:
        calls["git"] += 1
        raise AssertionError("git must not run before controller authentication")

    monkeypatch.setattr(controller, "_stable_read", drift)
    monkeypatch.setattr(controller, "validate_consumer_release_binding", consumer_call)
    monkeypatch.setattr(controller.subprocess, "run", git_call)
    assert controller.main(["--dry-run"]) == 2
    captured = capsys.readouterr()
    assert "controller_implementation_hash_drift" in captured.out
    assert calls == {"consumer": 0, "git": 0}


def test_teacher_implementation_drift_blocks_before_consumer_or_git(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_read = controller._stable_read
    calls = {"consumer": 0, "git": 0}

    def drift(path: Path, *, reason: str) -> bytes:
        raw = original_read(path, reason=reason)
        if path.resolve().name == "teacher.py":
            return raw + b"\n# authenticated-drift-test\n"
        return raw

    def consumer_call(*_: Any, **__: Any) -> None:
        calls["consumer"] += 1

    def git_call(*_: Any, **__: Any) -> Any:
        calls["git"] += 1
        raise AssertionError("git must not run before teacher authentication")

    monkeypatch.setattr(controller, "_stable_read", drift)
    monkeypatch.setattr(controller, "validate_consumer_release_binding", consumer_call)
    monkeypatch.setattr(controller.subprocess, "run", git_call)
    assert controller.main(["--dry-run"]) == 2
    captured = capsys.readouterr()
    assert "controller_teacher_implementation_drift" in captured.out
    assert calls == {"consumer": 0, "git": 0}


def test_secret_never_appears_in_repr() -> None:
    secret = b"fake-controller-credential"
    slots = batch.RuntimeSecretSlots.from_process_channel(
        lambda: secret,
        hmac_factory=lambda _: b"h" * 32,
    )
    assert secret.decode() not in repr(slots)
    slots.close()
