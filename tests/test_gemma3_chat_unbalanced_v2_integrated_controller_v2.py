from __future__ import annotations

import ast
import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from typing import Any

import pytest

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data import gemma3_chat_unbalanced_v2_integrated_controller_v2 as v2
from anchor_mvp.data import gemma3_chat_unbalanced_v2_live_controller as v1


ACTIVE_CONFIG_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_integrated_controller_"
    "consumer_rollover_v2.json"
)


def _inventory() -> batch.SourceInventory:
    return batch.SourceInventory(
        records=(),
        manifest_sha256="1" * 64,
        approval_sha256="2" * 64,
        partition_set_sha256="3" * 64,
        readonly_test_identity_sha256="4" * 64,
        source_identity_sha256="5" * 64,
        role_counts={},
        language_counts={},
        identity_counts={},
    )


def _runtime_slots() -> batch.RuntimeSecretSlots:
    return batch.RuntimeSecretSlots.from_process_channel(
        lambda: b"synthetic-process-only-credential",
        hmac_factory=lambda _: b"h" * 32,
    )


def _report(profile_id: str, phase: str) -> batch.RunReport:
    total = {
        "smoke_exact1": 1,
        "bounded_small_c1": 15,
        "bulk_c30": 3520,
        "bulk_c16": 3520,
    }[profile_id]
    return batch.RunReport(
        phase=phase,
        total=total,
        queued=0,
        succeeded=total,
        rejected=0,
        retried=0,
        requests=total,
        input_tokens=0,
        output_tokens=0,
        state="complete",
    )


def _bound_for_tmp(
    tmp_path: Path,
) -> tuple[
    v2.IntegratedControllerConfig,
    v1.BoundProfiles,
    dict[str, batch.AdapterConfig],
]:
    integrated = v2.load_integrated_config(ACTIVE_CONFIG_PATH)
    canonical = integrated.base_controller
    common = dict(canonical.common_identity)
    common["output_root"] = str(tmp_path / "data" / "integrated-run")
    altered = replace(canonical, common_identity=common)
    values: dict[str, batch.AdapterConfig] = {}
    for spec in altered.profiles:
        profile = batch.load_config(spec.path)
        output = dict(profile.output)
        output["root"] = common["output_root"]
        values[spec.profile_id] = replace(profile, output=output)

    def loader(path: Path) -> batch.AdapterConfig:
        spec = next(item for item in altered.profiles if item.path == path)
        return values[spec.profile_id]

    bound = v1.bind_profiles(altered, profile_loader=loader)
    return integrated, bound, values


def _profile_loader(
    bound: v1.BoundProfiles,
    values: dict[str, batch.AdapterConfig],
) -> Any:
    return lambda path: next(
        values[item.profile_id]
        for item in bound.controller.profiles
        if item.path == path
    )


def _teacher_status() -> dict[str, Any]:
    return {
        "state": "candidate_pending_independent_release_review",
        "records": 3520,
        "main_records": 3440,
        "router_records": 80,
        "train_identity_bundles": 20,
        "train_identity_records": 100,
        "runtime_hmac_consumed_before_close": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "live_authorized": False,
        "manifest_sha256": "a" * 64,
    }


def test_exact1_15_c30_then_finalizer_before_slot_close(tmp_path: Path) -> None:
    integrated, bound, values = _bound_for_tmp(tmp_path)
    slots = _runtime_slots()
    order: list[str] = []

    async def execute(
        profile: batch.AdapterConfig,
        *_: Any,
    ) -> batch.RunReport:
        order.append(profile.profile_id)
        return _report(profile.profile_id, profile.limits.allowed_phases[0])

    def finalize(
        profile: batch.AdapterConfig,
        inventory: batch.SourceInventory,
        runtime_slots: batch.RuntimeSecretSlots,
        controller_binding: dict[str, Any],
        controller_identity: dict[str, Any],
        final_report: batch.RunReport,
        config: v2.IntegratedControllerConfig,
    ) -> dict[str, Any]:
        del inventory, controller_binding, config
        assert profile.profile_id == "bulk_c30"
        assert final_report.succeeded == 3520
        assert runtime_slots.loaded is True
        assert len(runtime_slots.receipt_hmac_key()) == 32
        assert controller_identity["controller_ancestor_commit"] == (
            "adfa03e225441f010e80f3b2010aec5484a5a90b"
        )
        order.append("teacher_finalizer")
        return _teacher_status()

    result = asyncio.run(
        v2.IntegratedSingleProcessController(
            integrated,
            bound,
            slots,
            teacher_finalizer=finalize,
            profile_loader=_profile_loader(bound, values),
            source_loader=lambda _: _inventory(),
            heldout_validator=lambda _: {},
            stage_executor=execute,
            consumer_validator=lambda _: None,
            environ={},
        ).run()
    )
    assert order == [
        "smoke_exact1",
        "bounded_small_c1",
        "bulk_c30",
        "teacher_finalizer",
    ]
    assert result["state"] == "complete"
    assert result["teacher_final"]["manifest_sha256"] == "a" * 64
    assert result["runtime_secret_slots_shared_through_finalization"] is True
    assert slots.loaded is False


def test_finalizer_failure_prevents_complete_and_closes_slots(
    tmp_path: Path,
) -> None:
    integrated, bound, values = _bound_for_tmp(tmp_path)
    slots = _runtime_slots()
    saw_live_slot = False

    async def execute(
        profile: batch.AdapterConfig,
        *_: Any,
    ) -> batch.RunReport:
        return _report(profile.profile_id, profile.limits.allowed_phases[0])

    def fail(
        _profile: batch.AdapterConfig,
        _inventory: batch.SourceInventory,
        runtime_slots: batch.RuntimeSecretSlots,
        *_: Any,
    ) -> dict[str, Any]:
        nonlocal saw_live_slot
        saw_live_slot = runtime_slots.loaded
        raise batch.AdapterError("synthetic_teacher_finalizer_failure")

    with pytest.raises(
        batch.AdapterError,
        match="synthetic teacher finalizer failure",
    ):
        asyncio.run(
            v2.IntegratedSingleProcessController(
                integrated,
                bound,
                slots,
                teacher_finalizer=fail,
                profile_loader=_profile_loader(bound, values),
                source_loader=lambda _: _inventory(),
                heldout_validator=lambda _: {},
                stage_executor=execute,
                consumer_validator=lambda _: None,
                environ={},
            ).run()
        )
    assert saw_live_slot is True
    assert slots.loaded is False


def test_authenticated_c16_path_finalizes_c16_output(tmp_path: Path) -> None:
    integrated, bound, values = _bound_for_tmp(tmp_path)
    slots = _runtime_slots()
    order: list[str] = []

    async def execute(
        profile: batch.AdapterConfig,
        *_: Any,
    ) -> batch.RunReport:
        order.append(profile.profile_id)
        if profile.profile_id == "bulk_c30":
            raise batch.RateLimitError()
        return _report(profile.profile_id, profile.limits.allowed_phases[0])

    def finalize(
        profile: batch.AdapterConfig,
        _inventory: batch.SourceInventory,
        runtime_slots: batch.RuntimeSecretSlots,
        *_: Any,
    ) -> dict[str, Any]:
        assert profile.profile_id == "bulk_c16"
        assert runtime_slots.loaded
        order.append("teacher_finalizer")
        return _teacher_status()

    class TestController(v2.IntegratedSingleProcessController):
        async def _complete_authenticated_cooldown(self, **_: Any) -> None:
            return None

    result = asyncio.run(
        TestController(
            integrated,
            bound,
            slots,
            teacher_finalizer=finalize,
            profile_loader=_profile_loader(bound, values),
            source_loader=lambda _: _inventory(),
            heldout_validator=lambda _: {},
            stage_executor=execute,
            fallback_verifier=lambda *_: "provider_rate_limit",
            consumer_validator=lambda _: None,
            environ={},
        ).run()
    )
    assert order == [
        "smoke_exact1",
        "bounded_small_c1",
        "bulk_c30",
        "bulk_c16",
        "teacher_finalizer",
    ]
    assert result["selected_bulk_profile"] == "bulk_c16"
    assert result["fallback_reason_code"] == "provider_rate_limit"
    assert slots.loaded is False


def test_v1_bytes_remain_bound_and_v2_dry_run_is_non_live(
    capsys: pytest.CaptureFixture[str],
) -> None:
    integrated = v2.load_integrated_config(ACTIVE_CONFIG_PATH)
    assert integrated.base_controller.physical_sha256 == (
        "d3485bee084f1ca07e47220ba07bf4fa4aa59581a6922310b4c326ee62b19c22"
    )
    assert integrated.base_controller.controller_implementation_sha256 == (
        "7bb024e2eac4d8fbd1af9c64cf6ffa2b4203a6d3044df07892ab91a2b2064595"
    )
    assert integrated.base_controller.teacher_implementation_sha256 == (
        "5545d8775adcb3b0750ebbdaea662a9fbe3f13c360827dfefc88f474de5755f9"
    )
    assert v2.main(["--config", str(ACTIVE_CONFIG_PATH), "--dry-run"]) == 2
    output = capsys.readouterr().out
    assert "runtime_hmac_slot_unloaded" in output
    assert '"network_requests": 0' in output
    assert '"live_authorized": false' in output


def test_default_finalizer_loader_is_python310_compatible_and_not_cached() -> None:
    integrated = v2.load_integrated_config(ACTIVE_CONFIG_PATH)
    snapshot = integrated.snapshots["teacher_implementation"]
    tree = ast.parse(snapshot.raw, feature_version=(3, 10))
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "datetime"
        and any(alias.name == "UTC" for alias in node.names)
        for node in ast.walk(tree)
    )
    function = v2._load_finalizer_callable(integrated)
    module_name = f"_anchor_teacher_finalizer_{snapshot.sha256}"
    assert callable(function)
    assert function.__module__ == module_name
    assert module_name not in sys.modules


def test_default_finalizer_entrypoint_executes_and_fails_closed_body_free() -> None:
    integrated = v2.load_integrated_config(ACTIVE_CONFIG_PATH)
    profile = batch.load_config(integrated.base_controller.profile_map["bulk_c30"].path)
    slots = _runtime_slots()
    slots.close()
    with pytest.raises(
        batch.AdapterError,
        match="teacher final runtime not green",
    ):
        v2._default_teacher_finalizer(
            profile,
            _inventory(),
            slots,
            {"controller_run_id": "0" * 32},
            {},
            _report("bulk_c30", "bulk"),
            integrated,
        )


def test_cli_config_and_internal_file_same_content_symlinks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "controller.json"
    target.write_bytes(b'{"same":"bytes"}\n')
    config_link = root / "controller-link.json"
    internal_link = root / "internal-link.json"
    try:
        config_link.symlink_to(target)
        internal_link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    monkeypatch.setattr(batch, "REPO_ROOT", root)
    with pytest.raises(
        batch.AdapterError,
        match="integrated controller reparse path forbidden",
    ):
        v2.load_integrated_config(config_link)
    with pytest.raises(
        batch.AdapterError,
        match="integrated controller reparse path forbidden",
    ):
        v2._repo_file(
            "internal-link.json",
            reason="integrated_test_internal_path_invalid",
        )


def test_finalizer_implementation_drift_after_config_load_fails_closed(
    tmp_path: Path,
) -> None:
    integrated = v2.load_integrated_config(ACTIVE_CONFIG_PATH)
    canonical = integrated.snapshots["teacher_implementation"]
    copied_path = tmp_path / "teacher_finalizer.py"
    copied_path.write_bytes(canonical.raw)
    copied = v2._snapshot_file(
        copied_path,
        expected_sha256=canonical.sha256,
        reason="test_finalizer_copy_invalid",
    )
    snapshots = dict(integrated.snapshots)
    snapshots["teacher_implementation"] = copied
    paths = dict(integrated.teacher_finalizer_paths)
    paths["implementation"] = copied_path
    altered = replace(
        integrated,
        teacher_finalizer_paths=paths,
        snapshots=snapshots,
    )
    mutated = bytearray(canonical.raw)
    mutated[-1] = ord(" ") if mutated[-1] != ord(" ") else ord("\n")
    copied_path.write_bytes(bytes(mutated))
    with pytest.raises(
        batch.AdapterError,
        match="integrated teacher finalizer implementation drift",
    ):
        v2._load_finalizer_callable(altered)


@pytest.mark.parametrize(
    "snapshot_key",
    ["base_implementation", "profile_smoke_exact1"],
)
def test_base_and_profile_anchor_drift_fail_closed(
    tmp_path: Path,
    snapshot_key: str,
) -> None:
    integrated = v2.load_integrated_config(ACTIVE_CONFIG_PATH)
    raw = integrated.snapshots[snapshot_key].raw
    path = tmp_path / f"{snapshot_key}.bin"
    path.write_bytes(raw)
    copied = v2._snapshot_file(
        path,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        reason="test_anchor_copy_invalid",
    )
    snapshots = dict(integrated.snapshots)
    snapshots[snapshot_key] = copied
    altered = replace(integrated, snapshots=snapshots)
    path.write_bytes(raw + b"\n")
    with pytest.raises(
        batch.AdapterError,
        match="integrated controller artifact identity drift",
    ):
        v2._recheck_integrated(altered)
