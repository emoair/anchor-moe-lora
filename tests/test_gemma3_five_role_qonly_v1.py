from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from anchor_mvp.training import gemma3_five_role_qonly_v1 as runner


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return runner.load_config(runner.CONFIG_PATH)


def _write_receipt(path: Path, value: object) -> str:
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(data).hexdigest()


def _gui_processes() -> list[dict[str, object]]:
    return [
        {
            "pid": 4304,
            "process_name": "dwm.exe",
            "used_gpu_memory_mib": "[N/A]",
            "reported_name_was_permission_denied": True,
            "allowlisted_wddm_gui": True,
        }
    ]


def _compute_inventory_sha256() -> str:
    return _canonical_sha256(
        [
            {"pid": item["pid"], "process_name": item["process_name"]}
            for item in _gui_processes()
        ]
    )


def _attested_gpu_policy() -> dict[str, object]:
    return {
        "sample_count": 3,
        "sample_interval_seconds": 1,
        "command_timeout_seconds": 5,
        "expected_index": 0,
        "expected_total_memory_mib": 12288,
        "idle_used_memory_max_mib": 2048,
        "idle_free_memory_min_mib": 8192,
        "idle_utilization_max_percent": 15,
        "prestart_temperature_max_c": 75,
        "wddm_gui_process_allowlist": list(runner.WDDM_GUI_PROCESS_ALLOWLIST),
        "wddm_gui_inventory_must_be_stable_across_gate": True,
        "insufficient_permissions_pid_resolution_required": True,
        "unknown_or_non_allowlisted_compute_process_forbidden": True,
    }


def _python_runtime_receipt() -> dict[str, object]:
    return {
        "path": r"C:\Python\python.exe",
        "version": "3.11.9",
        "sha256": "e" * 64,
        "dependency_probe": runner.PYTHON_RUNTIME_DEPENDENCY_PROBE,
    }


def _materialize_code_bindings(root: Path, config: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for label, relative in {
        "config": runner.CONFIG_PATH,
        "implementation": runner.IMPLEMENTATION_PATH,
        "runner": runner.SCRIPT_PATH,
        "launcher": runner.LAUNCHER_PATH,
    }.items():
        source = ROOT.joinpath(*Path(relative).parts)
        destination = root.joinpath(*Path(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        result[label] = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert result["config"] == config["_config_sha256"]
    return result


def _gpu_sample(uuid: str, phase: str, ordinal: int) -> dict[str, object]:
    processes = _gui_processes()
    return {
        "phase": phase,
        "ordinal": ordinal,
        "observed_at_utc": "2026-07-23T00:00:00Z",
        "index": 0,
        "uuid": uuid,
        "name": "NVIDIA GeForce RTX 3080 Ti",
        "driver_model": "WDDM",
        "memory_total_mib": 12288,
        "memory_used_mib": 1373,
        "memory_free_mib": 10915,
        "utilization_percent": 5,
        "temperature_c": 55,
        "selected_gpu_compute_process_count": len(processes),
        "compute_inventory_sha256": _compute_inventory_sha256(),
        "compute_processes": processes,
        "wddm_desktop_baseline_tolerated": True,
    }


def test_config_locks_q_only_freshness_and_private_tail() -> None:
    config = _config()

    assert config["dataset"]["roles"] == list(runner.ROLES)
    assert config["dataset"]["concurrency"] == 1
    assert config["lora"] == {
        "profile": "q_only",
        "target_modules": ["q_proj"],
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "bias": "none",
        "expected_trainable_parameters_per_expert": 226_304,
        "o_proj_allowed": False,
    }
    assert config["training"]["sequence_length"] == 768
    assert config["training"]["truncation"] is False
    assert config["training"]["smoke_steps"] == 2
    assert config["training"]["full_steps"] == 160
    assert config["training"]["smoke_and_full_fresh_objects"] is True
    assert config["training"]["smoke_checkpoint_consumed_by_full"] is False
    assert config["training"]["resume"] is False
    assert config["training"]["optimizer"] == "adamw8bit"
    assert config["training"]["optimizer_library"] == "bitsandbytes"
    assert config["training"]["bitsandbytes_version"] == "0.48.2"
    assert config["training"]["optimizer_state_bits"] == 8
    assert config["training"]["compatibility_optim_bits_argument"] == 32
    assert config["training"]["min_8bit_size"] == 4096
    assert config["training"]["percentile_clipping"] == 100
    assert config["training"]["block_wise"] is True
    assert config["training"]["is_paged"] is False
    assert config["training"]["amsgrad"] is False
    assert config["gpu_policy"]["torch_peak_allocated_max_mib"] == 11264
    assert config["gpu_policy"]["torch_peak_reserved_max_mib"] == 11264
    assert config["training"]["adapter_effect_gate"] == {
        "view": "first_train_record_first_supervised_next_token_v1",
        "comparison": "enabled_vs_disable_adapter_after_training",
        "require_finite": True,
        "require_max_abs_gt_zero": True,
        "sample_body_included": False,
        "token_ids_included": False,
    }
    assert tuple(config["gpu_policy"]["wddm_gui_process_allowlist"]) == (
        runner.WDDM_GUI_PROCESS_ALLOWLIST
    )
    assert config["gpu_policy"]["expected_gpu_uuid"] == os.environ.get(
        "ANCHOR_GEMMA_GPU_UUID", "UNBOUND"
    )
    assert config["kv_runtime_boundary"] == {
        "shared_prefix_adapter_state": "off",
        "shared_prefix_read_only": True,
        "identical_ordered_prefix_lineage_only": True,
        "expert_activation": "q_only",
        "expert_private_tail_append_only": True,
        "private_tail_includes_post_activation_prompt_and_generated_tokens": True,
        "private_tail_cross_expert_reuse": False,
        "committed_text_reencoded_for_next_shared_context": True,
        "full_generation_kv_shared_claimed": False,
        "normal_in_stack_q_lora_exact_kv_sharing_claimed": False,
        "token_level_moe_claimed": False,
        "runtime_private_tail_materialized": False,
    }


def test_runtime_gpu_gate_allows_only_own_python_and_frozen_wddm_gui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(_config())
    gpu_uuid = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    config["gpu_policy"]["expected_gpu_uuid"] = gpu_uuid
    calls = 0

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if command[1].startswith("--query-gpu="):
            stdout = f"0, {gpu_uuid}, WDDM, 12288, 1373, 10915, 5, 55\n"
        else:
            stdout = (
                f'{gpu_uuid},999,"C:\\Python\\python.exe",[N/A]\n'
                f'{gpu_uuid},4304,"C:\\Windows\\System32\\dwm.exe",[N/A]\n'
            )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    observed = runner._query_runtime_gpu(config, allow_pid=999)

    assert calls == 2
    assert observed["uuid"] == gpu_uuid
    assert observed["allowlisted_wddm_gui_processes"] == [
        {"pid": 4304, "process_name": "dwm.exe"}
    ]


def test_runtime_gpu_gate_rejects_foreign_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(_config())
    gpu_uuid = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    config["gpu_policy"]["expected_gpu_uuid"] = gpu_uuid

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if command[1].startswith("--query-gpu="):
            stdout = f"0, {gpu_uuid}, WDDM, 12288, 1373, 10915, 5, 55\n"
        else:
            stdout = f'{gpu_uuid},777,"C:\\Python\\python.exe",[N/A]\n'
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="foreign_compute_process_detected",
    ):
        runner._query_runtime_gpu(config, allow_pid=999)


def test_adamw8bit_state_must_be_uint8_cuda() -> None:
    class FakeParameter:
        def __init__(self, elements: int) -> None:
            self.elements = elements

        def numel(self) -> int:
            return self.elements

    class FakeTensor:
        dtype = "uint8"
        device = SimpleNamespace(type="cuda")

        def __init__(self, elements: int) -> None:
            self.elements = elements

        def numel(self) -> int:
            return self.elements

    parameters = [FakeParameter(4096), FakeParameter(4608)]
    optimizer = SimpleNamespace(
        state={
            parameter: {
                "state1": FakeTensor(parameter.numel()),
                "state2": FakeTensor(parameter.numel()),
            }
            for parameter in parameters
        }
    )
    observed = runner._validate_adamw8bit_state(
        optimizer,
        parameters,
        torch=SimpleNamespace(uint8="uint8"),
        bitsandbytes_version="0.48.2",
    )

    assert observed == {
        "backend": "bitsandbytes",
        "class": "bitsandbytes.optim.AdamW8bit",
        "package_version": "0.48.2",
        "optimizer_state_bits": 8,
        "compatibility_optim_bits_argument": 32,
        "min_8bit_size": 4096,
        "percentile_clipping": 100,
        "block_wise": True,
        "is_paged": False,
        "amsgrad": False,
        "parameter_tensors": 2,
        "state_tensors": 4,
        "state_elements": 17_408,
        "state_dtype": "uint8",
        "state_device": "cuda",
    }

    optimizer.state[parameters[0]]["state1"].dtype = "float32"
    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="adamw8bit_state_not_uint8_cuda",
    ):
        runner._validate_adamw8bit_state(
            optimizer,
            parameters,
            torch=SimpleNamespace(uint8="uint8"),
            bitsandbytes_version="0.48.2",
        )


def test_runtime_gpu_gate_allows_netease_gameviewer_server_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(_config())
    gpu_uuid = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    config["gpu_policy"]["expected_gpu_uuid"] = gpu_uuid

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if command[0].casefold().endswith("tasklist.exe"):
            return SimpleNamespace(
                returncode=0,
                stdout=('"GameViewerServer.exe","8852","Console","1","10,000 K"\n'),
                stderr="",
            )
        if command[1].startswith("--query-gpu="):
            stdout = f"0, {gpu_uuid}, WDDM, 12288, 988, 11300, 13, 66\n"
        else:
            stdout = f"{gpu_uuid},8852,[Insufficient Permissions],[N/A]\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    observed = runner._query_runtime_gpu(config, allow_pid=999)

    assert observed["allowlisted_wddm_gui_processes"] == [
        {"pid": 8852, "process_name": "gameviewerserver.exe"}
    ]


@pytest.mark.parametrize(
    "foreign_name",
    ("GameViewer.exe", "GameViewerService.exe", "GameViewerServerHelper.exe"),
)
def test_runtime_gpu_gate_rejects_other_gameviewer_processes(
    monkeypatch: pytest.MonkeyPatch,
    foreign_name: str,
) -> None:
    config = copy.deepcopy(_config())
    gpu_uuid = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    config["gpu_policy"]["expected_gpu_uuid"] = gpu_uuid

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        stdout = (
            f"0, {gpu_uuid}, WDDM, 12288, 988, 11300, 13, 66\n"
            if command[1].startswith("--query-gpu=")
            else f'{gpu_uuid},8852,"C:\\Remote\\{foreign_name}",[N/A]\n'
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="foreign_compute_process_detected",
    ):
        runner._query_runtime_gpu(config, allow_pid=999)


def test_runtime_gpu_gate_accepts_no_running_processes_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(_config())
    gpu_uuid = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    config["gpu_policy"]["expected_gpu_uuid"] = gpu_uuid

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        stdout = (
            f"0, {gpu_uuid}, WDDM, 12288, 1373, 10915, 5, 55\n"
            if command[1].startswith("--query-gpu=")
            else "No running processes found\n"
        )
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    observed = runner._query_runtime_gpu(config, allow_pid=999)

    assert observed["allowlisted_wddm_gui_processes"] == []


def test_runtime_permission_denied_name_resolution_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='"dwm.exe","4304","Console","1","10,000 K"\n',
            stderr="",
        ),
    )
    assert (
        runner._runtime_process_basename(
            "Insufficient Permissions",
            pid=4304,
            timeout=5,
        )
        == "dwm.exe"
    )

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="INFO: No tasks are running which match the specified criteria.\n",
            stderr="",
        ),
    )
    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="runtime_compute_permission_pid_resolution_failed",
    ):
        runner._runtime_process_basename(
            "Insufficient Permissions",
            pid=4304,
            timeout=5,
        )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("lora", "target_modules", ["q_proj", "o_proj"]),
        ("training", "sequence_length", 512),
        ("training", "truncation", True),
        ("training", "smoke_checkpoint_consumed_by_full", True),
        ("dataset", "concurrency", 2),
        ("kv_runtime_boundary", "shared_prefix_adapter_state", "on"),
        ("kv_runtime_boundary", "private_tail_cross_expert_reuse", True),
        ("kv_runtime_boundary", "expert_private_tail_append_only", False),
        ("kv_runtime_boundary", "runtime_private_tail_materialized", True),
        ("claims", "formal", True),
    ],
)
def test_contract_drift_fails_closed(section: str, key: str, value: object) -> None:
    config = _config()
    config[section][key] = value

    with pytest.raises(Exception):
        runner.validate_config(config)


def test_adapter_effect_gate_drift_fails_closed() -> None:
    config = _config()
    config["training"]["adapter_effect_gate"]["require_max_abs_gt_zero"] = False

    with pytest.raises(Exception):
        runner.validate_config(config)


class _MockAdapterModel:
    def __init__(self, torch: object, effect: float) -> None:
        self.torch = torch
        self.effect = effect
        self.disabled = False
        self.disable_calls = 0

    def eval(self) -> "_MockAdapterModel":
        return self

    @contextmanager
    def disable_adapter(self):
        self.disable_calls += 1
        previous = self.disabled
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = previous

    def __call__(self, **_kwargs: object) -> SimpleNamespace:
        logits = self.torch.zeros((1, 3, 4), dtype=self.torch.float32)
        if not self.disabled:
            logits[0, 1, 2] = self.effect
        return SimpleNamespace(logits=logits)


class _MockShapeAdapterModel(_MockAdapterModel):
    def __init__(
        self,
        torch: object,
        enabled_shape: tuple[int, int, int],
        disabled_shape: tuple[int, int, int],
    ) -> None:
        super().__init__(torch, 0.25)
        self.enabled_shape = enabled_shape
        self.disabled_shape = disabled_shape

    def __call__(self, **_kwargs: object) -> SimpleNamespace:
        shape = self.disabled_shape if self.disabled else self.enabled_shape
        return SimpleNamespace(logits=self.torch.zeros(shape, dtype=self.torch.float32))


def test_enabled_vs_disabled_next_token_effect_is_measured() -> None:
    torch = pytest.importorskip("torch")
    model = _MockAdapterModel(torch, 0.25)
    batch = {
        "input_ids": torch.zeros((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }

    result = runner._enabled_vs_disabled_next_token_effect(
        model,
        batch,
        1,
        torch=torch,
    )

    assert result == {
        "finite": True,
        "max_abs": 0.25,
        "mean_abs": 0.0625,
        "vocabulary_logits": 4,
    }
    assert model.disable_calls == 1
    assert model.disabled is False


@pytest.mark.parametrize(
    ("enabled_shape", "disabled_shape", "error"),
    [
        (
            (1, 4, 4),
            (1, 4, 4),
            "adapter_effect_enabled_logits_shape_invalid",
        ),
        (
            (1, 3, 0),
            (1, 3, 0),
            "adapter_effect_enabled_logits_shape_invalid",
        ),
        (
            (1, 3, 4),
            (1, 2, 4),
            "adapter_effect_disabled_logits_shape_invalid",
        ),
    ],
)
def test_enabled_vs_disabled_next_token_effect_rejects_shape_drift(
    enabled_shape: tuple[int, int, int],
    disabled_shape: tuple[int, int, int],
    error: str,
) -> None:
    torch = pytest.importorskip("torch")
    model = _MockShapeAdapterModel(torch, enabled_shape, disabled_shape)
    batch = {
        "input_ids": torch.zeros((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }

    with pytest.raises(runner.GemmaFiveRoleError, match=error):
        runner._enabled_vs_disabled_next_token_effect(
            model,
            batch,
            1,
            torch=torch,
        )


def test_adapter_effect_view_stops_before_first_supervised_target() -> None:
    serialized = SimpleNamespace(
        input_ids=(10, 11, 12, 13, 14),
        labels=(-100, -100, 12, 13, 14),
    )

    view = runner._adapter_effect_prefix_view(serialized)

    assert view == {
        "input_prefix": (10, 11),
        "prediction_position": 1,
        "full_sequence_tokens": 5,
    }


@pytest.mark.parametrize(
    "serialized",
    [
        SimpleNamespace(input_ids=(10, 11), labels=(-100,)),
        SimpleNamespace(input_ids=(10, 11), labels=(10, 11)),
        SimpleNamespace(input_ids=(10, 11), labels=(-100, -100)),
    ],
)
def test_adapter_effect_view_rejects_invalid_supervision_boundary(
    serialized: SimpleNamespace,
) -> None:
    with pytest.raises(runner.GemmaFiveRoleError):
        runner._adapter_effect_prefix_view(serialized)


@pytest.mark.parametrize(
    ("effect", "error"),
    [
        (0.0, "adapter_output_effect_absent_or_invalid"),
        (float("nan"), "adapter_effect_nonfinite_logits"),
    ],
)
def test_enabled_vs_disabled_next_token_effect_fails_closed(
    effect: float,
    error: str,
) -> None:
    torch = pytest.importorskip("torch")
    model = _MockAdapterModel(torch, effect)
    batch = {
        "input_ids": torch.zeros((1, 3), dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }

    with pytest.raises(runner.GemmaFiveRoleError, match=error):
        runner._enabled_vs_disabled_next_token_effect(
            model,
            batch,
            1,
            torch=torch,
        )


def test_smoke_and_full_phase_receipts_bind_adapter_output_effect() -> None:
    source = (ROOT / runner.IMPLEMENTATION_PATH).read_text(encoding="utf-8")

    assert "adapter_effect = _fixed_training_view_adapter_effect(" in source
    assert '"adapter_effect": adapter_effect,' in source
    assert "example.record_id" in source
    assert '"token_ids_included": False' in source
    assert '"sample_body_included": False' in source
    assert '"future_target_suffix_forwarded": False' in source


def test_tokenizer_binding_final_identities_are_locked() -> None:
    config = _config()
    locked = config["bindings"]["tokenizer_binding"]

    assert (
        locked["config_sha256"]
        == "08f07feb07076a924bf0cedbbc722a2fd91ce77e03041f50fc3e3bf6859d1367"
    )
    assert (
        locked["implementation_sha256"]
        == "3194ede7eb78597902e05f7e246bc76b644135a5f6abffef2d5aa4f56cf6b011"
    )
    assert (
        locked["manifest_sha256"]
        == "2cfe0370f4c8f634e2258094e15acdb0571d6c3954bb66d5eb15677c0827f366"
    )
    assert (
        locked["manifest_sidecar_physical_sha256"]
        == "2c0df1565beaca7294e024ebafc1f889033ad1268b70f7146b244125fb15b31e"
    )


def test_build_preflight_is_model_free_and_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    fake_manifest = {
        "status": "passed_model_free_tokenizer_and_label_preflight_training_blocked",
        "identities": {
            "tokenizer_template_special_policy_sha256": config["bindings"][
                "tokenizer_binding"
            ]["tokenizer_template_special_policy_sha256"]
        },
        "summary": {
            "records": 1000,
            "ordered_input_ids_sha256": "1" * 64,
            "ordered_labels_sha256": "2" * 64,
        },
        "tokenization": {
            "sequence_length": 768,
            "observed_minimum_tokens": 449,
            "observed_maximum_tokens": 665,
            "records_over_sequence_length": 0,
            "truncation_used": False,
        },
    }
    datasets = {
        role: SimpleNamespace(train=tuple(range(160)), eval_proxy=tuple(range(40)))
        for role in runner.ROLES
    }
    monkeypatch.setattr(runner, "_authenticate_bound_files", lambda _config: {})
    monkeypatch.setattr(
        runner.binding, "audit_manifest", lambda *_args, **_kwargs: fake_manifest
    )
    monkeypatch.setattr(
        runner.budget,
        "audit_contract",
        lambda _root: {
            "status": "metadata_budget_ready_real_run_blocked",
            "base_parameters": 999_885_952,
            "training_authorized": False,
        },
    )
    monkeypatch.setattr(runner, "_binding_config", lambda _config: {})
    monkeypatch.setattr(
        runner.binding,
        "load_all_role_datasets",
        lambda _config: datasets,
    )

    report = runner.build_preflight(config)

    assert report["status"] == (
        "passed_model_free_ready_for_explicit_controlled_proxy_execute"
    )
    assert report["execution_plan"] == {
        "roles": list(runner.ROLES),
        "concurrency": 1,
        "phases_per_role": ["smoke", "full"],
        "steps": {"smoke": 2, "full": 160},
        "fresh_base_objects": 10,
        "fresh_adapters": 10,
        "single_private_authenticated_model_snapshot": True,
        "smoke_checkpoint_consumed_by_full": False,
        "resume": False,
        "optimizer": runner.OPTIMIZER_RUNTIME_CONTRACT,
    }
    assert report["claims"]["model_loaded"] is False
    assert report["claims"]["gpu_requested"] is False
    assert report["claims"]["runtime_private_tail_materialized"] is False
    assert report["kv_runtime_boundary"] == runner.KV_RUNTIME_BOUNDARY
    assert report["audit"] == {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 0,
        "gpu_requests": 0,
        "protected_body_reads": 0,
    }


def test_execute_receipts_bind_lock_gpu_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(_config())
    config["gpu_policy"]["expected_gpu_uuid"] = (
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    )
    root = tmp_path / "repo"
    code = _materialize_code_bindings(root, config)
    lock = root / "runs" / "formal-v3-training.lock"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"held")
    monkeypatch.setattr(runner, "_root", lambda: root)
    claims = {
        "diagnostic_only": True,
        "proxy_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "quality_claimed": False,
        "generalization_claimed": False,
    }
    common = {
        "schema_version": ("anchor.gemma3-1b-it-five-role-qonly-execution-lease.v1"),
        "status": "passed",
        "run_id": "test-run",
        "canonical_lock": "runs/formal-v3-training.lock",
        "canonical_lock_sha256": "f" * 64,
        "canonical_lock_held_at_publish": True,
        "launcher_pid": os.getppid(),
        "expected_gpu_index": 0,
        "expected_gpu_uuid": config["gpu_policy"]["expected_gpu_uuid"],
        "roles": list(runner.ROLES),
        "smoke_steps": 2,
        "full_steps": 160,
        "concurrency": 1,
        "config_sha256": config["_config_sha256"],
        "implementation_sha256": code["implementation"],
        "runner_script_sha256": code["runner"],
        "launcher_sha256": code["launcher"],
        "wddm_gui_process_allowlist_sha256": _canonical_sha256(
            list(runner.WDDM_GUI_PROCESS_ALLOWLIST)
        ),
        "compute_inventory_sha256": _compute_inventory_sha256(),
        "compute_processes": _gui_processes(),
        "python_runtime": _python_runtime_receipt(),
        "fresh_base_per_phase": True,
        "fresh_adapter_per_phase": True,
        "resume_allowed": False,
        "kv_runtime_boundary": dict(runner.KV_RUNTIME_BOUNDARY),
        "claims": claims,
    }
    lease_path = root / "lease.json"
    lease_sha = _write_receipt(lease_path, common)
    pre_samples = [
        _gpu_sample(config["gpu_policy"]["expected_gpu_uuid"], "pre_lock", ordinal)
        for ordinal in range(1, 4)
    ]
    post_samples = [
        _gpu_sample(config["gpu_policy"]["expected_gpu_uuid"], "post_lock", ordinal)
        for ordinal in range(1, 4)
    ]
    attestation = {
        **common,
        "schema_version": ("anchor.gemma3-1b-it-five-role-qonly-gpu-attestation.v1"),
        "lease_receipt_sha256": lease_sha,
        "gpu_policy": _attested_gpu_policy(),
        "sample_count": 3,
        "launcher": {
            "path": runner.LAUNCHER_PATH,
            "sha256": code["launcher"],
        },
        "runner": {"path": runner.SCRIPT_PATH, "sha256": code["runner"]},
        "implementation": {
            "path": runner.IMPLEMENTATION_PATH,
            "sha256": code["implementation"],
        },
        "config": {"path": runner.CONFIG_PATH, "sha256": code["config"]},
        "lock": {
            "path": "runs/formal-v3-training.lock",
            "content_sha256": "f" * 64,
            "file_mode": "CreateNew",
            "file_share": "None",
            "delete_on_close": True,
            "held_for_entire_orchestrator": True,
        },
        "compute_processes": _gui_processes(),
        "execution_plan": {
            "gpu_index": 0,
            "concurrency": 1,
            "roles": list(runner.ROLES),
            "smoke_steps_per_role": 2,
            "full_steps_per_role": 160,
            "phase_order": ["smoke", "full"],
            "fresh_base_per_phase": True,
            "fresh_adapter_per_phase": True,
            "resume_allowed": False,
        },
        "pre_lock_samples": pre_samples,
        "post_lock_samples": post_samples,
    }
    attestation_path = root / "attestation.json"
    attestation_sha = _write_receipt(attestation_path, attestation)

    lease, observed_attestation, snapshots = runner._validate_launch_receipts(
        config,
        run_id="test-run",
        lease_path=lease_path,
        lease_sha256=lease_sha,
        gpu_attestation_path=attestation_path,
        gpu_attestation_sha256=attestation_sha,
    )

    assert lease["run_id"] == "test-run"
    assert observed_attestation["gpu_policy"]["sample_count"] == 3
    assert lease["kv_runtime_boundary"] == runner.KV_RUNTIME_BOUNDARY
    assert observed_attestation["kv_runtime_boundary"] == runner.KV_RUNTIME_BOUNDARY
    assert snapshots[0].sha256 == lease_sha
    assert snapshots[1].sha256 == attestation_sha

    attestation["python_runtime"]["dependency_probe"] = "bitsandbytes==0.48.1"
    tampered_runtime_sha = _write_receipt(attestation_path, attestation)
    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="python_runtime_cross_binding_failed",
    ):
        runner._validate_launch_receipts(
            config,
            run_id="test-run",
            lease_path=lease_path,
            lease_sha256=lease_sha,
            gpu_attestation_path=attestation_path,
            gpu_attestation_sha256=tampered_runtime_sha,
        )
    attestation["python_runtime"] = _python_runtime_receipt()

    attestation["kv_runtime_boundary"]["private_tail_cross_expert_reuse"] = True
    tampered_sha = _write_receipt(attestation_path, attestation)
    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="launcher_non_authorizing_claims_drift",
    ):
        runner._validate_launch_receipts(
            config,
            run_id="test-run",
            lease_path=lease_path,
            lease_sha256=lease_sha,
            gpu_attestation_path=attestation_path,
            gpu_attestation_sha256=tampered_sha,
        )

    attestation["kv_runtime_boundary"]["private_tail_cross_expert_reuse"] = False
    attestation["pre_lock_samples"][0]["memory_used_mib"] = -1
    invalid_telemetry_sha = _write_receipt(attestation_path, attestation)
    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="gpu_attestation_gpu_identity_invalid",
    ):
        runner._validate_launch_receipts(
            config,
            run_id="test-run",
            lease_path=lease_path,
            lease_sha256=lease_sha,
            gpu_attestation_path=attestation_path,
            gpu_attestation_sha256=invalid_telemetry_sha,
        )


def test_wddm_attested_inventory_requires_canonical_pid_order() -> None:
    canonical = [
        {
            "pid": 4304,
            "process_name": "dwm.exe",
            "used_gpu_memory_mib": "[N/A]",
            "reported_name_was_permission_denied": False,
            "allowlisted_wddm_gui": True,
        },
        {
            "pid": 16180,
            "process_name": "explorer.exe",
            "used_gpu_memory_mib": "[N/A]",
            "reported_name_was_permission_denied": False,
            "allowlisted_wddm_gui": True,
        },
    ]

    assert runner._validate_attested_wddm_processes(canonical) == tuple(canonical)
    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="wddm_compute_inventory_not_canonical",
    ):
        runner._validate_attested_wddm_processes(list(reversed(canonical)))


def test_prevalidation_failure_publishes_fail_closed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(_config())
    root = tmp_path / "repo"
    monkeypatch.setattr(runner, "_root", lambda: root)
    monkeypatch.setattr(runner, "build_preflight", lambda _config: {})

    def fail_validation(*_args: object, **_kwargs: object) -> object:
        raise runner.GemmaFiveRoleError("wddm_compute_inventory_not_canonical")

    monkeypatch.setattr(runner, "_validate_launch_receipts", fail_validation)

    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="wddm_compute_inventory_not_canonical",
    ):
        runner.execute(
            config,
            run_id="prevalidation-failure",
            lease_path="unused-lease.json",
            lease_sha256="a" * 64,
            gpu_attestation_path="unused-attestation.json",
            gpu_attestation_sha256="b" * 64,
        )

    launch_root = root.joinpath(
        *runner.PurePosixPath(config["output"]["launch_root"]).parts
    )
    receipt_path = launch_root / "prevalidation-failure" / "failure_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt == {
        "schema_version": runner.FAILURE_RECEIPT_VERSION,
        "status": "blocked",
        "run_id": "prevalidation-failure",
        "error_code": "wddm_compute_inventory_not_canonical",
        "role": None,
        "phase": None,
        "sample_content_included": False,
        "automatic_retry": False,
        "claims": {
            "formal": False,
            "training_authorized": False,
            "formal_training_authorized": False,
        },
    }


@pytest.mark.parametrize(
    "run_id",
    ("..", "../escaped", "../../escaped", "C:\\escaped", "/escaped"),
)
def test_invalid_run_id_never_writes_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    config = copy.deepcopy(_config())
    root = tmp_path / "repo"
    monkeypatch.setattr(runner, "_root", lambda: root)
    monkeypatch.setattr(runner, "build_preflight", lambda _config: {})

    with pytest.raises(runner.GemmaFiveRoleError, match="run_id_invalid"):
        runner.execute(
            config,
            run_id=run_id,
            lease_path="unused-lease.json",
            lease_sha256="a" * 64,
            gpu_attestation_path="unused-attestation.json",
            gpu_attestation_sha256="b" * 64,
        )

    assert not root.exists()
    assert not (tmp_path / "escaped").exists()


def test_launcher_guard_requires_current_parent_and_canonical_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    root = tmp_path / "repo"
    lock = root / "runs" / "formal-v3-training.lock"
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"held")
    monkeypatch.setattr(runner, "_root", lambda: root)

    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="launcher_parent_process_not_active",
    ):
        runner._assert_launcher_lock_guard(
            config,
            {"launcher_pid": os.getppid() + 1},
        )

    runner._assert_launcher_lock_guard(
        config,
        {"launcher_pid": os.getppid()},
    )

    lock.unlink()
    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="canonical_gpu_lock_not_held",
    ):
        runner._assert_launcher_lock_guard(
            config,
            {"launcher_pid": os.getppid()},
        )


def test_atomic_publish_rechecks_launcher_lock_and_authenticated_inputs() -> None:
    source = (ROOT / runner.IMPLEMENTATION_PATH).read_text(encoding="utf-8")
    publish = source.index("os.rename(staging, destination)")
    guard = source.rfind("_assert_launcher_lock_guard(config, lease)", 0, publish)
    snapshot_recheck = source.rfind(
        "snapshot.assert_unchanged()",
        0,
        publish,
    )

    assert guard != -1
    assert snapshot_recheck != -1
    assert guard < snapshot_recheck < publish
    assert publish - snapshot_recheck < 500


def test_execute_receipts_reject_conflicting_handoff_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(_config())
    config["gpu_policy"]["expected_gpu_uuid"] = (
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    )
    root = tmp_path / "repo"
    code = _materialize_code_bindings(root, config)
    canonical = root / "runs" / "formal-v3-training.lock"
    conflict = root / "runs" / "distill-train-handoff-v3" / "gpu-job.lock"
    canonical.parent.mkdir(parents=True)
    conflict.parent.mkdir(parents=True)
    canonical.write_bytes(b"held")
    conflict.write_bytes(b"held")
    monkeypatch.setattr(runner, "_root", lambda: root)
    claims = {
        "diagnostic_only": True,
        "proxy_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
        "quality_claimed": False,
        "generalization_claimed": False,
    }
    common = {
        "schema_version": ("anchor.gemma3-1b-it-five-role-qonly-execution-lease.v1"),
        "status": "passed",
        "run_id": "test-run",
        "canonical_lock": "runs/formal-v3-training.lock",
        "canonical_lock_sha256": "f" * 64,
        "canonical_lock_held_at_publish": True,
        "launcher_pid": os.getppid(),
        "expected_gpu_index": 0,
        "expected_gpu_uuid": config["gpu_policy"]["expected_gpu_uuid"],
        "roles": list(runner.ROLES),
        "smoke_steps": 2,
        "full_steps": 160,
        "concurrency": 1,
        "config_sha256": config["_config_sha256"],
        "implementation_sha256": code["implementation"],
        "runner_script_sha256": code["runner"],
        "launcher_sha256": code["launcher"],
        "wddm_gui_process_allowlist_sha256": _canonical_sha256(
            list(runner.WDDM_GUI_PROCESS_ALLOWLIST)
        ),
        "compute_inventory_sha256": _compute_inventory_sha256(),
        "compute_processes": _gui_processes(),
        "python_runtime": _python_runtime_receipt(),
        "fresh_base_per_phase": True,
        "fresh_adapter_per_phase": True,
        "resume_allowed": False,
        "kv_runtime_boundary": dict(runner.KV_RUNTIME_BOUNDARY),
        "claims": claims,
    }
    lease_path = root / "lease.json"
    lease_sha = _write_receipt(lease_path, common)
    pre_samples = [
        _gpu_sample(config["gpu_policy"]["expected_gpu_uuid"], "pre_lock", ordinal)
        for ordinal in range(1, 4)
    ]
    post_samples = [
        _gpu_sample(config["gpu_policy"]["expected_gpu_uuid"], "post_lock", ordinal)
        for ordinal in range(1, 4)
    ]
    attestation_path = root / "attestation.json"
    attestation_sha = _write_receipt(
        attestation_path,
        {
            **common,
            "schema_version": (
                "anchor.gemma3-1b-it-five-role-qonly-gpu-attestation.v1"
            ),
            "lease_receipt_sha256": lease_sha,
            "gpu_policy": _attested_gpu_policy(),
            "sample_count": 3,
            "launcher": {
                "path": runner.LAUNCHER_PATH,
                "sha256": code["launcher"],
            },
            "runner": {"path": runner.SCRIPT_PATH, "sha256": code["runner"]},
            "implementation": {
                "path": runner.IMPLEMENTATION_PATH,
                "sha256": code["implementation"],
            },
            "config": {"path": runner.CONFIG_PATH, "sha256": code["config"]},
            "lock": {
                "path": "runs/formal-v3-training.lock",
                "content_sha256": "f" * 64,
                "file_mode": "CreateNew",
                "file_share": "None",
                "delete_on_close": True,
                "held_for_entire_orchestrator": True,
            },
            "compute_processes": _gui_processes(),
            "execution_plan": {
                "gpu_index": 0,
                "concurrency": 1,
                "roles": list(runner.ROLES),
                "smoke_steps_per_role": 2,
                "full_steps_per_role": 160,
                "phase_order": ["smoke", "full"],
                "fresh_base_per_phase": True,
                "fresh_adapter_per_phase": True,
                "resume_allowed": False,
            },
            "pre_lock_samples": pre_samples,
            "post_lock_samples": post_samples,
        },
    )

    with pytest.raises(
        runner.GemmaFiveRoleError,
        match="conflicting_handoff_gpu_lock_present",
    ):
        runner._validate_launch_receipts(
            config,
            run_id="test-run",
            lease_path=lease_path,
            lease_sha256=lease_sha,
            gpu_attestation_path=attestation_path,
            gpu_attestation_sha256=attestation_sha,
        )


def test_authenticated_copy_detects_source_or_destination_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"authenticated model bytes")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    runner._copy_authenticated_file(
        source,
        destination,
        expected_bytes=source.stat().st_size,
        expected_sha256=expected,
    )

    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(runner.GemmaFiveRoleError):
        runner._copy_authenticated_file(
            source,
            tmp_path / "bad.bin",
            expected_bytes=source.stat().st_size,
            expected_sha256="f" * 64,
        )


def test_smoke_and_full_must_be_fresh_and_share_only_initialization() -> None:
    smoke = {
        "initial_adapter_sha256": "a" * 64,
        "base_hash_before": "b" * 64,
        "train_record_order_sha256": "c" * 64,
        "optimizer_steps": 2,
        "smoke_checkpoint_consumed": False,
    }
    full = {
        "initial_adapter_sha256": "a" * 64,
        "base_hash_before": "b" * 64,
        "train_record_order_sha256": "d" * 64,
        "optimizer_steps": 160,
        "smoke_checkpoint_consumed": False,
    }

    runner._assert_smoke_full_freshness(smoke, full)
    full["smoke_checkpoint_consumed"] = True
    with pytest.raises(
        runner.GemmaFiveRoleError, match="smoke_full_freshness_gate_failed"
    ):
        runner._assert_smoke_full_freshness(smoke, full)


def test_main_execute_without_launcher_inputs_fails_before_model_or_gpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = runner.main(["--config", runner.CONFIG_PATH, "--execute"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["status"] == "blocked"
    assert output["error_code"] == "execute_missing_required_launcher_inputs"
    assert output["sample_content_included"] is False
    assert output["automatic_retry"] is False
