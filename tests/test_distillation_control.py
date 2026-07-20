from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
OBSERVABILITY = ROOT / "scripts" / "observability"
sys.path.insert(0, str(OBSERVABILITY))

import distillation_control as control  # noqa: E402
import distillation_dashboard as dashboard  # noqa: E402


SENTINEL = "sk-control-sentinel-never-persist"


class FakeProcess:
    next_pid = 41000

    def __init__(self) -> None:
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.done = threading.Event()
        self.return_code: int | None = None
        self.signals: list[int] = []

    def wait(self, timeout: float | None = None) -> int:
        if not self.done.wait(timeout):
            raise subprocess.TimeoutExpired(["fake-child"], timeout)
        assert self.return_code is not None
        return self.return_code

    def poll(self) -> int | None:
        return self.return_code if self.done.is_set() else None

    def send_signal(self, signal_value: int) -> None:
        self.signals.append(signal_value)

    def complete(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.done.set()


class FakeFactory:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[list[str], dict[str, object], FakeProcess]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> FakeProcess:
        if self.fail:
            raise OSError("synthetic spawn failure")
        process = FakeProcess()
        captured = dict(kwargs)
        environment = captured.get("env")
        if isinstance(environment, dict):
            captured["env"] = dict(environment)
        self.calls.append((list(argv), captured, process))
        return process


class FakeSignaler:
    def __init__(
        self, *, graceful_exit: bool = True, terminate_exit: bool = True
    ) -> None:
        self.graceful_exit = graceful_exit
        self.terminate_exit = terminate_exit
        self.actions: list[str] = []

    def popen_group_kwargs(self) -> dict[str, object]:
        return {"start_new_session": True}

    def graceful(self, process: FakeProcess) -> None:
        self.actions.append("graceful")
        if self.graceful_exit:
            process.complete(130)

    def terminate(self, process: FakeProcess) -> None:
        self.actions.append("terminate")
        if self.terminate_exit:
            process.complete(143)

    def kill(self, process: FakeProcess) -> None:
        self.actions.append("kill")
        process.complete(137)


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "configs" / "data").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "skills").mkdir()
    (root / "skills" / "planner.md").write_text("fixture SOP", encoding="utf-8")
    (root / "configs" / "data" / "task_cards.v1.yaml").write_text(
        "schema_version: fixture\n", encoding="utf-8"
    )
    (root / "configs" / "data" / "base.yaml").write_text(
        "\n".join(
            [
                "provider: custom-openai-responses",
                "protocol: openai_responses",
                "base_url: https://fixture.invalid/v1",
                "fallback_protocol: anthropic",
                "fallback_base_url: https://old-secret-destination.invalid/v1",
                "model: fixture-model",
                "force_model: true",
                "discover_models: false",
                "api_key_env: FIXTURE_KEY",
                "sop_dir: skills",
                "output_dir: data/reserved-base",
                "task_card_config: configs/data/task_cards.v1.yaml",
                "seed_index_offset: 0",
                "concurrency_stages: [1]",
                "stage_seed_counts: [2]",
                "raw_collection_target: 2",
                "max_requests: 20",
                "max_output_tokens_total: 2000",
                "cooldown_seconds: 60",
                "cooldown_poll_seconds: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return root


def _payload(*, output: str = "data/control_shards/run-a", offset: int = 100) -> dict:
    return {
        "base_config": "configs/data/base.yaml",
        "output_dir": output,
        "seed_index_offset": offset,
        "concurrency": 4,
        "base_url": "https://provider.invalid/api/v1",
        "protocol": "openai_responses",
        "api_key": SENTINEL,
        "model": "teacher-model-v1",
        "force_model": True,
        "task_card_config": "configs/data/task_cards.v1.yaml",
        "timeout_seconds": 30,
        "max_retries": 2,
        "reconnect_attempts": 1,
        "reconnect_backoff_seconds": 0.1,
        "cooldown_seconds": 300,
        "cooldown_poll_seconds": 5,
        "wall_clock_deadline_seconds": 120,
        "max_requests": 500,
        "max_output_tokens_total": 5_000_000,
        "discovery_timeout_seconds": 10,
        "wait_cooldown": True,
        "network_route": "direct",
    }


def _manager(
    root: Path,
    factory: FakeFactory | None = None,
    signaler: FakeSignaler | None = None,
    attached: list[tuple[str, Path]] | None = None,
) -> tuple[control.ControlPlane, FakeFactory, FakeSignaler]:
    child_factory = factory or FakeFactory()
    child_signaler = signaler or FakeSignaler()
    callback = (
        (lambda label, path: attached.append((label, path)))
        if attached is not None
        else None
    )
    manager = control.ControlPlane(
        root,
        popen_factory=child_factory,
        command_builder=lambda generated: [
            sys.executable,
            "fixture-child.py",
            "--config",
            str(generated.effective_config),
        ],
        signaler=child_signaler,
        attach_callback=callback,
        graceful_timeout_seconds=0.02,
        terminate_timeout_seconds=0.02,
    )
    return manager, child_factory, child_signaler


def _wait_state(manager: control.ControlPlane, expected: str) -> dict[str, object]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        public = manager.public()
        if public["process_state"] == expected:
            return public
        time.sleep(0.01)
    raise AssertionError(f"state did not become {expected}: {manager.public()}")


def _disk_bytes(root: Path) -> bytes:
    chunks: list[bytes] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        chunks.append(path.read_bytes())
    return b"\n".join(chunks)


def _write_formal_status(
    root: Path,
    *,
    state: str = "running",
    updated_at: datetime | None = None,
    completed: int = 3,
    rate: float = 2.0,
    eta: float | None = 300.0,
) -> Path:
    path = (
        root
        / "artifacts"
        / "swebench"
        / "full-bank-live-v1"
        / "status.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "anchor.swebench-ccswitch-status.v2",
        "control_run_id": "anchor-test-run",
        "checkpoint_id": "1" * 64,
        "config_sha256": "2" * 64,
        "execution_lock_sha256": "3" * 64,
        "resume_mode": True,
        "state": state,
        "submitted_tasks": completed + (1 if state in {"running", "starting"} else 0),
        "active_tasks": 1 if state in {"running", "starting"} else 0,
        "completed_tasks": completed,
        "expected_tasks": 19008,
        "counts": {"completed": completed, "blocked": 0, "failed": 0},
        "stage_counts": {
            "planner": completed,
            "tool_policy": completed,
            "domain_builder": completed,
            "domain_review": completed,
            "security": completed,
        },
        "failure_counts": {},
        "requests": {
            "provider_requests": 8,
            "provider_successes": 7,
            "provider_failures": 1,
            "retry_attempts": 1,
        },
        "request_failure_counts": {"http_499": 1},
        "tokens": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "cached_input_tokens": 0,
        },
        "elapsed_seconds": 60.0,
        "tasks_per_minute": rate,
        "provider_output_tokens_per_second": 0.5,
        "eta_seconds": eta,
        "updated_at": (updated_at or datetime.now(timezone.utc)).isoformat(),
        "last_error_code": None,
        "content_free": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_start_generates_secret_free_config_and_fixed_child_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workspace(tmp_path)
    attached: list[tuple[str, Path]] = []
    monkeypatch.setenv("OTHER_SECRET_TOKEN", "must-not-enter-child")
    manager, factory, _ = _manager(root, attached=attached)

    public = manager.start_new(_payload())
    argv, kwargs, process = factory.calls[0]
    child_env = kwargs["env"]

    assert public["process_state"] == "running"
    assert public["credential_loaded"] is True
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == str(root.resolve())
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert SENTINEL not in json.dumps(argv)
    assert isinstance(child_env, dict)
    assert child_env[control.CONTROL_KEY_ENV] == SENTINEL
    assert "OTHER_SECRET_TOKEN" not in child_env
    assert attached and attached[0][0] == "run-a"
    assert SENTINEL.encode() not in _disk_bytes(root)
    assert SENTINEL not in json.dumps(public)

    manifest = json.loads(
        (
            root
            / "runs"
            / "control-plane"
            / str(public["run_id"])
            / "control-manifest.json"
        ).read_text(encoding="utf-8")
    )
    effective = (
        root
        / "runs"
        / "control-plane"
        / str(public["run_id"])
        / "effective-config.yaml"
    ).read_text(encoding="utf-8")
    assert manifest["credential_persisted"] is False
    assert "api_key:" not in effective
    assert f"api_key_env: {control.CONTROL_KEY_ENV}" in effective
    assert "shell: false" in effective
    assert "network_route: direct" in effective
    assert "fallback_protocol" not in effective
    assert "fallback_base_url" not in effective
    assert child_env["NO_PROXY"] == "*"
    assert child_env["no_proxy"] == "*"
    assert "HTTP_PROXY" not in child_env
    assert "HTTPS_PROXY" not in child_env

    process.complete(0)
    terminal = _wait_state(manager, "exited")
    assert terminal["exit_code"] == 0
    assert terminal["credential_loaded"] is False
    assert not (root / "data" / "control_shards" / "run-a" / control.LOCK_NAME).exists()


def test_production_command_is_fixed_argv_list(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    policy = control.WorkspacePolicy(root)
    spec, _, base = control.parse_start_spec(_payload(), policy)
    generated = control.generate_run(policy, spec, base)
    manager = control.ControlPlane(root)

    argv = manager._default_command(generated)

    assert argv[:3] == [sys.executable, "-m", "anchor_mvp.data.automation"]
    assert argv[3:5] == ["--config", str(generated.effective_config)]
    assert argv[-1] == "--wait-cooldown"
    assert SENTINEL not in json.dumps(argv)
    manager.close()


def test_options_list_only_strict_automation_base_configs(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    (root / "configs" / "data" / "swebench_five_stage.example.yaml").write_text(
        "schema_version: swebench\ntasks: []\n", encoding="utf-8"
    )
    (root / "configs" / "data" / "execution_tasks_v0.yaml").write_text(
        "schema_version: execution\ntasks: []\n", encoding="utf-8"
    )
    (root / "configs" / "data" / "almost.yaml").write_text(
        "sop_dir: skills\noutput_dir: data/x\n", encoding="utf-8"
    )

    options = control.ControlPlane(root).options()

    assert options["base_configs"] == [{"id": "configs/data/base.yaml", "valid": True}]
    assert options["formal_route"] == {
        "component_ready": False,
        "live_route_container_reachable": None,
        "reachability_state": "not_probed_by_dashboard",
        "e2e_ready": False,
    }
    assert options["formal_dataset"] == {
        "bank_ready": False,
        "locale_assignment_counts": {},
        "language_routing_only": True,
        "zh_cn_localization_manifest_present": False,
    }
    assert options["formal_execution"]["bundle_present"] is False
    assert options["formal_execution"]["observed_tool_contract_version"] is None
    assert options["formal_execution"]["required_tool_contract_version"] == (
        "anchor.execution-tool-contract.v3"
    )
    assert options["formal_execution"]["ready"] is False
    assert options["formal_execution"]["reason_code"] == (
        "generic_train_execution_contract_not_ready"
    )
    assert options["formal_execution"]["remaining_gates"] == [
        "execution_lock_invalid"
    ]
    assert options["formal_execution"]["not_official_swebench_pass"] is True
    assert (
        options["formal_execution"]["official_evaluation_contract_ready"] is False
    )
    assert options["formal_execution"]["capability_gap"] == (
        "python_repository_validation_not_attested"
    )
    assert options["formal_gates"] == {
        "component_ready": False,
        "bank_ready": False,
        "execution_contract_ready": False,
        "live_start_allowed": False,
        "reason_code": "formal_component_not_ready",
    }
    assert options["limits"]["concurrency_default"] == 1
    assert options["limits"]["concurrency_max"] is None


def test_concurrency_has_no_product_hard_ceiling(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    payload = _payload()
    payload["concurrency"] = 512

    spec, _, _ = control.parse_start_spec(payload, control.WorkspacePolicy(root))

    assert spec.concurrency == 512


def test_formal_live_is_blocked_before_credential_is_retained(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manager, factory, _ = _manager(root)

    with pytest.raises(control.ControlError) as caught:
        manager.start_formal(
            {"api_key": SENTINEL, "concurrency": 1}, resume=False
        )

    assert caught.value.status == 409
    assert caught.value.code == "formal_component_not_ready"
    assert manager.formal_secret.configured is False
    assert factory.calls == []


def test_formal_gate_reason_is_ready_when_live_start_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = control.WorkspacePolicy(_workspace(tmp_path))
    monkeypatch.setattr(
        policy,
        "_formal_route_status",
        lambda: {"component_ready": True},
    )
    monkeypatch.setattr(
        policy,
        "_formal_dataset_status",
        lambda: {"bank_ready": True},
    )
    monkeypatch.setattr(
        policy,
        "_formal_execution_status",
        lambda: {
            "bundle_present": True,
            "ready": True,
            "reason_code": "generic_train_execution_contract_ready",
        },
    )

    gates = policy.options()["formal_gates"]

    assert gates["live_start_allowed"] is True
    assert gates["reason_code"] == "generic_train_execution_contract_ready"


def test_formal_status_keeps_global_official_result_non_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workspace(tmp_path)
    bundle = root / "artifacts/tooling/opencode-patched/bundle-manifest.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text(
        json.dumps(
            {
                "schema_version": "anchor.patched-opencode.bundle.v1",
                "source": {
                    "tool_contract_version": "anchor.execution-tool-contract.v3",
                    "tool_contract": {
                        "version": "anchor.execution-tool-contract.v3"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    coordinator_config = root / "configs/data/swebench_five_stage.ccswitch.yaml"
    coordinator_config.write_text(
        "execution_contract:\n"
        "  attestation: artifacts/attestation.json\n"
        "  lock: configs/tooling/lock.json\n"
        f"  lock_sha256: {'a' * 64}\n",
        encoding="utf-8",
    )
    fake = types.ModuleType("run_swebench_ccswitch")

    class FakeCoordinatorConfig:
        @staticmethod
        def load(path: Path) -> object:
            assert path == coordinator_config
            return object()

    fake.CoordinatorConfig = FakeCoordinatorConfig
    fake._distillation_execution_contract_gate = lambda _config: {
        "mode": "generic_train_repo_base_commit",
        "not_official_swebench_pass": True,
        "ready": True,
        "reason_code": "generic_train_execution_contract_ready",
        "remaining_gates": [],
        "official_evaluation_contract_ready": False,
        "official_evaluation_remaining_gates": [
            "official_testspec_behavior_probe_missing"
        ],
    }
    monkeypatch.setitem(sys.modules, "run_swebench_ccswitch", fake)

    result = control.WorkspacePolicy(root)._formal_execution_status()

    assert result["ready"] is True
    assert result["reason_code"] == "generic_train_execution_contract_ready"
    assert result["not_official_swebench_pass"] is True
    assert result["official_evaluation_contract_ready"] is False
    assert result["official_evaluation_remaining_gates"] == [
        "official_testspec_behavior_probe_missing"
    ]


def test_formal_preflight_is_content_free_and_never_reads_candidate_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workspace(tmp_path)
    candidate = root / "datasets" / "public" / "swebench-full-bank-v1" / "candidate-tasks"
    candidate.mkdir(parents=True)
    candidate.joinpath("tasks-00000.jsonl").write_text(
        '{"problem_statement":"DO-NOT-READ-CANDIDATE-BODY"}\n',
        encoding="utf-8",
    )
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        assert "candidate-tasks" not in path.parts
        assert "candidate-work-orders" not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    report = control.WorkspacePolicy(root).formal_preflight()

    assert report["content_free"] is True
    assert report["provider_requests"] == 0
    assert report["credentials_read"] is False
    assert report["sample_bodies_read"] is False
    assert report["sample_bodies_printed"] is False
    assert report["heldout_files_read"] is False
    assert report["live_started"] is False
    assert report["live_start_allowed"] is False
    assert report["reason_code"] == "formal_component_not_ready"
    assert "DO-NOT-READ-CANDIDATE-BODY" not in json.dumps(report)


def test_formal_runtime_v2_uses_explicit_attempt_local_rate_and_eta(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    _write_formal_status(root, completed=90, rate=2.5, eta=123.0)

    status = control.WorkspacePolicy(root).formal_runtime_status()

    assert status["state"] == "running"
    assert status["tasks_per_minute"] == 2.5
    assert status["eta_seconds"] == 123.0
    assert status["completed_tasks"] == 90
    assert status["request_failure_counts"] == {"http_499": 1}


def test_formal_runtime_starting_is_stale_and_future_timestamp_is_rejected(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    path = _write_formal_status(
        root,
        state="starting",
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=30),
    )
    status = control.WorkspacePolicy(root).formal_runtime_status()
    assert status["state"] == "starting"
    assert status["fresh"] is False

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["updated_at"] = (
        datetime.now(timezone.utc) + timedelta(minutes=1)
    ).isoformat()
    path.write_text(json.dumps(payload), encoding="utf-8")

    future = control.WorkspacePolicy(root).formal_runtime_status()
    assert future["state"] == "invalid_status"
    assert future["available"] is False


def test_formal_status_marks_unbound_or_stale_telemetry_untrusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workspace(tmp_path)
    manager, _, _ = _manager(root)
    manager.formal_gates = {
        "live_start_allowed": True,
        "reason_code": "formal_live_ready",
    }
    runtime = {
        "available": True,
        "state": "running",
        "fresh": True,
        "config_sha256": "2" * 64,
        "execution_lock_sha256": "3" * 64,
        "control_run_id": "old-process",
        "checkpoint_id": "1" * 64,
        "resume_mode": False,
    }
    monkeypatch.setattr(manager.policy, "formal_runtime_status", lambda: dict(runtime))
    monkeypatch.setattr(
        manager.policy,
        "formal_local_binding",
        lambda: {
            "ready": True,
            "config_sha256": "2" * 64,
            "execution_lock_sha256": "3" * 64,
            "status_exists": True,
            "checkpoint_exists": True,
        },
    )

    historical = manager.formal_status()
    assert historical["state"] == "historical_unbound"
    assert historical["telemetry_trusted"] is False

    process = FakeProcess()
    manager.formal_job = control.FormalJob(
        run_id="current-process",
        concurrency=1,
        resume_mode=False,
        config_sha256="2" * 64,
        execution_lock_sha256="3" * 64,
        expected_checkpoint_id=None,
        process_state="running",
        process=process,
    )
    runtime.update({"control_run_id": "current-process", "fresh": False})
    stale = manager.formal_status()
    assert stale["state"] == "stale_status"
    assert stale["telemetry_trusted"] is False
    assert stale["reason_code"] == "formal_status_stale"
    process.complete(130)
    manager.close()


@pytest.mark.parametrize("resume", [False, True])
def test_formal_start_and_continue_use_distinct_fixed_cli_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    root = _workspace(tmp_path)
    script = root / "scripts" / "tooling" / "run_swebench_ccswitch.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fixed fixture entrypoint\n", encoding="utf-8")
    config = root / "configs" / "data" / "swebench_five_stage.ccswitch.yaml"
    config.write_text("schema_version: fixture\n", encoding="utf-8")
    manager, factory, _ = _manager(root)
    ready_options = {
        "formal_gates": {
            "component_ready": True,
            "bank_ready": True,
            "execution_contract_ready": True,
            "live_start_allowed": True,
            "reason_code": "formal_live_ready",
        },
        "formal_execution": {"ready": True},
    }

    def options() -> dict[str, object]:
        manager.formal_gates = dict(ready_options["formal_gates"])
        manager.formal_execution = dict(ready_options["formal_execution"])
        return ready_options

    monkeypatch.setattr(manager, "options", options)
    runtime = {
        "available": resume,
        "state": "stopped_checkpoint_resumable" if resume else "not_started",
        "fresh": True,
        "config_sha256": "2" * 64 if resume else None,
        "execution_lock_sha256": "3" * 64 if resume else None,
        "checkpoint_id": "1" * 64 if resume else None,
        "control_run_id": "previous-attempt" if resume else None,
        "resume_mode": False if resume else None,
    }
    binding = {
        "ready": True,
        "config_sha256": "2" * 64,
        "execution_lock_sha256": "3" * 64,
        "status_exists": resume,
        "checkpoint_exists": resume,
    }
    monkeypatch.setattr(manager.policy, "formal_runtime_status", lambda: dict(runtime))
    monkeypatch.setattr(manager.policy, "formal_local_binding", lambda: dict(binding))

    manager.start_formal(
        {"api_key": SENTINEL, "concurrency": 7, "max_tasks": 16},
        resume=resume,
    )
    assert manager.formal_status()["max_tasks"] == 16
    argv, kwargs, process = factory.calls[0]

    assert argv[:2] == [sys.executable, str(script.resolve())]
    assert argv[2:5] == ["--config", str(config.resolve()), "--confirm-live"]
    assert argv[5] == "--control-run-id"
    assert argv[7:] == ["--concurrency", "7"] + (["--resume"] if resume else []) + [
        "--max-tasks",
        "16",
    ]
    assert kwargs["shell"] is False
    assert SENTINEL not in json.dumps(argv)
    assert kwargs["env"]["ARK_CODING_API_KEY"] == SENTINEL
    with pytest.raises(control.ControlError) as caught:
        manager.clear_credential()
    assert caught.value.code == "active_credential_resident"
    assert manager.formal_secret.configured is True
    process.complete(130)
    manager.close()


def test_zero_work_failed_start_can_only_launch_a_fresh_formal_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _workspace(tmp_path)
    script = root / "scripts" / "tooling" / "run_swebench_ccswitch.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fixed fixture entrypoint\n", encoding="utf-8")
    config = root / "configs" / "data" / "swebench_five_stage.ccswitch.yaml"
    config.write_text("schema_version: fixture\n", encoding="utf-8")
    manager, factory, _ = _manager(root)
    ready_options = {
        "formal_gates": {
            "component_ready": True,
            "bank_ready": True,
            "execution_contract_ready": True,
            "live_start_allowed": True,
            "reason_code": "formal_live_ready",
        },
        "formal_execution": {"ready": True},
    }

    def options() -> dict[str, object]:
        manager.formal_gates = dict(ready_options["formal_gates"])
        manager.formal_execution = dict(ready_options["formal_execution"])
        return ready_options

    monkeypatch.setattr(manager, "options", options)
    runtime = {
        "available": True,
        "state": "failed",
        "fresh": True,
        "config_sha256": "2" * 64,
        "execution_lock_sha256": "3" * 64,
        "checkpoint_id": "1" * 64,
        "control_run_id": "failed-startup",
        "resume_mode": False,
    }
    binding = {
        "ready": True,
        "config_sha256": "2" * 64,
        "execution_lock_sha256": "3" * 64,
        "status_exists": True,
        "checkpoint_exists": False,
        "failed_startup_rearmable": True,
    }
    monkeypatch.setattr(manager.policy, "formal_runtime_status", lambda: dict(runtime))
    monkeypatch.setattr(manager.policy, "formal_local_binding", lambda: dict(binding))

    public = manager.formal_status()
    assert public["can_start"] is True
    assert public["can_continue"] is False
    with pytest.raises(control.ControlError) as caught:
        manager.start_formal(
            {"api_key": SENTINEL, "concurrency": 1, "max_tasks": 1},
            resume=True,
        )
    assert caught.value.code == "formal_resume_binding_invalid"

    manager.start_formal(
        {"api_key": SENTINEL, "concurrency": 1, "max_tasks": 1},
        resume=False,
    )
    argv, _, process = factory.calls[0]
    assert "--resume" not in argv
    assert argv[-2:] == ["--max-tasks", "1"]
    process.complete(130)
    manager.close()


def test_clear_credential_refuses_while_legacy_child_still_holds_it(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    manager, factory, _ = _manager(root)
    manager.start_new(_payload())

    with pytest.raises(control.ControlError) as caught:
        manager.clear_credential()

    assert caught.value.status == 409
    assert caught.value.code == "active_credential_resident"
    assert manager.secret.configured is True
    factory.calls[0][2].complete(130)
    manager.close()


def test_formal_profile_requires_literal_max_reasoning(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    base = root / "configs" / "data" / "base.yaml"
    base.write_text(
        base.read_text(encoding="utf-8")
        + "reasoning_policy:\n  required: true\n  effort: max\n",
        encoding="utf-8",
    )
    policy = control.WorkspacePolicy(root)
    payload = _payload()
    payload.update({"reasoning_enabled": True, "reasoning_effort": "high"})

    with pytest.raises(control.ControlError) as caught:
        control.parse_start_spec(payload, policy)

    assert caught.value.status == 409
    assert caught.value.code == "formal_reasoning_required"

    payload["reasoning_effort"] = "max"
    spec, credential, _ = control.parse_start_spec(payload, policy)
    assert credential == SENTINEL
    assert spec.reasoning_enabled is True
    assert spec.reasoning_effort == "max"


def test_network_route_inherit_is_explicit_and_never_persists_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workspace(tmp_path)
    proxy = "http://proxy-user:proxy-password@127.0.0.1:9911"
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    manager, factory, _ = _manager(root)
    payload = _payload()
    payload["network_route"] = "inherit"

    started = manager.start_new(payload)
    child_env = factory.calls[0][1]["env"]

    assert isinstance(child_env, dict)
    assert child_env["HTTPS_PROXY"] == proxy
    assert proxy.encode() not in _disk_bytes(root)
    effective = (
        root
        / "runs"
        / "control-plane"
        / str(started["run_id"])
        / "effective-config.yaml"
    ).read_text(encoding="utf-8")
    assert "network_route: inherit" in effective
    assert "proxy-user" not in effective
    factory.calls[0][2].complete(0)
    _wait_state(manager, "exited")


def test_options_report_only_content_free_proxy_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _workspace(tmp_path)
    monkeypatch.setattr(
        control,
        "getproxies",
        lambda: {"https": "http://secret-user:secret-pass@proxy.invalid"},
    )

    serialized = json.dumps(control.ControlPlane(root).options())

    assert '"proxy_detected": true' in serialized
    assert "secret-user" not in serialized
    assert "proxy.invalid" not in serialized


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("base_url", "the provider URL", "invalid_base_url"),
        ("base_url", "file:///tmp/model", "invalid_base_url"),
        ("model", "model; calc.exe", "invalid_model"),
        ("output_dir", "../outside", "invalid_output_dir"),
        ("output_dir", "D:/outside", "invalid_output_dir"),
        ("concurrency", 0, "invalid_concurrency"),
        ("seed_index_offset", -1, "invalid_seed_index_offset"),
    ],
)
def test_start_rejects_unsafe_fields(
    tmp_path: Path, field: str, value: object, code: str
) -> None:
    root = _workspace(tmp_path)
    manager, _, _ = _manager(root)
    payload = _payload()
    payload[field] = value

    with pytest.raises(control.ControlError) as captured:
        manager.start_new(payload)

    assert captured.value.code == code
    assert manager.public()["process_state"] == "idle"
    assert SENTINEL.encode() not in _disk_bytes(root)


def test_offset_overlap_and_existing_output_fail_closed(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manager, _, _ = _manager(root)

    with pytest.raises(control.ControlError, match="reserved") as overlap:
        manager.start_new(_payload(offset=1))
    assert overlap.value.code == "offset_conflict"

    existing = root / "data" / "control_shards" / "existing"
    existing.mkdir(parents=True)
    (existing / "sentinel.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(control.ControlError) as collision:
        manager.start_new(_payload(output="data/control_shards/existing", offset=100))
    assert collision.value.code == "new_shard_required"
    assert (existing / "sentinel.txt").read_text(encoding="utf-8") == "preserve"


def test_output_path_rejects_lexical_link_or_windows_junction(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    target = root / "data" / "real-output"
    target.mkdir()
    linked = root / "data" / "linked-output"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.skip("Windows junction creation is unavailable")
    else:
        linked.symlink_to(target, target_is_directory=True)
    try:
        policy = control.WorkspacePolicy(root)
        with pytest.raises(control.ControlError) as rejected:
            policy.output_path("data/linked-output/run-a", must_exist=False)
        assert rejected.value.code == "invalid_output_dir"
    finally:
        if linked.is_symlink():
            linked.unlink()
        else:
            linked.rmdir()


def test_spawn_failure_clears_secret_and_removes_empty_output(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manager, _, _ = _manager(root, factory=FakeFactory(fail=True))

    with pytest.raises(control.ControlError) as captured:
        manager.start_new(_payload())

    assert captured.value.code == "spawn_failed"
    assert manager.public()["credential_loaded"] is False
    assert not (root / "data" / "control_shards" / "run-a").exists()
    assert SENTINEL.encode() not in _disk_bytes(root)


def test_spawn_failure_does_not_unlink_replaced_foreign_lock(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    output = root / "data" / "control_shards" / "run-a"

    class ReplacingFactory:
        def __call__(self, argv: list[str], **kwargs: object) -> FakeProcess:
            del argv, kwargs
            lock_path = output / control.LOCK_NAME
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": control.CONTROL_SCHEMA,
                        "run_id": "f" * 32,
                        "launch_config_sha256": "0" * 64,
                        "owner_token": "foreign-owner",
                    }
                ),
                encoding="utf-8",
            )
            raise OSError("synthetic ownership race")

    manager = control.ControlPlane(
        root,
        popen_factory=ReplacingFactory(),
        command_builder=lambda generated: [
            sys.executable,
            "fixture-child.py",
            "--config",
            str(generated.effective_config),
        ],
        signaler=FakeSignaler(),
    )

    with pytest.raises(control.ControlError) as captured:
        manager.start_new(_payload())

    assert captured.value.code == "spawn_failed"
    lock = json.loads((output / control.LOCK_NAME).read_text(encoding="utf-8"))
    assert lock["owner_token"] == "foreign-owner"
    assert manager.public()["credential_loaded"] is False


def test_stop_is_graceful_before_timeout_escalation(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manager, factory, signaler = _manager(root)
    started = manager.start_new(_payload())

    stopping = manager.stop(started["run_id"])

    assert stopping["process_state"] in {"stopping", "exited"}
    terminal = _wait_state(manager, "exited")
    assert terminal["exit_code"] == 130
    assert signaler.actions == ["graceful"]
    assert factory.calls[0][2].poll() == 130


def test_stop_timeout_terminates_then_reaper_finishes(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    signaler = FakeSignaler(graceful_exit=False, terminate_exit=True)
    manager, _, _ = _manager(root, signaler=signaler)
    started = manager.start_new(_payload())

    manager.stop(started["run_id"])
    terminal = _wait_state(manager, "exited")

    assert terminal["exit_code"] == 143
    assert signaler.actions == ["graceful", "terminate"]


def test_reconnect_wait_exposes_content_free_next_attempt_time(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manager, factory, _ = _manager(root)
    payload = _payload()
    payload["reconnect_backoff_seconds"] = 30
    started = manager.start_new(payload)

    factory.calls[0][2].complete(7)
    waiting = _wait_state(manager, "reconnect_wait")

    assert waiting["exit_code"] == 7
    assert waiting["reconnect"]["used"] == 1
    assert waiting["reconnect"]["maximum"] == 1
    assert isinstance(waiting["reconnect"]["next_at"], str)
    assert "sk-" not in waiting["reconnect"]["next_at"]

    manager.stop(started["run_id"])
    terminal = _wait_state(manager, "exited")
    assert terminal["reconnect"]["next_at"] is None


def test_continue_reuses_immutable_config_and_rejects_overrides(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manager, factory, _ = _manager(root)
    payload = _payload()
    payload["reconnect_attempts"] = 0
    started = manager.start_new(payload)
    first_config = (
        root
        / "runs"
        / "control-plane"
        / str(started["run_id"])
        / "effective-config.yaml"
    )
    first_sha = control._sha256_file(first_config)
    status = root / "data" / "control_shards" / "run-a" / "automation" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "state": "running",
                "current_worker": "frontend",
                "config_binding_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    factory.calls[0][2].complete(130)
    _wait_state(manager, "failed")

    with pytest.raises(control.ControlError) as override:
        manager.continue_run(
            {"run_id": started["run_id"], "api_key": SENTINEL, "model": "changed"}
        )
    assert override.value.code == "unknown_fields"

    resumed = manager.continue_run({"run_id": started["run_id"], "api_key": SENTINEL})
    assert resumed["process_state"] == "running"
    assert resumed["run_id"] == started["run_id"]
    assert control._sha256_file(first_config) == first_sha
    assert len(factory.calls) == 2
    factory.calls[1][2].complete(0)
    _wait_state(manager, "exited")


def test_continue_rejects_tampered_effective_config(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manager, factory, _ = _manager(root)
    started = manager.start_new(_payload())
    status = root / "data" / "control_shards" / "run-a" / "automation" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"state": "ready", "config_binding_sha256": "b" * 64}),
        encoding="utf-8",
    )
    factory.calls[0][2].complete(0)
    _wait_state(manager, "exited")
    config_path = (
        root
        / "runs"
        / "control-plane"
        / str(started["run_id"])
        / "effective-config.yaml"
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8"
    )

    with pytest.raises(control.ControlError) as captured:
        manager.continue_run({"run_id": started["run_id"], "api_key": SENTINEL})

    assert captured.value.code == "launch_config_changed"
    assert len(factory.calls) == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_id", "f" * 32),
        ("output_dir", "data/control_shards/run-b"),
        ("reconnect_attempts", 19),
        ("wait_cooldown", False),
    ],
)
def test_continue_rejects_manifest_field_tampering(
    tmp_path: Path, field: str, replacement: object
) -> None:
    root = _workspace(tmp_path)
    manager, factory, _ = _manager(root)
    started = manager.start_new(_payload())
    status = root / "data" / "control_shards" / "run-a" / "automation" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"state": "ready", "config_binding_sha256": "d" * 64}),
        encoding="utf-8",
    )
    factory.calls[0][2].complete(0)
    _wait_state(manager, "exited")
    manifest_path = (
        root
        / "runs"
        / "control-plane"
        / str(started["run_id"])
        / "control-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(control.ControlError) as captured:
        manager.continue_run({"run_id": started["run_id"], "api_key": SENTINEL})

    assert captured.value.code == "run_not_trusted"
    assert len(factory.calls) == 1


def test_continue_rejects_sop_tree_drift(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manager, factory, _ = _manager(root)
    started = manager.start_new(_payload())
    status = root / "data" / "control_shards" / "run-a" / "automation" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"state": "ready", "config_binding_sha256": "e" * 64}),
        encoding="utf-8",
    )
    factory.calls[0][2].complete(0)
    _wait_state(manager, "exited")
    (root / "skills" / "planner.md").write_text("drifted SOP", encoding="utf-8")

    with pytest.raises(control.ControlError) as captured:
        manager.continue_run({"run_id": started["run_id"], "api_key": SENTINEL})

    assert captured.value.code == "source_drift"
    assert len(factory.calls) == 1


def test_new_controller_cannot_resume_active_looking_unmanaged_shard(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    manager, factory, _ = _manager(root)
    started = manager.start_new(_payload())
    status = root / "data" / "control_shards" / "run-a" / "automation" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "state": "running",
                "current_worker": "frontend",
                "config_binding_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    factory.calls[0][2].complete(0)
    _wait_state(manager, "exited")

    restarted_controller, restarted_factory, _ = _manager(root)
    with pytest.raises(control.ControlError) as captured:
        restarted_controller.continue_run(
            {"run_id": started["run_id"], "api_key": SENTINEL}
        )

    assert captured.value.code == "unmanaged_owner"
    assert not restarted_factory.calls
    assert SENTINEL.encode() not in _disk_bytes(root)


def test_external_attach_is_monitor_only_and_does_not_write(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    external = root / "data" / "automated_v3_shards" / "external-c10"
    external.mkdir(parents=True)
    (external / "seeds.jsonl").write_text('{"seed_id":"seed-1"}\n', encoding="utf-8")
    before = _disk_bytes(external)
    attached: list[tuple[str, Path]] = []
    manager, _, _ = _manager(root, attached=attached)

    result = manager.attach_monitor(
        {"output_dir": "data/automated_v3_shards/external-c10", "label": "external-c10"}
    )

    assert result == {"attached": True, "label": "external-c10", "managed": False}
    assert attached == [("external-c10", external.resolve())]
    assert _disk_bytes(external) == before
    with pytest.raises(control.ControlError) as captured:
        manager.stop("f" * 32)
    assert captured.value.code == "stale_job"


def test_model_probe_uses_ram_key_and_returns_only_safe_models(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    observed: dict[str, object] = {}

    def probe(
        base_url: str, protocol: str, key: str, timeout: float
    ) -> dict[str, object]:
        observed.update(
            {"base_url": base_url, "protocol": protocol, "key": key, "timeout": timeout}
        )
        return {"status": "success", "models": ["safe-model", "bad model", SENTINEL]}

    manager = control.ControlPlane(root, probe_backend=probe)
    result = manager.probe_models(
        {
            "base_url": "https://provider.invalid/v1",
            "protocol": "openai",
            "api_key": SENTINEL,
            "model": "safe-model",
            "force_model": False,
            "timeout_seconds": 5,
        }
    )

    assert observed["key"] == SENTINEL
    assert result == {"status": "success", "models": ["safe-model"], "model_count": 1}
    assert manager.public()["credential_loaded"] is False
    assert SENTINEL.encode() not in _disk_bytes(root)


def test_concurrent_model_probes_never_share_credentials(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    first_entered = threading.Event()
    release_first = threading.Event()
    observed: list[tuple[str, str]] = []

    def probe(
        base_url: str, protocol: str, key: str, timeout: float
    ) -> dict[str, object]:
        del protocol, timeout
        observed.append((base_url, key))
        if base_url.endswith("provider-a.invalid/v1"):
            first_entered.set()
            assert release_first.wait(2)
        return {"status": "success", "models": []}

    manager = control.ControlPlane(root, probe_backend=probe)
    results: list[dict[str, object]] = []

    def call(base_url: str, key: str) -> None:
        results.append(
            manager.probe_models(
                {
                    "base_url": base_url,
                    "protocol": "openai",
                    "api_key": key,
                    "model": "fixture-model",
                    "force_model": False,
                    "timeout_seconds": 5,
                }
            )
        )

    key_a = "sk-probe-a-local-only"
    key_b = "sk-probe-b-local-only"
    first = threading.Thread(target=call, args=("https://provider-a.invalid/v1", key_a))
    second = threading.Thread(
        target=call, args=("https://provider-b.invalid/v1", key_b)
    )
    first.start()
    assert first_entered.wait(2)
    second.start()
    time.sleep(0.05)
    assert observed == [("https://provider-a.invalid/v1", key_a)]
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert observed == [
        ("https://provider-a.invalid/v1", key_a),
        ("https://provider-b.invalid/v1", key_b),
    ]
    assert len(results) == 2
    assert manager.public()["credential_loaded"] is False
    assert key_a.encode() not in _disk_bytes(root)
    assert key_b.encode() not in _disk_bytes(root)


def test_model_discovery_rejects_redirect_without_forwarding_key() -> None:
    received_authorization: list[str | None] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            received_authorization.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"should-not-arrive"}]}')

        def log_message(self, _format: str, *_args: object) -> None:
            return

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    target_url = f"http://127.0.0.1:{target.server_address[1]}/models"

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", target_url)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    redirect_thread = threading.Thread(target=redirect.serve_forever, daemon=True)
    redirect_thread.start()
    try:
        result = control.discover_models(
            f"http://127.0.0.1:{redirect.server_address[1]}/v1",
            "openai",
            "sk-redirect-must-not-forward",
            5,
        )
    finally:
        redirect.shutdown()
        target.shutdown()
        redirect.server_close()
        target.server_close()
        redirect_thread.join(timeout=2)
        target_thread.join(timeout=2)

    assert result == {"status": "invalid_response", "models": []}
    assert received_authorization == []


def test_windows_and_posix_process_group_contracts() -> None:
    process = FakeProcess()
    taskkill_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        taskkill_calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    windows = control.SystemSignaler("nt", run=fake_run)
    assert "creationflags" in windows.popen_group_kwargs()
    windows.graceful(process)
    windows.terminate(process)
    assert process.signals == [getattr(signal, "CTRL_BREAK_EVENT", 1)]
    assert taskkill_calls[0][0] == ["taskkill", "/PID", str(process.pid), "/T", "/F"]
    assert taskkill_calls[0][1]["shell"] is False

    group_calls: list[tuple[int, int]] = []
    posix = control.SystemSignaler(
        "posix",
        killpg=lambda group, sent_signal: group_calls.append((group, sent_signal)),
        getpgid=lambda pid: pid + 10,
    )
    assert posix.popen_group_kwargs() == {"start_new_session": True}
    posix.graceful(process)
    posix.terminate(process)
    posix.kill(process)
    assert group_calls == [
        (process.pid + 10, signal.SIGINT),
        (process.pid + 10, signal.SIGTERM),
        (process.pid + 10, getattr(signal, "SIGKILL", 9)),
    ]


def _server(
    root: Path,
) -> tuple[
    dashboard.DashboardServer, control.ControlPlane, FakeFactory, threading.Thread
]:
    manager, factory, _ = _manager(root)
    engine = dashboard.DashboardEngine([])
    manager.attach_callback = engine.attach_shard
    server = dashboard.DashboardServer(
        ("127.0.0.1", 0), engine, b"<p>control fixture</p>", manager
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, manager, factory, thread


def _session(port: int) -> tuple[str, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", "/")
    response = connection.getresponse()
    body = response.read()
    cookie = response.getheader("Set-Cookie")
    connection.close()
    assert response.status == 200 and cookie is not None
    return cookie.split(";", 1)[0], body


def _post(
    port: int,
    path: str,
    payload: object,
    cookie: str,
    *,
    origin: str | None = None,
    csrf: str = "1",
) -> tuple[int, bytes, dict[str, str]]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "Cookie": cookie,
        "X-Anchor-CSRF": csrf,
        "Origin": origin or f"http://127.0.0.1:{port}",
    }
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("POST", path, body=body, headers=headers)
    response = connection.getresponse()
    response_body = response.read()
    response_headers = dict(response.getheaders())
    connection.close()
    return response.status, response_body, response_headers


def test_http_control_requires_exact_host_origin_cookie_and_csrf(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    server, _, _, thread = _server(root)
    port = int(server.server_address[1])
    try:
        cookie, page = _session(port)
        assert server.session_cookie.encode() not in page

        status, _, headers = _post(port, "/api/control/clear-key", {}, cookie)
        assert status == 200
        assert not any(name.casefold().startswith("access-control") for name in headers)

        status, _, _ = _post(
            port,
            "/api/control/clear-key",
            {},
            cookie,
            origin="http://evil.invalid",
        )
        assert status == 403
        status, _, _ = _post(port, "/api/control/clear-key", {}, cookie, csrf="wrong")
        assert status == 403

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.putrequest("GET", "/api/snapshot", skip_host=True)
        connection.putheader("Host", "127.0.0.1.evil")
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        assert response.status == 403
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("OPTIONS", "/api/control/start")
        response = connection.getresponse()
        response.read()
        assert response.status == 405
        assert response.getheader("Access-Control-Allow-Origin") is None
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_start_never_returns_or_persists_key(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    server, manager, factory, thread = _server(root)
    port = int(server.server_address[1])
    try:
        cookie, _ = _session(port)
        status, body, _ = _post(port, "/api/control/start", _payload(), cookie)

        assert status == 200
        assert SENTINEL.encode() not in body
        assert SENTINEL.encode() not in _disk_bytes(root)
        assert factory.calls[0][1]["env"][control.CONTROL_KEY_ENV] == SENTINEL
        run_id = json.loads(body)["run_id"]
        status, stop_body, _ = _post(
            port, "/api/control/stop", {"run_id": run_id}, cookie
        )
        assert status == 200
        assert SENTINEL.encode() not in stop_body
        _wait_state(manager, "exited")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_rejects_duplicate_json_keys_and_oversize_before_spawn(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    server, _, factory, thread = _server(root)
    port = int(server.server_address[1])
    try:
        cookie, _ = _session(port)
        duplicate = b'{"run_id":"a","run_id":"b"}'
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST",
            "/api/control/stop",
            body=duplicate,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(duplicate)),
                "Cookie": cookie,
                "X-Anchor-CSRF": "1",
                "Origin": f"http://127.0.0.1:{port}",
            },
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 400
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.putrequest("POST", "/api/control/start")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(dashboard.MAX_POST_BODY_BYTES + 1))
        connection.putheader("Cookie", cookie)
        connection.putheader("X-Anchor-CSRF", "1")
        connection.putheader("Origin", f"http://127.0.0.1:{port}")
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        assert response.status == 413
        connection.close()
        assert not factory.calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_refuses_every_non_ipv4_loopback_bind(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manager, _, _ = _manager(root)
    engine = dashboard.DashboardEngine([])

    for host in ("localhost", "::1", "0.0.0.0"):
        with pytest.raises(ValueError, match="127.0.0.1"):
            dashboard.DashboardServer((host, 0), engine, b"fixture", manager)
