from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT / "scripts" / "research" / "run_gemma3_1b_it_five_role_qonly_v1.ps1"
)
GPU_UUID = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def _powershell() -> str:
    executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    return executable


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def _make_project(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    launcher = (
        project / "scripts" / "research" / "run_gemma3_1b_it_five_role_qonly_v1.ps1"
    )
    runner = project / "scripts" / "research" / "run_gemma3_1b_it_five_role_qonly_v1.py"
    implementation = (
        project / "src" / "anchor_mvp" / "training" / "gemma3_five_role_qonly_v1.py"
    )
    config = project / "configs" / "training" / "gemma3_1b_it_five_role_qonly_v1.yaml"
    fake_python = project / "test-bin" / "fake-python.cmd"
    fake_nvidia = project / "test-bin" / "nvidia-smi.cmd"
    python_log = project / "python-args.log"
    nvidia_log = project / "nvidia-args.log"

    _write_text(launcher, LAUNCHER.read_text(encoding="utf-8"))
    _write_text(runner, "# model-free test double\n")
    _write_text(implementation, "# model-free implementation test double\n")
    _write_text(config, "schema_version: test-only\n")
    _write_text(
        fake_python,
        (
            "@echo off\r\n"
            'if "%~1"=="-c" (\r\n'
            "  if defined ANCHOR_MOCK_PYTHON_PROBE_FAILURE exit /b 17\r\n"
            '  echo {"bitsandbytes_version":"0.48.2","missing":[],"schema_version":'
            '"anchor.python-runtime-probe.v1","version":[3,11,9]}\r\n'
            "  exit /b 0\r\n"
            ")\r\n"
            "if defined ANCHOR_REQUIRE_LOCK (\r\n"
            '  if not exist "%ANCHOR_TEST_PROJECT%\\runs\\formal-v3-training.lock" '
            "exit /b 23\r\n"
            ")\r\n"
            'echo UUID=%ANCHOR_GEMMA_GPU_UUID%>>"%ANCHOR_PYTHON_LOG%"\r\n'
            'echo %*>>"%ANCHOR_PYTHON_LOG%"\r\n'
            "exit /b 0\r\n"
        ),
    )
    _write_text(
        fake_nvidia,
        (
            "@echo off\r\n"
            'echo %*>>"%ANCHOR_NVIDIA_LOG%"\r\n'
            'if /i "%~1"=="--query-gpu=index,uuid,name,driver_model.current,'
            'memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu" (\r\n'
            f'  echo 0, {GPU_UUID}, "NVIDIA GeForce RTX 3080 Ti", '
            "WDDM, 12288, 1373, 10915, 5, 55\r\n"
            "  exit /b 0\r\n"
            ")\r\n"
            'if /i "%~1"=="--query-compute-apps=gpu_uuid,pid,process_name,'
            'used_gpu_memory" (\r\n'
            f"  echo {GPU_UUID}, 16180, "
            '"C:\\Windows\\explorer.exe", [N/A]\r\n'
            f"  echo {GPU_UUID}, 4304, "
            '"C:\\Windows\\System32\\dwm.exe", [N/A]\r\n'
            "  if defined ANCHOR_MOCK_FOREIGN (\r\n"
            f"    echo {GPU_UUID}, 777, "
            '"C:\\Python\\python.exe", [N/A]\r\n'
            "  )\r\n"
            "  exit /b 0\r\n"
            ")\r\n"
            "echo unexpected arguments 1>&2\r\n"
            "exit /b 19\r\n"
        ),
    )
    return {
        "project": project,
        "launcher": launcher,
        "runner": runner,
        "implementation": implementation,
        "config": config,
        "python": fake_python,
        "nvidia": fake_nvidia,
        "python_log": python_log,
        "nvidia_log": nvidia_log,
    }


def _run_launcher(
    paths: dict[str, Path],
    *extra_args: str,
    extra_env: dict[str, str] | None = None,
    include_python_arg: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "ANCHOR_PYTHON_LOG": str(paths["python_log"]),
            "ANCHOR_NVIDIA_LOG": str(paths["nvidia_log"]),
            "ANCHOR_TEST_PROJECT": str(paths["project"]),
        }
    )
    if extra_env:
        env.update(extra_env)
    command = [
        _powershell(),
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(paths["launcher"]),
        "-NvidiaSmiPath",
        str(paths["nvidia"]),
    ]
    if include_python_arg:
        command.extend(["-Python", str(paths["python"])])
    command.extend(extra_args)
    return subprocess.run(
        command,
        cwd=paths["project"],
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_launcher_static_contract_is_fail_closed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert re.search(r"C:\\Users\\[^\\]+", source) is None
    assert '[string]$Python = ""' in source
    assert "ANCHOR_PYTHON" in source
    assert "CONDA_PREFIX" in source
    assert "yaml,sentencepiece,torch,transformers,peft" in source
    assert "runs/formal-v3-training.lock" in source
    assert "runs/distill-train-handoff/gpu-job.lock" in source
    assert "runs/distill-train-handoff-v3/gpu-job.lock" in source
    assert "[IO.FileMode]::CreateNew" in source
    assert "[IO.FileShare]::None" in source
    assert "[IO.FileOptions]::DeleteOnClose" in source
    assert "Remove-Item" not in source
    assert "--dry-run" in source
    assert "--lease-receipt" in source
    assert "--lease-receipt-sha256" in source
    assert "--gpu-attestation" in source
    assert "--gpu-attestation-sha256" in source
    assert "--run-id" in source
    assert "smoke_steps_per_role = 2" in source
    assert "full_steps_per_role = 160" in source
    assert "resume_allowed = $false" in source
    assert "concurrency = 1" in source
    assert "expert_private_tail_append_only = $true" in source
    assert (
        "private_tail_includes_post_activation_prompt_and_generated_tokens = $true"
        in source
    )
    assert "private_tail_cross_expert_reuse = $false" in source
    assert "committed_text_reencoded_for_next_shared_context = $true" in source
    assert "wddm_gui_process_allowlist" in source
    assert '"chatgpt.exe"' in source
    assert '"promecefpluginhost.exe"' in source
    assert '"wechatappex.exe"' in source
    assert "Insufficient Permissions" in source
    for role in (
        "planner",
        "tool_policy",
        "frontend_gen",
        "frontend_review",
        "security_gate",
    ):
        assert f'"{role}"' in source
    assert re.search(
        r"if \(-not \$Execute\) \{.*?--dry-run.*?exit 0",
        source,
        re.DOTALL,
    )


def test_default_path_is_model_free_and_never_queries_nvidia(tmp_path: Path) -> None:
    paths = _make_project(tmp_path)
    result = _run_launcher(paths)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dry-run" in paths["python_log"].read_text(encoding="utf-8")
    assert not paths["nvidia_log"].exists()
    assert not (paths["project"] / "runs" / "formal-v3-training.lock").exists()


def test_python_runtime_can_be_selected_from_environment_without_personal_default(
    tmp_path: Path,
) -> None:
    paths = _make_project(tmp_path)
    result = _run_launcher(
        paths,
        include_python_arg=False,
        extra_env={"ANCHOR_PYTHON": str(paths["python"])},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dry-run" in paths["python_log"].read_text(encoding="utf-8")
    assert not paths["nvidia_log"].exists()


def test_python_runtime_probe_fails_closed_before_gpu_or_runner(tmp_path: Path) -> None:
    paths = _make_project(tmp_path)
    result = _run_launcher(
        paths,
        extra_env={"ANCHOR_MOCK_PYTHON_PROBE_FAILURE": "1"},
    )

    assert result.returncode != 0
    assert "version/dependency probe failed" in (result.stdout + result.stderr)
    assert not paths["python_log"].exists()
    assert not paths["nvidia_log"].exists()


def test_execute_requires_bound_uuid_before_any_gpu_probe(tmp_path: Path) -> None:
    paths = _make_project(tmp_path)
    result = _run_launcher(
        paths,
        "-Execute",
        extra_env={"ANCHOR_GEMMA_GPU_UUID": "UNBOUND"},
    )

    assert result.returncode != 0
    assert "requires a bound GPU UUID" in (result.stdout + result.stderr)
    assert not paths["nvidia_log"].exists()
    assert not paths["python_log"].exists()
    assert not (paths["project"] / "runs" / "formal-v3-training.lock").exists()


def test_mock_wddm_idle_gate_publishes_attestation_and_releases_lock(
    tmp_path: Path,
) -> None:
    paths = _make_project(tmp_path)
    result = _run_launcher(
        paths,
        "-Execute",
        "-ExpectedGpuUuid",
        GPU_UUID,
        extra_env={"ANCHOR_REQUIRE_LOCK": "1"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (paths["project"] / "runs" / "formal-v3-training.lock").exists()
    nvidia_calls = [
        line
        for line in paths["nvidia_log"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(nvidia_calls) == 12
    attestations = list(
        (
            paths["project"]
            / "runs"
            / "gemma3_1b_it_five_role_qonly_v1"
            / "gpu-attestations"
        ).glob("*/gpu_attestation.json")
    )
    assert len(attestations) == 1
    attestation_path = attestations[0]
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["canonical_lock"] == "runs/formal-v3-training.lock"
    assert payload["expected_gpu_index"] == 0
    assert payload["expected_gpu_uuid"] == GPU_UUID
    assert payload["smoke_steps"] == 2
    assert payload["full_steps"] == 160
    assert payload["concurrency"] == 1
    assert re.fullmatch(r"[0-9a-f]{64}", payload["config_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", payload["implementation_sha256"])
    assert payload["compute_processes"] == [
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
    assert len(payload["pre_lock_samples"]) == 3
    assert len(payload["post_lock_samples"]) == 3
    all_samples = payload["pre_lock_samples"] + payload["post_lock_samples"]
    assert all(
        sample["selected_gpu_compute_process_count"] == 2 for sample in all_samples
    )
    assert len({sample["compute_inventory_sha256"] for sample in all_samples}) == 1
    assert all(
        sample["compute_processes"] == payload["compute_processes"]
        for sample in all_samples
    )
    assert payload["compute_processes"] == sorted(
        payload["compute_processes"],
        key=lambda item: (item["pid"], item["process_name"]),
    )
    assert all(sample["wddm_desktop_baseline_tolerated"] for sample in all_samples)
    assert payload["lock"]["file_mode"] == "CreateNew"
    assert payload["lock"]["file_share"] == "None"
    assert payload["lock"]["delete_on_close"] is True
    assert payload["execution_plan"]["concurrency"] == 1
    assert payload["execution_plan"]["smoke_steps_per_role"] == 2
    assert payload["execution_plan"]["full_steps_per_role"] == 160
    expected_kv_boundary = {
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
    assert payload["kv_runtime_boundary"] == expected_kv_boundary
    sidecar = attestation_path.with_name("gpu_attestation.json.sha256")
    sidecar_text = sidecar.read_text(encoding="utf-8")
    assert re.fullmatch(r"[0-9a-f]{64}  gpu_attestation\.json\n", sidecar_text)
    lease_path = attestation_path.with_name("lease_receipt.json")
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    assert lease["status"] == "passed"
    assert lease["run_id"] == payload["run_id"]
    assert lease["canonical_lock"] == "runs/formal-v3-training.lock"
    assert lease["expected_gpu_uuid"] == GPU_UUID
    assert lease["roles"] == payload["roles"]
    assert lease["smoke_steps"] == 2
    assert lease["full_steps"] == 160
    assert lease["concurrency"] == 1
    assert lease["kv_runtime_boundary"] == expected_kv_boundary
    lease_sidecar = lease_path.with_name("lease_receipt.json.sha256")
    assert re.fullmatch(
        r"[0-9a-f]{64}  lease_receipt\.json\n",
        lease_sidecar.read_text(encoding="utf-8"),
    )
    python_args = paths["python_log"].read_text(encoding="utf-8")
    assert f"UUID={GPU_UUID}" in python_args
    assert "--execute" in python_args
    assert "--lease-receipt" in python_args
    assert "--lease-receipt-sha256" in python_args
    assert "--gpu-attestation" in python_args
    assert "--gpu-attestation-sha256" in python_args
    assert "--run-id" in python_args


def test_preexisting_canonical_lock_is_preserved_without_gpu_probe(
    tmp_path: Path,
) -> None:
    paths = _make_project(tmp_path)
    lock = paths["project"] / "runs" / "formal-v3-training.lock"
    _write_text(lock, "owned-by-another-process\n")

    result = _run_launcher(
        paths,
        "-Execute",
        "-ExpectedGpuUuid",
        GPU_UUID,
    )

    assert result.returncode != 0
    assert "canonical GPU training lock already exists" in (
        result.stdout + result.stderr
    )
    assert lock.read_text(encoding="utf-8") == "owned-by-another-process\n"
    assert not paths["nvidia_log"].exists()
    assert not paths["python_log"].exists()


def test_mock_quoted_foreign_compute_process_fails_before_lock(
    tmp_path: Path,
) -> None:
    paths = _make_project(tmp_path)
    result = _run_launcher(
        paths,
        "-Execute",
        "-ExpectedGpuUuid",
        GPU_UUID,
        extra_env={"ANCHOR_MOCK_FOREIGN": "1"},
    )

    assert result.returncode != 0
    error = result.stdout + result.stderr
    assert "foreign" in error
    assert "python.exe:777" in error
    assert not paths["python_log"].exists()
    assert not (paths["project"] / "runs" / "formal-v3-training.lock").exists()
    assert not (
        paths["project"]
        / "runs"
        / "gemma3_1b_it_five_role_qonly_v1"
        / "gpu-attestations"
    ).exists()
