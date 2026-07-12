import json
import os
from pathlib import Path
from types import SimpleNamespace

from anchor_mvp.tooling import (
    AnchorSandboxOptions,
    ControlledSessionCapture,
    OpenCodeExecutor,
    ToolPolicy,
)
from anchor_mvp.tooling.opencode_artifact import BinaryAttestation, sha256_file


def _attested_executor(tmp_path: Path, *, windows_shim: bool = False) -> OpenCodeExecutor:
    suffix = ".cmd" if windows_shim else ".exe" if os.name == "nt" else ""
    executable = tmp_path / f"opencode{suffix}"
    executable.write_bytes(b"audited-test-binary")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"test":true}\n', encoding="utf-8")
    executor = OpenCodeExecutor(executable=str(executable))
    executor._attestation = BinaryAttestation(
        executable=executable.resolve(),
        binary_sha256=sha256_file(executable),
        build_manifest=manifest.resolve(),
        build_manifest_sha256=sha256_file(manifest),
        patch_sha256="0" * 64,
        baseline_commit="test",
        opencode_version="test",
        behavioral_probe=True,
    )
    return executor


def test_live_environment_is_isolated_and_preserves_real_client_identity(tmp_path):
    config_path = tmp_path / ".anchor" / "opencode.json"
    executor = OpenCodeExecutor(
        extra_environment={
            "KIMI_CODE_API_KEY": "test-only",
            "OPENCODE_CONFIG_CONTENT": '{"permission":{"*":"allow"}}',
        }
    )

    environment = executor._environment(config_path)

    assert environment["OPENCODE_CLIENT"] == "cli"
    assert environment["OPENCODE_CONFIG"] == str(config_path)
    assert environment["OPENCODE_DISABLE_DEFAULT_PLUGINS"] == "true"
    assert environment["OPENCODE_DISABLE_CLAUDE_CODE"] == "true"
    assert environment["OPENCODE_DISABLE_MODELS_FETCH"] == "true"
    assert environment["OPENCODE_DISABLE_PROJECT_CONFIG"] == "true"
    assert "OPENCODE_CONFIG_CONTENT" not in environment
    assert environment["XDG_DATA_HOME"].startswith(str(tmp_path))
    assert environment["XDG_CACHE_HOME"].startswith(str(tmp_path))


def test_missing_opencode_returns_auditable_failure_without_api_call(tmp_path):
    executor = OpenCodeExecutor(executable="definitely-missing-opencode-binary")

    result = executor.run(
        sample_id="missing",
        prompt="No call",
        workspace=tmp_path,
        config_path=tmp_path / "config.json",
        policy=ToolPolicy(),
    )

    assert result.exit_code == 126
    assert result.error_codes == ("binary_attestation_missing_or_changed",)


def test_anchor_launch_failure_still_attempts_cleanup(monkeypatch, tmp_path):
    executor = _attested_executor(tmp_path)
    commands: list[list[str]] = []

    def failing_popen(*args, **kwargs):
        raise OSError("synthetic launch failure")

    def fake_run(command, *args, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("anchor_mvp.tooling.runner.subprocess.Popen", failing_popen)
    monkeypatch.setattr("anchor_mvp.tooling.runner.subprocess.run", fake_run)

    result = executor.run(
        sample_id="launch-failure",
        prompt="No call",
        workspace=tmp_path,
        config_path=tmp_path / "opencode.json",
        policy=ToolPolicy(),
    )

    assert result.error_codes == ("anchor_sandbox_launch_failed",)
    assert commands[0][1:3] == ["anchor", "cleanup"]


def test_windows_uses_the_launchable_npm_cmd_shim(tmp_path):
    if os.name != "nt":
        return
    command = _attested_executor(tmp_path, windows_shim=True).command(
        sample_id="shim",
        prompt="No call",
        workspace=tmp_path,
        config_path=tmp_path / "opencode.json",
    )

    assert Path(command[0]).suffix.casefold() == ".cmd"


def test_command_pins_audited_thinking_variant_without_printing_reasoning(tmp_path):
    linux = (tmp_path / "opencode-linux").resolve()
    executor = _attested_executor(tmp_path)
    executor.sandbox_options = AnchorSandboxOptions(
        linux_executable=linux,
        wsl_distro="Ubuntu-22.04",
        supervisor="wsl-root-systemd",
        memory="4G",
        cpus="2",
        pids=256,
        timeout_seconds=900,
    )
    argv = executor.command(
        sample_id="thinking",
        prompt="No call",
        workspace=tmp_path,
        config_path=tmp_path / "opencode.json",
    )

    assert argv[1:3] == ["anchor", "run"]
    assert argv[argv.index("--run-id") + 1] == "thinking"
    assert argv[argv.index("--workspace") + 1] == str(tmp_path.resolve())
    assert argv[argv.index("--config") + 1] == str((tmp_path / "opencode.json").resolve())
    assert argv[argv.index("--linux-executable") + 1] == str(linux)
    assert argv[argv.index("--wsl-distro") + 1] == "Ubuntu-22.04"
    assert argv[argv.index("--supervisor") + 1] == "wsl-root-systemd"
    assert argv[argv.index("--memory") + 1] == "4G"
    assert argv[argv.index("--cpus") + 1] == "2"
    assert argv[argv.index("--pids") + 1] == "256"
    assert argv[argv.index("--timeout") + 1] == "900"
    assert argv[argv.index("--model") + 1] == "anchor-kimi/kimi-for-coding"
    assert argv[argv.index("--agent") + 1] == "anchor-distiller"
    assert argv[argv.index("--variant") + 1] == "medium"
    assert "--dir" not in argv
    assert "--thinking" not in argv


def test_anchor_export_reuses_the_run_sandbox_options(tmp_path):
    executor = _attested_executor(tmp_path)
    executor.sandbox_options = AnchorSandboxOptions(
        linux_executable=(tmp_path / "opencode-linux").resolve(),
        wsl_distro="Ubuntu-22.04",
        supervisor="wsl-root-systemd",
        memory="4G",
        cpus="2",
        pids=256,
        timeout_seconds=900,
    )

    argv = executor._export_command(
        sample_id="export/test",
        session_id="ses_mock_1234",
        workspace=tmp_path,
        config_path=tmp_path / "opencode.json",
    )

    assert argv[1:3] == ["anchor", "export"]
    assert argv[argv.index("--run-id") + 1] == "export-test"
    assert argv[argv.index("--wsl-distro") + 1] == "Ubuntu-22.04"
    assert argv[argv.index("--supervisor") + 1] == "wsl-root-systemd"
    assert argv[argv.index("--memory") + 1] == "4G"
    assert argv[argv.index("--session") + 1] == "ses_mock_1234"


def test_anchor_run_id_is_normalized_before_cli_invocation(tmp_path):
    argv = _attested_executor(tmp_path).command(
        sample_id="plan/计算器:01",
        prompt="No call",
        workspace=tmp_path,
        config_path=tmp_path / "opencode.json",
    )

    assert argv[argv.index("--run-id") + 1] == "plan-u8ba1u7b97u5668-01"
    assert argv[argv.index("--title") + 1] == "anchor-distiller:plan-u8ba1u7b97u5668-01"


def test_live_agent_config_rejects_any_initial_tool_force():
    assert OpenCodeExecutor.is_unforced_agent_config({"options": {}})
    assert not OpenCodeExecutor.is_unforced_agent_config(
        {"requireInitialToolCall": True, "options": {}}
    )
    assert not OpenCodeExecutor.is_unforced_agent_config(
        {"options": {"requireInitialToolCall": True}}
    )


def test_patched_probe_uses_a_sibling_root_to_avoid_parent_config_merge(
    monkeypatch, tmp_path
):
    executor = _attested_executor(tmp_path)
    config_path = tmp_path / "preflight" / "opencode.json"
    config_path.parent.mkdir()
    config_path.write_text("{}", encoding="utf-8")
    observed: dict[str, Path] = {}

    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.verify_binary_attestation",
        lambda *args, **kwargs: executor._attestation,
    )
    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"options": {}}),
            stderr="",
        ),
    )

    def fake_behavioral_probe(executable, *, probe_root, environment):
        observed["probe_root"] = probe_root
        return True, "verified"

    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.run_behavioral_probe", fake_behavioral_probe
    )

    assert executor.probe_patched(config_path) == (True, "verified")
    assert config_path.parent not in observed["probe_root"].parents
    assert observed["probe_root"].parent == config_path.parent.parent


def test_export_precedes_isolated_runtime_deletion(monkeypatch, tmp_path):
    executor = _attested_executor(tmp_path)
    heldout_cases = tmp_path / "heldout.jsonl"
    heldout_cases.write_text("", encoding="utf-8")
    heldout_fixtures = tmp_path / "heldout-fixtures"
    heldout_fixtures.mkdir()
    heldout_manifest = tmp_path / "heldout-manifest.json"
    heldout_manifest.write_text("{}", encoding="utf-8")
    executor.session_capture = ControlledSessionCapture(
        candidates_path=(tmp_path / "candidates.jsonl").resolve(),
        quarantine_path=(tmp_path / "quarantine.jsonl").resolve(),
        heldout_cases=heldout_cases.resolve(),
        heldout_fixtures_root=heldout_fixtures.resolve(),
        heldout_manifest=heldout_manifest.resolve(),
    )

    outcome = {
        "schema_version": "anchor.public-outcome.v1",
        "status": "completed",
        "decision_trace": [
            {"check": "test", "evidence": "passed", "action": "finish"}
        ],
        "repair_summaries": [],
        "final_summary": "done",
    }
    stdout = json.dumps(
        {
            "type": "text",
            "sessionID": "ses_mock_1234",
            "part": {"type": "text", "text": json.dumps(outcome)},
        }
    )

    class FakeProcess:
        returncode = 0
        pid = 1

        def communicate(self, timeout=None):
            return stdout, ""

    monkeypatch.setattr("anchor_mvp.tooling.runner.subprocess.Popen", lambda *a, **k: FakeProcess())
    calls: list[list[str]] = []

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")

    monkeypatch.setattr("anchor_mvp.tooling.runner.subprocess.run", fake_run)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "run" / "opencode.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")

    execution = executor.run(
        sample_id="lifecycle",
        prompt="test",
        workspace=workspace,
        config_path=config,
        policy=ToolPolicy(),
    )

    export_path = Path(execution.controlled_export_path or "")
    runtime_path = Path(execution.isolated_runtime_path or "")
    assert export_path.read_bytes() == b"{}"
    assert runtime_path.is_dir()

    captured, code = executor.finalize_capture(
        execution=execution,
        sample_id="lifecycle",
        workspace=workspace,
        validators=(),
    )

    assert captured is False
    assert code is not None
    assert not runtime_path.exists()
    assert any(command[1:3] == ["anchor", "cleanup"] for command in calls)
    quarantine = (tmp_path / "quarantine.jsonl").read_text(encoding="utf-8")
    assert "lifecycle" in quarantine
    assert "ses_mock_1234" not in quarantine
