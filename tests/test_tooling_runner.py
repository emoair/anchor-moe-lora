import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from anchor_mvp.tooling import (
    AgentExecution,
    AnchorSandboxOptions,
    ControlledSessionCapture,
    OpenCodeExecutor,
    OpenCodeProvider,
    ToolPolicy,
)
from anchor_mvp.tooling.opencode_artifact import BinaryAttestation, sha256_file
from anchor_mvp.tooling.models import PublicDecisionStep, PublicOutcome
from anchor_mvp.tooling.runner import _public_outcome_capture
from anchor_mvp.tooling.session_export import credential_fingerprint


def _attested_executor(
    tmp_path: Path, *, windows_shim: bool = False
) -> OpenCodeExecutor:
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


def test_public_outcome_capture_uses_only_json_native_collections():
    outcome = PublicOutcome(
        status="completed",
        decision_trace=(
            PublicDecisionStep(check="test", evidence="passed", action="finish"),
        ),
        repair_summaries=("Repaired the controlled fixture.",),
        final_summary="done",
    )

    captured = _public_outcome_capture(outcome)

    assert captured == {
        "schema_version": "anchor.public-outcome.v1",
        "status": "completed",
        "decision_trace": [{"check": "test", "evidence": "passed", "action": "finish"}],
        "repair_summaries": ["Repaired the controlled fixture."],
        "final_summary": "done",
    }
    assert isinstance(captured["decision_trace"], list)
    assert isinstance(captured["repair_summaries"], list)


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


def test_probe_timeout_is_operator_configurable_but_cannot_restore_old_false_negative():
    assert OpenCodeExecutor(probe_timeout_seconds=225).probe_timeout_seconds == 225.0
    for value in (149.9, 0, True, "300"):
        with pytest.raises(
            ValueError, match="probe_timeout_seconds must be at least 150 seconds"
        ):
            OpenCodeExecutor(probe_timeout_seconds=value)


@pytest.mark.parametrize(
    "route_host",
    (
        "127.0.0.0",
        "127.255.255.255",
        "10.0.0.0",
        "10.255.255.255",
        "172.16.0.0",
        "172.31.255.255",
        "192.168.0.0",
        "192.168.255.255",
    ),
)
def test_anchor_sandbox_route_host_accepts_exact_audited_ipv4_ranges(route_host):
    options = AnchorSandboxOptions(route_host=route_host, route_port=1)

    assert options.command_options()[-4:] == [
        "--route-host",
        route_host,
        "--route-port",
        "1",
    ]


@pytest.mark.parametrize(
    "route_host",
    (
        "126.255.255.255",
        "128.0.0.0",
        "9.255.255.255",
        "11.0.0.0",
        "172.15.255.255",
        "172.32.0.0",
        "192.167.255.255",
        "192.169.0.0",
        "169.254.169.254",
        "192.0.0.1",
        "::1",
    ),
)
def test_anchor_sandbox_route_host_rejects_addresses_outside_audited_ranges(
    route_host,
):
    with pytest.raises(ValueError, match="route_host must be local"):
        AnchorSandboxOptions(route_host=route_host, route_port=18080)


@pytest.mark.parametrize("route_port", (1, 65535))
def test_anchor_sandbox_route_port_accepts_audited_boundaries(route_port):
    assert AnchorSandboxOptions(
        route_host="127.0.0.1", route_port=route_port
    ).route_port == route_port


@pytest.mark.parametrize("route_port", (0, 65536, True, 1.5))
def test_anchor_sandbox_route_port_rejects_values_outside_audited_boundaries(
    route_port,
):
    with pytest.raises(ValueError):
        AnchorSandboxOptions(route_host="127.0.0.1", route_port=route_port)


def test_provider_key_is_aliased_only_in_the_child_environment(monkeypatch, tmp_path):
    provider = OpenCodeProvider(
        provider_id="anchor-ark-glm52",
        npm="@ai-sdk/openai",
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="glm-5-2-260617",
        variant="max",
        key_env="ARK_CODING_API_KEY",
        route_host="ark.cn-beijing.volces.com",
    )
    monkeypatch.setenv("ARK_CODING_API_KEY", "test-process-only")
    monkeypatch.setenv("KIMI_CODE_API_KEY", "stale-wrong-provider-key")
    executor = OpenCodeExecutor(provider=provider)

    environment = executor._environment(tmp_path / "opencode.json")

    assert executor.api_key_present() is True
    assert "ARK_CODING_API_KEY" not in environment
    assert environment["KIMI_CODE_API_KEY"] == "test-process-only"


def test_selected_provider_key_is_reduced_to_nonplaintext_fingerprint(tmp_path):
    selected = "provider-value-ExactSyntheticCredential-012345"
    executor = OpenCodeExecutor(
        extra_environment={"KIMI_CODE_API_KEY": selected},
    )

    fingerprint = executor._selected_credential_fingerprint()

    assert fingerprint == credential_fingerprint(selected)
    assert selected not in repr(fingerprint)


def test_command_uses_configured_provider_model_and_variant(tmp_path):
    provider = OpenCodeProvider(
        provider_id="anchor-ark-glm52",
        npm="@ai-sdk/openai",
        base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        model="glm-5-2-260617",
        variant="max",
        key_env="ARK_CODING_API_KEY",
        route_host="ark.cn-beijing.volces.com",
    )
    executor = _attested_executor(tmp_path)
    executor.provider = provider

    argv = executor.command(
        sample_id="ark-wire",
        prompt="No call",
        workspace=tmp_path,
        config_path=tmp_path / "opencode.json",
    )

    assert argv[argv.index("--model") + 1] == "anchor-ark-glm52/glm-5-2-260617"
    assert argv[argv.index("--variant") + 1] == "max"
    assert executor.backend_name == "opencode-anchor-ark-glm52-anchor-sandbox"


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
    assert argv[argv.index("--config") + 1] == str(
        (tmp_path / "opencode.json").resolve()
    )
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


def test_anchor_cleanup_reuses_only_the_sandbox_routing_options(tmp_path):
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

    argv = executor._cleanup_command(sample_id="cleanup/test", workspace=tmp_path)

    assert argv[1:3] == ["anchor", "cleanup"]
    assert argv[argv.index("--run-id") + 1] == "cleanup-test"
    assert argv[argv.index("--workspace") + 1] == str(tmp_path.resolve())
    assert argv[argv.index("--wsl-distro") + 1] == "Ubuntu-22.04"
    assert argv[argv.index("--supervisor") + 1] == "wsl-root-systemd"
    assert "--linux-executable" not in argv
    assert "--memory" not in argv
    assert "--cpus" not in argv
    assert "--pids" not in argv
    assert "--timeout" not in argv


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
    attestation_kwargs: dict[str, object] = {}

    def fake_attestation(*args, **kwargs):
        attestation_kwargs.update(kwargs)
        return executor._attestation

    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.verify_binary_attestation",
        fake_attestation,
    )
    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"options": {}}),
            stderr="",
        ),
    )

    def fake_behavioral_probe(
        executable, *, probe_root, environment, timeout_seconds
    ):
        observed["probe_root"] = probe_root
        observed["timeout_seconds"] = timeout_seconds
        return True, "verified"

    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.run_behavioral_probe", fake_behavioral_probe
    )

    assert executor.probe_patched(config_path) == (True, "verified")
    assert attestation_kwargs["linux_executable"] is None
    assert config_path.parent not in observed["probe_root"].parents
    assert observed["probe_root"].parent == config_path.parent.parent
    assert observed["timeout_seconds"] == 300.0


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
        "decision_trace": [{"check": "test", "evidence": "passed", "action": "finish"}],
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

    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.subprocess.Popen", lambda *a, **k: FakeProcess()
    )
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


def test_collect_capture_lands_before_strict_success_even_when_tools_rejected(
    monkeypatch, tmp_path
):
    executor = _attested_executor(tmp_path)
    heldout_cases = tmp_path / "heldout.jsonl"
    heldout_cases.write_text("", encoding="utf-8")
    heldout_fixtures = tmp_path / "heldout-fixtures"
    heldout_fixtures.mkdir()
    heldout_manifest = tmp_path / "heldout-manifest.json"
    heldout_manifest.write_text("{}", encoding="utf-8")
    staging = tmp_path / "staging.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    executor.session_capture = ControlledSessionCapture(
        candidates_path=candidates.resolve(),
        quarantine_path=(tmp_path / "quarantine.jsonl").resolve(),
        heldout_cases=heldout_cases.resolve(),
        heldout_fixtures_root=heldout_fixtures.resolve(),
        heldout_manifest=heldout_manifest.resolve(),
        staging_path=staging.resolve(),
        mode="collect",
    )
    export_path = tmp_path / "session.json"
    export_path.write_text("{}", encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    execution = AgentExecution(
        exit_code=1,
        timed_out=False,
        duration_ms=1,
        rejected_events=1,
        controlled_session_id="ses_collect_1234",
        controlled_export_path=str(export_path),
        isolated_runtime_path=str(runtime),
        opencode_version="1.17.18",
    )
    staged = {
        "schema_version": "anchor.session-candidate-staging.v1",
        "sample_id": "collect-rejected",
        "quality": {"labels": ["tool_rejected"]},
    }
    observed_capture = {}

    def fake_staging_conversion(export_data, capture, policy):
        observed_capture.update(capture)
        return staged

    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.convert_controlled_session_staging",
        fake_staging_conversion,
    )
    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.convert_controlled_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("collect mode must defer strict conversion")
        ),
    )
    monkeypatch.setattr(executor, "_cleanup_sandbox", lambda **kwargs: None)

    captured, code = executor.finalize_capture(
        execution=execution,
        sample_id="collect-rejected",
        workspace=tmp_path,
        validators=(),
        final_diff=(
            {
                "file": "index.js",
                "patch": "--- a/index.js\n+++ b/index.js\n@@ -1 +1 @@\n-old\n+new",
                "additions": 1,
                "deletions": 1,
                "status": "modified",
            },
        ),
    )

    assert captured is True
    assert code is None
    assert observed_capture["final_diff"][0]["file"] == "index.js"
    assert json.loads(staging.read_text()) == staged
    assert not candidates.exists()
    assert not runtime.exists()
