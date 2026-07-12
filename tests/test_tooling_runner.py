import json
import os
from pathlib import Path
from types import SimpleNamespace

from anchor_mvp.tooling import ControlledSessionCapture, OpenCodeExecutor, ToolPolicy
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


def test_windows_uses_the_launchable_npm_cmd_shim(tmp_path):
    if os.name != "nt":
        return
    command = _attested_executor(tmp_path, windows_shim=True).command(
        sample_id="shim",
        prompt="No call",
        workspace=tmp_path,
    )

    assert Path(command[0]).suffix.casefold() == ".cmd"


def test_command_pins_audited_thinking_variant_without_printing_reasoning(tmp_path):
    command = _attested_executor(tmp_path).command(
        sample_id="thinking",
        prompt="No call",
        workspace=tmp_path,
    )

    assert command[command.index("--variant") + 1] == "thinking"
    assert "--thinking" not in command


def test_patched_capability_requires_first_class_agent_marker():
    assert OpenCodeExecutor.is_patched_agent_config(
        {"requireInitialToolCall": True, "options": {}}
    )
    assert not OpenCodeExecutor.is_patched_agent_config(
        {"options": {"requireInitialToolCall": True}}
    )
    assert not OpenCodeExecutor.is_patched_agent_config({"options": {}})


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
            stdout=json.dumps({"requireInitialToolCall": True, "options": {}}),
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
    monkeypatch.setattr(
        "anchor_mvp.tooling.runner.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=b"{}", stderr=b""),
    )
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
    quarantine = (tmp_path / "quarantine.jsonl").read_text(encoding="utf-8")
    assert "lifecycle" in quarantine
    assert "ses_mock_1234" not in quarantine
