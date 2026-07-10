import os
from pathlib import Path

from anchor_mvp.tooling import OpenCodeExecutor, ToolPolicy


def test_live_environment_is_isolated_and_preserves_real_client_identity(tmp_path):
    config_path = tmp_path / ".anchor" / "opencode.json"
    executor = OpenCodeExecutor(extra_environment={"KIMI_CODE_API_KEY": "test-only"})

    environment = executor._environment(config_path)

    assert environment["OPENCODE_CLIENT"] == "cli"
    assert environment["OPENCODE_CONFIG"] == str(config_path)
    assert environment["OPENCODE_DISABLE_DEFAULT_PLUGINS"] == "true"
    assert environment["OPENCODE_DISABLE_CLAUDE_CODE"] == "true"
    assert environment["OPENCODE_DISABLE_MODELS_FETCH"] == "true"
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

    assert result.exit_code == 127
    assert result.error_codes == ("opencode_not_installed",)


def test_windows_uses_the_launchable_npm_cmd_shim(tmp_path):
    if os.name != "nt":
        return
    command = OpenCodeExecutor().command(
        sample_id="shim",
        prompt="No call",
        workspace=tmp_path,
    )

    assert Path(command[0]).suffix.casefold() == ".cmd"
