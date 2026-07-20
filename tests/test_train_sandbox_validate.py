from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tooling" / "train_sandbox_validate.py"
SPEC = importlib.util.spec_from_file_location("train_sandbox_validate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "validator@example.invalid")
    _git(root, "config", "user.name", "Validator Test")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "app.js").write_text("const value = 1;\n", encoding="utf-8")
    _git(root, "add", "app.py", "app.js")
    _git(root, "commit", "--quiet", "-m", "baseline")
    return root


def test_python_compile_is_bound_and_side_effect_free(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

    changed, before = MODULE._workspace_state(root)
    code_paths = MODULE._changed_code_paths(changed)
    assert MODULE._compile(root, code_paths) == ("python-compile",)
    changed_after, after = MODULE._workspace_state(root)

    assert changed_after == changed
    assert after == before
    assert not list(root.rglob("__pycache__"))
    assert not list(root.rglob("*.pyc"))


def test_git_args_trust_only_the_exact_resolved_workspace(tmp_path: Path) -> None:
    root = _repo(tmp_path)

    args = MODULE._git_args(root, "status", "--porcelain")

    assert args[:2] == ["git", "-c"]
    assert args[2] == f"safe.directory={root.resolve(strict=True)}"
    assert "*" not in args[2]
    assert args[3:] == ["-C", str(root.resolve(strict=True)), "status", "--porcelain"]


def test_validation_state_changes_after_a_later_edit(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    _changed, first = MODULE._workspace_state(root)

    (root / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    _changed, second = MODULE._workspace_state(root)

    assert first != second


def test_js_validator_receives_every_changed_js_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    (root / "app.js").write_text("const value = 2;\n", encoding="utf-8")
    (root / "second.mjs").write_text("export const second = 2;\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_stream(args: list[str], *, root: Path, timeout: int) -> None:
        del root, timeout
        calls.append(args)

    monkeypatch.setattr(MODULE, "_stream", fake_stream)
    changed, _state = MODULE._workspace_state(root)
    validators = MODULE._compile(root, MODULE._changed_code_paths(changed))

    assert validators == ("node-check",)
    assert calls == [
        ["node", "--check", "app.js"],
        ["node", "--check", "second.mjs"],
    ]


def test_unsupported_changed_code_fails_closed(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "component.tsx").write_text("export const App = () => <main />;\n", encoding="utf-8")
    changed, _state = MODULE._workspace_state(root)

    with pytest.raises(MODULE.ValidationError, match="changed_code_language_unsupported"):
        MODULE._changed_code_paths(changed)


def test_deleted_code_fails_closed(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "app.py").unlink()

    with pytest.raises(MODULE.ValidationError, match="changed_code_deletion_not_validated"):
        MODULE._workspace_state(root)


def test_main_emits_content_free_final_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repo(tmp_path)
    (root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.chdir(root)

    assert MODULE.main(["compile"]) == 0
    value = json.loads(capsys.readouterr().out.splitlines()[-1])

    assert value["schema_version"] == "anchor.train-sandbox-validation.v1"
    assert value["validator_version"] == "1.0.1"
    assert value["validation_level"] == "syntax"
    assert value["success"] is True
    assert value["not_official_swebench_pass"] is True
    assert value["changed_paths"] == ["app.py"]
    assert len(value["changed_paths_sha256"]) == 64
    assert len(value["final_state_sha256"]) == 64
