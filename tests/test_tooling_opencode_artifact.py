from __future__ import annotations

import json
from pathlib import Path

import pytest

from anchor_mvp.tooling.behavioral_probe import (
    PROBE_MARKER,
    ProbeTranscript,
    _is_execution_request,
    _is_title_request,
    _response,
    _sse_lines,
)
from anchor_mvp.tooling.opencode_artifact import (
    sha256_file,
    verify_binary_attestation,
    verify_launch_identity,
)


def _write_attested_artifact(tmp_path: Path):
    patch = tmp_path / "audited.patch"
    patch.write_bytes(b"audited patch\n")
    source = tmp_path / "patch-manifest.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "anchor.opencode-patch-source.v1",
                "repository": "https://github.com/anomalyco/opencode.git",
                "baseline_commit": "1" * 40,
                "upstream_version": "1.17.18",
                "patch": patch.name,
                "patch_sha256": sha256_file(patch),
                "bun_version": "1.3.14",
                "tool_contract_version": "anchor.execution-tool-contract.v2",
                "tool_contract": {
                    "version": "anchor.execution-tool-contract.v2",
                    "tools": [
                        "apply_patch",
                        "bash",
                        "edit",
                        "glob",
                        "grep",
                        "list",
                        "read",
                        "write",
                    ],
                    "bash_commands": [
                        "npm run build --if-present",
                        "npm run lint --if-present",
                        "npm run test --if-present",
                    ],
                },
                "required_tests": {
                    "core": ["test/config/behavior.test.ts"],
                    "opencode": ["test/session/behavior.test.ts"],
                },
            }
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "opencode-anchor.exe"
    executable.write_bytes(b"audited binary")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "anchor.patched-opencode.v1",
                "repository": "https://github.com/anomalyco/opencode.git",
                "baseline_commit": "1" * 40,
                "opencode_version": "1.17.18",
                "patch_sha256": sha256_file(patch),
                "patch_source_manifest_sha256": sha256_file(source),
                "bun_version": "1.3.14",
                "tool_contract_version": "anchor.execution-tool-contract.v2",
                "tool_contract": {
                    "version": "anchor.execution-tool-contract.v2",
                    "tools": [
                        "apply_patch",
                        "bash",
                        "edit",
                        "glob",
                        "grep",
                        "list",
                        "read",
                        "write",
                    ],
                    "bash_commands": [
                        "npm run build --if-present",
                        "npm run lint --if-present",
                        "npm run test --if-present",
                    ],
                },
                "tests_executed": True,
                "required_tests": {
                    "core": ["test/config/behavior.test.ts"],
                    "opencode": ["test/session/behavior.test.ts"],
                },
                "typecheck_executed": True,
                "binary_sha256": sha256_file(executable),
                "binary": executable.name,
                "global_install_modified": False,
            }
        ),
        encoding="utf-8",
    )
    return executable, source, manifest


def test_artifact_attestation_rehashes_binary_and_manifest_before_launch(tmp_path: Path):
    executable, source, _ = _write_attested_artifact(tmp_path)
    attestation = verify_binary_attestation(executable, patch_manifest=source)
    verified = attestation.with_behavioral_probe()

    verify_launch_identity(verified)
    executable.write_bytes(b"changed after probe")

    with pytest.raises(ValueError, match="changed before launch"):
        verify_launch_identity(verified)


def test_artifact_attestation_requires_exact_behavioral_test_manifest(tmp_path: Path):
    executable, source, manifest = _write_attested_artifact(tmp_path)
    build = json.loads(manifest.read_text(encoding="utf-8"))
    build["required_tests"]["opencode"] = ["test/session/wrong.test.ts"]
    manifest.write_text(json.dumps(build), encoding="utf-8")

    with pytest.raises(ValueError, match="required_tests"):
        verify_binary_attestation(executable, patch_manifest=source)


def test_artifact_attestation_requires_converter_tool_contract(tmp_path: Path):
    executable, source, _ = _write_attested_artifact(tmp_path)
    source_data = json.loads(source.read_text(encoding="utf-8"))
    source_data["tool_contract"]["tools"].append("task")
    source.write_text(json.dumps(source_data), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the converter"):
        verify_binary_attestation(executable, patch_manifest=source)


def test_probe_transcript_preserves_automatic_tool_choice_across_tool_result():
    transcript = ProbeTranscript(
        requests=[
            {
                "tool_choice": "auto",
                "tools": [{"type": "function", "function": {"name": "read"}}],
                "messages": [{"role": "user", "content": "read the marker"}],
            },
            {
                "tool_choice": "auto",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "",
                        "tool_calls": [{"id": "call_anchor_probe_1"}],
                    },
                    {"role": "tool", "content": PROBE_MARKER},
                ],
            },
        ]
    )

    assert transcript.validate()[0] is True
    transcript.requests[0]["tool_choice"] = "required"
    assert transcript.validate() == (
        False,
        "first provider request did not preserve automatic tool choice",
    )


def test_probe_ignores_title_request_but_tracks_the_execution_turn():
    title_request = {
        "messages": [
            {"role": "system", "content": "You are a title generator. Output only a concise title."}
        ]
    }

    assert _is_title_request(title_request) is True
    assert _is_execution_request(title_request) is False
    assert _is_title_request({"tools": []}) is False
    assert _is_execution_request({"tools": []}) is True


def test_probe_rejects_extra_first_turn_tools():
    transcript = ProbeTranscript(
        requests=[
            {
                "tool_choice": "auto",
                "tools": [
                    {"type": "function", "function": {"name": "read"}},
                    {"type": "function", "function": {"name": "task"}},
                ],
            },
            {"tool_choice": "auto", "messages": [{"role": "tool", "content": PROBE_MARKER}]},
        ]
    )

    assert transcript.validate() == (
        False,
        "first provider request exposed tools outside the local read allowlist",
    )


def test_probe_rejects_missing_reasoning_content_on_tool_call_history():
    transcript = ProbeTranscript(
        requests=[
            {
                "tool_choice": "auto",
                "tools": [{"type": "function", "function": {"name": "read"}}],
            },
            {
                "tool_choice": "auto",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_anchor_probe_1"}],
                    },
                    {"role": "tool", "content": PROBE_MARKER},
                ],
            },
        ]
    )

    assert transcript.validate() == (
        False,
        "second provider request omitted reasoning_content from assistant tool call",
    )


def test_probe_stream_encodes_a_local_tool_call_as_openai_sse():
    transcript = ProbeTranscript(requests=[{"tools": [{"function": {"name": "read"}}]}])
    response = _response(transcript, transcript.requests[0])
    lines = _sse_lines(response)
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]

    assert chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read"
    assert chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"filePath": "probe.txt"}'
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert lines[-1] == "data: [DONE]\n\n"
