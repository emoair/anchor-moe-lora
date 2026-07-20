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
from anchor_mvp.tooling.tool_contract import (
    EXECUTION_TOOL_CONTRACT_V3_VERSION,
    v3_contract_descriptor,
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
                "tool_contract_version": EXECUTION_TOOL_CONTRACT_V3_VERSION,
                "tool_contract": v3_contract_descriptor(),
                "required_tests": {
                    "core": ["test/config/behavior.test.ts"],
                    "opencode": ["test/session/behavior.test.ts"],
                },
            }
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "opencode-anchor.exe"
    executable.write_bytes(b"audited windows binary")
    windows_binary = tmp_path / "windows-x64" / "opencode-anchor.exe"
    windows_binary.parent.mkdir()
    windows_binary.write_bytes(executable.read_bytes())
    linux_binary = tmp_path / "linux-x64" / "opencode-anchor"
    linux_binary.parent.mkdir()
    linux_binary.write_bytes(b"audited linux binary")
    required_tests = {
        "core": ["test/config/behavior.test.ts"],
        "opencode": ["test/session/behavior.test.ts"],
    }
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
                "tool_contract_version": EXECUTION_TOOL_CONTRACT_V3_VERSION,
                "tool_contract": v3_contract_descriptor(),
                "tests_executed": True,
                "required_tests": required_tests,
                "typecheck_executed": True,
                "binary_sha256": sha256_file(executable),
                "binary": executable.name,
                "global_install_modified": False,
            }
        ),
        encoding="utf-8",
    )
    source_contract = {
        "repository": "https://github.com/anomalyco/opencode.git",
        "baseline_commit": "1" * 40,
        "opencode_version": "1.17.18",
        "patch_sha256": sha256_file(patch),
        "patch_source_manifest_sha256": sha256_file(source),
        "bun_version": "1.3.14",
        "tool_contract_version": EXECUTION_TOOL_CONTRACT_V3_VERSION,
        "tool_contract": v3_contract_descriptor(),
        "lockfile_sha256": "2" * 64,
    }
    platform_manifests: dict[str, Path] = {}
    platform_binaries = {
        "windows-x64": windows_binary,
        "linux-x64": linux_binary,
    }
    for target, binary in platform_binaries.items():
        binary_relative = binary.relative_to(tmp_path).as_posix()
        platform_manifest = tmp_path / f"{target}.manifest.json"
        platform_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "anchor.patched-opencode.platform.v1",
                    "target": target,
                    "platform": {
                        "os": "windows" if target == "windows-x64" else "linux",
                        "arch": "x64",
                        "libc": None if target == "windows-x64" else "glibc",
                    },
                    "source": source_contract,
                    "checks": {
                        "tests_executed": True,
                        "required_tests": required_tests,
                        "typecheck_executed": True,
                        "build_smoke_executed": True,
                    },
                    "binary": {
                        "path": binary_relative,
                        "sha256": sha256_file(binary),
                    },
                    "global_install_modified": False,
                }
            ),
            encoding="utf-8",
        )
        platform_manifests[target] = platform_manifest
    bundle = tmp_path / "bundle-manifest.json"
    bundle.write_text(
        json.dumps(
            {
                "schema_version": "anchor.patched-opencode.bundle.v1",
                "source": source_contract,
                "platforms": {
                    target: {
                        "manifest": member.name,
                        "manifest_sha256": sha256_file(member),
                        "binary": {
                            "path": platform_binaries[target].relative_to(tmp_path).as_posix(),
                            "sha256": sha256_file(platform_binaries[target]),
                        },
                    }
                    for target, member in platform_manifests.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "executable": executable,
        "linux": linux_binary,
        "source": source,
        "manifest": manifest,
        "bundle": bundle,
        "members": platform_manifests,
        "windows_member": windows_binary,
    }


def test_artifact_attestation_rehashes_binary_and_manifest_before_launch(tmp_path: Path):
    artifact = _write_attested_artifact(tmp_path)
    attestation = verify_binary_attestation(
        artifact["executable"],
        patch_manifest=artifact["source"],
        linux_executable=artifact["linux"],
    )
    verified = attestation.with_behavioral_probe()

    verify_launch_identity(verified)
    artifact["executable"].write_bytes(b"changed after probe")

    with pytest.raises(ValueError, match="changed before launch"):
        verify_launch_identity(verified)


def test_artifact_attestation_rehashes_mounted_linux_before_launch(tmp_path: Path):
    artifact = _write_attested_artifact(tmp_path)
    verified = verify_binary_attestation(
        artifact["executable"],
        patch_manifest=artifact["source"],
        linux_executable=artifact["linux"],
    ).with_behavioral_probe()

    artifact["linux"].write_bytes(b"changed after probe")

    with pytest.raises(ValueError, match="changed before launch"):
        verify_launch_identity(verified)


def test_artifact_attestation_fails_closed_without_mounted_linux(tmp_path: Path):
    artifact = _write_attested_artifact(tmp_path)

    with pytest.raises(ValueError, match="mounted Linux OpenCode executable is required"):
        verify_binary_attestation(
            artifact["executable"], patch_manifest=artifact["source"]
        )


def test_artifact_attestation_requires_exact_behavioral_test_manifest(tmp_path: Path):
    artifact = _write_attested_artifact(tmp_path)
    manifest = artifact["manifest"]
    build = json.loads(manifest.read_text(encoding="utf-8"))
    build["required_tests"]["opencode"] = ["test/session/wrong.test.ts"]
    manifest.write_text(json.dumps(build), encoding="utf-8")

    with pytest.raises(ValueError, match="required_tests"):
        verify_binary_attestation(
            artifact["executable"],
            patch_manifest=artifact["source"],
            linux_executable=artifact["linux"],
        )


def test_artifact_attestation_requires_converter_tool_contract(tmp_path: Path):
    artifact = _write_attested_artifact(tmp_path)
    source = artifact["source"]
    source_data = json.loads(source.read_text(encoding="utf-8"))
    source_data["tool_contract"]["model_tools"].append("task")
    source.write_text(json.dumps(source_data), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the converter"):
        verify_binary_attestation(
            artifact["executable"],
            patch_manifest=source,
            linux_executable=artifact["linux"],
        )


@pytest.mark.parametrize(
    ("version", "nested_version"),
    [
        ("anchor.execution-tool-contract.v2", "anchor.execution-tool-contract.v2"),
        ("anchor.execution-tool-contract.v4", EXECUTION_TOOL_CONTRACT_V3_VERSION),
        (EXECUTION_TOOL_CONTRACT_V3_VERSION, "anchor.execution-tool-contract.v4"),
    ],
)
def test_artifact_attestation_rejects_legacy_or_drifted_contract_version(
    tmp_path: Path,
    version: str,
    nested_version: str,
):
    artifact = _write_attested_artifact(tmp_path)
    source = artifact["source"]
    source_data = json.loads(source.read_text(encoding="utf-8"))
    source_data["tool_contract_version"] = version
    source_data["tool_contract"]["version"] = nested_version
    source.write_text(json.dumps(source_data), encoding="utf-8")

    with pytest.raises(ValueError, match="tool contract"):
        verify_binary_attestation(
            artifact["executable"],
            patch_manifest=source,
            linux_executable=artifact["linux"],
        )


def test_artifact_attestation_requires_the_mounted_linux_bundle_member(tmp_path: Path):
    artifact = _write_attested_artifact(tmp_path)
    copied = tmp_path / "copied-opencode-linux"
    copied.write_bytes(artifact["linux"].read_bytes())

    with pytest.raises(ValueError, match="not the attested bundle member"):
        verify_binary_attestation(
            artifact["executable"],
            patch_manifest=artifact["source"],
            linux_executable=copied,
        )


def test_artifact_attestation_rejects_tampered_bundle_member_manifest(tmp_path: Path):
    artifact = _write_attested_artifact(tmp_path)
    linux_manifest = artifact["members"]["linux-x64"]
    value = json.loads(linux_manifest.read_text(encoding="utf-8"))
    value["checks"]["typecheck_executed"] = False
    linux_manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="member manifest SHA-256 mismatch"):
        verify_binary_attestation(
            artifact["executable"],
            patch_manifest=artifact["source"],
            linux_executable=artifact["linux"],
        )


def test_artifact_attestation_rejects_cross_platform_source_drift(tmp_path: Path):
    artifact = _write_attested_artifact(tmp_path)
    linux_manifest = artifact["members"]["linux-x64"]
    value = json.loads(linux_manifest.read_text(encoding="utf-8"))
    value["source"]["lockfile_sha256"] = "3" * 64
    linux_manifest.write_text(json.dumps(value), encoding="utf-8")
    bundle = json.loads(artifact["bundle"].read_text(encoding="utf-8"))
    bundle["platforms"]["linux-x64"]["manifest_sha256"] = sha256_file(linux_manifest)
    artifact["bundle"].write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(ValueError, match="source contract differs from the bundle"):
        verify_binary_attestation(
            artifact["executable"],
            patch_manifest=artifact["source"],
            linux_executable=artifact["linux"],
        )


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
