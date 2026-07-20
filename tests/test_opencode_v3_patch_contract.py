from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anchor_mvp.tooling.tool_contract import (
    EXECUTION_TOOL_CONTRACT_V3_VERSION,
    v3_contract_descriptor,
)


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "patches" / "opencode"
PATCH = PATCH_ROOT / "v1.17.18-anchor-distillation.patch"
MANIFEST = PATCH_ROOT / "patch-manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _added_patch_lines(*paths: str) -> str:
    wanted = set(paths)
    current: str | None = None
    added: list[str] = []
    for line in PATCH.read_text(encoding="utf-8").splitlines():
        if line.startswith("+++ b/"):
            current = line.removeprefix("+++ b/")
            continue
        if current in wanted and line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return "\n".join(added)


def test_patch_manifest_pins_exact_v3_contract_and_digest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["baseline_commit"] == "b1fc8113948b518835c2a39ece49553cffe9b30c"
    assert manifest["upstream_version"] == "1.17.18"
    assert manifest["patch"] == PATCH.name
    assert manifest["patch_sha256"] == _sha256(PATCH)
    assert manifest["tool_contract_version"] == EXECUTION_TOOL_CONTRACT_V3_VERSION
    assert manifest["tool_contract"] == v3_contract_descriptor()


def test_patch_production_source_encodes_formal_v3_execution_boundary():
    source = _added_patch_lines(
        "packages/opencode/src/anchor/sandbox.ts",
        "packages/opencode/src/cli/cmd/anchor.ts",
    )

    for marker in (
        'ANCHOR_TOOL_CONTRACT = "anchor.execution-tool-contract.v3"',
        'CANONICAL_TESTBED = "/testbed"',
        'ANCHOR_ROUTE_SOCKET = "/run/anchor-route/ccswitch.sock"',
        'ANCHOR_ROUTE_HOST = "127.0.0.1"',
        "ANCHOR_ROUTE_PORT = 18080",
        '"--network",',
        '"none",',
        "parts[0].upper()==b'CONNECT'",
        "hosts != [b'127.0.0.1:18080']",
        'command: "_route-bridge"',
        "cleanupRuntimeResources",
        "exportSandboxed",
    ):
        assert marker in source

    assert "slirp4netns" not in source
    assert "--network=slirp4netns" not in source
    schema_source = _added_patch_lines("packages/schema/src/v1/session.ts")
    assert 'policy: Schema.Literal("anchor.execution-tool-contract.v3")' in schema_source
    assert 'contract: Schema.Literal("anchor.execution-tool-contract.v3")' in schema_source
    assert "InitialToolCallError.EffectSchema" in schema_source
    assert "anchor.execution-tool-contract.v2" not in schema_source


def test_patch_docs_state_v2_is_not_ready_and_reproduction_is_clean_apply():
    english = (PATCH_ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (PATCH_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")

    assert "network=none" in english
    assert "/testbed" in english
    assert "git apply --check" in english
    assert "v2 binary or bundle must remain not-ready" in english
    assert "not an official-evaluation receipt" in english
    assert "validator_version_sha256" in english
    assert "validation_state_sha256" in english
    assert "final changed paths" in english
    assert "anchor.execution-tool-contract.v3" in chinese
    assert "/testbed" in chinese
    assert "not-ready" in chinese
    assert "不是正式评测" in chinese
    assert "validator_version_sha256" in chinese
    assert "validation_state_sha256" in chinese
    assert "最终\n变更路径的精确哈希集合" in chinese
