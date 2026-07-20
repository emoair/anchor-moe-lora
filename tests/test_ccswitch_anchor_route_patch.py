from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.tooling.validate_ccswitch_route import validate_manifest, validate_profile


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "cc-switch" / "v3.16.5-anchor-opencode-route.patch"
MANIFEST = ROOT / "artifacts" / "tooling" / "ccswitch-patched" / "route-manifest.json"
PROFILES = [
    ROOT / "patches" / "cc-switch" / "profiles" / "glm-5.2-max.json",
    ROOT / "patches" / "cc-switch" / "profiles" / "kimi-k3-max.json",
]
UPSTREAM = ROOT / "runs" / "cc-switch-build" / "upstream-v3.16.5"
LAUNCHER = ROOT / "scripts" / "tooling" / "start_patched_ccswitch_route.ps1"


def test_local_route_manifest_is_valid_for_its_recorded_state() -> None:
    if not MANIFEST.is_file():
        pytest.skip("local CC Switch build artifact is intentionally Git-ignored")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert not validate_manifest(manifest, ROOT, require_ready=False)
    errors = validate_manifest(manifest, ROOT, require_ready=True)
    if manifest.get("ready") is True:
        assert not errors
        assert isinstance(manifest.get("binary"), dict)
    else:
        assert "route is not ready" in errors
        assert manifest["binary"] is None


def test_patch_hash_and_runtime_contract() -> None:
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == manifest["patch"]["sha256"]
    text = PATCH.read_text(encoding="utf-8")
    for marker in [
        'APP_TYPE: &str = "anchor-opencode"',
        '"/anchor/v1/responses"',
        '"/anchor/v1/chat/completions"',
        '"api_key_env"',
        'ANCHOR_ROUTE_BASE_URL',
        'ANCHOR_ROUTE_MODEL',
        'ANCHOR_ROUTE_API_FORMAT',
        'ANCHOR_ROUTE_REASONING_FIELD',
        'ANCHOR_ROUTE_REASONING_EFFORT',
        'ANCHOR_ROUTE_PRICE_INPUT_PER_MILLION',
        'ANCHOR_ROUTE_NETWORK_MODE',
        'ANCHOR_ROUTE_PROXY_URL_ENV',
        '"direct" =>',
        '"proxy" =>',
        '"inherit" =>',
        'reasoning_override("reasoning.effort", "max")',
    ]:
        assert marker in text


@pytest.mark.parametrize("profile_path", PROFILES)
def test_formal_profiles_are_valid_and_lock_literal_max(profile_path: Path) -> None:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert not validate_profile(profile)
    assert profile["model_selection"]["force_manual_model"] is True
    assert profile["reasoning"] == {"field": "reasoning.effort", "effort": "max"}
    assert profile["network"] == {
        "mode": "direct",
        "proxy_url_env": None,
        "require_physical_route": True,
    }


def test_pricing_is_evidence_bounded() -> None:
    glm = json.loads(PROFILES[0].read_text(encoding="utf-8"))
    assert glm["pricing"] == {
        "input_per_million": "1.4",
        "output_per_million": "4.4",
        "cache_read_per_million": "0.26",
        "cache_creation_per_million": "0",
        "source": "local CC Switch/models.dev snapshot; glm-5.2 family match",
        "verified_at": "2026-07-17",
    }
    kimi = json.loads(PROFILES[1].read_text(encoding="utf-8"))
    assert kimi["pricing"] is None


def test_kimi_k3_formal_teacher_uses_ark_responses_endpoint() -> None:
    profile = json.loads(PROFILES[1].read_text(encoding="utf-8"))
    assert profile["protocol"] == "openai_responses"
    assert profile["base_url"] == "https://ark.cn-beijing.volces.com/api/coding/v3"
    assert profile["key_env"] == "ARK_CODING_API_KEY"
    assert profile["model_selection"]["manual_model_id"] == "kimi-k3"


def test_formal_teacher_profiles_use_distinct_coordinator_ports() -> None:
    glm = json.loads(PROFILES[0].read_text(encoding="utf-8"))
    kimi = json.loads(PROFILES[1].read_text(encoding="utf-8"))
    assert glm["route"]["port"] == 15731
    assert kimi["route"]["port"] == 15732
    assert glm["route"]["port"] != kimi["route"]["port"]


def test_launcher_fails_closed_when_direct_route_is_not_physical() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for marker in [
        "Assert-PhysicalProviderRoute",
        "Find-NetRoute -RemoteIPAddress",
        "Get-NetAdapter -Physical",
        "Direct mode is not physically direct",
        "This launcher will not alter system routes",
    ]:
        assert marker in text


def test_patch_applies_to_exact_pinned_source(tmp_path: Path) -> None:
    if not (UPSTREAM / ".git").is_dir() or not shutil.which("git"):
        pytest.skip("local pinned upstream checkout is unavailable")
    clean = tmp_path / "cc-switch-v3.16.5"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(UPSTREAM), str(clean)],
        check=True,
    )
    head = subprocess.check_output(["git", "-C", str(clean), "rev-parse", "HEAD"], text=True).strip()
    assert head == "8d1b3306d09a27b9d8fc29694791d8421aba5f93"
    subprocess.run(["git", "-C", str(clean), "apply", "--check", str(PATCH)], check=True)
