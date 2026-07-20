from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "anchor.ccswitch-route-manifest.v1"
PROFILE_SCHEMA = "anchor.ccswitch-route-profile.v1"
PINNED_COMMIT = "8d1b3306d09a27b9d8fc29694791d8421aba5f93"
SECRET_PATTERN = re.compile(r"(?:sk-[A-Za-z0-9_-]{12,}|ark-[A-Za-z0-9_-]{12,})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def resolve_repo_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def validate_profile(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if profile.get("schema_version") != PROFILE_SCHEMA:
        errors.append("profile schema_version mismatch")
    if profile.get("protocol") not in {"openai_responses", "openai_chat"}:
        errors.append("profile protocol is unsupported")
    base_url = str(profile.get("base_url", ""))
    if not base_url.startswith(("http://", "https://")):
        errors.append("profile base_url must be absolute HTTP(S)")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(profile.get("key_env", ""))):
        errors.append("profile key_env is invalid")
    selection = profile.get("model_selection") or {}
    if selection.get("mode") not in {"manual", "discover", "discover_or_manual"}:
        errors.append("profile model selection mode is invalid")
    if not str(selection.get("manual_model_id", "")).strip():
        errors.append("profile manual_model_id is required as a fallback")
    reasoning = profile.get("reasoning") or {}
    if reasoning.get("field") not in {"none", "reasoning.effort", "reasoning_effort"}:
        errors.append("profile reasoning field is invalid")
    if reasoning.get("field") != "none" and not str(reasoning.get("effort", "")).strip():
        errors.append("profile reasoning effort is required")
    network = profile.get("network") or {}
    network_mode = network.get("mode")
    if network_mode not in {"direct", "proxy", "inherit"}:
        errors.append("profile network mode must be direct, proxy, or inherit")
    proxy_url_env = network.get("proxy_url_env")
    if proxy_url_env is not None and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", str(proxy_url_env)
    ):
        errors.append("profile proxy_url_env is invalid")
    if network_mode == "proxy" and not proxy_url_env:
        errors.append("proxy network mode requires proxy_url_env")
    if not isinstance(network.get("require_physical_route"), bool):
        errors.append("profile require_physical_route must be boolean")
    if network.get("require_physical_route") and network_mode != "direct":
        errors.append("require_physical_route is only valid with direct mode")
    if SECRET_PATTERN.search(json.dumps(profile, ensure_ascii=False)):
        errors.append("profile appears to contain an API credential")
    return errors


def validate_manifest(
    manifest: dict[str, Any], repo_root: Path, require_ready: bool
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("manifest schema_version mismatch")
    upstream = manifest.get("upstream") or {}
    if upstream.get("commit") != PINNED_COMMIT:
        errors.append("manifest upstream commit is not pinned v3.16.5")
    patch = manifest.get("patch") or {}
    patch_path = resolve_repo_path(repo_root, str(patch.get("path", "")))
    if not patch_path.is_file():
        errors.append("manifest patch path is missing")
    elif sha256(patch_path) != patch.get("sha256"):
        errors.append("manifest patch SHA-256 mismatch")
    route = manifest.get("route") or {}
    if route.get("app_type") != "anchor-opencode":
        errors.append("manifest route app_type mismatch")
    if not str(route.get("base_url", "")).startswith("http://127.0.0.1:"):
        errors.append("manifest route must bind to loopback")
    if route.get("content_free_health_status") is not True:
        errors.append("manifest must declare content-free health/status")
    if route.get("default_network_mode") != "direct":
        errors.append("manifest route must default to direct networking")
    if route.get("supported_network_modes") != ["direct", "proxy", "inherit"]:
        errors.append("manifest route network-mode contract mismatch")
    if manifest.get("secret_persisted") is not False:
        errors.append("manifest must declare secret_persisted=false")
    if SECRET_PATTERN.search(json.dumps(manifest, ensure_ascii=False)):
        errors.append("manifest appears to contain an API credential")

    ready = manifest.get("ready") is True
    if require_ready and not ready:
        errors.append("route is not ready")
    binary = manifest.get("binary")
    if ready:
        if not isinstance(binary, dict):
            errors.append("ready manifest is missing binary metadata")
        else:
            binary_path = resolve_repo_path(repo_root, str(binary.get("path", "")))
            if not binary_path.is_file():
                errors.append("ready manifest binary is missing")
            elif sha256(binary_path) != binary.get("sha256"):
                errors.append("ready manifest binary SHA-256 mismatch")
        verified = manifest.get("verified_tests") or []
        if not verified or any(item.get("status") != "passed" for item in verified):
            errors.append("ready manifest requires all recorded tests to pass")
    elif binary is not None:
        errors.append("not-ready manifest must not advertise a binary")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the fail-closed CC Switch route")
    parser.add_argument(
        "--manifest",
        default="artifacts/tooling/ccswitch-patched/route-manifest.json",
    )
    parser.add_argument("--profile")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_repo_path(repo_root, args.manifest)
    errors: list[str] = []
    try:
        errors.extend(validate_manifest(load_json(manifest_path), repo_root, args.require_ready))
        if args.profile:
            profile_path = resolve_repo_path(repo_root, args.profile)
            errors.extend(validate_profile(load_json(profile_path)))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print("CC Switch Anchor route metadata is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
