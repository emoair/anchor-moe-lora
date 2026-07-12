#!/usr/bin/env bash
# Build the Linux x64 member of the pinned OpenCode patch bundle inside WSL.
#
# This script intentionally keeps its checkout and Bun cache on the WSL ext4 volume.
# The repository itself may live under /mnt/d, but building node_modules on DrvFS/9p is
# both slower and prone to cross-platform native dependency contamination.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_patched_opencode_wsl.sh [options]

  --checkout-root PATH  Fresh Linux-only checkout (default: ~/.cache/anchor-moe-lora/opencode-build/worktrees/linux-x64)
  --bun-path PATH       Required Bun 1.3.14 Linux x64 executable (or BUN_PATH)
  --bun-sha256 HASH     Optional expected SHA-256 for the Bun executable
  --skip-install        Do not run bun install (manifest records the skip)
  --skip-tests          Do not run focused tests (manifest records the skip)
  --skip-typecheck      Do not run the OpenCode typecheck (manifest records the skip)

No option resets, cleans, or reuses a dirty checkout. Pick a new --checkout-root
after an interrupted build. Build Windows separately with build_patched_opencode.ps1,
then run assemble_opencode_bundle.py after both platform manifests exist.
EOF
}

checkout_root="${CHECKOUT_ROOT:-$HOME/.cache/anchor-moe-lora/opencode-build/worktrees/linux-x64}"
bun_path="${BUN_PATH:-}"
bun_sha256="${BUN_SHA256:-}"
skip_install=0
skip_tests=0
skip_typecheck=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkout-root) checkout_root="$2"; shift 2 ;;
    --bun-path) bun_path="$2"; shift 2 ;;
    --bun-sha256) bun_sha256="$2"; shift 2 ;;
    --skip-install) skip_install=1; shift ;;
    --skip-tests) skip_tests=1; shift ;;
    --skip-typecheck) skip_typecheck=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command is missing: $1" >&2
    exit 2
  }
}

run_in() {
  local directory="$1"
  shift
  (
    cd "$directory"
    "$@"
  )
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

manifest_value() {
  local key="$1"
  python3 - "$patch_manifest" "$key" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8-sig") as handle:
    value = json.load(handle)
for part in sys.argv[2].split("."):
    value = value[part]
if not isinstance(value, (str, int, float)):
    raise SystemExit(f"manifest value is not scalar: {sys.argv[2]}")
print(value)
PY
}

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This script must run on Linux x86_64 (for example Ubuntu WSL2)." >&2
  exit 2
fi
# WSL normally appends Windows PATH entries.  Do not let Linux tests discover PE
# shims such as WindowsApps/rg; use only native Linux tool locations.
export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
require_command git
require_command python3
if (( ! skip_tests )); then
  require_command rg
fi
if (( ! skip_install )); then
  for command in make gcc g++; do
    require_command "$command"
  done
fi

project_root="${ANCHOR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
patch_manifest="$project_root/patches/opencode/patch-manifest.json"
output_root="$project_root/artifacts/tooling/opencode-patched"
target="linux-x64"

[[ -f "$patch_manifest" ]] || { echo "Patch source manifest is missing: $patch_manifest" >&2; exit 2; }
repository="$(manifest_value repository)"
baseline_commit="$(manifest_value baseline_commit)"
patch_name="$(manifest_value patch)"
expected_patch_sha="$(manifest_value patch_sha256 | tr '[:upper:]' '[:lower:]')"
expected_bun_version="$(manifest_value bun_version)"
upstream_version="$(manifest_value upstream_version)"
tool_contract_version="$(manifest_value tool_contract_version)"
patch_path="$(dirname "$patch_manifest")/$patch_name"

[[ "$repository" == "https://github.com/anomalyco/opencode.git" ]] || { echo "Patch manifest repository is not the audited upstream" >&2; exit 2; }
[[ "$baseline_commit" =~ ^[0-9a-f]{40}$ && "$expected_patch_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "Patch manifest has invalid source identity" >&2; exit 2; }
[[ -f "$patch_path" ]] || { echo "Patch is missing: $patch_path" >&2; exit 2; }
patch_sha="$(sha256_file "$patch_path")"
[[ "$patch_sha" == "$expected_patch_sha" ]] || { echo "Patch SHA-256 mismatch" >&2; exit 2; }

if [[ -e "$checkout_root" ]]; then
  [[ -e "$checkout_root/.git" ]] || { echo "CheckoutRoot exists but is not a Git checkout: $checkout_root" >&2; exit 2; }
  existing_origin="$(git -c core.autocrlf=false -C "$checkout_root" remote get-url origin)"
  [[ "$existing_origin" == "$repository" ]] || { echo "Checkout origin is not the audited repository: $existing_origin" >&2; exit 2; }
  [[ -z "$(git -c core.autocrlf=false -C "$checkout_root" status --porcelain=v1)" ]] || {
    echo "Checkout is dirty. Use a fresh --checkout-root; this script never resets user work." >&2
    exit 2
  }
else
  mkdir -p "$(dirname "$checkout_root")"
  git -c core.autocrlf=false clone --depth 1 --branch "v$upstream_version" --filter=blob:none --no-checkout "$repository" "$checkout_root"
fi

if ! git -c core.autocrlf=false -C "$checkout_root" cat-file -e "${baseline_commit}^{commit}" 2>/dev/null; then
  git -c core.autocrlf=false -C "$checkout_root" fetch --depth 1 origin "$baseline_commit"
fi
git -c core.autocrlf=false -C "$checkout_root" checkout --detach "$baseline_commit"
actual_commit="$(git -c core.autocrlf=false -C "$checkout_root" rev-parse HEAD)"
[[ "$actual_commit" == "$baseline_commit" ]] || { echo "Baseline mismatch: $actual_commit" >&2; exit 2; }
git -c core.autocrlf=false -C "$checkout_root" apply --check "$patch_path"
git -c core.autocrlf=false -C "$checkout_root" apply "$patch_path"

[[ -n "$bun_path" ]] || { echo "Provide --bun-path or BUN_PATH; no global Bun fallback is used." >&2; exit 2; }
[[ -x "$bun_path" ]] || { echo "Bun path is not executable: $bun_path" >&2; exit 2; }
bun_path="$(readlink -f "$bun_path")"
bun_version="$($bun_path --version)"
[[ "$bun_version" == "$expected_bun_version" ]] || { echo "Audited build requires Bun $expected_bun_version; got '$bun_version'" >&2; exit 2; }
bun_sha="$(sha256_file "$bun_path")"
if [[ -n "$bun_sha256" ]]; then
  bun_sha256="${bun_sha256,,}"
  [[ "$bun_sha256" =~ ^[0-9a-f]{64}$ && "$bun_sha" == "$bun_sha256" ]] || { echo "Bun SHA-256 does not match --bun-sha256" >&2; exit 2; }
fi

export BUN_CONFIG_MAX_HTTP_REQUESTS=4
export BUN_INSTALL_CACHE_DIR="${BUN_INSTALL_CACHE_DIR:-$HOME/.cache/anchor-moe-lora/opencode-build/bun-cache/$target}"
mkdir -p "$BUN_INSTALL_CACHE_DIR"
if (( ! skip_install )); then
  run_in "$checkout_root" "$bun_path" install --frozen-lockfile
fi

mapfile -t core_tests < <(python3 - "$patch_manifest" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    source = json.load(handle)
for item in source["required_tests"]["core"]:
    print(item)
PY
)
mapfile -t opencode_tests < <(python3 - "$patch_manifest" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8-sig") as handle:
    source = json.load(handle)
for item in source["required_tests"]["opencode"]:
    print(item)
PY
)
if (( ! skip_tests )); then
  run_in "$checkout_root/packages/core" "$bun_path" test --timeout 15000 "${core_tests[@]}"
  run_in "$checkout_root/packages/opencode" "$bun_path" test --timeout 15000 "${opencode_tests[@]}"
fi
if (( ! skip_typecheck )); then
  run_in "$checkout_root" "$bun_path" run --cwd packages/opencode typecheck
fi
run_in "$checkout_root" "$bun_path" run packages/opencode/script/build.ts --single --skip-install --skip-embed-web-ui

built="$checkout_root/packages/opencode/dist/opencode-linux-x64/bin/opencode"
[[ -f "$built" ]] || { echo "Build completed without the expected Linux x64 binary" >&2; exit 2; }
mkdir -p "$output_root/linux-x64"
destination="$output_root/linux-x64/opencode-anchor"
cp "$built" "$destination"
chmod 0755 "$destination"
binary_sha="$(sha256_file "$destination")"
patch_manifest_sha="$(sha256_file "$patch_manifest")"
lockfile_sha="$(sha256_file "$checkout_root/bun.lock")"
platform_manifest="$output_root/linux-x64.manifest.json"

python3 - "$patch_manifest" "$platform_manifest" "$target" "$binary_sha" "$bun_version" "$bun_sha" "$patch_sha" "$patch_manifest_sha" "$lockfile_sha" "$skip_install" "$skip_tests" "$skip_typecheck" <<'PY'
import json
import sys

(
    patch_manifest,
    destination,
    target,
    binary_sha,
    bun_version,
    bun_sha,
    patch_sha,
    patch_manifest_sha,
    lockfile_sha,
    skip_install,
    skip_tests,
    skip_typecheck,
) = sys.argv[1:]
with open(patch_manifest, encoding="utf-8-sig") as handle:
    source = json.load(handle)
platform = {
    "schema_version": "anchor.patched-opencode.platform.v1",
    "target": target,
    "platform": {"os": "linux", "arch": "x64", "libc": "glibc"},
    "source": {
        "repository": source["repository"],
        "baseline_commit": source["baseline_commit"],
        "opencode_version": source["upstream_version"],
        "patch_sha256": patch_sha,
        "patch_source_manifest_sha256": patch_manifest_sha,
        "bun_version": bun_version,
        "tool_contract_version": source["tool_contract_version"],
        "tool_contract": source["tool_contract"],
        "lockfile_sha256": lockfile_sha,
    },
    "bun": {"version": bun_version, "sha256": bun_sha},
    "node_gyp_version": None,
    "install": {"executed": skip_install == "0", "linker": "default", "cache_scope": target},
    "checks": {
        "tests_executed": skip_tests == "0",
        "required_tests": source["required_tests"],
        "test_exclusions": [],
        "typecheck_executed": skip_typecheck == "0",
        "build_smoke_executed": True,
    },
    "binary": {"path": "linux-x64/opencode-anchor", "sha256": binary_sha},
    "global_install_modified": False,
}
with open(destination, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(platform, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
PY

printf '%s\n' "$destination"
