[CmdletBinding()]
param(
    [switch]$ConfirmNetwork,
    [string]$Python = "$HOME\.conda\envs\anchor-mvp\python.exe",
    [string]$WslDistro = "Ubuntu-22.04",
    [string]$Checkout = "artifacts\tooling\swebench-harness",
    [string]$Attestation = "artifacts\tooling\opencode-patched\multilang-execution-attestation.json"
)

$ErrorActionPreference = "Stop"
$Repository = "https://github.com/SWE-bench/SWE-bench.git"
$Revision = "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"
$ExpectedVersion = "4.1.0"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$CheckoutPath = [IO.Path]::GetFullPath((Join-Path $Root $Checkout))
$ArtifactsRoot = [IO.Path]::GetFullPath((Join-Path $Root "artifacts\tooling"))

if (-not $CheckoutPath.StartsWith($ArtifactsRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Checkout must stay below ignored artifacts\tooling"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python runtime not found: $Python"
}
if (-not $ConfirmNetwork -and -not (Test-Path -LiteralPath (Join-Path $CheckoutPath ".git") -PathType Container)) {
    throw "First bootstrap needs explicit -ConfirmNetwork; it never downloads instance images"
}

Set-Location $Root
if (-not (Test-Path -LiteralPath (Join-Path $CheckoutPath ".git") -PathType Container)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $CheckoutPath) | Out-Null
    & git clone --filter=blob:none --no-checkout $Repository $CheckoutPath
    if ($LASTEXITCODE -ne 0) { throw "Pinned SWE-bench clone failed" }
}

$Origin = (& git -C $CheckoutPath remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or $Origin -ne $Repository) {
    throw "Existing checkout has the wrong official origin"
}
$Head = (& git -C $CheckoutPath rev-parse HEAD 2>$null).Trim()
$WorktreeReady = Test-Path -LiteralPath (Join-Path $CheckoutPath "pyproject.toml") -PathType Leaf
if ($Head -ne $Revision -or -not $WorktreeReady) {
    if (-not $ConfirmNetwork) {
        throw "Pinned revision is absent; rerun with explicit -ConfirmNetwork"
    }
    & git -C $CheckoutPath fetch --no-tags --depth 1 origin $Revision
    if ($LASTEXITCODE -ne 0) { throw "Pinned SWE-bench fetch failed" }
    & git -C $CheckoutPath checkout --detach $Revision
    if ($LASTEXITCODE -ne 0) { throw "Pinned SWE-bench checkout failed" }
}
$Head = (& git -C $CheckoutPath rev-parse HEAD).Trim()
$Dirty = (& git -C $CheckoutPath status --porcelain --untracked-files=all)
if ($Head -ne $Revision -or $Dirty) {
    throw "Official harness checkout is not the exact clean locked revision"
}

# This is an explicit dependency install into the selected project environment.
# No image pull/prune command exists in this bootstrap script.
# SWE-bench 4.1.0's wheel omits harness/constants/fixtures/*.Cargo.lock.
# Keep the exact clean checkout as the import source so the official harness is
# usable instead of accepting a wheel that installs successfully but fails at
# import time.
& $Python -m pip install --disable-pip-version-check --editable $CheckoutPath
if ($LASTEXITCODE -ne 0) { throw "Pinned SWE-bench harness install failed" }
$InstalledVersion = (& $Python -c "import importlib.metadata; print(importlib.metadata.version('swebench'))").Trim()
if ($LASTEXITCODE -ne 0 -or $InstalledVersion -ne $ExpectedVersion) {
    throw "Installed SWE-bench version does not match the lock"
}

# Provision the official-evaluation HMAC key entirely inside WSL.  Key bytes
# never cross stdout, argv, environment variables, Windows files, or Git.
$ReceiptKeyProvision = @'
set -eu
install -d -o 0 -g 0 -m 700 /var/lib/anchor/keys
install -d -o 0 -g 0 -m 700 /var/lib/anchor/swebench-v3
install -d -o 0 -g 0 -m 700 /var/lib/anchor/swebench-v3/image-cache
for key in \
  /var/lib/anchor/keys/official-eval-hmac-v1 \
  /var/lib/anchor/keys/distillation-execution-hmac-v1
do
  if [ ! -e "$key" ]; then
    umask 077
    KEY_PATH="$key" python3 -c 'import os,pathlib,secrets; pathlib.Path(os.environ["KEY_PATH"]).write_bytes(secrets.token_bytes(64))'
  fi
  chown 0:0 "$key"
  chmod 600 "$key"
  state=$(stat -c '%a:%u:%g:%F' "$key")
  size=$(stat -c '%s' "$key")
  [ "$state" = '600:0:0:regular file' ]
  [ "$size" -ge 32 ]
done
printf 'official_eval_receipt_key=ready\n'
printf 'distillation_execution_receipt_key=ready\n'
'@
$KeyStatus = & wsl.exe --distribution $WslDistro --user root --exec sh -lc $ReceiptKeyProvision
if (
    $LASTEXITCODE -ne 0 -or
    ($KeyStatus -join "`n").Trim() -ne (
        "official_eval_receipt_key=ready`n" +
        "distillation_execution_receipt_key=ready"
    )
) {
    throw "Official evaluation receipt key provisioning failed (content-free check)"
}

& $Python scripts\tooling\build_swebench_execution_attestation.py --output $Attestation
$ProbeExit = $LASTEXITCODE
if ($ProbeExit -notin @(0, 3)) {
    throw "Execution-attestation probe failed"
}
Write-Host "Harness lock installed and local probes completed."
Write-Host "No instance image was downloaded. ready=false is expected until every remaining gate passes."
exit 0
