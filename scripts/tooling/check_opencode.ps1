param(
    [switch]$RunNpmDryRun
)

$ErrorActionPreference = "Stop"

Write-Output "OpenCode preflight (no API calls, no installation)"

foreach ($name in @("opencode", "node", "npm", "wsl")) {
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        Write-Output ("{0}: MISSING" -f $name)
    } else {
        Write-Output ("{0}: {1}" -f $name, $command.Source)
    }
}

if (Get-Command node -ErrorAction SilentlyContinue) {
    node --version
}
if (Get-Command npm -ErrorAction SilentlyContinue) {
    npm --version
    npm config get registry
    npm config get prefix
}

Write-Output "Official Windows npm install command (NOT executed):"
Write-Output "npm install -g opencode-ai"
Write-Output "Safe metadata-resolution dry-run command:"
Write-Output "npm install -g opencode-ai --dry-run --ignore-scripts"

if ($RunNpmDryRun) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "npm is unavailable"
    }
    npm install -g opencode-ai --dry-run --ignore-scripts
}
