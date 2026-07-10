param(
    [int64]$MaxTrackedBytes = 10MB
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $repoRoot

git rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Run this audit inside an initialized Git worktree."
}

$tracked = @(git ls-files)
$forbiddenPaths = @(
    '(^|/)\.env($|\.)',
    '(^|/)models/',
    '(^|/)runs/',
    '(^|/)data/.+\.jsonl$',
    '(^|/)artifacts/adapters/',
    '(^|/)id_(?:rsa|ed25519)',
    '\.(?:pem|p12|key|gguf|safetensors|ckpt|pt|pth)$'
)

$pathViolations = @()
foreach ($path in $tracked) {
    $normalized = $path -replace '\\', '/'
    if ($normalized -eq '.env.example') {
        continue
    }
    if ($forbiddenPaths | Where-Object { $normalized -match $_ }) {
        $pathViolations += $path
    }
}
if ($pathViolations.Count -gt 0) {
    Write-Error ("Forbidden tracked paths:`n" + ($pathViolations -join "`n"))
}

$largeFiles = @()
foreach ($path in $tracked) {
    $size = git cat-file -s (":" + $path) 2>$null
    if ($LASTEXITCODE -eq 0 -and [int64]$size -gt $MaxTrackedBytes) {
        $largeFiles += "$path ($size bytes)"
    }
}
if ($largeFiles.Count -gt 0) {
    Write-Error ("Tracked files exceed $MaxTrackedBytes bytes:`n" + ($largeFiles -join "`n"))
}

$secretPattern = 'sk-kimi-|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (OPENSSH |RSA |EC )?PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]+'
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$secretFiles = @(git grep --cached -I -l -E $secretPattern -- . 2>$null)
$secretFiles = @($secretFiles | Where-Object { $_ -ne 'scripts/release/prepublish_audit.ps1' })
$grepExit = $LASTEXITCODE
$ErrorActionPreference = $previousErrorAction
if ($grepExit -notin @(0, 1)) {
    throw "git grep secret scan failed with exit code $grepExit"
}
if ($secretFiles.Count -gt 0) {
    Write-Error ("Potential secrets found; contents suppressed:`n" + ($secretFiles -join "`n"))
}

if (Get-Command gitleaks -ErrorAction SilentlyContinue) {
    gitleaks protect --staged --redact --no-banner
    if ($LASTEXITCODE -ne 0) {
        throw "gitleaks rejected the staged snapshot."
    }
} else {
    Write-Host "gitleaks: not installed (built-in path/size/secret gates passed)"
}

if ($pathViolations.Count -gt 0 -or $largeFiles.Count -gt 0 -or $secretFiles.Count -gt 0) {
    exit 2
}

Write-Host "prepublish_audit=PASS tracked_files=$($tracked.Count)"
