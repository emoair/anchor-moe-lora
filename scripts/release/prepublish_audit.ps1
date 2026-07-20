param(
    # Publication artifacts are sharded so every tracked file is strictly
    # smaller than 50 MiB.  Keep the Git gate aligned with that contract.
    [int64]$MaxTrackedBytes = 50MB
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
    if ($LASTEXITCODE -eq 0 -and [int64]$size -ge $MaxTrackedBytes) {
        $largeFiles += "$path ($size bytes)"
    }
}
if ($largeFiles.Count -gt 0) {
    Write-Error ("Tracked files exceed $MaxTrackedBytes bytes:`n" + ($largeFiles -join "`n"))
}

# Keep token boundaries aligned with the public-dataset scrubber.  Without the
# leading boundary, an ordinary identifier such as "spark-<uuid>" contains the
# substring "ark-<uuid>" and is a false positive rather than an Ark credential.
$secretPattern = '(^|[^A-Za-z0-9])sk-kimi-[A-Za-z0-9_-]{12,}([^A-Za-z0-9_-]|$)|(^|[^A-Za-z0-9])sk-proj-[A-Za-z0-9_-]{12,}([^A-Za-z0-9_-]|$)|(^|[^A-Za-z0-9])sk-[A-Za-z0-9]{40,}([^A-Za-z0-9]|$)|(^|[^A-Za-z0-9])ark-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(-[A-Za-z0-9_-]+)?([^A-Za-z0-9_-]|$)|(^|[^A-Za-z0-9])AKIA[0-9A-Z]{16}([^A-Za-z0-9]|$)|(^|[^A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}([^A-Za-z0-9]|$)|-----BEGIN (OPENSSH |RSA |EC )?PRIVATE KEY-----|(^|[^A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]+([^A-Za-z0-9-]|$)'
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
