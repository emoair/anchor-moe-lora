param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("frontend_gen", "code_review", "security_audit", "mixed_all")]
    [string]$Adapter,

    [switch]$Execute,
    [switch]$AllowModelDownload,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_adapter.ps1"

foreach ($Rank in @(16, 32, 64)) {
    $Arguments = @("-Adapter", $Adapter, "-Rank", $Rank)
    if ($Execute) { $Arguments += "-Execute" }
    if ($AllowModelDownload) { $Arguments += "-AllowModelDownload" }
    if ($Python) { $Arguments += @("-Python", $Python) }
    & $Runner @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Rank $Rank failed with exit code $LASTEXITCODE"
    }
}
