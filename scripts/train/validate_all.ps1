param([string]$Python = "")

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_adapter.ps1"

foreach ($Adapter in @("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate", "mixed_all")) {
    $Arguments = @("-Adapter", $Adapter, "-Rank", 16)
    if ($Python) { $Arguments += @("-Python", $Python) }
    & $Runner @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Dry run failed for $Adapter"
    }
}
