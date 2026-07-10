param(
    [switch]$Execute,
    [string]$Python = "C:\Users\Air\.conda\envs\anchor-mvp\python.exe"
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_adapter.ps1"
$Ranks = [ordered]@{
    planner = 3
    tool_policy = 3
    frontend_gen = 4
    frontend_review = 3
    security_gate = 3
}

foreach ($Entry in $Ranks.GetEnumerator()) {
    $Arguments = @("-Adapter", $Entry.Key, "-Rank", $Entry.Value, "-Python", $Python)
    if ($Execute) { $Arguments += "-Execute" }
    & $Runner @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$($Entry.Key) rank $($Entry.Value) failed with exit code $LASTEXITCODE"
    }
}
