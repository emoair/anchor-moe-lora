param(
    [ValidateSet("smoke", "B", "C", "D", "all")]
    [string]$Arm = "all",
    [string]$Python = "C:\Users\Air\.conda\envs\anchor-mvp\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$PreviousAllocatorConfig = $env:PYTORCH_CUDA_ALLOC_CONF
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:128"

function Invoke-Adapter([string]$Config, [string]$Adapter, [int]$Rank, [string]$Stage = "train") {
    & $Python -m anchor_mvp.training $Stage `
        --config (Join-Path $ProjectRoot $Config) `
        --adapter $Adapter `
        --rank $Rank `
        --execute
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage $Adapter rank $Rank failed with exit code $LASTEXITCODE"
    }
}

try {
    if ($Arm -in @("smoke", "all")) {
        Invoke-Adapter "configs/training/formal_v1_smoke.yaml" "frontend_gen" 16 "smoke-gate"
    }
    if ($Arm -in @("B", "all")) {
        Invoke-Adapter "configs/training/formal_v1_mixed.yaml" "mixed_all" 16
    }
    if ($Arm -in @("C", "all")) {
        foreach ($Adapter in @("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate")) {
            Invoke-Adapter "configs/training/formal_v1_common.yaml" $Adapter 16
        }
    }
    if ($Arm -in @("D", "all")) {
        $Ranks = [ordered]@{
            planner = 3
            tool_policy = 3
            frontend_gen = 4
            frontend_review = 3
            security_gate = 3
        }
        foreach ($Entry in $Ranks.GetEnumerator()) {
            Invoke-Adapter "configs/training/formal_v1_budget.yaml" $Entry.Key $Entry.Value
        }
    }
}
finally {
    if ($null -eq $PreviousAllocatorConfig) {
        Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
    } else {
        $env:PYTORCH_CUDA_ALLOC_CONF = $PreviousAllocatorConfig
    }
}
