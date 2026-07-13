param(
    [ValidateSet("preflight", "smoke", "probe", "B", "C", "D", "E", "F")]
    [string]$Arm = "preflight",
    [switch]$Execute,
    [string]$AllocationManifest = "",
    [string]$LockPath = "",
    [string]$Python = "C:\Users\Air\.conda\envs\anchor-mvp\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$PreviousPythonPath = $env:PYTHONPATH
$PreviousAllocatorConfig = $env:PYTORCH_CUDA_ALLOC_CONF
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:128"
$Specialists = @("planner", "tool_policy", "frontend_gen", "frontend_review", "security_gate")
$AllowedAdaptiveRanks = @(1, 2, 3, 4, 6, 8, 12, 16)
$LockStream = $null

function Resolve-ProjectPath([string]$Relative) {
    return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $Relative))
}

function Invoke-Preflight([string]$Config) {
    & $Python -m anchor_mvp.training preflight `
        --config (Resolve-ProjectPath $Config) --dry-run
    if ($LASTEXITCODE -ne 0) {
        throw "formal-v3 preflight failed for $Config (exit $LASTEXITCODE)"
    }
}

function Get-CurrentRunContract([string]$Config, [string]$Adapter, [int]$Rank) {
    $Code = "import json,sys; from anchor_mvp.training.config import load_training_config,select_adapter; from anchor_mvp.training.manifest import config_fingerprint; c=select_adapter(load_training_config(sys.argv[1]),sys.argv[2],int(sys.argv[3])); print(json.dumps({'fingerprint':config_fingerprint(c),'max_steps':c['training']['max_steps'],'rank':c['lora']['rank'],'alpha':c['lora']['alpha'],'target_modules':sorted(c['lora']['target_modules'])}))"
    $Raw = (& $Python -c $Code (Resolve-ProjectPath $Config) $Adapter $Rank).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "could not resolve current run contract for $Adapter rank $Rank"
    }
    try {
        $Value = $Raw | ConvertFrom-Json
    }
    catch {
        throw "current run contract is not valid JSON for $Adapter rank $Rank"
    }
    if ($Value.fingerprint -notmatch '^[0-9a-f]{64}$' -or
        $Value.rank -ne $Rank -or
        $Value.alpha -ne (2 * $Rank)) {
        throw "current run contract is invalid for $Adapter rank $Rank"
    }
    return $Value
}

function Test-CompletedRun(
    [string]$Config,
    [string]$Adapter,
    [int]$Rank,
    [string]$ManifestDir,
    [string]$AdapterDir,
    [string]$Stage = "train"
) {
    $RunName = "$Adapter-r$Rank"
    $ArtifactName = if ($Stage -eq "smoke-gate") { "smoke-gate-$RunName" } else { $RunName }
    $RunManifestPath = Resolve-ProjectPath "$ManifestDir/$ArtifactName.execute.json"
    $OutputDir = Resolve-ProjectPath "$AdapterDir/$ArtifactName"
    $MetadataPath = Join-Path $OutputDir "checkpoint_metadata.json"
    $AdapterConfigPath = Join-Path $OutputDir "adapter_config.json"
    $WeightCandidates = @(
        (Join-Path $OutputDir "adapter_model.safetensors"),
        (Join-Path $OutputDir "adapter_model.bin")
    )
    if (-not (Test-Path $RunManifestPath) -or
        -not (Test-Path $MetadataPath) -or
        -not (Test-Path $AdapterConfigPath) -or
        -not ($WeightCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1)) {
        return $false
    }
    try {
        $RunManifest = Get-Content -LiteralPath $RunManifestPath -Raw | ConvertFrom-Json
        $Metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
        $AdapterConfig = Get-Content -LiteralPath $AdapterConfigPath -Raw | ConvertFrom-Json
        $Snapshot = Get-Content -LiteralPath (
            Resolve-ProjectPath "artifacts/formal_v3/dataset/manifest.json"
        ) -Raw | ConvertFrom-Json
        $Contract = Get-CurrentRunContract $Config $Adapter $Rank
        $ExpectedTargets = @($Contract.target_modules | Sort-Object) -join ','
        $ObservedTargets = @($AdapterConfig.target_modules | Sort-Object) -join ','
        $WeightPath = $WeightCandidates | Where-Object {
            Test-Path -LiteralPath $_ -PathType Leaf
        } | Select-Object -First 1
        return (
            $RunManifest.mode -eq "execute" -and
            $RunManifest.run_name -eq $RunName -and
            $RunManifest.config_sha256 -eq $Contract.fingerprint -and
            $RunManifest.preflight.passed -eq $true -and
            $RunManifest.preflight.dataset_snapshot_manifest.passed -eq $true -and
            $RunManifest.preflight.dataset_snapshot_sha256 -eq $Snapshot.snapshot_sha256 -and
            $Metadata.run_name -eq $RunName -and
            $Metadata.adapter_name -eq $Adapter -and
            $Metadata.config_sha256 -eq $Contract.fingerprint -and
            $Metadata.global_step -eq $Contract.max_steps -and
            $Metadata.trainable_parameters -eq (649216 * $Rank) -and
            $Metadata.artifact_type -eq "peft_adapter" -and
            $Metadata.merge_status -eq "unmerged" -and
            $AdapterConfig.r -eq $Rank -and
            $AdapterConfig.lora_alpha -eq (2 * $Rank) -and
            $ObservedTargets -eq $ExpectedTargets -and
            $null -ne $WeightPath -and
            (Get-Item -LiteralPath $WeightPath).Length -gt 0
        )
    }
    catch {
        return $false
    }
}

function Invoke-Adapter(
    [string]$Config,
    [string]$Adapter,
    [int]$Rank,
    [string]$ManifestDir,
    [string]$AdapterDir,
    [string]$Stage = "train"
) {
    if ($Execute) {
        if (Test-CompletedRun $Config $Adapter $Rank $ManifestDir $AdapterDir $Stage) {
            Write-Host "SKIP verified completed job: $Adapter rank $Rank"
            return
        }
        $RunName = "$Adapter-r$Rank"
        $ArtifactName = if ($Stage -eq "smoke-gate") { "smoke-gate-$RunName" } else { $RunName }
        $RunManifestPath = Resolve-ProjectPath "$ManifestDir/$ArtifactName.execute.json"
        $OutputDir = Resolve-ProjectPath "$AdapterDir/$ArtifactName"
        $ProgressDir = Resolve-ProjectPath "$AdapterDir/$ArtifactName.progress"
        $ExistingTraces = @(
            $RunManifestPath,
            $OutputDir,
            $ProgressDir
        ) | Where-Object { Test-Path -LiteralPath $_ }
        if ($ExistingTraces.Count -gt 0) {
            throw (
                "partial/stale output exists for $RunName; automatic exact resume is not supported. " +
                "Preserve safety-checkpoints, progress, execute manifest, and output for audit. " +
                "Manually isolate the entire prior run before choosing a fresh step-zero run; " +
                "this launcher never deletes or overwrites those files. Existing traces: " +
                ($ExistingTraces -join ', ')
            )
        }
    }
    $Mode = if ($Execute) { "--execute" } else { "--dry-run" }
    $Arguments = @(
        "-m", "anchor_mvp.training", $Stage,
        "--config", (Resolve-ProjectPath $Config),
        "--adapter", $Adapter,
        "--rank", $Rank,
        $Mode
    )
    if (-not $Execute) { $Arguments += "--require-data" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage $Adapter rank $Rank failed with exit code $LASTEXITCODE"
    }
}

function Read-AdaptiveRanks([string]$ExpectedArm) {
    if (-not $AllocationManifest) {
        throw "Arm $ExpectedArm requires -AllocationManifest from frozen calibration"
    }
    $Path = if ([IO.Path]::IsPathRooted($AllocationManifest)) {
        [IO.Path]::GetFullPath($AllocationManifest)
    } else {
        [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $AllocationManifest))
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "allocation manifest does not exist: $Path"
    }
    $SidecarPath = "$Path.sha256"
    if (-not (Test-Path -LiteralPath $SidecarPath -PathType Leaf)) {
        throw "allocation manifest SHA-256 sidecar is missing: $SidecarPath"
    }
    $DeclaredManifestHash = (
        Get-Content -LiteralPath $SidecarPath -Raw
    ).Split(
        [char[]]" `t`r`n", [System.StringSplitOptions]::RemoveEmptyEntries
    )[0].ToLowerInvariant()
    $ObservedManifestHash = (
        Get-FileHash -LiteralPath $Path -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($DeclaredManifestHash -ne $ObservedManifestHash) {
        throw "allocation manifest SHA-256 sidecar mismatch"
    }
    $Value = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    $Snapshot = Get-Content -LiteralPath (
        Resolve-ProjectPath "artifacts/formal_v3/dataset/manifest.json"
    ) -Raw | ConvertFrom-Json
    if ($Value.schema_version -ne "anchor.lora-allocation.v1" -or
        $Value.arm -ne $ExpectedArm -or
        $Value.dataset_snapshot_sha256 -ne $Snapshot.snapshot_sha256 -or
        $Value.mechanism_id -ne "stage_complexity_calibration_pareto_v1" -or
        $Value.base_contract_id -ne "gemma4-12b-r56820d7-bnb-nf4-doublequant-bf16-v1" -or
        $Value.parameters_per_rank -ne 649216 -or
        $Value.allocation_frozen_before_heldout -ne $true -or
        $Value.heldout_access -ne "forbidden_until_allocation_frozen" -or
        $Value.heldout_opened -ne $false -or
        $null -ne $Value.heldout_opened_at) {
        throw "allocation manifest is not frozen to this formal-v3 snapshot/arm"
    }
    if ($Value.calibration_snapshot_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw "allocation manifest calibration_snapshot_sha256 is invalid"
    }
    $ExpectedTargets = @("q_proj", "v_proj") -join ','
    $ObservedTargets = @($Value.target_modules | Sort-Object) -join ','
    if ($ObservedTargets -ne $ExpectedTargets) {
        throw "allocation manifest target_modules must remain q_proj/v_proj"
    }
    $CreatedAt = [DateTimeOffset]::MinValue
    $FrozenAt = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse([string]$Value.created_at, [ref]$CreatedAt) -or
        -not [DateTimeOffset]::TryParse(
            [string]$Value.allocation_frozen_at, [ref]$FrozenAt
        ) -or
        $FrozenAt -lt $CreatedAt) {
        throw "allocation manifest timestamps do not prove pre-heldout freeze order"
    }
    $ExpectedObjectives = if ($ExpectedArm -eq "E") {
        @(
            "maximize_per_stage_calibration_quality",
            "minimize_materialized_parameters",
            "minimize_routed_latency",
            "minimize_peak_vram"
        )
    } else {
        @(
            "maximize_per_stage_calibration_quality",
            "minimize_routed_latency",
            "minimize_peak_vram"
        )
    }
    if ((@($Value.selection_objectives) -join ',') -ne
        ($ExpectedObjectives -join ',')) {
        throw "allocation manifest selection_objectives do not match arm $ExpectedArm"
    }
    if ($null -eq $Value.selected_ranks) {
        throw "allocation manifest selected_ranks is required"
    }
    $Ranks = [ordered]@{}
    foreach ($Expert in $Specialists) {
        $Rank = $Value.selected_ranks.$Expert
        if ($Rank -notin $AllowedAdaptiveRanks) {
            throw "invalid adaptive rank for ${Expert}: $Rank"
        }
        $Ranks[$Expert] = [int]$Rank
    }
    $ObservedExperts = @(
        $Value.selected_ranks.PSObject.Properties.Name | Sort-Object
    ) -join ','
    $ExpectedExperts = @($Specialists | Sort-Object) -join ','
    if ($ObservedExperts -ne $ExpectedExperts) {
        throw "allocation manifest selected_ranks must name exactly the five specialists"
    }
    $RankSum = ($Ranks.Values | Measure-Object -Sum).Sum
    $MaterializedParameters = 649216 * $RankSum
    if ($Value.materialized_trainable_parameters -ne $MaterializedParameters) {
        throw "allocation manifest materialized parameters do not match selected ranks"
    }
    $UniqueRanks = @($Ranks.Values | Sort-Object -Unique)
    if ($ExpectedArm -eq "E" -and $UniqueRanks.Count -lt 2) {
        throw "Arm E requires a non-uniform adaptive rank allocation"
    }
    if ($ExpectedArm -eq "F" -and
        ($RankSum -ne 16 -or $MaterializedParameters -ne 10387456)) {
        throw "Arm F must exactly match B: rank sum 16 and 10,387,456 parameters"
    }
    $Attempts = @($Value.attempted_allocations)
    if ($Attempts.Count -lt 1) {
        throw "allocation manifest attempted_allocations must be non-empty"
    }
    $SelectedSignature = @(
        $Specialists | ForEach-Object { "${_}=$($Ranks[$_])" }
    ) -join ';'
    $AttemptSignatures = @{}
    $SelectedWasAttempted = $false
    foreach ($Attempt in $Attempts) {
        if ($null -eq $Attempt -or $null -eq $Attempt.selected_ranks) {
            throw "every attempted allocation must include selected_ranks"
        }
        $AttemptExperts = @(
            $Attempt.selected_ranks.PSObject.Properties.Name | Sort-Object
        ) -join ','
        if ($AttemptExperts -ne $ExpectedExperts) {
            throw "attempted allocation must name exactly the five specialists"
        }
        $AttemptSignatureParts = @()
        foreach ($Expert in $Specialists) {
            $AttemptRank = $Attempt.selected_ranks.$Expert
            if ($AttemptRank -notin $AllowedAdaptiveRanks) {
                throw "attempted allocation contains an invalid rank"
            }
            $AttemptSignatureParts += "${Expert}=$AttemptRank"
        }
        $AttemptSignature = $AttemptSignatureParts -join ';'
        if ($AttemptSignatures.ContainsKey($AttemptSignature)) {
            throw "attempted_allocations contains a duplicate allocation"
        }
        $AttemptSignatures[$AttemptSignature] = $true
        if ($AttemptSignature -eq $SelectedSignature) {
            $SelectedWasAttempted = $true
        }
    }
    if (-not $SelectedWasAttempted) {
        throw "selected_ranks is absent from attempted_allocations"
    }
    return $Ranks
}

try {
    $ConfigByArm = @{
        preflight = "configs/training/formal_v3_lowmem_common.yaml"
        smoke = "configs/training/formal_v3_lowmem_smoke.yaml"
        probe = "configs/training/formal_v3_lowmem_probe.yaml"
        B = "configs/training/formal_v3_lowmem_mixed.yaml"
        C = "configs/training/formal_v3_lowmem_common.yaml"
        D = "configs/training/formal_v3_lowmem_budget.yaml"
        E = "configs/training/formal_v3_lowmem_adaptive.yaml"
        F = "configs/training/formal_v3_lowmem_adaptive_budget.yaml"
    }
    $Config = $ConfigByArm[$Arm]
    Invoke-Preflight $Config
    if ($Arm -eq "preflight") { return }

    if ($Execute) {
        $ResolvedLockPath = if ($LockPath) {
            if ([IO.Path]::IsPathRooted($LockPath)) {
                [IO.Path]::GetFullPath($LockPath)
            } else {
                [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $LockPath))
            }
        } else {
            Resolve-ProjectPath "runs/formal-v3-training.lock"
        }
        [IO.Directory]::CreateDirectory(
            [IO.Path]::GetDirectoryName($ResolvedLockPath)
        ) | Out-Null
        try {
            $LockStream = [IO.File]::Open(
                $ResolvedLockPath,
                [IO.FileMode]::CreateNew,
                [IO.FileAccess]::Write,
                [IO.FileShare]::None
            )
        }
        catch {
            throw "another formal-v3 GPU launcher owns $ResolvedLockPath"
        }
    }

    if ($Arm -eq "smoke") {
        Invoke-Adapter $Config "frontend_gen" 16 `
            "artifacts/formal_v3/smoke/manifests" `
            "artifacts/formal_v3/smoke/adapters" "smoke-gate"
    }
    elseif ($Arm -eq "probe") {
        Invoke-Adapter $Config "frontend_gen" 16 `
            "artifacts/formal_v3/probe/manifests" `
            "artifacts/formal_v3/probe/adapters"
    }
    elseif ($Arm -eq "B") {
        Invoke-Adapter $Config "mixed_all" 16 `
            "artifacts/formal_v3/B/manifests" `
            "artifacts/formal_v3/B/adapters"
    }
    else {
        if ($Arm -eq "C") {
            $Ranks = [ordered]@{
                planner = 16; tool_policy = 16; frontend_gen = 16
                frontend_review = 16; security_gate = 16
            }
        }
        elseif ($Arm -eq "D") {
            $Ranks = [ordered]@{
                planner = 3; tool_policy = 3; frontend_gen = 4
                frontend_review = 3; security_gate = 3
            }
        }
        else {
            $Ranks = Read-AdaptiveRanks $Arm
        }
        foreach ($Entry in $Ranks.GetEnumerator()) {
            Invoke-Adapter $Config $Entry.Key $Entry.Value `
                "artifacts/formal_v3/$Arm/manifests" `
                "artifacts/formal_v3/$Arm/adapters"
        }
    }
}
finally {
    if ($null -ne $LockStream) {
        $LockPath = $LockStream.Name
        $LockStream.Dispose()
        Remove-Item -LiteralPath $LockPath -ErrorAction SilentlyContinue
    }
    if ($null -eq $PreviousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $PreviousPythonPath
    }
    if ($null -eq $PreviousAllocatorConfig) {
        Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
    } else {
        $env:PYTORCH_CUDA_ALLOC_CONF = $PreviousAllocatorConfig
    }
}
