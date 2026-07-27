[CmdletBinding()]
param(
    [ValidateSet("SmokeOnly", "Full")]
    [string]$Mode = "SmokeOnly",

    [Parameter(Mandatory = $true)]
    [string]$TeacherFinalBinding,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{64}$")]
    [string]$TeacherFinalBindingSha256,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedGpuUuid,

    [string]$ProducerRepository = "",

    [string]$SmokeRunReceipt = "",

    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,

    [string]$ConfigPath = (
        "configs\training\" +
        "gemma3_1b_it_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.yaml"
    )
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\..")
)
$CanonicalLockRelative = "runs/formal-v3-training.lock"
$CanonicalLockPath = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot $CanonicalLockRelative)
)
$ConfigFullPath = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot $ConfigPath)
)
$TeacherFullPath = [IO.Path]::GetFullPath($TeacherFinalBinding)
$RunnerPath = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot (
        "scripts\research\" +
        "run_gemma3_chat_unbalanced_v2_multiarm_q8_qlora_runtime_v2.py"
    ))
)
$HelperPath = [IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot "scripts\research\gemma3_q8_launcher_helpers_v2.ps1")
)

if (-not [IO.File]::Exists($PythonExecutable) -or
    -not [IO.File]::Exists($ConfigFullPath) -or
    -not [IO.File]::Exists($TeacherFullPath) -or
    -not [IO.File]::Exists($RunnerPath) -or
    -not [IO.File]::Exists($HelperPath)) {
    throw "required runtime input is missing"
}
if ([string]::IsNullOrWhiteSpace($ExpectedGpuUuid) -or
    $ExpectedGpuUuid -eq "UNBOUND") {
    throw "ExpectedGpuUuid must be explicitly bound"
}
if ($Mode -eq "Full" -and [string]::IsNullOrWhiteSpace($SmokeRunReceipt)) {
    throw "Full mode requires a passed smoke run receipt"
}
if ($Mode -eq "SmokeOnly" -and -not [string]::IsNullOrWhiteSpace(
    $SmokeRunReceipt
)) {
    throw "SmokeOnly must not consume a prior checkpoint or receipt"
}

$ValidationArguments = @(
    "-I",
    $RunnerPath,
    "--config",
    $ConfigFullPath,
    "--validate"
)
if (-not [string]::IsNullOrWhiteSpace($ProducerRepository)) {
    $ValidationArguments += @(
        "--producer-repository",
        [IO.Path]::GetFullPath($ProducerRepository)
    )
}
$ValidationRaw = & $PythonExecutable @ValidationArguments
if ($LASTEXITCODE -ne 0) {
    throw "model-free runtime validation failed"
}
try {
    $Validation = $ValidationRaw | ConvertFrom-Json
}
catch {
    throw "model-free runtime validation output is invalid"
}
if ($Validation.status -ne "passed" -or
    $Validation.gpu_execution_ready -ne $true) {
    throw ("GPU execution remains fail-closed: " +
        [string]$Validation.gpu_execution_blocker)
}

$TeacherPreflightArguments = @(
    "-I",
    $RunnerPath,
    "--config",
    $ConfigFullPath,
    "--preflight-teacher-final",
    "--teacher-final-binding",
    $TeacherFullPath,
    "--teacher-final-binding-sha256",
    $TeacherFinalBindingSha256
)
if (-not [string]::IsNullOrWhiteSpace($ProducerRepository)) {
    $TeacherPreflightArguments += @(
        "--producer-repository",
        [IO.Path]::GetFullPath($ProducerRepository)
    )
}
$TeacherPreflightRaw = & $PythonExecutable @TeacherPreflightArguments
if ($LASTEXITCODE -ne 0) {
    throw "Teacher FINAL physical preflight failed"
}
try {
    $TeacherPreflight = $TeacherPreflightRaw | ConvertFrom-Json
}
catch {
    throw "Teacher FINAL physical preflight output is invalid"
}
if ($TeacherPreflight.status -ne "passed" -or
    $TeacherPreflight.operation -ne "teacher_final_physical_preflight" -or
    $TeacherPreflight.teacher_final.binding_sha256 -ne
    $TeacherFinalBindingSha256 -or
    $TeacherPreflight.all_binding_manifest_receipt_attestation_shard_identities_authenticated -ne
    $true -or
    $TeacherPreflight.gpu_requested -ne $false) {
    throw "Teacher FINAL physical preflight evidence is incomplete"
}

. $HelperPath

function Get-LowerSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-BoundJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Value
    )
    $Json = $Value | ConvertTo-Json -Depth 30 -Compress
    $Bytes = [Text.UTF8Encoding]::new($false).GetBytes($Json + "`n")
    $Stream = [IO.File]::Open(
        $Path,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None
    )
    try {
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    }
    finally {
        $Stream.Dispose()
    }
    $Sha = Get-LowerSha256 -Path $Path
    $Sidecar = "$Sha  $([IO.Path]::GetFileName($Path))`n"
    [IO.File]::WriteAllText(
        "$Path.sha256",
        $Sidecar,
        [Text.UTF8Encoding]::new($false)
    )
    return $Sha
}

function Get-GpuSample {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][int]$Ordinal
    )
    $Line = & nvidia-smi `
        --query-gpu=index,uuid,driver_model.current,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi GPU query failed"
    }
    $Rows = @($Line | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($Rows.Count -ne 1) {
        throw "exactly one GPU is required"
    }
    $Cells = @($Rows[0].Split(",") | ForEach-Object { $_.Trim() })
    if ($Cells.Count -ne 8) {
        throw "nvidia-smi GPU row is invalid"
    }
    $Sample = [ordered]@{
        phase = $Phase
        ordinal = $Ordinal
        observed_at_utc = [DateTime]::UtcNow.ToString("o")
        index = [int]$Cells[0]
        uuid = $Cells[1]
        driver_model = $Cells[2]
        memory_total_mib = [int]$Cells[3]
        memory_used_mib = [int]$Cells[4]
        memory_free_mib = [int]$Cells[5]
        utilization_percent = [int]$Cells[6]
        temperature_c = [int]$Cells[7]
        external_compute_process_policy = "observe_only"
    }
    if ($Sample.index -ne 0 -or
        $Sample.uuid -ne $ExpectedGpuUuid -or
        $Sample.memory_total_mib -ne 12288 -or
        $Sample.memory_used_mib -gt 2048 -or
        $Sample.memory_free_mib -lt 8192 -or
        $Sample.utilization_percent -gt 15 -or
        $Sample.temperature_c -gt 75) {
        throw "$Phase GPU idle/identity/temperature gate failed"
    }
    return $Sample
}

if ([IO.File]::Exists($CanonicalLockPath)) {
    throw "canonical GPU training lock already exists"
}
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($CanonicalLockPath)) |
    Out-Null

$PreLockSamples = @()
foreach ($Ordinal in 1..3) {
    $PreLockSamples += Get-GpuSample -Phase "pre_lock" -Ordinal $Ordinal
    if ($Ordinal -lt 3) {
        Start-Sleep -Seconds 1
    }
}

$RunId = [Guid]::NewGuid().ToString("N")
$Timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfff")
$LaunchRoot = Join-Path $ProjectRoot (
    "runs\gemma3-chat-unbalanced-v2-multiarm-q8-runtime-v2\" +
    "$Timestamp-$RunId-launch"
)
[IO.Directory]::CreateDirectory($LaunchRoot) | Out-Null
$ConfigSha = Get-LowerSha256 -Path $ConfigFullPath
$TeacherSha = $TeacherFinalBindingSha256
$TeacherSidecar = "$TeacherFullPath.sha256"
if (-not [IO.File]::Exists($TeacherSidecar)) {
    throw "Teacher FINAL binding sidecar is missing"
}
$ExpectedTeacherSidecar = "$TeacherSha  $([IO.Path]::GetFileName($TeacherFullPath))`n"
$ObservedTeacherSidecar = [IO.File]::ReadAllText($TeacherSidecar)
if ($ObservedTeacherSidecar -ne $ExpectedTeacherSidecar) {
    throw "Teacher FINAL binding sidecar mismatch"
}
$IdentityJson = & $PythonExecutable -I $RunnerPath `
    --config $ConfigFullPath --emit-implementation-identity
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($IdentityJson)) {
    throw "runtime implementation identity preflight failed"
}
$ImplementationIdentity = $IdentityJson | ConvertFrom-Json
$ImplementationSetSha = [string]$ImplementationIdentity.set_sha256
if ($ImplementationSetSha -notmatch "^[0-9a-f]{64}$") {
    throw "runtime implementation identity is invalid"
}

$LockStream = $null
try {
    $LockStream = [IO.FileStream]::new(
        $CanonicalLockPath,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::ReadWrite,
        [IO.FileShare]::None,
        4096,
        [IO.FileOptions]::DeleteOnClose -bor [IO.FileOptions]::WriteThrough
    )
    $LockBody = [Text.UTF8Encoding]::new($false).GetBytes(
        "run_id=$RunId`nmode=$Mode`nlauncher_pid=$PID`n"
    )
    $LockStream.Write($LockBody, 0, $LockBody.Length)
    $LockStream.Flush($true)
    Assert-AnchorGemmaQ8CanonicalLockHeld `
        -CanonicalLockStream $LockStream `
        -CanonicalLockPath $CanonicalLockPath

    $PostLockSamples = @()
    foreach ($Ordinal in 1..3) {
        $PostLockSamples += Get-GpuSample -Phase "post_lock" -Ordinal $Ordinal
        if ($Ordinal -lt 3) {
            Start-Sleep -Seconds 1
        }
    }

    $RuntimeMode = if ($Mode -eq "SmokeOnly") { "smoke_only" } else { "full" }
    $Lease = [ordered]@{
        schema_version = (
            "anchor.gemma3-chat-unbalanced-v2-multiarm-q8-" +
            "execution-lease.v2"
        )
        status = "active"
        run_id = $RunId
        mode = $RuntimeMode
        launcher_pid = $PID
        config_sha256 = $ConfigSha
        teacher_binding_sha256 = $TeacherSha
        implementation_set_sha256 = $ImplementationSetSha
        canonical_lock_path = $CanonicalLockRelative.Replace("\", "/")
        concurrency = 1
        resume = $false
        fresh_base_per_phase = $true
        fresh_adapter_per_phase = $true
        automatic_retry = $false
        created_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    $LeasePath = Join-Path $LaunchRoot "execution_lease.json"
    $LeaseSha = Write-BoundJson -Path $LeasePath -Value $Lease
    $Attestation = [ordered]@{
        schema_version = (
            "anchor.gemma3-chat-unbalanced-v2-multiarm-q8-" +
            "gpu-attestation.v2"
        )
        status = "passed"
        run_id = $RunId
        mode = $RuntimeMode
        config_sha256 = $ConfigSha
        teacher_binding_sha256 = $TeacherSha
        implementation_set_sha256 = $ImplementationSetSha
        expected_gpu_uuid = $ExpectedGpuUuid
        canonical_lock_path = $CanonicalLockRelative.Replace("\", "/")
        external_compute_process_policy = "observe_only"
        pre_lock_samples = $PreLockSamples
        post_lock_samples = $PostLockSamples
        created_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    $AttestationPath = Join-Path $LaunchRoot "gpu_attestation.json"
    $AttestationSha = Write-BoundJson -Path $AttestationPath -Value $Attestation

    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:HF_DATASETS_OFFLINE = "1"
    $env:TOKENIZERS_PARALLELISM = "false"
    $env:ANCHOR_GEMMA_GPU_UUID = $ExpectedGpuUuid
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    if (-not [string]::IsNullOrWhiteSpace($ProducerRepository)) {
        $env:ANCHOR_GEMMA3_UNBALANCED_V2_PRODUCER_REPOSITORY = (
            [IO.Path]::GetFullPath($ProducerRepository)
        )
    }

    $Arguments = @(
        "-I",
        $RunnerPath,
        "--config",
        $ConfigFullPath,
        "--run-id",
        $RunId,
        "--teacher-final-binding",
        $TeacherFullPath,
        "--teacher-final-binding-sha256",
        $TeacherSha,
        "--execution-lease",
        $LeasePath,
        "--execution-lease-sha256",
        $LeaseSha,
        "--gpu-attestation",
        $AttestationPath,
        "--gpu-attestation-sha256",
        $AttestationSha
    )
    if ($Mode -eq "SmokeOnly") {
        $Arguments += "--execute-smoke-only"
    }
    else {
        $SmokeFullPath = [IO.Path]::GetFullPath($SmokeRunReceipt)
        if (-not [IO.File]::Exists($SmokeFullPath) -or
            -not [IO.File]::Exists("$SmokeFullPath.sha256")) {
            throw "passed smoke run receipt and sidecar are required"
        }
        $SmokeSha = Get-LowerSha256 -Path $SmokeFullPath
        $Arguments += @(
            "--execute-full",
            "--smoke-run-receipt",
            $SmokeFullPath,
            "--smoke-run-receipt-sha256",
            $SmokeSha
        )
    }
    if (-not [string]::IsNullOrWhiteSpace($ProducerRepository)) {
        $Arguments += @(
            "--producer-repository",
            [IO.Path]::GetFullPath($ProducerRepository)
        )
    }
    $ProcessResult = Invoke-AnchorGemmaQ8PythonAndWait `
        -PythonExecutable $PythonExecutable `
        -ArgumentList $Arguments `
        -CanonicalLockStream $LockStream `
        -CanonicalLockPath $CanonicalLockPath `
        -WorkingDirectory $ProjectRoot
    $ExitCode = [int]$ProcessResult.exit_code
    if ($ExitCode -ne 0) {
        throw "runtime exited with code $ExitCode"
    }
}
finally {
    if ($null -ne $LockStream) {
        $LockStream.Dispose()
    }
}
