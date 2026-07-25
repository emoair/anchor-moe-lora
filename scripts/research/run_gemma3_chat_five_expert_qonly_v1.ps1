[CmdletBinding(DefaultParameterSetName = "Preflight")]
param(
    [Parameter(ParameterSetName = "Preflight")]
    [switch]$Preflight,

    [Parameter(Mandatory = $true, ParameterSetName = "Execute")]
    [switch]$Execute,

    [string]$RunId = "",
    [string]$ExpectedGpuUuid = "",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EntryPoint = Join-Path $ProjectRoot `
    "scripts\research\run_gemma3_chat_five_expert_qonly_v1.py"
$CanonicalConfig = Join-Path $ProjectRoot `
    "configs\training\gemma3_1b_it_chat_five_expert_qonly_rank1024_v1.yaml"
$Implementation = Join-Path $ProjectRoot `
    "src\anchor_mvp\training\gemma3_chat_five_expert_qonly_v1.py"
$TokenizerBinding = Join-Path $ProjectRoot `
    "src\anchor_mvp\training\gemma3_tokenizer_binding_v1.py"
$TokenizerConsumer = Join-Path $ProjectRoot `
    "src\anchor_mvp\training\qwen_synthetic_five_role_qonly_v2.py"
$TokenizerProducer = Join-Path $ProjectRoot `
    "src\anchor_mvp\research\synthetic_five_role_qonly_diagnostic_v1.py"
$TokenizerScaffold = Join-Path $ProjectRoot `
    "src\anchor_mvp\research\synthetic_nl_scaffold_diagnostic_v1.py"
$Q8Runtime = Join-Path $ProjectRoot `
    "src\anchor_mvp\training\gemma3_five_role_q8_qlora_v2.py"
$DiagnosticRuntime = Join-Path $ProjectRoot `
    "src\anchor_mvp\training\qwen_lora_diagnostic.py"
$SnapshotRuntime = Join-Path $ProjectRoot `
    "src\anchor_mvp\training\qwen_synthetic_scaffold_diagnostic.py"
$ConfigRuntime = Join-Path $ProjectRoot `
    "src\anchor_mvp\training\config.py"
$Q8ReliabilityRuntime = Join-Path $ProjectRoot `
    "src\anchor_mvp\training\gemma3_q8_reliability_v2.py"
$BudgetRuntime = Join-Path $ProjectRoot `
    "src\anchor_mvp\research\gemma3_qonly_parameter_budget.py"
$ManifestRuntime = Join-Path $ProjectRoot `
    "src\anchor_mvp\training\manifest.py"
$TokenizerPolicy = Join-Path $ProjectRoot `
    "configs\research\gemma3_1b_it_chat_template_policy_v1.json"
$RunRootRelative = `
    "runs/gemma3_1b_it_chat_five_expert_qonly_rank1024_v1/"
$LauncherHelper = Join-Path $ProjectRoot `
    "scripts\research\gemma3_q8_launcher_helpers_v2.ps1"
$CanonicalLockRelative = "runs/formal-v3-training.lock"
$CanonicalLock = Join-Path $ProjectRoot $CanonicalLockRelative
$LockReceiptDirectoryRelative = (
    $RunRootRelative + "gpu-locks/"
)
$ConflictingLocks = @(
    (Join-Path $ProjectRoot "runs/distill-train-handoff/gpu-job.lock"),
    (Join-Path $ProjectRoot "runs/distill-train-handoff-v3/gpu-job.lock")
)
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$Ascii = [Text.Encoding]::ASCII
$LockStream = $null
$LockSha256 = ""
$PublishedReceipt = ""
$PublishedSidecar = ""

function Resolve-ChatPython([string]$Requested) {
    $Candidates = New-Object Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        $Candidates.Add($Requested)
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:ANCHOR_TRAINING_PYTHON)) {
        $Candidates.Add([string]$env:ANCHOR_TRAINING_PYTHON)
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:ANCHOR_PYTHON)) {
        $Candidates.Add([string]$env:ANCHOR_PYTHON)
    }
    else {
        if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
            $Candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
        }
        $Candidates.Add((Join-Path $ProjectRoot ".venv\Scripts\python.exe"))
        $Candidates.Add(
            (Join-Path $env:USERPROFILE ".conda\envs\anchor-mvp\python.exe")
        )
        $Candidates.Add("python.exe")
        $Candidates.Add("python")
    }
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
        $Command = Get-Command $Candidate -CommandType Application `
            -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $Command) {
            return $Command.Source
        }
    }
    throw (
        "Python 3.10+ not found. Set -Python, ANCHOR_TRAINING_PYTHON, " +
        "or ANCHOR_PYTHON."
    )
}

function Get-ChatSha256([byte[]]$Bytes) {
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try {
        return (
            [BitConverter]::ToString($Hasher.ComputeHash($Bytes))
        ).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $Hasher.Dispose()
    }
}

function New-ChatLockNonce {
    $Bytes = New-Object byte[] 32
    $Generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $Generator.GetBytes($Bytes)
    }
    finally {
        $Generator.Dispose()
    }
    return (
        [BitConverter]::ToString($Bytes)
    ).Replace("-", "").ToLowerInvariant()
}

function Assert-ChatLockStreamDigest(
    [IO.FileStream]$Stream,
    [string]$ExpectedSha256
) {
    if ($null -eq $Stream -or
        -not $Stream.CanRead -or
        -not $Stream.CanWrite -or
        $Stream.SafeFileHandle.IsClosed) {
        throw "canonical lock stream is not readable and writable"
    }
    $OriginalPosition = $Stream.Position
    try {
        $Stream.Flush($true)
        if ($Stream.Length -gt [int]::MaxValue) {
            throw "canonical lock stream unexpectedly exceeds the byte cap"
        }
        $Length = [int]$Stream.Length
        $Bytes = New-Object byte[] $Length
        $Stream.Position = 0
        $Offset = 0
        while ($Offset -lt $Length) {
            $Read = $Stream.Read($Bytes, $Offset, $Length - $Offset)
            if ($Read -le 0) {
                throw "canonical lock stream ended before its declared length"
            }
            $Offset += $Read
        }
        $ObservedSha256 = Get-ChatSha256 -Bytes $Bytes
        if ($ObservedSha256 -cne $ExpectedSha256) {
            throw "canonical lock stream content changed"
        }
    }
    finally {
        $Stream.Position = $OriginalPosition
    }
}

function Publish-ChatLockOwnerReceipt(
    [string]$Destination,
    [byte[]]$OwnerBytes,
    [string]$OwnerSha256
) {
    $Parent = [IO.Path]::GetDirectoryName($Destination)
    [IO.Directory]::CreateDirectory($Parent) | Out-Null
    if ([IO.Directory]::Exists($Destination) -or
        [IO.File]::Exists($Destination)) {
        throw "GPU lock owner receipt already exists: $Destination"
    }
    $Staging = Join-Path $Parent (
        "." + [IO.Path]::GetFileName($Destination) +
        ".tmp-" + [Guid]::NewGuid().ToString("N")
    )
    [IO.Directory]::CreateDirectory($Staging) | Out-Null
    $Receipt = Join-Path $Staging "lock_owner.json"
    $Sidecar = Join-Path $Staging "lock_owner.json.sha256"
    $SidecarBytes = $Ascii.GetBytes(
        "$OwnerSha256  lock_owner.json`n"
    )
    try {
        $ReceiptStream = [IO.FileStream]::new(
            $Receipt,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None,
            4096,
            [IO.FileOptions]::WriteThrough
        )
        try {
            $ReceiptStream.Write($OwnerBytes, 0, $OwnerBytes.Length)
            $ReceiptStream.Flush($true)
        }
        finally {
            $ReceiptStream.Dispose()
        }
        $SidecarStream = [IO.FileStream]::new(
            $Sidecar,
            [IO.FileMode]::CreateNew,
            [IO.FileAccess]::Write,
            [IO.FileShare]::None,
            4096,
            [IO.FileOptions]::WriteThrough
        )
        try {
            $SidecarStream.Write($SidecarBytes, 0, $SidecarBytes.Length)
            $SidecarStream.Flush($true)
        }
        finally {
            $SidecarStream.Dispose()
        }
        [IO.Directory]::Move($Staging, $Destination)
    }
    catch {
        if ([IO.File]::Exists($Receipt)) {
            [IO.File]::Delete($Receipt)
        }
        if ([IO.File]::Exists($Sidecar)) {
            [IO.File]::Delete($Sidecar)
        }
        if ([IO.Directory]::Exists($Staging)) {
            [IO.Directory]::Delete($Staging, $false)
        }
        throw
    }
}

function Assert-ChatPublishedLockReceipt(
    [string]$Receipt,
    [string]$Sidecar,
    [string]$ExpectedSha256
) {
    if ([string]::IsNullOrWhiteSpace($Receipt) -or
        [string]::IsNullOrWhiteSpace($Sidecar) -or
        -not [IO.File]::Exists($Receipt) -or
        -not [IO.File]::Exists($Sidecar)) {
        throw "published GPU lock owner receipt disappeared"
    }
    $ObservedSha256 = (
        Get-FileHash -LiteralPath $Receipt -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $ObservedSidecar = $Ascii.GetString(
        [IO.File]::ReadAllBytes($Sidecar)
    )
    if ($ObservedSha256 -cne $ExpectedSha256 -or
        $ObservedSidecar -cne "$ExpectedSha256  lock_owner.json`n") {
        throw "published GPU lock owner receipt failed its digest check"
    }
}

if (-not (Test-Path -LiteralPath $EntryPoint -PathType Leaf)) {
    throw "Chat five-expert entry point is unavailable: $EntryPoint"
}
if (-not (Test-Path -LiteralPath $CanonicalConfig -PathType Leaf)) {
    throw "Canonical chat training config is unavailable: $CanonicalConfig"
}
if (-not (Test-Path -LiteralPath $LauncherHelper -PathType Leaf)) {
    throw "Q8 process-lifetime helper is unavailable: $LauncherHelper"
}
$PythonExecutable = Resolve-ChatPython -Requested $Python

if (-not $Execute) {
    Write-Host "Gemma 3 chat five-expert model-free preflight"
    Write-Host "Receipts: $RunRootRelative"
    $PreflightArguments = @(
        $EntryPoint,
        "--config", $CanonicalConfig,
        "--preflight"
    )
    if ($RunId) {
        $PreflightArguments += @("--run-id", $RunId)
    }
    & $PythonExecutable @PreflightArguments
    exit $LASTEXITCODE
}

# The first command is intentionally model/GPU-free. With any pending producer
# identity it exits 2 before a dataset path, GPU lock, tokenizer, or model load.
Write-Host "Gemma 3 chat five-expert execute prerequisite dry-run"
& $PythonExecutable $EntryPoint --config $CanonicalConfig --dry-run
$DryRunExit = $LASTEXITCODE
if ($DryRunExit -ne 0) {
    Write-Host (
        "Execute remains blocked. No GPU lock or model load was attempted " +
        "(dry-run exit $DryRunExit)."
    )
    exit $DryRunExit
}

if ([string]::IsNullOrWhiteSpace($ExpectedGpuUuid)) {
    $ExpectedGpuUuid = [string]$env:ANCHOR_GEMMA_GPU_UUID
}
$ExpectedGpuUuid = $ExpectedGpuUuid.Trim()
if ([string]::IsNullOrWhiteSpace($ExpectedGpuUuid) -or
    $ExpectedGpuUuid -ieq "UNBOUND" -or
    $ExpectedGpuUuid -notmatch "^GPU-[0-9A-Fa-f-]{32,64}$") {
    throw (
        "Execute requires -ExpectedGpuUuid or ANCHOR_GEMMA_GPU_UUID. " +
        "Empty, UNBOUND, and malformed values are refused."
    )
}
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = [Guid]::NewGuid().ToString("N")
}
if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$") {
    throw "RunId must satisfy ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
}

$PreviousEnvironment = [ordered]@{
    ANCHOR_GEMMA_GPU_UUID = $env:ANCHOR_GEMMA_GPU_UUID
    CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES
    HF_HUB_OFFLINE = $env:HF_HUB_OFFLINE
    TRANSFORMERS_OFFLINE = $env:TRANSFORMERS_OFFLINE
    HF_DATASETS_OFFLINE = $env:HF_DATASETS_OFFLINE
    TOKENIZERS_PARALLELISM = $env:TOKENIZERS_PARALLELISM
    ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_RECEIPT = `
        $env:ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_RECEIPT
    ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_SHA256 = `
        $env:ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_SHA256
    ANCHOR_CHAT_EXTERNAL_LOCK_NONCE = `
        $env:ANCHOR_CHAT_EXTERNAL_LOCK_NONCE
    ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID = `
        $env:ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID
}
try {
    foreach ($Conflict in $ConflictingLocks) {
        if (Test-Path -LiteralPath $Conflict) {
            throw "Conflicting GPU handoff lock exists: $Conflict"
        }
    }
    if (Test-Path -LiteralPath $CanonicalLock) {
        throw "Canonical GPU training lock already exists: $CanonicalLock"
    }
    [IO.Directory]::CreateDirectory(
        [IO.Path]::GetDirectoryName($CanonicalLock)
    ) | Out-Null
    $LockNonce = New-ChatLockNonce
    $LockReceiptRelative = (
        $LockReceiptDirectoryRelative + $RunId + "/lock_owner.json"
    )
    $LockReceiptDirectory = Join-Path $ProjectRoot (
        $LockReceiptDirectoryRelative + $RunId
    )
    $ExecutionDependencySha256 = [ordered]@{
        "src/anchor_mvp/training/gemma3_tokenizer_binding_v1.py" = (
            Get-FileHash -LiteralPath $TokenizerBinding -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/training/qwen_synthetic_five_role_qonly_v2.py" = (
            Get-FileHash -LiteralPath $TokenizerConsumer -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/research/synthetic_five_role_qonly_diagnostic_v1.py" = (
            Get-FileHash -LiteralPath $TokenizerProducer -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/research/synthetic_nl_scaffold_diagnostic_v1.py" = (
            Get-FileHash -LiteralPath $TokenizerScaffold -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/training/gemma3_five_role_q8_qlora_v2.py" = (
            Get-FileHash -LiteralPath $Q8Runtime -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/training/qwen_lora_diagnostic.py" = (
            Get-FileHash -LiteralPath $DiagnosticRuntime -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/training/qwen_synthetic_scaffold_diagnostic.py" = (
            Get-FileHash -LiteralPath $SnapshotRuntime -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/training/config.py" = (
            Get-FileHash -LiteralPath $ConfigRuntime -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/training/gemma3_q8_reliability_v2.py" = (
            Get-FileHash -LiteralPath $Q8ReliabilityRuntime -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/research/gemma3_qonly_parameter_budget.py" = (
            Get-FileHash -LiteralPath $BudgetRuntime -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "src/anchor_mvp/training/manifest.py" = (
            Get-FileHash -LiteralPath $ManifestRuntime -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "configs/research/gemma3_1b_it_chat_template_policy_v1.json" = (
            Get-FileHash -LiteralPath $TokenizerPolicy -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        "scripts/research/gemma3_q8_launcher_helpers_v2.ps1" = (
            Get-FileHash -LiteralPath $LauncherHelper -Algorithm SHA256
        ).Hash.ToLowerInvariant()
    }
    $LockOwner = [ordered]@{
        schema_version = `
            "anchor.gemma3-1b-it-chat-five-expert-qonly-lock-owner.v1"
        run_id = $RunId
        launcher_pid = $PID
        canonical_lock = $CanonicalLockRelative
        owner_receipt = $LockReceiptRelative
        expected_gpu_index = 0
        expected_gpu_uuid = $ExpectedGpuUuid
        roles = @(
            "humor",
            "serious",
            "angry_style",
            "tool_call",
            "review_audit"
        )
        concurrency = 1
        smoke_steps_per_role = 2
        full_steps_per_role = 160
        fresh_base_per_phase = $true
        fresh_adapter_per_phase = $true
        resume = $false
        config_sha256 = (
            Get-FileHash -LiteralPath $CanonicalConfig -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        implementation_sha256 = (
            Get-FileHash -LiteralPath $Implementation -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        runner_script_sha256 = (
            Get-FileHash -LiteralPath $EntryPoint -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        launcher_sha256 = (
            Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        launcher_helper_sha256 = (
            Get-FileHash -LiteralPath $LauncherHelper -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        execution_dependency_sha256 = $ExecutionDependencySha256
        nonce = $LockNonce
        owner = "powershell_create_new_file_share_none"
        file_mode = "CreateNew"
        file_access = "ReadWrite"
        file_share = "None"
        delete_on_close = $true
        write_through = $true
        held_for_entire_python_lifetime = $true
    }
    $LockBytes = $Utf8NoBom.GetBytes(
        (($LockOwner | ConvertTo-Json -Depth 8 -Compress) + "`n")
    )
    $LockSha256 = Get-ChatSha256 -Bytes $LockBytes
    $LockOptions = [IO.FileOptions]::DeleteOnClose -bor `
        [IO.FileOptions]::WriteThrough
    $LockStream = [IO.FileStream]::new(
        $CanonicalLock,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::ReadWrite,
        [IO.FileShare]::None,
        4096,
        $LockOptions
    )
    $LockStream.Write($LockBytes, 0, $LockBytes.Length)
    $LockStream.Flush($true)
    Assert-ChatLockStreamDigest `
        -Stream $LockStream `
        -ExpectedSha256 $LockSha256
    Publish-ChatLockOwnerReceipt `
        -Destination $LockReceiptDirectory `
        -OwnerBytes $LockBytes `
        -OwnerSha256 $LockSha256
    $PublishedReceipt = Join-Path $LockReceiptDirectory "lock_owner.json"
    $PublishedSidecar = Join-Path (
        $LockReceiptDirectory
    ) "lock_owner.json.sha256"
    Assert-ChatPublishedLockReceipt `
        -Receipt $PublishedReceipt `
        -Sidecar $PublishedSidecar `
        -ExpectedSha256 $LockSha256
    . $LauncherHelper
    foreach ($Conflict in $ConflictingLocks) {
        if (Test-Path -LiteralPath $Conflict) {
            throw "Conflicting GPU handoff lock appeared: $Conflict"
        }
    }

    $env:ANCHOR_GEMMA_GPU_UUID = $ExpectedGpuUuid
    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:HF_DATASETS_OFFLINE = "1"
    $env:TOKENIZERS_PARALLELISM = "false"
    $env:ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_RECEIPT = $PublishedReceipt
    $env:ANCHOR_CHAT_EXTERNAL_LOCK_OWNER_SHA256 = $LockSha256
    $env:ANCHOR_CHAT_EXTERNAL_LOCK_NONCE = $LockNonce
    $env:ANCHOR_CHAT_EXTERNAL_LOCK_LAUNCHER_PID = [string]$PID
    Write-Host (
        "Starting explicit diagnostic execute: run=$RunId, " +
        "GPU=$ExpectedGpuUuid, roles=5, smoke=2, full=160, concurrency=1"
    )
    $ExecuteArguments = @(
        $EntryPoint,
        "--config", $CanonicalConfig,
        "--execute",
        "--run-id", $RunId
    )
    $PythonResult = Invoke-AnchorGemmaQ8PythonAndWait `
        -PythonExecutable $PythonExecutable `
        -ArgumentList $ExecuteArguments `
        -CanonicalLockStream $LockStream `
        -CanonicalLockPath $CanonicalLock `
        -WorkingDirectory $ProjectRoot
    Assert-ChatLockStreamDigest `
        -Stream $LockStream `
        -ExpectedSha256 $LockSha256
    Assert-ChatPublishedLockReceipt `
        -Receipt $PublishedReceipt `
        -Sidecar $PublishedSidecar `
        -ExpectedSha256 $LockSha256
    exit ([int]$PythonResult.exit_code)
}
finally {
    try {
        if ($null -ne $LockStream -and
            -not [string]::IsNullOrWhiteSpace($LockSha256)) {
            Assert-ChatLockStreamDigest `
                -Stream $LockStream `
                -ExpectedSha256 $LockSha256
        }
        if (-not [string]::IsNullOrWhiteSpace($PublishedReceipt)) {
            Assert-ChatPublishedLockReceipt `
                -Receipt $PublishedReceipt `
                -Sidecar $PublishedSidecar `
                -ExpectedSha256 $LockSha256
        }
    }
    finally {
        foreach ($Name in $PreviousEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable(
                $Name,
                $PreviousEnvironment[$Name],
                [EnvironmentVariableTarget]::Process
            )
        }
        if ($null -ne $LockStream) {
            $LockStream.Dispose()
            $LockStream = $null
        }
    }
}
