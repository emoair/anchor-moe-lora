param(
    [switch]$Execute,
    [string]$ExpectedGpuUuid = "",
    [string]$Python = "",
    [string]$Config = "configs/training/gemma3_1b_it_five_role_qonly_v1.yaml",
    [string]$NvidiaSmiPath = "nvidia-smi.exe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$RunnerRelativePath = "scripts/research/run_gemma3_1b_it_five_role_qonly_v1.py"
$ImplementationRelativePath = "src/anchor_mvp/training/gemma3_five_role_qonly_v1.py"
$ConfigRelativePath = "configs/training/gemma3_1b_it_five_role_qonly_v1.yaml"
$CanonicalLockRelativePath = "runs/formal-v3-training.lock"
$HandoffLockRelativePaths = @(
    "runs/distill-train-handoff/gpu-job.lock",
    "runs/distill-train-handoff-v3/gpu-job.lock"
)
$AttestationRootRelativePath = "runs/gemma3_1b_it_five_role_qonly_v1/gpu-attestations"
$Roles = @(
    "planner",
    "tool_policy",
    "frontend_gen",
    "frontend_review",
    "security_gate"
)

# This policy is intentionally fixed in the launcher. Changing it requires a
# versioned launcher/config update rather than an ad-hoc command-line override.
$GpuPolicy = [ordered]@{
    expected_index = 0
    expected_total_memory_mib = 12288
    sample_count = 3
    sample_interval_seconds = 1
    command_timeout_seconds = 5
    idle_used_memory_max_mib = 2048
    idle_free_memory_min_mib = 8192
    idle_utilization_max_percent = 15
    prestart_temperature_max_c = 75
    wddm_gui_process_allowlist = @(
        "applicationframehost.exe",
        "chatgpt.exe",
        "codex.exe",
        "dwm.exe",
        "explorer.exe",
        "flclash.exe",
        "gamebar.exe",
        "gamebarftserver.exe",
        "gameviewerserver.exe",
        "lockapp.exe",
        "msedgewebview2.exe",
        "nvidia broadcast.exe",
        "nvidia overlay.exe",
        "promecefpluginhost.exe",
        "searchhost.exe",
        "shellexperiencehost.exe",
        "shellhost.exe",
        "startmenuexperiencehost.exe",
        "systemsettings.exe",
        "systemsettingsbroker.exe",
        "tabtip.exe",
        "taskmgr.exe",
        "textinputhost.exe",
        "wechat.exe",
        "wechatappex.exe",
        "widgets.exe",
        "widgetservice.exe",
        "wps.exe",
        "wpscenter.exe",
        "wpscloudsvr.exe"
    )
    wddm_gui_inventory_must_be_stable_across_gate = $true
    insufficient_permissions_pid_resolution_required = $true
    unknown_or_non_allowlisted_compute_process_forbidden = $true
}

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$LockStream = $null
$TemporaryAttestationDirectory = $null

function Resolve-ProjectPath([string]$RelativePath) {
    return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $RelativePath))
}

function Get-RelativeProjectPath([string]$AbsolutePath) {
    $RootWithSeparator = $ProjectRoot.TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    ) + [IO.Path]::DirectorySeparatorChar
    if (-not $AbsolutePath.StartsWith(
        $RootWithSeparator,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "path escapes the project root: $AbsolutePath"
    }
    return $AbsolutePath.Substring($RootWithSeparator.Length).Replace("\", "/")
}

function Resolve-Executable([string]$Candidate, [string]$Label) {
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }
    $Command = Get-Command $Candidate -CommandType Application, ExternalScript -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $Command) {
        throw "$Label executable was not found: $Candidate"
    }
    return $Command.Source
}

function Test-PythonRuntime([string]$Executable) {
    $ProbeCode = (
        "import importlib.util,json,sys;" +
        "mods=('yaml','sentencepiece','torch','transformers','peft');" +
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];" +
        "print(json.dumps({'schema_version':'anchor.python-runtime-probe.v1'," +
        "'version':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro]," +
        "'missing':missing},sort_keys=True,separators=(',',':')));" +
        "raise SystemExit(0 if sys.version_info>=(3,10) and not missing else 17)"
    )
    $Raw = @(& $Executable -c $ProbeCode 2>&1)
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        return $null
    }
    $Text = (($Raw | ForEach-Object { [string]$_ }) -join "`n").Trim()
    try {
        $Probe = $Text | ConvertFrom-Json
    }
    catch {
        return $null
    }
    if ($null -eq $Probe) {
        return $null
    }
    $ProbeProperties = @($Probe.PSObject.Properties.Name)
    if ("schema_version" -notin $ProbeProperties -or
        "version" -notin $ProbeProperties -or
        "missing" -notin $ProbeProperties -or
        $Probe.schema_version -ne "anchor.python-runtime-probe.v1" -or
        @($Probe.version).Count -ne 3 -or
        @($Probe.missing).Count -ne 0 -or
        [int]$Probe.version[0] -lt 3 -or
        ([int]$Probe.version[0] -eq 3 -and [int]$Probe.version[1] -lt 10)) {
        return $null
    }
    return [pscustomobject]@{
        path = $Executable
        version = (@($Probe.version) -join ".")
        sha256 = Get-Sha256 $Executable
        dependency_probe = "yaml,sentencepiece,torch,transformers,peft"
    }
}

function Resolve-PythonRuntime([string]$Requested) {
    $Candidates = New-Object Collections.Generic.List[object]
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        $Candidates.Add([pscustomobject]@{ label = "-Python"; value = $Requested })
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:ANCHOR_PYTHON)) {
        $Candidates.Add([pscustomobject]@{
            label = "ANCHOR_PYTHON"
            value = [string]$env:ANCHOR_PYTHON
        })
    }
    else {
        if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
            $Candidates.Add([pscustomobject]@{
                label = "CONDA_PREFIX"
                value = (Join-Path $env:CONDA_PREFIX "python.exe")
            })
        }
        foreach ($Relative in @(
            ".venv/Scripts/python.exe",
            ".venv/bin/python",
            "venv/Scripts/python.exe",
            "venv/bin/python"
        )) {
            $Candidates.Add([pscustomobject]@{
                label = "repository virtual environment"
                value = (Resolve-ProjectPath $Relative)
            })
        }
        $UserHome = [Environment]::GetFolderPath(
            [Environment+SpecialFolder]::UserProfile
        )
        if (-not [string]::IsNullOrWhiteSpace($UserHome)) {
            $Candidates.Add([pscustomobject]@{
                label = "user Conda anchor-mvp environment"
                value = (Join-Path $UserHome ".conda/envs/anchor-mvp/python.exe")
            })
        }
        foreach ($Name in @("python.exe", "python")) {
            $Candidates.Add([pscustomobject]@{ label = "PATH"; value = $Name })
        }
    }

    $Failures = New-Object Collections.Generic.List[string]
    $Seen = @{}
    foreach ($Candidate in $Candidates) {
        try {
            $Executable = Resolve-Executable `
                -Candidate ([string]$Candidate.value) `
                -Label "Python"
        }
        catch {
            $Failures.Add("$($Candidate.label): not found")
            continue
        }
        $Key = $Executable.ToLowerInvariant()
        if ($Seen.ContainsKey($Key)) {
            continue
        }
        $Seen[$Key] = $true
        $Runtime = Test-PythonRuntime -Executable $Executable
        if ($null -ne $Runtime) {
            return $Runtime
        }
        $Failures.Add("$($Candidate.label): version/dependency probe failed")
        if (-not [string]::IsNullOrWhiteSpace($Requested) -or
            -not [string]::IsNullOrWhiteSpace($env:ANCHOR_PYTHON)) {
            break
        }
    }
    throw (
        "no reproducible Python >=3.10 runtime with yaml, sentencepiece, torch, " +
        "transformers, and peft was found; set -Python or ANCHOR_PYTHON. " +
        "Attempts: $($Failures -join '; ')"
    )
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-BytesSha256([byte[]]$Bytes) {
    $Sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($Sha.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $Sha.Dispose()
    }
}

function ConvertTo-CanonicalJsonBytes([object]$Value) {
    $Json = ($Value | ConvertTo-Json -Depth 20 -Compress) + "`n"
    return $Utf8NoBom.GetBytes($Json)
}

function Write-NewFileDurable(
    [string]$Path,
    [byte[]]$Bytes
) {
    $Stream = [IO.FileStream]::new(
        $Path,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None,
        4096,
        [IO.FileOptions]::WriteThrough
    )
    try {
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    }
    finally {
        $Stream.Dispose()
    }
}

function Assert-NoCompetingLocks([string]$Phase) {
    $CanonicalLockPath = Resolve-ProjectPath $CanonicalLockRelativePath
    if (Test-Path -LiteralPath $CanonicalLockPath) {
        throw "$Phase refused: canonical GPU training lock already exists at $CanonicalLockRelativePath"
    }
    foreach ($RelativePath in $HandoffLockRelativePaths) {
        if (Test-Path -LiteralPath (Resolve-ProjectPath $RelativePath)) {
            throw "$Phase refused: handoff GPU lock already exists at $RelativePath"
        }
    }
}

function Assert-NoHandoffLocks([string]$Phase) {
    foreach ($RelativePath in $HandoffLockRelativePaths) {
        if (Test-Path -LiteralPath (Resolve-ProjectPath $RelativePath)) {
            throw "$Phase refused: handoff GPU lock appeared at $RelativePath"
        }
    }
}

function Invoke-ProcessWithTimeout(
    [string]$Executable,
    [string]$Arguments,
    [int]$TimeoutSeconds
) {
    $Info = New-Object Diagnostics.ProcessStartInfo
    $Extension = [IO.Path]::GetExtension($Executable)
    if ($Extension -ieq ".cmd" -or $Extension -ieq ".bat") {
        if ([string]::IsNullOrWhiteSpace($env:ComSpec)) {
            throw "ComSpec is unavailable for the command shim $Executable"
        }
        $QuotedArgumentTokens = @(
            $Arguments -split " " |
                Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
                ForEach-Object { "`"$($_.Replace('"', '""'))`"" }
        )
        $Info.FileName = $env:ComSpec
        $Info.Arguments = (
            "/d /s /c `"`"$Executable`" " +
            ($QuotedArgumentTokens -join " ") +
            "`""
        )
    }
    else {
        $Info.FileName = $Executable
        $Info.Arguments = $Arguments
    }
    $Info.UseShellExecute = $false
    $Info.CreateNoWindow = $true
    $Info.RedirectStandardOutput = $true
    $Info.RedirectStandardError = $true
    $Info.StandardOutputEncoding = $Utf8NoBom
    $Info.StandardErrorEncoding = $Utf8NoBom

    $Process = New-Object Diagnostics.Process
    $Process.StartInfo = $Info
    try {
        if (-not $Process.Start()) {
            throw "failed to start $Executable"
        }
        $StdoutTask = $Process.StandardOutput.ReadToEndAsync()
        $StderrTask = $Process.StandardError.ReadToEndAsync()
        if (-not $Process.WaitForExit($TimeoutSeconds * 1000)) {
            try {
                $Process.Kill()
            }
            catch {
                # The process may have exited between WaitForExit and Kill.
            }
            $Process.WaitForExit()
            throw "command timed out after $TimeoutSeconds seconds: $Executable"
        }
        $Stdout = $StdoutTask.Result
        $Stderr = $StderrTask.Result
        if ($Process.ExitCode -ne 0) {
            $ErrorSummary = $Stderr.Trim()
            if ([string]::IsNullOrWhiteSpace($ErrorSummary)) {
                $ErrorSummary = "<empty stderr>"
            }
            throw "command exited $($Process.ExitCode): $Executable; stderr=$ErrorSummary"
        }
        return [pscustomobject]@{
            stdout = [string]$Stdout
            stderr = [string]$Stderr
        }
    }
    finally {
        $Process.Dispose()
    }
}

function ConvertFrom-StrictCsv(
    [string]$Text,
    [string[]]$Headers,
    [string]$Label,
    [switch]$AllowEmpty
) {
    $Lines = @(
        $Text -split "\r?\n" |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($Lines.Count -eq 1 -and $Lines[0] -match "^No running processes found\.?$") {
        $Lines = @()
    }
    if ($Lines.Count -eq 0) {
        if ($AllowEmpty) {
            return @()
        }
        throw "$Label returned no CSV rows"
    }
    try {
        $Rows = @($Lines | ConvertFrom-Csv -Header $Headers)
    }
    catch {
        throw "$Label returned malformed CSV: $($_.Exception.Message)"
    }
    foreach ($Row in $Rows) {
        $ObservedHeaders = @($Row.PSObject.Properties.Name)
        if ($ObservedHeaders.Count -ne $Headers.Count) {
            throw "$Label CSV row has an unexpected column count"
        }
        foreach ($Header in $Headers) {
            if ($Header -notin $ObservedHeaders -or $null -eq $Row.$Header) {
                throw "$Label CSV row is missing $Header"
            }
        }
    }
    return $Rows
}

function ConvertTo-StrictInteger(
    [object]$Value,
    [string]$Label,
    [int]$Minimum,
    [int]$Maximum
) {
    $Parsed = 0
    $Text = ([string]$Value).Trim()
    if (-not [int]::TryParse(
        $Text,
        [Globalization.NumberStyles]::Integer,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$Parsed
    )) {
        throw "$Label is not an integer: $Text"
    }
    if ($Parsed -lt $Minimum -or $Parsed -gt $Maximum) {
        throw "$Label is outside [$Minimum,$Maximum]: $Parsed"
    }
    return $Parsed
}

function Resolve-ComputeProcessBasename(
    [int]$PidValue,
    [string]$ReportedName
) {
    $Name = $ReportedName.Trim()
    $WasPermissionDenied = (
        $Name -match "^\[?Insufficient Permissions\]?$"
    )
    if ($WasPermissionDenied) {
        try {
            $Process = Get-Process -Id $PidValue -ErrorAction Stop
            $Name = [string]$Process.ProcessName
        }
        catch {
            throw (
                "compute-process PID $PidValue reported Insufficient Permissions " +
                "and could not be resolved through the local process table"
            )
        }
    }
    if ([string]::IsNullOrWhiteSpace($Name) -or
        $Name -match "^(N/A|\[N/A\]|Unknown)$") {
        throw "compute-process PID $PidValue has an unresolved process name"
    }
    $Basename = [IO.Path]::GetFileName($Name.Trim('"')).Trim()
    if ([string]::IsNullOrWhiteSpace([IO.Path]::GetExtension($Basename))) {
        $Basename = "$Basename.exe"
    }
    $Normalized = $Basename.ToLowerInvariant()
    if ($Normalized -notmatch "^[a-z0-9][a-z0-9 ._+-]*\.exe$") {
        throw "compute-process PID $PidValue has a non-canonical executable basename"
    }
    return [pscustomobject]@{
        process_name = $Normalized
        reported_name_was_permission_denied = $WasPermissionDenied
    }
}

function Get-ComputeInventorySha256([object[]]$Processes) {
    $Identity = @(
        $Processes |
            Sort-Object pid, process_name |
            ForEach-Object {
                [ordered]@{
                    pid = [int]$_.pid
                    process_name = [string]$_.process_name
                }
            }
    )
    return Get-BytesSha256 (ConvertTo-CanonicalJsonBytes $Identity)
}

function Get-GpuSample(
    [string]$NvidiaSmi,
    [string]$ExpectedUuid,
    [string]$Phase,
    [int]$Ordinal
) {
    $GpuArguments = (
        "--query-gpu=index,uuid,name,driver_model.current,memory.total," +
        "memory.used,memory.free,utilization.gpu,temperature.gpu " +
        "--format=csv,noheader,nounits"
    )
    $ComputeArguments = (
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory " +
        "--format=csv,noheader,nounits"
    )
    $GpuResult = Invoke-ProcessWithTimeout `
        -Executable $NvidiaSmi `
        -Arguments $GpuArguments `
        -TimeoutSeconds $GpuPolicy.command_timeout_seconds
    $GpuRows = @(ConvertFrom-StrictCsv `
        -Text $GpuResult.stdout `
        -Headers @(
            "index",
            "uuid",
            "name",
            "driver_model",
            "memory_total_mib",
            "memory_used_mib",
            "memory_free_mib",
            "utilization_percent",
            "temperature_c"
        ) `
        -Label "nvidia-smi GPU query")

    $SeenIndices = @{}
    $SeenUuids = @{}
    $SelectedRows = @()
    foreach ($Row in $GpuRows) {
        $Index = ConvertTo-StrictInteger -Value $Row.index -Label "GPU index" -Minimum 0 -Maximum 1024
        $Uuid = ([string]$Row.uuid).Trim()
        if ($Uuid -notmatch "^GPU-[0-9A-Fa-f-]{32,64}$") {
            throw "nvidia-smi returned an invalid GPU UUID: $Uuid"
        }
        $NormalizedUuid = $Uuid.ToLowerInvariant()
        if ($SeenIndices.ContainsKey($Index)) {
            throw "nvidia-smi returned duplicate GPU index $Index"
        }
        if ($SeenUuids.ContainsKey($NormalizedUuid)) {
            throw "nvidia-smi returned duplicate GPU UUID $Uuid"
        }
        $SeenIndices[$Index] = $true
        $SeenUuids[$NormalizedUuid] = $true
        if ($Index -eq $GpuPolicy.expected_index) {
            $SelectedRows += $Row
        }
    }
    if ($SelectedRows.Count -ne 1) {
        throw "expected exactly one GPU at index $($GpuPolicy.expected_index), observed $($SelectedRows.Count)"
    }

    $Gpu = $SelectedRows[0]
    $ObservedUuid = ([string]$Gpu.uuid).Trim()
    if (-not $ObservedUuid.Equals($ExpectedUuid, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Phase sample $Ordinal GPU UUID mismatch: expected $ExpectedUuid, observed $ObservedUuid"
    }
    $Name = ([string]$Gpu.name).Trim()
    $DriverModel = ([string]$Gpu.driver_model).Trim()
    if ([string]::IsNullOrWhiteSpace($Name) -or [string]::IsNullOrWhiteSpace($DriverModel)) {
        throw "$Phase sample $Ordinal has an empty GPU name or driver model"
    }
    $TotalMiB = ConvertTo-StrictInteger `
        -Value $Gpu.memory_total_mib -Label "GPU total memory MiB" -Minimum 1 -Maximum 1048576
    $UsedMiB = ConvertTo-StrictInteger `
        -Value $Gpu.memory_used_mib -Label "GPU used memory MiB" -Minimum 0 -Maximum 1048576
    $FreeMiB = ConvertTo-StrictInteger `
        -Value $Gpu.memory_free_mib -Label "GPU free memory MiB" -Minimum 0 -Maximum 1048576
    $Utilization = ConvertTo-StrictInteger `
        -Value $Gpu.utilization_percent -Label "GPU utilization percent" -Minimum 0 -Maximum 100
    $Temperature = ConvertTo-StrictInteger `
        -Value $Gpu.temperature_c -Label "GPU temperature C" -Minimum -50 -Maximum 200

    if ($TotalMiB -ne $GpuPolicy.expected_total_memory_mib) {
        throw "$Phase sample $Ordinal total memory mismatch: expected $($GpuPolicy.expected_total_memory_mib), observed $TotalMiB"
    }
    if ($UsedMiB -gt $GpuPolicy.idle_used_memory_max_mib) {
        throw "$Phase sample $Ordinal used memory exceeds idle gate: $UsedMiB MiB"
    }
    if ($FreeMiB -lt $GpuPolicy.idle_free_memory_min_mib) {
        throw "$Phase sample $Ordinal free memory is below idle gate: $FreeMiB MiB"
    }
    if ($Utilization -gt $GpuPolicy.idle_utilization_max_percent) {
        throw "$Phase sample $Ordinal utilization exceeds idle gate: $Utilization percent"
    }
    if ($Temperature -gt $GpuPolicy.prestart_temperature_max_c) {
        throw "$Phase sample $Ordinal temperature exceeds prestart gate: $Temperature C"
    }

    $ComputeResult = Invoke-ProcessWithTimeout `
        -Executable $NvidiaSmi `
        -Arguments $ComputeArguments `
        -TimeoutSeconds $GpuPolicy.command_timeout_seconds
    $ComputeRows = @(ConvertFrom-StrictCsv `
        -Text $ComputeResult.stdout `
        -Headers @("gpu_uuid", "pid", "process_name", "used_gpu_memory_mib") `
        -Label "nvidia-smi compute-process query" `
        -AllowEmpty)
    $AllowedProcesses = @()
    $ForeignProcesses = @()
    $SeenPids = @{}
    foreach ($Row in $ComputeRows) {
        $GpuUuid = ([string]$Row.gpu_uuid).Trim()
        if ($GpuUuid -notmatch "^GPU-[0-9A-Fa-f-]{32,64}$") {
            throw "compute-process query returned an invalid GPU UUID: $GpuUuid"
        }
        $PidValue = ConvertTo-StrictInteger `
            -Value $Row.pid -Label "compute-process PID" -Minimum 1 -Maximum ([int]::MaxValue)
        $ReportedProcessName = ([string]$Row.process_name).Trim()
        if ([string]::IsNullOrWhiteSpace($ReportedProcessName)) {
            throw "compute-process query returned an empty process name for PID $PidValue"
        }
        $MemoryText = ([string]$Row.used_gpu_memory_mib).Trim()
        if ($MemoryText -notmatch "^(N/A|\[N/A\]|Not Supported)$") {
            [void](ConvertTo-StrictInteger `
                -Value $MemoryText `
                -Label "compute-process used memory MiB" `
                -Minimum 0 `
                -Maximum 1048576)
        }
        if ($GpuUuid.Equals($ExpectedUuid, [StringComparison]::OrdinalIgnoreCase)) {
            if ($SeenPids.ContainsKey($PidValue)) {
                throw "compute-process query returned duplicate PID $PidValue on the selected GPU"
            }
            $SeenPids[$PidValue] = $true
            $ResolvedProcess = Resolve-ComputeProcessBasename `
                -PidValue $PidValue `
                -ReportedName $ReportedProcessName
            # Emit a real PSObject so Sort-Object resolves pid/process_name as
            # properties. An OrderedDictionary preserves insertion order but
            # Sort-Object can otherwise retain the nvidia-smi row order.
            $ProcessRecord = [pscustomobject][ordered]@{
                pid = $PidValue
                process_name = $ResolvedProcess.process_name
                used_gpu_memory_mib = $MemoryText
                reported_name_was_permission_denied = (
                    $ResolvedProcess.reported_name_was_permission_denied
                )
                allowlisted_wddm_gui = (
                    $DriverModel -ieq "WDDM" -and
                    $GpuPolicy.wddm_gui_process_allowlist -contains (
                        $ResolvedProcess.process_name
                    )
                )
            }
            if ($ProcessRecord.allowlisted_wddm_gui) {
                $AllowedProcesses += $ProcessRecord
            }
            else {
                $ForeignProcesses += $ProcessRecord
            }
        }
    }
    if ($ForeignProcesses.Count -ne 0) {
        $ProcessSummary = (
            ($ForeignProcesses | ForEach-Object {
                "$($_.process_name):$($_.pid)"
            }) -join ","
        )
        throw (
            "$Phase sample $Ordinal found a foreign, Python/llama, unknown, " +
            "or non-allowlisted compute process on $ExpectedUuid ($ProcessSummary)"
        )
    }
    $AllowedProcesses = @($AllowedProcesses | Sort-Object pid, process_name)
    $ComputeInventorySha256 = Get-ComputeInventorySha256 $AllowedProcesses

    # WDDM desktop allocation is accepted only because every selected-GPU
    # process resolved to an exact frozen GUI basename and all scalar gates pass.
    return [ordered]@{
        phase = $Phase
        ordinal = $Ordinal
        observed_at_utc = [DateTime]::UtcNow.ToString("o")
        index = $GpuPolicy.expected_index
        uuid = $ObservedUuid
        name = $Name
        driver_model = $DriverModel
        memory_total_mib = $TotalMiB
        memory_used_mib = $UsedMiB
        memory_free_mib = $FreeMiB
        utilization_percent = $Utilization
        temperature_c = $Temperature
        selected_gpu_compute_process_count = $AllowedProcesses.Count
        compute_inventory_sha256 = $ComputeInventorySha256
        compute_processes = $AllowedProcesses
        wddm_desktop_baseline_tolerated = ($DriverModel -ieq "WDDM" -and $UsedMiB -gt 0)
    }
}

function Get-GpuSampleSeries(
    [string]$NvidiaSmi,
    [string]$ExpectedUuid,
    [string]$Phase
) {
    $Samples = @()
    for ($Index = 1; $Index -le $GpuPolicy.sample_count; $Index++) {
        $Samples += Get-GpuSample `
            -NvidiaSmi $NvidiaSmi `
            -ExpectedUuid $ExpectedUuid `
            -Phase $Phase `
            -Ordinal $Index
        if ($Index -lt $GpuPolicy.sample_count) {
            Start-Sleep -Seconds $GpuPolicy.sample_interval_seconds
        }
    }
    return $Samples
}

function Assert-StableGpuIdentity(
    [object[]]$PreLockSamples,
    [object[]]$PostLockSamples
) {
    $AllSamples = @($PreLockSamples) + @($PostLockSamples)
    $Reference = $AllSamples[0]
    foreach ($Sample in $AllSamples) {
        if (-not ([string]$Sample.uuid).Equals(
            [string]$Reference.uuid,
            [StringComparison]::OrdinalIgnoreCase
        ) -or
            $Sample.index -ne $Reference.index -or
            $Sample.memory_total_mib -ne $Reference.memory_total_mib -or
            $Sample.driver_model -ne $Reference.driver_model -or
            $Sample.compute_inventory_sha256 -ne $Reference.compute_inventory_sha256) {
            throw (
                "GPU identity or frozen WDDM GUI process inventory drifted " +
                "across the pre-lock/post-lock samples"
            )
        }
    }
}

function Publish-ExecutionGateArtifacts(
    [object]$LeaseReceipt,
    [object]$Attestation,
    [string]$RunId
) {
    $Parent = Resolve-ProjectPath $AttestationRootRelativePath
    [IO.Directory]::CreateDirectory($Parent) | Out-Null
    $FinalDirectory = Join-Path $Parent $RunId
    if (Test-Path -LiteralPath $FinalDirectory) {
        throw "GPU attestation output already exists: $(Get-RelativeProjectPath $FinalDirectory)"
    }
    $script:TemporaryAttestationDirectory = Join-Path $Parent (
        ".tmp-$RunId-$([Guid]::NewGuid().ToString('N'))"
    )
    [IO.Directory]::CreateDirectory($script:TemporaryAttestationDirectory) | Out-Null
    $LeasePath = Join-Path $script:TemporaryAttestationDirectory "lease_receipt.json"
    $LeaseSidecarPath = Join-Path $script:TemporaryAttestationDirectory "lease_receipt.json.sha256"
    $JsonPath = Join-Path $script:TemporaryAttestationDirectory "gpu_attestation.json"
    $SidecarPath = Join-Path $script:TemporaryAttestationDirectory "gpu_attestation.json.sha256"
    $LeaseBytes = ConvertTo-CanonicalJsonBytes $LeaseReceipt
    $LeaseSha256 = Get-BytesSha256 $LeaseBytes
    $LeaseSidecarBytes = $Utf8NoBom.GetBytes("$LeaseSha256  lease_receipt.json`n")
    $JsonBytes = ConvertTo-CanonicalJsonBytes $Attestation
    $DeclaredSha256 = Get-BytesSha256 $JsonBytes
    $SidecarBytes = $Utf8NoBom.GetBytes("$DeclaredSha256  gpu_attestation.json`n")
    Write-NewFileDurable -Path $LeasePath -Bytes $LeaseBytes
    Write-NewFileDurable -Path $LeaseSidecarPath -Bytes $LeaseSidecarBytes
    Write-NewFileDurable -Path $JsonPath -Bytes $JsonBytes
    Write-NewFileDurable -Path $SidecarPath -Bytes $SidecarBytes
    if ((Get-Sha256 $LeasePath) -ne $LeaseSha256) {
        throw "execution lease receipt staging hash mismatch"
    }
    if ((Get-Sha256 $JsonPath) -ne $DeclaredSha256) {
        throw "GPU attestation staging hash mismatch"
    }
    [IO.Directory]::Move($script:TemporaryAttestationDirectory, $FinalDirectory)
    $script:TemporaryAttestationDirectory = $null
    $FinalLeasePath = Join-Path $FinalDirectory "lease_receipt.json"
    $FinalLeaseSidecarPath = Join-Path $FinalDirectory "lease_receipt.json.sha256"
    $FinalJsonPath = Join-Path $FinalDirectory "gpu_attestation.json"
    $FinalSidecarPath = Join-Path $FinalDirectory "gpu_attestation.json.sha256"
    if ((Get-Sha256 $FinalLeasePath) -ne $LeaseSha256) {
        throw "execution lease receipt final hash mismatch"
    }
    if ((Get-Sha256 $FinalJsonPath) -ne $DeclaredSha256) {
        throw "GPU attestation final hash mismatch"
    }
    $ExpectedLeaseSidecar = "$LeaseSha256  lease_receipt.json`n"
    if ([IO.File]::ReadAllText($FinalLeaseSidecarPath, $Utf8NoBom) -cne $ExpectedLeaseSidecar) {
        throw "execution lease receipt mandatory sidecar mismatch"
    }
    $ExpectedSidecar = "$DeclaredSha256  gpu_attestation.json`n"
    if ([IO.File]::ReadAllText($FinalSidecarPath, $Utf8NoBom) -cne $ExpectedSidecar) {
        throw "GPU attestation mandatory sidecar mismatch"
    }
    return [pscustomobject]@{
        lease_path = $FinalLeasePath
        lease_sha256 = $LeaseSha256
        lease_sidecar_path = $FinalLeaseSidecarPath
        attestation_path = $FinalJsonPath
        attestation_sha256 = $DeclaredSha256
        attestation_sidecar_path = $FinalSidecarPath
    }
}

function Remove-OwnedTemporaryAttestation {
    if ([string]::IsNullOrWhiteSpace($script:TemporaryAttestationDirectory)) {
        return
    }
    $ExpectedParent = Resolve-ProjectPath $AttestationRootRelativePath
    $ResolvedParent = [IO.Path]::GetDirectoryName($script:TemporaryAttestationDirectory)
    $Leaf = [IO.Path]::GetFileName($script:TemporaryAttestationDirectory)
    if (-not $ResolvedParent.Equals($ExpectedParent, [StringComparison]::OrdinalIgnoreCase) -or
        -not $Leaf.StartsWith(".tmp-", [StringComparison]::Ordinal)) {
        throw "refused to clean an unexpected temporary attestation directory"
    }
    foreach ($Name in @(
        "lease_receipt.json",
        "lease_receipt.json.sha256",
        "gpu_attestation.json",
        "gpu_attestation.json.sha256"
    )) {
        $Path = Join-Path $script:TemporaryAttestationDirectory $Name
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            [IO.File]::Delete($Path)
        }
    }
    if (Test-Path -LiteralPath $script:TemporaryAttestationDirectory -PathType Container) {
        [IO.Directory]::Delete($script:TemporaryAttestationDirectory, $false)
    }
    $script:TemporaryAttestationDirectory = $null
}

$RunnerPath = Resolve-ProjectPath $RunnerRelativePath
$ImplementationPath = Resolve-ProjectPath $ImplementationRelativePath
$CanonicalConfigPath = Resolve-ProjectPath $ConfigRelativePath
$RequestedConfigPath = if ([IO.Path]::IsPathRooted($Config)) {
    [IO.Path]::GetFullPath($Config)
}
else {
    Resolve-ProjectPath $Config
}
if (-not $RequestedConfigPath.Equals(
    $CanonicalConfigPath,
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw "only the canonical frozen config is accepted: $ConfigRelativePath"
}
if (-not (Test-Path -LiteralPath $RunnerPath -PathType Leaf)) {
    throw "five-role Python runner is unavailable: $RunnerRelativePath"
}
if (-not (Test-Path -LiteralPath $ImplementationPath -PathType Leaf)) {
    throw "five-role Python implementation is unavailable: $ImplementationRelativePath"
}
if (-not (Test-Path -LiteralPath $CanonicalConfigPath -PathType Leaf)) {
    throw "five-role training config is unavailable: $ConfigRelativePath"
}
$PythonRuntime = Resolve-PythonRuntime -Requested $Python
$PythonExecutable = $PythonRuntime.path

if (-not $Execute) {
    & $PythonExecutable $RunnerPath --config $CanonicalConfigPath --dry-run
    if ($LASTEXITCODE -ne 0) {
        throw "Gemma five-role model-free preflight failed with exit $LASTEXITCODE"
    }
    exit 0
}

if ([string]::IsNullOrWhiteSpace($ExpectedGpuUuid)) {
    $ExpectedGpuUuid = [string]$env:ANCHOR_GEMMA_GPU_UUID
}
$ExpectedGpuUuid = $ExpectedGpuUuid.Trim()
if ([string]::IsNullOrWhiteSpace($ExpectedGpuUuid) -or
    $ExpectedGpuUuid -ieq "UNBOUND" -or
    $ExpectedGpuUuid -notmatch "^GPU-[0-9A-Fa-f-]{32,64}$") {
    throw (
        "Execute requires a bound GPU UUID via -ExpectedGpuUuid or " +
        "ANCHOR_GEMMA_GPU_UUID; empty, UNBOUND, and malformed values are refused"
    )
}

$NvidiaSmiExecutable = Resolve-Executable -Candidate $NvidiaSmiPath -Label "nvidia-smi"
$CanonicalLockPath = Resolve-ProjectPath $CanonicalLockRelativePath
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($CanonicalLockPath)) | Out-Null

$PreviousEnvironment = [ordered]@{
    ANCHOR_GEMMA_GPU_UUID = $env:ANCHOR_GEMMA_GPU_UUID
    CUDA_VISIBLE_DEVICES = $env:CUDA_VISIBLE_DEVICES
    HF_HUB_OFFLINE = $env:HF_HUB_OFFLINE
    TRANSFORMERS_OFFLINE = $env:TRANSFORMERS_OFFLINE
    TOKENIZERS_PARALLELISM = $env:TOKENIZERS_PARALLELISM
}

try {
    Assert-NoCompetingLocks -Phase "pre-lock"
    $PreLockSamples = @(Get-GpuSampleSeries `
        -NvidiaSmi $NvidiaSmiExecutable `
        -ExpectedUuid $ExpectedGpuUuid `
        -Phase "pre_lock")
    # Use nvidia-smi's canonical spelling for all downstream cross-bindings.
    $ExpectedGpuUuid = [string]$PreLockSamples[0].uuid

    $RunId = [Guid]::NewGuid().ToString("N")
    $LockOwner = [ordered]@{
        schema_version = "anchor.gemma3-1b-it-five-role-qonly-lock-owner.v1"
        run_id = $RunId
        launcher_pid = $PID
        launcher_process_start_utc = (
            (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString("o")
        )
        acquired_at_utc = [DateTime]::UtcNow.ToString("o")
        lock_path = $CanonicalLockRelativePath
        expected_gpu_uuid = $ExpectedGpuUuid
        gpu_index = 0
        launcher_sha256 = Get-Sha256 $PSCommandPath
        runner_sha256 = Get-Sha256 $RunnerPath
        implementation_path = $ImplementationRelativePath
        implementation_sha256 = Get-Sha256 $ImplementationPath
        config_sha256 = Get-Sha256 $CanonicalConfigPath
        roles = $Roles
        concurrency = 1
        smoke_steps_per_role = 2
        full_steps_per_role = 160
        fresh_base_per_phase = $true
        fresh_adapter_per_phase = $true
        resume_allowed = $false
        diagnostic_only = $true
    }
    $LockBytes = ConvertTo-CanonicalJsonBytes $LockOwner
    $LockSha256 = Get-BytesSha256 $LockBytes
    $LockOptions = [IO.FileOptions]::DeleteOnClose -bor [IO.FileOptions]::WriteThrough
    $LockStream = [IO.FileStream]::new(
        $CanonicalLockPath,
        [IO.FileMode]::CreateNew,
        [IO.FileAccess]::Write,
        [IO.FileShare]::None,
        4096,
        $LockOptions
    )
    $LockStream.Write($LockBytes, 0, $LockBytes.Length)
    $LockStream.Flush($true)

    Assert-NoHandoffLocks -Phase "post-lock"
    $PostLockSamples = @(Get-GpuSampleSeries `
        -NvidiaSmi $NvidiaSmiExecutable `
        -ExpectedUuid $ExpectedGpuUuid `
        -Phase "post_lock")
    Assert-StableGpuIdentity `
        -PreLockSamples $PreLockSamples `
        -PostLockSamples $PostLockSamples

    $LauncherSha256 = Get-Sha256 $PSCommandPath
    $ImplementationSha256 = Get-Sha256 $ImplementationPath
    $RunnerScriptSha256 = Get-Sha256 $RunnerPath
    $ConfigSha256 = Get-Sha256 $CanonicalConfigPath
    $ComputeBaseline = @($PreLockSamples[0].compute_processes)
    $ComputeInventorySha256 = [string]$PreLockSamples[0].compute_inventory_sha256
    $WddmGuiAllowlistSha256 = Get-BytesSha256 (
        ConvertTo-CanonicalJsonBytes @($GpuPolicy.wddm_gui_process_allowlist)
    )
    $KvRuntimeBoundary = [ordered]@{
        shared_prefix_adapter_state = "off"
        shared_prefix_read_only = $true
        identical_ordered_prefix_lineage_only = $true
        expert_activation = "q_only"
        expert_private_tail_append_only = $true
        private_tail_includes_post_activation_prompt_and_generated_tokens = $true
        private_tail_cross_expert_reuse = $false
        committed_text_reencoded_for_next_shared_context = $true
        full_generation_kv_shared_claimed = $false
        normal_in_stack_q_lora_exact_kv_sharing_claimed = $false
        token_level_moe_claimed = $false
        runtime_private_tail_materialized = $false
    }
    $CommonClaims = [ordered]@{
        diagnostic_only = $true
        proxy_only = $true
        training_authorized = $false
        formal_training_authorized = $false
        formal = $false
        quality_claimed = $false
        generalization_claimed = $false
    }
    $LeaseReceipt = [ordered]@{
        schema_version = "anchor.gemma3-1b-it-five-role-qonly-execution-lease.v1"
        status = "passed"
        run_id = $RunId
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        canonical_lock = $CanonicalLockRelativePath
        canonical_lock_sha256 = $LockSha256
        canonical_lock_held_at_publish = $true
        launcher_pid = $PID
        expected_gpu_index = 0
        expected_gpu_uuid = $ExpectedGpuUuid
        roles = $Roles
        smoke_steps = 2
        full_steps = 160
        concurrency = 1
        config_sha256 = $ConfigSha256
        implementation_sha256 = $ImplementationSha256
        runner_script_sha256 = $RunnerScriptSha256
        launcher_sha256 = $LauncherSha256
        python_runtime = $PythonRuntime
        wddm_gui_process_allowlist_sha256 = $WddmGuiAllowlistSha256
        compute_inventory_sha256 = $ComputeInventorySha256
        compute_processes = $ComputeBaseline
        fresh_base_per_phase = $true
        fresh_adapter_per_phase = $true
        resume_allowed = $false
        kv_runtime_boundary = $KvRuntimeBoundary
        claims = $CommonClaims
    }
    $LeaseReceiptSha256 = Get-BytesSha256 (
        ConvertTo-CanonicalJsonBytes $LeaseReceipt
    )
    $Attestation = [ordered]@{
        schema_version = "anchor.gemma3-1b-it-five-role-qonly-gpu-attestation.v1"
        status = "passed"
        run_id = $RunId
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        canonical_lock = $CanonicalLockRelativePath
        canonical_lock_sha256 = $LockSha256
        launcher_pid = $PID
        expected_gpu_index = 0
        expected_gpu_uuid = $ExpectedGpuUuid
        roles = $Roles
        smoke_steps = 2
        full_steps = 160
        concurrency = 1
        config_sha256 = $ConfigSha256
        implementation_sha256 = $ImplementationSha256
        runner_script_sha256 = $RunnerScriptSha256
        lease_receipt_sha256 = $LeaseReceiptSha256
        wddm_gui_process_allowlist_sha256 = $WddmGuiAllowlistSha256
        compute_inventory_sha256 = $ComputeInventorySha256
        launcher = [ordered]@{
            path = Get-RelativeProjectPath $PSCommandPath
            sha256 = $LauncherSha256
        }
        runner = [ordered]@{
            path = $RunnerRelativePath
            sha256 = $RunnerScriptSha256
        }
        implementation = [ordered]@{
            path = $ImplementationRelativePath
            sha256 = $ImplementationSha256
        }
        config = [ordered]@{
            path = $ConfigRelativePath
            sha256 = $ConfigSha256
        }
        python_runtime = $PythonRuntime
        nvidia_smi = [ordered]@{
            path = $NvidiaSmiExecutable
            sha256 = Get-Sha256 $NvidiaSmiExecutable
            calls = 12
        }
        lock = [ordered]@{
            path = $CanonicalLockRelativePath
            content_sha256 = $LockSha256
            file_mode = "CreateNew"
            file_share = "None"
            delete_on_close = $true
            held_for_entire_orchestrator = $true
        }
        gpu_policy = $GpuPolicy
        sample_count = 3
        pre_lock_samples = $PreLockSamples
        post_lock_samples = $PostLockSamples
        compute_processes = $ComputeBaseline
        execution_plan = [ordered]@{
            gpu_index = 0
            concurrency = 1
            roles = $Roles
            smoke_steps_per_role = 2
            full_steps_per_role = 160
            phase_order = @("smoke", "full")
            fresh_base_per_phase = $true
            fresh_adapter_per_phase = $true
            resume_allowed = $false
        }
        kv_runtime_boundary = $KvRuntimeBoundary
        claims = $CommonClaims
        prestart_counters = [ordered]@{
            provider_requests = 0
            network_requests = 0
            protected_body_reads = 0
            model_loads = 0
            gpu_training_requests = 0
            nvidia_smi_telemetry_calls = 12
        }
    }
    $PublishedGates = Publish-ExecutionGateArtifacts `
        -LeaseReceipt $LeaseReceipt `
        -Attestation $Attestation `
        -RunId $RunId

    $env:ANCHOR_GEMMA_GPU_UUID = $ExpectedGpuUuid
    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:TOKENIZERS_PARALLELISM = "false"

    Write-Host (
        "Starting Gemma 3 1B IT Q-only diagnostic: " +
        "GPU=$ExpectedGpuUuid, config=$ConfigSha256, " +
        "smoke=2/role, full=160/role, concurrency=1, lock=$CanonicalLockRelativePath"
    )
    & $PythonExecutable $RunnerPath `
        --config $CanonicalConfigPath `
        --execute `
        --lease-receipt $PublishedGates.lease_path `
        --lease-receipt-sha256 $PublishedGates.lease_sha256 `
        --gpu-attestation $PublishedGates.attestation_path `
        --gpu-attestation-sha256 $PublishedGates.attestation_sha256 `
        --run-id $RunId
    if ($LASTEXITCODE -ne 0) {
        throw "Gemma five-role diagnostic orchestrator failed with exit $LASTEXITCODE"
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
    Remove-OwnedTemporaryAttestation
}
