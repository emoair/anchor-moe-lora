# One anonymous-stdin credential feeds the controller's full in-process phase chain.
[CmdletBinding()]
param(
    [string]$ProducerRoot = "D:\LLM\anchor-moe-lora-gemma3-chat-rollover-v3",
    [string]$PythonPath = "D:\LLM\envs\gemma3-keras-torch\Scripts\python.exe",
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")]
    [string]$ImplementerId = "producer.implementer",
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9_.:-]{2,127}$")]
    [string]$ReviewerId = "independent.reviewer",
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$moduleName = "anchor_mvp.data.gemma3_chat_unbalanced_v2_consumer_identity_rollover_v3"
$executeBootstrapBase64 = (
    "aW1wb3J0IG9zLHJ1bnB5LHN5cztzaW5rPW9zLm9wZW4ob3MuZGV2bnVsbCxvcy5PX1dST05M" +
    "WSk7b3MuZHVwMihzaW5rLDEpIGlmIHNpbmshPTEgZWxzZSBOb25lO29zLmR1cDIoc2luaywy" +
    "KSBpZiBzaW5rIT0yIGVsc2UgTm9uZTtvcy5jbG9zZShzaW5rKSBpZiBzaW5rPjIgZWxzZSBO" +
    "b25lO3N5cy5zdGRvdXQ9b3Blbihvcy5kZXZudWxsLCd3JyxlbmNvZGluZz0ndXRmLTgnKTtz" +
    "eXMuc3RkZXJyPW9wZW4ob3MuZGV2bnVsbCwndycsZW5jb2Rpbmc9J3V0Zi04Jyk7bW9kdWxl" +
    "PXN5cy5hcmd2WzFdO3N5cy5hcmd2PVttb2R1bGUsKnN5cy5hcmd2WzI6XV07cnVucHkucnVu" +
    "X21vZHVsZShtb2R1bGUscnVuX25hbWU9J19fbWFpbl9fJyk="
)
$executeBootstrap = (
    "import base64;exec(base64.b64decode('$executeBootstrapBase64'))"
)
$relativeOutputRoot = "data\gemma3_chat_five_expert_qonly_unbalanced_v2_teacher_alignment_v1\shard-0001"

function Resolve-ContainedPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Child
    )

    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $resolvedChild = [System.IO.Path]::GetFullPath($Child)
    $prefix = $resolvedRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedChild.StartsWith(
        $prefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "secure_session_path_escaped"
    }
    return $resolvedChild
}

function Write-SecureAsciiLine {
    param(
        [Parameter(Mandatory = $true)]
        [System.Security.SecureString]$Secret,
        [Parameter(Mandatory = $true)]
        [System.IO.Stream]$Stream
    )

    $bstr = [System.IntPtr]::Zero
    try {
        $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secret)
        $byteLength = [System.Runtime.InteropServices.Marshal]::ReadInt32(
            $bstr,
            -4
        )
        $characterCount = [int]($byteLength / 2)
        if ($characterCount -lt 1) {
            throw "secure_session_credential_empty"
        }
        for ($index = 0; $index -lt $characterCount; $index++) {
            $codePoint = [System.Runtime.InteropServices.Marshal]::ReadInt16(
                $bstr,
                $index * 2
            )
            if ($codePoint -lt 0x21 -or $codePoint -gt 0x7e) {
                throw "secure_session_credential_not_ascii"
            }
        }
        for ($index = 0; $index -lt $characterCount; $index++) {
            $codePoint = [System.Runtime.InteropServices.Marshal]::ReadInt16(
                $bstr,
                $index * 2
            )
            $Stream.WriteByte([byte]$codePoint)
        }
        $Stream.WriteByte(0x0a)
        $Stream.Flush()
    }
    finally {
        if ($bstr -ne [System.IntPtr]::Zero) {
            [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
    }
}

function New-ControllerStartInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Arguments,
        [Parameter(Mandatory = $true)]
        [bool]$RedirectInput,
        [Parameter(Mandatory = $true)]
        [bool]$RedirectOutput
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $pythonResolved
    $startInfo.Arguments = $Arguments
    $startInfo.WorkingDirectory = $producerRootResolved
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardInput = $RedirectInput
    $startInfo.RedirectStandardOutput = $RedirectOutput
    $startInfo.RedirectStandardError = $RedirectOutput
    $startInfo.CreateNoWindow = $true
    $startInfo.EnvironmentVariables["PYTHONPATH"] = $sourceRoot
    return $startInfo
}

function Get-ControllerPreflight {
    $startInfo = New-ControllerStartInfo `
        -Arguments ("-m {0} --validate-only" -f $moduleName) `
        -RedirectInput $false `
        -RedirectOutput $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "secure_session_preflight_start_failed"
    }
    try {
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(120000)) {
            try {
                $process.Kill()
            }
            catch [System.InvalidOperationException] {
                # The process exited between the timeout and Kill().
            }
            [void]$process.WaitForExit(5000)
            throw "secure_session_preflight_timeout"
        }
        $stdout = $stdoutTask.Result.Trim()
        $stderr = $stderrTask.Result
        $exitCode = $process.ExitCode
    }
    finally {
        $process.Dispose()
    }
    if (-not $stdout) {
        throw "secure_session_preflight_output_missing"
    }
    try {
        $result = $stdout | ConvertFrom-Json
    }
    catch {
        throw "secure_session_preflight_output_invalid"
    }
    if ($stderr) {
        throw "secure_session_preflight_stderr_nonempty"
    }
    if ($exitCode -notin @(0, 2)) {
        throw "secure_session_preflight_exit_invalid"
    }
    return $result
}

$producerRootResolved = [System.IO.Path]::GetFullPath($ProducerRoot)
if ($ImplementerId.Equals(
    $ReviewerId,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "secure_session_reviewer_separation_invalid"
}
if (-not (Test-Path -LiteralPath $producerRootResolved -PathType Container)) {
    throw "secure_session_producer_root_missing"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "secure_session_python_missing"
}
$pythonResolved = [System.IO.Path]::GetFullPath($PythonPath)
$venvRoot = Split-Path -Parent (Split-Path -Parent $pythonResolved)
$venvConfig = Join-Path $venvRoot "pyvenv.cfg"
if (Test-Path -LiteralPath $venvConfig -PathType Leaf) {
    $homeLine = Get-Content -LiteralPath $venvConfig |
        Where-Object { $_ -match "^\s*home\s*=\s*(.+?)\s*$" } |
        Select-Object -First 1
    if ($homeLine -and $homeLine -match "^\s*home\s*=\s*(.+?)\s*$") {
        $directPython = Join-Path $Matches[1] "python.exe"
        if (-not (Test-Path -LiteralPath $directPython -PathType Leaf)) {
            throw "secure_session_direct_python_missing"
        }
        $pythonResolved = [System.IO.Path]::GetFullPath($directPython)
    }
}
$sourceRoot = Resolve-ContainedPath `
    -Root $producerRootResolved `
    -Child (Join-Path $producerRootResolved "src")
$dataRoot = Resolve-ContainedPath `
    -Root $producerRootResolved `
    -Child (Join-Path $producerRootResolved "data")
$outputRoot = Resolve-ContainedPath `
    -Root $dataRoot `
    -Child (Join-Path $producerRootResolved $relativeOutputRoot)

$hashAlgorithm = [System.Security.Cryptography.SHA256]::Create()
try {
    $mutexDigest = $hashAlgorithm.ComputeHash(
        [System.Text.Encoding]::UTF8.GetBytes($outputRoot.ToLowerInvariant())
    )
}
finally {
    $hashAlgorithm.Dispose()
}
$mutexSuffix = -join (
    $mutexDigest[0..15] | ForEach-Object { $_.ToString("x2") }
)
$mutex = [System.Threading.Mutex]::new(
    $false,
    "Local\anchor-gemma3-glm-rollover-$mutexSuffix"
)
$ownsMutex = $false
try {
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) {
        throw "secure_session_already_running"
    }

    $preflight = Get-ControllerPreflight
    $allowedBlockers = @(
        "controller_credential_slot_unloaded",
        "runtime_hmac_slot_unloaded"
    )
    $unexpectedBlockers = @(
        $preflight.blockers | Where-Object { $_ -notin $allowedBlockers }
    )
    if (Test-Path -LiteralPath $outputRoot) {
        $outputItem = Get-Item -LiteralPath $outputRoot -Force
        if (-not $outputItem.PSIsContainer) {
            $unexpectedBlockers += "controller_output_root_invalid"
        }
        elseif (
            ($outputItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
        ) {
            $unexpectedBlockers += "controller_output_root_reparse_forbidden"
        }
        elseif (
            [System.IO.Directory]::EnumerateFileSystemEntries(
                $outputRoot
            ).GetEnumerator().MoveNext()
        ) {
            $unexpectedBlockers += "controller_existing_output_not_resumable"
        }
    }
    $unexpectedBlockers = @($unexpectedBlockers | Sort-Object -Unique)
    $ready = $unexpectedBlockers.Count -eq 0
    if ($ValidateOnly -or -not $ready) {
        [pscustomobject]@{
            schema_version = "anchor.gemma3-glm-rollover-secure-session.v1"
            state = if ($ready) { "ready" } else { "blocked" }
            blockers = $unexpectedBlockers
            credential_prompt_would_open = $ready
            credential_inputs_per_controller = 1
            credential_transport = "anonymous_stdin"
            credential_persisted = $false
            cross_process_resume = $false
            output_root_relative = $relativeOutputRoot.Replace("\", "/")
        } | ConvertTo-Json -Compress
        if ($ready) {
            exit 0
        }
        exit 2
    }

    $credential = Read-Host `
        -Prompt "Ark API key (one input; memory-only for this controller)" `
        -AsSecureString
    try {
        $startInfo = New-ControllerStartInfo `
            -Arguments (
                (
                    '-c "{0}" {1} --execute --credential-stdin ' +
                    "--implementer-id {2} --reviewer-id {3}"
                ) -f (
                    $executeBootstrap,
                    $moduleName,
                    $ImplementerId,
                    $ReviewerId
                )
            ) `
            -RedirectInput $true `
            -RedirectOutput $false
        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) {
            throw "secure_session_controller_start_failed"
        }
        try {
            try {
                Write-SecureAsciiLine `
                    -Secret $credential `
                    -Stream $process.StandardInput.BaseStream
                $process.StandardInput.Close()
            }
            catch {
                try {
                    if (-not $process.HasExited) {
                        $process.Kill()
                    }
                }
                catch [System.InvalidOperationException] {
                    # The process exited between HasExited and Kill().
                }
                if (-not $process.WaitForExit(5000)) {
                    throw "secure_session_child_termination_unconfirmed"
                }
                try {
                    $process.StandardInput.Close()
                }
                catch {
                    # The child may already have closed the anonymous pipe.
                }
                throw
            }
            $process.WaitForExit()
            $exitCode = $process.ExitCode
        }
        finally {
            $process.Dispose()
        }

        [pscustomobject]@{
            schema_version = "anchor.gemma3-glm-rollover-secure-session.v1"
            state = if ($exitCode -eq 0) { "complete" } else { "blocked" }
            controller_exit_code = $exitCode
            credential_inputs = 1
            credential_persisted = $false
            cross_process_resume = $false
        } | ConvertTo-Json -Compress
        exit $exitCode
    }
    finally {
        if ($null -ne $credential) {
            $credential.Dispose()
        }
    }
}
finally {
    if ($ownsMutex) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
