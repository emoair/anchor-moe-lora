# Additive v2 process-lifetime helpers for the Gemma 3 Q8 QLoRA launcher.
#
# Dot-source this file from the v2 launcher after it has acquired the canonical
# FileShare.None/DeleteOnClose lock.  The helper starts Python as a tracked
# process, attaches it to a kill-on-close Windows job, and does not return until
# Python exits.  Consequently the caller keeps its canonical lock handle for
# the complete Python lifetime, while an abnormal launcher termination closes
# the job handle and prevents a detached training orphan.

if ($null -eq ("Anchor.GemmaQ8LauncherV2.KillOnCloseJob" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

namespace Anchor.GemmaQ8LauncherV2
{
    public sealed class KillOnCloseJob : IDisposable
    {
        private const UInt32 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
        private IntPtr handle;

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public Int64 PerProcessUserTimeLimit;
            public Int64 PerJobUserTimeLimit;
            public UInt32 LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public UInt32 ActiveProcessLimit;
            public UIntPtr Affinity;
            public UInt32 PriorityClass;
            public UInt32 SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IO_COUNTERS
        {
            public UInt64 ReadOperationCount;
            public UInt64 WriteOperationCount;
            public UInt64 OtherOperationCount;
            public UInt64 ReadTransferCount;
            public UInt64 WriteTransferCount;
            public UInt64 OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        private enum JOBOBJECTINFOCLASS
        {
            JobObjectExtendedLimitInformation = 9
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObject(
            IntPtr jobAttributes,
            String name
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetInformationJobObject(
            IntPtr job,
            JOBOBJECTINFOCLASS informationClass,
            ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION information,
            UInt32 informationLength
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool AssignProcessToJobObject(
            IntPtr job,
            IntPtr process
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CloseHandle(IntPtr value);

        public KillOnCloseJob()
        {
            handle = CreateJobObject(IntPtr.Zero, null);
            if (handle == IntPtr.Zero)
            {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "CreateJobObject failed"
                );
            }

            JOBOBJECT_EXTENDED_LIMIT_INFORMATION information =
                new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            information.BasicLimitInformation.LimitFlags =
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!SetInformationJobObject(
                handle,
                JOBOBJECTINFOCLASS.JobObjectExtendedLimitInformation,
                ref information,
                (UInt32)Marshal.SizeOf(
                    typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
                )
            ))
            {
                Int32 error = Marshal.GetLastWin32Error();
                CloseHandle(handle);
                handle = IntPtr.Zero;
                throw new Win32Exception(
                    error,
                    "SetInformationJobObject failed"
                );
            }
        }

        public void Assign(IntPtr processHandle)
        {
            if (handle == IntPtr.Zero)
            {
                throw new ObjectDisposedException("KillOnCloseJob");
            }
            if (!AssignProcessToJobObject(handle, processHandle))
            {
                throw new Win32Exception(
                    Marshal.GetLastWin32Error(),
                    "AssignProcessToJobObject failed"
                );
            }
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero)
            {
                CloseHandle(handle);
                handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~KillOnCloseJob()
        {
            Dispose();
        }
    }

    public static class CommandLine
    {
        public static String QuoteArgument(String argument)
        {
            if (argument == null)
            {
                throw new ArgumentNullException("argument");
            }
            if (argument.Length > 0 &&
                argument.IndexOfAny(
                    new Char[] { ' ', '\t', '\r', '\n', '\v', '"' }
                ) < 0)
            {
                return argument;
            }

            StringBuilder result = new StringBuilder();
            result.Append('"');
            Int32 backslashes = 0;
            foreach (Char value in argument)
            {
                if (value == '\\')
                {
                    backslashes++;
                }
                else if (value == '"')
                {
                    result.Append('\\', (backslashes * 2) + 1);
                    result.Append('"');
                    backslashes = 0;
                }
                else
                {
                    result.Append('\\', backslashes);
                    result.Append(value);
                    backslashes = 0;
                }
            }
            result.Append('\\', backslashes * 2);
            result.Append('"');
            return result.ToString();
        }

        public static String BuildArguments(String[] arguments)
        {
            if (arguments == null)
            {
                throw new ArgumentNullException("arguments");
            }
            StringBuilder result = new StringBuilder();
            for (Int32 index = 0; index < arguments.Length; index++)
            {
                if (index > 0)
                {
                    result.Append(' ');
                }
                result.Append(QuoteArgument(arguments[index]));
            }
            return result.ToString();
        }
    }
}
"@
}

function Assert-AnchorGemmaQ8CanonicalLockHeld {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [IO.FileStream]$CanonicalLockStream,

        [Parameter(Mandatory = $true)]
        [string]$CanonicalLockPath
    )

    if ($null -eq $CanonicalLockStream -or
        -not $CanonicalLockStream.CanWrite -or
        $CanonicalLockStream.SafeFileHandle.IsClosed) {
        throw "canonical training lock stream is not held"
    }
    $ExpectedPath = [IO.Path]::GetFullPath($CanonicalLockPath)
    $StreamPath = [IO.Path]::GetFullPath($CanonicalLockStream.Name)
    if (-not [StringComparer]::OrdinalIgnoreCase.Equals(
        $ExpectedPath,
        $StreamPath
    )) {
        throw "canonical training lock stream/path mismatch"
    }
    if (-not [IO.File]::Exists($ExpectedPath)) {
        throw "canonical training lock path disappeared"
    }
}

function Invoke-AnchorGemmaQ8PythonAndWait {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExecutable,

        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [string[]]$ArgumentList,

        [Parameter(Mandatory = $true)]
        [IO.FileStream]$CanonicalLockStream,

        [Parameter(Mandatory = $true)]
        [string]$CanonicalLockPath,

        [string]$WorkingDirectory = "",

        [ValidateRange(25, 5000)]
        [int]$PollIntervalMilliseconds = 250
    )

    Assert-AnchorGemmaQ8CanonicalLockHeld `
        -CanonicalLockStream $CanonicalLockStream `
        -CanonicalLockPath $CanonicalLockPath
    if (-not [IO.File]::Exists([IO.Path]::GetFullPath($PythonExecutable))) {
        throw "Python executable does not exist"
    }
    if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory) -and
        -not [IO.Directory]::Exists([IO.Path]::GetFullPath($WorkingDirectory))) {
        throw "Python working directory does not exist"
    }

    $StartInfo = [Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = [IO.Path]::GetFullPath($PythonExecutable)
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        $StartInfo.WorkingDirectory = [IO.Path]::GetFullPath($WorkingDirectory)
    }

    $ArgumentListProperty = $StartInfo.GetType().GetProperty("ArgumentList")
    if ($null -ne $ArgumentListProperty) {
        $NativeArguments = $ArgumentListProperty.GetValue($StartInfo, $null)
        foreach ($Argument in $ArgumentList) {
            [void]$NativeArguments.Add([string]$Argument)
        }
    }
    else {
        $StartInfo.Arguments = (
            [Anchor.GemmaQ8LauncherV2.CommandLine]::BuildArguments($ArgumentList)
        )
    }

    $Job = [Anchor.GemmaQ8LauncherV2.KillOnCloseJob]::new()
    $Process = [Diagnostics.Process]::new()
    $Process.StartInfo = $StartInfo
    $Started = $false
    try {
        $Started = $Process.Start()
        if (-not $Started) {
            throw "Python process failed to start"
        }
        $ProcessId = $Process.Id
        try {
            $Job.Assign($Process.Handle)
        }
        catch {
            if (-not $Process.HasExited) {
                throw
            }
        }

        while (-not $Process.WaitForExit($PollIntervalMilliseconds)) {
            Assert-AnchorGemmaQ8CanonicalLockHeld `
                -CanonicalLockStream $CanonicalLockStream `
                -CanonicalLockPath $CanonicalLockPath
        }
        # Required by System.Diagnostics when asynchronous stream machinery is
        # present; harmless for inherited stdout/stderr and closes the wait.
        $Process.WaitForExit()
        Assert-AnchorGemmaQ8CanonicalLockHeld `
            -CanonicalLockStream $CanonicalLockStream `
            -CanonicalLockPath $CanonicalLockPath

        return [PSCustomObject]@{
            process_id = $ProcessId
            exit_code = [int]$Process.ExitCode
            waited = $true
            canonical_lock_held_through_exit = $true
            kill_on_launcher_exit = $true
        }
    }
    finally {
        # On an exception this closes the job first, terminating only the
        # Python process launched above (and any descendants in the same job).
        # On normal completion the process has already exited.
        if ($null -ne $Job) {
            $Job.Dispose()
        }
        if ($Started -and -not $Process.HasExited) {
            [void]$Process.WaitForExit(5000)
        }
        $Process.Dispose()
    }
}
