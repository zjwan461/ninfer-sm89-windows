<#
.SYNOPSIS
  NInfer Launcher bootstrap: locate a usable Python 3.10+, download the official
  embeddable Python when the machine has none, install the launcher dependencies,
  then start the WebUI.

.DESCRIPTION
  Interpreter lookup order (first usable wins):
    1. the NINFER_LAUNCHER_PYTHON environment variable
    2. a previously bootstrapped runtime
       (%LOCALAPPDATA%\NInfer-Launcher\runtime\python)
    3. the Windows py launcher (py -3.13 / -3.12 / -3.11 / -3.10)
    4. python / python3 on PATH
    5. common install locations (python.org installer, miniconda/anaconda, scoop, ...)
    6. conda on PATH (conda info --base, conda env list --json)

  When nothing is found, python.org's official embeddable package
  (python-<version>-embed-amd64.zip) is downloaded once, site-packages is enabled in
  python*._pth, pip is installed with get-pip.py, and the launcher dependencies are
  installed into that private runtime. The system Python is never modified.

  Proxy: HTTPS_PROXY / HTTP_PROXY (or the lowercase forms) are honored for all
  downloads and for pip.

  Environment overrides:
    NINFER_LAUNCHER_PYTHON           force a specific interpreter
    NINFER_LAUNCHER_HOME             launcher data dir (default %LOCALAPPDATA%\NInfer-Launcher)
    NINFER_LAUNCHER_PYTHON_VERSION   embeddable Python version (default 3.11.9)
    NINFER_LAUNCHER_PYTHON_URL       full URL of the embeddable zip (overrides the version)

  This file is intentionally ASCII-only without a BOM: Windows PowerShell 5.1 decodes
  .ps1 sources as ANSI, and non-ASCII text would corrupt the parser.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$LauncherArgs
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$MinMinor = 10
$PinnedVersion = if ($env:NINFER_LAUNCHER_PYTHON_VERSION) { $env:NINFER_LAUNCHER_PYTHON_VERSION } else { '3.11.9' }
# SHA256 of python-3.11.9-embed-amd64.zip; other versions are downloaded unverified.
$PinnedSha256 = '009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b'

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LauncherScript = Join-Path $PSScriptRoot 'launcher.py'
$Requirements = Join-Path $PSScriptRoot 'requirements.txt'

$HomeDir = if ($env:NINFER_LAUNCHER_HOME) {
    $env:NINFER_LAUNCHER_HOME
} elseif ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA 'NInfer-Launcher'
} else {
    Join-Path (Join-Path $env:USERPROFILE 'AppData\Local') 'NInfer-Launcher'
}
$RuntimeDir = Join-Path $HomeDir 'runtime'
$PythonDir = Join-Path $RuntimeDir 'python'
$PortableExe = Join-Path $PythonDir 'python.exe'

$EmbedUrl = if ($env:NINFER_LAUNCHER_PYTHON_URL) {
    $env:NINFER_LAUNCHER_PYTHON_URL
} else {
    "https://www.python.org/ftp/python/$PinnedVersion/python-$PinnedVersion-embed-amd64.zip"
}

function Write-Step([string]$Message) {
    Write-Host "[launcher] $Message" -ForegroundColor Cyan
}

# Native tools write warnings to stderr; with ErrorActionPreference=Stop that would
# terminate the script. Every native call goes through these wrappers instead.
function Invoke-Native([string]$FilePath, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Collect output first, then echo it, so nothing leaks into the return value.
        $lines = & $FilePath @Arguments 2>&1
        foreach ($line in $lines) { Write-Host "$line" }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Invoke-NativeCapture([string]$FilePath, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $FilePath @Arguments 2>$null
        return @{ Code = $LASTEXITCODE; Output = $output }
    } finally {
        $ErrorActionPreference = $previous
    }
}

# Run a long command with stdout/stderr captured, echo them, and return the exit code.
# ProcessStartInfo avoids cmd.exe quoting entirely.
function Invoke-NativeStreaming([string]$FilePath, [string[]]$Arguments) {
    $quoted = @()
    foreach ($argument in $Arguments) {
        if ($argument -match '[\s"]') {
            $quoted += '"' + ($argument -replace '"', '\"') + '"'
        } else {
            $quoted += $argument
        }
    }
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = ($quoted -join ' ')
    $startInfo.WorkingDirectory = (Get-Location).Path
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    try {
        $process = [System.Diagnostics.Process]::Start($startInfo)
    } catch {
        Write-Host "failed to start ${FilePath}: $($_.Exception.Message)"
        return 1
    }
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    if ($stdout) { Write-Host $stdout.TrimEnd() }
    if ($stderr) { Write-Host $stderr.TrimEnd() }
    return $process.ExitCode
}

# Standard proxy environment variables are honored by every download below and by pip.
function Get-Proxy {
    if ($env:HTTPS_PROXY) { return $env:HTTPS_PROXY }
    if ($env:HTTP_PROXY) { return $env:HTTP_PROXY }
    if ($env:https_proxy) { return $env:https_proxy }
    if ($env:http_proxy) { return $env:http_proxy }
    return $null
}

function Invoke-Download([string]$Uri, [string]$OutFile) {
    $proxy = Get-Proxy
    $arguments = @{ Uri = $Uri; OutFile = $OutFile; UseBasicParsing = $true }
    if ($proxy) {
        $arguments['Proxy'] = $proxy
        Write-Step "Using proxy $proxy"
    }
    Invoke-WebRequest @arguments
}

function Test-PythonUsable([string]$Path) {
    if (-not $Path) { return $false }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $result = Invoke-NativeCapture $Path @('-c', 'import sys; print(sys.version_info[0], sys.version_info[1])')
    if ($result.Code -ne 0 -or -not $result.Output) { return $false }
    $parts = "$($result.Output)".Trim() -split '\s+'
    if ($parts.Count -lt 2) { return $false }
    if ($parts[0] -ne '3') { return $false }
    return ([int]$parts[1] -ge $MinMinor)
}

function Resolve-FromPyLauncher {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $launcher) { return $null }
    foreach ($minor in @('3.13', '3.12', '3.11', '3.10')) {
        $result = Invoke-NativeCapture $launcher.Source @("-$minor", '-c', 'import sys; print(sys.executable)')
        if ($result.Code -ne 0 -or -not $result.Output) { continue }
        $exe = "$($result.Output)".Trim()
        if (Test-PythonUsable $exe) { return $exe }
    }
    return $null
}

function Resolve-FromPath {
    foreach ($name in @('python', 'python3')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        if (Test-PythonUsable $command.Source) { return $command.Source }
    }
    return $null
}

function Resolve-FromKnownLocations {
    $patterns = @()
    if ($env:LOCALAPPDATA) {
        $patterns += (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python3*\python.exe')
        $patterns += (Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\python3.exe')
    }
    if ($env:ProgramFiles) {
        $patterns += (Join-Path $env:ProgramFiles 'Python3*\python.exe')
    }
    if (${env:ProgramFiles(x86)}) {
        $patterns += (Join-Path ${env:ProgramFiles(x86)} 'Python3*\python.exe')
    }
    if ($env:USERPROFILE) {
        $patterns += (Join-Path $env:USERPROFILE 'miniconda3\python.exe')
        $patterns += (Join-Path $env:USERPROFILE 'anaconda3\python.exe')
        $patterns += (Join-Path $env:USERPROFILE 'miniforge3\python.exe')
        $patterns += (Join-Path $env:USERPROFILE 'mambaforge\python.exe')
        $patterns += (Join-Path $env:USERPROFILE 'scoop\apps\python\current\python.exe')
    }
    if ($env:ProgramData) {
        $patterns += (Join-Path $env:ProgramData 'miniconda3\python.exe')
        $patterns += (Join-Path $env:ProgramData 'anaconda3\python.exe')
    }
    foreach ($pattern in $patterns) {
        try {
            $found = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        } catch {
            continue
        }
        if ($found -and (Test-PythonUsable $found.FullName)) { return $found.FullName }
    }
    return $null
}

function Resolve-FromConda {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $conda) { return $null }
    $base = Invoke-NativeCapture $conda.Source @('info', '--base')
    if ($base.Code -eq 0 -and $base.Output) {
        $candidate = Join-Path "$($base.Output)".Trim() 'python.exe'
        if (Test-PythonUsable $candidate) { return $candidate }
    }
    $json = Invoke-NativeCapture $conda.Source @('env', 'list', '--json')
    if ($json.Code -ne 0 -or -not $json.Output) { return $null }
    try {
        $parsed = "$($json.Output)" | ConvertFrom-Json
    } catch {
        return $null
    }
    if (-not $parsed -or -not $parsed.envs) { return $null }
    foreach ($envDir in $parsed.envs) {
        $candidate = Join-Path $envDir 'python.exe'
        if (Test-PythonUsable $candidate) { return $candidate }
    }
    return $null
}

function Find-Python {
    if ($env:NINFER_LAUNCHER_PYTHON) {
        if (Test-PythonUsable $env:NINFER_LAUNCHER_PYTHON) { return $env:NINFER_LAUNCHER_PYTHON }
        Write-Warning "NINFER_LAUNCHER_PYTHON=$($env:NINFER_LAUNCHER_PYTHON) is not a usable Python 3.$MinMinor+ interpreter; probing further."
    }
    if (Test-PythonUsable $PortableExe) { return $PortableExe }
    foreach ($resolver in @('Resolve-FromPyLauncher', 'Resolve-FromPath', 'Resolve-FromKnownLocations', 'Resolve-FromConda')) {
        $found = & $resolver
        if ($found) { return $found }
    }
    return $null
}

function Enable-SitePackages([string]$Directory) {
    $pth = Get-ChildItem -LiteralPath $Directory -Filter 'python*._pth' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $pth) { throw "Cannot find python*._pth under $Directory" }

    $kept = @()
    foreach ($line in @(Get-Content -LiteralPath $pth.FullName)) {
        if ($line -match '^\s*#?\s*import\s+site\s*$') { continue }
        if ($line -match 'site-packages') { continue }
        $kept += $line
    }
    $kept += 'Lib\site-packages'
    $kept += 'import site'
    Set-Content -LiteralPath $pth.FullName -Value $kept -Encoding Ascii
}

function Install-Pip([string]$Python) {
    $check = Invoke-NativeCapture $Python @('-m', 'pip', '--version')
    if ($check.Code -eq 0) { return }

    Write-Step 'Installing pip into the private runtime.'
    $getPip = Join-Path $RuntimeDir 'get-pip.py'
    try {
        Invoke-Download -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getPip
    } catch {
        throw "Failed to download get-pip.py: $($_.Exception.Message)`n  URL: https://bootstrap.pypa.io/get-pip.py"
    }
    $arguments = @($getPip, '--no-warn-script-location', '--disable-pip-version-check')
    $proxy = Get-Proxy
    if ($proxy) { $arguments += @('--proxy', $proxy) }
    $size = 0
    if (Test-Path -LiteralPath $getPip) { $size = (Get-Item -LiteralPath $getPip).Length }
    Write-Step "get-pip.py: $size bytes; running it."
    $code = Invoke-NativeStreaming $Python $arguments
    if ($code -ne 0) {
        throw "get-pip.py failed with exit code $code (script $size bytes). Check the network (or HTTPS_PROXY) and retry."
    }
    Remove-Item -LiteralPath $getPip -Force -ErrorAction SilentlyContinue
}

function Install-PortablePython {
    Write-Step "No Python 3.$MinMinor+ found; downloading the official embeddable Python $PinnedVersion (one time, about 11 MB)."
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    if (Test-Path -LiteralPath $PythonDir) {
        Remove-Item -LiteralPath $PythonDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Force -Path $PythonDir | Out-Null

    # Drop a partial download left by an interrupted run.
    Get-ChildItem -LiteralPath $RuntimeDir -Filter 'python-*-embed-amd64.zip' -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }

    $archive = Join-Path $RuntimeDir "python-$PinnedVersion-embed-amd64.zip"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    try {
        Invoke-Download -Uri $EmbedUrl -OutFile $archive
    } catch {
        $reason = $_.Exception.Message
        throw "Failed to download the embeddable Python: $reason`n  URL: $EmbedUrl`n  You can also install Python 3.$MinMinor+ manually or set NINFER_LAUNCHER_PYTHON."
    }

    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLower()
    if ($PinnedVersion -eq '3.11.9' -and -not $env:NINFER_LAUNCHER_PYTHON_URL -and $actual -ne $PinnedSha256) {
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
        throw 'SHA256 mismatch for the downloaded embeddable Python; the file was deleted.'
    }

    Write-Step 'Extracting the embeddable Python.'
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    $code = 1
    if ($tar) { $code = Invoke-Native $tar.Source @('-xf', $archive, '-C', $PythonDir) }
    if ($code -ne 0) {
        Expand-Archive -Path $archive -DestinationPath $PythonDir -Force
    }
    # tar can briefly keep the archive open; retry removal a few times.
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        if (-not (Test-Path -LiteralPath $archive)) { break }
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $archive) { Start-Sleep -Milliseconds 300 }
    }

    Enable-SitePackages $PythonDir
    Install-Pip $PortableExe

    if (-not (Test-PythonUsable $PortableExe)) {
        throw "The embeddable Python is not usable: $PortableExe"
    }
    return $PortableExe
}

function Ensure-Dependencies([string]$Python) {
    $check = Invoke-NativeCapture $Python @('-c', 'import fastapi, uvicorn, httpx')
    if ($check.Code -eq 0) { return }

    $pip = Invoke-NativeCapture $Python @('-m', 'pip', '--version')
    if ($pip.Code -ne 0) {
        Write-Step 'pip is missing; trying ensurepip.'
        Invoke-Native $Python @('-m', 'ensurepip', '--upgrade') | Out-Null
        $pip = Invoke-NativeCapture $Python @('-m', 'pip', '--version')
        if ($pip.Code -ne 0) {
            throw "pip is not available for $Python. Install pip or set NINFER_LAUNCHER_PYTHON to another interpreter."
        }
    }

    Write-Step 'Installing launcher dependencies (fastapi, uvicorn, httpx).'
    $arguments = @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', $Requirements)
    $proxy = Get-Proxy
    if ($proxy) { $arguments += @('--proxy', $proxy) }
    $code = Invoke-NativeStreaming $Python $arguments
    if ($code -ne 0) {
        throw "Dependency installation failed. Check the network (or HTTPS_PROXY) and retry, or run manually:`n  $Python -m pip install -r $Requirements"
    }
    $check = Invoke-NativeCapture $Python @('-c', 'import fastapi, uvicorn, httpx')
    if ($check.Code -ne 0) { throw 'Dependencies still cannot be imported after installation.' }
}

# ------------------------------------------------------------------ main

$python = Find-Python
if (-not $python) {
    $python = Install-PortablePython
} else {
    Write-Step "Using Python: $python"
}
Ensure-Dependencies $python

if (-not (Test-Path -LiteralPath $LauncherScript)) {
    throw "Launcher entry point not found: $LauncherScript"
}

& $python $LauncherScript @LauncherArgs
exit $LASTEXITCODE