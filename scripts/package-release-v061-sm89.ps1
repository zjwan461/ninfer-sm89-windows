$ErrorActionPreference = 'Stop'

# Fork-local release packager for the native Windows sm_89 port.
# Mirrors scripts\package-release-v060.ps1, but targets this fork's Ninja build trees instead of
# the upstream RTX 3090 MSVC trees (build-sm86-*), and bundles both compile-time SM profiles.
#
# This fork is an sm_89 port: the attention wave geometry is a compile-time constant
# (NINFER_TARGET_SM_COUNT, see CMakeLists.txt), so the RTX 4090 (128 SMs) and the RTX 4080 SUPER
# (80 SMs) need their own binaries. The two profiles differ only in that constant, so they ship as
# two subdirectories of one archive instead of two look-alike downloads.
#
# Pack-only, exactly like v040/v050/v060: it assembles an existing verified build and throws when a
# build directory or product is missing. Build both profiles first (see
# ninfer-windows-build-manual.md for the 80-SM tree), then run from the repository root:
#
#   powershell -ExecutionPolicy Bypass -File scripts\package-release-v061-sm89.ps1
#
# A Bash counterpart (package-release-v061-sm89.sh) packages the Linux sm_89 builds so the
# scripts\ pairs stay symmetric (scripts\check-linux-scripts.sh enforces that).

$ReleaseTag = '0.6.1'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DistRoot = Join-Path $RepoRoot 'dist'
$ProductName = "ninfer-sm89-windows-x64-$ReleaseTag"
$ProductRoot = Join-Path $DistRoot $ProductName
$ArchivePath = Join-Path $DistRoot "$ProductName.zip"
$ChecksumPath = Join-Path $DistRoot "SHA256SUMS-v$ReleaseTag-sm89.txt"

# Per card: the subdirectory it occupies, the SM profile to verify, and the build directories to
# look for (newest name first).
$Variants = @(
    @{ Card = 'rtx4090'; Subdir = 'sm128-rtx4090'; SmCount = 128; BuildDirs = @('build-4090', 'build') },
    @{ Card = 'rtx4080s'; Subdir = 'sm80-rtx4080s'; SmCount = 80; BuildDirs = @('build-4080s') }
)

# Ninja (single config) puts the products directly in apps\; the apps\Release\ candidates keep the
# script working under a multi-config generator. ninfer-perplexity.exe replaces the 3090 bundle's
# ninfer_bench.exe because NINFER_BUILD_BENCHMARKS defaults to OFF.
$Products = @(
    @{ Name = 'ninfer.exe'; Candidates = @('apps\ninfer.exe', 'apps\Release\ninfer.exe') },
    @{ Name = 'ninfer-serve.exe'; Candidates = @('apps\ninfer-serve.exe', 'apps\Release\ninfer-serve.exe') },
    @{ Name = 'ninfer-perplexity.exe'; Candidates = @('apps\ninfer-perplexity.exe', 'apps\Release\ninfer-perplexity.exe') }
)

# The server launchers resolve `%~dp0ninfer-serve.exe`, so each SM profile needs its own copy
# inside its subdirectory; the shared docs sit at the archive root.
$Launchers = @('start-bonsai-server.bat', 'start-qwen38-server.bat')

function Resolve-BuildRoot {
    param([System.Collections.IDictionary]$Variant)
    foreach ($candidate in $Variant.BuildDirs) {
        $path = Join-Path $RepoRoot $candidate
        if (Test-Path -LiteralPath $path) {
            Write-Host "SM $($Variant.SmCount) ($($Variant.Card)): using build directory '$candidate'"
            return $path
        }
    }
    throw "No build directory for SM $($Variant.SmCount) ($($Variant.Card)); tried: $($Variant.BuildDirs -join ', ')"
}

function Find-Product {
    param([string]$BuildRoot, [System.Collections.IDictionary]$Product)
    foreach ($candidate in $Product.Candidates) {
        $path = Join-Path $BuildRoot $candidate
        if (Test-Path -LiteralPath $path) { return $path }
    }
    throw "Missing release product: $($Product.Candidates -join ' or ') under $BuildRoot"
}

# Guard against shipping a mislabeled dual-SM bundle: the two profiles differ only by this
# constant, so a wrong pairing is invisible after the fact. compile_commands.json always exists
# because CMAKE_EXPORT_COMPILE_COMMANDS is ON.
function Assert-SmProfile {
    param([string]$BuildRoot, [int]$SmCount, [string]$Card)
    $compileCommands = Join-Path $BuildRoot 'compile_commands.json'
    if (-not (Test-Path -LiteralPath $compileCommands)) {
        Write-Warning "No compile_commands.json under $BuildRoot; skipping the SM profile check for $Card"
        return
    }
    if (-not (Select-String -LiteralPath $compileCommands -Pattern "NINFER_TARGET_SM_COUNT=$SmCount" -SimpleMatch -Quiet)) {
        throw "Build $BuildRoot does not carry NINFER_TARGET_SM_COUNT=$SmCount; refusing to label it $Card"
    }
}

# --- Pre-flight --------------------------------------------------------------
# Resolve and validate every profile before touching dist/, so a missing build or a wrong SM
# profile leaves dist/ untouched instead of a half-written bundle directory.
$ResolvedRoots = @{}
foreach ($variant in $Variants) {
    $buildRoot = Resolve-BuildRoot -Variant $variant
    Assert-SmProfile -BuildRoot $buildRoot -SmCount $variant.SmCount -Card $variant.Card
    foreach ($product in $Products) {
        [void](Find-Product -BuildRoot $buildRoot -Product $product)
    }
    $ResolvedRoots[$variant.Subdir] = $buildRoot
}

# --- Pack --------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
$resolvedDist = (Resolve-Path -LiteralPath $DistRoot).Path
$resolvedProductParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $ProductRoot))
if ($resolvedProductParent -ne $resolvedDist -or (Split-Path -Leaf $ProductRoot) -ne $ProductName) {
    throw "Refusing to package outside the expected dist directory: $ProductRoot"
}
if (Test-Path -LiteralPath $ProductRoot) {
    Remove-Item -LiteralPath $ProductRoot -Recurse -Force
}
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
New-Item -ItemType Directory -Path $ProductRoot | Out-Null

foreach ($variant in $Variants) {
    $buildRoot = $ResolvedRoots[$variant.Subdir]
    $variantRoot = Join-Path $ProductRoot $variant.Subdir
    New-Item -ItemType Directory -Path $variantRoot | Out-Null

    foreach ($product in $Products) {
        $source = Find-Product -BuildRoot $buildRoot -Product $product
        Copy-Item -LiteralPath $source -Destination (Join-Path $variantRoot $product.Name)
    }

    # vcpkg's app-local deployment drops the runtime DLLs next to the executables; fall back to the
    # manifest install tree if a build skipped that step.
    $dllSource = Join-Path $buildRoot 'apps'
    if (-not (Test-Path -LiteralPath $dllSource)) { $dllSource = Join-Path $buildRoot 'apps\Release' }
    if (Test-Path -LiteralPath $dllSource) {
        Get-ChildItem -LiteralPath $dllSource -Filter '*.dll' | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $variantRoot
        }
    }

    # Repositories without the fork launchers still package cleanly.
    foreach ($launcher in $Launchers) {
        $source = Join-Path $RepoRoot $launcher
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination $variantRoot
        }
    }

    # VERSION in the repository reports the upstream RTX 3090 tag; stamp each profile with the tag
    # and card it actually is.
    Set-Content -LiteralPath (Join-Path $variantRoot 'VERSION') -Value "$ReleaseTag-$($variant.Card)" -Encoding ascii
}

# --- Shared root files -------------------------------------------------------
Set-Content -LiteralPath (Join-Path $ProductRoot 'VERSION') -Value "$ReleaseTag-sm89" -Encoding ascii
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'WINDOWS_PORT.md') -Destination $ProductRoot
foreach ($doc in @('ninfer-windows-build-manual.md')) {
    $source = Join-Path $RepoRoot $doc
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination $ProductRoot
    }
}

$readme = @'
# NInfer sm_89 - native Windows bundle (__TAG__)

This archive carries both compile-time SM profiles of the sm_89 port. Pick the folder that matches
your card, then run its `start-bonsai-server.bat` or `start-qwen38-server.bat` from inside that
folder (they resolve `%~dp0ninfer-serve.exe`), passing the model path as the first argument:

    start-bonsai-server.bat D:\models\bonsai2_27b_vl_mtp_q4q5.ninfer

A directory containing a single `.ninfer` file works too. Without an argument the launcher falls
back to `NINFER_MODEL`, then `NINFER_MODEL_DIR`, then a default file name under `models\`.

| Folder | Cards | NINFER_TARGET_SM_COUNT |
| --- | --- | --- |
| `sm128-rtx4090/` | RTX 4090 | 128 |
| `sm80-rtx4080s/` | RTX 4080 SUPER | 80 |

Both folders contain `ninfer.exe`, `ninfer-serve.exe`, `ninfer-perplexity.exe` and the vcpkg
runtime DLLs. The profiles differ only in the attention wave geometry constant; 128 matches the
RTX 4090 the project is tuned on. Either binary runs on either card, but the matching profile
avoids the attention tail-wave tax.

- `WINDOWS_PORT.md` - what this native Windows port changes and the verified performance.
- `ninfer-windows-build-manual.md` - how the Windows sm_89 build is configured (128/80 SM).
- `SHA256SUMS.txt` - hashes of every file in this bundle; the archive hash is in
  `SHA256SUMS-v__TAG__-sm89.txt` next to the download.

Model artifacts are not included. Download the `.ninfer` artifact for your model from the
repositories linked in the project README, and verify its published SHA-256 separately.
'@
$readme.Replace('__TAG__', $ReleaseTag) |
    Set-Content -LiteralPath (Join-Path $ProductRoot 'README.md') -Encoding ascii

# --- Hashes and archive ------------------------------------------------------
$innerHashes = Get-ChildItem -LiteralPath $ProductRoot -Recurse -File |
    Where-Object { $_.Name -ne 'SHA256SUMS.txt' } |
    Sort-Object FullName |
    ForEach-Object {
        $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
        $relative = $_.FullName.Substring($ProductRoot.Length + 1).Replace('\', '/')
        "$($hash.Hash.ToLowerInvariant())  $relative"
    }
$innerHashes | Set-Content -LiteralPath (Join-Path $ProductRoot 'SHA256SUMS.txt') -Encoding ascii

Compress-Archive -LiteralPath $ProductRoot -DestinationPath $ArchivePath -CompressionLevel Optimal
$archiveHash = Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256
"$($archiveHash.Hash.ToLowerInvariant())  $(Split-Path -Leaf $ArchivePath)" |
    Set-Content -LiteralPath $ChecksumPath -Encoding ascii

Get-Item -LiteralPath $ArchivePath, $ChecksumPath |
    Select-Object Name, @{ Name = 'SizeMB'; Expression = { [math]::Round($_.Length / 1MB, 2) } }