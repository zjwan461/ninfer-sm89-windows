param(
    # Optional subset of the two SM profiles to package, by card name. Packaging only the
    # available build (for example on an RTX 4080 SUPER machine without a 128-SM tree) keeps the
    # pack-only contract honest instead of forcing a throw:
    #   powershell -ExecutionPolicy Bypass -File scripts\package-release-v061-sm89.ps1 -Cards rtx4080s
    [ValidateSet('rtx4090', 'rtx4080s', 'rtx5060ti')]
    [string[]]$Cards = @('rtx4090', 'rtx4080s', 'rtx5060ti')
)

$ErrorActionPreference = 'Stop'

# Fork-local release packager for the native Windows port.
# Targets this fork's Ninja build trees (build/, build-4080s/, build-5060ti/) and bundles the three
# qualified compile-time profiles: RTX 4090 and RTX 4080 SUPER (sm_89), RTX 5060 Ti (sm_120a).
# The archive and product name keep the historical `sm89` token so existing links stay valid; the
# RTX 5060 Ti folder inside it is sm_120a.
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
# To ship only the profiles whose build trees exist, restrict the set with -Cards, for example on a
# machine that only carries the 80-SM tree:
#
#   powershell -ExecutionPolicy Bypass -File scripts\package-release-v061-sm89.ps1 -Cards rtx4080s
#
# A Bash counterpart (package-release-v061-sm89.sh) packages the Linux sm_89 builds so the
# scripts\ pairs stay symmetric (scripts\check-linux-scripts.sh enforces that).

$ReleaseTag = '0.7.0'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DistRoot = Join-Path $RepoRoot 'dist'
$ProductName = "ninfer-sm89-windows-x64-$ReleaseTag"
$ProductRoot = Join-Path $DistRoot $ProductName
$ArchivePath = Join-Path $DistRoot "$ProductName.zip"
$ChecksumPath = Join-Path $DistRoot "SHA256SUMS-v$ReleaseTag-sm89.txt"

# Per card: the subdirectory it occupies, the SM profile to verify, the build directories to look
# for (newest name first), and the launcher pair that profile ships. The launchers resolve
# `%~dp0ninfer-serve.exe`, so each profile carries its own pair inside its subdirectory; they
# differ only in the prefill chunk the matching wave geometry wants.
$AllVariants = @(
    @{ Card = 'rtx4090'; Subdir = 'sm128-rtx4090'; Arch = '89'; SmCount = 128; BuildDirs = @('build-4090', 'build');
       Models = 'Bonsai, Qwen3.8'; Note = '';
       Launchers = @('start-bonsai-rtx4090.bat', 'start-qwen38-rtx4090.bat') },
    @{ Card = 'rtx4080s'; Subdir = 'sm80-rtx4080s'; Arch = '89'; SmCount = 80; BuildDirs = @('build-4080s');
       Models = 'Bonsai, Qwen3.8'; Note = '';
       Launchers = @('start-bonsai-rtx4080s.bat', 'start-qwen38-rtx4080s.bat') },
    @{ Card = 'rtx5060ti'; Subdir = 'sm36-rtx5060ti'; Arch = '120a'; SmCount = 36; BuildDirs = @('build-5060ti');
       Models = 'Bonsai'; Note = 'Bonsai only: the Qwen3.8-27B weights do not fit in 16 GB, so this folder ships the Bonsai launcher alone.'
       Launchers = @('start-bonsai-rtx5060ti.bat') }
)
$Variants = @($AllVariants | Where-Object { $Cards -contains $_.Card })

# Ninja (single config) puts the products directly in apps\; the apps\Release\ candidates keep the
# script working under a multi-config generator. ninfer-perplexity.exe replaces the 3090 bundle's
# ninfer_bench.exe because NINFER_BUILD_BENCHMARKS defaults to OFF.
$Products = @(
    @{ Name = 'ninfer.exe'; Candidates = @('apps\ninfer.exe', 'apps\Release\ninfer.exe') },
    @{ Name = 'ninfer-serve.exe'; Candidates = @('apps\ninfer-serve.exe', 'apps\Release\ninfer-serve.exe') },
    @{ Name = 'ninfer-perplexity.exe'; Candidates = @('apps\ninfer-perplexity.exe', 'apps\Release\ninfer-perplexity.exe') }
)


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

# Guard against shipping a mislabeled bundle: the profiles differ by target architecture and by this
# constant, so a wrong pairing is invisible after the fact. compile_commands.json always exists
# because CMAKE_EXPORT_COMPILE_COMMANDS is ON.
function Assert-Profile {
    param([string]$BuildRoot, [string]$Arch, [int]$SmCount, [string]$Card)
    $compileCommands = Join-Path $BuildRoot 'compile_commands.json'
    if (-not (Test-Path -LiteralPath $compileCommands)) {
        Write-Warning "No compile_commands.json under $BuildRoot; skipping the profile check for $Card"
        return
    }
    if (-not (Select-String -LiteralPath $compileCommands -Pattern "NINFER_TARGET_SM_COUNT=$SmCount" -SimpleMatch -Quiet)) {
        throw "Build $BuildRoot does not carry NINFER_TARGET_SM_COUNT=$SmCount; refusing to label it $Card"
    }
    if (-not (Select-String -LiteralPath $compileCommands -Pattern "compute_$Arch" -SimpleMatch -Quiet)) {
        throw "Build $BuildRoot was not compiled for sm_$Arch; refusing to label it $Card"
    }
}

# --- Pre-flight --------------------------------------------------------------
# Resolve and validate every profile before touching dist/, so a missing build or a wrong SM
# profile leaves dist/ untouched instead of a half-written bundle directory.
$ResolvedRoots = @{}
foreach ($variant in $Variants) {
    $buildRoot = Resolve-BuildRoot -Variant $variant
    Assert-Profile -BuildRoot $buildRoot -Arch $variant.Arch -SmCount $variant.SmCount -Card $variant.Card
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
    foreach ($launcher in $variant.Launchers) {
        $source = Join-Path $RepoRoot $launcher
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination $variantRoot
        }
    }

    # The repository VERSION is the product-level tag; stamp each profile with the tag and the card
    # it actually is, so the launcher badge distinguishes the directories of one archive.
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

# The archive may carry any subset of the three profiles (see -Cards), so the folder table, the
# profile list and the example launcher line are generated from the selected variants.
$profileTable = ($Variants | ForEach-Object {
    '| `' + $_.Subdir + '/` | ' + $_.Card + ' | ' + $_.SmCount + ' | ' + $_.Models + ' |'
}) -join "`r`n"
$noteLines = ($Variants | Where-Object { $_.Note } | ForEach-Object {
    '- `' + $_.Subdir + '/`: ' + $_.Note
}) -join "`r`n"
if (-not $noteLines) {
    $noteLines = 'Every profile in this archive runs both models.'
}
$exampleLauncher = (Join-Path $Variants[0].Subdir $Variants[0].Launchers[0])
$profileList = ($Variants | ForEach-Object {
    $_.Card + ' (`sm_' + $_.Arch + '`, ' + $_.SmCount + ' SMs)'
}) -join ', '
if ($Variants.Count -eq 1) {
    $profileSentence = 'This archive carries the compile-time profile for ' + $profileList +
        '. Run the launcher for your model from inside that folder'
    $foldersWord = 'The folder'
    $containWord = 'contains'
} else {
    $profileSentence = 'This archive carries ' + $Variants.Count + ' compile-time profiles: ' +
        $profileList + '. Pick the folder that matches your card, then run the launcher for your model from inside that folder'
    $foldersWord = 'The folders'
    $containWord = 'contain'
}
$readme = @'
# NInfer - native Windows bundle (__TAG__)

__PROFILE_SENTENCE__ (each launcher resolves
`%~dp0ninfer-serve.exe`), passing the model path as the first argument:

    __EXAMPLE_LAUNCHER__ D:\models\bonsai2_27b_vl_mtp_q4q5.ninfer

Each profile folder ships the launchers for the models its card fits; the pairs differ only in
`--prefill-chunk`.

__NOTES__

A directory containing a single `.ninfer` file works too. Without an argument the launcher falls
back to `NINFER_MODEL`, then `NINFER_MODEL_DIR`, then a default file name under `models\`.

| Folder | Card | NINFER_TARGET_SM_COUNT | Models |
| --- | --- | --- | --- |
__PROFILE_TABLE__

__FOLDERS_WORD__ __CONTAIN_WORD__ `ninfer.exe`, `ninfer-serve.exe`, `ninfer-perplexity.exe` and the vcpkg
runtime DLLs. The profiles differ in the target architecture (`sm_89` against `sm_120a`) and in the
attention wave geometry constant. A binary only runs on the architecture it was built for; within
one architecture, the matching SM profile avoids the attention tail-wave tax.

- `WINDOWS_PORT.md` - what this native Windows port changes and the verified performance.
- `ninfer-windows-build-manual.md` - how the Windows build is configured (128/80/36 SM, `sm_89`/`sm_120a`).
- `SHA256SUMS.txt` - hashes of every file in this bundle; the archive hash is in
  `SHA256SUMS-v__TAG__-sm89.txt` next to the download.

Model artifacts are not included. Download the `.ninfer` artifact for your model from the
repositories linked in the project README, and verify its published SHA-256 separately.
'@
$readme.Replace('__TAG__', $ReleaseTag).
    Replace('__PROFILE_SENTENCE__', $profileSentence).
    Replace('__EXAMPLE_LAUNCHER__', $exampleLauncher).
    Replace('__PROFILE_TABLE__', $profileTable).
    Replace('__NOTES__', $noteLines).
    Replace('__FOLDERS_WORD__', $foldersWord).
    Replace('__CONTAIN_WORD__', $containWord) |
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