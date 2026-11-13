# Release process

This is the single authority for cutting a Windows release of this fork. The workflow that
implements it is [.github/workflows/windows-release.yml](../../.github/workflows/windows-release.yml);
the packaging itself is [scripts/package-release-v061-sm89.ps1](../../scripts/package-release-v061-sm89.ps1).

## What a release contains

One archive per target architecture, because the three compile-time profiles together are about
2.3 GiB and GitHub caps a release asset at 2 GiB:

| Archive | Profiles |
|---|---|
| `ninfer-sm89-windows-x64-<version>.zip` | `sm128-rtx4090/`, `sm80-rtx4080s/` |
| `ninfer-sm120-windows-x64-<version>.zip` | `sm36-rtx5060ti/` (Bonsai only) |

Plus one `SHA256SUMS-v<version>-<arch>.txt` per archive. The `sm89` token in the first name is
historical and means the two `sm_89` cards; it is kept so published links stay valid.

## Cutting a release

1. **Land the change on `main`** and let the [Checks](../../.github/workflows/checks.yml) workflow go
   green. It covers the Python tooling, the launcher tests, script parsing and documentation
   links -- everything the release build does not exercise.
2. **Validate the three-profile build** before tagging. Run *Windows release* from the Actions tab
   on `main` with `package` enabled. It builds the three presets, runs the packager twice and
   uploads both archives as a build artifact (kept 14 days). Nothing is published yet.
3. **Tag the commit the validation ran on.** The tag rebuilds from that exact commit, creates the
   release and attaches both archives plus their checksum files:

   ```bash
   git tag -a v0.8.0 -m "..." <commit>
   git push origin v0.8.0
   ```

4. **Verify the assets** before announcing anything: both archives are present, their sizes are
   plausible (about 1.5 GiB and 0.7 GiB), and each `SHA256SUMS` line matches the archive digest
   GitHub reports for the asset.
5. **Smoke-test on a real card.** The hosted runner has no GPU: the release proves that the tree
   configures, compiles, links and packages, not that it generates tokens. Unpack the archive on
   the target card and run one request:

   ```bat
   sm80-rtx4080s\start-qwen38-rtx4080s.bat D:\models\qwen3_8_27b_a8.ninfer
   ```

   A tok/s reading on `http://127.0.0.1:8080/monitor` is the acceptance signal.

## Constraints that have already cost a run each

- **CUDA action version.** `Jimver/cuda-toolkit` only knows the CUDA versions in its own link
  table (`src/links/windows-links.ts` at the pinned tag). v0.2.19 predates 12.8.0 and fails with
  `Error: Version not available`; the workflow pins v0.2.36 with 13.3.1. Check the table before
  changing either number.
- **`cl.exe` is not on PATH.** A hosted Windows runner does not start in a developer prompt, and
  the image also carries MinGW, which CMake will otherwise select. The job imports the MSVC
  environment and pins `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` and `CMAKE_CUDA_HOST_COMPILER`
  to `cl`.
- **2 GiB per asset.** `Compress-Archive` has its own 2 GB ceiling, so the packager writes with
  `ZipFile`; the archives are split per architecture instead of shipping one bundle.
- **Wall clock.** The job takes about two hours on a 4-core runner: 23 minutes the first time
  vcpkg builds ffmpeg (cached afterwards), then three full builds. The job timeout is 300
  minutes.
- **Environment names are case-insensitive.** A workflow that declares the same variable twice
  with different case (`NO_PROXY` and `no_proxy`) is rejected as a whole: the run completes as a
  failure in the same second with zero jobs and no log, and the run list shows the workflow's
  file path instead of its name. Declare one spelling and let the platform case-fold it.

## When it fails

Re-run the failed job from the Actions tab; the vcpkg binary cache makes a retry much cheaper
than the first attempt. If the failure is in the build itself, fix it on `main` and start again
from step 2 -- do not move a tag that has already been published. If a release shipped with a
wrong artifact, delete the release and the tag, fix, and re-tag (a republished tag is acceptable
only while nothing has consumed it).
