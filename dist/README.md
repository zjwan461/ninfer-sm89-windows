# Release bundles

## sm_89 / sm_120a native Windows bundle (this fork)

This fork's attention wave geometry is a compile-time constant (`NINFER_TARGET_SM_COUNT`), so the
RTX 4090 (128 SMs), the RTX 4080 SUPER (80 SMs) and the RTX 5060 Ti (36 SMs) need their own
binaries. One archive carries one architecture: the three profiles together exceed GitHub's 2 GiB
release-asset cap. Build them first, then run the packager once per architecture from the
repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package-release-v061-sm89.ps1 -Cards rtx4090,rtx4080s
powershell -ExecutionPolicy Bypass -File scripts\package-release-v061-sm89.ps1 -Cards rtx5060ti
```

The first run creates `dist/ninfer-sm89-windows-x64-<version>/` with `sm128-rtx4090/` and
`sm80-rtx4080s/`; the second creates `dist/ninfer-sm120-windows-x64-<version>/` with
`sm36-rtx5060ti/`. Each folder holds the executables, the vcpkg DLLs, its own `VERSION` and the
launcher pair for that profile (`start-bonsai-rtx<card>.bat` + `start-qwen38-rtx<card>.bat`; the
5060 Ti folder carries the Bonsai launcher only, because the Qwen3.8-27B weights alone are
15.9 GiB). Each run writes `<product>.zip` and `dist/SHA256SUMS-v<version>-<arch>.txt` next to it.

A single invocation cannot mix architectures: `-Cards rtx4090,rtx5060ti` is rejected. The
`sm89` product name is historical and means the two `sm_89` cards; the `sm_120a` card ships as
`sm120`. `.github/workflows/windows-release.yml` builds all three profiles on a Windows runner and
runs the packager once per architecture, on a `v*` tag or on demand.

Before packaging, each profile's `NINFER_TARGET_SM_COUNT` and target architecture are asserted
against its build directory's `compile_commands.json`, so a mislabeled archive cannot be produced
silently. The script is pack-only: it fails when a build directory or product is missing.

Generated binaries and archives are ignored by Git because GitHub source repositories should not
contain build products. Upload the `.zip` and versioned checksum files as GitHub Release assets.
The packaging guide itself is tracked.

Model artifacts are not included. Download them from the repositories linked in the project README
and verify the published SHA-256 separately.

The Windows bundle includes its FFmpeg/curl/zlib DLLs and requires NVIDIA driver 595 or newer and
the Microsoft Visual C++ 2015-2022 x64 runtime.

The Windows script is the delivered one; `scripts/package-release-v061-sm89.sh` is its Bash
counterpart for the Linux `sm_89` builds (the repository requires a `.sh` twin for every `.ps1`).
