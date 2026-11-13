# Release bundles

## sm_89 / sm_120a native Windows bundle (this fork)

This fork's attention wave geometry is a compile-time constant (`NINFER_TARGET_SM_COUNT`), so the
RTX 4090 (128 SMs) and the RTX 4080 SUPER (80 SMs) need their own binaries. Both profiles ship in
one archive. Build them first, then run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package-release-v061-sm89.ps1
```

It creates `dist/ninfer-sm89-windows-x64-<version>/` containing the `sm128-rtx4090/` and
`sm80-rtx4080s/` subdirectories (each with the executables, vcpkg DLLs, its own `VERSION`, and the
launcher pair for that profile: `start-bonsai-rtx4090.bat` + `start-qwen38-rtx4090.bat`, or the
`-rtx4080s` pair) plus the shared docs, archives it as
`ninfer-sm89-windows-x64-<version>.zip`, and writes `dist/SHA256SUMS-v<version>-sm89.txt`. The two
pairs differ only in `--prefill-chunk`, which is the one knob tied to the attention wave geometry.

The RTX 5060 Ti 16 GB is `sm_120a` (36 SMs) and has no prebuilt archive: an `sm_89` binary cannot
load on it. Build that profile from source with `-DCMAKE_CUDA_ARCHITECTURES=120a
-DNINFER_TARGET_SM_COUNT=36`; there is no `sm_120a` packager yet.

Before packaging, each profile's `NINFER_TARGET_SM_COUNT` is asserted against its build directory's
`compile_commands.json`, so a mislabeled dual-SM archive cannot be produced silently. The script is
pack-only: it fails when a build directory or product is missing.

Generated binaries and archives are ignored by Git because GitHub source repositories should not
contain build products. Upload the `.zip` and versioned checksum file as GitHub Release assets. The
packaging guide itself is tracked.

Model artifacts are not included. Download them from the repositories linked in the project README
and verify the published SHA-256 separately.

The Windows bundle includes its FFmpeg/curl/zlib DLLs and requires NVIDIA driver 595 or newer and
the Microsoft Visual C++ 2015-2022 x64 runtime.

The Windows script is the delivered one; `scripts/package-release-v061-sm89.sh` is its Bash
counterpart for the Linux `sm_89` builds (the repository requires a `.sh` twin for every `.ps1`).
