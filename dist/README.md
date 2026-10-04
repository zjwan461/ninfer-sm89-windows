# Release bundles

## RTX 3090 (upstream `sm_86` trees)

For v0.4.0, run the Windows-only packaging script from the repository root after the verified
native build exists:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package-release-v040.ps1
```

It creates a versioned directory and archive under `dist/`:

- `ninfer-rtx3090-windows-x64-*`: native Windows CLI, server, benchmark, and vcpkg DLLs;
- `SHA256SUMS-v0.4.0.txt`: archive hash for release verification.

Generated binaries and archives are ignored by Git because GitHub source repositories should not
contain build products. Upload the `.zip` and versioned checksum file as GitHub Release assets.
The packaging guide itself is tracked.

Model artifacts are not included. Download either the 16.29 GiB `qwen3_6_27b.ninfer` or 20.84 GiB
`qwen3_6_35b_a3b.ninfer` artifact from the repositories linked in the project README and verify its
published SHA-256 separately. The compact 35B artifact is text-only; leave `--vision` disabled.

The Windows bundle includes its FFmpeg/curl/zlib DLLs and requires the NVIDIA driver and Microsoft
Visual C++ 2022 runtime.

## sm_89 native Windows bundle (this fork)

This fork is an `sm_89` port whose attention wave geometry is a compile-time constant
(`NINFER_TARGET_SM_COUNT`), so the RTX 4090 (128 SMs) and the RTX 4080 SUPER (80 SMs) need their own
binaries. Both profiles ship in one archive. Build them first, then run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package-release-v061-sm89.ps1
```

It creates `dist/ninfer-sm89-windows-x64-<version>/` containing the `sm128-rtx4090/` and
`sm80-rtx4080s/` subdirectories (each with the executables, vcpkg DLLs, its own `VERSION`, and the
`start-*-server.bat` launchers) plus the shared docs, archives it as
`ninfer-sm89-windows-x64-<version>.zip`, and writes `dist/SHA256SUMS-v<version>-sm89.txt`.

Before packaging, each profile's `NINFER_TARGET_SM_COUNT` is asserted against its build directory's
`compile_commands.json`, so a mislabeled dual-SM archive cannot be produced silently. The script is
pack-only, like the v0x0 ones above: it fails when a build directory or product is missing.

The Windows script is the delivered one; `scripts/package-release-v061-sm89.sh` is its Bash
counterpart for the Linux `sm_89` builds (the repository requires a `.sh` twin for every `.ps1`).
