# NInfer-sm89 for Windows v0.7.0

This release turns the fork into three qualified consumer-card profiles of one `sm_89`/`sm_120a`
source tree, and adds the DSpine draft backend. Model execution, artifact compatibility and the
serving protocol are unchanged.

## Supported cards

| Card | Architecture | `NINFER_TARGET_SM_COUNT` | Delivery |
|---|---|---:|---|
| RTX 4090 24 GB | `sm_89` | 128 | prebuilt, `sm128-rtx4090/` |
| RTX 4080 SUPER 16/32 GB | `sm_89` | 80 | prebuilt, `sm80-rtx4080s/` |
| RTX 5060 Ti 16 GB | `sm_120a` | 36 | prebuilt, `sm36-rtx5060ti/` (Bonsai only) |

Two archives, one per architecture: `ninfer-sm89-windows-x64-0.7.0.zip` carries the RTX 4090 and
RTX 4080 SUPER folders, `ninfer-sm120-windows-x64-0.7.0.zip` the RTX 5060 Ti one. They are split
because the three profiles together exceed GitHub's 2 GiB release-asset cap, and a binary only
runs on the architecture it was built for. The 16 GB card is qualified on Bonsai: the Qwen3.8-27B
weights alone are 15.9 GiB, so its folder ships one launcher.
The RTX 3090 (`sm_86`) port belongs to the upstream [Don-Chad/ninfer-3090] line; this tree accepts
only `89` and `120a` and does not build `sm_86`.

[Don-Chad/ninfer-3090]: https://github.com/Don-Chad/ninfer-3090

## Changes

- **The target SM count is a build option.** `NINFER_TARGET_SM_COUNT` selects the attention wave
  geometry at configure time; `CMAKE_CUDA_ARCHITECTURES` selects `89` or `120a`. The value is
  validated at configure time and again by a `static_assert` in the causal-attention geometry, and
  the release packager asserts it against each build's `compile_commands.json` so a mislabeled
  archive cannot be produced silently.
- **RTX 4080 SUPER (80 SM) profile.** Measured baseline in `NINFER_4080S_BASELINE.md`: 37.8 tok/s
  decode without speculation, 74.8-81.0 tok/s with MTP 3, 3.41k tok/s prefill, and `262144` context
  at two lanes inside 29.5 GB. The 80-SM constant makes no measurable difference on the same card
  compared with a 128-SM build, which is why the profile is a build option rather than a rewrite.
- **RTX 5060 Ti 16 GB (`sm_120a`, 36 SM) profile.** Measured baseline in
  `NINFER_5060TI_BASELINE.md`: 59.2 tok/s decode without speculation, 111.5 tok/s with MTP 2 at
  8192 context, and `131072` context at two lanes in 8.88 GiB.
- **DSpine draft backend** (`--spec dspine`): a five-layer draft with a 16-token block, an 8192-token
  sliding window, a per-layer code bridge and serial last-write refinement, re-scoring the
  sixteen full-vocabulary candidates with the target head. On the RTX 4080 SUPER with the int8
  artifact it accepts 10.60 tokens per round on a merge-sorted-function prompt (233 tok/s) at
  `--draft-tokens 15`. Mathematics, rounding points and the deliberate deviations from the
  reference implementation are in `docs/maintainer/dspine.md`.
- **Per-profile launchers and a local WebUI.** `start-bonsai-*.bat` and `start-qwen38-*.bat` exist
  for all three profiles, and `start-launcher.bat` configures and supervises `ninfer-serve.exe`
  from a browser form, shipping the launcher profiles as presets.
- **Windows release CI.** `.github/workflows/windows-release.yml` builds the three presets on a
  hosted Windows runner, packages one archive per architecture with the repository's own script,
  uploads both as build artifacts and attaches them to a `v*` tag's release. The runner has no GPU, so it proves the
  tree configures, compiles and links; kernel correctness still comes from `ctest` on a card.
- **One release packager.** `scripts/package-release-v061-sm89.ps1` (and its Bash counterpart)
  replaces the upstream v0.4.0-v0.6.0 packagers; the older scripts and the `sm_86` Linux guide are
  removed, together with the completed fork work plans that were still in the repository root.

## Verification

Every figure in this document was measured on the card named next to it, with the conditions
recorded in `NINFER_4080S_BASELINE.md`, `NINFER_5060TI_BASELINE.md` and the README tables. The C++
and CUDA test suite is unchanged: `ctest --test-dir build --output-on-failure` after a
`-DBUILD_TESTING=ON` configure.

## Limits

- One card per build and per process. A binary built for one architecture does not load on the
  other, and one built for the wrong SM count only loses wave utilization -- it stays correct.
- The Linux `Dockerfile` builds the 128-SM `sm_89` profile and is not validated by this fork.
- The `sm89` archive name is historical and now means the two `sm_89` cards; the `sm_120a` card
  ships in `ninfer-sm120-windows-x64-<version>.zip`.
