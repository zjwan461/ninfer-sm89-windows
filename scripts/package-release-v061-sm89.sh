#!/usr/bin/env bash
# Bash counterpart of scripts/package-release-v061-sm89.ps1.
#
# This fork is delivered on Windows, so the Windows script is the one that ships. This script
# packages the equivalent sm_89 Linux builds (the upstream fork's native platform) with the same
# two-profile layout, and exists so the scripts/ .bat/.ps1 pairs stay symmetric --
# scripts/check-linux-scripts.sh fails when a .ps1 has no executable .sh counterpart.
#
# Pack-only, like v040/v050/v060: it assembles existing verified builds and fails when a build
# directory or product is missing.
set -euo pipefail

release_tag='0.6.1'
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dist_root="$repo_root/dist"
product_name="ninfer-sm89-linux-x64-$release_tag"
product_root="$dist_root/$product_name"
archive_name="$product_name.tar.gz"
archive_path="$dist_root/$archive_name"
checksum_path="$dist_root/SHA256SUMS-v$release_tag-sm89-linux.txt"

# subdir | sm count | comma-separated candidate build directories (newest name first)
variants=(
  'sm128-rtx4090|128|build-linux-sm89-128,build-linux-sm89'
  'sm80-rtx4080s|80|build-linux-sm89-80'
)
products=(ninfer ninfer-serve ninfer-perplexity)

mkdir -p -- "$dist_root"
case "$product_root" in
  "$dist_root/$product_name") ;;
  *) printf 'Refusing to package outside dist: %s\n' "$product_root" >&2; exit 1 ;;
esac
rm -rf -- "$product_root"
rm -f -- "$archive_path"
mkdir -- "$product_root"

resolve_build_root() {
  local card="$1" sm="$2" candidates="$3" candidate path
  IFS=',' read -r -a candidate <<< "$candidates"
  for path in "${candidate[@]}"; do
    if [[ -d "$repo_root/$path" ]]; then
      printf 'SM %s (%s): using build directory %s\n' "$sm" "$card" "$path"
      printf '%s\n' "$repo_root/$path"
      return 0
    fi
  done
  printf 'No build directory for SM %s (%s); tried: %s\n' "$sm" "$card" "$candidates" >&2
  return 1
}

# Guard against shipping a mislabeled dual-SM bundle: the profiles differ only by this constant.
assert_sm_profile() {
  local build_root="$1" sm="$2" card="$3" compile_commands="$build_root/compile_commands.json"
  if [[ ! -f "$compile_commands" ]]; then
    printf 'No compile_commands.json under %s; skipping the SM profile check for %s\n' \
      "$build_root" "$card" >&2
    return 0
  fi
  if ! grep -qF "NINFER_TARGET_SM_COUNT=$sm" "$compile_commands"; then
    printf 'Build %s does not carry NINFER_TARGET_SM_COUNT=%s; refusing to label it %s\n' \
      "$build_root" "$sm" "$card" >&2
    exit 1
  fi
}

for entry in "${variants[@]}"; do
  IFS='|' read -r subdir sm candidates <<< "$entry"
  build_root="$(resolve_build_root "$subdir" "$sm" "$candidates")"
  assert_sm_profile "$build_root" "$sm" "$subdir"

  variant_root="$product_root/$subdir"
  mkdir -- "$variant_root"
  for product in "${products[@]}"; do
    source_path="$build_root/apps/$product"
    if [[ ! -f "$source_path" ]]; then
      printf 'Missing release product: %s\n' "$source_path" >&2
      exit 1
    fi
    cp -- "$source_path" "$variant_root/$product"
  done
  # Bundle any app-local shared libraries a build staged next to the executables; system
  # dependencies (FFmpeg, curl) stay distribution-provided, like the upstream Linux bundles.
  while IFS= read -r -d '' library; do
    cp -- "$library" "$variant_root/"
  done < <(find "$build_root/apps" -maxdepth 1 -name '*.so*' -type f -print0 2>/dev/null)

  # VERSION reports the upstream RTX 3090 tag; stamp each profile with the tag and card it is.
  printf '%s\n' "$release_tag-$subdir" > "$variant_root/VERSION"
done

# --- Shared root files -------------------------------------------------------
printf '%s\n' "$release_tag-sm89" > "$product_root/VERSION"
cp -- "$repo_root/LICENSE" "$product_root/"
cp -- "$repo_root/WINDOWS_PORT.md" "$product_root/"
if [[ -f "$repo_root/ninfer-4080s-build-manual.md" ]]; then
  cp -- "$repo_root/ninfer-4080s-build-manual.md" "$product_root/"
fi

cat > "$product_root/README.md" <<EOF
# NInfer sm_89 - Linux bundle ($release_tag)

This archive carries both compile-time SM profiles of the sm_89 port. Pick the folder that matches
your card and run \`ninfer-serve\` from inside it.

| Folder | Cards | NINFER_TARGET_SM_COUNT |
| --- | --- | --- |
| \`sm128-rtx4090/\` | RTX 4090 | 128 |
| \`sm80-rtx4080s/\` | RTX 4080 SUPER | 80 |

Both folders contain \`ninfer\`, \`ninfer-serve\` and \`ninfer-perplexity\`. The profiles differ only in
the attention wave geometry constant; 128 matches the RTX 4090 the project is tuned on. Either
binary runs on either card, but the matching profile avoids the attention tail-wave tax.

FFmpeg and curl come from the distribution, not from this archive. Model artifacts are not
included either: download the \`.ninfer\` artifact for your model from the repositories linked in
the project README and verify its published SHA-256 separately.

\`SHA256SUMS.txt\` covers every file in this bundle; the archive hash is in
\`SHA256SUMS-v$release_tag-sm89-linux.txt\` next to the download.
EOF

(
  cd -- "$product_root"
  mapfile -d '' files < <(find . -type f ! -name SHA256SUMS.txt -print0 | LC_ALL=C sort -z)
  sha256sum -- "${files[@]}" > SHA256SUMS.txt
)
tar -C "$dist_root" -czf "$archive_path" "$product_name"
(
  cd -- "$dist_root"
  sha256sum -- "$archive_name" > "$(basename -- "$checksum_path")"
)
du -h -- "$archive_path" "$checksum_path"