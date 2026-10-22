#!/usr/bin/env bash
# Download the SPICE v2.0 structure sets from the GRACE-OFF Zenodo record (18662355).
#   biased_test_set.xyz : 10k random SPICE structures  -> LoRA training pool (~38 MB)
#   GRACE_test_set.xyz  : 10k structures held out from GRACE-OFF training -> evaluation (~38 MB)
# (SPICE_V2.0_cleaned.xyz on that record is identical to GRACE_test_set.xyz, so it is skipped.)
#
# Usage: bash scripts/download_data.sh [target_dir]   (default: ./data)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-$ROOT/data}"
mkdir -p "$TARGET"

for f in biased_test_set.xyz GRACE_test_set.xyz; do
  if [ -s "$TARGET/$f" ]; then
    echo "exists: $TARGET/$f"
  else
    echo "downloading: $f"
    curl -fL --retry 3 -o "$TARGET/$f" "https://zenodo.org/api/records/18662355/files/$f/content"
  fi
done
# verify the content against the pinned SHA-256s (scripts/checksums.py)
python3 "$ROOT/scripts/checksums.py" verify "$TARGET/biased_test_set.xyz" "$TARGET/GRACE_test_set.xyz"
echo "Done -> $TARGET"
