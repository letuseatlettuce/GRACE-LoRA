#!/usr/bin/env bash
# Download the GRACE student (foundation model + training checkpoint) and the
# GRACE-OFF teacher (heid-lab/grace-off, 2-layer medium, trained on SPICE v2.0).
#
# Usage (inside the activated venv):
#   bash scripts/download_models.sh
#   STUDENT=GRACE-2L-SMAX-OMAT-medium bash scripts/download_models.sh   # other student
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUDENT="${STUDENT:-GRACE-2L-OMAT-medium-ft-AM}"
TEACHER_DIR="$ROOT/models/grace-off"
TEACHER_SUBDIR="models/2l/b_off_medium"

echo "== Student: $STUDENT"
# SavedModel for inference -> ~/.cache/grace/<name>
grace_models download "$STUDENT"
# Checkpoint + model.yaml (needed to rebuild the model with LoRA adapters)
# -> ~/.cache/grace/checkpoints/<name>
grace_models checkpoint "$STUDENT"

echo "== Teacher: GRACE-OFF ($TEACHER_SUBDIR)"
# Sparse clone: the full repo is ~700 MB (MACE models, notebooks); we only need one model.
if [ ! -d "$TEACHER_DIR/.git" ]; then
  mkdir -p "$ROOT/models"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/heid-lab/grace-off.git "$TEACHER_DIR"
fi
# --no-cone: cone mode would also pull files in parent dirs (~80 MB of MACE models)
git -C "$TEACHER_DIR" sparse-checkout set --no-cone "/$TEACHER_SUBDIR/"

echo "== Verifying checksums"
CACHE="${GRACE_CACHE:-$HOME/.cache/grace}"
python3 "$ROOT/scripts/checksums.py" verify-dir "$CACHE/$STUDENT" "$CACHE/checkpoints/$STUDENT" \
  "$TEACHER_DIR/$TEACHER_SUBDIR/seed/1/saved_model"

echo
echo "Done."
echo "  Student model:      ~/.cache/grace/$STUDENT"
echo "  Student checkpoint: ~/.cache/grace/checkpoints/$STUDENT"
echo "  Teacher model:      $TEACHER_DIR/$TEACHER_SUBDIR/seed/1/saved_model  (fp64; not casted_model)"
