#!/bin/bash
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "$(pwd)/extracted" ]]; then
  ROOT="$(pwd)"
elif [[ -d "$PKG_ROOT/../extracted" ]]; then
  ROOT="$(cd "$PKG_ROOT/.." && pwd)"
elif [[ -d "$PKG_ROOT/../../extracted" ]]; then
  ROOT="$(cd "$PKG_ROOT/../.." && pwd)"
else
  ROOT="${MPDD_ROOT:-$(pwd)}"
fi

cd "$ROOT"

export PYTHONPATH="$PKG_ROOT/src:${PYTHONPATH:-}"

python "$PKG_ROOT/scripts/run_teacher_bank_student.py" \
  --checkpoint "$PKG_ROOT/checkpoints/young_gp_selected_params_checkpoint.npz" \
  --test-root "$ROOT/extracted/Test-MPDD-Young/Young" \
  --personality-npy "$ROOT/extracted/Train-MPDD-Young/Young/descriptions_embeddings_with_ids.npy" \
  --out-dir "$PKG_ROOT/output"

echo "Wrote $PKG_ROOT/output/submission/submission.zip"
