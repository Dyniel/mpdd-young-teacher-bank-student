#!/bin/bash
set -euo pipefail

EVIDENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$EVIDENCE_ROOT/../young_generalization_submission_params}"
MPDD_ROOT="${MPDD_ROOT:-$(cd "$EVIDENCE_ROOT/../.." && pwd)}"

python "$EVIDENCE_ROOT/scripts/run_ladder_from_teacher_bank.py" \
  --package-root "$PACKAGE_ROOT" \
  --evidence-root "$EVIDENCE_ROOT" \
  --mpdd-root "$MPDD_ROOT"
