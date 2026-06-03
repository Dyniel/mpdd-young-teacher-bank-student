#!/bin/bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPDD_ROOT="${MPDD_ROOT:-$(cd "$BUNDLE_ROOT/../.." && pwd)}"

python -m py_compile $(find "$BUNDLE_ROOT/young_generalization_submission_params" -name '*.py')
python -m py_compile "$BUNDLE_ROOT/young_reviewer_teacher_bank_evidence/scripts/run_ladder_from_teacher_bank.py"
bash -n "$BUNDLE_ROOT/young_generalization_submission_params/run_inference.sh"
bash -n "$BUNDLE_ROOT/young_reviewer_teacher_bank_evidence/run_ladder_from_teacher_bank.sh"

rm -rf "$BUNDLE_ROOT/young_generalization_submission_params/output"
MPDD_ROOT="$MPDD_ROOT" "$BUNDLE_ROOT/young_generalization_submission_params/run_inference.sh"

rm -rf "$BUNDLE_ROOT/young_reviewer_teacher_bank_evidence/generated_submissions"
MPDD_ROOT="$MPDD_ROOT" "$BUNDLE_ROOT/young_reviewer_teacher_bank_evidence/run_ladder_from_teacher_bank.sh"

python - "$BUNDLE_ROOT" <<'PY'
from __future__ import annotations

import filecmp
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
evidence = bundle / "young_reviewer_teacher_bank_evidence"
missing = []
for expected in sorted((evidence / "example_submissions").iterdir()):
    if not expected.is_dir():
        continue
    generated = evidence / "generated_submissions" / expected.name / "submission"
    for name in ("binary.csv", "ternary.csv"):
        if not filecmp.cmp(expected / "submission" / name, generated / name, shallow=False):
            missing.append(f"{expected.name}/{name}")
if missing:
    raise SystemExit("mismatched regenerated submissions: " + ", ".join(missing))
print("young review bundle validation OK")
PY
