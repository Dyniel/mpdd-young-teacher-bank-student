# MPDD Young G+P Review Bundle

This bundle puts the Young G+P reproducibility materials in one place.

## Directories

- `young_generalization_submission_params/` - final compact student package.
- `young_reviewer_teacher_bank_evidence/` - compact teacher-bank ladder with
  representative model-family prediction artifacts.
- `train_cv_baseline_reference/` - light train/CV reference submissions and
  local CV scores.
- `archives/` - standalone ZIP copies of the final package and evidence pack.
- `docs/` - reviewer notes, rebuild-cost estimate, and validation summary.
- `scripts/` - bundle-level validation helper.

## Main Reproduction

```bash
./delivery/young_final_reviewer_bundle/young_generalization_submission_params/run_inference.sh
```

## Teacher-Bank Ladder

```bash
./delivery/young_final_reviewer_bundle/young_reviewer_teacher_bank_evidence/run_ladder_from_teacher_bank.sh
```

The ladder table is:

```text
young_reviewer_teacher_bank_evidence/teacher_bank_ladder_metrics.csv
```

## Scope

The full heavyweight model bank is not included. It is represented by compact
prediction artifacts that can be distilled deterministically into small student
checkpoints.
