# Young G+P Compact Student Deliverables

This package contains the compact reproducible inference path for the selected
Young G+P submission. 

## Included

- Source code for deterministic feature extraction and teacher-bank student
  inference.
- A frozen compact checkpoint:
  `checkpoints/young_gp_selected_params_checkpoint.npz`.
- Selected configuration:
  `configs/selected_optimization_params.json`.
- Reproducible runner:
  `run_inference.sh`.
- Teacher-bank build/audit utilities:
  `scripts/distill_prediction_bank.py` and `scripts/audit_teacher_bank.py`.

## Output

Running `run_inference.sh` writes:

```text
output/submission/binary.csv
output/submission/ternary.csv
output/submission/submission.zip
```

