# Young Teacher-Bank Evidence

This bundle is a compact evidence set for the Young G+P teacher-bank
distillation strategy. It replaces the original heavyweight model/checkpoint
library with small prediction artifacts from representative model families.

## Contents

- `teacher_bank_inputs/` - compact prediction artifacts used as teacher signals.
- `example_submissions/` - generated submissions for each ladder step.
- `teacher_bank_ladder_metrics.csv` - progression from train/CV references to
  the selected high-CCC final family.
- `configs/teacher_bank_ladder_steps.json` - step definitions.
- `scripts/run_ladder_from_teacher_bank.py` - regeneration script.

## Reproducing

Place this directory next to `young_generalization_submission_params/` and run:

```bash
./run_ladder_from_teacher_bank.sh
```

The generated outputs are written to `generated_submissions/`.


