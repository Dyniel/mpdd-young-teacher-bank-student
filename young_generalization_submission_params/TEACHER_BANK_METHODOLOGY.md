# Teacher-Bank Methodology

The teacher-bank is a compact replacement for the original heavyweight model
library. Instead of shipping every trained model and intermediate checkpoint, the
package uses prior prediction artifacts as teacher signals.

## Procedure

1. Generate multiple model/prediction families from train/CV modeling and
   stability-oriented searches.
2. Align prediction artifacts by test subject ID.
3. Aggregate PHQ-9 by weighted mean.
4. Aggregate binary and ternary labels by weighted vote.
5. Fit a compact ridge student to the resulting teacher signal.
6. Select the stable high-CCC endpoint and store the student checkpoint for
   deterministic inference.

## Why This Is Compact

The original teacher families can require substantial compute and storage. Their
prediction artifacts are small, deterministic, and sufficient to reproduce the
selected teacher-bank student.

