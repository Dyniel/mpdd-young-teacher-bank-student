# Method Description

The Young G+P solution is delivered as a compact teacher-bank student.

The original modeling campaign produced several train/CV model families and
submission families. To avoid distributing the full heavyweight model bank, the
selected family is represented by compact prediction artifacts. Those artifacts
are aligned by subject ID and distilled into a ridge student over deterministic
gait/personality features.

## Feature Stack

- IMU gait features are extracted deterministically from the Young test gait
  arrays.
- Personality embeddings are loaded from the released Young embedding file.
- Candidate feature banks include `base` and `segment` gait features.
- Candidate normalizations include `standard` and `none`.

## Student Fit

The student is a dual ridge regressor fit to three targets:

- PHQ-9 score,
- binary class,
- ternary class.

Class predictions are produced by rounding and clipping the corresponding
student outputs. PHQ predictions are clipped to the PHQ range.

## Grid Search

The selected high-CCC checkpoint was chosen from:

- 2 gait banks,
- 2 normalization modes,
- 25 ridge alpha values,
- total grid size 100.

The selection criterion minimizes class mismatches against the teacher-bank
signal, requires PHQ values to stay within tolerance, and prefers the largest
acceptable regularization strength.


## Reproduction

The final checkpoint is frozen in this package. A reviewer can reproduce the
submission by running:

```bash
./run_inference.sh
```
