# Young G+P Teacher-Bank Student Package

This repository contains the compact teacher-bank student verification package for the Young G+P prediction files.

## Motivation

We initially approached the Young G+P task as a conventional applied classification problem: build a robust baseline, combine the available gait and personality modalities, tune the model, and verify that the final system does not overfit. In practice, the task setting was unusually constrained. Both the training and test splits are small, and standard tabular-learning tools such as gradient-boosted trees can overfit easily when the feature space is high-dimensional relative to the number of subjects.

The personality modality was also challenging in a specific way. The released personality information is available through embeddings rather than directly interpretable raw questionnaire variables. This made it difficult to inspect the feature distribution in a clinically or behaviorally meaningful way, or to reason directly about which embedding dimensions should drive the final predictions.

For this reason, we treated the final system as a small-sample optimization and stability problem rather than a large-scale representation-learning problem. The goal was to construct a stable, deterministic prediction artifact that performs well under the specific Young G+P evaluation conditions while avoiding fragile dependence on a single model family, seed, or parameter setting.

Our first step was to establish conservative baselines and tune their parameters using standard optimization procedures, including grid search and Optuna-style parameter exploration. The key concern was stability: we wanted a baseline that was not trivially overfit and whose predictions were not dominated by one unstable configuration.

The final package uses a compact teacher-bank student approach. Inspired by practical experience with model distillation and ensemble compression, we built a teacher bank from selected team-generated prediction artifacts. These teacher predictions summarize multiple locally strong solutions and preserve useful variation across model configurations. A compact student checkpoint then distills this selected teacher-consensus signal into a deterministic inference artifact.

This design allowed us to combine several useful optimization centers without allowing one fragile configuration to dominate the final output. The delivered checkpoint should therefore be understood as a compact, frozen student artifact for reproducing the selected Young G+P prediction profile, not as a large general-purpose pretrained model.

## Running inference

Run inference from a checkout placed next to, or inside, the standard Young data layout:

bash ./run_inference.sh 

If the data lives elsewhere, point MPDD_ROOT at the repository/data root that contains extracted/:

bash MPDD_ROOT=/path/to/mpdd-young-pipeline ./run_inference.sh 

Expected output:

text output/submission/submission.zip 

The output archive contains the required prediction files:

text binary.csv ternary.csv 

## Package contents

The package includes:

text METHOD_DESCRIPTION.md TEACHER_BANK_METHODOLOGY.md DELIVERABLES.md requirements.txt requirements-elder-gp.txt requirements-elder-gp-gpu.txt run_inference.sh checkpoints/ configs/ scripts/ src/elder_gp/ 

The repository contains the method documentation, source code, configuration files, environment requirements, run script, and the frozen compact student checkpoint required for deterministic verification.

## Review scope

The delivered checkpoint should be reviewed as a compact teacher-bank distillation artifact. The package is intended to support deterministic reproduction of the selected Young G+P prediction files from the included frozen checkpoint and released challenge inputs.

The original heavy model-family library is not included in this repository. Instead, the selected teacher family is represented by compact prediction artifacts, and the included source code deterministically distills those artifacts into a small student checkpoint.

No private datasets are used in the included code paths.
