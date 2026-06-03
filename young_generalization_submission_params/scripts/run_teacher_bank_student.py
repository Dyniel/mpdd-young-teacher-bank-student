"""Run the frozen teacher-bank student checkpoint on Young G+P inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elder_gp.teacher_bank_student import (
    build_teacher_student_feature_table,
    load_checkpoint,
    predict_with_checkpoint,
    write_prediction_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-root", default="extracted/Test-MPDD-Young/Young")
    parser.add_argument(
        "--personality-npy",
        default="extracted/Train-MPDD-Young/Young/descriptions_embeddings_with_ids.npy",
    )
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()

    checkpoint = load_checkpoint(Path(args.checkpoint).resolve())
    table = build_teacher_student_feature_table(
        Path(args.test_root).resolve(),
        Path(args.personality_npy).resolve(),
        checkpoint.feature_spec,
    )
    phq, binary, ternary = predict_with_checkpoint(table, checkpoint)
    diagnostics = {
        "run_kind": "teacher_bank_student_inference",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_diagnostics": checkpoint.diagnostics,
        "method": "meta-supervised teacher-bank distillation student",
    }
    zip_path = write_prediction_artifacts(Path(args.out_dir).resolve(), table.ids, phq, binary, ternary, diagnostics)
    print(json.dumps(diagnostics, indent=2))
    print(f"[submission] {zip_path}")


if __name__ == "__main__":
    main()
