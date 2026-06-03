from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the Young teacher-bank ladder submissions.")
    parser.add_argument("--package-root", default="../young_generalization_submission_params")
    parser.add_argument("--evidence-root", default=".")
    parser.add_argument("--mpdd-root", default=".")
    parser.add_argument("--out-dir", default="generated_submissions")
    args = parser.parse_args()

    evidence_root = Path(args.evidence_root).resolve()
    package_root = Path(args.package_root).resolve()
    mpdd_root = Path(args.mpdd_root).resolve()
    out_root = (evidence_root / args.out_dir).resolve()
    config = json.loads((evidence_root / "configs/teacher_bank_ladder_steps.json").read_text(encoding="utf-8"))
    distill_script = package_root / "scripts/distill_prediction_bank.py"
    pythonpath = str(package_root / "src")

    for step in config["steps"]:
        step_out = out_root / step["step"]
        inputs = [evidence_root / "teacher_bank_inputs" / name for name in step["teachers"]]
        cmd = [
            sys.executable,
            str(distill_script),
            *[str(path) for path in inputs],
            "--weights",
            ",".join(str(weight) for weight in step["weights"]),
            "--test-root",
            str(mpdd_root / "extracted/Test-MPDD-Young/Young"),
            "--personality-npy",
            str(mpdd_root / "extracted/Train-MPDD-Young/Young/descriptions_embeddings_with_ids.npy"),
            "--out-dir",
            str(step_out),
            "--gait-banks",
            "base,segment",
            "--normalizations",
            "standard,none",
            "--phq-tolerance",
            "0.25",
        ]
        subprocess.run(cmd, check=True, env={"PYTHONPATH": pythonpath})

    print(f"Regenerated Young ladder submissions under {out_root}")


if __name__ == "__main__":
    main()
