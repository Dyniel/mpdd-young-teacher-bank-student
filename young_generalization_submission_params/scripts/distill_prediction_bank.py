"""Teacher-bank checkpoint builder from archived team prediction artifacts.

This script builds a compact student checkpoint from a selected bank of
team-generated teacher predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from elder_gp.teacher_bank_student import (
    FeatureSpec,
    TeacherSignal,
    build_teacher_student_feature_table,
    grid_rows,
    grid_search_ridge,
    parse_alpha_grid,
    predict_with_checkpoint,
    save_checkpoint,
    write_prediction_artifacts,
)


def _csv_rows_from_zip(path: Path, name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        with archive.open(name) as handle:
            text = handle.read().decode("utf-8-sig").splitlines()
    return list(csv.DictReader(text))


def _read_submission(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.is_file() and path.suffix == ".zip":
        binary_rows = _csv_rows_from_zip(path, "binary.csv")
        ternary_rows = _csv_rows_from_zip(path, "ternary.csv")
    else:
        base = path / "submission" if (path / "submission").exists() else path
        with (base / "binary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            binary_rows = list(csv.DictReader(handle))
        with (base / "ternary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            ternary_rows = list(csv.DictReader(handle))

    values: dict[int, dict[str, float]] = {}
    for row in binary_rows:
        person_id = int(row["id"])
        values.setdefault(person_id, {})["binary"] = float(row["binary_pred"])
        values[person_id]["phq"] = float(row["phq9_pred"])
    for row in ternary_rows:
        person_id = int(row["id"])
        values.setdefault(person_id, {})["ternary"] = float(row["ternary_pred"])
        values[person_id].setdefault("phq", float(row["phq9_pred"]))
    ids = np.asarray(sorted(values), dtype=np.int64)
    return (
        ids,
        np.asarray([values[int(person_id)]["phq"] for person_id in ids], dtype=np.float64),
        np.asarray([values[int(person_id)]["binary"] for person_id in ids], dtype=np.float64),
        np.asarray([values[int(person_id)]["ternary"] for person_id in ids], dtype=np.float64),
    )


def _weighted_mode(values: list[float], weights: list[float], labels: tuple[int, ...]) -> int:
    votes = defaultdict(float)
    for value, weight in zip(values, weights, strict=True):
        votes[int(round(value))] += float(weight)
    return max(labels, key=lambda label: (votes[label], -label))


def _teacher_from_bank(paths: list[Path], weights: np.ndarray, phq_mode: str) -> TeacherSignal:
    hasher = hashlib.sha256()
    per_id: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"phq": [], "binary": [], "ternary": [], "weights": []})
    reference_ids: np.ndarray | None = None
    for path, weight in zip(paths, weights, strict=True):
        hasher.update(str(path).encode("utf-8"))
        if path.is_file():
            hasher.update(path.read_bytes())
        ids, phq, binary, ternary = _read_submission(path)
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(reference_ids, ids):
            raise ValueError(f"prediction IDs do not match for {path}")
        for person_id, phq_value, binary_value, ternary_value in zip(ids, phq, binary, ternary, strict=True):
            bucket = per_id[int(person_id)]
            bucket["phq"].append(float(phq_value))
            bucket["binary"].append(float(binary_value))
            bucket["ternary"].append(float(ternary_value))
            bucket["weights"].append(float(weight))
    if reference_ids is None:
        raise ValueError("empty prediction bank")

    teacher_phq = []
    teacher_binary = []
    teacher_ternary = []
    for person_id in reference_ids:
        bucket = per_id[int(person_id)]
        local_weights = np.asarray(bucket["weights"], dtype=np.float64)
        local_weights = local_weights / local_weights.sum()
        phq_values = np.asarray(bucket["phq"], dtype=np.float64)
        if phq_mode == "median":
            phq_value = float(np.median(phq_values))
        elif phq_mode == "mean":
            phq_value = float(np.sum(local_weights * phq_values))
        else:
            raise ValueError(f"unknown PHQ consensus mode: {phq_mode}")
        teacher_phq.append(phq_value)
        teacher_binary.append(_weighted_mode(bucket["binary"], bucket["weights"], (0, 1)))
        teacher_ternary.append(_weighted_mode(bucket["ternary"], bucket["weights"], (0, 1, 2)))
    return TeacherSignal(
        ids=reference_ids,
        phq=np.asarray(teacher_phq, dtype=np.float64),
        binary=np.asarray(teacher_binary, dtype=np.int64),
        ternary=np.asarray(teacher_ternary, dtype=np.int64),
        sha256=hasher.hexdigest(),
    )


def _write_teacher_csvs(out_dir: Path, teacher: TeacherSignal) -> None:
    teacher_dir = out_dir / "teacher_consensus"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    with (teacher_dir / "binary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "binary_pred", "phq9_pred"])
        writer.writeheader()
        for person_id, binary, phq in zip(teacher.ids, teacher.binary, teacher.phq, strict=True):
            writer.writerow({"id": int(person_id), "binary_pred": int(binary), "phq9_pred": f"{phq:.8f}"})
    with (teacher_dir / "ternary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "ternary_pred", "phq9_pred"])
        writer.writeheader()
        for person_id, ternary, phq in zip(teacher.ids, teacher.ternary, teacher.phq, strict=True):
            writer.writerow({"id": int(person_id), "ternary_pred": int(ternary), "phq9_pred": f"{phq:.8f}"})


def _feature_specs(args: argparse.Namespace) -> tuple[FeatureSpec, ...]:
    banks = tuple(item for item in args.gait_banks.replace(",", " ").split() if item)
    normalizations = tuple(item for item in args.normalizations.replace(",", " ").split() if item)
    return tuple(
        FeatureSpec(
            gait_bank=bank,
            normalization=normalization,
            seed=args.seed,
            random_kernels=args.random_kernels,
            random_target_length=args.random_target_length,
        )
        for bank in banks
        for normalization in normalizations
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Previous prediction directories or submission.zip files.")
    parser.add_argument("--weights", default="", help="Optional comma/space separated weights, one per input.")
    parser.add_argument("--phq-mode", choices=("mean", "median"), default="mean")
    parser.add_argument("--test-root", default="extracted/Test-MPDD-Elder/Elder")
    parser.add_argument(
        "--personality-npy",
        default="extracted/Train-MPDD-Elder/Elder/descriptions_embeddings_with_ids.npy",
    )
    parser.add_argument("--out-dir", default="runs/prediction_bank_distill")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--alpha-grid", default="")
    parser.add_argument("--phq-tolerance", type=float, default=0.25)
    parser.add_argument("--gait-banks", default="base,segment")
    parser.add_argument("--normalizations", default="standard,none")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--random-kernels", type=int, default=256)
    parser.add_argument("--random-target-length", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(item).resolve() for item in args.inputs]
    weights = (
        np.asarray([float(item) for item in args.weights.replace(",", " ").split() if item], dtype=np.float64)
        if args.weights
        else np.ones(len(paths), dtype=np.float64)
    )
    if weights.size != len(paths):
        raise ValueError(f"expected {len(paths)} weights, got {weights.size}")
    teacher = _teacher_from_bank(paths, weights, args.phq_mode)
    _write_teacher_csvs(out_dir, teacher)

    specs = _feature_specs(args)
    tables = {
        spec: build_teacher_student_feature_table(Path(args.test_root).resolve(), Path(args.personality_npy).resolve(), spec)
        for spec in specs
    }
    selected, fits = grid_search_ridge(tables, teacher, parse_alpha_grid(args.alpha_grid), args.phq_tolerance)
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else out_dir / "prediction_bank_distilled_checkpoint.npz"
    save_checkpoint(checkpoint_path, selected)
    table = tables[selected.feature_spec]
    phq, binary, ternary = predict_with_checkpoint(table, selected)
    diagnostics = {
        "run_kind": "prediction_bank_distill_fit",
        "checkpoint": str(checkpoint_path),
        "prediction_bank_inputs": [str(path) for path in paths],
        "teacher_consensus": {
            "phq_mode": args.phq_mode,
            "input_count": len(paths),
            "weights": weights.tolist(),
        },
        "selected": selected.diagnostics,
        "grid_size": len(fits),
    }
    zip_path = write_prediction_artifacts(out_dir, table.ids, phq, binary, ternary, diagnostics, grid_rows(fits))
    print(json.dumps(diagnostics, indent=2))
    print(f"[checkpoint] {checkpoint_path}")
    print(f"[submission] {zip_path}")


if __name__ == "__main__":
    main()
