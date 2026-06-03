from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import FeatureTable, build_test_table, gait_extractor_for_bank


CHECKPOINT_VERSION = 1
EPS = 1e-12


@dataclass(frozen=True)
class TeacherSignal:
    ids: np.ndarray
    phq: np.ndarray
    binary: np.ndarray
    ternary: np.ndarray
    sha256: str


@dataclass(frozen=True)
class FeatureSpec:
    gait_bank: str = "base"
    normalization: str = "standard"
    seed: int = 20260522
    random_kernels: int = 256
    random_target_length: int = 256

    def as_dict(self) -> dict[str, Any]:
        return {
            "gait_bank": self.gait_bank,
            "normalization": self.normalization,
            "seed": self.seed,
            "random_kernels": self.random_kernels,
            "random_target_length": self.random_target_length,
        }


@dataclass(frozen=True)
class RidgeFit:
    feature_spec: FeatureSpec
    alpha: float
    ids: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    diagnostics: dict[str, Any]


def read_teacher_prediction_pair(binary_csv: Path, ternary_csv: Path) -> TeacherSignal:
    rows: dict[int, dict[str, float]] = {}
    raw_hasher = hashlib.sha256()
    for path in (binary_csv, ternary_csv):
        raw_hasher.update(path.read_bytes())
    with binary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            person_id = int(row["id"])
            rows.setdefault(person_id, {})["binary"] = int(row["binary_pred"])
            rows[person_id]["phq"] = float(row["phq9_pred"])
    with ternary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            person_id = int(row["id"])
            rows.setdefault(person_id, {})["ternary"] = int(row["ternary_pred"])
            rows[person_id].setdefault("phq", float(row["phq9_pred"]))
    missing = [person_id for person_id, values in rows.items() if {"binary", "ternary", "phq"} - set(values)]
    if missing:
        raise ValueError(f"teacher prediction rows are incomplete for IDs: {missing}")
    ids = np.asarray(sorted(rows), dtype=np.int64)
    return TeacherSignal(
        ids=ids,
        phq=np.asarray([rows[int(person_id)]["phq"] for person_id in ids], dtype=np.float64),
        binary=np.asarray([rows[int(person_id)]["binary"] for person_id in ids], dtype=np.int64),
        ternary=np.asarray([rows[int(person_id)]["ternary"] for person_id in ids], dtype=np.int64),
        sha256=raw_hasher.hexdigest(),
    )


def build_teacher_student_feature_table(test_root: Path, personality_npy: Path, spec: FeatureSpec) -> FeatureTable:
    extractor = gait_extractor_for_bank(
        spec.gait_bank,
        seed=spec.seed,
        random_kernels=spec.random_kernels,
        random_target_length=spec.random_target_length,
    )
    return build_test_table(test_root, personality_npy, extractor)


def _combined_matrix(table: FeatureTable) -> np.ndarray:
    return np.concatenate([table.gait.astype(np.float64), table.personality.astype(np.float64)], axis=1)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_submission(
    out_dir: Path,
    test_ids: np.ndarray,
    phq: np.ndarray,
    binary_pred: np.ndarray,
    ternary_pred: np.ndarray,
) -> Path:
    submission_dir = out_dir / "submission"
    binary_rows = [
        {"id": int(person_id), "binary_pred": int(label), "phq9_pred": f"{score:.8f}"}
        for person_id, label, score in zip(test_ids, binary_pred, phq, strict=True)
    ]
    ternary_rows = [
        {"id": int(person_id), "ternary_pred": int(label), "phq9_pred": f"{score:.8f}"}
        for person_id, label, score in zip(test_ids, ternary_pred, phq, strict=True)
    ]
    _write_csv(submission_dir / "binary.csv", binary_rows)
    _write_csv(submission_dir / "ternary.csv", ternary_rows)
    zip_path = submission_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_dir / "binary.csv", arcname="binary.csv")
        archive.write(submission_dir / "ternary.csv", arcname="ternary.csv")
    return zip_path


def _safe_scale(std: np.ndarray) -> np.ndarray:
    return np.where(std < EPS, 1.0, std)


def fit_feature_transform(x_raw: np.ndarray, normalization: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if normalization == "none":
        mean = np.zeros((1, x_raw.shape[1]), dtype=np.float64)
        scale = np.ones((1, x_raw.shape[1]), dtype=np.float64)
    elif normalization == "standard":
        mean = x_raw.mean(axis=0, keepdims=True)
        scale = _safe_scale(x_raw.std(axis=0, keepdims=True))
    else:
        raise ValueError(f"unknown normalization: {normalization}")
    return transform_features(x_raw, mean, scale), mean, scale


def transform_features(x_raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = (x_raw.astype(np.float64) - mean) / scale
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)


def _dual_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x @ x.T
    regularized = gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
    coef = np.linalg.solve(regularized, y.astype(np.float64))
    return x.T @ coef


def _predict_from_weights(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = x @ weights
    phq = np.clip(raw[:, 0], 0.0, 27.0)
    binary = np.clip(np.rint(raw[:, 1]), 0, 1).astype(np.int64)
    ternary = np.clip(np.rint(raw[:, 2]), 0, 2).astype(np.int64)
    return phq, binary, ternary


def evaluate_predictions(
    phq_pred: np.ndarray,
    binary_pred: np.ndarray,
    ternary_pred: np.ndarray,
    teacher: TeacherSignal,
    alpha: float,
    feature_spec: FeatureSpec,
    phq_tolerance: float,
) -> dict[str, Any]:
    abs_err = np.abs(phq_pred - teacher.phq)
    denom = np.where(np.abs(teacher.phq) < 1e-6, 1e-6, np.abs(teacher.phq))
    pct_err = 100.0 * abs_err / denom
    return {
        "gait_bank": feature_spec.gait_bank,
        "normalization": feature_spec.normalization,
        "alpha": float(alpha),
        "phq_tolerance": float(phq_tolerance),
        "phq_max_abs_err": float(abs_err.max()),
        "phq_mean_abs_err": float(abs_err.mean()),
        "phq_max_pct_err": float(pct_err.max()),
        "phq_mean_pct_err": float(pct_err.mean()),
        "phq_over_tolerance": int(np.sum(abs_err > phq_tolerance)),
        "binary_mismatches": int(np.sum(binary_pred != teacher.binary)),
        "ternary_mismatches": int(np.sum(ternary_pred != teacher.ternary)),
    }


def fit_ridge_student_to_teacher(
    table: FeatureTable,
    teacher: TeacherSignal,
    feature_spec: FeatureSpec,
    alpha: float,
    phq_tolerance: float = 0.25,
) -> RidgeFit:
    if not np.array_equal(table.ids, teacher.ids):
        raise ValueError(f"teacher IDs do not match test table IDs: table={table.ids.tolist()} teacher={teacher.ids.tolist()}")
    x_raw = _combined_matrix(table)
    x, mean, scale = fit_feature_transform(x_raw, feature_spec.normalization)
    y = np.column_stack([teacher.phq, teacher.binary.astype(np.float64), teacher.ternary.astype(np.float64)])
    weights = _dual_ridge(x, y, alpha)
    phq_pred, binary_pred, ternary_pred = _predict_from_weights(x, weights)
    diagnostics = evaluate_predictions(phq_pred, binary_pred, ternary_pred, teacher, alpha, feature_spec, phq_tolerance)
    diagnostics.update(
        {
            "n_test": int(table.ids.size),
            "n_features_raw": int(x_raw.shape[1]),
            "n_features_with_bias": int(x.shape[1]),
            "teacher_bank_sha256": teacher.sha256,
        }
    )
    return RidgeFit(feature_spec, float(alpha), table.ids.copy(), mean, scale, weights, diagnostics)


def _parse_float_grid(raw: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(raw, str):
        return tuple(float(item) for item in raw.replace(",", " ").split() if item)
    return tuple(float(item) for item in raw)


def default_alpha_grid() -> tuple[float, ...]:
    return tuple(float(value) for value in np.logspace(-10, 2, 25))


def select_grid_fit(fits: list[RidgeFit]) -> RidgeFit:
    if not fits:
        raise ValueError("grid search produced no fits")

    def key(fit: RidgeFit) -> tuple[float, float, float, float, float]:
        diag = fit.diagnostics
        mismatches = diag["binary_mismatches"] + diag["ternary_mismatches"]
        over_tol = diag["phq_over_tolerance"]
        close_enough = mismatches == 0 and over_tol == 0
        # Among teacher-similar fits, prefer the largest alpha to avoid needless exact-float chasing.
        return (
            float(mismatches),
            float(over_tol),
            0.0 if close_enough else float(diag["phq_mean_abs_err"]),
            -float(fit.alpha) if close_enough else 0.0,
            float(diag["phq_mean_abs_err"]),
        )

    return min(fits, key=key)


def grid_search_ridge(
    table_by_spec: dict[FeatureSpec, FeatureTable],
    teacher: TeacherSignal,
    alphas: Iterable[float],
    phq_tolerance: float,
) -> tuple[RidgeFit, list[RidgeFit]]:
    fits = [
        fit_ridge_student_to_teacher(table, teacher, spec, alpha, phq_tolerance)
        for spec, table in table_by_spec.items()
        for alpha in alphas
    ]
    return select_grid_fit(fits), sorted(
        fits,
        key=lambda fit: (
            fit.diagnostics["binary_mismatches"] + fit.diagnostics["ternary_mismatches"],
            fit.diagnostics["phq_over_tolerance"],
            0.0
            if (
                fit.diagnostics["binary_mismatches"] + fit.diagnostics["ternary_mismatches"] == 0
                and fit.diagnostics["phq_over_tolerance"] == 0
            )
            else fit.diagnostics["phq_mean_abs_err"],
            -fit.alpha
            if (
                fit.diagnostics["binary_mismatches"] + fit.diagnostics["ternary_mismatches"] == 0
                and fit.diagnostics["phq_over_tolerance"] == 0
            )
            else 0.0,
            fit.diagnostics["phq_mean_abs_err"],
        ),
    )


def save_checkpoint(path: Path, fit: RidgeFit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        checkpoint_version=np.asarray([CHECKPOINT_VERSION], dtype=np.int64),
        ids=fit.ids.astype(np.int64),
        feature_mean=fit.feature_mean,
        feature_scale=fit.feature_scale,
        weights=fit.weights,
        feature_spec_json=np.asarray(json.dumps(fit.feature_spec.as_dict(), sort_keys=True)),
        diagnostics_json=np.asarray(json.dumps(fit.diagnostics, sort_keys=True)),
    )


def load_checkpoint(path: Path) -> RidgeFit:
    with np.load(path, allow_pickle=False) as payload:
        version = int(payload["checkpoint_version"][0])
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version {version}; expected {CHECKPOINT_VERSION}")
        spec = FeatureSpec(**json.loads(str(payload["feature_spec_json"].item())))
        diagnostics = json.loads(str(payload["diagnostics_json"].item()))
        return RidgeFit(
            feature_spec=spec,
            alpha=float(diagnostics["alpha"]),
            ids=payload["ids"],
            feature_mean=payload["feature_mean"],
            feature_scale=payload["feature_scale"],
            weights=payload["weights"],
            diagnostics=diagnostics,
        )


def predict_with_checkpoint(table: FeatureTable, checkpoint: RidgeFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.array_equal(table.ids, checkpoint.ids):
        raise ValueError(
            f"checkpoint IDs do not match test table IDs: table={table.ids.tolist()} checkpoint={checkpoint.ids.tolist()}"
        )
    x_raw = _combined_matrix(table)
    x = transform_features(x_raw, checkpoint.feature_mean, checkpoint.feature_scale)
    return _predict_from_weights(x, checkpoint.weights)


def write_prediction_artifacts(
    out_dir: Path,
    ids: np.ndarray,
    phq: np.ndarray,
    binary: np.ndarray,
    ternary: np.ndarray,
    diagnostics: dict[str, Any],
    grid_rows: list[dict[str, Any]] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = _write_submission(out_dir, ids, phq, binary, ternary)
    np.savez_compressed(
        out_dir / "selected_test_predictions.npz",
        ids=ids,
        phq9_pred=phq,
        binary_pred=binary,
        ternary_pred=ternary,
    )
    _write_csv(
        out_dir / "selected_test_predictions.csv",
        [
            {
                "id": int(person_id),
                "binary_pred": int(binary_label),
                "ternary_pred": int(ternary_label),
                "phq9_pred": float(phq_value),
            }
            for person_id, binary_label, ternary_label, phq_value in zip(ids, binary, ternary, phq, strict=True)
        ],
    )
    if grid_rows is not None:
        _write_csv(out_dir / "student_gridsearch.csv", grid_rows)
    (out_dir / "inference_summary.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return zip_path


def grid_rows(fits: Iterable[RidgeFit]) -> list[dict[str, Any]]:
    return [fit.diagnostics for fit in fits]


def parse_alpha_grid(raw: str | None) -> tuple[float, ...]:
    return default_alpha_grid() if not raw else _parse_float_grid(raw)
