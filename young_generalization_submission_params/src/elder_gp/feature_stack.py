from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import save_variant_bundle
from .data import FeatureTable, build_test_table, build_train_table
from .metrics import (
    TrackScore,
    official_track_score,
    phq_to_binary,
    phq_to_ternary,
    task_classification_metrics,
)


RIDGE_ALPHAS = np.logspace(-3, 4, 15)
Z_MIN = 0.0
Z_MAX = float(np.log1p(27.0))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    view: str
    family: str


@dataclass
class VariantPredictions:
    name: str
    oof_z: np.ndarray
    test_z: np.ndarray
    oof_binary_proba: np.ndarray
    test_binary_proba: np.ndarray
    oof_ternary_proba: np.ndarray
    test_ternary_proba: np.ndarray


@dataclass(frozen=True)
class Decision:
    source: str
    thresholds: tuple[float, ...]
    quality: float


@dataclass(frozen=True)
class ZCalibration:
    name: str
    center: float
    scale: float
    shift: float

    def transform(self, z_pred: np.ndarray) -> np.ndarray:
        calibrated = self.center + self.scale * (z_pred - self.center) + self.shift
        return np.clip(calibrated, Z_MIN, Z_MAX)


@dataclass
class Candidate:
    subset: tuple[str, ...]
    calibration: ZCalibration
    binary: Decision
    ternary: Decision
    score: TrackScore

    def as_row(self) -> dict[str, Any]:
        return {
            "subset": "+".join(self.subset),
            "score": self.score.score,
            "cls_f1": self.score.cls_f1,
            "cls_ccc": self.score.cls_ccc,
            "cls_kappa": self.score.cls_kappa,
            "z_calibration": self.calibration.name,
            "z_scale": self.calibration.scale,
            "z_shift": self.calibration.shift,
            "binary_source": self.binary.source,
            "binary_thresholds": ",".join(f"{value:.4f}" for value in self.binary.thresholds),
            "ternary_source": self.ternary.source,
            "ternary_thresholds": ",".join(f"{value:.4f}" for value in self.ternary.thresholds),
            **self.score.as_dict(),
        }


def _default_specs() -> list[ModelSpec]:
    return [
        ModelSpec("gait_ridge", "gait", "linear"),
        ModelSpec("personality_ridge", "personality", "linear"),
        ModelSpec("fusion_ridge", "fusion", "linear"),
        ModelSpec("gait_trees", "gait", "trees"),
        ModelSpec("personality_trees", "personality", "trees"),
        ModelSpec("fusion_trees", "fusion", "trees"),
    ]


def _combined_matrix(table: FeatureTable) -> np.ndarray:
    return np.concatenate([table.gait.astype(np.float64), table.personality.astype(np.float64)], axis=1)


def _pca_components(requested: int, samples: int, personality_dim: int) -> int:
    return max(1, min(int(requested), samples - 1, personality_dim))


def _preprocessor(
    view: str,
    gait_dim: int,
    personality_dim: int,
    pca_components: int,
    train_samples: int,
    seed: int,
) -> Pipeline | ColumnTransformer:
    gait_slice = slice(0, gait_dim)
    personality_slice = slice(gait_dim, gait_dim + personality_dim)
    if view == "gait":
        return Pipeline([("gait", ColumnTransformer([("scale", StandardScaler(), gait_slice)]))])
    personality_pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=_pca_components(pca_components, train_samples, personality_dim),
                    random_state=seed,
                ),
            ),
        ]
    )
    if view == "personality":
        return ColumnTransformer([("personality", personality_pipe, personality_slice)])
    if view == "fusion":
        return ColumnTransformer(
            [
                ("gait", StandardScaler(), gait_slice),
                ("personality", personality_pipe, personality_slice),
            ]
        )
    raise ValueError(f"unknown feature view: {view}")


def _pipeline(
    spec: ModelSpec,
    target: str,
    gait_dim: int,
    personality_dim: int,
    pca_components: int,
    train_samples: int,
    seed: int,
    n_jobs: int,
) -> Pipeline:
    transform = _preprocessor(spec.view, gait_dim, personality_dim, pca_components, train_samples, seed)
    if spec.family == "linear":
        if target == "regression":
            estimator = RidgeCV(alphas=RIDGE_ALPHAS)
        else:
            estimator = LogisticRegression(
                C=0.2,
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            )
    elif spec.family == "trees":
        tree_kwargs = {
            "n_estimators": 384,
            "min_samples_leaf": 3,
            "max_features": 0.5,
            "n_jobs": n_jobs,
            "random_state": seed,
        }
        estimator = (
            ExtraTreesRegressor(**tree_kwargs)
            if target == "regression"
            else ExtraTreesClassifier(class_weight="balanced", **tree_kwargs)
        )
    else:
        raise ValueError(f"unknown model family: {spec.family}")
    return Pipeline([("transform", transform), ("estimator", estimator)])


def _aligned_proba(model: Pipeline, features: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    predicted = model.predict_proba(features)
    classes = model.named_steps["estimator"].classes_
    result = np.zeros((features.shape[0], len(labels)), dtype=np.float64)
    for src_idx, label in enumerate(classes):
        if int(label) in labels:
            result[:, labels.index(int(label))] = predicted[:, src_idx]
    row_sum = result.sum(axis=1, keepdims=True)
    return result / np.where(row_sum <= 0.0, 1.0, row_sum)


def fit_variants(
    train: FeatureTable,
    test: FeatureTable,
    n_splits: int,
    n_repeats: int,
    seed: int,
    pca_components: int,
    n_jobs: int,
    reporter: Callable[[str], None] = print,
) -> dict[str, VariantPredictions]:
    if not train.has_labels:
        raise ValueError("training table must contain labels")
    x_train = _combined_matrix(train)
    x_test = _combined_matrix(test)
    y_z = np.log1p(train.phq9)
    gait_dim = train.gait.shape[1]
    personality_dim = train.personality.shape[1]
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    variants: dict[str, VariantPredictions] = {}
    specs = _default_specs()
    for spec in specs:
        reporter(f"[variant] {spec.name}")
        oof_z = np.zeros(train.ids.size, dtype=np.float64)
        oof_binary = np.zeros((train.ids.size, 2), dtype=np.float64)
        oof_ternary = np.zeros((train.ids.size, 3), dtype=np.float64)
        oof_count = np.zeros(train.ids.size, dtype=np.float64)
        test_z_parts = []
        test_binary_parts = []
        test_ternary_parts = []
        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(x_train, train.label3), start=1):
            fold_seed = seed + 1000 * fold_idx + len(spec.name)
            regressor = _pipeline(
                spec,
                "regression",
                gait_dim,
                personality_dim,
                pca_components,
                len(fit_idx),
                fold_seed,
                n_jobs,
            )
            binary_classifier = _pipeline(
                spec,
                "binary",
                gait_dim,
                personality_dim,
                pca_components,
                len(fit_idx),
                fold_seed + 1,
                n_jobs,
            )
            ternary_classifier = _pipeline(
                spec,
                "ternary",
                gait_dim,
                personality_dim,
                pca_components,
                len(fit_idx),
                fold_seed + 2,
                n_jobs,
            )
            regressor.fit(x_train[fit_idx], y_z[fit_idx])
            binary_classifier.fit(x_train[fit_idx], train.label2[fit_idx])
            ternary_classifier.fit(x_train[fit_idx], train.label3[fit_idx])

            oof_z[val_idx] += np.clip(regressor.predict(x_train[val_idx]), Z_MIN, Z_MAX)
            oof_binary[val_idx] += _aligned_proba(binary_classifier, x_train[val_idx], (0, 1))
            oof_ternary[val_idx] += _aligned_proba(ternary_classifier, x_train[val_idx], (0, 1, 2))
            oof_count[val_idx] += 1.0
            test_z_parts.append(np.clip(regressor.predict(x_test), Z_MIN, Z_MAX))
            test_binary_parts.append(_aligned_proba(binary_classifier, x_test, (0, 1)))
            test_ternary_parts.append(_aligned_proba(ternary_classifier, x_test, (0, 1, 2)))
        if np.any(oof_count <= 0.0):
            raise RuntimeError(f"OOF coverage failed for {spec.name}")
        variants[spec.name] = VariantPredictions(
            name=spec.name,
            oof_z=oof_z / oof_count,
            test_z=np.mean(test_z_parts, axis=0),
            oof_binary_proba=oof_binary / oof_count[:, None],
            test_binary_proba=np.mean(test_binary_parts, axis=0),
            oof_ternary_proba=oof_ternary / oof_count[:, None],
            test_ternary_proba=np.mean(test_ternary_parts, axis=0),
        )
    return variants


def _quality(label_true: np.ndarray, label_pred: np.ndarray, labels: tuple[int, ...]) -> float:
    metrics = task_classification_metrics(label_true, label_pred, labels)
    return float(metrics["macro_f1"] + metrics["kappa"])


def _best_binary(phq: np.ndarray, proba: np.ndarray, y_true: np.ndarray) -> list[Decision]:
    decisions = []
    best_raw = max(
        (
            Decision("phq_threshold", (float(threshold),), _quality(y_true, phq_to_binary(phq, threshold), (0, 1)))
            for threshold in np.arange(3.5, 7.01, 0.25)
        ),
        key=lambda item: item.quality,
    )
    best_proba = max(
        (
            Decision(
                "classifier_proba",
                (float(threshold),),
                _quality(y_true, (proba[:, 1] >= threshold).astype(np.int64), (0, 1)),
            )
            for threshold in np.arange(0.25, 0.751, 0.025)
        ),
        key=lambda item: item.quality,
    )
    decisions.extend([best_raw, best_proba])
    return decisions


def _best_ternary(phq: np.ndarray, proba: np.ndarray, y_true: np.ndarray) -> list[Decision]:
    threshold_decisions = []
    for mild in np.arange(3.5, 7.01, 0.25):
        for severe in np.arange(8.0, 13.01, 0.25):
            threshold_decisions.append(
                Decision(
                    "phq_thresholds",
                    (float(mild), float(severe)),
                    _quality(y_true, phq_to_ternary(phq, mild, severe), (0, 1, 2)),
                )
            )
    classifier = Decision(
        "classifier_argmax",
        (),
        _quality(y_true, np.argmax(proba, axis=1), (0, 1, 2)),
    )
    weighted = max(
        (
            Decision(
                "classifier_weighted",
                (float(mild_weight), float(severe_weight)),
                _quality(
                    y_true,
                    _weighted_ternary_argmax(proba, float(mild_weight), float(severe_weight)),
                    (0, 1, 2),
                ),
            )
            for mild_weight in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
            for severe_weight in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
            if mild_weight != 1.0 or severe_weight != 1.0
        ),
        key=lambda item: item.quality,
    )
    return [max(threshold_decisions, key=lambda item: item.quality), classifier, weighted]


def _phq_from_z(z_pred: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(np.clip(z_pred, Z_MIN, Z_MAX)), 0.0, 27.0)


def _z_calibrations(z_pred: np.ndarray, phq_true: np.ndarray) -> list[ZCalibration]:
    pred = np.asarray(z_pred, dtype=np.float64)
    true = np.log1p(np.clip(np.asarray(phq_true, dtype=np.float64), 0.0, 27.0))
    identity = ZCalibration("identity", 0.0, 1.0, 0.0)
    if float(pred.std()) <= 1e-12:
        return [identity]
    scale = float(true.std() / pred.std())
    shift = float(true.mean() - pred.mean())
    return [
        identity,
        ZCalibration("mean", float(pred.mean()), 1.0, shift),
        ZCalibration("moment_soft", float(pred.mean()), 0.75 * scale, shift),
        ZCalibration("moment", float(pred.mean()), scale, shift),
    ]


def _subset_predictions(
    variants: dict[str, VariantPredictions],
    subset: tuple[str, ...],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_key = f"{split}_z"
    binary_key = f"{split}_binary_proba"
    ternary_key = f"{split}_ternary_proba"
    z = np.mean([getattr(variants[name], z_key) for name in subset], axis=0)
    binary = np.mean([getattr(variants[name], binary_key) for name in subset], axis=0)
    ternary = np.mean([getattr(variants[name], ternary_key) for name in subset], axis=0)
    return z, binary, ternary


def _binary_predictions(decision: Decision, phq: np.ndarray, proba: np.ndarray) -> np.ndarray:
    if decision.source == "phq_threshold":
        return phq_to_binary(phq, decision.thresholds[0])
    if decision.source == "classifier_proba":
        return (proba[:, 1] >= decision.thresholds[0]).astype(np.int64)
    raise ValueError(f"unsupported binary decision: {decision.source}")


def _weighted_ternary_argmax(proba: np.ndarray, mild_weight: float, severe_weight: float) -> np.ndarray:
    weights = np.asarray([1.0, mild_weight, severe_weight], dtype=np.float64)
    return np.argmax(proba * weights[None, :], axis=1).astype(np.int64)


def _ternary_predictions(decision: Decision, phq: np.ndarray, proba: np.ndarray) -> np.ndarray:
    if decision.source == "phq_thresholds":
        return phq_to_ternary(phq, decision.thresholds[0], decision.thresholds[1])
    if decision.source == "classifier_argmax":
        return np.argmax(proba, axis=1).astype(np.int64)
    if decision.source == "classifier_weighted":
        return _weighted_ternary_argmax(proba, decision.thresholds[0], decision.thresholds[1])
    raise ValueError(f"unsupported ternary decision: {decision.source}")


def search_candidates(
    train: FeatureTable,
    variants: dict[str, VariantPredictions],
    max_subset_size: int | None = None,
) -> list[Candidate]:
    names = tuple(variants)
    candidates = []
    subset_limit = len(names) if not max_subset_size else min(max_subset_size, len(names))
    for subset_size in range(1, subset_limit + 1):
        for subset in itertools.combinations(names, subset_size):
            raw_oof_z, binary_proba, ternary_proba = _subset_predictions(variants, subset, "oof")
            for calibration in _z_calibrations(raw_oof_z, train.phq9):
                phq = _phq_from_z(calibration.transform(raw_oof_z))
                binary_choices = _best_binary(phq, binary_proba, train.label2)
                ternary_choices = _best_ternary(phq, ternary_proba, train.label3)
                for binary in binary_choices:
                    for ternary in ternary_choices:
                        score = official_track_score(
                            train.label2,
                            train.label3,
                            train.phq9,
                            _binary_predictions(binary, phq, binary_proba),
                            _ternary_predictions(ternary, phq, ternary_proba),
                            phq,
                        )
                        candidates.append(Candidate(subset, calibration, binary, ternary, score))
    return sorted(candidates, key=lambda item: item.score.score, reverse=True)


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


def _selected_oof_rows(
    train: FeatureTable,
    candidate: Candidate,
    phq: np.ndarray,
    binary_pred: np.ndarray,
    ternary_pred: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "id": int(person_id),
            "label2": int(label2),
            "label3": int(label3),
            "phq9": float(true_phq),
            "binary_pred": int(binary),
            "ternary_pred": int(ternary),
            "phq9_pred": float(pred_phq),
            "subset": "+".join(candidate.subset),
        }
        for person_id, label2, label3, true_phq, binary, ternary, pred_phq in zip(
            train.ids,
            train.label2,
            train.label3,
            train.phq9,
            binary_pred,
            ternary_pred,
            phq,
            strict=True,
        )
    ]


def finish_variant_run(
    out_dir: Path,
    train: FeatureTable,
    test: FeatureTable,
    variants: dict[str, VariantPredictions],
    args: argparse.Namespace,
    started: float,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = search_candidates(train, variants, getattr(args, "max_subset_size", None))
    selected = candidates[0]
    oof_z, oof_binary_proba, oof_ternary_proba = _subset_predictions(variants, selected.subset, "oof")
    oof_phq = _phq_from_z(selected.calibration.transform(oof_z))
    oof_binary = _binary_predictions(selected.binary, oof_phq, oof_binary_proba)
    oof_ternary = _ternary_predictions(selected.ternary, oof_phq, oof_ternary_proba)
    test_z, test_binary_proba, test_ternary_proba = _subset_predictions(variants, selected.subset, "test")
    test_phq = _phq_from_z(selected.calibration.transform(test_z))
    test_binary = _binary_predictions(selected.binary, test_phq, test_binary_proba)
    test_ternary = _ternary_predictions(selected.ternary, test_phq, test_ternary_proba)
    zip_path = _write_submission(out_dir, test.ids, test_phq, test_binary, test_ternary)
    _write_csv(out_dir / "candidate_scores.csv", [candidate.as_row() for candidate in candidates])
    _write_csv(
        out_dir / "selected_oof_predictions.csv",
        _selected_oof_rows(train, selected, oof_phq, oof_binary, oof_ternary),
    )
    np.savez_compressed(
        out_dir / "selected_test_predictions.npz",
        ids=test.ids,
        phq9_pred=test_phq,
        binary_pred=test_binary,
        ternary_pred=test_ternary,
        binary_proba=test_binary_proba,
        ternary_proba=test_ternary_proba,
    )
    save_variant_bundle(
        out_dir / "variant_predictions.npz",
        train.ids,
        test.ids,
        train.label2,
        train.label3,
        train.phq9,
        variants,
    )
    summary = {
        "selected_candidate": selected.as_row(),
        "train_count": int(train.ids.size),
        "test_count": int(test.ids.size),
        "gait_feature_dim": int(train.gait.shape[1]),
        "personality_dim": int(train.personality.shape[1]),
        "variant_names": list(variants),
        "candidate_count": len(candidates),
        "submission_zip": str(zip_path),
        "seconds": round(time.time() - started, 3),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    summary.update(summary_extra or {})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[selected]", json.dumps(selected.as_row(), indent=2))
    print(f"[submission] {zip_path}")
    return summary


def run_feature_stack(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_dir = Path(args.out_dir).resolve()
    train_root = Path(args.train_root).resolve()
    test_root = Path(args.test_root).resolve()
    label_csv = Path(args.label_csv).resolve()
    personality_npy = Path(args.personality_npy).resolve()
    print("[data] extracting train features")
    train = build_train_table(train_root, label_csv, personality_npy)
    print("[data] extracting test features")
    test = build_test_table(test_root, personality_npy)
    print(
        f"[data] train={train.ids.size} test={test.ids.size} "
        f"gait_features={train.gait.shape[1]} personality={train.personality.shape[1]}"
    )
    variants = fit_variants(
        train,
        test,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
        pca_components=args.pca_components,
        n_jobs=args.n_jobs,
    )
    return finish_variant_run(out_dir, train, test, variants, args, started)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOF feature stack for MPDD-AVG 2026 Elder G+P.")
    parser.add_argument(
        "--train-root",
        default="extracted/Train-MPDD-Elder/Elder",
        help="Elder train root containing IMU/, labels, and personality embedding file.",
    )
    parser.add_argument("--test-root", default="extracted/Test-MPDD-Elder/Elder")
    parser.add_argument(
        "--label-csv",
        default="extracted/Train-MPDD-Elder/Elder/split_labels_train.csv",
    )
    parser.add_argument(
        "--personality-npy",
        default="extracted/Train-MPDD-Elder/Elder/descriptions_embeddings_with_ids.npy",
    )
    parser.add_argument("--out-dir", default="runs/feature_stack")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--max-subset-size",
        type=int,
        default=0,
        help="Limit searched ensemble width; 0 searches every non-empty variant subset.",
    )
    return parser


def main() -> None:
    run_feature_stack(build_parser().parse_args())


if __name__ == "__main__":
    main()
