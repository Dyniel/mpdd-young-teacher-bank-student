from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def _as_1d(values: np.ndarray | list[float] | list[int]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"expected a 1D array, got shape={array.shape}")
    return array


def concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_1d(y_true).astype(np.float64)
    y_pred = _as_1d(y_pred).astype(np.float64)
    if y_true.size != y_pred.size:
        raise ValueError("CCC inputs must have the same length")
    cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    denom = y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2
    return float(2.0 * cov / denom) if denom > EPS else 0.0


def log_phq_ccc(phq_true: np.ndarray, phq_pred: np.ndarray) -> float:
    true = np.log1p(np.clip(_as_1d(phq_true).astype(np.float64), 0.0, 27.0))
    pred = np.log1p(np.clip(_as_1d(phq_pred).astype(np.float64), 0.0, 27.0))
    return concordance_ccc(true, pred)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int), strict=True):
        if true in label_to_index and pred in label_to_index:
            matrix[label_to_index[true], label_to_index[pred]] += 1.0
    return matrix


def macro_f1(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    labels: tuple[int, ...],
) -> float:
    confusion = _confusion(_as_1d(y_true), _as_1d(y_pred), labels)
    scores = []
    for idx in range(confusion.shape[0]):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom <= EPS else float(2.0 * tp / denom))
    return float(np.mean(scores))


def cohen_kappa(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    labels: tuple[int, ...],
) -> float:
    confusion = _confusion(_as_1d(y_true), _as_1d(y_pred), labels)
    total = confusion.sum()
    if total <= EPS:
        return 0.0
    observed = np.trace(confusion) / total
    expected = float(confusion.sum(axis=1) @ confusion.sum(axis=0) / (total * total))
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 - EPS else 0.0


def task_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[int, ...],
) -> dict[str, float]:
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)
    if true.size != pred.size:
        raise ValueError("classification inputs must have the same length")
    return {
        "macro_f1": macro_f1(true, pred, labels),
        "kappa": cohen_kappa(true, pred, labels),
        "accuracy": float(np.mean(true.astype(int) == pred.astype(int))),
    }


@dataclass(frozen=True)
class TrackScore:
    score: float
    cls_f1: float
    cls_ccc: float
    cls_kappa: float
    binary_f1: float
    binary_kappa: float
    binary_ccc: float
    ternary_f1: float
    ternary_kappa: float
    ternary_ccc: float

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.__dict__.items()}


def official_track_score(
    label2_true: np.ndarray,
    label3_true: np.ndarray,
    phq_true: np.ndarray,
    binary_pred: np.ndarray,
    ternary_pred: np.ndarray,
    binary_phq_pred: np.ndarray,
    ternary_phq_pred: np.ndarray | None = None,
) -> TrackScore:
    binary = task_classification_metrics(label2_true, binary_pred, (0, 1))
    ternary = task_classification_metrics(label3_true, ternary_pred, (0, 1, 2))
    binary_ccc = log_phq_ccc(phq_true, binary_phq_pred)
    ternary_ccc = log_phq_ccc(
        phq_true,
        binary_phq_pred if ternary_phq_pred is None else ternary_phq_pred,
    )
    cls_f1 = (binary["macro_f1"] + ternary["macro_f1"]) / 2.0
    cls_kappa = (binary["kappa"] + ternary["kappa"]) / 2.0
    cls_ccc = (binary_ccc + ternary_ccc) / 2.0
    return TrackScore(
        score=(cls_f1 + cls_ccc + cls_kappa) / 3.0,
        cls_f1=cls_f1,
        cls_ccc=cls_ccc,
        cls_kappa=cls_kappa,
        binary_f1=binary["macro_f1"],
        binary_kappa=binary["kappa"],
        binary_ccc=binary_ccc,
        ternary_f1=ternary["macro_f1"],
        ternary_kappa=ternary["kappa"],
        ternary_ccc=ternary_ccc,
    )


def phq_to_binary(phq_pred: np.ndarray, threshold: float) -> np.ndarray:
    return (_as_1d(phq_pred).astype(np.float64) >= float(threshold)).astype(np.int64)


def phq_to_ternary(phq_pred: np.ndarray, mild_threshold: float, severe_threshold: float) -> np.ndarray:
    if severe_threshold <= mild_threshold:
        raise ValueError("severe_threshold must be greater than mild_threshold")
    phq = _as_1d(phq_pred).astype(np.float64)
    result = np.zeros(phq.shape, dtype=np.int64)
    result[phq >= float(mild_threshold)] = 1
    result[phq >= float(severe_threshold)] = 2
    return result

