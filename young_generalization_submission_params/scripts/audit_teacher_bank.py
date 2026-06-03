"""Audit stability of a teacher prediction bank.

The audit is intentionally lightweight: it reads prior prediction artifacts,
builds a consensus teacher, then reports agreement margins, PHQ dispersion,
leave-one-teacher-out flips, and bootstrap consensus variability. It does not
need external labels.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from distill_prediction_bank import _read_submission, _weighted_mode


def _load_bank(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids_ref: np.ndarray | None = None
    phq_rows = []
    binary_rows = []
    ternary_rows = []
    for path in paths:
        ids, phq, binary, ternary = _read_submission(path)
        if ids_ref is None:
            ids_ref = ids
        elif not np.array_equal(ids_ref, ids):
            raise ValueError(f"prediction IDs do not match for {path}")
        phq_rows.append(phq)
        binary_rows.append(binary)
        ternary_rows.append(ternary)
    if ids_ref is None:
        raise ValueError("empty teacher bank")
    return ids_ref, np.stack(phq_rows), np.stack(binary_rows), np.stack(ternary_rows)


def _vote_margin(labels: np.ndarray, weights: np.ndarray, choices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    margins = []
    consensus = []
    norm = weights / weights.sum()
    for col in labels.T:
        scores = defaultdict(float)
        for value, weight in zip(col, norm, strict=True):
            scores[int(round(value))] += float(weight)
        ranked = sorted((scores[label], label) for label in choices)
        consensus.append(int(ranked[-1][1]))
        second = ranked[-2][0] if len(ranked) > 1 else 0.0
        margins.append(float(ranked[-1][0] - second))
    return np.asarray(consensus, dtype=np.int64), np.asarray(margins, dtype=np.float64)


def _consensus(
    phq: np.ndarray,
    binary: np.ndarray,
    ternary: np.ndarray,
    weights: np.ndarray,
    phq_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = weights / weights.sum()
    if phq_mode == "median":
        phq_c = np.median(phq, axis=0)
    elif phq_mode == "mean":
        phq_c = np.sum(phq * norm[:, None], axis=0)
    else:
        raise ValueError(phq_mode)
    binary_c, _ = _vote_margin(binary, weights, (0, 1))
    ternary_c, _ = _vote_margin(ternary, weights, (0, 1, 2))
    return phq_c, binary_c, ternary_c


def _leave_one_out(
    phq: np.ndarray,
    binary: np.ndarray,
    ternary: np.ndarray,
    weights: np.ndarray,
    phq_mode: str,
) -> dict[str, Any]:
    full_phq, full_binary, full_ternary = _consensus(phq, binary, ternary, weights, phq_mode)
    rows = []
    for idx in range(weights.size):
        keep = np.ones(weights.size, dtype=bool)
        keep[idx] = False
        if not np.any(keep):
            continue
        loo_phq, loo_binary, loo_ternary = _consensus(phq[keep], binary[keep], ternary[keep], weights[keep], phq_mode)
        rows.append(
            {
                "teacher_index": idx,
                "binary_flip_count": int(np.sum(loo_binary != full_binary)),
                "ternary_flip_count": int(np.sum(loo_ternary != full_ternary)),
                "phq_max_abs_delta": float(np.max(np.abs(loo_phq - full_phq))),
                "phq_mean_abs_delta": float(np.mean(np.abs(loo_phq - full_phq))),
            }
        )
    return {
        "rows": rows,
        "max_binary_flip_count": int(max((row["binary_flip_count"] for row in rows), default=0)),
        "max_ternary_flip_count": int(max((row["ternary_flip_count"] for row in rows), default=0)),
        "max_phq_mean_abs_delta": float(max((row["phq_mean_abs_delta"] for row in rows), default=0.0)),
    }


def _bootstrap(
    phq: np.ndarray,
    binary: np.ndarray,
    ternary: np.ndarray,
    weights: np.ndarray,
    phq_mode: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    full_phq, full_binary, full_ternary = _consensus(phq, binary, ternary, weights, phq_mode)
    binary_flips = []
    ternary_flips = []
    phq_mean_delta = []
    for _ in range(repeats):
        sample = rng.integers(0, weights.size, size=weights.size)
        boot_phq, boot_binary, boot_ternary = _consensus(phq[sample], binary[sample], ternary[sample], weights[sample], phq_mode)
        binary_flips.append(int(np.sum(boot_binary != full_binary)))
        ternary_flips.append(int(np.sum(boot_ternary != full_ternary)))
        phq_mean_delta.append(float(np.mean(np.abs(boot_phq - full_phq))))
    return {
        "repeats": int(repeats),
        "binary_flip_mean": float(np.mean(binary_flips)),
        "binary_flip_std": float(np.std(binary_flips)),
        "binary_flip_max": int(np.max(binary_flips)),
        "ternary_flip_mean": float(np.mean(ternary_flips)),
        "ternary_flip_std": float(np.std(ternary_flips)),
        "ternary_flip_max": int(np.max(ternary_flips)),
        "phq_mean_abs_delta_mean": float(np.mean(phq_mean_delta)),
        "phq_mean_abs_delta_std": float(np.std(phq_mean_delta)),
        "phq_mean_abs_delta_max": float(np.max(phq_mean_delta)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Prior prediction dirs or submission.zip files.")
    parser.add_argument("--weights", default="", help="Optional comma/space separated weights, one per input.")
    parser.add_argument("--phq-mode", choices=("mean", "median"), default="mean")
    parser.add_argument("--bootstrap-repeats", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--out-dir", default="runs/teacher_bank_audit")
    args = parser.parse_args()

    paths = [Path(item).resolve() for item in args.inputs]
    weights = (
        np.asarray([float(item) for item in args.weights.replace(",", " ").split() if item], dtype=np.float64)
        if args.weights
        else np.ones(len(paths), dtype=np.float64)
    )
    if weights.size != len(paths):
        raise ValueError(f"expected {len(paths)} weights, got {weights.size}")
    ids, phq, binary, ternary = _load_bank(paths)
    consensus_phq, consensus_binary, consensus_ternary = _consensus(phq, binary, ternary, weights, args.phq_mode)
    _, binary_margin = _vote_margin(binary, weights, (0, 1))
    _, ternary_margin = _vote_margin(ternary, weights, (0, 1, 2))
    phq_std = np.std(phq, axis=0)
    summary = {
        "teacher_count": len(paths),
        "subject_count": int(ids.size),
        "phq_mode": args.phq_mode,
        "binary_margin_mean": float(np.mean(binary_margin)),
        "binary_margin_min": float(np.min(binary_margin)),
        "ternary_margin_mean": float(np.mean(ternary_margin)),
        "ternary_margin_min": float(np.min(ternary_margin)),
        "phq_teacher_std_mean": float(np.mean(phq_std)),
        "phq_teacher_std_max": float(np.max(phq_std)),
        "leave_one_teacher_out": _leave_one_out(phq, binary, ternary, weights, args.phq_mode),
        "bootstrap": _bootstrap(phq, binary, ternary, weights, args.phq_mode, args.bootstrap_repeats, args.seed),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "teacher_bank_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "teacher_consensus_predictions.npz",
        ids=ids,
        phq9_pred=consensus_phq,
        binary_pred=consensus_binary,
        ternary_pred=consensus_ternary,
        binary_margin=binary_margin,
        ternary_margin=ternary_margin,
        phq_teacher_std=phq_std,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
