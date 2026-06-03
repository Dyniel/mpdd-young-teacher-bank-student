from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


TARGET_CHANNELS = 12
EPS = 1e-8


@dataclass(frozen=True)
class FeatureTable:
    ids: np.ndarray
    gait: np.ndarray
    personality: np.ndarray
    label2: np.ndarray | None = None
    label3: np.ndarray | None = None
    phq9: np.ndarray | None = None

    @property
    def has_labels(self) -> bool:
        return self.label2 is not None and self.label3 is not None and self.phq9 is not None


@dataclass(frozen=True)
class RandomConvBank:
    weights: tuple[np.ndarray, ...]
    dilations: tuple[int, ...]
    biases: tuple[float, ...]
    target_length: int


GaitExtractor = Callable[[np.ndarray], np.ndarray]


def read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"ID", "label2", "label3", "PHQ-9"}
    missing = expected.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"label CSV misses columns: {sorted(missing)}")
    return rows


def load_personality_embeddings(path: Path) -> dict[int, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    mapping: dict[int, np.ndarray] = {}
    for item in data:
        person_id = int(item["id"])
        embedding = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
        mapping[person_id] = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
    if not mapping:
        raise ValueError(f"no personality embeddings loaded from {path}")
    return mapping


TRAIT_NAMES = ("extraversion", "agreeableness", "openness", "neuroticism", "conscientiousness")
COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _trait_score(description: str, trait: str) -> float | None:
    escaped = re.escape(trait)
    patterns = (
        rf"{escaped}(?:\s+personality\s+trait)?\s+score(?:\s+is|\s+of)?\s+(\d+(?:\.\d+)?)",
        rf"score(?:s)?\s+of\s+(\d+(?:\.\d+)?)\s+(?:for|in)\s+{escaped}",
        rf"score\s+of\s+(\d+(?:\.\d+)?)\s+on\s+the\s+{escaped}(?:\s+scale)?",
        rf"(\d+(?:\.\d+)?)\s+(?:for|in)\s+{escaped}",
    )
    for pattern in patterns:
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _stress_features(description: str) -> np.ndarray:
    text = description.lower()
    levels = (
        bool(re.search(r"\bno financial stress\b", text)),
        bool(re.search(r"\blow financial stress\b|\bfinancial stress(?: score)? [^.]{0,28}\b(?:low|relatively low)\b", text)),
        bool(re.search(r"\bmoderate financial stress\b", text)),
        bool(re.search(r"\bhigh financial stress\b", text)),
    )
    known = any(levels)
    return np.asarray([*levels, not known], dtype=np.float32)


def _family_features(description: str) -> np.ndarray:
    match = re.search(
        r"\blive(?:s)? with(?: only)?\s+(?:a\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+family member",
        description,
        flags=re.IGNORECASE,
    )
    if not match:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    raw = match.group(1).lower()
    count = float(COUNT_WORDS.get(raw, int(raw) if raw.isdigit() else 0))
    return np.asarray([min(count, 10.0) / 10.0, 0.0], dtype=np.float32)


def extract_structured_description_features(description: str) -> np.ndarray:
    """Extract small numeric context explicitly stated in the released descriptions."""
    scores = [_trait_score(description, trait) for trait in TRAIT_NAMES]
    trait_values = np.asarray([0.0 if score is None else min(score, 20.0) / 20.0 for score in scores], dtype=np.float32)
    trait_missing = np.asarray([score is None for score in scores], dtype=np.float32)
    text = description.lower()
    disease_flags = np.asarray(
        [
            "healthy disease" in text,
            "neurological disease" in text,
            "other disease" in text,
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            trait_values,
            trait_missing,
            _stress_features(description),
            _family_features(description),
            disease_flags,
        ]
    )
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def load_structured_description_features(path: Path) -> dict[int, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"id", "description"}
    missing = expected.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"description CSV misses columns: {sorted(missing)}")
    mapping = {
        int(row["id"]): extract_structured_description_features(row["description"])
        for row in rows
        if row["id"] and row["description"]
    }
    if not mapping:
        raise ValueError(f"no structured description features loaded from {path}")
    return mapping


def find_train_gait(train_root: Path, person_id: int) -> Path:
    path = train_root / "IMU" / "train" / str(person_id) / f"{person_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing train gait for ID={person_id}: {path}")
    return path


def discover_test_gait(test_root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted((test_root / "IMU").glob("*/*.npy")):
        try:
            person_id = int(path.parent.name)
        except ValueError:
            continue
        result[person_id] = path
    if not result:
        raise FileNotFoundError(f"no test IMU files found below {test_root / 'IMU'}")
    return result


def _clean_and_pad_imu(array: np.ndarray, target_channels: int = TARGET_CHANNELS) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"expected a non-empty [T, C] IMU array, got shape={array.shape}")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    keep = min(array.shape[1], target_channels)
    padded = np.zeros((array.shape[0], target_channels), dtype=np.float32)
    padded[:, :keep] = array[:, :keep]
    present = np.zeros(target_channels, dtype=np.float32)
    present[:keep] = 1.0
    return padded, present


def prepare_imu_sequence(
    array: np.ndarray,
    target_length: int,
    keep_channels: int = TARGET_CHANNELS,
    append_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if target_length < 2:
        raise ValueError("target_length must be >= 2")
    if keep_channels < 1 or keep_channels > TARGET_CHANNELS:
        raise ValueError(f"keep_channels must be in [1, {TARGET_CHANNELS}]")
    gait, present = _clean_and_pad_imu(array)
    gait = gait[:, :keep_channels]
    present = present[:keep_channels]
    mean = gait.mean(axis=0, keepdims=True)
    std = gait.std(axis=0, keepdims=True)
    gait = np.clip((gait - mean) / np.where(std < EPS, 1.0, std), -8.0, 8.0)
    source = np.linspace(0.0, 1.0, gait.shape[0], dtype=np.float64)
    target = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    resampled = np.stack([np.interp(target, source, gait[:, idx]) for idx in range(gait.shape[1])], axis=1)
    resampled = resampled.astype(np.float32)
    if append_mask:
        mask = np.broadcast_to(present[None, :], resampled.shape)
        resampled = np.concatenate([resampled, mask], axis=1)
    return resampled, present.astype(np.float32, copy=False)


def _spectral_features(channel: np.ndarray) -> list[float]:
    centered = channel.astype(np.float64) - float(np.mean(channel))
    power = np.square(np.abs(np.fft.rfft(centered)))
    if power.size > 0:
        power[0] = 0.0
    total = float(power.sum())
    if total <= EPS or power.size <= 1:
        return [0.0] * 6
    frequencies = np.linspace(0.0, 1.0, power.size, dtype=np.float64)
    chunks = np.array_split(power[1:], 3)
    fractions = [float(chunk.sum() / total) if chunk.size else 0.0 for chunk in chunks]
    peak_index = int(np.argmax(power))
    return [
        float(np.sum(frequencies * power) / total),
        float(frequencies[peak_index]),
        float(power[peak_index] / total),
        *fractions,
    ]


def _lag_features(channel: np.ndarray) -> list[float]:
    centered = channel.astype(np.float64) - float(np.mean(channel))
    denom = float(np.dot(centered, centered))
    if denom <= EPS:
        return [0.0] * 5
    features = []
    for lag in (1, 2, 4, 8, 16):
        if centered.size <= lag:
            features.append(0.0)
        else:
            features.append(float(np.dot(centered[:-lag], centered[lag:]) / denom))
    return features


def extract_gait_features(array: np.ndarray, target_channels: int = TARGET_CHANNELS) -> np.ndarray:
    gait, present = _clean_and_pad_imu(array, target_channels)
    length = gait.shape[0]
    quantiles = np.quantile(gait, (0.05, 0.25, 0.5, 0.75, 0.95), axis=0)
    abs_gait = np.abs(gait)
    base = [
        gait.mean(axis=0),
        gait.std(axis=0),
        gait.min(axis=0),
        gait.max(axis=0),
        abs_gait.mean(axis=0),
        np.sqrt(np.square(gait).mean(axis=0)),
        *list(quantiles),
    ]

    deltas = np.diff(gait, axis=0)
    if deltas.size == 0:
        delta_summary = [np.zeros(target_channels, dtype=np.float32)] * 3
    else:
        delta_summary = [
            np.abs(deltas).mean(axis=0),
            deltas.std(axis=0),
            np.max(np.abs(deltas), axis=0),
        ]

    windows = [window for window in np.array_split(gait, min(8, length)) if window.size]
    window_means = np.stack([window.mean(axis=0) for window in windows])
    window_stds = np.stack([window.std(axis=0) for window in windows])
    window_summary = [
        window_means.std(axis=0),
        np.median(window_means, axis=0),
        window_stds.mean(axis=0),
        window_stds.std(axis=0),
    ]

    scaled = gait - gait.mean(axis=0, keepdims=True)
    std = gait.std(axis=0, keepdims=True)
    scaled = scaled / np.where(std < EPS, 1.0, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(scaled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr_upper = corr[np.triu_indices(target_channels, k=1)]

    spectral = []
    lags = []
    for channel_idx in range(target_channels):
        spectral.extend(_spectral_features(gait[:, channel_idx]))
        lags.extend(_lag_features(gait[:, channel_idx]))

    extras = np.asarray(
        [
            float(length),
            float(np.log1p(length)),
            float(present.sum()),
            float(np.mean(np.linalg.norm(gait[:, : min(3, target_channels)], axis=1))),
            float(np.std(np.linalg.norm(gait[:, : min(3, target_channels)], axis=1))),
        ],
        dtype=np.float32,
    )
    feature_blocks: Iterable[np.ndarray] = [
        *base,
        *delta_summary,
        *window_summary,
        corr_upper.astype(np.float32),
        np.asarray(spectral, dtype=np.float32),
        np.asarray(lags, dtype=np.float32),
        present,
        extras,
    ]
    features = np.concatenate([block.reshape(-1) for block in feature_blocks]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _segment_window_features(window: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(window, (0.1, 0.5, 0.9), axis=0)
    deltas = np.diff(window, axis=0)
    if deltas.size:
        delta_abs = np.abs(deltas).mean(axis=0)
        jerk_std = deltas.std(axis=0)
    else:
        delta_abs = np.zeros(window.shape[1], dtype=np.float32)
        jerk_std = np.zeros(window.shape[1], dtype=np.float32)
    blocks = [
        window.mean(axis=0),
        window.std(axis=0),
        np.sqrt(np.square(window).mean(axis=0)),
        np.abs(window).mean(axis=0),
        window.max(axis=0) - window.min(axis=0),
        delta_abs,
        jerk_std,
        *list(quantiles),
    ]
    spectral = []
    lags = []
    for channel_idx in range(window.shape[1]):
        spectral.extend(_spectral_features(window[:, channel_idx])[:3])
        lags.extend(_lag_features(window[:, channel_idx])[:3])
    return np.concatenate(
        [
            *[block.reshape(-1) for block in blocks],
            np.asarray(spectral, dtype=np.float32),
            np.asarray(lags, dtype=np.float32),
        ]
    ).astype(np.float32)


def extract_segment_gait_features(
    array: np.ndarray,
    target_channels: int = TARGET_CHANNELS,
    window_counts: tuple[int, ...] = (4, 8, 16),
) -> np.ndarray:
    gait, present = _clean_and_pad_imu(array, target_channels)
    all_blocks = []
    for window_count in window_counts:
        windows = [window for window in np.array_split(gait, min(window_count, gait.shape[0])) if window.size]
        per_window = np.stack([_segment_window_features(window) for window in windows])
        all_blocks.extend(
            [
                per_window.mean(axis=0),
                per_window.std(axis=0),
                np.quantile(per_window, 0.25, axis=0),
                np.quantile(per_window, 0.75, axis=0),
                per_window.min(axis=0),
                per_window.max(axis=0),
            ]
        )
    extras = np.asarray([gait.shape[0], np.log1p(gait.shape[0]), present.sum()], dtype=np.float32)
    features = np.concatenate([*[block.reshape(-1) for block in all_blocks], present, extras]).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def build_random_conv_bank(
    seed: int,
    n_kernels: int = 256,
    target_length: int = 256,
    target_channels: int = TARGET_CHANNELS,
) -> RandomConvBank:
    if n_kernels < 1:
        raise ValueError("n_kernels must be >= 1")
    rng = np.random.default_rng(seed)
    weights = []
    dilations = []
    biases = []
    for _ in range(n_kernels):
        length = int(rng.choice((5, 7, 9)))
        dilation = int(rng.choice((1, 2, 4, 8, 16)))
        while (length - 1) * dilation + 1 >= target_length:
            dilation = max(1, dilation // 2)
        selected_count = int(rng.integers(1, min(4, target_channels) + 1))
        selected = rng.choice(target_channels, size=selected_count, replace=False)
        kernel = np.zeros((target_channels, length), dtype=np.float32)
        raw = rng.normal(size=(selected_count, length)).astype(np.float32)
        raw = raw - raw.mean(axis=1, keepdims=True)
        raw = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), EPS)
        kernel[selected] = raw / np.sqrt(float(selected_count))
        weights.append(kernel)
        dilations.append(dilation)
        biases.append(float(rng.uniform(-1.0, 1.0)))
    return RandomConvBank(tuple(weights), tuple(dilations), tuple(biases), target_length)


def extract_random_conv_features(array: np.ndarray, bank: RandomConvBank) -> np.ndarray:
    gait, present = prepare_imu_sequence(array, target_length=bank.target_length)
    features = []
    for weights, dilation, bias in zip(bank.weights, bank.dilations, bank.biases, strict=True):
        length = weights.shape[1]
        output_length = gait.shape[0] - (length - 1) * dilation
        response = np.zeros(output_length, dtype=np.float32)
        for offset in range(length):
            response += gait[offset * dilation : offset * dilation + output_length] @ weights[:, offset]
        response += bias
        features.extend(
            [
                float(response.max()),
                float(response.mean()),
                float(response.std()),
                float(np.mean(response > 0.0)),
            ]
        )
    extras = [float(array.shape[0]), float(np.log1p(array.shape[0])), *present.tolist()]
    return np.nan_to_num(np.asarray([*features, *extras], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def gait_extractor_for_bank(
    bank: str,
    seed: int,
    random_kernels: int = 256,
    random_target_length: int = 256,
) -> GaitExtractor:
    if bank == "base":
        return extract_gait_features
    if bank == "segment":
        return extract_segment_gait_features
    if bank == "rocket":
        random_bank = build_random_conv_bank(seed, random_kernels, random_target_length)
        return lambda array: extract_random_conv_features(array, random_bank)
    raise ValueError(f"unknown gait bank: {bank}")


def _personality_vector(mapping: dict[int, np.ndarray], person_id: int, dim: int) -> np.ndarray:
    vector = mapping.get(person_id)
    if vector is None:
        return np.zeros(dim, dtype=np.float32)
    if vector.size != dim:
        raise ValueError(f"personality embedding ID={person_id} has dim={vector.size}, expected {dim}")
    return vector.astype(np.float32, copy=False)


def build_train_table(
    train_root: Path,
    label_csv: Path,
    personality_npy: Path,
    gait_extractor: GaitExtractor = extract_gait_features,
) -> FeatureTable:
    rows = read_label_rows(label_csv)
    personality = load_personality_embeddings(personality_npy)
    personality_dim = next(iter(personality.values())).size
    ids = []
    gait_features = []
    personality_features = []
    label2 = []
    label3 = []
    phq9 = []
    for row in sorted(rows, key=lambda item: int(item["ID"])):
        person_id = int(row["ID"])
        ids.append(person_id)
        gait_features.append(gait_extractor(np.load(find_train_gait(train_root, person_id))))
        personality_features.append(_personality_vector(personality, person_id, personality_dim))
        label2.append(int(float(row["label2"])))
        label3.append(int(float(row["label3"])))
        phq9.append(float(row["PHQ-9"]))
    return FeatureTable(
        ids=np.asarray(ids, dtype=np.int64),
        gait=np.stack(gait_features),
        personality=np.stack(personality_features),
        label2=np.asarray(label2, dtype=np.int64),
        label3=np.asarray(label3, dtype=np.int64),
        phq9=np.asarray(phq9, dtype=np.float64),
    )


def build_test_table(
    test_root: Path,
    personality_npy: Path,
    gait_extractor: GaitExtractor = extract_gait_features,
) -> FeatureTable:
    personality = load_personality_embeddings(personality_npy)
    personality_dim = next(iter(personality.values())).size
    gait_paths = discover_test_gait(test_root)
    ids = sorted(gait_paths)
    return FeatureTable(
        ids=np.asarray(ids, dtype=np.int64),
        gait=np.stack([gait_extractor(np.load(gait_paths[person_id])) for person_id in ids]),
        personality=np.stack([_personality_vector(personality, person_id, personality_dim) for person_id in ids]),
    )
