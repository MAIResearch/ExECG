"""Cohort aggregation shared by median-beat algorithms.

Given per-ECG beats (already on a common time base), these helpers reduce them
to one beat robustly: optional per-recording amplitude normalization,
one-vote-per-patient reduction, a robust per-sample statistic (median / trimmed
mean), and optional morphology clustering. They are algorithm agnostic, so any
extractor can reuse them.
"""

from typing import List, Optional, Sequence

import numpy as np


def normalize_beats(beats: np.ndarray, method: str) -> np.ndarray:
    """Per-recording, per-lead amplitude normalization.

    Args:
        beats: ``(n_ecgs, n_leads, beat_len)``.
        method: ``"none"``, ``"zscore"``, or ``"unit"`` (peak-R).
    """
    if method == "none":
        return beats
    out = beats.copy()
    for rec_idx in range(out.shape[0]):
        for lead_idx in range(out.shape[1]):
            lead = out[rec_idx, lead_idx]
            if not np.isfinite(lead).any():
                continue
            if method == "zscore":
                mean, std = np.nanmean(lead), np.nanstd(lead)
                if std > 0:
                    out[rec_idx, lead_idx] = (lead - mean) / std
            elif method == "unit":
                peak = np.nanmax(np.abs(lead))
                if peak > 0:
                    out[rec_idx, lead_idx] = lead / peak
    return out


def reduce_per_patient(beats: np.ndarray, subject_ids: Sequence) -> np.ndarray:
    """Collapse beats to one (median) beat per patient."""
    subject_ids = np.asarray(subject_ids)
    patient_beats = [
        np.nanmedian(beats[subject_ids == sid], axis=0)
        for sid in np.unique(subject_ids)
    ]
    return np.stack(patient_beats)


def aggregate(beats: np.ndarray, method: str = "median", trim: float = 0.1) -> np.ndarray:
    """Robust per-sample aggregation across the first axis."""
    if method == "median":
        return np.nanmedian(beats, axis=0)
    if method == "trimmed_mean":
        return nan_trimmed_mean(beats, trim, axis=0)
    raise ValueError(f"Unknown aggregate method '{method}'")


def nan_trimmed_mean(data: np.ndarray, trim: float, axis: int = 0) -> np.ndarray:
    """NaN-aware trimmed mean along ``axis``.

    Sorts along ``axis`` (NaNs sort to the end), then averages the central
    ``1 - 2*trim`` fraction of the values per position.
    """
    data = np.moveaxis(data, axis, 0)
    n = data.shape[0]
    lo = int(np.floor(n * trim))
    hi = max(int(np.ceil(n * (1 - trim))), lo + 1)
    ordered = np.sort(data, axis=0)
    return np.nanmean(ordered[lo:hi], axis=0)


def cluster_beats(
    beats: np.ndarray, k: int, aggregate_method: str = "median", trim: float = 0.1
) -> List[dict]:
    """Cluster per-ECG beats by morphology and aggregate each cluster.

    Returns a list of ``{"beat", "n"}`` dicts, largest cluster first. Empty if
    fewer than two usable beats.
    """
    from sklearn.cluster import KMeans

    valid_mask = np.isfinite(beats.reshape(beats.shape[0], -1)).all(axis=1)
    features = beats[valid_mask].reshape(valid_mask.sum(), -1)
    k = min(k, features.shape[0])
    if k < 2:
        return []
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(features)
    valid_indices = np.where(valid_mask)[0]
    clusters = []
    for cluster_id in range(k):
        members = valid_indices[labels == cluster_id]
        clusters.append(
            {"beat": aggregate(beats[members], aggregate_method, trim), "n": len(members)}
        )
    clusters.sort(key=lambda c: c["n"], reverse=True)
    return clusters
