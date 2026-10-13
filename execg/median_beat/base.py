"""Median beat: summarize a cohort of ECG recordings into a single beat.

A median beat is built in two stages:

  1. reduce each recording to one per-ECG beat on a common time base
     (:meth:`MedianBeatExtractor.per_ecg_beats`, algorithm-specific);
  2. aggregate those beats across the cohort with a robust per-sample statistic
     (shared, in :mod:`.aggregate`).

Algorithms only implement stage 1 -- e.g. :class:`.rlign.RlignMedianBeat`. New
algorithms can be added as sibling modules without touching the aggregation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from execg.median_beat import aggregate as agg
from execg.median_beat.harmonize import LEAD_NAMES_12


@dataclass
class MedianBeat:
    """A cohort median beat and the metadata describing how it was built.

    Attributes:
        beat: Median beat, shape ``(n_leads, beat_len)``, in mV.
        lead_names: Lead names corresponding to the first axis of ``beat``.
        sampling_rate: Sampling rate of the beat in Hz.
        template_bpm: Heart rate (bpm) the beats were normalized to, or None
            when the algorithm does not warp the time axis.
        n_ecgs: Number of ECG recordings that contributed.
        n_patients: Number of distinct patients that contributed.
        n_dropped: Number of recordings dropped (e.g. R-peak detection failure).
        label: Optional group label (e.g. "hyperkalemia").
        per_beat: Optional per-ECG beats, shape ``(n_ecgs, n_leads, beat_len)``.
        clusters: Optional per-morphology sub-beats.
        info: Free-form extra metadata.
    """

    beat: np.ndarray
    lead_names: List[str]
    sampling_rate: float
    template_bpm: Optional[float]
    n_ecgs: int
    n_patients: int
    n_dropped: int
    label: Optional[str] = None
    per_beat: Optional[np.ndarray] = None
    clusters: Optional[List["MedianBeat"]] = None
    info: Dict = field(default_factory=dict)


class MedianBeatExtractor(ABC):
    """Interface every median-beat algorithm implements.

    Holds the shared configuration and the whole :meth:`build` flow; subclasses
    supply only stage 1 via :meth:`per_ecg_beats`. Keeping one interface lets
    callers swap algorithms freely.
    """

    #: Heart rate the beats are normalized to. Stays None for algorithms that
    #: do not warp the time axis; the ones that do set it in ``__init__``.
    template_bpm: Optional[float] = None

    def __init__(
        self,
        target_fs: float = 500.0,
        seconds: float = 10.0,
        leads: Sequence[str] = LEAD_NAMES_12,
        aggregate_method: str = "median",
        trim: float = 0.1,
        normalize: str = "none",
        per_patient: bool = True,
        cluster: Optional[int] = None,
        keep_per_beat: bool = False,
        verbose: bool = True,
    ):
        """Configure the shared cohort-aggregation behaviour.

        Args:
            target_fs: Sampling rate (Hz) the input signals are expected at.
            seconds: Recording duration (s) of the input signals.
            leads: Output lead order.
            aggregate_method: Cohort statistic, ``"median"`` or
                ``"trimmed_mean"``.
            trim: Tail fraction for ``trimmed_mean``.
            normalize: Per-recording amplitude normalization
                (``"none"`` / ``"zscore"`` / ``"unit"``).
            per_patient: Reduce to one beat per patient before aggregating, so
                patients with many recordings do not dominate.
            cluster: If set, number of morphology clusters (the largest becomes
                the main beat; all are returned in ``clusters``).
            keep_per_beat: Store the per-ECG beats on the result.
            verbose: Print progress.
        """
        self.target_fs = target_fs
        self.seconds = seconds
        self.leads = list(leads)
        self.aggregate_method = aggregate_method
        self.trim = trim
        self.normalize = normalize
        self.per_patient = per_patient
        self.cluster = cluster
        self.keep_per_beat = keep_per_beat
        self.verbose = verbose

    @abstractmethod
    def per_ecg_beats(self, signals: np.ndarray) -> np.ndarray:
        """Reduce each recording to one beat on a common time base.

        Args:
            signals: ``(n_ecgs, n_leads, length)`` in mV.

        Returns:
            Per-ECG beats ``(n_ecgs, n_leads, beat_len)``, row-aligned with the
            input; recordings that could not be processed are all-NaN rows.
        """

    def aggregate_beats(
        self, beats: np.ndarray, subject_ids: Optional[Sequence] = None
    ) -> np.ndarray:
        """Run the shared cohort aggregation (stage 2) on per-ECG beats."""
        beats = agg.normalize_beats(beats, self.normalize)
        if self.per_patient and subject_ids is not None:
            beats = agg.reduce_per_patient(beats, subject_ids)
        return agg.aggregate(beats, self.aggregate_method, self.trim)

    def build(
        self,
        signals: np.ndarray,
        subject_ids: Optional[Sequence] = None,
        label: Optional[str] = None,
    ) -> MedianBeat:
        """Build a median beat from a batch of ECG signals.

        Args:
            signals: ECG array ``(n_ecgs, n_leads, length)`` in mV, harmonized
                to the extractor's sampling rate / length / lead order (see
                :func:`.harmonize.harmonize`).
            subject_ids: Optional per-recording patient ids.
            label: Optional group label stored on the result.
        """
        signals = np.asarray(signals, dtype=float)
        if signals.ndim != 3:
            raise ValueError(
                f"signals must be (n_ecgs, n_leads, length); got {signals.shape}"
            )
        if signals.shape[1] != len(self.leads):
            raise ValueError(
                f"signals has {signals.shape[1]} leads but the extractor is "
                f"configured for {len(self.leads)}: {self.leads}"
            )
        sids = np.asarray(subject_ids) if subject_ids is not None else None
        if sids is not None and len(sids) != signals.shape[0]:
            raise ValueError("subject_ids length must match signals")

        beats = self.per_ecg_beats(signals)

        usable = np.isfinite(beats.reshape(beats.shape[0], -1)).all(axis=1)
        n_dropped = int((~usable).sum())
        if not usable.any():
            raise ValueError(
                f"no usable beats: all {n_dropped} recording(s) failed stage 1 "
                f"({type(self).__name__}.per_ecg_beats returned only NaN rows)"
            )
        usable_beats = beats[usable]
        usable_sids = sids[usable] if sids is not None else None

        beat = self.aggregate_beats(usable_beats, usable_sids)
        n_patients = (
            int(np.unique(usable_sids).size)
            if usable_sids is not None
            else int(usable.sum())
        )

        clusters = self._cluster_beats(usable_beats)
        if clusters:
            beat = clusters[0].beat  # largest cluster is the main beat

        return MedianBeat(
            beat=beat,
            lead_names=self.leads,
            sampling_rate=self.target_fs,
            template_bpm=self.template_bpm,
            n_ecgs=int(usable.sum()),
            n_patients=n_patients,
            n_dropped=n_dropped,
            label=label,
            per_beat=usable_beats if self.keep_per_beat else None,
            clusters=clusters,
        )

    def build_groups(
        self,
        groups: Dict[str, np.ndarray],
        subject_ids: Optional[Dict[str, Sequence]] = None,
    ) -> Dict[str, MedianBeat]:
        """Build a median beat for each named group of signals."""
        out = {}
        for label, signals in groups.items():
            sids = subject_ids.get(label) if subject_ids else None
            out[label] = self.build(signals, subject_ids=sids, label=label)
        return out

    def _cluster_beats(self, beats: np.ndarray) -> Optional[list]:
        """Split beats into morphology clusters (largest first), or None."""
        if not self.cluster or self.cluster < 2:
            return None
        found = agg.cluster_beats(
            beats, self.cluster, self.aggregate_method, self.trim
        )
        return [
            MedianBeat(
                beat=c["beat"], lead_names=self.leads,
                sampling_rate=self.target_fs, template_bpm=self.template_bpm,
                n_ecgs=c["n"], n_patients=c["n"], n_dropped=0,
                label=f"cluster_{i}",
            )
            for i, c in enumerate(found)
        ] or None
