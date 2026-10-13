"""Median beat via the Rlign algorithm.

Stage 1 uses Rlign (Plagwitz et al., 2024, vendored under ``_vendor/rlign``):
R-peak detection, optional heart-rate-corrected (HRC) alignment, and a
per-recording median beat. Stage 2 (cohort aggregation) is inherited from
:class:`MedianBeatExtractor`.

``scale_method`` selects how the time axis is treated:

    ``"none"``    R-aligned fixed window, no time warping -- the traditional
                  median beat, comparable to what ECG carts (GE 12SL, UNI-G)
                  produce. Beat length comes from ``beat_window_ms``.
    ``"linear"``  each RR interval linearly resampled onto the template, which
                  stretches the QRS along with the ST-T segment.
    ``"hrc"``     heart-rate corrected: the QRS core is kept rate-independent
                  while the ST-T region is rescaled per the QT-RR relation.
                  Use this when comparing cohorts at differing heart rates.

References:
    Plagwitz et al., "Rlign ... heart rate-corrected ECG alignment",
    Eur Heart J Digital Health (2025). https://github.com/imi-ms/rlign
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from execg.median_beat.base import MedianBeatExtractor
from execg.median_beat.harmonize import LEAD_NAMES_12

# ExECG names the non-warping mode "none"; Rlign calls it "identity".
_SCALE_METHODS = {
    "none": "identity",
    "identity": "identity",
    "linear": "linear",
    "hrc": "hrc",
}


class RlignMedianBeat(MedianBeatExtractor):
    """Median beat whose per-ECG beats come from the Rlign algorithm.

    Example:
        >>> extractor = RlignMedianBeat(scale_method="none")   # traditional
        >>> beat = extractor.build(signals, subject_ids=sids, label="normal")
        >>> beat.beat.shape  # (12, 600) at 500 Hz with the default window
    """

    def __init__(
        self,
        scale_method: str = "hrc",
        template_bpm: float = 60.0,
        beat_window_ms: Tuple[float, float] = (400.0, 800.0),
        select_lead: str = "II",
        correct_artifacts: bool = True,
        neurokit_method: str = "neurokit",
        num_workers: int = 4,
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
        """Initialize the Rlign median-beat extractor.

        Args:
            scale_method: ``"none"`` (no time warping, traditional median
                beat), ``"linear"``, or ``"hrc"`` (heart-rate corrected).
            template_bpm: Heart rate the beats are warped to. Ignored when
                ``scale_method="none"``; beat length is ``60/template_bpm``
                seconds for the other two.
            beat_window_ms: ``(before, after)`` the R-peak, in ms. Only used
                when ``scale_method="none"``.
            select_lead: Lead used for R-peak detection.
            correct_artifacts: NeuroKit2 ectopic/noisy beat correction.
            neurokit_method: NeuroKit2 R-peak detection method.
            num_workers: Processes used for per-recording alignment.

        Other args (``target_fs``, ``seconds``, ``leads``, ``aggregate_method``,
        ``trim``, ``normalize``, ``per_patient``, ``cluster``, ``keep_per_beat``,
        ``verbose``) configure the shared cohort aggregation -- see
        :class:`MedianBeatExtractor`.
        """
        super().__init__(
            target_fs=target_fs,
            seconds=seconds,
            leads=leads,
            aggregate_method=aggregate_method,
            trim=trim,
            normalize=normalize,
            per_patient=per_patient,
            cluster=cluster,
            keep_per_beat=keep_per_beat,
            verbose=verbose,
        )
        if scale_method not in _SCALE_METHODS:
            raise ValueError(
                f"Unknown scale_method '{scale_method}'; expected one of "
                f"{sorted(set(_SCALE_METHODS))}"
            )
        self.scale_method = scale_method
        self.beat_window_ms = beat_window_ms
        self.select_lead = select_lead
        self.correct_artifacts = correct_artifacts
        self.neurokit_method = neurokit_method
        self.num_workers = num_workers
        # "none" does not warp the time axis, so no template rate applies.
        self.template_bpm = None if self._rlign_scale == "identity" else template_bpm

    @property
    def _rlign_scale(self) -> str:
        return _SCALE_METHODS[self.scale_method]

    @property
    def beat_length(self) -> int:
        """Length in samples of the per-ECG beats this extractor produces."""
        if self._rlign_scale == "identity":
            before, after = self._window_samples()
            return before + after
        return int(round(60.0 * self.target_fs / self.template_bpm))

    def _window_samples(self) -> Tuple[int, int]:
        before, after = self.beat_window_ms
        return (
            int(round(before / 1000.0 * self.target_fs)),
            int(round(after / 1000.0 * self.target_fs)),
        )

    def per_ecg_beats(self, signals: np.ndarray) -> np.ndarray:
        from execg.median_beat._vendor import rlign as _rlign

        select_idx = (
            self.leads.index(self.select_lead)
            if self.select_lead in self.leads
            else 0
        )
        aligner = _rlign.Rlign(
            sampling_rate=int(round(self.target_fs)),
            seconds_len=int(round(self.seconds)),
            scale_method=self._rlign_scale,
            select_lead=select_idx,
            # Rlign always needs a template_bpm internally, even when unused.
            template_bpm=self.template_bpm or 60.0,
            agg_beat="median",
            correct_artifacts=self.correct_artifacts,
            neurokit_method=self.neurokit_method,
            num_workers=self.num_workers,
            identity_window=self._window_samples(),
            silent=not self.verbose,
        )
        beats = np.asarray(aligner.transform(np.nan_to_num(signals)), dtype=float)

        # On failure Rlign substitutes a raw-ECG slice, which is finite and so
        # would survive the aggregation; use its own per-record flags instead.
        failed = np.asarray(aligner.fails, dtype=bool)
        flat = beats.reshape(beats.shape[0], -1)
        failed |= np.all(flat == 0, axis=1) | ~np.isfinite(flat).all(axis=1)
        beats[failed] = np.nan
        return beats
