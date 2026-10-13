"""Tests for the median beat module.

These tests use synthetic signals only and do not require access to the ECG
data store, so they run anywhere.
"""

import numpy as np
import pytest

from execg.median_beat import RlignMedianBeat, harmonize
from execg.median_beat.harmonize import (
    LEAD_NAMES_12,
    assemble_leads,
    crop_or_pad,
    resample_to,
)
from execg.median_beat.aggregate import nan_trimmed_mean, normalize_beats
from execg.median_beat.base import MedianBeat, MedianBeatExtractor


def _synthetic_ecg(fs=500, seconds=10, rr=1.0, n_leads=12, amp=1.0):
    """Periodic QRS+T 12-lead signal that NeuroKit2 can detect R-peaks on."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    qrs = np.zeros(n)
    twave = np.zeros(n)
    for r in np.arange(0.5, seconds, rr):
        qrs += np.exp(-((t - r) ** 2) / (2 * 0.01**2))
        twave += 0.25 * np.exp(-((t - (r + 0.3)) ** 2) / (2 * 0.04**2))
    beat = amp * (qrs + twave)
    return np.tile(beat, (n_leads, 1))


# =============================================================================
# IO harmonization
# =============================================================================


class TestIOHarmonization:
    def test_resample_changes_length(self):
        sig = np.random.randn(12, 1000)
        out = resample_to(sig, 250, 500)
        assert out.shape == (12, 2000)

    def test_resample_noop_when_equal(self):
        sig = np.random.randn(3, 100)
        assert resample_to(sig, 500, 500) is sig

    def test_fix_length_crop_and_pad(self):
        sig = np.ones((2, 100))
        assert crop_or_pad(sig, 60).shape == (2, 60)
        padded = crop_or_pad(sig, 140)
        assert padded.shape == (2, 140)
        assert np.all(padded[:, 100:] == 0)

    def test_assemble_derives_augmented_leads(self):
        length = 50
        lead_I = np.random.randn(length)
        lead_II = np.random.randn(length)
        data = {"I": lead_I.tolist(), "II": lead_II.tolist()}
        mat, present = assemble_leads(data, LEAD_NAMES_12)
        assert mat.shape == (12, length)
        # III = II - I
        np.testing.assert_allclose(mat[2], lead_II - lead_I)
        # aVR = -(I + II) / 2
        np.testing.assert_allclose(mat[3], -(lead_I + lead_II) / 2)
        assert "I" in present and "V1" not in present

    def test_assemble_nan_fills_missing(self):
        data = {"II": np.random.randn(20).tolist()}
        mat, present = assemble_leads(data, LEAD_NAMES_12)
        assert present == ["II"]
        assert np.isnan(mat[LEAD_NAMES_12.index("V1")]).all()

    def test_harmonize_resamples_and_fixes_length(self):
        sig = np.random.randn(12, 1000)  # 250 Hz, 4 s
        out = harmonize(sig, fs_in=250, target_fs=500, seconds=10)
        assert out.shape == (12, 5000)


# =============================================================================
# Stage 2 aggregation
# =============================================================================


class TestAggregation:
    def test_trimmed_mean_rejects_outlier(self):
        a = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        out = nan_trimmed_mean(a[:, None], trim=0.2, axis=0)
        assert out[0] < 1.0  # outlier trimmed away

    def test_trimmed_mean_handles_nan(self):
        a = np.array([1.0, 1.0, 1.0, np.nan])
        out = nan_trimmed_mean(a[:, None], trim=0.0, axis=0)
        np.testing.assert_allclose(out[0], 1.0)

    def test_median_aggregation_shape(self):
        extractor = RlignMedianBeat(verbose=False)
        beats = np.random.randn(7, 12, 500)
        assert extractor.aggregate_beats(beats).shape == (12, 500)

    def test_per_patient_weighting(self):
        # 4 beats from 2 patients; patient A contributes 3, B contributes 1.
        # Per-patient reduction must weight each patient equally.
        extractor = RlignMedianBeat(per_patient=True, verbose=False)
        beats = np.stack(
            [
                np.full((12, 500), 1.0),
                np.full((12, 500), 1.0),
                np.full((12, 500), 1.0),
                np.full((12, 500), 5.0),
            ]
        )
        sids = ["A", "A", "A", "B"]
        rep = extractor.aggregate_beats(beats, subject_ids=sids)
        # Per-patient medians are 1.0 (A) and 5.0 (B); their median is 3.0.
        np.testing.assert_allclose(rep, 3.0)

    def test_nan_beats_ignored(self):
        extractor = RlignMedianBeat(verbose=False)
        beats = np.stack([np.full((12, 500), 2.0), np.full((12, 500), np.nan)])
        np.testing.assert_allclose(extractor.aggregate_beats(beats), 2.0)

    def test_trimmed_mean_selected_by_aggregate_method(self):
        extractor = RlignMedianBeat(
            aggregate_method="trimmed_mean", trim=0.2, per_patient=False,
            verbose=False,
        )
        beats = np.concatenate(
            [np.full((4, 12, 10), 1.0), np.full((1, 12, 10), 100.0)]
        )
        assert extractor.aggregate_beats(beats).max() < 2.0


# =============================================================================
# scale_method configuration
# =============================================================================


class TestScaleMethods:
    def test_none_beat_length_follows_window_and_fs(self):
        assert RlignMedianBeat(scale_method="none").beat_length == 600
        # The window is in ms, so it must track the sampling rate.
        at_250 = RlignMedianBeat(scale_method="none", target_fs=250)
        assert at_250.beat_length == 300
        custom = RlignMedianBeat(scale_method="none", beat_window_ms=(200.0, 400.0))
        assert custom.beat_length == 300

    def test_warping_beat_length_follows_template_bpm(self):
        assert RlignMedianBeat(scale_method="hrc", template_bpm=60).beat_length == 500
        assert RlignMedianBeat(scale_method="hrc", template_bpm=50).beat_length == 600
        assert RlignMedianBeat(scale_method="linear", template_bpm=60).beat_length == 500

    def test_none_reports_no_template_bpm(self):
        # "none" does not warp the time axis, so no template rate applies.
        assert RlignMedianBeat(scale_method="none", template_bpm=60).template_bpm is None
        assert RlignMedianBeat(scale_method="hrc", template_bpm=60).template_bpm == 60

    def test_identity_is_accepted_as_alias_for_none(self):
        assert RlignMedianBeat(scale_method="identity").template_bpm is None

    def test_unknown_scale_method_rejected(self):
        with pytest.raises(ValueError, match="Unknown scale_method"):
            RlignMedianBeat(scale_method="bogus")


# =============================================================================
# Stage 1 (Rlign) + normalization
# =============================================================================


class TestPerEcgBeats:
    def test_per_ecg_beats_shape_and_finite(self):
        pytest.importorskip("neurokit2")  # vendored rlign needs neurokit2
        extractor = RlignMedianBeat(num_workers=1, verbose=False)
        signals = np.stack([_synthetic_ecg(), _synthetic_ecg(rr=0.8)])
        beats = extractor.per_ecg_beats(signals)
        assert beats.shape == (2, 12, extractor.beat_length)
        # At least one recording should align successfully.
        assert np.isfinite(beats.reshape(2, -1)).all(axis=1).any()

    @pytest.mark.parametrize("scale_method", ["none", "linear", "hrc"])
    def test_build_from_array(self, scale_method):
        pytest.importorskip("neurokit2")  # vendored rlign needs neurokit2
        extractor = RlignMedianBeat(
            scale_method=scale_method, num_workers=1, verbose=False
        )
        signals = np.stack([_synthetic_ecg(), _synthetic_ecg(rr=0.8)])
        beat = extractor.build(signals, subject_ids=["a", "b"], label="t")
        assert isinstance(beat, MedianBeat)
        assert beat.beat.shape == (12, extractor.beat_length)
        assert beat.label == "t"
        assert beat.n_ecgs + beat.n_dropped == 2
        assert beat.sampling_rate == 500.0
        assert beat.lead_names == LEAD_NAMES_12

    def test_flat_signal_is_dropped_not_averaged_in(self):
        """A signal with no detectable R-peaks must not leak into the result."""
        pytest.importorskip("neurokit2")
        extractor = RlignMedianBeat(
            scale_method="none", num_workers=1, per_patient=False, verbose=False
        )
        signals = np.stack([_synthetic_ecg(), np.zeros((12, 5000))])
        beats = extractor.per_ecg_beats(signals)
        assert np.isnan(beats[1]).all()

    def test_keep_per_beat_stores_contributions(self):
        pytest.importorskip("neurokit2")
        extractor = RlignMedianBeat(
            num_workers=1, verbose=False, keep_per_beat=True
        )
        signals = np.stack([_synthetic_ecg(), _synthetic_ecg(rr=0.8)])
        beat = extractor.build(signals)
        assert beat.per_beat is not None
        assert beat.per_beat.shape[0] == beat.n_ecgs

    def test_normalize_unit_scales_peak(self):
        beats = (np.random.randn(3, 12, 500)) * 5.0
        out = normalize_beats(beats, "unit")
        assert np.nanmax(np.abs(out)) <= 1.0 + 1e-6

    def test_build_rejects_bad_ndim(self):
        extractor = RlignMedianBeat(verbose=False)
        with pytest.raises(ValueError, match="n_ecgs, n_leads, length"):
            extractor.build(np.random.randn(12, 500))

    def test_build_rejects_lead_count_mismatch(self):
        extractor = RlignMedianBeat(verbose=False)
        with pytest.raises(ValueError, match="leads"):
            extractor.build(np.random.randn(2, 8, 5000))


# =============================================================================
# Pluggable extractor interface
# =============================================================================


class _ConstantBeat(MedianBeatExtractor):
    """A non-Rlign stage-1: emit a fixed beat. Proves a new algorithm slots in
    behind the same interface without touching stages 0/2 or any dependency."""

    def per_ecg_beats(self, signals):
        return np.full((signals.shape[0], len(self.leads), 50), 2.0)


class TestPluggableInterface:
    def test_custom_extractor_builds(self):
        beat = _ConstantBeat(verbose=False).build(
            np.random.randn(3, 12, 500), subject_ids=["a", "a", "b"], label="g"
        )
        assert beat.beat.shape == (12, 50)
        assert beat.n_ecgs == 3 and beat.n_patients == 2 and beat.label == "g"
        np.testing.assert_allclose(beat.beat, 2.0)

    def test_custom_extractor_has_no_template_bpm(self):
        beat = _ConstantBeat(verbose=False).build(np.random.randn(1, 12, 500))
        assert beat.template_bpm is None

    def test_build_groups_keys_each_group(self):
        groups = {"x": np.random.randn(2, 12, 500), "y": np.random.randn(1, 12, 500)}
        beats = _ConstantBeat(verbose=False).build_groups(groups)
        assert set(beats) == {"x", "y"}
        assert beats["x"].label == "x" and beats["y"].n_ecgs == 1

    def test_per_ecg_beats_is_required(self):
        with pytest.raises(TypeError):
            MedianBeatExtractor()

    def test_all_recordings_dropped_raises_clear_error(self):
        class _AlwaysFails(MedianBeatExtractor):
            def per_ecg_beats(self, signals):
                return np.full((signals.shape[0], len(self.leads), 50), np.nan)

        with pytest.raises(ValueError, match="no usable beats"):
            _AlwaysFails(verbose=False).build(np.random.randn(2, 12, 500))


# =============================================================================
# Visualization
# =============================================================================


class TestVisualization:
    def test_plot_median_beats_returns_figure(self):
        import matplotlib

        matplotlib.use("Agg")
        from execg.visualizer import plot_median_beats

        beats = {
            "a": np.random.randn(12, 500),
            "b": np.random.randn(12, 500),
        }
        fig, axes = plot_median_beats(beats, leads=["II", "V2"])
        assert fig is not None
        assert axes.size >= 2

    def test_plot_accepts_median_beat_objects(self):
        import matplotlib

        matplotlib.use("Agg")
        from execg.visualizer import plot_median_beats

        beat = _ConstantBeat(verbose=False).build(np.random.randn(2, 12, 500))
        fig, _ = plot_median_beats({"g": beat}, leads=["II"])
        assert fig is not None
