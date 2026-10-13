# Median Beat

## Overview

The `median_beat` module builds a single representative beat from a cohort of
ECG recordings — anywhere from 1 to N recordings, each possibly with a different
heart rate, sampling rate and lead set. It is designed to avoid the failure
modes of naive cross-recording averaging:

| Failure mode of naive averaging | Cause | How `median_beat` fixes it |
|---|---|---|
| Result is a blurred "smear" when many ECGs are pooled | mean over misaligned / mixed-morphology beats | two-stage pipeline (per-ECG beat first), **align before averaging**, robust **median / trimmed-mean** |
| Every group looks almost identical | grand average converges to the population mean | **per-patient weighting**, optional **morphology clustering** |
| Mixing different heart rates breaks morphology (T-wave drift, QT smear) | fixed window ignores rate dependence | optional **heart-rate-corrected (HRC) alignment** via [Rlign](https://github.com/imi-ms/rlign) |

### Pipeline

```
Stage 0  Harmonize    load each ECG -> mV, resample to common fs, fixed length, 12-lead order
Stage 1  Per-ECG beat R-peak detection (+ optional HRC warping) -> one median beat per recording
Stage 2  Aggregate    (one beat per patient) -> per-sample median / trimmed-mean across the cohort
```

### Structure

Every algorithm implements the same interface (`MedianBeatExtractor`, returning
a `MedianBeat`), so they are interchangeable and new ones slot in without
touching callers. Algorithms only implement stage 1; stages 0 and 2 are shared:

```
median_beat/
  base.py          MedianBeat (output), MedianBeatExtractor (interface + build flow)
  harmonize.py     shared signal harmonization      (stage 0)
  aggregate.py     shared cohort aggregation        (stage 2)
  rlign.py         RlignMedianBeat                  (stage 1)
  _vendor/rlign    vendored Rlign, used by rlign.py
```

## Installation

The module's dependencies are included in the standard installation:

```bash
pip install -e .
```

The alignment library **rlign is vendored in-tree** under
`execg/median_beat/_vendor/rlign` (MIT; see its `VENDOR.md`), so the
feature keeps working even if the package is removed from PyPI. Its runtime
dependencies (numpy, scipy, pandas, neurokit2, scikit-learn) are included in
the core package dependencies.

The core package does not depend on any data store. Source-specific loaders
remain outside the package.

## Choosing `scale_method`

This is the main decision. It controls how stage 1 treats the time axis:

| `scale_method` | Time axis | Beat length | Use when |
|---|---|---|---|
| `"none"` | untouched — R-aligned fixed window | `beat_window_ms`, default 1.2 s | You want the **traditional median beat**, comparable to what ECG carts (GE 12SL, UNI-G) output. |
| `"linear"` | each RR interval linearly resampled | `60/template_bpm` s | Rarely — it stretches the QRS along with the ST-T. |
| `"hrc"` | heart-rate corrected: QRS kept rate-independent, ST-T rescaled per the QT-RR relation | `60/template_bpm` s | Comparing cohorts that **differ in heart rate**, so T-wave position is not confounded by rate. |

`"none"` and `"hrc"` produce different beat lengths and different time axes, so
beats from the two are not directly stackable or comparable sample-by-sample.

## Quick Start

A runnable example using the bundled ECG sample is available
in [example/median_beat.ipynb](../example/median_beat.ipynb).

The core is **array-based and decoupled from any data store**: `build` takes a
harmonized `(n_ecgs, n_leads, length)` array in mV. Use `harmonize` to bring raw
signals (any sampling rate) onto the common length first, and a small
source-specific loader to produce the array.

```python
import numpy as np
from execg.median_beat import RlignMedianBeat, harmonize
from execg.visualizer import plot_median_beats

extractor = RlignMedianBeat(
    scale_method="none",     # traditional median beat, no heart-rate correction
    target_fs=500,           # the signals you pass are expected at this fs
    aggregate_method="median",
    normalize="none",        # keep mV (amplitude is clinically meaningful)
    per_patient=True,        # one vote per patient
)

# signals: (n_ecgs, 12, 5000) in mV, e.g. np.stack of harmonize(raw, fs_in) rows
beat = extractor.build(signals, subject_ids=subject_ids, label="normal")
beat.beat.shape   # (12, 600) -> 12 leads, 1.2 s at 500 Hz
```

For a heart-rate-corrected beat instead:

```python
extractor = RlignMedianBeat(scale_method="hrc", template_bpm=60)
beat = extractor.build(signals, subject_ids=subject_ids, label="normal")
beat.beat.shape   # (12, 500) -> one 60 bpm beat

fig, _ = plot_median_beats({"normal": beat}, sampling_rate=500)
```

### Comparing groups

```python
beats = extractor.build_groups(
    {"hypo": hypo_signals, "normal": normal_signals, "hyper": hyper_signals},
    subject_ids={"hypo": hypo_sids, "normal": normal_sids, "hyper": hyper_sids},
)
fig, _ = plot_median_beats(beats, sampling_rate=500)
```

## API Reference

### `RlignMedianBeat`

Algorithm-specific arguments:

| Argument | Default | Description |
|---|---|---|
| `scale_method` | `"hrc"` | `"none"` (traditional), `"linear"`, or `"hrc"`. See the table above. |
| `template_bpm` | `60` | Heart rate the beats are warped to. Ignored when `scale_method="none"`. |
| `beat_window_ms` | `(400, 800)` | `(before, after)` the R-peak, in ms. Only used when `scale_method="none"`. |
| `select_lead` | `"II"` | Lead used for R-peak detection. |
| `correct_artifacts` | `True` | NeuroKit2 ectopic/noisy beat correction. |
| `neurokit_method` | `"neurokit"` | NeuroKit2 R-peak detection method. |
| `num_workers` | `4` | Processes used for per-recording alignment. |

Shared arguments (defined on `MedianBeatExtractor`, accepted by every algorithm):

| Argument | Default | Description |
|---|---|---|
| `target_fs` | `500` | Sampling rate (Hz) the input signals are expected at. |
| `seconds` | `10` | Duration (s) of the input signals. |
| `leads` | 12-lead | Output lead order. Missing limb-augmented leads are derived from I/II by `assemble_leads`. |
| `aggregate_method` | `"median"` | `"median"` or `"trimmed_mean"`. |
| `trim` | `0.1` | Tail fraction for `trimmed_mean`. |
| `normalize` | `"none"` | `"none"` (mV), `"zscore"`, or `"unit"` (peak-R) per recording. |
| `per_patient` | `True` | Reduce to one beat per patient before cohort aggregation. |
| `cluster` | `None` | If set, number of morphology clusters (the largest becomes the main beat). |
| `keep_per_beat` | `False` | Store the per-ECG beats on the result. |
| `verbose` | `True` | Print progress. |

**Methods**

- `build(signals, subject_ids=None, label=None) -> MedianBeat` — `signals` is
  `(n_ecgs, n_leads, length)` in mV, harmonized to `target_fs`/`seconds`.
- `build_groups(groups, subject_ids=None) -> dict[str, MedianBeat]` — `groups`
  maps label to a signal array.
- `per_ecg_beats(signals) -> np.ndarray` — Stage 1 only.
- `aggregate_beats(beats, subject_ids=None) -> np.ndarray` — Stage 2 only.
- `beat_length -> int` — length in samples of the beats this extractor produces.

**Harmonization helpers** (`execg.median_beat`):
`harmonize(signal, fs_in, target_fs, seconds)` and `assemble_leads(data, leads)`
build the array the core expects from raw signals.

### `MedianBeat`

Dataclass with: `beat` `(n_leads, beat_len)`, `lead_names`, `sampling_rate`,
`template_bpm` (`None` when the algorithm does not warp time), `n_ecgs`,
`n_patients`, `n_dropped`, `label`, `per_beat` (optional), `clusters`
(optional), `info`.

### Adding an algorithm

Subclass `MedianBeatExtractor` and implement stage 1 only:

```python
from execg.median_beat import MedianBeatExtractor

class MyMedianBeat(MedianBeatExtractor):
    def per_ecg_beats(self, signals):
        # (n_ecgs, n_leads, length) -> (n_ecgs, n_leads, beat_len),
        # row-aligned with the input; all-NaN rows for recordings that failed.
        ...
```

`build`, `build_groups` and the cohort aggregation are inherited.

## Scratch experiments: serum potassium

The potassium experiments used to develop this module live under `tmp/`
(git-ignored) in the source ExECG development checkout, together with the MAI
data loader (`tmp/mai_loader.py`, requiring the internal `mai.data` package).
These scratch scripts and datasets are not included in this repository:

- `tmp/representation_beat/potassium_cross_sectional.py` — cross-sectional groups
  (hypo / normal / moderate- and extreme-hyperkalemia), per-patient sampling.
- `tmp/representation_beat/potassium_sameday_cohort.py` — same-day
  acute-derangement cohort.

Findings: the median beats are sharp (no smearing) and hypokalemia shows
consistently **flattened T-waves** across the precordial leads. The hyperkalemic
*peaked T* is a within-patient change relative to each patient's own baseline,
which is swamped by inter-patient variation in a cross-sectional median; a
paired (same-day, same patient) difference recovers a monotonic dose-response in
the T-region (hypo < normal < moderate < extreme hyperkalemia).

`tmp/medianbeat_comparison/` validates the `scale_method="none"` output against
GE 12SL and UNI-G reference median beats on PTB-XL.

## References

- Plagwitz et al., *Rlign — heart rate-corrected ECG alignment*, Eur Heart J
  Digital Health (2025). https://github.com/imi-ms/rlign
- Abboud & Sadeh, *Cross-correlation for ECG alignment and extrasystole
  rejection*, Comput Biomed Res 17:258–266 (1984).
