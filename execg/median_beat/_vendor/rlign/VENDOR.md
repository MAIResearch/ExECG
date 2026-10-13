# Vendored: rlign

Vendored copy of the `rlign` heart-rate-corrected ECG alignment library, kept
in-tree so the median_beat feature does not depend on the package staying
available on PyPI.

- Upstream: https://github.com/imi-ms/rlign
- Commit: `1cbbb7b111319f60cebd92fcd90736a228649f6a`
- License: MIT (see `LICENSE.txt`; © 2024 Lucas Bickmann, Lucas Plagwitz)
- Paper: Plagwitz et al., *Rlign — heart rate-corrected ECG alignment*,
  Eur Heart J Digital Health (2025).

## Local modifications

The internal imports were changed from absolute to relative so the package
works under `execg.median_beat._vendor.rlign` instead of top-level `rlign`:

- `__init__.py`: `from rlign.rlign import Rlign` -> `from .rlign import Rlign`,
  `from rlign.utils import Template` -> `from .utils import Template`
- `rlign.py`: `from rlign.utils import (...)` -> `from .utils import (...)`

Additionally, the already-deprecated `median_beat` constructor argument (and its
deprecation-warning branch) was removed from `Rlign.__init__`; use `agg_beat`
instead (which is what this codebase already does).

Bug fix in `rlign.py` `_normalize_interval`: the `scale_method="identity"` branch
returned a bare array while every other branch (and `transform`'s
`res[0], res[1]` unpacking) expects an `(array, fail_flag)` tuple, so identity
output collapsed to `(1, 600)` (lead 0 mistaken for the whole beat). It now returns
`normalized_ecg, 0` and guards the no-beat case with an all-NaN beat + fail flag.
This is what makes `RlignMedianBeat(scale_method="none")` (traditional,
non-HR-corrected median beat) usable.

Two further fixes, both no-ops at the default configuration:

- The `identity` beat window was hard-coded at `rpeak-200 : rpeak+400` samples,
  i.e. correct only at 500 Hz. It is now derived from `sampling_rate` (400 ms
  before / 800 ms after the R-peak, reproducing 200/400 at 500 Hz) and can be
  overridden with the new `identity_window` constructor argument, which
  `RlignMedianBeat` drives from its `beat_window_ms`. The two fallback paths in
  `_template_transform` follow the same window instead of a literal `600`.
- `_template_transform` called `find_rpeaks(ecg_lead, self.sampling_rate)`
  without forwarding `neurokit_method` / `correct_artifacts`, so both
  constructor arguments were silently inert. They are now passed through; the
  defaults match `find_rpeaks`'s own, so default output is unchanged.

No other logic was changed. Runtime deps (numpy, scipy, neurokit2, scikit-learn,
pandas) are declared in the core dependencies of `execg`.

## To update

Re-clone upstream, copy `rlign/*.py` + `LICENSE.txt` here, re-apply the relative
import edits above, and bump the commit hash.
