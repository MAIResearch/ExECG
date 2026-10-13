"""Signal harmonization utilities for median beat construction.

Data-source agnostic: brings a raw ECG array (any sampling rate, any subset of
leads) onto the common representation the builder expects -- fixed sampling rate,
fixed length, a known lead order, in mV. Loading from a specific store (e.g. MAI
object ids) belongs in a thin source-specific wrapper.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

# Must match execg.visualizer.ecg_chart.LEAD_NAMES_12.
LEAD_NAMES_12 = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def assemble_leads(
    data: Dict[str, List[float]], leads: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Stack raw lead signals into a matrix following ``leads`` order.

    Missing limb-augmented leads (III, aVR, aVL, aVF) are derived from I and II
    via Einthoven/Goldberger relations when those base leads are available.
    Leads that are still missing are filled with NaN.

    Args:
        data: Mapping from lead name to its raw sample list.
        leads: Desired output lead order.

    Returns:
        Tuple of (matrix of shape ``(len(leads), length)``, list of lead names
        that are genuinely present, i.e. not NaN-filled).
    """
    length = len(next(iter(data.values())))
    present: List[str] = []

    def lead_signal(name: str) -> Optional[np.ndarray]:
        if name in data and data[name] is not None:
            values = np.asarray(data[name], dtype=float)
            if values.size == length:
                return values
        return None

    lead_i = lead_signal("I")
    lead_ii = lead_signal("II")
    derivable = {}
    if lead_i is not None and lead_ii is not None:
        derivable = {
            "III": lead_ii - lead_i,
            "aVR": -(lead_i + lead_ii) / 2.0,
            "aVL": lead_i - lead_ii / 2.0,
            "aVF": lead_ii - lead_i / 2.0,
        }

    rows = []
    for name in leads:
        signal = lead_signal(name)
        if signal is None and name in derivable:
            signal = derivable[name]
        if signal is None:
            rows.append(np.full(length, np.nan))
        else:
            rows.append(signal)
            present.append(name)

    return np.vstack(rows), present


def resample_to(signal: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample a ``(n_leads, length)`` signal from ``fs_in`` to ``fs_out`` Hz."""
    if fs_in == fs_out:
        return signal
    out_length = int(round(signal.shape[1] * fs_out / fs_in))
    out = np.empty((signal.shape[0], out_length))
    gcd = np.gcd(int(round(fs_in)), int(round(fs_out)))
    up = int(round(fs_out)) // gcd
    down = int(round(fs_in)) // gcd
    for i, lead in enumerate(signal):
        # All-NaN (absent) leads must stay NaN; resample the rest.
        if np.all(np.isnan(lead)):
            out[i] = np.nan
            continue
        resampled = resample_poly(np.nan_to_num(lead), up, down)
        out[i] = (
            resampled[:out_length]
            if resampled.shape[0] >= out_length
            else np.pad(resampled, (0, out_length - resampled.shape[0]))
        )
    return out


def crop_or_pad(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or zero-pad the time axis of ``(n_leads, length)``."""
    length = signal.shape[1]
    if length == target_len:
        return signal
    if length > target_len:
        start = (length - target_len) // 2
        return signal[:, start : start + target_len]
    return np.pad(signal, ((0, 0), (0, target_len - length)))


def harmonize(
    signal: np.ndarray,
    fs_in: float,
    target_fs: float = 500.0,
    seconds: float = 10.0,
) -> np.ndarray:
    """Resample a ``(n_leads, length)`` signal and fix it to a common length.

    Args:
        signal: ECG matrix, shape ``(n_leads, length)``.
        fs_in: Sampling rate of ``signal`` in Hz.
        target_fs: Target sampling rate in Hz.
        seconds: Target duration in seconds (output length = round(seconds*fs)).

    Returns:
        Harmonized signal of shape ``(n_leads, round(seconds * target_fs))``.
    """
    signal = resample_to(np.asarray(signal, dtype=float), fs_in, target_fs)
    return crop_or_pad(signal, int(round(seconds * target_fs)))
