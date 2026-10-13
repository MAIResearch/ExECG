"""Visualization for median beats.

Provides an overlay comparison of median beats across groups, designed
to make morphological differences (e.g. peaked T-waves in hyperkalemia)
immediately visible across the standard 12 leads.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from execg.median_beat.harmonize import LEAD_NAMES_12


def _as_beat_array(beat) -> np.ndarray:
    """Accept a MedianBeat or a raw array and return ``(n_leads, len)``."""
    arr = getattr(beat, "beat", beat)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _percentile_band(beat, pct: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Central ``pct`` band across the per-ECG beats, if they are available.

    Returns (low, high) each shaped ``(n_leads, len)``, or None when the item
    carries no per-beat distribution.
    """
    per_beat = getattr(beat, "per_beat", None)
    if per_beat is None:
        return None
    per_beat = np.asarray(per_beat)
    lo = (100.0 - pct) / 2.0
    return (
        np.nanpercentile(per_beat, lo, axis=0),
        np.nanpercentile(per_beat, 100.0 - lo, axis=0),
    )


def plot_median_beats(
    beats: Union[Dict[str, object], Sequence[object]],
    labels: Optional[Sequence[str]] = None,
    sampling_rate: float = 500.0,
    lead_names: Sequence[str] = LEAD_NAMES_12,
    leads: Optional[Sequence[Union[int, str]]] = None,
    colors: Optional[Sequence] = None,
    title: str = "Median Beat Comparison",
    ncols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    linewidth: float = 1.3,
    alpha: float = 0.8,
    xlim_ms: Optional[Tuple[float, float]] = None,
    min_yrange: float = 0.4,
    align_baseline: bool = True,
    baseline_ms: float = 40.0,
    show_band: bool = False,
    band_pct: float = 95.0,
    bands: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """Overlay median beats from several groups across leads.

    Args:
        beats: Either a mapping ``{label: MedianBeat|array}`` or a
            sequence of beats (then ``labels`` must be given). Each beat is a
            ``MedianBeat`` or an array of shape ``(n_leads, beat_len)``.
        labels: Group labels (required if ``beats`` is a sequence).
        sampling_rate: Sampling rate of the beats in Hz.
        lead_names: Names for each lead row in the beat arrays.
        leads: Which leads to plot, as indices or names. Defaults to all leads.
        colors: One color per group.
        title: Figure title.
        ncols: Number of subplot columns.
        figsize: Figure size; auto-computed if None.
        linewidth: Line width for the beats.
        alpha: Line opacity (<1 keeps overlapping curves readable).
        xlim_ms: Optional ``(start_ms, end_ms)`` to zoom the time axis.
        min_yrange: Minimum y-axis span (mV) per panel, so low-voltage leads are
            not auto-zoomed to the point of amplifying noise. 0 disables.
        align_baseline: Subtract each curve's baseline (mean of the first
            ``baseline_ms``) so all groups start from a common zero.
        baseline_ms: Window length (ms) used for the baseline estimate.
        show_band: Shade a ``band_pct`` band per group (needs per-ECG beats on a
            ``MedianBeat`` or an explicit ``bands`` argument).
        band_pct: Central percentage of the beat distribution to shade.
        bands: Optional explicit ``{label: (low, high)}`` bands, each
            ``(n_leads, len)``; overrides auto bands from ``per_beat``.
        save_path: If given, save the figure here.
        dpi: DPI for the saved figure.

    Returns:
        Tuple of (fig, axes).
    """
    if isinstance(beats, dict):
        labels = list(beats.keys())
        items = [beats[k] for k in labels]
    else:
        items = list(beats)
        if labels is None:
            labels = [f"group {i}" for i in range(len(items))]

    beat_list = [_as_beat_array(it) for it in items]
    if bands is not None:
        band_list = [bands.get(lbl) for lbl in labels]
    elif show_band:
        band_list = [_percentile_band(it, band_pct) for it in items]
    else:
        band_list = [None] * len(items)

    baseline_n = max(1, int(baseline_ms / 1000.0 * sampling_rate))
    if align_baseline:
        for i, beat in enumerate(beat_list):
            offset = beat[:, :baseline_n].mean(axis=1, keepdims=True)
            beat_list[i] = beat - offset
            if band_list[i] is not None:
                lo, hi = band_list[i]
                band_list[i] = (lo - offset, hi - offset)

    lead_names = list(lead_names)
    if leads is None:
        leads = list(range(beat_list[0].shape[0]))
    lead_indices = [
        lead_names.index(lead) if isinstance(lead, str) else int(lead)
        for lead in leads
    ]

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(beat_list), 3)))

    n_panels = len(lead_indices)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (3.2 * ncols, 2.4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for panel, lead_idx in enumerate(lead_indices):
        ax = axes_flat[panel]
        for beat, band, label, color in zip(beat_list, band_list, labels, colors):
            if lead_idx >= beat.shape[0]:
                continue
            sig = beat[lead_idx]
            time_ms = np.arange(sig.shape[0]) / sampling_rate * 1000.0
            if band is not None:
                ax.fill_between(
                    time_ms, band[0][lead_idx], band[1][lead_idx],
                    color=color, alpha=0.15, linewidth=0,
                )
            ax.plot(time_ms, sig, color=color, linewidth=linewidth,
                    alpha=alpha, label=label)
        name = lead_names[lead_idx] if lead_idx < len(lead_names) else f"Lead {lead_idx}"
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axhline(0, color="0.7", linewidth=0.5, zorder=0)
        ax.grid(True, alpha=0.25)
        if xlim_ms is not None:
            ax.set_xlim(*xlim_ms)
        if min_yrange > 0:
            lo, hi = ax.get_ylim()
            if hi - lo < min_yrange:
                center = (lo + hi) / 2
                ax.set_ylim(center - min_yrange / 2, center + min_yrange / 2)
        if panel % ncols == 0:
            ax.set_ylabel("mV")
        if panel // ncols == nrows - 1:
            ax.set_xlabel("Time (ms)")

    for panel in range(n_panels, len(axes_flat)):
        axes_flat[panel].axis("off")

    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes
