"""ECG Chart visualization with XAI overlay for ExECG.

Based on ECGSheetViewer.py, refactored to accept array input instead of JSON,
with added support for counterfactual overlay and attribution heatmap.
"""

from math import ceil
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from .attribution import bin_attribution


# Standard 12-lead order
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


def _normalize_ecg_input(ecg: np.ndarray) -> np.ndarray:
    """Normalize ECG array to shape (n_leads, length)."""
    if not isinstance(ecg, np.ndarray):
        ecg = ecg.detach().cpu().numpy()

    if ecg.ndim == 3:
        ecg = ecg.squeeze(0)
    elif ecg.ndim == 1:
        ecg = ecg.reshape(1, -1)

    return ecg


def _split_ecg_continuous(ecg: np.ndarray, columns: int) -> np.ndarray:
    """Split ECG for partial continuous view.

    Each lead gets assigned to a column based on its index.
    The lead extracts only the time segment corresponding to its column.

    For columns=4 with 12 leads:
    - Leads 0,1,2 (I, II, III) in column 0 → show time 0-2.5s
    - Leads 3,4,5 (aVR, aVL, aVF) in column 1 → show time 2.5-5s
    - Leads 6,7,8 (V1, V2, V3) in column 2 → show time 5-7.5s
    - Leads 9,10,11 (V4, V5, V6) in column 3 → show time 7.5-10s
    """
    n_leads, length = ecg.shape
    rows = int(ceil(n_leads / columns))
    split_length = length // columns

    split_ecg = np.zeros((n_leads, split_length), dtype=ecg.dtype)

    for lead_idx in range(n_leads):
        col = lead_idx // rows
        start = col * split_length
        end = start + split_length
        split_ecg[lead_idx, :] = ecg[lead_idx, start:end]

    return split_ecg


def _set_grid(
    ax,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    display_factor: float,
    major_grid_col: tuple,
    minor_grid_col: tuple,
) -> None:
    """Set ECG-style grid on axis."""
    ax.set_axisbelow(False)

    ax.set_xticks(np.arange(x_min, x_max, 0.2))
    ax.set_yticks(np.arange(y_min, y_max, 0.5))

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax.grid(
        which="major",
        linestyle="-",
        linewidth=0.8 * display_factor,
        color=major_grid_col,
        zorder=2.5,
    )
    ax.grid(
        which="minor",
        linestyle="-",
        linewidth=0.5 * display_factor,
        color=minor_grid_col,
        zorder=2.5,
    )

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)


def _draw_attribution_heatmap(
    ax,
    attribution: np.ndarray,
    x_offset: float,
    y_offset: float,
    secs: float,
    row_height: float,
    bin_size: int,
    cmap: str,
    global_min: float = 0.0,
    global_max: float = 1.0,
) -> None:
    """Draw attribution heatmap below ECG signal.

    Args:
        global_min, global_max: Global min/max values for normalization.
            This ensures consistent coloring across all leads.
    """
    cmap_obj = plt.cm.get_cmap(cmap)

    binned = bin_attribution(attribution, bin_size)

    if global_max > global_min:
        binned = (binned - global_min) / (global_max - global_min)
        binned = np.clip(binned, 0, 1)

    n_bins = len(binned)
    bin_width = secs / n_bins
    heatmap_height = row_height * 0.04
    heatmap_y = y_offset - row_height / 5 - heatmap_height * 0.5

    for i, val in enumerate(binned):
        x_start = x_offset + i * bin_width
        color = cmap_obj(val)
        rect = plt.Rectangle(
            (x_start, heatmap_y),
            bin_width,
            heatmap_height,
            facecolor=color,
            edgecolor="none",
            zorder=10,
        )
        ax.add_patch(rect)


def plot_ecg_chart(
    ecg: np.ndarray,
    sample_rate: float = 250,
    cf_ecg: Optional[np.ndarray] = None,
    attribution: Optional[np.ndarray] = None,
    title: str = "Electrocardiogram",
    meta: Optional[Dict] = None,
    style: str = "clinical",
    columns: int = 4,
    row_height: int = 6,
    lead_names: Optional[List[str]] = None,
    partial_continuous: bool = True,
    show_lead_name: bool = True,
    show_separate_line: bool = True,
    show_grid: bool = True,
    show_full_single_last: bool = False,
    show_calibration: bool = False,
    ecg_linewidth: float = 0.5,
    cf_color: Union[tuple, str] = (1, 0, 0),
    cf_alpha: float = 0.7,
    cf_label: str = "CF",
    attr_bin_size: int = 25,
    attr_cmap: str = "Reds",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot 12-lead ECG chart with optional XAI overlays.

    This function creates a clinical-style ECG visualization based on ECGSheetViewer,
    with additional support for counterfactual overlay and attribution heatmap.

    Args:
        ecg: ECG signal array. Shape: (1, n_leads, length) or (n_leads, length).
        sample_rate: Sampling rate in Hz. Default: 250.
        cf_ecg: Counterfactual ECG array with same shape as ecg. Optional.
        attribution: Attribution scores array with same shape as ecg. Optional.
        title: Chart title. Default: "Electrocardiogram".
        meta: Dictionary containing metadata to display. Optional.
        style: Visual style - "clinical"/"rb" (red grid, blue ecg) or "paper"/"bb" (gray grid, black ecg).
        columns: Number of columns for lead layout. Default: 4.
        row_height: Height of each lead row (number of major grid squares). Default: 6.
        lead_names: List of lead names. If None, uses standard 12-lead names.
        partial_continuous: If True, each column shows continuous time segments. Default: True.
        show_lead_name: Show lead name at the beginning of each lead. Default: True.
        show_separate_line: Show separation line between columns. Default: True.
        show_grid: Whether to show ECG grid. Default: True.
        show_full_single_last: Show full Lead II at bottom. Default: False.
        show_calibration: Show calibration pulse before each lead. Default: False.
        ecg_linewidth: Line width for ECG signals. Default: 0.5.
        cf_color: Color for counterfactual ECG. Default: red (1, 0, 0).
        cf_alpha: Alpha transparency for counterfactual. Default: 0.7.
        cf_label: Label for counterfactual in legend. Default: "CF".
        attr_bin_size: Number of samples per attribution bin. Default: 25.
        attr_cmap: Colormap for attribution heatmap. Default: "Reds".
        figsize: Figure size (width, height). If None, auto-calculated.
        save_path: Path to save figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default: 300.

    Returns:
        Tuple of (fig, axes) matplotlib objects.

    Example:
        >>> import numpy as np
        >>> from execg.visualizer import plot_ecg_chart
        >>>
        >>> # Basic ECG visualization (standard clinical layout)
        >>> ecg = np.random.randn(1, 12, 2500)
        >>> fig, axes = plot_ecg_chart(ecg, sample_rate=250)
        >>>
        >>> # With counterfactual overlay
        >>> cf_ecg = np.random.randn(1, 12, 2500)
        >>> fig, axes = plot_ecg_chart(ecg, cf_ecg=cf_ecg, style="clinical")
        >>>
        >>> # With attribution heatmap
        >>> attribution = np.random.rand(1, 12, 2500)
        >>> fig, axes = plot_ecg_chart(ecg, attribution=attribution)
    """
    if row_height <= 0:
        raise ValueError("row_height must be positive")

    ecg = _normalize_ecg_input(ecg)
    n_leads, seq_length = ecg.shape

    if cf_ecg is not None:
        cf_ecg = _normalize_ecg_input(cf_ecg)
        if cf_ecg.shape != ecg.shape:
            raise ValueError(
                f"cf_ecg shape {cf_ecg.shape} must match ecg shape {ecg.shape}"
            )

    if attribution is not None:
        attribution = _normalize_ecg_input(attribution)
        if attribution.shape != ecg.shape:
            raise ValueError(
                f"attribution shape {attribution.shape} must match ecg shape {ecg.shape}"
            )

    if lead_names is None:
        if n_leads == 12:
            lead_names = LEAD_NAMES_12.copy()
        else:
            lead_names = [f"Lead {i+1}" for i in range(n_leads)]

    if style in ("clinical", "rb"):
        color_grid_major = (1, 0, 0)
        color_grid_minor = (1, 0.7, 0.7)
        color_ecg_line = (0, 0, 0.7)
    elif style in ("paper", "bb"):
        color_grid_major = (0.4, 0.4, 0.4)
        color_grid_minor = (0.75, 0.75, 0.75)
        color_ecg_line = (0, 0, 0)
    else:
        color_grid_major = (1, 0, 0)
        color_grid_minor = (1, 0.7, 0.7)
        color_ecg_line = (0, 0, 0.7)

    if "II" in lead_names:
        single_idx = lead_names.index("II")
        single = ecg[single_idx, :].copy()
        single_title = "II"
        single_cf = cf_ecg[single_idx, :].copy() if cf_ecg is not None else None
        single_attr = (
            attribution[single_idx, :].copy() if attribution is not None else None
        )
    elif "I" in lead_names:
        single_idx = lead_names.index("I")
        single = ecg[single_idx, :].copy()
        single_title = "I"
        single_cf = cf_ecg[single_idx, :].copy() if cf_ecg is not None else None
        single_attr = (
            attribution[single_idx, :].copy() if attribution is not None else None
        )
    else:
        single = np.zeros(seq_length)
        single_title = "No lead"
        single_cf = None
        single_attr = None

    if partial_continuous:
        ecg = _split_ecg_continuous(ecg, columns)
        if cf_ecg is not None:
            cf_ecg = _split_ecg_continuous(cf_ecg, columns)
        if attribution is not None:
            attribution = _split_ecg_continuous(attribution, columns)

    if attribution is not None:
        binned_attrs = [bin_attribution(attribution[i], attr_bin_size) for i in range(len(attribution))]
        if single_attr is not None:
            binned_attrs.append(bin_attribution(single_attr, attr_bin_size))
        all_binned = np.concatenate(binned_attrs)
        attr_global_min = all_binned.min()
        attr_global_max = all_binned.max()
    else:
        attr_global_min = 0.0
        attr_global_max = 1.0

    lead_order = list(range(0, len(ecg)))
    secs = len(ecg[0]) / sample_rate
    leads = len(lead_order)
    rows = int(ceil(leads / columns))
    display_factor = 1
    line_width = ecg_linewidth
    second_calib = 0.4

    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    if show_calibration:
        x_max += second_calib

    show_meta = meta is not None
    additional_n = show_meta + show_full_single_last
    n_figs = 1 + additional_n

    height_ratio = [rows]
    if show_meta:
        height_ratio = [1] + height_ratio
    if show_full_single_last:
        height_ratio = height_ratio + [1]

    if figsize is None:
        if partial_continuous:
            # Total duration is original length, not split length
            fig_width = len(single) / sample_rate * display_factor
        else:
            fig_width = secs * columns * display_factor
        fig_height = (rows + additional_n) * row_height / 5 * display_factor
        figsize = (fig_width, fig_height)

    fig, axes = plt.subplots(
        n_figs,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratio},
    )

    axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes
    for ax in axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    main_ax_pos = 0
    if show_meta:
        main_ax_pos += 1
        text_ax = axes[0]
        text_ax.set_ylim(0, 10)
        text_ax.set_xlim(0, 1)
        text_ax.axis("off")

        text_ax.text(0.5, 8, title, fontsize=12, fontweight="bold", ha="center")

        meta_text = "\n".join([f"{k}: {v}" for k, v in meta.items()])
        text_ax.text(0.5, 4, meta_text, fontsize=9, ha="center", va="center")

    if show_full_single_last:
        step = 1.0 / sample_rate
        last_ax = axes[-1]
        min_ = -(row_height / 2 / 2)
        max_ = row_height / 2 / 2

        if show_grid:
            _set_grid(
                last_ax,
                x_min,
                x_max,
                min_,
                max_,
                display_factor,
                color_grid_major,
                color_grid_minor,
            )

        x_offset = 0

        if show_calibration:
            start = 0
            end = second_calib
            a_quarters = start + end * 0.25
            three_quarters = start + end * 0.75
            last_ax.plot(
                [start, a_quarters, a_quarters, three_quarters, three_quarters, end],
                [0, 0, 0 + 1, 0 + 1, 0, 0],
                linewidth=line_width * display_factor,
                color=color_ecg_line,
            )
            x_offset += second_calib

        last_ax.plot(
            np.arange(0, len(single) * step, step) + x_offset,
            single,
            linewidth=line_width * display_factor,
            color=color_ecg_line,
            label="Original",
            zorder=3,
        )

        if single_cf is not None:
            last_ax.plot(
                np.arange(0, len(single_cf) * step, step) + x_offset,
                single_cf,
                linewidth=line_width * display_factor,
                color=cf_color,
                alpha=cf_alpha,
                label=cf_label,
                zorder=3,
            )

        if single_attr is not None:
            single_secs = len(single) / sample_rate
            _draw_attribution_heatmap(
                last_ax,
                single_attr,
                x_offset,
                0,
                single_secs,
                row_height,
                attr_bin_size,
                attr_cmap,
                attr_global_min,
                attr_global_max,
            )

        last_ax.text(
            0 + 0.07 + x_offset,
            0 - 0.5,
            single_title,
            fontsize=9 * display_factor,
        )

    ax = axes[0] if len(axes) == 1 else axes[main_ax_pos]

    fig.subplots_adjust(
        hspace=0,
        wspace=0,
        left=0,
        right=1,
        bottom=0,
        top=1,
    )

    if not show_meta:
        axes[0].set_title(title)

    if show_grid:
        _set_grid(
            ax,
            x_min,
            x_max,
            y_min,
            y_max,
            display_factor,
            color_grid_major,
            color_grid_minor,
        )

    for c in range(0, columns):
        for i in range(0, rows):
            if c * rows + i < leads:
                t_lead = lead_order[c * rows + i]
                y_offset = -(row_height / 2) * ceil(i % rows)

                x_offset = 0

                if show_calibration:
                    start = 0
                    end = second_calib
                    a_quarters = start + end * 0.25
                    three_quarters = start + end * 0.75
                    ax.plot(
                        [
                            start,
                            a_quarters,
                            a_quarters,
                            three_quarters,
                            three_quarters,
                            end,
                        ],
                        [
                            y_offset,
                            y_offset,
                            y_offset + 1,
                            y_offset + 1,
                            y_offset,
                            y_offset,
                        ],
                        linewidth=line_width * display_factor,
                        color=color_ecg_line,
                    )
                    x_offset += second_calib

                if c > 0:
                    x_offset = secs * c
                    if show_calibration:
                        x_offset += second_calib
                    if show_separate_line:
                        ax.plot(
                            [x_offset, x_offset],
                            [
                                ecg[t_lead][0] + y_offset - 0.3,
                                ecg[t_lead][0] + y_offset + 0.3,
                            ],
                            linewidth=line_width * display_factor * 2,
                            color=color_ecg_line,
                        )

                step = 1.0 / sample_rate

                if show_lead_name:
                    ax.text(
                        x_offset + 0.07,
                        y_offset - 0.5,
                        lead_names[t_lead],
                        fontsize=9 * display_factor,
                    )

                if attribution is not None:
                    _draw_attribution_heatmap(
                        ax,
                        attribution[t_lead],
                        x_offset,
                        y_offset,
                        secs,
                        row_height,
                        attr_bin_size,
                        attr_cmap,
                        attr_global_min,
                        attr_global_max,
                    )

                ax.plot(
                    np.arange(0, len(ecg[t_lead]) * step, step) + x_offset,
                    ecg[t_lead] + y_offset,
                    linewidth=line_width * display_factor,
                    color=color_ecg_line,
                    label="Original" if c == 0 and i == 0 else None,
                    zorder=3,
                )

                if cf_ecg is not None:
                    ax.plot(
                        np.arange(0, len(cf_ecg[t_lead]) * step, step) + x_offset,
                        cf_ecg[t_lead] + y_offset,
                        linewidth=line_width * display_factor,
                        color=cf_color,
                        alpha=cf_alpha,
                        label=cf_label if c == 0 and i == 0 else None,
                        zorder=3,
                    )

    if cf_ecg is not None:
        ax.legend(loc="upper right", fontsize=8)

    if attribution is not None:
        cmap_obj = plt.cm.get_cmap(attr_cmap)
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes,
            orientation="horizontal",
            fraction=0.02,
            pad=0.02,
            aspect=50,
        )
        cbar.set_label("Attribution Score", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes


def plot_ecg_comparison(
    ecg_list: List[np.ndarray],
    labels: List[str],
    sample_rate: float = 250,
    colors: Optional[List[str]] = None,
    lead_idx: Union[int, List[int]] = 1,
    title: str = "ECG Comparison",
    figsize: Tuple[float, float] = (14, 4),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple ECG signals for comparison.

    Args:
        ecg_list: List of ECG arrays, each with shape (1, 12, length) or (12, length).
        labels: List of labels for each ECG.
        sample_rate: Sampling rate in Hz.
        colors: List of colors. If None, uses default color cycle.
        lead_idx: Lead index or list of indices to plot.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
        dpi: DPI for saved figure.

    Returns:
        Tuple of (fig, axes) matplotlib objects.
    """
    # Normalize inputs
    ecg_list = [_normalize_ecg_input(ecg) for ecg in ecg_list]

    if isinstance(lead_idx, int):
        lead_idx = [lead_idx]

    n_leads = len(lead_idx)
    n_ecgs = len(ecg_list)

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_ecgs))

    fig, axes = plt.subplots(
        n_leads,
        1,
        figsize=(figsize[0], figsize[1] * n_leads),
        squeeze=False,
        sharex=True,
    )

    for row, lid in enumerate(lead_idx):
        ax = axes[row, 0]

        for ecg, label, color in zip(ecg_list, labels, colors):
            time = np.arange(ecg.shape[1]) / sample_rate
            ax.plot(time, ecg[lid], color=color, linewidth=0.8, alpha=0.8, label=label)

        lead_name = LEAD_NAMES_12[lid] if lid < 12 else f"Lead {lid}"
        ax.set_ylabel(lead_name)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if row == n_leads - 1:
            ax.set_xlabel("Time (s)")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes
