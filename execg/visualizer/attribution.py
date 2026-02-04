"""Attribution visualization for ECG-XAI."""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def bin_attribution(attr_data: np.ndarray, bin_size: int = 10) -> np.ndarray:
    """Bin attribution data by averaging samples.

    Args:
        attr_data: Attribution data array (1D).
        bin_size: Number of samples per bin.

    Returns:
        Binned attribution array.
    """
    n_samples = len(attr_data)
    n_bins = n_samples // bin_size

    binned = np.array(
        [attr_data[i * bin_size : (i + 1) * bin_size].mean() for i in range(n_bins)]
    )

    remainder = n_samples % bin_size
    if remainder > 0:
        last_bin = attr_data[n_bins * bin_size :].mean()
        binned = np.append(binned, last_bin)

    return binned


def normalize_attribution(binned_list: List[np.ndarray]) -> List[np.ndarray]:
    """Global min-max normalize multiple lead attributions.

    Uses actual min-max to ensure full contrast in visualization.

    Args:
        binned_list: List of binned attribution arrays.

    Returns:
        List of normalized attribution arrays.
    """
    all_values = np.concatenate(binned_list)
    global_min = all_values.min()
    global_max = all_values.max()

    normalized_list = []
    for binned in binned_list:
        if global_max > global_min:
            normalized = (binned - global_min) / (global_max - global_min)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.zeros_like(binned)
        normalized_list.append(normalized)

    return normalized_list


def bin_and_normalize(attr_data: np.ndarray, bin_size: int = 10) -> np.ndarray:
    """Bin and min-max normalize attribution data (single lead).

    Uses actual min-max to ensure full contrast in visualization.

    Args:
        attr_data: Attribution data array (1D).
        bin_size: Number of samples per bin.

    Returns:
        Binned and normalized attribution array.
    """
    binned = bin_attribution(attr_data, bin_size)

    global_min = binned.min()
    global_max = binned.max()
    if global_max > global_min:
        binned = (binned - global_min) / (global_max - global_min)
        binned = np.clip(binned, 0, 1)
    else:
        binned = np.zeros_like(binned)

    return binned


def add_attribution_heatmap(
    ax: plt.Axes,
    binned_attr: np.ndarray,
    bin_size: int,
    n_samples: int,
    cmap: str = "Reds",
    height_ratio: float = 0.15,
) -> Tuple[float, float]:
    """Add attribution heatmap below an existing plot.

    Args:
        ax: Matplotlib axes to add heatmap to.
        binned_attr: Normalized binned attribution array (values 0-1).
        bin_size: Number of samples per bin.
        n_samples: Total number of samples in the signal.
        cmap: Matplotlib colormap name.
        height_ratio: Height of heatmap as ratio of y-axis range.

    Returns:
        Tuple of (heatmap_bottom, heatmap_height) for ylim adjustment.
    """
    cmap_obj = plt.cm.get_cmap(cmap)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    heatmap_height = y_range * height_ratio
    heatmap_bottom = y_min - heatmap_height * 1.2

    for i, val in enumerate(binned_attr):
        x_start = i * bin_size
        x_end = min((i + 1) * bin_size, n_samples)
        color = cmap_obj(val)
        rect = plt.Rectangle(
            (x_start, heatmap_bottom),
            x_end - x_start,
            heatmap_height,
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(rect)

    return heatmap_bottom, heatmap_height


def plot_attribution(
    ecg: np.ndarray,
    attribution: np.ndarray,
    title: str,
    lead_idx: Optional[Union[int, List[int]]] = None,
    bin_size: int = 10,
    cmap: str = "Reds",
    figsize: Tuple[int, int] = (14, 5),
    lead_names: Optional[List[str]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> None:
    """Plot ECG signal with attribution heatmap.

    Args:
        ecg: ECG signal array of shape (n_leads, seq_length).
        attribution: Attribution result array.
        title: Plot title.
        lead_idx: Lead index to visualize. Can be int, list[int], or None (all leads).
        bin_size: Number of samples per bin.
        cmap: Matplotlib colormap name.
        figsize: Figure size (width, height_per_lead).
        lead_names: List of lead names. If None, uses "Lead 0", "Lead 1", etc.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.
    """
    n_leads_total = ecg.shape[0]

    if lead_idx is None:
        leads_to_plot = list(range(n_leads_total))
    elif isinstance(lead_idx, int):
        leads_to_plot = [lead_idx]
    else:
        leads_to_plot = list(lead_idx)

    n_leads = len(leads_to_plot)

    if lead_names is None:
        lead_names = [f"Lead {i}" for i in range(n_leads_total)]

    binned_attrs = []
    for lead in leads_to_plot:
        if attribution.ndim == 1 or attribution.shape[0] == 1:
            attr_data = attribution.flatten()
        else:
            attr_data = attribution[lead]
        binned_attrs.append(bin_attribution(attr_data, bin_size))

    normalized_attrs = normalize_attribution(binned_attrs)

    fig_width, fig_height_per_lead = figsize
    adjusted_figsize = (fig_width, fig_height_per_lead * n_leads)

    fig, axes = plt.subplots(n_leads, 1, figsize=adjusted_figsize, squeeze=False)
    axes = axes.flatten()

    cmap_obj = plt.cm.get_cmap(cmap)

    for plot_idx, lead in enumerate(leads_to_plot):
        ax = axes[plot_idx]
        ecg_data = ecg[lead]
        n_samples = len(ecg_data)
        binned_attr = normalized_attrs[plot_idx]

        ax.plot(ecg_data, "k-", linewidth=0.8)
        ax.set_xlim(0, n_samples)
        ax.set_ylabel("Amplitude")

        if plot_idx == 0:
            ax.set_title(f"{title} (bin_size={bin_size})\n{lead_names[lead]}")
        else:
            ax.set_title(lead_names[lead])

        if plot_idx == n_leads - 1:
            ax.set_xlabel("Sample")

        heatmap_bottom, heatmap_height = add_attribution_heatmap(
            ax, binned_attr, bin_size, n_samples, cmap
        )

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(heatmap_bottom - heatmap_height * 0.1, y_max + y_range * 0.05)

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.01)
        if plot_idx == 0:
            cbar.set_label("Attribution Score")

    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_attribution_comparison(
    ecg: np.ndarray,
    methods_dict: Dict[str, Tuple[np.ndarray, str]],
    lead_idx: Optional[Union[int, List[int]]] = None,
    bin_size: int = 50,
    figsize: Tuple[int, int] = (16, 8),
    lead_names: Optional[List[str]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> None:
    """Compare multiple attribution methods with heatmaps below ECG.

    Args:
        ecg: ECG signal array of shape (n_leads, seq_length).
        methods_dict: Dictionary of {method_name: (attribution_result, cmap)}.
        lead_idx: Lead index to visualize. Can be int, list[int], or None (all leads).
        bin_size: Number of samples per bin.
        figsize: Figure size (width, height_per_lead).
        lead_names: List of lead names. If None, uses "Lead 0", "Lead 1", etc.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.
    """
    n_leads_total = ecg.shape[0]
    n_methods = len(methods_dict)

    if lead_idx is None:
        leads_to_plot = list(range(n_leads_total))
    elif isinstance(lead_idx, int):
        leads_to_plot = [lead_idx]
    else:
        leads_to_plot = list(lead_idx)

    n_leads = len(leads_to_plot)

    if lead_names is None:
        lead_names = [f"Lead {i}" for i in range(n_leads_total)]

    normalized_methods = {}
    for method_name, (attr_result, cmap_name) in methods_dict.items():
        binned_attrs = []
        for lead in leads_to_plot:
            if attr_result.ndim == 1 or attr_result.shape[0] == 1:
                attr_data = attr_result.flatten()
            else:
                attr_data = attr_result[lead]
            binned_attrs.append(bin_attribution(attr_data, bin_size))

        normalized_attrs = normalize_attribution(binned_attrs)
        normalized_methods[method_name] = (normalized_attrs, cmap_name)

    fig_width, fig_height_per_lead = figsize
    adjusted_figsize = (fig_width, fig_height_per_lead * n_leads)

    fig, axes = plt.subplots(n_leads, 1, figsize=adjusted_figsize, squeeze=False)
    axes = axes.flatten()

    n_samples = ecg.shape[1]

    for plot_idx, lead in enumerate(leads_to_plot):
        ax = axes[plot_idx]
        ecg_data = ecg[lead]

        ax.plot(ecg_data, "k-", linewidth=0.8)
        ax.set_xlim(0, n_samples)
        ax.set_ylabel("Amplitude")

        if plot_idx == 0:
            ax.set_title(
                f"Attribution Methods Comparison (bin_size={bin_size})\n{lead_names[lead]}"
            )
        else:
            ax.set_title(lead_names[lead])

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        heatmap_height = y_range * 0.12
        heatmap_gap = heatmap_height * 0.15
        total_heatmap_height = n_methods * (heatmap_height + heatmap_gap)

        for method_idx, (method_name, (normalized_attrs, cmap_name)) in enumerate(
            normalized_methods.items()
        ):
            binned_attr = normalized_attrs[plot_idx]

            heatmap_bottom = (
                y_min
                - total_heatmap_height
                + method_idx * (heatmap_height + heatmap_gap)
            )

            cmap_obj = plt.cm.get_cmap(cmap_name)
            for i, val in enumerate(binned_attr):
                x_start = i * bin_size
                x_end = min((i + 1) * bin_size, n_samples)
                color = cmap_obj(val)
                rect = plt.Rectangle(
                    (x_start, heatmap_bottom),
                    x_end - x_start,
                    heatmap_height,
                    facecolor=color,
                    edgecolor="none",
                )
                ax.add_patch(rect)

            ax.text(
                -n_samples * 0.01,
                heatmap_bottom + heatmap_height / 2,
                method_name,
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_ylim(y_min - total_heatmap_height - heatmap_gap, y_max + y_range * 0.05)

        ax.set_xlim(-n_samples * 0.12, n_samples)

        if plot_idx == n_leads - 1:
            ax.set_xlabel("Sample")

        sm = plt.cm.ScalarMappable(cmap="gray_r", norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.01)
        if plot_idx == 0:
            cbar.set_label("Attribution Score")

    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")
    plt.show()
