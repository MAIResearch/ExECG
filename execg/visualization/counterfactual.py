"""Counterfactual visualization for ECG-XAI."""

from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .attribution import add_attribution_heatmap, bin_attribution, normalize_attribution


def plot_counterfactual_overlay(
    original_ecg: np.ndarray,
    cf_ecg: np.ndarray,
    lead_idx: int = 1,
    figsize: Tuple[int, int] = (14, 4),
    lead_names: Optional[List[str]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> None:
    """Plot original and counterfactual ECG overlay.

    Args:
        original_ecg: Original ECG array of shape (n_leads, seq_length).
        cf_ecg: Counterfactual ECG array of shape (n_leads, seq_length).
        lead_idx: Lead index to visualize.
        figsize: Figure size.
        lead_names: List of lead names.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.
    """
    if lead_names is None:
        lead_names = [f"Lead {i}" for i in range(original_ecg.shape[0])]

    plt.figure(figsize=figsize)
    plt.plot(original_ecg[lead_idx], "b-", linewidth=1, label="Original", alpha=0.8)
    plt.plot(cf_ecg[lead_idx], "r-", linewidth=1, label="Counterfactual", alpha=0.8)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(f"Original vs Counterfactual ECG ({lead_names[lead_idx]})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_counterfactual_diff(
    original_ecg: np.ndarray,
    cf_ecg: np.ndarray,
    lead_idx: int = 1,
    figsize: Tuple[int, int] = (14, 6),
    lead_names: Optional[List[str]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> None:
    """Plot original, counterfactual, and difference.

    Args:
        original_ecg: Original ECG array of shape (n_leads, seq_length).
        cf_ecg: Counterfactual ECG array of shape (n_leads, seq_length).
        lead_idx: Lead index to visualize.
        figsize: Figure size.
        lead_names: List of lead names.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.
    """
    if lead_names is None:
        lead_names = [f"Lead {i}" for i in range(original_ecg.shape[0])]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    axes[0].plot(original_ecg[lead_idx], "b-", linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Original ECG ({lead_names[lead_idx]})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cf_ecg[lead_idx], "r-", linewidth=0.8)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"Counterfactual ECG ({lead_names[lead_idx]})")
    axes[1].grid(True, alpha=0.3)

    diff = cf_ecg[lead_idx] - original_ecg[lead_idx]
    axes[2].plot(diff, "g-", linewidth=0.8)
    axes[2].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[2].set_ylabel("Difference")
    axes[2].set_xlabel("Sample")
    axes[2].set_title("Difference (CF - Original)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_counterfactual_progress(
    all_probs: List[float],
    target_value: float,
    target_idx: int = 0,
    figsize: Tuple[int, int] = (10, 4),
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> None:
    """Plot counterfactual generation progress.

    Args:
        all_probs: List of probabilities during optimization.
        target_value: Target probability value.
        target_idx: Target class index.
        figsize: Figure size.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.
    """
    plt.figure(figsize=figsize)
    plt.plot(all_probs, "b-o", markersize=4)
    plt.axhline(y=target_value, color="r", linestyle="--", label=f"Target: {target_value}")
    plt.xlabel("Iteration (improvement steps)")
    plt.ylabel(f"Class {target_idx} Probability")
    plt.title("Counterfactual Generation Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_counterfactual_evolution(
    all_cf: List[np.ndarray],
    all_probs: List[float],
    lead_idx: int = 1,
    n_steps: int = 5,
    figsize: Tuple[int, int] = (14, 3),
    lead_names: Optional[List[str]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> None:
    """Plot counterfactual ECG evolution during optimization.

    Args:
        all_cf: List of counterfactual ECGs during optimization.
        all_probs: List of probabilities during optimization.
        lead_idx: Lead index to visualize.
        n_steps: Number of steps to show.
        figsize: Figure size (width, height_per_step).
        lead_names: List of lead names.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.
    """
    if lead_names is None:
        lead_names = [f"Lead {i}" for i in range(all_cf[0].shape[0])]

    n_steps = min(n_steps, len(all_cf))
    step_indices = np.linspace(0, len(all_cf) - 1, n_steps, dtype=int)

    fig_width, fig_height_per_step = figsize
    fig, axes = plt.subplots(
        n_steps, 1, figsize=(fig_width, fig_height_per_step * n_steps), sharex=True
    )

    if n_steps == 1:
        axes = [axes]

    for i, idx in enumerate(step_indices):
        axes[i].plot(all_cf[idx][lead_idx], "b-", linewidth=0.8)
        axes[i].set_ylabel(f"Step {idx}")
        prob = all_probs[idx] if idx < len(all_probs) else all_probs[-1]
        axes[i].set_title(f"Probability: {prob:.4f}")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample")
    plt.suptitle(f"Counterfactual Evolution ({lead_names[lead_idx]})", y=1.02)
    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_counterfactual_all_leads(
    original_ecg: np.ndarray,
    cf_ecg: np.ndarray,
    original_prob: float,
    cf_prob: float,
    target_idx: int = 0,
    figsize: Tuple[int, int] = (16, 18),
    lead_names: Optional[List[str]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> None:
    """Plot 12-lead comparison of original and counterfactual ECG.

    Args:
        original_ecg: Original ECG array of shape (n_leads, seq_length).
        cf_ecg: Counterfactual ECG array of shape (n_leads, seq_length).
        original_prob: Original prediction probability.
        cf_prob: Counterfactual prediction probability.
        target_idx: Target class index.
        figsize: Figure size.
        lead_names: List of lead names.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.
    """
    n_leads = original_ecg.shape[0]

    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        if n_leads != 12:
            lead_names = [f"Lead {i}" for i in range(n_leads)]

    fig, axes = plt.subplots(n_leads, 1, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for i in range(n_leads):
        axes[i].plot(original_ecg[i], "b-", linewidth=0.6, alpha=0.7, label="Original")
        axes[i].plot(cf_ecg[i], "r-", linewidth=0.6, alpha=0.7, label="CF")
        axes[i].set_title(f"{lead_names[i]}")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right")

    plt.suptitle(
        f"12-Lead ECG Comparison\n"
        f"Original prob (class {target_idx}): {original_prob:.4f} → CF prob: {cf_prob:.4f}",
        y=1.01,
    )
    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_counterfactual_diff_with_attribution(
    original_ecg: np.ndarray,
    cf_ecg: np.ndarray,
    attribution: np.ndarray,
    lead_idx: int = 1,
    bin_size: int = 50,
    cmap: str = "Reds",
    figsize: Tuple[int, int] = (14, 8),
    lead_names: Optional[List[str]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> None:
    """Plot original ECG with attribution, counterfactual, and difference.

    Args:
        original_ecg: Original ECG array of shape (n_leads, seq_length).
        cf_ecg: Counterfactual ECG array of shape (n_leads, seq_length).
        attribution: Attribution array for original ECG.
        lead_idx: Lead index to visualize.
        bin_size: Number of samples per bin for attribution.
        cmap: Matplotlib colormap name for attribution heatmap.
        figsize: Figure size.
        lead_names: List of lead names.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.
    """
    if lead_names is None:
        lead_names = [f"Lead {i}" for i in range(original_ecg.shape[0])]

    # Prepare attribution data
    if attribution.ndim == 1 or attribution.shape[0] == 1:
        attr_data = attribution.flatten()
    else:
        attr_data = attribution[lead_idx]

    binned_attr = bin_attribution(attr_data, bin_size)
    normalized_attr = normalize_attribution([binned_attr])[0]

    n_samples = original_ecg.shape[1]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Original ECG with attribution heatmap
    axes[0].plot(original_ecg[lead_idx], "b-", linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Original ECG with Attribution ({lead_names[lead_idx]})")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, n_samples)

    heatmap_bottom, heatmap_height = add_attribution_heatmap(
        axes[0], normalized_attr, bin_size, n_samples, cmap
    )
    y_min, y_max = axes[0].get_ylim()
    y_range = y_max - y_min
    axes[0].set_ylim(heatmap_bottom - heatmap_height * 0.1, y_max + y_range * 0.05)

    # Add colorbar for attribution
    cmap_obj = plt.cm.get_cmap(cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], orientation="vertical", fraction=0.02, pad=0.01)
    cbar.set_label("Attribution")

    # Counterfactual ECG
    axes[1].plot(cf_ecg[lead_idx], "r-", linewidth=0.8)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title(f"Counterfactual ECG ({lead_names[lead_idx]})")
    axes[1].grid(True, alpha=0.3)

    # Difference
    diff = cf_ecg[lead_idx] - original_ecg[lead_idx]
    axes[2].plot(diff, "g-", linewidth=0.8)
    axes[2].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[2].set_ylabel("Difference")
    axes[2].set_xlabel("Sample")
    axes[2].set_title("Difference (CF - Original)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")
    plt.show()
