"""TCAV visualization for ECG-XAI."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_tcav_scores(
    result_dict: Dict[str, Any],
    cmap: str = "coolwarm",
    figsize: Optional[Tuple[int, int]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot TCAV scores as a heatmap.

    Args:
        result_dict: Dictionary from TCAV.explain() method.
            Format: {concept_name: {layer_name: (mean_score, (ci_lower, ci_upper))}}
        cmap: Matplotlib colormap name.
        figsize: Figure size. If None, auto-calculated based on data.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.

    Returns:
        Tuple of (fig, ax) matplotlib objects.
    """
    targets = list(result_dict.keys())
    layers = sorted({layer for v in result_dict.values() for layer in v.keys()})

    means = pd.DataFrame(index=layers, columns=targets, dtype=float)
    annotations = pd.DataFrame(index=layers, columns=targets, dtype=str)

    for target in targets:
        for layer in layers:
            if layer in result_dict[target]:
                mean_val, (lower, upper) = result_dict[target][layer]
                means.at[layer, target] = mean_val
                annotations.at[layer, target] = (
                    f"{mean_val:.2f}\n[{lower:.2f}, {upper:.2f}]"
                )
            else:
                means.at[layer, target] = np.nan
                annotations.at[layer, target] = ""

    if figsize is None:
        figsize = (max(8, len(targets) * 2), max(4, len(layers) * 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(means.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)

    for i in range(len(layers)):
        for j in range(len(targets)):
            ax.text(j, i, annotations.iat[i, j], ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="TCAV Score")
    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_tcav_bar(
    result_dict: Dict[str, Any],
    layer: str,
    figsize: Tuple[int, int] = (10, 5),
    colors: Optional[List[str]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot TCAV scores as a bar chart for a specific layer.

    Args:
        result_dict: Dictionary from TCAV.explain() method.
        layer: Layer name to plot.
        figsize: Figure size.
        colors: List of colors for each concept. If None, uses default colormap.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.

    Returns:
        Tuple of (fig, ax) matplotlib objects.
    """
    concepts = list(result_dict.keys())
    scores = []
    errors_low = []
    errors_high = []

    for concept in concepts:
        if layer in result_dict[concept]:
            mean_val, (lower, upper) = result_dict[concept][layer]
            scores.append(mean_val)
            errors_low.append(mean_val - lower)
            errors_high.append(upper - mean_val)
        else:
            scores.append(0)
            errors_low.append(0)
            errors_high.append(0)

    if colors is None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(concepts)))

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(concepts))
    bars = ax.bar(
        x,
        scores,
        yerr=[errors_low, errors_high],
        capsize=5,
        color=colors,
        alpha=0.8,
    )

    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5, label="Baseline (0.5)")
    ax.set_xlabel("Concepts")
    ax.set_ylabel("TCAV Score")
    ax.set_title(f"TCAV Scores - Layer: {layer}")
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_tcav_comparison(
    result_dict: Dict[str, Any],
    layers: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    save_file_path: Optional[str] = None,
    dpi: int = 350,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot TCAV scores comparison across multiple layers.

    Args:
        result_dict: Dictionary from TCAV.explain() method.
        layers: List of layer names to compare.
        figsize: Figure size. If None, auto-calculated.
        save_file_path: Path to save the figure. If None, figure is not saved.
        dpi: DPI for saved figure. Default is 350.

    Returns:
        Tuple of (fig, axes) matplotlib objects.
    """
    concepts = list(result_dict.keys())
    n_layers = len(layers)

    if figsize is None:
        figsize = (6 * n_layers, 5)

    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    if n_layers == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(concepts)))

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        scores = []
        errors_low = []
        errors_high = []

        for concept in concepts:
            if layer in result_dict[concept]:
                mean_val, (lower, upper) = result_dict[concept][layer]
                scores.append(mean_val)
                errors_low.append(mean_val - lower)
                errors_high.append(upper - mean_val)
            else:
                scores.append(0)
                errors_low.append(0)
                errors_high.append(0)

        x = np.arange(len(concepts))
        ax.bar(
            x,
            scores,
            yerr=[errors_low, errors_high],
            capsize=5,
            color=colors,
            alpha=0.8,
        )

        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5, label="Baseline (0.5)")
        ax.set_xlabel("Concepts")
        ax.set_ylabel("TCAV Score")
        ax.set_title(f"Layer: {layer}")
        ax.set_xticks(x)
        ax.set_xticklabels(concepts, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("TCAV Scores by Layer", fontsize=14)
    plt.tight_layout()
    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=dpi, bbox_inches="tight")

    return fig, axes
