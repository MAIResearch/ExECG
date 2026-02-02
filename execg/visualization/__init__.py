"""Visualization module for ECG-XAI.

This module provides visualization functions for different XAI methods:
- attribution: Saliency maps, GradCAM, etc.
- counterfactual: StyleGAN-based counterfactual explanations
- tcav: Testing with Concept Activation Vectors
"""

from .attribution import (
    add_attribution_heatmap,
    bin_attribution,
    normalize_attribution,
    plot_attribution,
    plot_attribution_comparison,
)

from .counterfactual import (
    plot_counterfactual_all_leads,
    plot_counterfactual_diff,
    plot_counterfactual_diff_with_attribution,
    plot_counterfactual_evolution,
    plot_counterfactual_overlay,
    plot_counterfactual_progress,
)

from .tcav import (
    plot_tcav_bar,
    plot_tcav_comparison,
    plot_tcav_scores,
)

__all__ = [
    "add_attribution_heatmap",
    "bin_attribution",
    "bin_and_normalize",
    "normalize_attribution",
    "plot_attribution",
    "plot_attribution_comparison",
    "plot_counterfactual_overlay",
    "plot_counterfactual_diff",
    "plot_counterfactual_diff_with_attribution",
    "plot_counterfactual_progress",
    "plot_counterfactual_evolution",
    "plot_counterfactual_all_leads",
    "plot_tcav_scores",
    "plot_tcav_bar",
    "plot_tcav_comparison",
]
