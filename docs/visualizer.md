# Visualization

## Overview

ExECG provides comprehensive visualization tools for ECG-XAI analysis. All visualization functions are designed to work seamlessly with the output of XAI methods (Attribution, Counterfactual, TCAV).

```python
from execg.visualizer import (
    # Attribution
    plot_attribution,
    plot_attribution_comparison,
    # Counterfactual
    plot_counterfactual_overlay,
    plot_counterfactual_diff,
    plot_counterfactual_progress,
    plot_counterfactual_evolution,
    plot_counterfactual_all_leads,
    plot_counterfactual_diff_with_attribution,
    # TCAV
    plot_tcav_scores,
    plot_tcav_bar,
    plot_tcav_comparison,
    # ECG Chart
    plot_ecg_chart,
    plot_ecg_comparison,
)
```

## Quick Start

```python
import torch
from execg.models import TorchModelWrapper
from execg.attribution import GradCAM
from execg.visualizer import plot_attribution, plot_ecg_chart

# Setup
wrapper = TorchModelWrapper(model)
gradcam = GradCAM(wrapper)

# Generate attribution
ecg_tensor = torch.randn(1, 12, 2500)
result = gradcam.explain(ecg_tensor, target=1, target_layers="conv3")

# Visualize
plot_attribution(
    ecg=result["inputs"].squeeze(),
    attribution=result["results"],
    title="GradCAM Attribution",
    lead_idx=1,
    bin_size=25
)
```

---

## Attribution Visualization

### plot_attribution()

Plot ECG signal with attribution heatmap below.

```python
plot_attribution(
    ecg,                    # (n_leads, seq_length)
    attribution,            # Attribution scores
    title,                  # Plot title
    lead_idx=None,          # int, list, or None (all leads)
    bin_size=10,            # Samples per heatmap bin
    cmap="Reds",            # Colormap
    figsize=(14, 5),        # (width, height_per_lead)
    lead_names=None,        # Custom lead names
    save_file_path=None,    # Save path
    dpi=350
)
```

**Example:**

```python
from execg.visualizer import plot_attribution

result = gradcam.explain(ecg_tensor, target=1, target_layers="conv3")

plot_attribution(
    ecg=result["inputs"].squeeze(),
    attribution=result["results"],
    title="GradCAM Attribution",
    lead_idx=[0, 1, 6],  # Leads I, II, V1
    bin_size=25,
    cmap="Reds"
)
```

### plot_attribution_comparison()

Compare multiple attribution methods side by side.

```python
plot_attribution_comparison(
    ecg,                    # (n_leads, seq_length)
    methods_dict,           # {method_name: (attribution, cmap)}
    lead_idx=None,          # int, list, or None
    bin_size=50,
    figsize=(16, 8),
    lead_names=None,
    save_file_path=None,
    dpi=350
)
```

**Example:**

```python
from execg.visualizer import plot_attribution_comparison

methods = {
    "GradCAM": (gradcam_result["results"], "Reds"),
    "Saliency": (saliency_result["results"], "Blues"),
    "SmoothGrad": (smoothgrad_result["results"], "Greens"),
}

plot_attribution_comparison(
    ecg=ecg_tensor.squeeze().numpy(),
    methods_dict=methods,
    lead_idx=1,
    bin_size=50
)
```

### Utility Functions

```python
from execg.visualizer import bin_attribution, normalize_attribution, bin_and_normalize

# Bin attribution for smoother visualization
binned = bin_attribution(attribution_1d, bin_size=25)

# Global min-max normalize multiple leads
normalized_list = normalize_attribution([attr_lead0, attr_lead1])

# Bin and normalize in one step (single lead)
processed = bin_and_normalize(attribution_1d, bin_size=25)
```

---

## Counterfactual Visualization

### plot_counterfactual_overlay()

Overlay original and counterfactual ECG signals.

```python
plot_counterfactual_overlay(
    original_ecg,           # (n_leads, seq_length)
    cf_ecg,                 # (n_leads, seq_length)
    lead_idx=1,
    figsize=(14, 4),
    lead_names=None,
    original_prob=None,     # Display in legend
    cf_prob=None,           # Display in legend
    save_file_path=None,
    dpi=350
)
```

**Example:**

```python
from execg.visualizer import plot_counterfactual_overlay

cf_ecg, cf_prob, info = cf_explainer.explain(ecg_tensor, target=1, target_value=0.9)

plot_counterfactual_overlay(
    original_ecg=ecg_tensor.squeeze().cpu().numpy(),
    cf_ecg=cf_ecg,
    lead_idx=1,
    original_prob=info["original_prob"],
    cf_prob=cf_prob
)
```

### plot_counterfactual_diff()

Show original, counterfactual, and difference in three subplots.

```python
plot_counterfactual_diff(
    original_ecg,           # (n_leads, seq_length)
    cf_ecg,                 # (n_leads, seq_length)
    lead_idx=1,
    figsize=(14, 6),
    lead_names=None,
    save_file_path=None,
    dpi=350
)
```

### plot_counterfactual_progress()

Visualize probability change during CF optimization.

```python
plot_counterfactual_progress(
    all_probs,              # List of probabilities from info["all_probs"]
    target_value,           # Target probability
    target=0,               # Target class index
    figsize=(10, 4),
    save_file_path=None,
    dpi=350
)
```

**Example:**

```python
from execg.visualizer import plot_counterfactual_progress

cf_ecg, cf_prob, info = cf_explainer.explain(ecg_tensor, target=1, target_value=0.9)

plot_counterfactual_progress(
    all_probs=info["all_probs"],
    target_value=0.9,
    target=1
)
```

### plot_counterfactual_evolution()

Show ECG changes at multiple steps during optimization.

```python
plot_counterfactual_evolution(
    all_cf,                 # List of CF ECGs from info["all_cf"]
    all_probs,              # List of probabilities
    lead_idx=1,
    n_steps=5,              # Number of steps to show
    figsize=(14, 3),
    lead_names=None,
    save_file_path=None,
    dpi=350
)
```

### plot_counterfactual_all_leads()

Compare all 12 leads between original and counterfactual.

```python
plot_counterfactual_all_leads(
    original_ecg,           # (n_leads, seq_length)
    cf_ecg,                 # (n_leads, seq_length)
    original_prob,          # Original prediction probability
    cf_prob,                # CF prediction probability
    target=0,               # Target class
    figsize=(16, 18),
    lead_names=None,
    save_file_path=None,
    dpi=350
)
```

### plot_counterfactual_diff_with_attribution()

Combine counterfactual overlay with attribution heatmap.

```python
plot_counterfactual_diff_with_attribution(
    original_ecg,           # (n_leads, seq_length)
    cf_ecg,                 # (n_leads, seq_length)
    attribution,            # Attribution array
    lead_idx=1,
    bin_size=50,
    cmap="Reds",
    figsize=(14, 5),
    lead_names=None,
    original_prob=None,
    cf_prob=None,
    save_file_path=None,
    dpi=350
)
```

**Example:**

```python
from execg.visualizer import plot_counterfactual_diff_with_attribution

# Generate both attribution and counterfactual
attr_result = gradcam.explain(ecg_tensor, target=1, target_layers="conv3")
cf_ecg, cf_prob, info = cf_explainer.explain(ecg_tensor, target=1, target_value=0.9)

plot_counterfactual_diff_with_attribution(
    original_ecg=ecg_tensor.squeeze().cpu().numpy(),
    cf_ecg=cf_ecg,
    attribution=attr_result["results"].squeeze(),
    lead_idx=1,
    original_prob=info["original_prob"],
    cf_prob=cf_prob
)
```

---

## TCAV Visualization

### plot_tcav_scores()

Plot TCAV scores as a heatmap.

```python
fig, ax = plot_tcav_scores(
    result_dict,            # Output from TCAV.explain()
    cmap="coolwarm",
    figsize=None,           # Auto-calculated if None
    save_file_path=None,
    dpi=350
)
```

**Example:**

```python
from execg.visualizer import plot_tcav_scores

results = tcav.explain(inputs, target=1)

fig, ax = plot_tcav_scores(results, cmap="RdYlGn")
ax.set_title("TCAV Scores for AFib Detection")
plt.show()
```

### plot_tcav_bar()

Plot TCAV scores as a bar chart for a specific layer.

```python
fig, ax = plot_tcav_bar(
    result_dict,            # Output from TCAV.explain()
    layer,                  # Layer name to plot
    figsize=(10, 5),
    colors=None,            # Custom colors
    save_file_path=None,
    dpi=350
)
```

**Example:**

```python
from execg.visualizer import plot_tcav_bar

fig, ax = plot_tcav_bar(results, layer="conv3")
plt.show()
```

### plot_tcav_comparison()

Compare TCAV scores across multiple layers.

```python
fig, axes = plot_tcav_comparison(
    result_dict,            # Output from TCAV.explain()
    layers,                 # List of layer names
    figsize=None,           # Auto-calculated if None
    save_file_path=None,
    dpi=350
)
```

**Example:**

```python
from execg.visualizer import plot_tcav_comparison

fig, axes = plot_tcav_comparison(
    results,
    layers=["conv2", "conv3", "conv4"]
)
plt.suptitle("TCAV Scores by Layer")
plt.show()
```

---

## ECG Chart Visualization

### plot_ecg_chart()

Create clinical-style 12-lead ECG chart with optional XAI overlays.

```python
fig, axes = plot_ecg_chart(
    ecg,                    # (1, n_leads, length) or (n_leads, length)
    sample_rate=250,
    cf_ecg=None,            # Counterfactual overlay
    attribution=None,       # Attribution heatmap
    title="Electrocardiogram",
    meta=None,              # Metadata dict to display
    style="clinical",       # "clinical" (red grid) or "paper" (gray grid)
    columns=4,
    row_height=6,
    lead_names=None,
    partial_continuous=True,
    show_lead_name=True,
    show_separate_line=True,
    show_grid=True,
    show_full_single_last=False,  # Show full Lead II at bottom
    show_calibration=False,
    ecg_linewidth=0.5,
    cf_color=(1, 0, 0),
    cf_alpha=0.7,
    cf_label="CF",
    attr_bin_size=25,
    attr_cmap="Reds",
    figsize=None,           # Auto-calculated if None
    save_path=None,
    dpi=300
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ecg` | np.ndarray | required | ECG signal array |
| `sample_rate` | float | `250` | Sampling rate in Hz |
| `cf_ecg` | np.ndarray | `None` | Counterfactual ECG to overlay |
| `attribution` | np.ndarray | `None` | Attribution scores for heatmap |
| `style` | str | `"clinical"` | `"clinical"` (red) or `"paper"` (gray) |
| `columns` | int | `4` | Number of columns for layout |
| `partial_continuous` | bool | `True` | Show continuous time segments |
| `show_full_single_last` | bool | `False` | Show full Lead II at bottom |
| `show_calibration` | bool | `False` | Show calibration pulse |

**Example - Basic:**

```python
from execg.visualizer import plot_ecg_chart

fig, axes = plot_ecg_chart(
    ecg,
    sample_rate=250,
    title="12-Lead ECG",
    style="clinical"
)
```

**Example - With XAI Overlays:**

```python
meta = {
    "Patient": "P001",
    "Original Pred": "Normal (0.95)",
    "CF Pred": "AFib (0.82)",
}

fig, axes = plot_ecg_chart(
    ecg,
    sample_rate=250,
    cf_ecg=cf_ecg,
    attribution=attribution,
    meta=meta,
    title="ECG with XAI Analysis",
    style="clinical",
    show_full_single_last=True,
    show_calibration=True,
    save_path="ecg_xai_report.png"
)
```

### plot_ecg_comparison()

Compare multiple ECG signals side by side.

```python
fig, axes = plot_ecg_comparison(
    ecg_list,               # List of ECG arrays
    labels,                 # List of labels
    sample_rate=250,
    colors=None,
    lead_idx=1,             # int or list of indices
    title="ECG Comparison",
    figsize=(14, 4),
    save_path=None,
    dpi=300
)
```

**Example:**

```python
from execg.visualizer import plot_ecg_comparison

fig, axes = plot_ecg_comparison(
    ecg_list=[original_ecg, cf_ecg],
    labels=["Original", "Counterfactual"],
    lead_idx=[1, 6],  # Lead II and V1
    title="Original vs Counterfactual"
)
```

---

## Complete Example

```python
import torch
import numpy as np
from execg.models import TorchModelWrapper
from execg.attribution import GradCAM
from execg.counterfactual import StyleGANCF
from execg.visualizer import (
    plot_attribution,
    plot_counterfactual_overlay,
    plot_ecg_chart
)

# Setup
model = YourECGClassifier()
wrapper = TorchModelWrapper(model)

# Load ECG data
ecg_tensor = torch.randn(1, 12, 2500)

# Generate Attribution
gradcam = GradCAM(wrapper)
attr_result = gradcam.explain(ecg_tensor, target=1, target_layers="conv3")

# Generate Counterfactual
cf_explainer = StyleGANCF(
    model=wrapper,
    generator_name="stylegan250",
    model_dir="models/stylegan/",
    sampling_rate=250,
    download=True
)
cf_ecg, cf_prob, info = cf_explainer.explain(
    ecg_tensor, target=1, target_value=0.9
)

# Visualize Attribution
plot_attribution(
    ecg=attr_result["inputs"].squeeze(),
    attribution=attr_result["results"],
    title="GradCAM Attribution",
    lead_idx=1
)

# Visualize Counterfactual
plot_counterfactual_overlay(
    original_ecg=ecg_tensor.squeeze().cpu().numpy(),
    cf_ecg=cf_ecg,
    lead_idx=1,
    original_prob=info["original_prob"],
    cf_prob=cf_prob
)

# Clinical ECG Chart with all XAI
meta = {
    "Patient": "P001",
    "Attribution": "GradCAM",
    "Original": f"Normal ({info['original_prob']:.3f})",
    "CF": f"AFib ({cf_prob:.3f})",
}

fig, axes = plot_ecg_chart(
    ecg_tensor.squeeze().cpu().numpy(),
    sample_rate=250,
    cf_ecg=cf_ecg,
    attribution=attr_result["results"],
    meta=meta,
    title="Complete XAI Analysis",
    style="clinical",
    show_full_single_last=True,
    save_path="xai_report.png"
)
```
