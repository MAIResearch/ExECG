# TCAV (Testing with Concept Activation Vectors)

## Overview

TCAV quantifies the influence of user-defined concepts on model predictions. For ECG models, this helps understand how clinically meaningful concepts (e.g., atrial fibrillation, T-wave abnormality) affect the model's decision-making process.

ExECG wraps Captum's TCAV implementation and provides ECG-specific concept dataset generation from PhysioNet2021.

## How It Works

1. **Concept Definition**: Define concepts as sets of ECG samples containing specific features
2. **CAV Training**: Train linear classifiers to distinguish concept examples from random examples
3. **Sensitivity Calculation**: Measure how model predictions change when moving toward the concept direction
4. **Statistical Testing**: Compare against random concepts for significance

## Quick Start

```python
import torch
from execg.concept import TCAV

# Initialize TCAV
tcav = TCAV(
    model=your_model,
    model_layers_list=["conv3"],
    model_input_sampling_rate=250,
    model_input_duration=10,
    data_name="physionet2021",
    data_dir="/path/to/physionet_numpy",
    target_concepts=["atrial fibrillation", "sinus rhythm"]
)

# Run TCAV analysis
inputs = torch.randn(10, 12, 2500)  # Batch of ECG samples
results = tcav.explain(inputs, target=1)

# Visualize results
fig, ax = TCAV.plot_tcav_scores(results)
```

## API Reference

### TCAV

#### Constructor

```python
TCAV(
    model,
    model_layers_list,
    model_input_sampling_rate,
    model_input_duration,
    data_name,
    data_dir,
    target_concepts,
    num_random_concepts=10,
    num_samples=200,
    random_seed=42
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | required | PyTorch model to explain |
| `model_layers_list` | List[str] | required | Layers to analyze |
| `model_input_sampling_rate` | float | required | Model input sampling rate (Hz) |
| `model_input_duration` | float | required | Model input duration (seconds) |
| `data_name` | str | required | Dataset name (`"physionet2021"`) |
| `data_dir` | str | required | Directory containing `.npz` ECG files |
| `target_concepts` | List[str] | required | Concept names to analyze |
| `num_random_concepts` | int | `10` | Number of random concepts for baseline |
| `num_samples` | int | `200` | Samples per concept |
| `random_seed` | int | `42` | Random seed for reproducibility |

#### explain()

```python
explain(inputs, target=1, n_steps=50, score_type="sign_count")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | torch.Tensor | required | Input tensor `(batch, n_leads, seq_length)` |
| `target` | int | `1` | Target class index |
| `n_steps` | int | `50` | Steps for integrated gradients |
| `score_type` | str | `"sign_count"` | Score type (`"sign_count"` or `"magnitude"`) |

**Returns:**

```python
{
    "concept_name": {
        "layer_name": (mean, (lower_ci, upper_ci)),
        ...
    },
    ...
}
```

## Available Concepts (PhysioNet2021)

The following concepts are available from the PhysioNet2021 dataset:

| Category | Concepts |
|----------|----------|
| **Rhythm** | atrial fibrillation, atrial flutter, sinus rhythm, sinus bradycardia, sinus tachycardia |
| **Conduction** | bundle branch block, 1st degree av block, left bundle branch block, right bundle branch block |
| **Morphology** | t wave abnormal, t wave inversion, qwave abnormal, st depression, st elevation |
| **Intervals** | prolonged qt interval, short qt interval, prolonged pr interval |
| **Voltage** | low qrs voltages, left ventricular hypertrophy |

See `execg/concept/labels/physionet2021.csv` for the complete list.

## Data Format

ECG data files should be in `.npz` format:

```python
# File structure: {filename}.npz
# Contents: arr_0 with shape (12, samples)
# Example: A0001.npz contains 12-lead ECG at 500Hz

import numpy as np
data = np.load("A0001.npz")
ecg = data["arr_0"]  # Shape: (12, 5000) for 10s at 500Hz
```

## Interpretation

TCAV scores range from 0 to 1:

| Score | Interpretation |
|-------|---------------|
| **> 0.5** | Concept positively influences the prediction |
| **< 0.5** | Concept negatively influences the prediction |
| **≈ 0.5** | Concept has no significant influence |

Confidence intervals help assess statistical significance. A score is significant if the confidence interval does not include 0.5.

## Usage Examples

### Basic Analysis

```python
from execg.concept import TCAV

tcav = TCAV(
    model=afib_model,
    model_layers_list=["conv2", "conv3", "conv4"],
    model_input_sampling_rate=250,
    model_input_duration=10,
    data_name="physionet2021",
    data_dir="/path/to/data",
    target_concepts=[
        "atrial fibrillation",
        "sinus rhythm",
        "atrial flutter"
    ],
    num_random_concepts=10,
    num_samples=200
)

# Get test samples
test_inputs = torch.randn(20, 12, 2500)

# Run TCAV for AFib class (target=1)
results = tcav.explain(test_inputs, target=1)

# Access specific scores
afib_score = results["atrial fibrillation"]["conv3"]
mean, (lower, upper) = afib_score
print(f"AFib concept score: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
```

### Multi-Layer Analysis

```python
# Analyze multiple layers to see where concepts are important
tcav = TCAV(
    model=model,
    model_layers_list=["conv1", "conv2", "conv3", "conv4"],
    ...
)

results = tcav.explain(inputs, target=1)

# Check concept importance across layers
for layer in ["conv1", "conv2", "conv3", "conv4"]:
    mean, _ = results["atrial fibrillation"][layer]
    print(f"{layer}: {mean:.3f}")
```

## Visualization

ExECG provides three visualization functions for TCAV results.

### plot_tcav_scores()

Plot TCAV scores as a heatmap.

```python
from execg.visualization import plot_tcav_scores

fig, ax = plot_tcav_scores(
    result_dict=results,
    cmap="coolwarm",
    figsize=(10, 6)
)
plt.savefig("tcav_heatmap.png")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result_dict` | dict | required | Output from `explain()` |
| `cmap` | str | `"coolwarm"` | Matplotlib colormap |
| `figsize` | tuple | auto | Figure size |

**Returns:** Tuple of `(fig, ax)` matplotlib objects.

### plot_tcav_bar()

Plot TCAV scores as a bar chart for a specific layer.

```python
from execg.visualization import plot_tcav_bar

fig, ax = plot_tcav_bar(
    result_dict=results,
    layer="conv3",
    figsize=(10, 5)
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result_dict` | dict | required | Output from `explain()` |
| `layer` | str | required | Layer name to plot |
| `figsize` | tuple | `(10, 5)` | Figure size |
| `colors` | list | `None` | Custom colors for bars |

### plot_tcav_comparison()

Compare TCAV scores across multiple layers.

```python
from execg.visualization import plot_tcav_comparison

fig, axes = plot_tcav_comparison(
    result_dict=results,
    layers=["conv2", "conv3", "conv4"],
    figsize=(18, 5)
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result_dict` | dict | required | Output from `explain()` |
| `layers` | List[str] | required | Layers to compare |
| `figsize` | tuple | auto | Figure size |

## Complete Example

```python
import torch
from execg.concept import TCAV
from execg.visualization import (
    plot_tcav_scores,
    plot_tcav_bar,
    plot_tcav_comparison
)

# Setup
model = AFibClassifier()
model.load_state_dict(torch.load("afib_model.pt"))
model.eval()

# Initialize TCAV
tcav = TCAV(
    model=model,
    model_layers_list=["conv2", "conv3", "conv4"],
    model_input_sampling_rate=250,
    model_input_duration=10,
    data_name="physionet2021",
    data_dir="/data/physionet",
    target_concepts=[
        "atrial fibrillation",
        "sinus rhythm",
        "atrial flutter",
        "sinus bradycardia"
    ],
    num_random_concepts=10,
    num_samples=200
)

# Run analysis
test_data = torch.randn(50, 12, 2500)
results = tcav.explain(test_data, target=1)

# Visualize - Heatmap
fig, ax = plot_tcav_scores(results)
plt.title("TCAV Scores for AFib Detection")
plt.savefig("tcav_heatmap.png")

# Visualize - Bar chart for specific layer
fig, ax = plot_tcav_bar(results, layer="conv3")
plt.savefig("tcav_conv3.png")

# Visualize - Layer comparison
fig, axes = plot_tcav_comparison(results, layers=["conv2", "conv3", "conv4"])
plt.savefig("tcav_layers.png")

# Print results
print("\nTCAV Results (conv3):")
for concept, layers in results.items():
    mean, (lower, upper) = layers["conv3"]
    significance = "***" if lower > 0.5 or upper < 0.5 else ""
    print(f"  {concept}: {mean:.3f} [{lower:.3f}, {upper:.3f}] {significance}")
```
