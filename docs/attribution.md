# Attribution Methods

## Overview

Attribution methods explain model predictions by assigning importance scores to input features. For ECG models, this helps identify which parts of the signal (e.g., P-wave, QRS complex, T-wave) contribute most to the model's decision.

ExECG provides two main attribution classes:
- **GradCAM**: Gradient-weighted Class Activation Mapping for convolutional layers
- **SaliencyMap**: Gradient-based input attribution methods

## Quick Start

```python
import torch
from execg.models import TorchModelWrapper
from execg.attribution import GradCAM, SaliencyMap

# 1. Wrap your model
wrapper = TorchModelWrapper(your_model)

# 2. Create explainer
gradcam = GradCAM(wrapper)

# 3. Generate attribution
ecg_data = torch.randn(1, 12, 2500)  # (batch, n_leads, seq_length)
result = gradcam.explain(ecg_data, target=0, target_layer_name="conv3")

# 4. Access results
inputs = result["inputs"]      # numpy array (1, 12, 2500)
attribution = result["results"]  # numpy array
```

## API Reference

### GradCAM

Gradient-weighted Class Activation Mapping for 1D ECG signals.

#### Constructor

```python
GradCAM(model, relu_attributions=True, normalize_attributions=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | TorchModelWrapper | required | Wrapped model instance |
| `relu_attributions` | bool | `True` | Apply ReLU to keep only positive contributions |
| `normalize_attributions` | bool | `True` | Normalize output to [0, 1] |

#### explain()

```python
explain(inputs, target=None, target_layer_name=None, method="gradcam", **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | torch.Tensor | required | ECG tensor `(1, n_leads, seq_length)` |
| `target` | int, list, tensor | `None` | Target class index. If `None`, uses predicted class |
| `target_layer_name` | str | required | Name of convolutional layer for CAM |
| `method` | str | `"gradcam"` | Attribution method |

**Available methods:**

| Method | Description | Output Shape |
|--------|-------------|--------------|
| `"gradcam"` | Standard Grad-CAM | `(1, seq_length)` |
| `"gradcam_pp"` | Grad-CAM++ with improved weighting | `(1, seq_length)` |
| `"guided_gradcam"` | Combines Grad-CAM with guided backpropagation | `(1, n_leads, seq_length)` |

**Returns:**
```python
{
    "inputs": np.ndarray,   # Original input (1, n_leads, seq_length)
    "results": np.ndarray   # Attribution scores
}
```

---

### SaliencyMap

Gradient-based saliency maps for ECG signals.

#### Constructor

```python
SaliencyMap(model, absolute_gradients=True, normalize_gradients=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | TorchModelWrapper | required | Wrapped model instance |
| `absolute_gradients` | bool | `True` | Take absolute value of gradients |
| `normalize_gradients` | bool | `True` | Normalize output to [0, 1] |

#### explain()

```python
explain(inputs, target=None, method="vanilla_saliency", **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | torch.Tensor | required | ECG tensor `(1, n_leads, seq_length)` |
| `target` | int, list, tensor | `None` | Target class index. If `None`, uses predicted class |
| `method` | str | `"vanilla_saliency"` | Attribution method |

**Available methods:**

| Method | Description | Additional Parameters |
|--------|-------------|----------------------|
| `"vanilla_saliency"` | Simple input gradients | - |
| `"smooth_grad"` | Averaged gradients from noisy inputs | `n_samples` (default: 50), `noise_level` (default: 0.1) |
| `"integrated_gradients"` | Path-integrated gradients | `steps` (default: 50), `baseline` (default: zeros) |

**Returns:**
```python
{
    "inputs": np.ndarray,   # Original input (1, n_leads, seq_length)
    "results": np.ndarray   # Attribution scores (1, n_leads, seq_length)
}
```

## Usage Examples

### GradCAM

```python
from execg.attribution import GradCAM

gradcam = GradCAM(wrapper)

# Standard Grad-CAM
result = gradcam.explain(
    ecg_data,
    target=0,
    target_layer_name="conv3",
    method="gradcam"
)

# Grad-CAM++
result = gradcam.explain(
    ecg_data,
    target=0,
    target_layer_name="conv3",
    method="gradcam_pp"
)

# Guided Grad-CAM (returns per-lead attribution)
result = gradcam.explain(
    ecg_data,
    target=0,
    target_layer_name="conv3",
    method="guided_gradcam"
)
```

### SaliencyMap

```python
from execg.attribution import SaliencyMap

saliency = SaliencyMap(wrapper)

# Vanilla Saliency
result = saliency.explain(ecg_data, target=0, method="vanilla_saliency")

# SmoothGrad
result = saliency.explain(
    ecg_data,
    target=0,
    method="smooth_grad",
    n_samples=50,
    noise_level=0.1
)

# Integrated Gradients
result = saliency.explain(
    ecg_data,
    target=0,
    method="integrated_gradients",
    steps=50
)
```

## Finding Layer Names

To find available layer names for GradCAM:

```python
wrapper = TorchModelWrapper(model)
layers = wrapper.get_layer_names()
print(layers)
# ['', 'conv1', 'bn1', 'relu', 'conv2', 'conv3', 'fc1', 'fc2']

# Use the last convolutional layer
conv_layers = [l for l in layers if 'conv' in l.lower()]
target_layer = conv_layers[-1]  # e.g., "conv3"
```

## Visualization

ExECG provides built-in visualization functions for attribution results.

### plot_attribution()

Plot ECG signal with attribution heatmap below.

```python
from execg.visualization import plot_attribution

result = gradcam.explain(ecg_data, target=0, target_layer_name="conv3")

plot_attribution(
    ecg=result["inputs"].squeeze(),    # (12, seq_length)
    attribution=result["results"],
    title="GradCAM Attribution",
    lead_idx=[0, 1, 6],                # Leads I, II, V1
    bin_size=25,
    cmap="Reds"
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ecg` | np.ndarray | required | ECG signal `(n_leads, seq_length)` |
| `attribution` | np.ndarray | required | Attribution scores |
| `title` | str | required | Plot title |
| `lead_idx` | int, list, None | `None` | Leads to plot. `None` = all leads |
| `bin_size` | int | `10` | Samples per heatmap bin |
| `cmap` | str | `"Reds"` | Matplotlib colormap |
| `figsize` | tuple | `(14, 5)` | Figure size (width, height_per_lead) |
| `lead_names` | list | `None` | Custom lead names |

### plot_attribution_comparison()

Compare multiple attribution methods side by side.

```python
from execg.visualization import plot_attribution_comparison

# Generate attributions with different methods
gradcam_result = gradcam.explain(ecg_data, target=0, target_layer_name="conv3")
saliency_result = saliency.explain(ecg_data, target=0, method="vanilla_saliency")
smoothgrad_result = saliency.explain(ecg_data, target=0, method="smooth_grad")

# Compare methods
methods_dict = {
    "GradCAM": (gradcam_result["results"], "Reds"),
    "Saliency": (saliency_result["results"], "Blues"),
    "SmoothGrad": (smoothgrad_result["results"], "Greens"),
}

plot_attribution_comparison(
    ecg=ecg_data.squeeze().numpy(),
    methods_dict=methods_dict,
    lead_idx=1,           # Lead II
    bin_size=50
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ecg` | np.ndarray | required | ECG signal `(n_leads, seq_length)` |
| `methods_dict` | dict | required | `{method_name: (attribution, cmap)}` |
| `lead_idx` | int, list, None | `None` | Leads to plot |
| `bin_size` | int | `50` | Samples per heatmap bin |
| `figsize` | tuple | `(16, 8)` | Figure size |

### Utility Functions

```python
from execg.visualization import bin_attribution, normalize_attribution

# Bin attribution data for smoother visualization
binned = bin_attribution(attribution_1d, bin_size=25)

# Normalize multiple lead attributions globally
normalized_list = normalize_attribution([attr_lead0, attr_lead1, attr_lead2])
```
