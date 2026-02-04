# Counterfactual Explanation

## Overview

Counterfactual explanation answers the question: "What minimal changes to the input would change the model's prediction?" For ECG models, this reveals which signal characteristics drive the classification or regression output.

ExECG uses a StyleGAN-based approach to generate realistic counterfactual ECG signals by optimizing in the latent space.

> **Note**: This module requires **12-lead ECG** input because the StyleGAN generator is trained on 12-lead ECG data. For other lead configurations, use Attribution or TCAV methods instead.

## How It Works

1. **W-Inversion**: Encode the input ECG into StyleGAN's W+ latent space
2. **Latent Optimization**: Optimize the latent vector to change model prediction toward target value
3. **Generation**: Generate counterfactual ECG from optimized latent vector

## Quick Start

```python
import torch
from execg.models import TorchModelWrapper
from execg.counterfactual import StyleGANCF

# 1. Wrap your model
wrapper = TorchModelWrapper(your_model)

# 2. Create explainer with StyleGAN generator
cf_explainer = StyleGANCF(
    model=wrapper,
    generator_name="stylegan250",
    model_dir="models/stylegan/",
    sampling_rate=250,
    download=True  # Auto-download if not found
)

# 3. Generate counterfactual (input must be 10 seconds)
ecg_data = torch.randn(1, 12, 2500)  # (batch, n_leads, seq_length)
cf_ecg, cf_prob, etc = cf_explainer.explain(
    ecg_data,
    target=0,
    target_value=1.0
)
```

## API Reference

### StyleGANCF

StyleGAN-based counterfactual explanation generator for ECG.

#### Constructor

```python
StyleGANCF(
    model,
    generator_name,
    model_dir,
    sampling_rate,
    download=False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | TorchModelWrapper | required | Wrapped target model |
| `generator_name` | str | required | Generator name (e.g., `"stylegan250"`) |
| `model_dir` | str | required | Directory to store/load model files (code and weights) |
| `sampling_rate` | float | required | Input ECG sampling rate (Hz) |
| `download` | bool | `False` | Auto-download code and weights if not found |

**Available generators:**

| Generator | Output | Description |
|-----------|--------|-------------|
| `"stylegan250"` | 250Hz, 2500 samples | Standard 10-second ECG at 250Hz |

#### explain()

```python
explain(inputs, target, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | torch.Tensor | required | ECG tensor `(1, 12, seq_length)` - must be 10 seconds |
| `target` | int | required | Target output index |
| `target_value` | float | required | Desired prediction value |
| `inversion_steps` | int | `1000` | Steps for W-inversion |
| `inversion_lr` | float | `0.0005` | Learning rate for inversion |
| `cf_steps` | int | `500` | Steps for CF generation |
| `cf_lr` | float | `0.0005` | Learning rate for CF |
| `layer_use` | list | `None` | StyleGAN layers to optimize (`None` = all) |
| `verbose` | bool | `False` | Print progress |
| `show_plot` | bool | `False` | Show plots during optimization |

**Returns:**

```python
(cf_ecg, cf_prob, etc)
```

| Return | Type | Description |
|--------|------|-------------|
| `cf_ecg` | np.ndarray | Counterfactual ECG `(12, seq_length)` |
| `cf_prob` | float | Final prediction probability |
| `etc` | dict | Contains `all_cf` and `all_probs` for visualization |

## Input Requirements

- **Leads**: Must be exactly **12 leads** (StyleGAN is trained on 12-lead ECG)
- **Duration**: Must be exactly 10 seconds
- **Shape**: `(1, 12, seq_length)` or `(12, seq_length)`
- **Sampling Rate**: Any rate (auto-resampled to 250Hz internally)

| Input Rate | Input Shape | Internal Shape | Output Shape |
|------------|-------------|----------------|--------------|
| 250Hz | `(1, 12, 2500)` | `(1, 12, 2500)` | `(12, 2500)` |
| 500Hz | `(1, 12, 5000)` | `(1, 12, 2500)` | `(12, 5000)` |

## Usage Examples

### Classification Model

```python
from execg.counterfactual import StyleGANCF

cf_explainer = StyleGANCF(
    model=wrapper,
    generator_name="stylegan250",
    model_dir="models/stylegan/",
    sampling_rate=250
)

# Get original prediction
original_pred = wrapper.predict(ecg_data)
original_class = original_pred.argmax().item()
original_prob = original_pred[0, original_class].item()

# Generate CF for opposite class
target_class = 1 - original_class
cf_ecg, cf_prob, etc = cf_explainer.explain(
    ecg_data,
    target=target_class,
    target_value=1.0,
    inversion_steps=500,
    cf_steps=300,
    verbose=True
)

print(f"Original: class {original_class} ({original_prob:.3f})")
print(f"CF: class {target_class} ({cf_prob:.3f})")
```

### Regression Model

```python
# Get original prediction
original_value = wrapper.predict(ecg_data)[0, 0].item()

# Generate CF for higher value
cf_ecg, cf_prob, etc = cf_explainer.explain(
    ecg_data,
    target=0,  # Regression has single output
    target_value=original_value + 1.0,
    inversion_steps=500,
    cf_steps=300
)

print(f"Original: {original_value:.3f}")
print(f"CF: {cf_prob:.3f}")
```

### Layer-Specific Optimization

StyleGAN layers control different ECG characteristics:
- **Lower layers (0-4)**: R-R intervals, rhythm
- **Higher layers (5-9)**: QRS morphology, wave shapes

```python
# Only optimize high-level features (morphology)
cf_ecg, cf_prob, etc = cf_explainer.explain(
    ecg_data,
    target=0,
    target_value=1.0,
    layer_use=[6, 7, 8, 9]  # Higher layers only
)

# Only optimize rhythm features
cf_ecg, cf_prob, etc = cf_explainer.explain(
    ecg_data,
    target=0,
    target_value=1.0,
    layer_use=[0, 1, 2, 3]  # Lower layers only
)
```

## Parameter Tuning

### Inversion Quality

If reconstruction is poor:
- Increase `inversion_steps` (e.g., 1000-2000)
- Decrease `inversion_lr` (e.g., 0.0001)

### Counterfactual Quality

If CF doesn't reach target:
- Increase `cf_steps` (e.g., 500-1000)
- Adjust `cf_lr` (0.0001-0.001)

If CF is too different from original:
- Use `layer_use` to restrict optimization
- Decrease `cf_steps`

## Visualization

ExECG provides built-in visualization functions for counterfactual results.

### plot_counterfactual_overlay()

Overlay original and counterfactual ECG.

```python
from execg.visualizer import plot_counterfactual_overlay

plot_counterfactual_overlay(
    original_ecg=ecg_data.squeeze().numpy(),
    cf_ecg=cf_ecg,
    lead_idx=1,  # Lead II
    figsize=(14, 4)
)
```

### plot_counterfactual_diff()

Show original, counterfactual, and difference.

```python
from execg.visualizer import plot_counterfactual_diff

plot_counterfactual_diff(
    original_ecg=ecg_data.squeeze().numpy(),
    cf_ecg=cf_ecg,
    lead_idx=1,  # Lead II
    figsize=(14, 6)
)
```

### plot_counterfactual_progress()

Visualize optimization progress.

```python
from execg.visualizer import plot_counterfactual_progress

cf_ecg, cf_prob, etc = cf_explainer.explain(...)

plot_counterfactual_progress(
    all_probs=etc["all_probs"],
    target_value=1.0,
    target=0
)
```

### plot_counterfactual_evolution()

Show ECG changes during optimization.

```python
from execg.visualizer import plot_counterfactual_evolution

plot_counterfactual_evolution(
    all_cf=etc["all_cf"],
    all_probs=etc["all_probs"],
    lead_idx=1,
    n_steps=5  # Show 5 intermediate steps
)
```

### plot_counterfactual_all_leads()

Compare all 12 leads.

```python
from execg.visualizer import plot_counterfactual_all_leads

original_prob = wrapper.predict(ecg_data)[0, target_class].item()

plot_counterfactual_all_leads(
    original_ecg=ecg_data.squeeze().numpy(),
    cf_ecg=cf_ecg,
    original_prob=original_prob,
    cf_prob=cf_prob,
    target=target_class,
    figsize=(16, 18)
)
```

## Complete Example

```python
import torch
from execg.models import TorchModelWrapper
from execg.counterfactual import StyleGANCF
from execg.visualizer import (
    plot_counterfactual_overlay,
    plot_counterfactual_diff,
    plot_counterfactual_progress
)

# Setup
model = YourECGClassifier()
wrapper = TorchModelWrapper(model)

cf_explainer = StyleGANCF(
    model=wrapper,
    generator_name="stylegan250",
    model_dir="models/stylegan/",
    sampling_rate=250
)

# Generate counterfactual
ecg_data = torch.randn(1, 12, 2500)
cf_ecg, cf_prob, etc = cf_explainer.explain(
    ecg_data,
    target=1,
    target_value=0.9,
    verbose=True
)

# Visualize
plot_counterfactual_overlay(ecg_data.squeeze().numpy(), cf_ecg, lead_idx=1)
plot_counterfactual_diff(ecg_data.squeeze().numpy(), cf_ecg, lead_idx=1)
plot_counterfactual_progress(etc["all_probs"], target_value=0.9, target=1)
```
