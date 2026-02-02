# Model Wrapper

## Overview

The `TorchModelWrapper` is the foundation of ExECG. It provides a standardized interface between your PyTorch model and all XAI methods. By wrapping your model, you gain access to gradient computation, layer introspection, and consistent input/output handling.

**Why do we need a wrapper?**

Different models have different input formats, output formats, and internal structures. The wrapper normalizes these differences so that all XAI methods can work with any model without modification.

```python
from execg.models import TorchModelWrapper

wrapper = TorchModelWrapper(model)
```

## Input/Output Convention

ExECG follows a standard convention for ECG data:

### Input Format

| Property | Format | Example |
|----------|--------|---------|
| Shape | `(1, n_leads, seq_length)` | `(1, 12, 2500)` |
| Type | `torch.Tensor` | `torch.float32` |
| Batch Size | Always 1 | Single sample processing |

### Output Format

| Task | Shape | Description |
|------|-------|-------------|
| Regression | `(1, 1)` | Single continuous value |
| Binary Classification | `(1, 2)` | Two class probabilities |
| Multi-class | `(1, N)` | N class probabilities |
| Multi-label | `(1, N)` | N independent probabilities |

## Basic Usage

### Simple Wrapping

If your model already follows the convention (input: `(1, n_leads, seq_length)`, output: `(1, N)`):

```python
import torch
from execg.models import TorchModelWrapper

# Your model
model = YourECGClassifier()
model.load_state_dict(torch.load("model.pt"))

# Wrap it
wrapper = TorchModelWrapper(model)

# Make predictions
ecg = torch.randn(1, 12, 2500)
output = wrapper.predict(ecg)  # (1, 2) for binary classification
```

### Custom Input Format (preprocess)

If your model expects a different input format, use the `preprocess` parameter:

```python
# Model expects (1, seq_length, n_leads) instead of (1, n_leads, seq_length)
wrapper = TorchModelWrapper(
    model,
    preprocess=lambda x: x.transpose(1, 2)
)

# Input: (1, 12, 2500) → preprocess → (1, 2500, 12) → model
```

**Common preprocess examples:**

```python
# Transpose dimensions
preprocess=lambda x: x.transpose(1, 2)

# Add channel dimension: (1, 12, 2500) → (1, 1, 12, 2500)
preprocess=lambda x: x.unsqueeze(1)

# Normalize input
preprocess=lambda x: (x - x.mean()) / x.std()

# Combine multiple transformations
def my_preprocess(x):
    x = (x - x.mean()) / x.std()  # Normalize
    x = x.transpose(1, 2)          # Transpose
    return x

wrapper = TorchModelWrapper(model, preprocess=my_preprocess)
```

### Custom Output Format (postprocess)

If your model's output doesn't follow the convention, use the `postprocess` parameter:

```python
# Model outputs single logit, convert to binary probabilities
wrapper = TorchModelWrapper(
    model,
    postprocess=lambda x: torch.stack([1 - x.sigmoid(), x.sigmoid()], dim=-1).squeeze(-2)
)

# Output: logit → postprocess → (1, 2) probabilities
```

**Common postprocess examples:**

```python
# Logit to binary probabilities
postprocess=lambda x: torch.cat([1-x.sigmoid(), x.sigmoid()], dim=-1)

# Apply softmax
postprocess=lambda x: torch.softmax(x, dim=-1)

# Reshape output
postprocess=lambda x: x.view(1, -1)

# Extract specific outputs from tuple
postprocess=lambda x: x[0] if isinstance(x, tuple) else x
```

### Combined Example

```python
# Model expects (batch, length, leads) and outputs raw logits
wrapper = TorchModelWrapper(
    model,
    preprocess=lambda x: x.transpose(1, 2),
    postprocess=lambda x: torch.softmax(x, dim=-1)
)
```

## Core Methods

### predict()

Make predictions with the wrapped model.

```python
output = wrapper.predict(
    inputs,                  # (1, 12, seq_length)
    output_idx=None,         # Optional: select specific output
    requires_grad=False      # Enable gradient tracking
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | `torch.Tensor` | required | ECG tensor `(1, n_leads, seq_length)` |
| `output_idx` | `int` or `None` | `None` | If set, returns only `output[:, idx:idx+1]` |
| `requires_grad` | `bool` | `False` | Enable gradient computation |

**Examples:**

```python
# Basic prediction
output = wrapper.predict(ecg)  # (1, 2)

# Get probability of class 0 only
prob_class0 = wrapper.predict(ecg, output_idx=0)  # (1, 1)

# Enable gradients for XAI methods
output = wrapper.predict(ecg, requires_grad=True)
```

### get_gradients()

Compute gradients of model output with respect to input.

```python
gradients = wrapper.get_gradients(
    inputs,              # (1, 12, seq_length)
    target_class=None    # Which output to differentiate
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | `torch.Tensor` | required | ECG tensor `(1, n_leads, seq_length)` |
| `target_class` | `int` or `None` | `None` | Class index. If `None`, uses `argmax` |

**Returns:** `np.ndarray` with shape `(1, n_leads, seq_length)`

**Examples:**

```python
# Gradient w.r.t. predicted class
grads = wrapper.get_gradients(ecg)

# Gradient w.r.t. specific class
grads = wrapper.get_gradients(ecg, target_class=1)

# For regression models
grads = wrapper.get_gradients(ecg, target_class=0)
```

### get_layer_names()

List all layers available for GradCAM analysis.

```python
layer_names = wrapper.get_layer_names()
```

**Returns:** `List[str]` of layer names

**Example:**

```python
layers = wrapper.get_layer_names()
print(layers)
# ['', 'conv1', 'bn1', 'relu', 'conv2', 'bn2', 'conv3', 'fc1', 'fc2']

# Use the last convolutional layer for GradCAM
target_layer = [l for l in layers if 'conv' in l][-1]
```

### get_layer_gradients()

Get activations and gradients of a specific layer (used internally by GradCAM).

```python
activations, gradients = wrapper.get_layer_gradients(
    inputs,           # (1, 12, seq_length)
    target_class,     # Class index
    layer_name        # Layer name from get_layer_names()
)
```

**Returns:** Tuple of `(activations, gradients)` as `np.ndarray`

### to()

Move the wrapper to a different device.

```python
wrapper.to(torch.device("cuda:0"))
wrapper.to("cpu")
```

## Device Handling

The wrapper automatically manages device placement:

```python
# Model on GPU
model = model.cuda()
wrapper = TorchModelWrapper(model)

# Check device
print(wrapper.device)  # cuda:0

# Input is automatically moved to model's device
ecg = torch.randn(1, 12, 2500)  # CPU tensor
output = wrapper.predict(ecg)    # Works! Input moved internally
```

## Error Handling

The wrapper validates inputs and provides helpful error messages:

```python
# Wrong type
wrapper.predict(np.random.randn(1, 12, 2500))
# TypeError: Expected torch.Tensor, got ndarray

# Wrong dimensions
wrapper.predict(torch.randn(12, 2500))
# ValueError: Expected 3D tensor (1, n_leads, seq_length), got 2D tensor

# Wrong batch size
wrapper.predict(torch.randn(4, 12, 2500))
# ValueError: Expected batch size 1 and 3D tensor (1, n_leads, seq_length), got 4
```

## Integration with XAI Methods

All XAI methods in ExECG accept a `TorchModelWrapper`:

```python
from execg.models import TorchModelWrapper
from execg.attribution import GradCAM, SaliencyMap
from execg.counterfactual import StyleGANCF

# Wrap once
wrapper = TorchModelWrapper(model)

# Use with any XAI method
gradcam = GradCAM(wrapper)
saliency = SaliencyMap(wrapper)
```

## Best Practices

### 1. Verify Output Shape

Before using XAI methods, verify your wrapper produces correct output shape:

```python
wrapper = TorchModelWrapper(model)
ecg = torch.randn(1, 12, 2500)
output = wrapper.predict(ecg)

print(f"Output shape: {output.shape}")
# Binary classification: (1, 2)
# Regression: (1, 1)
```

### 2. Check Layer Names for GradCAM

Find the appropriate layer before using GradCAM:

```python
layers = wrapper.get_layer_names()
conv_layers = [l for l in layers if 'conv' in l.lower()]
print(f"Available conv layers: {conv_layers}")
```

### 3. Test Gradients

Verify gradient computation works:

```python
grads = wrapper.get_gradients(ecg, target_class=0)
print(f"Gradient shape: {grads.shape}")  # Should match input shape
print(f"Gradient range: [{grads.min():.4f}, {grads.max():.4f}]")
```

## Complete Example

```python
import torch
import torch.nn as nn
from execg.models import TorchModelWrapper
from execg.attribution import GradCAM

# Define a simple model
class SimpleECGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, 7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, 7, padding=3)
        self.conv3 = nn.Conv1d(64, 128, 7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return torch.softmax(self.fc(x), dim=-1)

# Create and wrap model
model = SimpleECGClassifier()
wrapper = TorchModelWrapper(model)

# Verify
ecg = torch.randn(1, 12, 2500)
print(f"Output: {wrapper.predict(ecg)}")
print(f"Layers: {wrapper.get_layer_names()}")

# Use with GradCAM
gradcam = GradCAM(wrapper)
result = gradcam.explain(ecg, target=1, target_layer_name="conv3")
print(f"Attribution shape: {result['results'].shape}")
```
