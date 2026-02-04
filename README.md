<p align="center">
  <img src="docs/img/logo.png" alt="ExECG Logo" width="400">
</p>

# ExECG (Explainable AI for ECG models)
**Explainable AI Library for ECG Deep Learning Models**

ExECG provides a unified framework for interpreting ECG (electrocardiogram) deep learning models through three complementary explanation approaches.

## Philosophy

Understanding *why* an ECG model makes a prediction is as important as the prediction itself. ExECG addresses this through three distinct but complementary perspectives:

| Approach | Question Answered | Method |
|----------|-------------------|--------|
| **Attribution** | *Which parts of the signal influenced the prediction?* | GradCAM, SaliencyMap |
| **Counterfactual** | *What minimal changes would alter the prediction?* | StyleGAN-based generation |
| **Concept (TCAV)** | *How do clinical concepts influence the model?* | Concept Activation Vectors |

All methods share a unified interface built on the **Model Wrapper Pattern**, ensuring consistent behavior regardless of your model architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your PyTorch Model                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TorchModelWrapper                         │
│  • Standardized input/output interface                       │
│  • Gradient computation                                      │
│  • Layer introspection                                       │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ Attribution │     │Counterfactual│    │    TCAV     │
   │             │     │             │     │             │
   │ .explain()  │     │ .explain()  │     │ .explain()  │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Visualization                           │
│  • plot_attribution()        • plot_counterfactual_*()      │
│  • plot_attribution_comparison()  • plot_tcav_scores()      │
└─────────────────────────────────────────────────────────────┘
```

The wrapper pattern decouples explanation methods from model implementations, allowing any PyTorch ECG model to be explained without modification.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.7.0
- See `requirements.txt` for full dependencies

## Quick Start

### 1. Wrap Your Model

```python
import torch
from execg.models import TorchModelWrapper

# Any PyTorch model that takes (batch, n_leads, seq_length) input
model = YourECGModel()
wrapper = TorchModelWrapper(model)
```

### 2. Choose Your Explanation Method

**Attribution** - Identify important signal regions:
```python
from execg.attribution import GradCAM

explainer = GradCAM(wrapper)
result = explainer.explain(
    ecg_data,                    # (1, n_leads, seq_length)
    target=0,                    # class index
    target_layers="conv3"        # layer to analyze
)
attribution = result["results"]  # importance scores
```

**Counterfactual** - Generate alternative signals:
```python
from execg.counterfactual import StyleGANCF

explainer = StyleGANCF(
    wrapper,
    generator_name="stylegan250",
    weight_path="path/to/weights.pt",
    sampling_rate=250
)
cf_ecg, cf_prob, _ = explainer.explain(
    ecg_data,
    target=0,
    target_value=1.0
)
```

**TCAV** - Measure concept influence:
```python
from execg.concept import TCAV

explainer = TCAV(
    model=model,
    target_layers=["conv3"],
    sampling_rate=250,
    duration=10,
    data_name="physionet2021",
    data_dir="/path/to/data",
    target_concepts=["atrial fibrillation", "sinus rhythm"]
)
results = explainer.explain(ecg_batch, target=1)
```

## Project Structure

```
execg/
├── models/          # Model wrapper pattern
│   ├── base.py      # Abstract interface
│   └── wrapper.py   # PyTorch implementation
├── attribution/     # Feature importance methods
│   ├── gradcam.py   # GradCAM, GradCAM++, Guided GradCAM
│   └── saliency.py  # Vanilla, SmoothGrad, Integrated Gradients
├── counterfactual/  # Counterfactual generation
│   └── stylegan_cf.py
├── concept/         # Concept-based explanations
│   └── tcav/        # TCAV implementation
└── visualizer/   # Plotting utilities
```

## Documentation

- [Model Wrapper](docs/wrapper.md) - How to wrap your PyTorch model
- [Attribution Methods](docs/attribution.md) - GradCAM, SaliencyMap
- [Counterfactual Explanation](docs/counterfactual.md) - StyleGAN-based generation
- [TCAV Analysis](docs/tcav.md) - Concept-based explanations
- [Visualizer](docs/visualizer.md) - ECG chart and XAI visualization


## License

MIT License
