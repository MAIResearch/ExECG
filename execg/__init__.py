"""ECG-XAI: Explainable AI library for ECG deep learning models."""

from .models import TorchModelWrapper
from .attribution import AttributionBase, GradCAM, SaliencyMap

__version__ = "0.1.0"

__all__ = [
    "TorchModelWrapper",
    "AttributionBase",
    "GradCAM",
    "SaliencyMap",
]
