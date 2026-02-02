"""Model wrappers for ECG-XAI.

This module provides wrappers that standardize the interface for
different deep learning models, enabling consistent usage across
all XAI methods in the library.

Example:
    >>> from execg.models import TorchModelWrapper
    >>> import torch
    >>>
    >>> model = torch.jit.load("ecg_classifier.pt")
    >>> wrapper = TorchModelWrapper(model)
    >>> prediction = wrapper.predict(ecg_data)
"""

from .base import BaseModelWrapper
from .wrapper import TorchModelWrapper

__all__ = ["BaseModelWrapper", "TorchModelWrapper"]
