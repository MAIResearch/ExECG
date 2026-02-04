"""Base class for model wrappers in ECG-XAI."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch


class BaseModelWrapper(ABC):
    """Abstract base class for all model wrappers in ECG-XAI.

    All XAI algorithms in ECG-XAI require a wrapped model that follows
    the standard input/output conventions defined in this class.

    Input Convention:
        - Shape: (1, n_leads, seq_length)
        - Example: (1, 12, 2500) for 12-lead ECG, 250Hz, 10 seconds

    Output Convention:
        - Regression: (1, 1)
        - Binary: (1, 2) probabilities
        - Multiclass/Multilabel: (1, N) probabilities
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device where the model is located."""
        pass

    @abstractmethod
    def predict(
        self,
        inputs: torch.Tensor,
        target: Optional[int] = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Make predictions using the wrapped model.

        Args:
            inputs: Input ECG tensor of shape (1, n_leads, seq_length).
            target: If specified, return only the output at this index.
            requires_grad: If True, enable gradient computation.

        Returns:
            Model predictions as torch.Tensor with shape (1, N).
        """
        pass

    @abstractmethod
    def get_gradients(
        self, inputs: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """Calculate gradients of the output with respect to inputs.

        Args:
            inputs: Input ECG tensor of shape (1, n_leads, seq_length).
            target_class: Target class index for gradient calculation.

        Returns:
            Gradients as numpy array with shape (1, n_leads, seq_length).
        """
        pass

    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """Get names of all available layers in the model.

        Returns:
            List of layer names.
        """
        pass

    def _validate_input(self, inputs: torch.Tensor) -> None:
        """Validate input tensor shape and type.

        Args:
            inputs: Input tensor to validate.

        Raises:
            TypeError: If inputs is not a torch.Tensor.
            ValueError: If inputs does not have 3 dimensions (1, n_leads, seq_length).
        """
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(inputs).__name__}")
        if inputs.dim() != 3:
            raise ValueError(
                f"Expected 3D tensor (1, n_leads, seq_length), got {inputs.dim()}D tensor"
            )
        if inputs.size(0) != 1:
            raise ValueError(
                f"Expected batch size 1 and 3D tensor (1, n_leads, seq_length), got {inputs.size(0)}"
            )
