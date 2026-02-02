"""PyTorch model wrapper for ECG-XAI."""

from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModelWrapper


class TorchModelWrapper(BaseModelWrapper):
    """Wrapper for PyTorch models to use with ECG-XAI.

    This wrapper provides a standardized interface for PyTorch models,
    enabling gradient computation and consistent input/output handling
    for all XAI methods.

    Input Convention (XAI repo standard):
        (1, n_leads, seq_length) e.g., (1, 12, 2500)
        Use `preprocess` if your model expects different format.

    Output Convention (XAI repo standard):
        (1, N) where N is:
        - Regression: N=1
        - Binary: N=2 (probabilities)
        - Multiclass/Multilabel: N=num_classes (probabilities)
        Use `postprocess` if your model outputs different format.

    Example:
        >>> wrapper = TorchModelWrapper(model)
        >>> ecg = torch.randn(1, 12, 2500)  # (1, lead, length)
        >>> prediction = wrapper.predict(ecg)  # (1, N)

        # Model expects (1, length, lead)
        >>> wrapper = TorchModelWrapper(
        ...     model,
        ...     preprocess=lambda x: x.transpose(1, 2)
        ... )

        # Model outputs single logit, convert to (1, 2) binary probs
        >>> wrapper = TorchModelWrapper(
        ...     model,
        ...     postprocess=lambda x: torch.cat([1-x.sigmoid(), x.sigmoid()], dim=-1)
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Initialize the wrapper with a PyTorch model.

        Args:
            model: PyTorch model instance (nn.Module or TorchScript).
                The model will be set to eval mode automatically.
            preprocess: Optional function to transform input (1, lead, length)
                to the format expected by your model.
            postprocess: Optional function to transform model output
                to standard format (1, N).
        """
        self.model = model
        self._device = next(model.parameters()).device
        self.model.to(self._device)
        self.model.eval()
        self._preprocess = preprocess
        self._postprocess = postprocess

    @property
    def device(self) -> torch.device:
        """Return the device where the model is located."""
        return self._device

    def predict(
        self,
        inputs: torch.Tensor,
        output_idx: Optional[int] = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Make predictions using the wrapped model.

        Args:
            inputs: Input ECG tensor of shape (1, n_leads, seq_length).
                Example: (1, 12, 2500) for 12-lead, 250Hz, 10 seconds.
            output_idx: If specified, return only the output at this index.
            requires_grad: If True, enable gradient computation.

        Returns:
            Model predictions as torch.Tensor with shape (1, N).
            - Regression: (1, 1)
            - Binary: (1, 2)
            - Multiclass: (1, num_classes)
            - If output_idx specified: (1, 1)
        """
        self._validate_input(inputs)

        x = self._preprocess(inputs) if self._preprocess else inputs
        x = x.to(self._device)

        if requires_grad:
            outputs = self.model(x)
        else:
            with torch.no_grad():
                outputs = self.model(x)

        if self._postprocess:
            outputs = self._postprocess(outputs)

        if output_idx is not None:
            return outputs[:, output_idx : output_idx + 1]

        return outputs

    def get_gradients(
        self, inputs: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """Calculate gradients of the output with respect to inputs.

        Args:
            inputs: Input ECG tensor of shape (1, n_leads, seq_length).
            target_class: Target class index for gradient calculation.
                - Classification: If None, uses argmax.
                - Regression: Should be None.

        Returns:
            Gradients as numpy array with shape (1, n_leads, seq_length).
        """
        self.model.zero_grad()

        x = inputs.clone().requires_grad_(True)
        outputs = self.predict(x, requires_grad=True)

        if target_class is None:
            target_class = outputs.argmax().item()
        outputs[0, target_class].backward()

        return x.grad.cpu().numpy()

    def get_layer_names(self) -> List[str]:
        """Get names of all available layers in the model.

        Returns:
            List of layer names.
        """
        return [name for name, _ in self.model.named_modules()]

    def get_layer_gradients(
        self, inputs: torch.Tensor, target_class: int, layer_name: str
    ) -> tuple:
        """Get activation and gradients of a specific layer for Grad-CAM.

        Args:
            inputs: Input ECG tensor of shape (1, n_leads, seq_length).
            target_class: Target class index for gradient calculation.
            layer_name: Name of the target convolutional layer.

        Returns:
            Tuple of (activations, gradients) as numpy arrays.
            Both have shape (1, num_channels, spatial_length).
        """
        self.model.zero_grad()

        activations = None
        gradients = None

        def forward_hook(_, __, out):
            nonlocal activations
            activations = out

        def backward_hook(_, __, grad_out):
            nonlocal gradients
            gradients = grad_out[0]

        target_layer = dict(self.model.named_modules()).get(layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model.")

        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        try:
            x = inputs.clone().requires_grad_(True)
            outputs = self.predict(x, requires_grad=True)
            outputs[0, target_class].backward()
        finally:
            fh.remove()
            bh.remove()

        return activations.detach().cpu().numpy(), gradients.detach().cpu().numpy()

    def __repr__(self) -> str:
        """Return string representation of the wrapper."""
        return f"TorchModelWrapper(model={self.model.__class__.__name__}, device={self._device})"

    def to(self, device):
        self._device = device
        self.model.to(self._device)
