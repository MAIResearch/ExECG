from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from execg.misc import set_random_seed
from execg.models.wrapper import TorchModelWrapper

from .base import AttributionBase


class SaliencyMap(AttributionBase):
    """Vanilla Gradient/Saliency Map for ECG models.

    Saliency maps highlight the input features that have the greatest influence
    on the model's output by computing the gradient of the output with respect
    to the input.

    This implementation is adapted for 1D ECG signals.
    """

    def __init__(
        self,
        model: TorchModelWrapper,
        absolute: bool = True,
        normalize: bool = True,
        random_seed: Optional[int] = None,
    ):
        """Initialize SaliencyMap.

        Args:
            model: TorchModelWrapper instance containing the model to explain.
            absolute: Whether to take the absolute value of gradients.
            normalize: Whether to normalize the gradients.
            random_seed: Random seed for reproducibility (used in smooth_grad).
        """
        super().__init__(model)
        self.absolute = absolute
        self.normalize = normalize
        self.random_seed = random_seed
    
    def explain(
        self,
        inputs: torch.Tensor,
        target: Optional[Union[int, List[int], torch.Tensor]] = None,
        method: str = "vanilla_saliency",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate and return saliency map attribution scores for the given inputs.

        Args:
            inputs: Input tensor of shape (batch_size, channels, seq_length).
            target: Target class(es) to explain. If None, uses model's prediction.
            method: Attribution method ('vanilla_saliency', 'smooth_grad',
                'integrated_gradients').
            **kwargs: Additional parameters for the specific method.

        Returns:
            Dictionary with 'inputs' and 'results' as numpy arrays.

        Raises:
            ValueError: If method is not recognized.
        """
        inputs = inputs.to(self.model.device)
        outputs = self.model.predict(inputs, requires_grad=True)
        target_indices = self._format_target(target, outputs)

        if method == "vanilla_saliency":
            attributions = self._vanilla_saliency(inputs, target_indices, **kwargs)
        elif method == "smooth_grad":
            attributions = self._smooth_grad(inputs, target_indices, **kwargs)
        elif method == "integrated_gradients":
            attributions = self._integrated_gradients(inputs, target_indices, **kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Available methods: vanilla_saliency, smooth_grad, integrated_gradients"
            )

        if self.absolute:
            if isinstance(attributions, np.ndarray):
                attributions = np.abs(attributions)
            else:
                attributions = torch.abs(attributions)

        if self.normalize:
            attributions = self.normalize_attribution(attributions)

        if isinstance(attributions, torch.Tensor):
            attributions = attributions.detach().cpu().numpy()

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()

        return {"inputs": inputs, "results": attributions}
    
    def _vanilla_saliency(
        self,
        inputs: torch.Tensor,
        target_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Generate vanilla saliency map attribution scores for the given inputs.

        Args:
            inputs: Input tensor of shape (batch_size, channels, seq_length).
            target_indices: Target class indices to explain.
            **kwargs: Additional parameters.

        Returns:
            Attribution scores of shape (batch_size, channels, seq_length).
        """
        return self._compute_gradients(inputs, target_indices)
    
    def _smooth_grad(
        self,
        inputs: torch.Tensor,
        target_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Generate SmoothGrad attribution scores for the given inputs.

        SmoothGrad averages the gradients from multiple noisy versions of the input
        to reduce visual noise in the saliency maps.

        Args:
            inputs: Input tensor of shape (1, n_leads, seq_length).
            target_indices: Target class indices to explain.
            **kwargs: Additional parameters including 'n_samples' (default: 50),
                'noise_level' (default: 0.1), and 'random_seed' (default: None).

        Returns:
            Attribution scores of shape (1, n_leads, seq_length).
        """
        n_samples = kwargs.get('n_samples', 50)
        noise_level = kwargs.get('noise_level', 0.1)
        seed = kwargs.get('random_seed', self.random_seed)

        if seed is not None:
            set_random_seed(seed)

        x = inputs.clone().detach()

        min_val = torch.min(x)
        max_val = torch.max(x)
        stdev = noise_level * (max_val - min_val)

        total_gradients = torch.zeros_like(x)

        for _ in range(n_samples):
            noise = torch.randn_like(x) * stdev
            noisy_input = (x + noise).to(self.model.device)
            sample_gradients = self._compute_gradients(noisy_input, target_indices)
            total_gradients += sample_gradients

        return total_gradients / n_samples
    
    def _integrated_gradients(
        self,
        inputs: torch.Tensor,
        target_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Generate Integrated Gradients attribution scores for the given inputs.

        Integrated Gradients computes the path integral of the gradients along a
        straight line from a baseline to the input, providing a theoretically
        sound attribution method.

        Args:
            inputs: Input tensor of shape (1, n_leads, seq_length).
            target_indices: Target class indices to explain.
            **kwargs: Additional parameters including 'n_steps' (default: 50)
                and 'baseline' (default: None, uses zeros).

        Returns:
            Attribution scores of shape (1, n_leads, seq_length).
        """
        n_steps = kwargs.get('n_steps', 50)
        baseline = kwargs.get('baseline', None)

        x = inputs.clone().detach()

        if baseline is None:
            baseline = torch.zeros_like(x)

        if isinstance(baseline, np.ndarray):
            baseline = torch.tensor(baseline, device=x.device, dtype=x.dtype)

        total_gradients = torch.zeros_like(x)

        for step in range(n_steps):
            alpha = step / (n_steps - 1) if n_steps > 1 else 1.0
            interpolated_input = (baseline + alpha * (x - baseline)).to(self.model.device)
            sample_gradients = self._compute_gradients(interpolated_input, target_indices)
            total_gradients += sample_gradients

        avg_gradients = total_gradients / n_steps
        attributions = avg_gradients * (x - baseline)

        return attributions