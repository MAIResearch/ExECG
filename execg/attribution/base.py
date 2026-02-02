import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Optional, Dict, Any

from execg.models.wrapper import TorchModelWrapper


class AttributionBase(ABC):
    """Base class for ECG attribution methods.

    This abstract class defines the interface for all attribution methods
    that explain ECG model predictions by attributing importance to input features.
    """

    def __init__(self, model: TorchModelWrapper):
        """Initialize the attribution method.

        Args:
            model: TorchModelWrapper instance containing the model to explain.
        """
        self.model = model
        
    def _format_target(
        self,
        target: Optional[Union[int, List[int], torch.Tensor]],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Format the target to be compatible with the attribution method.

        Args:
            target: Target class(es) to explain.
            output: Model output tensor of shape (1, N).

        Returns:
            Formatted target tensor.
        """
        if target is None:
            if output.shape[-1] == 1:
                return torch.tensor([0], device=output.device)
            return torch.tensor([torch.argmax(output).item()], device=output.device)

        if isinstance(target, int):
            return torch.tensor([target], device=output.device)

        if isinstance(target, list):
            return torch.tensor(target, device=output.device)

        return target
    
    @abstractmethod
    def explain(
        self,
        inputs: torch.Tensor,
        target: Optional[Union[int, List[int], torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate and return attribution scores for the given inputs.

        Args:
            inputs: Input tensor of shape (batch_size, channels, seq_length).
            target: Target class(es) to explain. If None, uses the model's prediction.
            **kwargs: Additional method-specific parameters.

        Returns:
            Dictionary containing visualization data with keys 'inputs' and 'results'.
        """
        pass
    
    def normalize_attribution(
        self, attribution: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Normalize attribution scores to the range [0, 1].

        Args:
            attribution: Attribution scores as tensor or numpy array.

        Returns:
            Normalized attribution scores in the same format as input.
        """
        if isinstance(attribution, torch.Tensor):
            if torch.all(attribution == 0):
                return attribution

            min_val = torch.min(attribution)
            max_val = torch.max(attribution)

            if min_val == max_val:
                return torch.zeros_like(attribution)

            return (attribution - min_val) / (max_val - min_val)
        else:
            if np.all(attribution == 0):
                return attribution

            min_val = np.min(attribution)
            max_val = np.max(attribution)

            if min_val == max_val:
                return np.zeros_like(attribution)

            return (attribution - min_val) / (max_val - min_val) 
        
    def _compute_gradients(
        self, inputs: torch.Tensor, target_indices: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """Compute gradients of the target class with respect to the input.

        Args:
            inputs: Input tensor of shape (1, n_leads, seq_length).
            target_indices: Target class indices (tensor with single element).

        Returns:
            Gradients tensor of shape (1, n_leads, seq_length).
        """
        if isinstance(target_indices, np.ndarray):
            target_indices = torch.tensor(target_indices, device=self.model.device)

        idx = target_indices[0].item() if target_indices.numel() > 0 else 0
        sample_grad = self.model.get_gradients(inputs, target_class=idx)

        if isinstance(sample_grad, np.ndarray):
            sample_grad = torch.tensor(sample_grad, device=self.model.device)

        return sample_grad