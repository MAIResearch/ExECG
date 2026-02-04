import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any

from execg.models.wrapper import TorchModelWrapper
from .base import AttributionBase


class GradCAM(AttributionBase):
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for ECG models.

    Grad-CAM uses the gradients of a target concept flowing into the final
    convolutional layer to produce a coarse localization map highlighting
    important regions in the input for prediction.

    This implementation is adapted for 1D ECG signals.
    """

    def __init__(
        self,
        model: TorchModelWrapper,
        absolute: bool = True,
        normalize: bool = True,
    ):
        """Initialize GradCAM.

        Args:
            model: TorchModelWrapper instance containing the model to explain.
            absolute: Whether to apply ReLU to the attribution scores.
            normalize: Whether to normalize the attribution scores.
        """
        super().__init__(model)
        self.absolute = absolute
        self.normalize = normalize

    def explain(
        self,
        inputs: torch.Tensor,
        target: Optional[Union[int, List[int], torch.Tensor]] = None,
        target_layers: str = None,
        method: str = "gradcam",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate and return GradCAM attribution scores for the given inputs.

        Args:
            inputs: Input tensor of shape (batch_size, channels, seq_length).
            target: Target class(es) to explain. If None, uses model's prediction.
            target_layers: Name of the layer to use for GradCAM analysis.
            method: Attribution method ('gradcam', 'guided_gradcam', 'gradcam_pp').
            **kwargs: Additional parameters for the specific method.

        Returns:
            Dictionary with 'inputs' and 'results' as numpy arrays.

        Raises:
            ValueError: If target_layers is None or not found in model.
        """
        if target_layers is None:
            raise ValueError("target_layers must be provided for GradCAM")

        if target_layers not in self.model.get_layer_names():
            raise ValueError(
                f"Layer '{target_layers}' not found in model. "
                f"Available layers: {self.model.get_layer_names()}"
            )

        inputs = inputs.to(self.model.device)
        outputs = self.model.predict(inputs, requires_grad=True)
        target_indices = self._format_target(target, outputs)

        if method == "gradcam":
            attributions = self._gradcam(
                inputs, target_indices, target_layers, **kwargs
            )
        elif method == "guided_gradcam":
            attributions = self._guided_gradcam(
                inputs, target_indices, target_layers, **kwargs
            )
        elif method == "gradcam_pp":
            attributions = self._gradcam_pp(
                inputs, target_indices, target_layers, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Available methods: gradcam, guided_gradcam, gradcam_pp"
            )

        if self.normalize:
            attributions = self.normalize_attribution(attributions)

        if isinstance(attributions, torch.Tensor):
            attributions = attributions.detach().cpu().numpy()

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()

        return {"inputs": inputs, "results": attributions}

    def _gradcam(
        self,
        inputs: torch.Tensor,
        target_indices: torch.Tensor,
        target_layers: str,
        **kwargs,
    ) -> torch.Tensor:
        """Generate Grad-CAM attribution scores for the given inputs.

        Grad-CAM computes importance weights by global-average-pooling the gradients
        of the target class score with respect to the target layer's activations,
        then creates a weighted combination of the activation maps.

        Args:
            inputs: Input tensor of shape (1, n_leads, seq_length).
            target_indices: Target class indices to explain.
            target_layers: Name of the convolutional layer to use.
            **kwargs: Additional parameters.

        Returns:
            Attribution scores of shape (1, n_leads, seq_length).
        """
        n_leads = inputs.shape[1]
        seq_length = inputs.shape[-1]
        target_class = target_indices[0].item()

        activations, gradients = self.model.get_layer_gradients(
            inputs, target_class, target_layers
        )

        activations = torch.tensor(activations, device=self.model.device)
        gradients = torch.tensor(gradients, device=self.model.device)

        if activations.dim() == 3:
            activations = activations.squeeze(0)
        if gradients.dim() == 3:
            gradients = gradients.squeeze(0)

        weights = torch.mean(gradients, dim=-1)
        cam = torch.sum(weights.unsqueeze(-1) * activations, dim=0)

        if self.absolute:
            cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=seq_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

        # Expand to match input channels: (1, seq_length) -> (1, n_leads, seq_length)
        cam = cam.unsqueeze(1).expand(-1, n_leads, -1)

        return cam

    def _guided_backprop(
        self,
        inputs: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Guided Backpropagation gradients.

        Guided Backpropagation modifies the backward pass through ReLU layers
        to only propagate positive gradients, producing sharper saliency maps.

        Reference:
            Springenberg et al., "Striving for Simplicity: The All
            Convolutional Net" (ICLR 2015 Workshop)

        Args:
            inputs: Input tensor of shape (1, n_leads, seq_length).
            target_indices: Target class indices to explain.

        Returns:
            Guided gradients of shape (1, n_leads, seq_length).
        """
        target_class = target_indices[0].item()
        handles = []

        def guided_relu_backward_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0),)

        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_backward_hook(guided_relu_backward_hook)
                handles.append(handle)

        try:
            self.model.model.zero_grad()

            x = inputs.clone().to(self.model.device)
            x.requires_grad_(True)

            outputs = self.model.predict(x, requires_grad=True)

            if outputs.shape[-1] == 1:
                target_output = outputs[0, 0]
            else:
                target_output = outputs[0, target_class]

            target_output.backward()
            guided_grads = x.grad.clone()
        finally:
            for handle in handles:
                handle.remove()

        return guided_grads

    def _guided_gradcam(
        self,
        inputs: torch.Tensor,
        target_indices: torch.Tensor,
        target_layers: str,
        **kwargs,
    ) -> torch.Tensor:
        """Generate Guided Grad-CAM attribution scores for the given inputs.

        Guided Grad-CAM combines Guided Backpropagation with Grad-CAM to produce
        high-resolution, class-discriminative visualizations. The element-wise
        product highlights features that are both important for the prediction
        (Grad-CAM) and have fine-grained detail (Guided Backprop).

        Reference:
            Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
            Networks via Gradient-based Localization" (ICCV 2017)

        Args:
            inputs: Input tensor of shape (1, n_leads, seq_length).
            target_indices: Target class indices to explain.
            target_layers: Name of the convolutional layer to use.
            **kwargs: Additional parameters.

        Returns:
            Attribution scores of shape (1, n_leads, seq_length).
        """
        # _gradcam now returns (1, n_leads, seq_length)
        gradcam_attr = self._gradcam(
            inputs, target_indices, target_layers, **kwargs
        )
        guided_gradients = self._guided_backprop(inputs, target_indices)

        # Element-wise multiplication (both are now (1, n_leads, seq_length))
        guided_gradcam = guided_gradients * gradcam_attr

        return guided_gradcam

    def _gradcam_pp(
        self,
        inputs: torch.Tensor,
        target_indices: torch.Tensor,
        target_layers: str,
        **kwargs,
    ) -> torch.Tensor:
        """Generate Grad-CAM++ attribution scores for the given inputs.

        Grad-CAM++ improves upon Grad-CAM by using pixel-wise weighting of gradients
        based on second and third order derivatives. This provides better localization,
        especially for multiple instances of a class in the input.

        Reference:
            Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based Visual
            Explanations for Deep Convolutional Networks" (WACV 2018)

        Args:
            inputs: Input tensor of shape (1, n_leads, seq_length).
            target_indices: Target class indices to explain.
            target_layers: Name of the convolutional layer to use.
            **kwargs: Additional parameters including 'eps' for numerical stability.

        Returns:
            Attribution scores of shape (1, n_leads, seq_length).
        """
        eps = kwargs.get("eps", 1e-7)
        n_leads = inputs.shape[1]
        seq_length = inputs.shape[-1]
        target_class = target_indices[0].item()

        activations, gradients = self.model.get_layer_gradients(
            inputs, target_class, target_layers
        )

        activations = torch.tensor(activations, device=self.model.device)
        gradients = torch.tensor(gradients, device=self.model.device)

        if activations.dim() == 3:
            activations = activations.squeeze(0)
        if gradients.dim() == 3:
            gradients = gradients.squeeze(0)

        grad_2 = gradients**2
        grad_3 = gradients**3

        sum_activations = torch.sum(F.relu(activations), dim=-1, keepdim=True)

        alpha_numer = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + eps

        alpha_denom = torch.where(
            torch.abs(alpha_denom) < eps, torch.full_like(alpha_denom, eps), alpha_denom
        )
        alpha = alpha_numer / alpha_denom

        positive_gradients = F.relu(gradients)
        weights = torch.sum(alpha * positive_gradients, dim=-1)

        cam = torch.sum(weights.unsqueeze(-1) * activations, dim=0)

        if self.absolute:
            cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=seq_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

        cam = cam.unsqueeze(1).expand(-1, n_leads, -1)

        return cam
