"""StyleGAN-based Counterfactual Explanation for ECG."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import clear_output

from execg.counterfactual.models import MODEL_REGISTRY
from execg.misc import get_model, match_shape, set_random_seed
from execg.models.wrapper import TorchModelWrapper


class StyleGANCF:
    """StyleGAN-based Counterfactual Explanation generator for ECG.

    This class generates counterfactual explanations by optimizing
    the latent space of a StyleGAN generator to change model predictions.

    The generator internally works at 250Hz with 2500 samples (10 seconds).
    Input data must be exactly 10 seconds. Different sampling rates are
    automatically resampled using nearest neighbor interpolation.

    Args:
        model: TorchModelWrapper instance wrapping the target model.
        generator_name: Name of the generator model (e.g., "stylegan250").
            Available models: {list(MODEL_REGISTRY.keys())}
        model_dir: Directory to store/load model files (code and weights).
        sampling_rate: Sampling rate of the input ECG data in Hz.
        download: If True, download code and weights from Google Drive if not found.

    Example:
        >>> wrapper = TorchModelWrapper(classifier)
        >>> cf_explainer = StyleGANCF(
        ...     wrapper,
        ...     generator_name="stylegan250",
        ...     model_dir="models/stylegan/",
        ...     sampling_rate=500,
        ...     download=True
        ... )
        >>> ecg_data = torch.randn(1, 12, 5000)
        >>> cf_ecg, cf_prob, etc = cf_explainer.explain(
        ...     ecg_data, target_idx=0, target_value=1.0
        ... )
    """

    AVAILABLE_GENERATORS = list(MODEL_REGISTRY.keys())
    GENERATOR_SAMPLING_RATE = 250
    GENERATOR_LENGTH = 2500
    REQUIRED_DURATION = 10.0

    def __init__(
        self,
        model: TorchModelWrapper,
        generator_name: str,
        model_dir: str,
        sampling_rate: float,
        download: bool = False,
        random_seed: Optional[int] = None,
    ):
        if not isinstance(model, TorchModelWrapper):
            raise TypeError(
                f"model must be TorchModelWrapper, got {type(model).__name__}. "
                "Wrap your model first: TorchModelWrapper(your_model)"
            )

        if generator_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown generator_name '{generator_name}'. "
                f"Available generators: {self.AVAILABLE_GENERATORS}"
            )

        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")

        self.model = model
        self.device = model.device
        self.sampling_rate = sampling_rate
        self.random_seed = random_seed

        if random_seed is not None:
            set_random_seed(random_seed)

        generator = get_model(
            name=generator_name,
            model_dir=model_dir,
            registry=MODEL_REGISTRY,
            download=download,
        )
        self.generator = generator.to(self.device)

    def _resample(
        self, x: torch.Tensor, from_rate: float, to_rate: float
    ) -> torch.Tensor:
        """Resample signal using nearest neighbor interpolation.

        Args:
            x: Input tensor of shape (batch, channels, length) or (channels, length)
            from_rate: Source sampling rate in Hz
            to_rate: Target sampling rate in Hz

        Returns:
            Resampled tensor with adjusted length
        """
        if from_rate == to_rate:
            return x

        orig_length = x.shape[-1]
        target_length = int(orig_length * to_rate / from_rate)

        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        resampled = F.interpolate(x, size=target_length, mode="nearest")

        if squeeze_batch:
            resampled = resampled.squeeze(0)

        return resampled

    def _validate_and_prepare_input(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Validate input and prepare for generator (resample to 250Hz).

        Args:
            inputs: Input ECG tensor of shape (1, 12, seq_length) or (12, seq_length)
                Must be exactly 10 seconds of data.

        Returns:
            Tuple of (prepared tensor at 250Hz with shape (1, 12, 2500), original sequence length)

        Raises:
            ValueError: If input is not exactly 10 seconds
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)

        if inputs.dim() != 3 or inputs.shape[1] != 12:
            raise ValueError(
                f"Expected input shape (1, 12, seq_length) or (12, seq_length), "
                f"got {tuple(inputs.shape)}"
            )

        original_length = inputs.shape[-1]
        expected_length = int(self.sampling_rate * self.REQUIRED_DURATION)

        if original_length != expected_length:
            raise ValueError(
                f"Input must be exactly {self.REQUIRED_DURATION:.0f} seconds. "
                f"Expected {expected_length} samples at {self.sampling_rate}Hz, "
                f"got {original_length} samples."
            )

        if self.sampling_rate != self.GENERATOR_SAMPLING_RATE:
            inputs = self._resample(
                inputs, self.sampling_rate, self.GENERATOR_SAMPLING_RATE
            )

        return inputs, original_length

    def _restore_output(
        self, cf_ecg: np.ndarray, original_length: int
    ) -> np.ndarray:
        """Restore counterfactual output to original sampling rate and length.

        Args:
            cf_ecg: Counterfactual ECG at 250Hz, shape (12, 2500)
            original_length: Original sequence length at original sampling rate

        Returns:
            Counterfactual ECG at original sampling rate, shape (12, original_length)
        """
        cf_tensor = torch.tensor(cf_ecg, dtype=torch.float32)

        if self.sampling_rate != self.GENERATOR_SAMPLING_RATE:
            cf_tensor = self._resample(
                cf_tensor, self.GENERATOR_SAMPLING_RATE, self.sampling_rate
            )

        current_length = cf_tensor.shape[-1]
        if current_length > original_length:
            cf_tensor = cf_tensor[..., :original_length]
        elif current_length < original_length:
            pad_length = original_length - current_length
            cf_tensor = F.pad(cf_tensor, (0, pad_length), mode="constant", value=0)

        return cf_tensor.numpy()

    def explain(
        self,
        inputs: torch.Tensor,
        target_idx: Optional[Union[int, List[int], torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Generate counterfactual explanation for ECG input.

        Args:
            inputs: Input ECG tensor of shape (1, 12, seq_length) or (12, seq_length).
                Must be exactly 10 seconds of data at the specified sampling_rate.
                The input will be automatically resampled to 250Hz for the generator,
                and the output will be resampled back to the original sampling rate.
            target_idx: Target output index for counterfactual generation.
            **kwargs: Additional arguments:
                - inversion_steps: W-inversion optimization steps (default: 1000)
                - inversion_lr: W-inversion learning rate (default: 0.0005)
                - cf_steps: CF generation steps (default: 500)
                - cf_lr: CF learning rate (default: 0.0005)
                - target_value: Target prediction value (required)
                - layer_use: StyleGAN layers to optimize (default: all)
                - verbose: Print progress (default: False)
                - show_plot: Show plots during optimization (default: False)
                - random_seed: Random seed for reproducibility (default: None)

        Returns:
            Tuple containing:
                - cf_ecg: Counterfactual ECG as numpy array (12, original_length)
                - cf_prob: Final prediction probability
                - etc: Dict with 'all_cf' and 'all_probs' for visualization

        Raises:
            ValueError: If input is not exactly 10 seconds
        """
        seed = kwargs.get('random_seed', self.random_seed)
        if seed is not None:
            set_random_seed(seed)

        prepared_inputs, original_length = self._validate_and_prepare_input(inputs)

        prepared_inputs_2d = prepared_inputs.squeeze(0)

        recon_ecg, w_inversion, noise = self._get_w_inversion(prepared_inputs_2d, **kwargs)
        cf_ecg_250hz, cf_prob, etc = self._get_cf(
            prepared_inputs_2d.unsqueeze(0), w_inversion, noise, target_idx, **kwargs
        )

        cf_ecg = self._restore_output(cf_ecg_250hz, original_length)

        if "all_cf" in etc:
            etc["all_cf"] = [
                self._restore_output(cf, original_length) for cf in etc["all_cf"]
            ]

        return cf_ecg, cf_prob, etc

    def _get_w_inversion(self, inputs: torch.Tensor, **kwargs):
        """Perform W space inversion for the input ECG tensor.

        Args:
            inputs: ECG tensor of shape (12, 2500).
            **kwargs: Additional arguments including 'inversion_steps',
                'inversion_lr', 'pix_weight', 'inversion_epsilon',
                'verbose', and 'show_plot'.

        Returns:
            Tuple of (reconstructed ECG, optimized W latent vector, noise).
        """
        steps = kwargs.get("inversion_steps", 1000)
        lr = kwargs.get("inversion_lr", 0.0005)
        pix_weight = kwargs.get("pix_weight", 1.0)
        epsilon = kwargs.get("inversion_epsilon", 0.015)
        verbose = kwargs.get("verbose", False)
        show_plot = kwargs.get("show_plot", False)

        ecg_tensor = inputs.clone().to(self.device)
        if len(ecg_tensor.shape) == 2:
            ecg_tensor = ecg_tensor.unsqueeze(0)

        ecg_tensor.requires_grad = False

        with torch.no_grad():
            _, w_styles, noise, _ = self.generator(2500, batch_ecg=ecg_tensor)

        w_styles = w_styles.detach().clone()
        w_styles.requires_grad_()

        optimizer = torch.optim.AdamW([w_styles], lr=lr)

        for idx in range(steps):
            recon_ecg, _, _, _ = self.generator(
                2500, batch_w_plus=w_styles, noise=noise
            )

            loss_pix = torch.nn.functional.mse_loss(
                ecg_tensor.squeeze(), recon_ecg.squeeze()
            ) + torch.nn.functional.l1_loss(ecg_tensor.squeeze(), recon_ecg.squeeze())
            loss = loss_pix * pix_weight

            if show_plot and idx % 100 == 0:
                clear_output(wait=True)
                plt.figure(figsize=(12, 3))
                plt.plot(ecg_tensor.squeeze()[1].cpu().numpy())
                plt.plot(recon_ecg.squeeze()[1].cpu().detach().numpy())
                plt.show()

            if verbose and idx % 100 == 0:
                print(f"[{idx}]Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < epsilon:
                break

        return recon_ecg, w_styles, noise

    def _get_cf(
        self,
        original_ecg: torch.Tensor,
        w_styles: torch.Tensor,
        noise: torch.Tensor,
        target_idx: Optional[Union[int, List[int], torch.Tensor]] = None,
        **kwargs,
    ):
        """Generate counterfactual examples using the inverted W latent vector.

        Args:
            original_ecg: Original reconstructed ECG.
            w_styles: Inverted W latent vector.
            noise: Noise vector.
            target_idx: Target class index for counterfactual.
            **kwargs: Additional arguments including 'cf_steps', 'cf_lr',
                'target_value', 'layer_use', 'verbose', and 'show_plot'.

        Returns:
            Tuple of (counterfactual ECG, probability, metadata dict).

        Raises:
            ValueError: If target_idx or target_value is not specified.
        """
        steps = kwargs.get("cf_steps", 500)
        lr = kwargs.get("cf_lr", 0.0005)
        target_value = kwargs.get("target_value", None)
        verbose = kwargs.get("verbose", False)
        show_plot = kwargs.get("show_plot", False)
        layer_use = kwargs.get("layer_use", None)

        if target_idx is None:
            raise ValueError(
                "target_idx must be specified for counterfactual generation"
            )

        if target_value is None:
            raise ValueError(
                "target_value must be specified for counterfactual generation"
            )

        w_styles = w_styles.to(self.device)
        noise = noise.to(self.device)

        w_delta = torch.zeros_like(w_styles).to(self.device)
        w_delta.requires_grad = True

        if layer_use is not None:
            mask = torch.zeros_like(w_delta)
            mask[0, layer_use, :] = 1
        else:
            mask = torch.ones_like(w_delta)

        original_shape = original_ecg.shape
        original_ecg = original_ecg.detach().clone()

        ecg_for_predict = (
            original_ecg.squeeze(0) if original_ecg.dim() == 3 else original_ecg
        )
        original_prob = self.model.predict(ecg_for_predict, output_idx=target_idx)

        cf_direction = 1 if target_value - original_prob > 0 else -1

        optimizer = torch.optim.Adam([w_delta], lr=lr)

        best_cf_ecg = original_ecg.clone()
        best_cf_prob = original_prob.clone()

        cf_viz_results = [original_ecg.squeeze().detach().clone().cpu().numpy()]
        cf_viz_preds = [original_prob.detach().cpu().numpy()]

        for idx in range(steps):
            w_delta_masked = w_delta * mask

            cf_ecg, _, _, _ = self.generator(
                2500, batch_w_plus=w_styles + w_delta_masked, noise=noise
            )
            cf_ecg = match_shape(cf_ecg.squeeze(), original_shape)

            cf_ecg_for_predict = cf_ecg.squeeze(0) if cf_ecg.dim() == 3 else cf_ecg
            cf_prob = self.model.predict(
                cf_ecg_for_predict, output_idx=target_idx, requires_grad=True
            )

            cf_prob_value = cf_prob.item()
            best_cf_prob_value = best_cf_prob.item()

            if (cf_direction > 0 and cf_prob_value > best_cf_prob_value) or (
                cf_direction < 0 and cf_prob_value < best_cf_prob_value
            ):
                best_cf_ecg = cf_ecg.clone()
                best_cf_prob = cf_prob.clone()

                cf_viz_results.append(
                    best_cf_ecg.squeeze().detach().clone().cpu().numpy()
                )
                cf_viz_preds.append(best_cf_prob.detach().cpu().numpy())

                if (cf_direction > 0 and cf_prob_value > 0.95) or (
                    cf_direction < 0 and cf_prob_value < 0.05
                ):
                    if verbose:
                        print(
                            f"Target reached at iteration {idx}: prob = {cf_prob_value:.4f}"
                        )

                    break

            loss = torch.nn.functional.mse_loss(
                cf_prob, torch.tensor(target_value).float().to(self.device)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if show_plot and idx % 50 == 0:
                clear_output(wait=True)
                plt.figure(figsize=(12, 3))
                plt.plot(original_ecg.squeeze()[1].cpu().numpy(), label="Original ECG")
                plt.plot(
                    cf_ecg.squeeze()[1].cpu().detach().numpy(),
                    label="CF ECG",
                    alpha=0.7,
                )
                plt.legend(loc="upper right")
                plt.title(
                    f"Target: {target_value:.1f}, Current: {cf_prob.item():.4f}, Original: {original_prob.item():.4f}"
                )
                plt.show()

            if verbose and idx % 50 == 0:
                print(
                    f"[{idx}] Loss: {loss.item():.5f}, Prob: {cf_prob.item():.4f}, Target: {target_value:.1f}"
                )

        return (
            best_cf_ecg.squeeze().detach().cpu().numpy(),
            best_cf_prob.item(),
            {
                "all_cf": cf_viz_results,
                "all_probs": cf_viz_preds,
            },
        )
