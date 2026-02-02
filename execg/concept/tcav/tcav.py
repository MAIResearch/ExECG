"""TCAV (Testing with Concept Activation Vectors) for ECG models."""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
from captum.concept import TCAV as CaptumTCAV
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader
from scipy import stats

from execg.misc import set_random_seed
from execg.models.wrapper import TorchModelWrapper

from .dataset_generator import generate_datasets


class TCAV:
    """Testing with Concept Activation Vectors (TCAV) for ECG models.

    TCAV quantifies the influence of user-defined concepts (e.g., atrial fibrillation,
    T-wave abnormality) on the predictions of a deep neural network model.

    This class wraps Captum's TCAV implementation and provides ECG-specific
    concept dataset generation from PhysioNet2021.

    Args:
        model: PyTorch model to explain.
        model_layers_list: List of layer names to analyze with TCAV.
        model_input_sampling_rate: Model input sampling rate (e.g., 250).
        model_input_duration: Model input duration in seconds (e.g., 10).
        data_name: Dataset name (currently supports "physionet2021").
        data_dir: Directory containing ECG data (.npz files).
        target_concepts: List of concept names to analyze.
        num_random_concepts: Number of random concepts for statistical testing.
        num_samples: Number of samples per concept.
        random_seed: Random seed for reproducibility.
        output_dir: Directory to save TCAV results and intermediate data.

    Example:
        >>> tcav = TCAV(
        ...     model=model,
        ...     model_layers_list=["conv3"],
        ...     model_input_sampling_rate=250,
        ...     model_input_duration=10,
        ...     data_name="physionet2021",
        ...     data_dir="/path/to/physionet_numpy",
        ...     target_concepts=["atrial fibrillation", "sinus rhythm"],
        ...     output_dir="./tcav_results"
        ... )
        >>> results = tcav.explain(inputs, target=1)
    """

    def __init__(
        self,
        model: nn.Module,
        model_layers_list: List[str],
        model_input_sampling_rate: float,
        model_input_duration: float,
        data_name: str,
        data_dir: str,
        target_concepts: List[str],
        num_random_concepts: int = 10,
        num_samples: int = 200,
        random_seed: int = 42,
        output_dir: Optional[str] = None,
    ):
        self.random_seed = random_seed
        set_random_seed(random_seed)

        self.model = TorchModelWrapper(model)
        self.model_layers = model_layers_list
        self.device = next(model.parameters()).device
        self.model_input_sampling_rate = model_input_sampling_rate
        self.model_input_duration = model_input_duration
        self.target_concepts = target_concepts
        self.output_dir = output_dir

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        available_layers = self.model.get_layer_names()
        invalid_layers = set(model_layers_list) - set(available_layers)
        if invalid_layers:
            raise ValueError(
                f"Invalid layers: {invalid_layers}. "
                f"Available layers: {available_layers}"
            )

        self.layer_attr_method = LayerIntegratedGradients(
            self.model.model, None, multiply_by_inputs=False
        )

        target_concept_datasets, random_concept_datasets = generate_datasets(
            data_name=data_name,
            data_dir=data_dir,
            model_input_sampling_rate=model_input_sampling_rate,
            model_input_duration=model_input_duration,
            target_concepts=target_concepts,
            num_random_concepts=num_random_concepts,
            num_samples=num_samples,
            random_seed=random_seed,
            device=self.device
        )

        self.target_concept_dict = {}
        for idx, (name, concept_dataset) in enumerate(target_concept_datasets.items()):
            concept_iter = dataset_to_dataloader(concept_dataset)
            self.target_concept_dict[name] = Concept(idx, name, concept_iter)

        self.random_concept_dict = {}
        for idx, (name, concept_dataset) in enumerate(random_concept_datasets.items()):
            concept_iter = dataset_to_dataloader(concept_dataset)
            self.random_concept_dict[name] = Concept(idx + 100, name, concept_iter)

        self.tcav = CaptumTCAV(
            model=self.model.model,
            layers=self.model_layers,
            layer_attr_method=self.layer_attr_method
        )
        
    
    def explain(
        self,
        inputs: torch.Tensor,
        target: int = 1,
        n_steps: int = 50,
        score_type: str = "sign_count",
    ) -> Dict[str, Any]:
        """Run TCAV interpretation on the provided inputs.

        Args:
            inputs: Input tensor of shape (batch, n_leads, seq_length).
            target: Target class index.
            n_steps: Number of steps for integrated gradients.
            score_type: Score type ("sign_count" or "magnitude").

        Returns:
            Dictionary mapping concept names to layer-wise TCAV scores.
            Each score is a tuple of (mean, (lower_ci, upper_ci)).

        Raises:
            ValueError: If input shape doesn't match model requirements.
            ValueError: If score_type is invalid.
        """
        expected_length = int(self.model_input_sampling_rate * self.model_input_duration)
        if inputs.shape[-1] != expected_length:
            raise ValueError(
                f"Input length {inputs.shape[-1]} doesn't match expected {expected_length}"
            )

        if score_type not in ["sign_count", "magnitude"]:
            raise ValueError(
                f"score_type must be 'sign_count' or 'magnitude', got '{score_type}'"
            )

        results = defaultdict(lambda: defaultdict(list))

        for target_concept_name, target_concept in self.target_concept_dict.items():
            for random_concept_name, random_concept in self.random_concept_dict.items():
                experimental_sets = [[target_concept, random_concept]]

                tcav_output = self.tcav.interpret(
                    inputs=inputs.to(self.device),
                    experimental_sets=experimental_sets,
                    target=target,
                    n_steps=n_steps
                )

                for layer_name, tcav_score in list(tcav_output.values())[0].items():
                    tcav_scores = tcav_score[score_type]
                    results[target_concept_name][layer_name].append(tcav_scores)

        for target_concept_name, target_results in results.items():
            for layer_name, layer_results in target_results.items():
                layer_results = np.array(layer_results)[:, 0]
                results[target_concept_name][layer_name] = self._mean_confidence_interval(
                    layer_results
                )

        results = dict(results)

        if self.output_dir is not None:
            self._save_results(results, target, score_type)

        return results

    def _save_results(
        self, results: Dict[str, Any], target: int, score_type: str
    ) -> None:
        """Save TCAV results to output directory.

        Args:
            results: TCAV results dictionary.
            target: Target class index.
            score_type: Score type used.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        concepts_str = "_".join([c.replace(" ", "-") for c in self.target_concepts[:3]])
        layers_str = "_".join([l.replace(".", "-") for l in self.model_layers[:2]])

        filename = f"tcav_results_target{target}_{score_type}_{concepts_str}_{layers_str}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        serializable_results = {}
        for concept_name, layer_results in results.items():
            serializable_results[concept_name] = {}
            for layer_name, (mean, (ci_lower, ci_upper)) in layer_results.items():
                serializable_results[concept_name][layer_name] = {
                    "mean": float(mean),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                }

        output_data = {
            "metadata": {
                "target_class": target,
                "score_type": score_type,
                "target_concepts": self.target_concepts,
                "model_layers": self.model_layers,
                "random_seed": self.random_seed,
                "timestamp": timestamp,
            },
            "results": serializable_results,
        }

        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"TCAV results saved to: {filepath}")
    
    @staticmethod
    def _mean_confidence_interval(data: np.ndarray, confidence: float = 0.95):
        """Calculate mean and confidence interval.

        Args:
            data: Array of values.
            confidence: Confidence level (default: 0.95).

        Returns:
            Tuple of (mean, (lower_bound, upper_bound)).
        """
        a = np.array(data)
        n = len(a)
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return m, (np.clip(m - h, 0, 1), np.clip(m + h, 0, 1))

    @staticmethod
    def plot_tcav_scores(result_dict: Dict[str, Any], cmap: str = "coolwarm"):
        """Plot TCAV scores as a heatmap.

        Args:
            result_dict: Dictionary from explain() method.
            cmap: Matplotlib colormap name.

        Returns:
            Tuple of (fig, ax) matplotlib objects.

        Note:
            This is a convenience method. For more visualization options,
            use execg.visualization.tcav module directly.
        """
        from execg.visualization.tcav import plot_tcav_scores as _plot_tcav_scores

        return _plot_tcav_scores(result_dict, cmap=cmap)

