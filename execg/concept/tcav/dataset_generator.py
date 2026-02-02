"""Dataset generator for TCAV concept datasets."""

import os
import random
from random import shuffle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


DATA_CONFIG = {
    "physionet2021": {
        "sampling_rate": 500,
        "duration": 10,
        "label_filename": "physionet2021.csv",
        "label_list": [
            "atrial fibrillation",
            "atrial flutter",
            "bundle branch block",
            "bradycardia",
            "complete left bundle branch block, left bundle...",
            "complete right bundle branch block, right bund...",
            "1st degree av block",
            "incomplete right bundle branch block",
            "left axis deviation",
            "left anterior fascicular block",
            "prolonged pr interval",
            "low qrs voltages",
            "prolonged qt interval",
            "nonspecific intraventricular conduction disorder",
            "sinus rhythm",
            "premature atrial contraction, supraventricular...",
            "pacing rhythm",
            "poor R wave Progression",
            "premature ventricular contractions, ventricula...",
            "qwave abnormal",
            "right axis deviation",
            "sinus arrhythmia",
            "sinus bradycardia",
            "sinus tachycardia",
            "t wave abnormal",
            "t wave inversion",
        ],
    }
}


def generate_concept_dict(
    data_name: str,
    target_concepts: List[str],
    num_random_concepts: int = 10,
    num_samples: int = 200,
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Generate dictionaries mapping concept names to sample dataframes.

    Args:
        data_name: Dataset name (e.g., 'physionet2021').
        target_concepts: List of target concept names.
        num_random_concepts: Number of random concepts to generate.
        num_samples: Number of samples per concept.
        random_seed: Random seed for reproducibility.
        verbose: Print progress messages.

    Returns:
        Tuple of (target_concept_dict, random_concept_dict).

    Raises:
        ValueError: If data_name or target_concepts are invalid.
    """
    if data_name not in DATA_CONFIG:
        raise ValueError(f"data_name must be one of {list(DATA_CONFIG.keys())}")

    config = DATA_CONFIG[data_name]
    label_filename = config["label_filename"]
    label_list = config["label_list"]

    invalid_concepts = set(target_concepts) - set(label_list)
    if invalid_concepts:
        raise ValueError(f"Invalid concepts: {invalid_concepts}")

    script_dir = os.path.dirname(__file__)
    label_path = os.path.join(script_dir, "../labels/", label_filename)
    label_df = pd.read_csv(label_path)

    label_dist = label_df[[str(i) for i in range(len(label_list))]].sum(axis=0).tolist()
    sorted_label_indices = sorted(range(len(label_dist)), key=lambda k: label_dist[k])
    label_dist = np.array([label_dist[i] for i in sorted_label_indices])

    selected_idx_list = [label_list.index(concept) for concept in target_concepts]

    target_concept_dict = {}
    random_concept_dict = {}

    total_filename_set = set(label_df.filename.tolist())
    control_filename_set = set(label_df.filename.tolist())

    for idx in np.argsort(label_dist[selected_idx_list]):
        select_idx = selected_idx_list[idx]
        name = label_list[select_idx]

        if verbose:
            print(f"Processing concept: {name}")

        target_df = label_df[label_df[str(select_idx)] == 1].copy()
        target_df["count"] = (
            target_df[[str(i) for i in range(len(label_list))]].sum(axis=1).tolist()
        )

        exist_filename_df = pd.DataFrame(list(total_filename_set), columns=["filename"])
        target_df = pd.merge(target_df, exist_filename_df, on="filename", how="inner")

        random.seed(random_seed)
        random_number = list(range(len(target_df)))
        shuffle(random_number)
        target_df["random_seed"] = random_number

        source_list = target_df.source.value_counts(ascending=True).index.tolist()
        source_sample_list = []

        remain_n = num_samples
        each_n = int(num_samples / len(source_list))

        for i, source in enumerate(source_list):
            source_sample_df = target_df[target_df.source == source]
            if i == len(source_list) - 1:
                target_sample_df = source_sample_df.sort_values(
                    ["count", "random_seed"]
                ).head(remain_n)
            else:
                target_sample_df = source_sample_df.sort_values(
                    ["count", "random_seed"]
                ).head(each_n)
                remain_n -= len(target_sample_df)

            if verbose:
                print(f"  {source}: {len(target_sample_df)} samples")
            source_sample_list.append(target_sample_df)

        target_sample_df = pd.concat(source_sample_list)
        filename_list = target_sample_df.filename.tolist()
        target_concept_dict[name] = target_sample_df

        total_filename_set = total_filename_set - set(filename_list)
        control_filename_set = control_filename_set - set(target_df.filename.tolist())

        if verbose:
            if len(filename_list) < num_samples:
                print(f"  [Warning] Insufficient samples: {len(filename_list)}")
            else:
                print(f"  [OK] {len(filename_list)} samples prepared")

    for random_idx in range(num_random_concepts):
        random_sample_df = pd.DataFrame(
            list(control_filename_set), columns=["filename"]
        )
        random_sample_df = pd.merge(
            random_sample_df, label_df, on="filename", how="inner"
        )

        each_n = int(num_samples / len(random_sample_df.source.unique()))

        random_sample_df = random_sample_df.groupby("source").sample(
            each_n, random_state=random_seed + random_idx
        )

        random_concept_dict[f"random_concept_{random_idx}"] = random_sample_df

        if verbose:
            print(f"Random concept {random_idx}: {len(random_sample_df)} samples")

    return target_concept_dict, random_concept_dict


class ConceptDataset(data.Dataset):
    """Dataset class for TCAV concepts.

    Args:
        data_dir: Directory containing ECG .npz files.
        data_sampling_rate: Original sampling rate of the data.
        file_df: DataFrame with 'filename' column.
        transform: Optional transform to apply.
        device: Device to load tensors to.
    """

    def __init__(
        self,
        data_dir: str,
        data_sampling_rate: int,
        file_df: pd.DataFrame,
        transform=None,
        device: str = "cpu",
    ):
        self.data_dir = data_dir
        self.data_sampling_rate = data_sampling_rate
        self.file_df = file_df.reset_index(drop=True)
        self.transform = transform
        self.file_list = file_df.filename.tolist()
        self.device = device

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_dir, f"{filename}.npz")

        signal = np.load(file_path)["arr_0"]
        if self.transform:
            signal = self.transform(signal, self.data_sampling_rate)

        return signal.to(self.device)


def generate_datasets(
    data_name: str,
    data_dir: str,
    target_concepts: List[str],
    model_input_sampling_rate: int,
    model_input_duration: int,
    num_random_concepts: int = 10,
    num_samples: int = 200,
    random_seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[Dict[str, ConceptDataset], Dict[str, ConceptDataset]]:
    """Generate TCAV concept datasets.

    Args:
        data_name: Dataset name (e.g., 'physionet2021').
        data_dir: Directory containing ECG .npz files.
        target_concepts: List of target concept names.
        model_input_sampling_rate: Target sampling rate for model input.
        model_input_duration: Target duration in seconds for model input.
        num_random_concepts: Number of random concepts to generate.
        num_samples: Number of samples per concept.
        random_seed: Random seed for reproducibility.
        device: Device to load tensors to.
        verbose: Print progress messages.

    Returns:
        Tuple of (target_concept_datasets, random_concept_datasets).
    """
    dataset_sampling_rate = DATA_CONFIG[data_name]["sampling_rate"]

    target_concept_dict, random_concept_dict = generate_concept_dict(
        data_name=data_name,
        target_concepts=target_concepts,
        num_random_concepts=num_random_concepts,
        num_samples=num_samples,
        random_seed=random_seed,
        verbose=verbose,
    )

    transform = SignalTransform(model_input_sampling_rate, model_input_duration)

    target_concept_datasets = {}
    for concept_name, concept_df in target_concept_dict.items():
        target_concept_datasets[concept_name] = ConceptDataset(
            data_dir=data_dir,
            data_sampling_rate=dataset_sampling_rate,
            file_df=concept_df,
            transform=transform,
            device=device,
        )
        if verbose:
            print(f"Created dataset: '{concept_name}' ({len(target_concept_datasets[concept_name])} samples)")

    random_concept_datasets = {}
    for concept_name, concept_df in random_concept_dict.items():
        random_concept_datasets[concept_name] = ConceptDataset(
            data_dir=data_dir,
            data_sampling_rate=dataset_sampling_rate,
            file_df=concept_df,
            transform=transform,
            device=device,
        )
        if verbose:
            print(f"Created dataset: '{concept_name}' ({len(random_concept_datasets[concept_name])} samples)")

    return target_concept_datasets, random_concept_datasets


class SignalTransform:
    """Transform for resampling and padding ECG signals.

    Args:
        model_input_sampling_rate: Target sampling rate (Hz).
        model_input_duration: Target duration (seconds).
    """

    def __init__(self, model_input_sampling_rate: int, model_input_duration: int):
        self.model_input_sampling_rate = model_input_sampling_rate
        self.model_input_duration = model_input_duration
        self.target_length = int(model_input_sampling_rate * model_input_duration)

    def __call__(self, signal: np.ndarray, original_sampling_rate: int) -> torch.Tensor:
        """Apply transformation to the signal.

        Args:
            signal: Input ECG signal of shape (n_leads, n_samples).
            original_sampling_rate: Original sampling rate of the signal.

        Returns:
            Transformed signal tensor of shape (n_leads, target_length).
        """
        n_leads, n_samples = signal.shape

        if original_sampling_rate != self.model_input_sampling_rate:
            ratio = self.model_input_sampling_rate / original_sampling_rate
            new_length = int(n_samples * ratio)

            resampled_signal = np.zeros((n_leads, new_length))
            for i in range(n_leads):
                original_times = np.linspace(0, 1, n_samples)
                new_times = np.linspace(0, 1, new_length)
                resampled_signal[i] = np.interp(new_times, original_times, signal[i])

            signal = resampled_signal

        signal = torch.FloatTensor(signal)
        _, current_length = signal.shape

        if current_length < self.target_length:
            padding = torch.zeros(n_leads, self.target_length - current_length)
            signal = torch.cat([signal, padding], dim=1)
        elif current_length > self.target_length:
            signal = signal[:, :self.target_length]

        return signal
