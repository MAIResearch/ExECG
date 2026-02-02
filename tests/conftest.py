"""Pytest configuration and fixtures for ECG-XAI tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from execg.models import TorchModelWrapper


# =============================================================================
# Path Configuration
# =============================================================================

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
MODELS_DIR = SAMPLES_DIR / "models" / "target_models"
DATA_DIR = SAMPLES_DIR / "data"


# =============================================================================
# Test Models (for fast testing without loading real models)
# =============================================================================


class SimpleCNN(nn.Module):
    """Simple CNN for classification testing.

    Input: (B, 12, 2500) -> Output: (B, num_classes)
    Uses Global Average Pooling to reduce parameters.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleRegressionCNN(nn.Module):
    """Simple CNN for regression testing.

    Input: (B, 12, 2500) -> Output: (B, 1)
    Uses Global Average Pooling to reduce parameters.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# ECG Data Fixtures
# =============================================================================

# Lead order for 12-lead ECG
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _load_ecg_from_json(file_path: Path, seq_length: int = 500) -> torch.Tensor:
    """Load ECG data from JSON file.

    Args:
        file_path: Path to JSON file.
        seq_length: Desired sequence length (truncates if longer).

    Returns:
        torch.Tensor: Shape (12, seq_length).
    """
    with open(file_path) as f:
        data = json.load(f)

    ecg_data = data["data"]
    ecg = np.array([ecg_data[lead][:seq_length] for lead in LEAD_ORDER])
    return torch.tensor(ecg, dtype=torch.float32)


@pytest.fixture(scope="session")
def single_ecg_data():
    """Load a single ECG sample from sample data.

    Returns:
        torch.Tensor: Shape (1, 12, 500) - batch=1, 12 leads, 500 time points.
    """
    sample_file = DATA_DIR / "sample.json"
    if not sample_file.exists():
        pytest.skip(f"Sample data not found: {sample_file}")

    ecg = _load_ecg_from_json(sample_file, seq_length=500)
    return ecg.unsqueeze(0)  # (12, 500) -> (1, 12, 500)


@pytest.fixture(scope="session")
def synthetic_ecg_data(single_ecg_data):
    """Legacy fixture - returns single ECG data for backward compatibility.

    Returns:
        torch.Tensor: Shape (12, 500) - 12 leads, 500 time points.
    """
    return single_ecg_data


@pytest.fixture(scope="session")
def sample_ecg_data():
    """Load sample ECG data from JSON file (full length).

    Returns:
        torch.Tensor: Shape (1, 12, 5000) - batch=1, 12 leads, 500Hz, 10 seconds.
    """
    sample_file = DATA_DIR / "sample.json"
    if not sample_file.exists():
        pytest.skip(f"Sample data not found: {sample_file}")

    ecg = _load_ecg_from_json(sample_file, seq_length=5000)
    return ecg.unsqueeze(0)  # (12, 5000) -> (1, 12, 5000)


# =============================================================================
# Model Fixtures (Simple test models)
# =============================================================================


@pytest.fixture(scope="session")
def classification_model():
    """Create a simple classification model for testing."""
    torch.manual_seed(42)
    model = SimpleCNN(num_classes=2)
    model.eval()
    return model


@pytest.fixture(scope="session")
def regression_model():
    """Create a simple regression model for testing."""
    torch.manual_seed(42)
    model = SimpleRegressionCNN()
    model.eval()
    return model


@pytest.fixture(scope="session")
def classification_wrapper(classification_model):
    """Create a TorchModelWrapper for classification model."""
    return TorchModelWrapper(classification_model)


@pytest.fixture(scope="session")
def regression_wrapper(regression_model):
    """Create a TorchModelWrapper for regression model."""
    return TorchModelWrapper(regression_model)


# =============================================================================
# Legacy Fixtures (for backward compatibility)
# =============================================================================


@pytest.fixture(scope="session")
def get_model_and_wrapper(classification_model, classification_wrapper):
    """Legacy fixture for backward compatibility."""
    return classification_model, classification_wrapper


@pytest.fixture(scope="session")
def afib_classification_wrapper():
    """Create AFib binary classification model (dummy).

    Returns:
        TorchModelWrapper: Wrapped classification model (2 classes).
    """
    torch.manual_seed(42)
    model = SimpleCNN(num_classes=2)
    model.eval()
    return TorchModelWrapper(model)


@pytest.fixture(scope="session")
def potassium_regression_wrapper():
    """Create potassium regression model (dummy).

    Returns:
        TorchModelWrapper: Wrapped regression model.
    """
    torch.manual_seed(42)
    model = SimpleRegressionCNN()
    model.eval()
    return TorchModelWrapper(model)


# =============================================================================
# Counterfactual Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def ecg_data_2500():
    """Load ECG data with 2500 length for counterfactual testing.

    Returns:
        torch.Tensor: Shape (1, 12, 2500) - batch=1, 12 leads, 250Hz, 10 seconds.
    """
    sample_file = DATA_DIR / "sample.json"
    if not sample_file.exists():
        pytest.skip(f"Sample data not found: {sample_file}")

    ecg = _load_ecg_from_json(sample_file, seq_length=2500)
    return ecg.unsqueeze(0)  # (12, 2500) -> (1, 12, 2500)
