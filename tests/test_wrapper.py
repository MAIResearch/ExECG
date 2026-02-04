"""Tests for TorchModelWrapper."""

import numpy as np
import pytest
import torch

from execg.models import TorchModelWrapper


class TestTorchModelWrapperClassification:
    """Tests for TorchModelWrapper with classification models."""

    def test_init(self, classification_wrapper):
        """Test wrapper initialization."""
        assert classification_wrapper.model is not None
        assert classification_wrapper.device is not None
        assert isinstance(classification_wrapper.device, torch.device)

    def test_repr(self, classification_wrapper):
        """Test string representation."""
        repr_str = repr(classification_wrapper)
        assert "TorchModelWrapper" in repr_str
        assert "SimpleCNN" in repr_str

    def test_predict_output_shape(self, classification_wrapper, single_ecg_data):
        """Test predict output shape for classification."""
        output = classification_wrapper.predict(single_ecg_data)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 2)  # (1, n_classes)

    def test_predict_target(self, classification_wrapper, single_ecg_data):
        """Test predict with target specified."""
        output_full = classification_wrapper.predict(single_ecg_data)
        target0 = classification_wrapper.predict(single_ecg_data, target=0)

        assert target0.shape == (1, 1)
        assert target0.item() == pytest.approx(output_full[0, 0].item())

    def test_predict_requires_grad(self, classification_wrapper, single_ecg_data):
        """Test predict with requires_grad=True."""
        output = classification_wrapper.predict(single_ecg_data, requires_grad=True)
        assert output.requires_grad is True

    def test_get_layer_names(self, classification_wrapper):
        """Test get_layer_names returns list of strings."""
        layer_names = classification_wrapper.get_layer_names()

        assert isinstance(layer_names, list)
        assert len(layer_names) > 0
        assert all(isinstance(name, str) for name in layer_names)
        assert "conv1" in layer_names
        assert "conv3" in layer_names

    def test_get_gradients(self, classification_wrapper, single_ecg_data):
        """Test get_gradients returns valid gradients."""
        gradients = classification_wrapper.get_gradients(
            single_ecg_data, target_class=0
        )

        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == tuple(single_ecg_data.shape)  # (1, 12, 500)

    def test_get_gradients_auto_target(self, classification_wrapper, single_ecg_data):
        """Test get_gradients with automatic target class selection."""
        gradients = classification_wrapper.get_gradients(
            single_ecg_data, target_class=None
        )

        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == tuple(single_ecg_data.shape)  # (1, 12, 500)


class TestTorchModelWrapperRegression:
    """Tests for TorchModelWrapper with regression models."""

    def test_predict_output_shape(self, regression_wrapper, single_ecg_data):
        """Test predict output shape for regression."""
        output = regression_wrapper.predict(single_ecg_data)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1)  # (1, 1) for regression

    def test_predict_target(self, regression_wrapper, single_ecg_data):
        """Test predict with target=0 for regression."""
        output_full = regression_wrapper.predict(single_ecg_data)
        target0 = regression_wrapper.predict(single_ecg_data, target=0)

        assert target0.shape == (1, 1)
        assert target0.item() == pytest.approx(output_full[0, 0].item())

    def test_get_gradients(self, regression_wrapper, single_ecg_data):
        """Test get_gradients for regression model."""
        gradients = regression_wrapper.get_gradients(
            single_ecg_data, target_class=0
        )

        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == tuple(single_ecg_data.shape)  # (1, 12, 500)


class TestTorchModelWrapperEdgeCases:
    """Edge case tests for TorchModelWrapper."""

    def test_gradient_not_modified_original(self, classification_wrapper, single_ecg_data):
        """Test that get_gradients doesn't modify original tensor."""
        original_data = single_ecg_data.clone()

        _ = classification_wrapper.get_gradients(single_ecg_data, target_class=0)

        assert torch.allclose(single_ecg_data, original_data)


class TestTorchModelWrapperInputValidation:
    """Tests for input validation."""

    def test_invalid_input_type(self, classification_wrapper):
        """Test that non-tensor input raises TypeError."""
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            classification_wrapper.predict(np.random.randn(1, 12, 500))

    def test_invalid_input_dim_2d(self, classification_wrapper):
        """Test that 2D tensor raises ValueError."""
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            classification_wrapper.predict(torch.randn(12, 500))

    def test_invalid_input_dim_4d(self, classification_wrapper):
        """Test that 4D tensor raises ValueError."""
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            classification_wrapper.predict(torch.randn(1, 1, 12, 500))

    def test_invalid_batch_size(self, classification_wrapper):
        """Test that batch size > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Expected batch size 1"):
            classification_wrapper.predict(torch.randn(2, 12, 500))
