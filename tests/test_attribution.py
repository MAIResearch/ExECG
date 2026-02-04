"""Tests for Attribution module."""

import numpy as np
import pytest
import torch

from execg.attribution import GradCAM, SaliencyMap


# =============================================================================
# GradCAM Tests
# =============================================================================


class TestGradCAMExplainValidation:
    """Tests for GradCAM.explain input validation."""

    def test_explain_missing_layer_raises_error(
        self, classification_wrapper, ecg_data_2500
    ):
        """Test that missing target_layers raises ValueError."""
        gradcam = GradCAM(classification_wrapper)
        with pytest.raises(ValueError, match="target_layers must be provided"):
            gradcam.explain(ecg_data_2500)

    def test_explain_invalid_layer_raises_error(
        self, classification_wrapper, ecg_data_2500
    ):
        """Test that invalid target_layers raises ValueError."""
        gradcam = GradCAM(classification_wrapper)
        with pytest.raises(ValueError, match="not found in model"):
            gradcam.explain(ecg_data_2500, target_layers="invalid_layer")

    def test_explain_invalid_method_raises_error(
        self, classification_wrapper, ecg_data_2500
    ):
        """Test that invalid method raises ValueError."""
        gradcam = GradCAM(classification_wrapper)
        with pytest.raises(ValueError, match="Unknown method"):
            gradcam.explain(
                ecg_data_2500, target_layers="conv3", method="invalid_method"
            )


class TestGradCAMClassification:
    """Tests for GradCAM with classification model."""

    def test_gradcam_method(self, classification_wrapper, ecg_data_2500):
        """Test GradCAM explain with gradcam method."""
        gradcam = GradCAM(classification_wrapper)
        result = gradcam.explain(
            ecg_data_2500, target=0, target_layers="conv3", method="gradcam"
        )

        assert "inputs" in result
        assert "results" in result
        assert result["inputs"].shape == ecg_data_2500.shape
        assert result["results"].shape[0] == 1  # (1, seq_length)
        assert np.min(result["results"]) >= 0
        assert np.max(result["results"]) <= 1

    def test_guided_gradcam_method(self, classification_wrapper, ecg_data_2500):
        """Test GradCAM explain with guided_gradcam method."""
        gradcam = GradCAM(classification_wrapper)
        result = gradcam.explain(
            ecg_data_2500, target=0, target_layers="conv3", method="guided_gradcam"
        )

        assert "inputs" in result
        assert "results" in result
        assert result["inputs"].shape == ecg_data_2500.shape
        assert result["results"].shape == ecg_data_2500.shape
        assert np.min(result["results"]) >= 0
        assert np.max(result["results"]) <= 1

    def test_gradcam_pp_method(self, classification_wrapper, ecg_data_2500):
        """Test GradCAM explain with gradcam_pp method."""
        gradcam = GradCAM(classification_wrapper)
        result = gradcam.explain(
            ecg_data_2500, target=0, target_layers="conv3", method="gradcam_pp"
        )

        assert "inputs" in result
        assert "results" in result
        assert result["inputs"].shape == ecg_data_2500.shape
        assert result["results"].shape[0] == 1
        assert np.min(result["results"]) >= 0
        assert np.max(result["results"]) <= 1


class TestGradCAMRegression:
    """Tests for GradCAM with regression model."""

    def test_gradcam_method(self, regression_wrapper, ecg_data_2500):
        """Test GradCAM explain with gradcam method for regression."""
        gradcam = GradCAM(regression_wrapper)
        result = gradcam.explain(
            ecg_data_2500, target=0, target_layers="conv3", method="gradcam"
        )

        assert "inputs" in result
        assert "results" in result
        assert result["inputs"].shape == ecg_data_2500.shape


# =============================================================================
# SaliencyMap Tests
# =============================================================================


class TestSaliencyMapExplainValidation:
    """Tests for SaliencyMap.explain input validation."""

    def test_explain_invalid_method_raises_error(
        self, classification_wrapper, ecg_data_2500
    ):
        """Test that invalid method raises ValueError."""
        saliency = SaliencyMap(classification_wrapper)
        with pytest.raises(ValueError, match="Unknown method"):
            saliency.explain(ecg_data_2500, method="invalid_method")


class TestSaliencyMapClassification:
    """Tests for SaliencyMap with classification model."""

    def test_vanilla_saliency_method(self, classification_wrapper, ecg_data_2500):
        """Test SaliencyMap explain with vanilla_saliency method."""
        saliency = SaliencyMap(classification_wrapper)
        result = saliency.explain(ecg_data_2500, target=0, method="vanilla_saliency")

        assert "inputs" in result
        assert "results" in result
        assert result["inputs"].shape == ecg_data_2500.shape
        assert result["results"].shape == ecg_data_2500.shape
        assert np.min(result["results"]) >= 0
        assert np.max(result["results"]) <= 1

    def test_smooth_grad_method(self, classification_wrapper, ecg_data_2500):
        """Test SaliencyMap explain with smooth_grad method."""
        saliency = SaliencyMap(classification_wrapper)
        result = saliency.explain(
            ecg_data_2500,
            target=0,
            method="smooth_grad",
            n_samples=5,  # Small for faster test
            noise_level=0.1,
        )

        assert "inputs" in result
        assert "results" in result
        assert result["inputs"].shape == ecg_data_2500.shape
        assert result["results"].shape == ecg_data_2500.shape
        assert np.min(result["results"]) >= 0
        assert np.max(result["results"]) <= 1

    def test_integrated_gradients_method(self, classification_wrapper, ecg_data_2500):
        """Test SaliencyMap explain with integrated_gradients method."""
        saliency = SaliencyMap(classification_wrapper)
        result = saliency.explain(
            ecg_data_2500,
            target=0,
            method="integrated_gradients",
            n_steps=5,  # Small for faster test
        )

        assert "inputs" in result
        assert "results" in result
        assert result["inputs"].shape == ecg_data_2500.shape
        assert result["results"].shape == ecg_data_2500.shape
        assert np.min(result["results"]) >= 0
        assert np.max(result["results"]) <= 1


class TestSaliencyMapRegression:
    """Tests for SaliencyMap with regression model."""

    def test_vanilla_saliency_method(self, regression_wrapper, ecg_data_2500):
        """Test SaliencyMap explain with vanilla_saliency method for regression."""
        saliency = SaliencyMap(regression_wrapper)
        result = saliency.explain(ecg_data_2500, target=0, method="vanilla_saliency")

        assert "inputs" in result
        assert "results" in result
        assert result["inputs"].shape == ecg_data_2500.shape
        assert result["results"].shape == ecg_data_2500.shape


# =============================================================================
# Utility Function Tests
# =============================================================================
class TestNormalizeAttribution:
    """Tests for normalize_attribution utility function."""

    def test_normalize_tensor(self, classification_wrapper):
        """Test normalization with torch tensor."""
        gradcam = GradCAM(classification_wrapper)
        tensor = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
        normalized = gradcam.normalize_attribution(tensor)
        expected = torch.tensor([[0.0, 1 / 3, 2 / 3, 1.0]])
        assert torch.allclose(normalized, expected, atol=1e-4)

    def test_normalize_numpy(self, classification_wrapper):
        """Test normalization with numpy array."""
        gradcam = GradCAM(classification_wrapper)
        array = np.array([[-1.0, 0.0, 1.0, 2.0]])
        normalized = gradcam.normalize_attribution(array)
        expected = np.array([[0.0, 1 / 3, 2 / 3, 1.0]])
        assert np.allclose(normalized, expected, atol=1e-4)

    def test_normalize_zero_tensor(self, classification_wrapper):
        """Test normalization with zero tensor."""
        gradcam = GradCAM(classification_wrapper)
        zero_tensor = torch.zeros((2, 3))
        normalized = gradcam.normalize_attribution(zero_tensor)
        assert torch.all(normalized == 0)

    def test_normalize_constant_tensor(self, classification_wrapper):
        """Test normalization with constant tensor."""
        gradcam = GradCAM(classification_wrapper)
        const_tensor = torch.ones((2, 3))
        normalized = gradcam.normalize_attribution(const_tensor)
        assert torch.all(normalized == 0)


class TestFormatTarget:
    """Tests for _format_target utility function."""

    def test_format_target_none(self, classification_wrapper, ecg_data_2500):
        """Test _format_target with None (uses model prediction)."""
        gradcam = GradCAM(classification_wrapper)
        output = classification_wrapper.predict(ecg_data_2500)  # (1, N)
        target_indices = gradcam._format_target(None, output)
        assert target_indices.shape[0] == 1

    def test_format_target_int(self, classification_wrapper, ecg_data_2500):
        """Test _format_target with integer target."""
        gradcam = GradCAM(classification_wrapper)
        output = classification_wrapper.predict(ecg_data_2500)  # (1, N)
        target_indices = gradcam._format_target(1, output)
        assert torch.all(target_indices == 1)

    def test_format_target_list(self, classification_wrapper, ecg_data_2500):
        """Test _format_target with list target."""
        gradcam = GradCAM(classification_wrapper)
        output = classification_wrapper.predict(ecg_data_2500)  # (1, N)
        target_indices = gradcam._format_target([0], output)
        assert torch.all(target_indices == torch.tensor([0]))

    def test_format_target_tensor(self, classification_wrapper, ecg_data_2500):
        """Test _format_target with tensor target."""
        gradcam = GradCAM(classification_wrapper)
        output = classification_wrapper.predict(ecg_data_2500)  # (1, N)
        target_tensor = torch.tensor([1])
        target_indices = gradcam._format_target(target_tensor, output)
        assert torch.all(target_indices == target_tensor)
