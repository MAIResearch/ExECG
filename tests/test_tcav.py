"""Tests for TCAV module."""

import numpy as np
import pytest
import torch

from execg.concept.tcav.dataset_generator import (
    DATA_CONFIG,
    SignalTransform,
    generate_concept_dict,
)


# =============================================================================
# SignalTransform Tests
# =============================================================================


class TestSignalTransform:
    """Tests for SignalTransform class."""

    def test_init(self):
        """Test SignalTransform initialization."""
        transform = SignalTransform(
            model_input_sampling_rate=250,
            model_input_duration=10
        )
        assert transform.model_input_sampling_rate == 250
        assert transform.model_input_duration == 10
        assert transform.target_length == 2500

    def test_no_resampling_needed(self):
        """Test transform when no resampling is needed."""
        transform = SignalTransform(
            model_input_sampling_rate=500,
            model_input_duration=10
        )
        signal = np.random.randn(12, 5000)
        result = transform(signal, original_sampling_rate=500)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (12, 5000)

    def test_downsample(self):
        """Test downsampling from 500Hz to 250Hz."""
        transform = SignalTransform(
            model_input_sampling_rate=250,
            model_input_duration=10
        )
        signal = np.random.randn(12, 5000)  # 500Hz, 10s
        result = transform(signal, original_sampling_rate=500)

        assert result.shape == (12, 2500)

    def test_upsample(self):
        """Test upsampling from 250Hz to 500Hz."""
        transform = SignalTransform(
            model_input_sampling_rate=500,
            model_input_duration=10
        )
        signal = np.random.randn(12, 2500)  # 250Hz, 10s
        result = transform(signal, original_sampling_rate=250)

        assert result.shape == (12, 5000)

    def test_truncate_long_signal(self):
        """Test truncation of signals longer than target."""
        transform = SignalTransform(
            model_input_sampling_rate=250,
            model_input_duration=10
        )
        signal = np.random.randn(12, 5000)  # Longer than 2500
        result = transform(signal, original_sampling_rate=250)

        assert result.shape == (12, 2500)

    def test_pad_short_signal(self):
        """Test padding of signals shorter than target."""
        transform = SignalTransform(
            model_input_sampling_rate=250,
            model_input_duration=10
        )
        signal = np.random.randn(12, 1000)  # Shorter than 2500
        result = transform(signal, original_sampling_rate=250)

        assert result.shape == (12, 2500)
        # Check that padding is zeros
        assert torch.all(result[:, 1000:] == 0)


# =============================================================================
# DATA_CONFIG Tests
# =============================================================================


class TestDataConfig:
    """Tests for DATA_CONFIG."""

    def test_physionet2021_config_exists(self):
        """Test that physionet2021 config exists."""
        assert "physionet2021" in DATA_CONFIG

    def test_physionet2021_has_required_keys(self):
        """Test that physionet2021 config has required keys."""
        config = DATA_CONFIG["physionet2021"]
        assert "sampling_rate" in config
        assert "duration" in config
        assert "label_filename" in config
        assert "label_list" in config

    def test_physionet2021_sampling_rate(self):
        """Test physionet2021 sampling rate is 500Hz."""
        assert DATA_CONFIG["physionet2021"]["sampling_rate"] == 500

    def test_physionet2021_label_list_not_empty(self):
        """Test that label list is not empty."""
        label_list = DATA_CONFIG["physionet2021"]["label_list"]
        assert len(label_list) > 0
        assert "atrial fibrillation" in label_list
        assert "sinus rhythm" in label_list


# =============================================================================
# generate_concept_dict Tests
# =============================================================================


class TestGenerateConceptDict:
    """Tests for generate_concept_dict function."""

    def test_invalid_data_name_raises_error(self):
        """Test that invalid data_name raises ValueError."""
        with pytest.raises(ValueError, match="data_name must be one of"):
            generate_concept_dict(
                data_name="invalid_dataset",
                target_concepts=["atrial fibrillation"],
                verbose=False
            )

    def test_invalid_concept_raises_error(self):
        """Test that invalid concept raises ValueError."""
        with pytest.raises(ValueError, match="Invalid concepts"):
            generate_concept_dict(
                data_name="physionet2021",
                target_concepts=["invalid_concept"],
                verbose=False
            )


# =============================================================================
# TCAV Integration Tests (require data)
# =============================================================================


@pytest.mark.skip(reason="Requires PhysioNet data")
class TestTCAVIntegration:
    """Integration tests for TCAV (require actual data)."""

    def test_tcav_explain(self):
        """Test TCAV explain method."""
        pass
