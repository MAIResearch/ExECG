import torch
import numpy as np

from execg.attribution import GradCAM

# Test compute gradients function
def test_compute_gradients(get_model_and_wrapper, synthetic_ecg_data):
    """Test the _compute_gradients function."""
    _, model_wrapper = get_model_and_wrapper
    gradcam = GradCAM(model_wrapper)
    
    # Get target indices
    output = model_wrapper.predict(synthetic_ecg_data)
    if isinstance(output, np.ndarray):
        output = torch.tensor(output, device=synthetic_ecg_data.device)
    target_indices = gradcam._format_target(None, output)
    
    # Compute gradients
    gradients = gradcam._compute_gradients(synthetic_ecg_data, target_indices)
    
    # Check shape
    assert gradients.shape == synthetic_ecg_data.shape 