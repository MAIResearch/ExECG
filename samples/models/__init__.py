from typing import Any, Dict, Optional, Union

import torch

from execg.misc import get_model as _get_model

from .registry import MODEL_REGISTRY

__all__ = ["get_model", "MODEL_REGISTRY"]


def get_model(
    name: str,
    model_dir: str,
    download: bool = False,
    device: Union[str, torch.device] = "cpu",
    config: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    """Load a sample classifier model by name.

    Available models: afib_binary, potassium_regression

    Args:
        name: Name of the model.
        model_dir: Directory to store/load model files.
        download: If True, download from Google Drive if not found.
        device: Device to load the model on.
        config: Optional config dict to override the default config.

    Returns:
        Loaded model in eval mode.
    """
    return _get_model(
        name=name,
        model_dir=model_dir,
        registry=MODEL_REGISTRY,
        download=download,
        device=device,
        config=config,
    )
