import importlib.util
import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import requests
import torch
from tqdm import tqdm


def set_random_seed(seed: int):
    """Set random seed for reproducibility across all random number generators.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(
    name: str,
    model_dir: str,
    registry: Dict[str, Dict[str, Any]] = None,
    download: bool = False,
    device: Union[str, torch.device] = "cpu",
    config: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    """Load a model by name from a registry with automatic download support.

    Downloads model code (.py) and weights (.pt) to the specified directory.
    Both files are stored in the same folder for easy management.

    Args:
        name: Name of the model in the registry.
        model_dir: Directory to store/load model files (code and weights).
        registry: Dictionary containing model configurations.
            Each entry should have: model_class_name, gdrive_code_id, gdrive_weight_id.
        download: If True, download code and weights from Google Drive if not found.
        device: Device to load the model on.
        config: Optional config dict to override the default config.

    Returns:
        Loaded model in eval mode.

    Raises:
        ValueError: If name is not in registry.
        FileNotFoundError: If files don't exist and download is False.

    Example:
        >>> # Using registry (downloads code and weights to model_dir)
        >>> model = get_model("afib_binary", "./models/afib/", MODEL_REGISTRY, download=True)
        >>> # This creates:
        >>> #   ./models/afib/ResNetDropoutAfib.py  (model code)
        >>> #   ./models/afib/ResNetDropoutAfib.pt  (weights)
    """
    if registry is None or name not in registry:
        if registry is not None:
            available = list(registry.keys())
            raise ValueError(f"Unknown model '{name}'. Available: {available}")
        else:
            raise ValueError("Registry must be provided")

    model_info = registry[name]
    model_class_name = model_info.get("model_class_name")
    gdrive_code_id = model_info.get("gdrive_code_id")
    gdrive_weight_id = model_info.get("gdrive_weight_id")

    os.makedirs(model_dir, exist_ok=True)

    code_path = os.path.join(model_dir, f"{model_class_name}.py")
    weight_path = os.path.join(model_dir, f"{model_class_name}.pt")

    if not os.path.exists(code_path):
        if download and gdrive_code_id:
            print(f"Downloading {name} model code to {code_path}...")
            download_from_gdrive(gdrive_code_id, code_path)
        else:
            raise FileNotFoundError(
                f"Code file not found at '{code_path}'. "
                "Set download=True to download automatically."
            )

    spec = importlib.util.spec_from_file_location("model_module", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, model_class_name):
        model_cls = getattr(module, model_class_name)
    else:
        raise ValueError(f"Could not find class '{model_class_name}' in '{code_path}'")

    model = model_cls()

    if not os.path.exists(weight_path):
        if download:
            if gdrive_weight_id is None:
                raise ValueError(f"No weight download URL configured for '{name}'")
            print(f"Downloading {name} weights to {weight_path}...")
            download_from_gdrive(gdrive_weight_id, weight_path)
        else:
            raise FileNotFoundError(
                f"Weight file not found at '{weight_path}'. "
                "Set download=True to download automatically."
            )

    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    return model


def download_from_gdrive(file_id: str, destination: str) -> None:
    """Download a file from Google Drive with progress bar.

    Uses gdown library if available (recommended for large files),
    otherwise falls back to requests-based download.

    Args:
        file_id: Google Drive file ID.
        destination: Local path to save the file.

    Example:
        >>> download_from_gdrive("1SeyvAWn5yJA0byXf6l4VLgKh4QZtJM1q", "./weights/model.pt")
    """
    parent_dir = os.path.dirname(destination)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    try:
        import gdown

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        print(f"Downloaded to {destination}")
        return
    except ImportError:
        print("gdown not installed, using requests (pip install gdown recommended)")

    URL = "https://drive.usercontent.google.com/download"

    session = requests.Session()
    params = {
        "id": file_id,
        "export": "download",
        "confirm": "t",
    }

    response = session.get(URL, params=params, stream=True)

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        raise RuntimeError(
            f"Failed to download file from Google Drive. "
            f"Please install gdown (pip install gdown) or download manually from: "
            f"https://drive.google.com/file/d/{file_id}/view "
            f"and save to: {destination}"
        )

    total_size = int(response.headers.get("content-length", 0))

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading",
        ) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"Downloaded to {destination}")


class ConfigClass:
    """Configuration class that allows dot notation access to nested attributes.

    Example:
        >>> config = ConfigClass({"model": {"hidden_size": 256}, "lr": 0.001})
        >>> config.model.hidden_size
        256
        >>> config.lr
        0.001
    """

    def __init__(self, config_dict: dict):
        """Convert a dictionary to a nested attribute object.

        Args:
            config_dict: Dictionary to convert into nested attributes.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigClass(value))
            else:
                setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for checking key existence.

        Args:
            key: Key to check for existence.

        Returns:
            True if key exists in the config, False otherwise.
        """
        return key in self.__dict__

    def to_dict(self) -> dict:
        """Convert ConfigClass object back to a dictionary.

        Returns:
            Dictionary representation of the config.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigClass):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def match_shape(tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Reshape tensor to match target shape by finding broadcast positions.

    Finds the positions of non-1 dimensions in target_shape and reshapes the tensor
    to align with those positions for proper broadcasting.

    Args:
        tensor: Input tensor to reshape.
        target_shape: Target shape to match (may contain 1s for broadcast dimensions).

    Returns:
        Reshaped tensor that can be broadcast to target_shape.

    Raises:
        ValueError: If tensor dimensions don't match non-1 dimensions in target_shape.
    """
    tensor_shape = list(tensor.shape)
    target_shape = list(target_shape)

    non_1_dims = [i for i, s in enumerate(target_shape) if s != 1]

    if len(tensor_shape) != len(non_1_dims):
        raise ValueError(
            f"Cannot match shapes: tensor shape {tensor_shape}, target shape {target_shape}"
        )

    new_shape = [1] * len(target_shape)
    for i, dim in zip(non_1_dims, tensor_shape):
        new_shape[i] = dim

    return tensor.reshape(new_shape)
