"""Model registry for configurable Task 1 model instantiation."""

from typing import Any, Callable, Dict, List

import torch.nn as nn

from .classifier import DeepLenseResNet, PhysicsInformedSwin
from .pi_haf import PIHAFBackbone

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "deeplense_resnet": DeepLenseResNet,
    "physics_informed_swin": PhysicsInformedSwin,
    "pi_haf": PIHAFBackbone,
}


def list_available_models() -> List[str]:
    """Return sorted list of registered model names."""
    return sorted(MODEL_REGISTRY.keys())


def build_model(model_config: Dict[str, Any]) -> nn.Module:
    """Build model from validated model config dictionary."""
    if "architecture" not in model_config:
        raise KeyError("model_config must include 'architecture'")

    architecture = model_config["architecture"]
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture '{architecture}'. "
            f"Available models: {list_available_models()}"
        )

    model_cls = MODEL_REGISTRY[architecture]
    kwargs = {k: v for k, v in model_config.items() if k != "architecture"}
    return model_cls(**kwargs)


def register_model(name: str, model_cls: Callable[..., nn.Module]) -> None:
    """Register a new model architecture for future ablation experiments."""
    if not name:
        raise ValueError("Model name cannot be empty")
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model name '{name}' is already registered")
    MODEL_REGISTRY[name] = model_cls


__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "list_available_models",
    "register_model",
]
