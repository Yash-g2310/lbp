"""Model architectures for Task 1 and Task 6."""

from .classifier import DeepLenseResNet, PhysicsInformedSwin
from .pi_haf import PIHAFBackbone, PIHAFConfig
from .registry import MODEL_REGISTRY, build_model, list_available_models, register_model
from .super_res import DiffusionSuperResolution, UNetSuperResolution

__all__ = [
    "DeepLenseResNet",
    "DiffusionSuperResolution",
    "MODEL_REGISTRY",
    "PIHAFBackbone",
    "PIHAFConfig",
    "PhysicsInformedSwin",
    "UNetSuperResolution",
    "build_model",
    "list_available_models",
    "register_model",
]
