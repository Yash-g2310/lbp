"""Classification models for DeepLense Task 1."""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class DeepLenseResNet(nn.Module):
    """ResNet-18 classifier adapted for 1-channel DeepLense inputs."""

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_rate: float = 0.0,
        input_size: tuple[int, int] = (150, 150),
    ):
        super().__init__()
        self.input_size = input_size

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        original_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # Preserve pretrained edge detectors by collapsing RGB filters to 1 channel.
                new_conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.backbone.conv1 = new_conv1

        in_features = self.backbone.fc.in_features
        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PhysicsInformedSwin(nn.Module):
    """Placeholder for future physics-informed Swin Transformer implementation."""

    def __init__(self, num_classes: int = 3, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "PhysicsInformedSwin is a stub. Implement shifted-window transformer blocks "
            "with physics-informed inductive biases in a later phase."
        )


__all__ = [
    "DeepLenseResNet",
    "PhysicsInformedSwin",
]
