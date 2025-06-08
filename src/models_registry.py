from dataclasses import dataclass, field

import torch.nn as nn

from torchvision import models


class ModelFactoryBase:
    def __init__(self, model_name: str, weights=None):
        self.model_name = model_name
        self.weights = weights

    def __call__(self, n_classes: int) -> nn.Module:
        raise NotImplementedError("This method should be overridden by subclasses.")


class ResNetFactory(ModelFactoryBase):
    def __call__(self, n_classes: int) -> nn.Module:
        model = getattr(models, self.model_name)(weights=self.weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model


class EfficientNetFactory(ModelFactoryBase):
    def __call__(self, n_classes: int) -> nn.Module:
        model = getattr(models, self.model_name)(weights=self.weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        return model


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    factory: type
    weights_enum: object


MODEL_SPECS = {
    "resnet18": ModelSpec(
        model_name="resnet18",
        factory=ResNetFactory,
        weights_enum=models.ResNet18_Weights.IMAGENET1K_V1,
    ),
    "resnet34": ModelSpec(
        model_name="resnet34",
        factory=ResNetFactory,
        weights_enum=models.ResNet34_Weights.IMAGENET1K_V1,
    ),
    "efficientnet_v2_s": ModelSpec(
        model_name="efficientnet_v2_s",
        factory=EfficientNetFactory,
        weights_enum=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    ),
    "efficientnet_v2_m": ModelSpec(
        model_name="efficientnet_v2_m",
        factory=EfficientNetFactory,
        weights_enum=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
    ),
}

AVAILABLE_MODEL_TYPES = list(MODEL_SPECS.keys())


def get_model(model_type: str, n_classes: int, pretrained: bool) -> nn.Module:
    spec = MODEL_SPECS.get(model_type)
    if not spec:
        available_model_types = AVAILABLE_MODEL_TYPES
        msg = f"Model type '{model_type}' is not supported. Available types: {available_model_types}."
        raise ValueError(msg)

    weights = spec.weights_enum if pretrained else None
    factory = spec.factory(spec.model_name, weights=weights)
    return factory(n_classes=n_classes)
