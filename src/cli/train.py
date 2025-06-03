from config import Config
from dataset_manager import DatasetManager
from torch import nn
from torchvision import models
from trainer import Trainer


def main():
    config = Config()
    dataset_manager = DatasetManager(config)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=dataset_manager.n_classes)

    trainer = Trainer(
        config,
        model=model,
        dataloaders=dataset_manager.get_loaders(),
    )

    trainer.fit()
    dataset_manager.save_idx_to_class()


if __name__ == "__main__":
    main()