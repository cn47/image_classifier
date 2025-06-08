from config import Config
from dataset_manager import DatasetManager
from trainer import Trainer


def main():
    config = Config()
    dataset_manager = DatasetManager(config)

    model_type = "resnet18"

    trainer = Trainer(
        config,
        model_type=model_type,
        dataloaders=dataset_manager.get_loaders(),
    )

    trainer.fit()


if __name__ == "__main__":
    main()