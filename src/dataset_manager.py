import pandas as pd
import torch

from config import Config
from PIL import ImageFile
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets


# PILは極端に大きな画像など高速にロードできない画像はロードしないでスルーするが、
# それを回避するためにLOAD_TRUNCATED_IMAGESをTrueにする。
# これをしないとエラーになることがある
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.transforms = self.config.dataset_manager.transforms

        self._prepare_datasets()
        self._prepare_loaders()

    def _prepare_datasets(self) -> None:
        full_dataset = datasets.ImageFolder(root=self.config.path.train_data_dir, transform=None)
        self._class_names = full_dataset.classes
        self._class_to_idx = full_dataset.class_to_idx

        # train valid split
        train_size = int(self.config.dataset_manager.train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # set transform
        self.train_dataset.dataset.transform = self.transforms.train
        self.val_dataset.dataset.transform = self.transforms.val

    def _prepare_loaders(self) -> None:
        # train_loader with sampling
        targets = torch.tensor(self.train_dataset.dataset.targets)[self.train_dataset.indices]

        class_counts = torch.bincount(targets)
        total_samples = len(self.train_dataset)
        class_weights = total_samples / class_counts.float()

        weights = class_weights[targets]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            **self.config.dataset_manager.loader.asdict(),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            **self.config.dataset_manager.loader.asdict(),
        )

    def get_loaders(self) -> dict[str, DataLoader]:
        return {"train": self.train_loader, "val": self.val_loader}

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def n_classes(self) -> int:
        return len(self._class_names)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self._class_to_idx

    @property
    def class_data_counts(self) -> pd.DataFrame:
        train_counts = torch.bincount(torch.tensor(self.train_dataset.dataset.targets)[self.train_dataset.indices])
        val_counts = torch.bincount(torch.tensor(self.val_dataset.dataset.targets)[self.val_dataset.indices])

        return pd.DataFrame(
            {
                "class": self.class_names,
                "train": train_counts,
                "val": val_counts,
            },
        )

    def save_class_data_counts(self) -> None:
        output_file = self.config.path.output_dir / "train_class_data_counts.tsv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.class_data_counts.to_csv(output_file, index=False, sep="\t")
