import os

from dataclasses import asdict, dataclass, field
from pathlib import Path

from torchvision import transforms


_pj_dir = Path(__file__).parents[1]


### Path関連
@dataclass
class PathConfig:
    train_data_dir: Path = _pj_dir / "datasets" / "train"
    output_dir: Path = _pj_dir / "outputs"


### Trainer関連
@dataclass
class EarlyStoppingConfig:
    patience: int = 5
    verbose: bool = True
    delta: float = 0.0


@dataclass
class TrainerConfig:
    n_epochs: int = 10
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    optimizer: dict[str, float] = field(
        default_factory=lambda: {
            "lr": 1e-4,  # 学習率
            "weight_decay": 1e-4,  # 重み減衰
        },
    )
    scheduler: dict[str, int | float] = field(
        default_factory=lambda: {
            "step_size": 5,  # 学習率を減衰させるステップ数
            "gamma": 0.1,  # 学習率を減衰させる倍率
        },
    )


### Dataset関連の設定
@dataclass
class TransformConfig:
    train: transforms.Compose = field(
        default_factory=lambda: transforms.Compose(
            [
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # ImageNetの平均値、標準偏差をハードコード
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        ),
    )
    val: transforms.Compose = field(
        default_factory=lambda: transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                # ImageNetの平均値、標準偏差をハードコード
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        ),
    )


@dataclass
class DatasetLoaderConfig:
    batch_size: int = 32
    shuffle: bool = False  # trainはsamplerを使うからFalse
    num_workers: int = 1  # os.cpu_count()
    pin_memory: bool = True

    def asdict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetManagerConfig:
    train_ratio: float = 0.7
    loader: DatasetLoaderConfig = field(default_factory=lambda: DatasetLoaderConfig())
    transforms: TransformConfig = field(default_factory=lambda: TransformConfig())


### 集約Config
@dataclass
class Config:
    path: PathConfig = field(default_factory=lambda: PathConfig())
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig())
    dataset_manager: DatasetManagerConfig = field(default_factory=lambda: DatasetManagerConfig())

    def asdict(self) -> dict:
        return asdict(self)
