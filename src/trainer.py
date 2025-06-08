from pathlib import Path

import numpy as np
import torch

from config import Config
from IPython.display import clear_output
from models_registry import get_model
from plot import plot_classification_report, plot_learning_curve
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, config: Config, output_model_file: Path) -> None:
        self.config = config
        self.counter = 0  # 連続で改善しなかった回数
        self.best_score = None
        self.early_stop = False  # 早期終了フラグ
        self.best_model_weights = None  # ベストモデルの重み保存用
        self.output_model_file = output_model_file

    def __call__(self, val_score: float, model: nn.Module) -> None:
        patience = self.config.trainer.early_stopping.patience
        verbose = self.config.trainer.early_stopping.verbose
        delta = self.config.trainer.early_stopping.delta

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.best_model_weights = model.state_dict()
            self.save_checkpoint(model)
        elif score < self.best_score + delta:
            self.counter += 1
            if verbose:
                print(f"(EarlyStopping count: {self.counter} out of {patience})")
            if self.counter >= patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_weights = model.state_dict()
            self.save_checkpoint(model)
            self.counter = 0  # 連続改善回数リセット

    def save_checkpoint(self, model: nn.Module) -> None:
        verbose = self.config.trainer.early_stopping.verbose  # 進捗表示の有無

        self.output_model_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.output_model_file)
        if verbose:
            print(f"Validation score improved, model saved to {self.output_model_file}")


class Trainer:
    def __init__(
        self,
        config: Config,
        model_type: str,
        dataloaders: dict[str, DataLoader],
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.dataloaders = dataloaders
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(f"{self.config.path.model_output_dir}_{model_type}")

        self.model_type = model_type
        self.model = get_model(model_type=self.model_type, n_classes=len(self._get_class_names()), pretrained=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), **self.config.trainer.optimizer)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **self.config.trainer.scheduler)

        # AMP用scaler
        self.scaler = GradScaler(device=self.device)
        self.early_stopping = EarlyStopping(config, output_model_file=self.output_dir / "classifier.pth")

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_auc": [],
            "classification_report": [],
        }

        self.model.to(self.device)
        torch.backends.cudnn.benchmark = True

    def _get_class_names(self) -> list[str]:
        """クラス名を取得する"""
        return self.dataloaders["train"].dataset.dataset.classes

    @property
    def class_names(self) -> list[str]:
        return self._get_class_names()

    def generate_train_report(self):
        """トレーニングの進捗と結果を表示・保存する

        1. 学習曲線を表示
        2. 最終エポックのAUCを表示
        3. 分類レポートを表示・保存
        4. ROC-AUCをテキストで保存
        """
        clear_output(wait=True)
        plot_learning_curve(self.history, output_file=self.output_dir / "learning_curve.png")

        report = self.history["classification_report"][-1]
        auc_score = self.history["val_auc"][-1]
        print("\n-- Validation AUC --")
        print(f"val AUC:  {auc_score:.4f}")

        print("\n-- Classification Report --")
        plot_classification_report(report, output_file=self.output_dir / "classification_report.png")

        with (self.output_dir / "roc_auc.txt").open("w") as fp:
            fp.write(f"val AUC: {auc_score:.4f}")

    def save_model_type(self) -> None:
        """モデルのタイプを保存する"""
        output_file = self.output_dir / "model_type.json"
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"model_type": self.model_type}
        with output_file.open("w") as fp:
            json.dump(data, fp, indent=4, ensure_ascii=False)

    @property
    def idx_to_class(self) -> dict[str, int]:
        return {v: k for k, v in self.dataloaders["train"].dataset.dataset.class_to_idx.items()}

    def save_idx_to_class(self) -> None:
        output_file = self.output_dir / "idx_to_class.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as fp:
            json.dump(idx_to_class, fp, indent=4, ensure_ascii=False)

    def fit(self) -> None:
        n_epochs = self.config.trainer.n_epochs
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")

            for phase in ["train", "val"]:
                self._run_epoch(phase)

                if phase == "val" and self.early_stopping.early_stop:
                    self.model.load_state_dict(self.early_stopping.best_model_weights)
                    self.generate_train_report()
                    print("Early stopping triggered. Stopping training.")
                    print(f"Epoch {epoch + 1} completed.\n")
                    return

            self.generate_train_report()
            print(f"Epoch {epoch + 1} completed.\n")

        self.save_model_type()
        self.save_idx_to_class()

    def _run_epoch(self, phase: str) -> None:
        is_train = phase == "train"
        self.model.train() if is_train else self.model.eval()

        running_loss = 0.0
        correct = 0
        all_labels = []
        all_preds = []

        loop = tqdm(self.dataloaders[phase], leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                with autocast(device_type=self.device.type, enabled=is_train):  # AMP(自動混合精度)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                if is_train:
                    # AMP scalerを使って安全にbackward
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)

                all_labels.append(labels.detach().cpu())
                all_preds.append(torch.softmax(outputs.detach().cpu(), dim=1))  # logits -> 確率に変換

        epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
        epoch_acc = correct.double() / len(self.dataloaders[phase].dataset)

        self.history[f"{phase}_loss"].append(epoch_loss)
        self.history[f"{phase}_acc"].append(epoch_acc.item())

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if is_train:
            self.scheduler.step()
        else:
            self.early_stopping(epoch_loss, self.model)
            self._evaluate(all_labels, all_preds)

    def _evaluate(self, all_labels: torch.Tensor, all_preds: torch.Tensor) -> None:
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        preds_cls = torch.argmax(all_preds, dim=1)

        try:
            if all_preds.shape[1] == 2:
                auc_score = roc_auc_score(all_labels, all_preds[:, 1])
            else:
                auc_score = roc_auc_score(all_labels, all_preds, multi_class="ovr")
        except ValueError:
            auc_score = np.nan

        self.history["val_auc"].append(auc_score)

        # classごとの精度レポート
        report = classification_report(all_labels, preds_cls, target_names=self.class_names, output_dict=True)
        self.history["classification_report"].append(report)
