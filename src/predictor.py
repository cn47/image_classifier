import json

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchvision.models as models

from PIL import Image, ImageFile, UnidentifiedImageError


if TYPE_CHECKING:
    import torchvision


# PILは極端に大きな画像など高速にロードできない画像はロードしないでスルーするが、
# それを回避するためにLOAD_TRUNCATED_IMAGESをTrueにする。
# これをしないとエラーになることがある
ImageFile.LOAD_TRUNCATED_IMAGES = True


### Predictorの抽象クラス設計
class Predictor:
    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        model_dir: Path,
        device: torch.device | None = None,
    ):
        self.transform = transform
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._idx_to_class = None

    @property
    def model_file(self) -> Path:
        return self.model_dir / "classifier.pth"

    @property
    def idx_to_class_file(self) -> Path:
        return self.model_dir / "idx_to_class.json"

    @property
    def idx_to_class(self) -> dict:
        if self._idx_to_class is None:
            with Path.open(self.idx_to_class_file) as fp:
                self._idx_to_class = json.load(fp)
            self._idx_to_class = {int(k): v for k, v in self._idx_to_class.items()}
        return self._idx_to_class

    def load_model(self) -> None:
        """モデルをロードする"""
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.idx_to_class))
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def _forward_img(self, img_file: Path) -> torch.Tensor:
        """画像ファイルを読み込み、前処理を行った後にモデルへ入力し、出力テンソルを返す。"""
        try:
            img = Image.open(img_file).convert("RGB")
        except UnidentifiedImageError:
            print(f"未対応フォーマットor壊れてます: {img_file}")
            print(f"Skipping file {img_file} due to UnidentifiedImageError.")
            raise
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        return outputs

    def predict(self, img_file: Path) -> dict:
        """単一の画像を予測する"""
        outputs = self._forward_img(img_file)
        _, pred = torch.max(outputs, 1)
        label = self.idx_to_class[pred.item()]

        return {"class_id": pred.item(), "class_name": label}

    def predict_proba(self, img_file: Path) -> dict:
        """単一の画像のクラス確率を予測する"""
        outputs = self._forward_img(img_file)
        probas = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()

        return {
            "proba": {self.idx_to_class[i]: float(p) for i, p in enumerate(probas)},
            "class_id": int(probas.argmax()),
            "class_name": self.idx_to_class[int(probas.argmax())],
        }

    def batch_predict(self, img_dir: Path) -> list[dict]:
        """ディレクトリ内の画像を一括で予測する"""
        results = []
        img_files = [f for f in img_dir.iterdir() if f.is_file()]
        for img_file in img_files:
            try:
                outputs = self._forward_img(img_file)
                _, pred = torch.max(outputs, 1)
                label = self.idx_to_class[pred.item()]
                results.append(
                    {
                        "class_id": int(pred.item()),
                        "class_name": label,
                        "img_file": img_file.resolve(),
                    },
                )
            except Exception as e:
                print(f"skip {img_file}: {e}")
        return results

    def batch_predict_proba(self, img_dir: Path) -> list[dict]:
        """ディレクトリ内の画像を一括でクラス確率を予測する"""
        results = []
        img_files = [f for f in img_dir.iterdir() if f.is_file()]
        for img_file in img_files:
            try:
                outputs = self._forward_img(img_file)
                probas = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
                proba_dict = {self.idx_to_class[i]: float(p) for i, p in enumerate(probas)}
                pred_idx = int(probas.argmax())
                results.append(
                    {
                        "proba": proba_dict,
                        "class_id": pred_idx,
                        "class_name": self.idx_to_class[pred_idx],
                        "img_file": img_file.resolve(),
                    },
                )
            except Exception as e:
                print(f"skip {img_file}: {e}")
        return results
