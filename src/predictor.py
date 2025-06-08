import json

from pathlib import Path

import torch

from models_registry import get_model
from PIL import Image, ImageFile, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm


# PILは極端に大きな画像など高速にロードできない画像はロードしないでスルーするが、
# それを回避するためにLOAD_TRUNCATED_IMAGESをTrueにする。
# これをしないとエラーになることがある
ImageFile.LOAD_TRUNCATED_IMAGES = True


### Predictorの抽象クラス設計
class Predictor:
    def __init__(
        self,
        transform: transforms.Compose,
        model_dir: Path,
        device: torch.device | None = None,
    ):
        self.transform = transform
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        self._model_dir = model_dir
        self._idx_to_class = None
        self._model_type = None

    @property
    def model_file(self) -> Path:
        return self._model_dir / "classifier.pth"

    @property
    def idx_to_class_file(self) -> Path:
        return self._model_dir / "idx_to_class.json"

    @property
    def model_type_file(self) -> Path:
        return self._model_dir / "model_type.json"

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @model_dir.setter
    def model_dir(self, value: Path) -> None:
        self._model_dir = value
        self._idx_to_class = None
        self._model_type = None

    @property
    def idx_to_class(self) -> dict:
        if self._idx_to_class is None:
            with Path.open(self.idx_to_class_file) as fp:
                self._idx_to_class = json.load(fp)
            self._idx_to_class = {int(k): v for k, v in self._idx_to_class.items()}
        return self._idx_to_class

    @property
    def model_type(self) -> str:
        if self._model_type is None:
            with Path.open(self.model_type_file) as fp:
                self._model_type = json.load(fp)
        return self._model_type["model_type"]

    def load_model(self) -> None:
        """モデルをロードする"""
        self.model = get_model(model_type=self.model_type, n_classes=len(self.idx_to_class), pretrained=False)
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
        """単一の画像のクラス確率を予測し、確率分布と所属確率が最も高いクラスを出力する

        所属確率は降順にソートされる

        """
        outputs = self._forward_img(img_file)
        probas = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()

        # クラス確率を降順にソート
        sorted_probas = {self.idx_to_class[i]: float(p) for i, p in enumerate(probas)}
        sorted_probas = dict(sorted(sorted_probas.items(), key=lambda item: item[1], reverse=True))

        return {
            "class_id": int(probas.argmax()),
            "class_name": self.idx_to_class[int(probas.argmax())],
            "proba": sorted_probas,
        }

    def batch_predict(self, img_dir: Path) -> tuple[list[dict], list[dict]]:
        """ディレクトリ内の各画像のクラス確率を予測し、確率分布と所属確率が最も高いクラスを出力する"""
        results = []
        error_results = []
        img_files = [f for f in img_dir.iterdir() if f.is_file()]
        for img_file in tqdm(img_files):
            try:
                result = self.predict(img_file)
                result["img_file"] = img_file.resolve()
                results.append(result)
            except Exception as e:
                print(f"skip {img_file}: {e}")
                error_results.append({"img_file": img_file.resolve(), "error": str(e)})

        return results, error_results
