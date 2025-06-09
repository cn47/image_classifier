import pandas as pd

from config import Config
from IPython.display import display
from PIL import Image, ImageFile
from predictor import Predictor


# PILは極端に大きな画像など高速にロードできない画像はロードしないでスルーするが、
# それを回避するためにLOAD_TRUNCATED_IMAGESをTrueにする。
# これをしないとエラーになることがある
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_image_and_proba(result: dict):
    """画像とクラス確率を表示する"""
    img_file = result["img_file"]
    img = Image.open(img_file).convert("RGB")
    img.thumbnail(size=(400, 400), resample=Image.LANCZOS)

    proba_df = pd.DataFrame(result["proba"].items(), columns=["class_name", "proba [%]"])
    proba_df = proba_df.sort_values(by="proba [%]", ascending=False)
    proba_df["proba [%]"] = (proba_df["proba [%]"] * 100).round(2)

    return img, proba_df


if __name__ == "__main__":
    config = Config()

    model_dir = config.path.pj_dir / "models"
    model_dir = max(model_dir.iterdir(), key=lambda p: p.stat().st_mtime)

    # 出力ディレクトリ名にモデルのバージョンを含める
    input_dir = config.path.pj_dir / "requests" / "inputs"

    # Predictorのインスタンスを作成
    predictor = Predictor(
        transform=config.dataset_manager.transforms.val,
        model_dir=model_dir,
    )
    predictor.load_model()

    # 入力ディレクトリに対してバッチ推論
    # results = predictor.batch_predict(input_dir)
    results, errors = predictor.batch_predict(input_dir)

    img, proba_df = get_image_and_proba(results[1])
    display(img, proba_df.head())
