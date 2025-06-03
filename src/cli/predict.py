import shutil

from config import Config
from predictor import Predictor


def main():
    config = Config()

    input_dir = config.path.pj_dir / "requests" / "inputs"
    output_dir = config.path.pj_dir / "requests" / "outputs"
    model_dir = config.path.pj_dir / "models"
    model_dir = max(model_dir.iterdir(), key=lambda p: p.stat().st_mtime)

    # Predictorのインスタンスを作成
    predictor = Predictor(
        transform=config.dataset_manager.transforms.val,
        model_dir=model_dir,
    )
    predictor.load_model()

    # 入力ディレクトリに対してバッチ推論
    results = predictor.batch_predict(input_dir)

    # クラス別に出力ディレクトリを作成
    for class_name in predictor.idx_to_class.values():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    # 推論結果をもとに画像を各クラスに分類保存
    for result in results:
        class_name = result["class_name"]
        img_file = result["img_file"]
        output_file = output_dir / class_name / img_file.name
        shutil.copy(img_file, output_file)

    # Notes:
    # 画像のコピーのときに同ファイル名があった場合は上書きされる。
    # もし上書きしたくない場合は、output_file.exists()をチェックして、
    # 既存のファイル名に連番を付けるなどの処理を追加する必要がある。


if __name__ == "__main__":
    main()
