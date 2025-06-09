import shutil

from config import Config
from predictor import Predictor


def main():
    config = Config()

    # 最新のモデルを取得
    model_dir = config.path.pj_dir / "models"
    model_dir = max(model_dir.iterdir(), key=lambda p: p.stat().st_mtime)

    # 出力ディレクトリ名にモデルのバージョンを含める
    input_dir = config.path.pj_dir / "requests" / "inputs"
    output_dir = config.path.pj_dir / "requests" / "outputs" / model_dir.parts[-1]

    # Predictorのインスタンスを作成
    predictor = Predictor(transform=config.dataset_manager.transforms.val, model_dir=model_dir)
    predictor.load_model()

    # 入力ディレクトリに対してバッチ推論
    results, errors = predictor.batch_predict(input_dir)

    # クラス別に出力ディレクトリを作成
    for class_name in predictor.idx_to_class.values():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    # 推論結果をもとに画像を各クラスに分類保存
    for result in results:
        class_name = result["class_name"]
        img_file = result["img_file"]
        output_file = output_dir / class_name / img_file.name

        if output_file.exists():
            print(f"File {output_file} already exists. Overwriting...")
            continue

        shutil.copy(img_file, output_file)

    print(f"Results saved to {output_dir}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
