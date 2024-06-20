import argparse
from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("model_path", type=str)

    return parser.parse_args()


def main(
        model_path
        ):

    # 학습 모델 load
    model = YOLO(model_path) 

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered

    print("map50-95", metrics.box.map)
    print("map50", metrics.box.map50)


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))