import argparse
from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default="", help="model path")
    parser.add_argument("--half", action='store_true', help='true -> fp16 tensorrt export')

    return parser.parse_args()


def main(
        model_path,
        half
        ):

    # 학습 모델 load
    model = YOLO(model_path) 
    
    # 변환
    model.export(format='engine', device=0, half=half)
    
    # 변환 된 모델 검증 
    metrics = model.val()  

    print("map50-95 : ", metrics.box.map)
    print("map50 : ", metrics.box.map50)
    print("speed : ", metrics.speed['inference'])


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))