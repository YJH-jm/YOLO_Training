import argparse
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--yaml", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--half", action='store_true', help='true -> fp16 tensorrt export')
    

    return parser.parse_args()


def main(
        yaml,
        model_path,
        half):
    
    model = YOLO(model_path)
    results = model.val(data=yaml, batch=1, imgsz=640, plots=False, device=0, half=half, verbose=False)
    metric, speed = results.results_dict['metrics/mAP50-95(B)'], results.speed['inference']
   
    print(metric, speed)

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
