import os
import argparse
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import numpy as np


def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("model_path", type=str, default="")
    parser.add_argument("img_path", type=str, default="")

    return parser.parse_args()



def main(
        model_path,
        img_path
    ):


    model = YOLO(model_path)
    result_folder = "detect/result"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    
    color_dict = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    img_src = cv2.imread(img_path)

    results = model(img_src)

    for result in results:
        annotator = Annotator(img_src)
        boxes = result.boxes
        for box in boxes:
            box_xyxy = box.xyxy[0] 
            box_cls = box.cls
            annotator.box_label(box_xyxy, model.names[int(box_cls)], color_dict[int(box_cls)])

    img_src = annotator.result()

    cv2.imshow("result", img_src)
    
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()



if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))