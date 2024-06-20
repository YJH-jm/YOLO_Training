import argparse
import torch
from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("model_path", type=str)
    parser.add_argument("config", type=str)
    parser.add_argument("name", type=str)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30)




    return parser.parse_args()

def main(model_path, config, name, epochs, batch, patience) :
    
    model = YOLO(model_path)

    result = model.train(model=model_path, data=config, epochs=epochs, batch=batch, device=0, patience=patience, name=name)



if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))