import os
import argparse
import json
from glob import glob
import yaml
import random
import shutil
import xml.etree.ElementTree as ET

from utils import pascal_to_yolo_bbox, make_config

# Training/바운딩박스/서울특별시/[원천]서울특별시_FLRR_1_01.zip   - 압축풀어서 이미지를 img폴더에 .png 파일들만 넣어줌
# Training/바운딩박스/서울특별시/[라벨]서울특별시_FLRR_1_01.zip   - 압축풀어서 라벨파일들을 label폴더에 .json 파일들만 넣어줌

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Preprocessing ")
    parser.add_argument("data_dir", type=str, help="data download path")
    
    return parser.parse_args()


def road(args):
    
    # 데이터 저장 폴더 생성
    save_path = "datasets/road/"
    if not os.path.isdir(os.path.join(save_path, 'labels')):
        os.makedirs(os.path.join(save_path, 'labels'))

    if not os.path.isdir(os.path.join(save_path, 'images')):
        os.makedirs(os.path.join(save_path, 'images'))
    
    
    city = ['서울특별시']
    region = ['영등포구', '종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구', '노원구','은평구', \
    '서대문구', '마포구', '양천구', '강서구', '구로구','금천구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구']
    weather = ['맑음']
    period = ['주간', '야간']
    location = ['실내', '실외']
    
    classes = ['car', 'pedestrian']
    cars = ['일반차량', '목적차량(특장차)', '이륜차']


    label_path = os.path.join(args.data_dir, "label")
    img_path = os.path.join(args.data_dir, "img")
    label_list = os.listdir(label_path)
 
    
    
    for file in label_list:
        with open(os.path.join(label_path, file), 'r', encoding='utf-8') as f: # json 폴더명 깨져서 수정
            data = json.load(f)
            
            if not os.path.isfile(os.path.join(img_path, data['filename'])):
                print("There is no image file")
                break

            # 파일 한글명 제거
            name, img_ext = data['filename'].split('.')
            p = name.split("_")
            p[1] = str(city.index(p[1]))
            p[3] = f"{region.index(p[3]):02d}"
            p[4] = str(weather.index(p[4]))
            p[5] = str(period.index(p[5]))
            p[6] = str(location.index(p[6]))
            new_name = "_".join(p) 

            # 이미지 파일 이동
            shutil.copyfile(os.path.join(img_path, data['filename']), os.path.join(save_path, 'images',new_name + "."+ img_ext))

            height, width = int(data['camera']['resolution_height']), int(data['camera']['resolution_width']) 

            with open(os.path.join(save_path, 'labels', new_name + '.txt'), "w", encoding="utf-8") as txt_file:
                for annot in data['annotations']:
                    if annot['label'] in cars or annot['label'] == "보행자":
                        if annot['label'] in cars:
                            label_idx = 0
                        else:
                            label_idx = 1
                        
                        bbox = annot['points']
                        yolo_bbox = pascal_to_yolo_bbox([int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])], width,height)
                        bbox_string = " ".join([str(x) for x in yolo_bbox])
                        result = f"{label_idx} {bbox_string}\n"
                        txt_file.write(result)


    # train, test split
    folder_list = [os.path.join(save_path , 'train'), os.path.join(save_path , 'val'), \
                       os.path.join(save_path , 'train', 'images'),  os.path.join(save_path , 'train', 'labels'), \
                       os.path.join(save_path , 'val', 'images'), os.path.join(save_path , 'val', 'labels')]

    for folder in folder_list:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    test_ratio = 0.1
    random.seed(2020)

    img_path = os.path.join(save_path, 'images') 
    img_list = os.listdir(img_path)
    random.shuffle(img_list)

    val_list = img_list[:int(len(img_list) * test_ratio)]
    train_list = img_list[int(len(img_list) * test_ratio):]

    for file in val_list:
        name, ext = file.split('.')
        shutil.copyfile(os.path.join(save_path, 'images', file), os.path.join(save_path, 'val','images',file))
        shutil.copyfile(os.path.join(save_path, 'labels', name+".txt"), os.path.join(save_path, 'val','labels',name+".txt"))
    
    for file in train_list:
        name, ext = file.split('.')
        shutil.copyfile(os.path.join(save_path, 'images', file), os.path.join(save_path, 'train','images',file))
        shutil.copyfile(os.path.join(save_path, 'labels', name+".txt"), os.path.join(save_path, 'train','labels',name+".txt"))

    make_config(save_path, classes, "road")

if __name__ == "__main__":
    args = parse_arguments()
    road(args)
