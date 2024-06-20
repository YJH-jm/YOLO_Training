import os
import glob
import yaml
import random
import shutil
import xml.etree.ElementTree as ET

class RoadSign:
    def __init__(self, folder_path = "./road_sign"):
        self.folder_path = folder_path
        self.annot_path = os.path.join(self.folder_path ,"annotations")
        self.img_path = os.path.join(self.folder_path ,"images")
        self.label_path = os.path.join(self.folder_path ,"labels")
        self.save_path  = "datasets/road_sign"
        self.classes = []
        self.label_ = '.xml'
        self.yolo_label_=".txt"
        self.img_ = '.png'
        self.annot_files = glob.glob(os.path.join(self.annot_path, "*" + self.label_))
        self.img_files = glob.glob(os.path.join(self.img_path, "*" + self.img_))
        assert len(self.img_files) == len(self.annot_files)


        if not os.path.exists(self.label_path):
            os.makedirs(self.label_path)


        # 저장할 경로 만들기
        folder_list = [self.save_path , os.path.join(self.save_path , 'train'), os.path.join(self.save_path , 'val'), \
                       os.path.join(self.save_path , 'train', 'images'),  os.path.join(self.save_path , 'train', 'labels'), \
                       os.path.join(self.save_path , 'val', 'images'), os.path.join(self.save_path , 'val', 'labels')]
        
        # 저장 경로 만들기
     
        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def preprocess(self):
        
        # Pascal to Yolo      
        for file in self.annot_files:
            basename = os.path.basename(file)
            
            filename = os.path.splitext(basename)[0]
            result = []
        
            # xml 파일을 읽기 위한 처리
            tree = ET.parse(file)
            root = tree.getroot()
            
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            for obj in root.findall('object'): # 모든 object에 대하여
                label = obj.find("name").text
              
                if label not in self.classes: # label index화 
                    self.classes.append(label)
                index = self.classes.index(label)
                pil_bbox = [int(x.text) for x in obj.find("bndbox")]
                yolo_bbox = self.pascal_to_yolo_bbox(pil_bbox, width, height)
                bbox_string = " ".join([str(x) for x in yolo_bbox])
                result.append(f"{index} {bbox_string}")
        
            if result:
                with open(os.path.join(self.label_path, f"{filename}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(result))



        
        # 데이터 나누고 옮기기
        random.shuffle(self.annot_files)
        test_ratio = 0.1
        test_list = self.annot_files[:int(len(self.annot_files)*test_ratio)]
        train_list = self.annot_files[int(len(self.annot_files)*test_ratio):]

        print(f"train의 개수 : {len(train_list)}, test의 개수 : {len(test_list)}")

        for i in test_list:
            f_base = os.path.basename(i)
            f_name = os.path.splitext(f_base)[0]
            print(f_name)
            shutil.copyfile(os.path.join(self.img_path, (f_name+self.img_)), os.path.join(self.save_path, 'val/images', (f_name+self.img_)))
            shutil.copyfile(os.path.join(self.label_path, (f_name+self.yolo_label_)), os.path.join(self.save_path, 'val/labels', (f_name+self.yolo_label_)))
        
        for i in train_list:
            f_base = os.path.basename(i)
            f_name = os.path.splitext(f_base)[0]

            shutil.copyfile(os.path.join(self.img_path, (f_name+self.img_)), os.path.join(self.save_path, 'train/images', (f_name+self.img_)))
            shutil.copyfile(os.path.join(self.label_path, (f_name+self.yolo_label_)), os.path.join(self.save_path, 'train/labels', (f_name+self.yolo_label_)))
        
        self.make_config()
    
    def pascal_to_yolo_bbox(self, bbox, w, h):
        
        # xmin, ymin, xmax, ymax
        x_center = ((bbox[2] + bbox[0]) / 2) / w
        y_center = ((bbox[3] + bbox[1]) / 2) / h
        width = (bbox[2]-bbox[0]) / w
        height = (bbox[3]-bbox[1]) / h

        return [x_center, y_center, width, height]


    def make_config(self):
        data = dict()
        data['train'] = os.path.join(os.getcwd(),self.save_path, 'train')
        data['val'] = os.path.join(os.getcwd(),self.save_path, 'val')
        data['test'] = os.path.join(os.getcwd(),self.save_path, 'test')

        data['nc'] = 4
        data['names'] = self.classes

        with open('road_sign.yaml', 'w') as f:
            yaml.dump(data, f)