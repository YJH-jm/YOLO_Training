import os
import yaml
import random
import shutil

class YoloMask:
    def __init__(self, folder_path):
        self.folder_ppath=folder_path # 데이터 경로 폴더
        self.save_path = "./datasets/mask" # 저장할 폴더 경로 
        
        self.label_ = ".txt"
        self.img_ = ".jpg"
        self.label_file_list = [file for file in os.listdir(folder_path) if file.endswith(self.label_)]
        self.img_file_list = [file for file in os.listdir(folder_path) if file.endswith(self.label_)]
    
        assert len(self.img_file_list) == len(self.label_file_list)

        
        # 저장할 경로 만들기
        folder_list = [self.save_path , os.path.join(self.save_path , 'train'), os.path.join(self.save_path , 'val'), \
                       os.path.join(self.save_path , 'train', 'images'),  os.path.join(self.save_path , 'train', 'labels'), \
                       os.path.join(self.save_path , 'val', 'images'), os.path.join(self.save_path , 'val', 'labels')]
        
        for folder in folder_list:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def preprocess(self):

        # 데이터 나누기
        random.shuffle(self.label_file_list)
        
        test_ratio = 0.1
        test_list = self.label_file_list[:int(len(self.label_file_list)*test_ratio)]
        train_list = self.label_file_list[int(len(self.label_file_list)*test_ratio):]
        
        # 파일 저장 경로로 이동 
        for i in test_list:
            f_name = os.path.splitext(i)[0] # 파일 이름

            shutil.copyfile(os.path.join(self.folder_path, (f_name+self.img_)), os.path.join(self.save_path , "val/images", (f_name+self.img_)))
            shutil.copyfile(os.path.join(self.folder_path, (f_name+self.label_)), os.path.join(self.save_path , "val/labels", (f_name+self.label_)))

        for i in train_list:
            f_name = os.path.splitext(i)[0] # 파일 이름
            shutil.copyfile(os.path.join(self.folder_path, (f_name+self.img_)), os.path.join(self.save_path , "train/images", (f_name+self.img_)))
            shutil.copyfile(os.path.join(self.folder_path, (f_name+self.label_)), os.path.join(self.save_path , "train/labels",(f_name+self.label_)))                             
            
    def make_config(self):
        data = dict()
        data['train'] = os.path.join(os.getcwd(),self.save_path, 'train')
        data['val'] = os.path.join(os.getcwd(),self.save_path, 'val')
        data['test'] = os.path.join(os.getcwd(),self.save_path, 'test')

        data['nc'] = 3
        data['names'] = ['mask', 'improporly', 'no mask']

        with open('mask.yaml', 'w') as f:
            yaml.dump(data, f)
