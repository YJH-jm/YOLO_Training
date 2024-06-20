
import os 
import yaml

def pascal_to_yolo_bbox(bbox, w, h):
        
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2]-bbox[0]) / w
    height = (bbox[3]-bbox[1]) / h

    return [x_center, y_center, width, height]


def make_config(path, classes, name):
    data = dict()
    data['train'] = os.path.join(os.getcwd(), path, 'train')
    data['val'] = os.path.join(os.getcwd(), path, 'val')
    if os.path.isdir(os.path.join(os.getcwd(), path, 'test')):
        data['test'] = os.path.join(os.getcwd(), path, 'test')
    else:
        data['test'] = os.path.join(os.getcwd(), path, 'val')

    data['nc'] = len(classes)
    data['names'] = classes

    with open(f'config/{name}.yaml', 'w') as f:
        yaml.dump(data, f)