"""
Name: data.py
Function: define data loaders for object detection
"""

import os
import numpy as numpy
from PIL import Image
import yaml
import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F, InterpolationMode, transforms as T

#dictionary transforming a label to index
label_to_num = {'motorcycle': 0, 'crutch': 1, 'sunglasses': 2, 'harp': 3, 'fox': 4, 'whale': 5, 'cattle': 6, 
'hippopotamus': 7, 'turtle': 8, 'otter': 9, 'sofa': 10, 'violin': 11, 'bowl': 12, 'jellyfish': 13, 'apple': 14, 'tie': 15, 
'red panda': 16, 'skunk': 17, 'rabbit': 18, 'bench': 19, 'frog': 20, 'drum': 21, 'cup or mug': 22, 'lemon': 23, 'beaker': 24, 
'mushroom': 25, 'dragonfly': 26, 'bookshelf': 27, 'cucumber': 28, 'backpack': 29, 'airplane': 30, 'salt or pepper shaker': 31, 
'antelope': 32, 'bird': 33, 'piano': 34, 'koala bear': 35, 'guitar': 36, 'cream': 37, 'domestic cat': 38, 'bicycle': 39, 
'croquet ball': 40, 'ray': 41, 'pomegranate': 42, 'coffee maker': 43, 'flower pot': 44, 'lizard': 45, 'fig': 46, 'ski': 47, 
'pitcher': 48, 'elephant': 49, 'monkey': 50, 'banana': 51, 'person': 52, 'table': 53, 'sheep': 54, 'orange': 55, 'bus': 56, 
'artichoke': 57, 'horse': 58, 'dumbbell': 59, 'miniskirt': 60, 'traffic light': 61, 'laptop': 62, 'goldfish': 63, 'dog': 64, 
'bagel': 65, 'wine bottle': 66, 'baby bed': 67, 'car': 68, 'nail': 69, 'helmet': 70, 'butterfly': 71, 'stove': 72, 'bear': 73, 
'seal': 74, 'cart': 75, 'axe': 76, 'tiger': 77, 'tape player': 78, 'chair': 79, 'computer keyboard': 80, 'porcupine': 81, 
'train': 82, 'strawberry': 83, 'lobster': 84, 'starfish': 85, 'ladybug': 86, 'camel': 87, 'swine': 88, 'pretzel': 89, 
'hat with a wide brim': 90, 'bell pepper': 91, 'snake': 92, 'tv or monitor': 93, 'bathing cap': 94, 'zebra': 95, 'lamp': 96, 
'purse': 97, 'watercraft': 98, 'microphone': 99}




def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.Resize((224,224)))
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class labelled_data(torch.utils.data.Dataset):
    def __init__(self, data_dir, d_type, transforms = None):
        self.data_dir = data_dir
        self.d_type = d_type
        self.transforms = transforms

        self.images = list(sorted(os.listdir(os.path.join(data_dir,d_type, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(data_dir,d_type, "labels"))))
    
    def __getitem__(self, idx):
        #load images and labels
        image_path = os.path.join(self.data_dir, self.d_type, "images", self.images[idx])
        label_path = os.path.join(self.data_dir, self.d_type, "labels", self.labels[idx])

        #make RGB PIL image
        image  = Image.open(image_path).convert("RGB")

        #convert to tensors and apply transforms
        with open(label_path) as f:
            labels = yaml.safe_load(f)

        labels['bboxes_norm'] = torch.as_tensor([[bbox[0] / labels['image_size'][0], bbox[1] / labels['image_size'][1],
        bbox[2] / labels['image_size'][0], bbox[3] / labels['image_size'][1]] for bbox in labels['bboxes']], dtype=torch.float32)
        labels['bboxes'] = torch.as_tensor(labels['bboxes'], dtype=torch.int64)
        labels['image_size'] = torch.as_tensor(labels['image_size'], dtype=torch.int64)
        labels['labels'] = torch.as_tensor(torch.nn.functional.one_hot(torch.as_tensor([ label_to_num[label] for label in labels['labels']], dtype=torch.int64), num_classes=100), dtype=torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image,labels

    def __len__(self):
        return len(self.images)
