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
label_to_num = {'car': 0, 'chair': 1, 'horse': 2, 'fox': 3, 'laptop': 4, 'tie': 5, 'bathing cap': 6, 'baby bed': 7, 
'snake': 8, 'orange': 9, 'sheep': 10, 'koala bear': 11, 'lemon': 12, 'guitar': 13, 'bagel': 14, 'dog': 15, 
'airplane': 16, 'monkey': 17, 'ski': 18, 'sofa': 19, 'watercraft': 20, 'tiger': 21, 'wine bottle': 22, 
'sunglasses': 23, 'butterfly': 24, 'whale': 25, 'goldfish': 26, 'hippopotamus': 27, 'drum': 28, 
'coffee maker': 29, 'stove': 30, 'cart': 31, 'red panda': 32, 'mushroom': 33, 'traffic light': 34, 
'dragonfly': 35, 'harp': 36, 'croquet ball': 37, 'bookshelf': 38, 'computer keyboard': 39, 'rabbit': 40,
 'helmet': 41, 'hat with a wide brim': 42, 'strawberry': 43, 'antelope': 44, 'purse': 45, 'lobster': 46, 
 'skunk': 47, 'fig': 48, 'bird': 49, 'apple': 50, 'bicycle': 51, 'piano': 52, 'miniskirt': 53, 'bear': 54, 
 'cattle': 55, 'dumbbell': 56, 'person': 57, 'flower pot': 58, 'tape player': 59, 'tv or monitor': 60,
  'ray': 61, 'crutch': 62, 'microphone': 63, 'pitcher': 64, 'starfish': 65, 'elephant': 66, 'camel': 67, 
  'banana': 68, 'swine': 69, 'seal': 70, 'jellyfish': 71, 'artichoke': 72, 'domestic cat': 73, 'bus': 74,
   'frog': 75, 'salt or pepper shaker': 76, 'cup or mug': 77, 'backpack': 78, 'pretzel': 79, 'pomegranate': 80, 
   'lamp': 81, 'otter': 82, 'violin': 83, 'motorcycle': 84, 'bench': 85, 'bowl': 86, 'train': 87, 'axe': 88, 
   'ladybug': 89, 'table': 90, 'porcupine': 91, 'bell pepper': 92, 'lizard': 93, 'cream': 94, 'nail': 95, 'turtle': 96}




def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
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

        labels['bboxes'] = torch.as_tensor(labels['bboxes'], dtype=torch.int64)
        labels['image_size'] = torch.as_tensor(labels['image_size'], dtype=torch.int64)
        labels['labels'] = torch.as_tensor([ label_to_num[label] for label in labels['labels']], dtype=torch.int64)

        if self.transforms is not None:
            image = self.transforms(image)

        return image,labels

    def __len__(self):
        return len(slef.images)
