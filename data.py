"""
Name: data.py
Function: define data loaders for object detection
"""

import os
import numpy as numpy
import torch
from PIL import Image
import yaml

class labelled_data(torch.utils.data.Dataset):
    def __init__(self, data_dir, d_type):
        self.data_dir = data_dir
        self.d_type = d_type

        self.images = list(sorted(os.listdir(os.path.join(data_dir,d_type, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(data_dir,d_type, "labels"))))
    
    def __getitem__(self, idx):
        #load images and labels
        image_path = os.path.join(self.data_dir, self.d_type, "images", self.images[idx])
        label_path = os.path.join(self.data_dir, self.d_type, "labels", self.labels[idx])

        image  = Image.open(image_path).convert("RGB")

        with open(label_path) as f:
            labels = yaml.safe_load(f)

        return image,labels

    def __len__(self):
        return len(slef.images)
