#######################################################
#               Define Dataset Class
#######################################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as T
import glob

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from PIL import Image



class RotationDataset(Dataset):
    def __init__(self, image_paths, angle=None,transform=None):
        self.image_paths = image_paths
        self.angle = angle
        self.rotation = T.RandomRotation((angle,angle))
        self.transform=transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx])
        label = 0
        if self.angle is not None:
            image = self.rotation(image)
            label = self.angle
        if(self.transform is not None):
            image = self.transform(image)
        
        return image, label
    
#######################################################
#                  Create Dataset
#######################################################

# train_dataset = LandmarkDataset(train_image_paths,train_transforms)
# valid_dataset = LandmarkDataset(valid_image_paths,test_transforms) #test transforms are applied
# test_dataset = LandmarkDataset(test_image_paths,test_transforms)