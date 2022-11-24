from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
import os
from torchvision.utils import save_image
import glob

# data_path = "C:/Users/varsh/Documents/Courses/DL/Project/data/unlabeled_vv"
# plt.rcParams["savefig.bbox"] = 'tight'
# orig_img = Image.open(f"{data_path}/9.PNG")
# "C:\Users\varsh\Documents\Courses\DL\Project\data\unlabeled_vv\9.PNG"
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)

        

def pretextTask_DataGeneration(data_path):

    images_list = glob.glob(f'{data_path}/*.PNG')
    number_images = len(images_list)

    angles = [0,90,180,270]
    folders = ["train","val"]
    target_folder = os.path.normpath(data_path + os.sep + os.pardir)
    for folder in folders:
        for angle in angles:
            if(os.path.exists(f"{target_folder}/backbone/{folder}/{angle}")):
                pass
            else:
                os.makedirs(f"{target_folder}/backbone/{folder}/{angle}")

    for indx,image_loc in enumerate(images_list):

        image = Image.open(image_loc)
        image_name = os.path.split(image_loc)[1]
        for angle in angles:
            rotater = T.RandomRotation((angle,angle))
            # save_image(rotater(image),f"{data_path}/{image_name[:-4]}+_90.PNG")
            if(indx>int(0.8*number_images)):
                (rotater(image)).save(f"{target_folder}/backbone/val/{angle}/{image_name}")
            else:
                (rotater(image)).save(f"{target_folder}/backbone/train/{angle}/{image_name}")
        
        
