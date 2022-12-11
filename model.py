import torchvision
import torch
import os
import numpy as np
import torch
from PIL import Image
import yaml
from common_functions import * 
from torchvision import datasets, models, transforms
from torch.nn import CrossEntropyLoss,MSELoss
from tqdm import tqdm
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch import nn
#import transforms as T
from torchvision import datasets, models, transforms
from engine import *
from utils import *
import time

def create_model(res50_custom,num_classes=100):
    # Load the pretrained features.

    backbone = torch.nn.Sequential(*(list(res50_custom.children())[:-1]))
    # for param in backbone.parameters():
    #     param.requires_grad = False

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    backbone.out_channels = 2048

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model



def get_model():
    '''Step1 : RPN Training'''
    resnet50 = torch.load('models/best_backbone_2022-12-05.pt')
    model = create_model(resnet50,get_num_classes())
    checkpoint = torch.load('models/best_stage3.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model