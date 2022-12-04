import torch
import numpy as np
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
import torch.nn.functional as TF
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

load_path = "saved_model.pt"

#resize dimensions for model
image_x = 224
image_y = 224

#max number of boxes predictable
num_boxes = 5

#score threshold for inference
score_threshold = 0.5

#device
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VGG(torch.nn.Module):
    def __init__(self, num_classes=100, in_channels=3, num_boxes=num_boxes, score_threshold=score_threshold,
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(VGG, self).__init__()
        # resize image
        image_x = 224
        image_y = 224
        # device where tensors are loaded
        self.device = device
        # number of classes
        self.num_classes = num_classes
        # number of in_channels
        self.in_channels = in_channels
        # number of bounding boxes
        self.num_boxes = num_boxes
        # score threshold
        self.score_threshold = score_threshold
        # transforms are resizing and converting to tensor
        self.transforms = T.Compose([T.Resize((image_x,image_y)),T.ConvertImageDtype(torch.float)])
        
        # conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # flatten the layer
        self.flatten = torch.nn.Flatten()

        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=(512*7*7), out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.labelout = nn.Linear(4096, 100*num_boxes)
        self.labelsoftmax = nn.Softmax(dim=1)
        self.bboxoutstart = nn.Linear(4096, 2*num_boxes)
        self.bboxoutsize = nn.Linear(4096, 2*num_boxes)
        self.scoreout = nn.Linear(4096, num_boxes)
   
    def forward(self, x):

        #get the scale reduction from image resize
        scale_x = torch.tensor([[[img.shape[1], 1, img.shape[1], 1]]*num_boxes for img in x])
        scale_y = torch.tensor([[[1, img.shape[2], 1, img.shape[2]]]*num_boxes for img in x])

        #send tensors to device
        scale_x = scale_x.to(self.device)
        scale_y = scale_y.to(self.device)

        #data input processing with transforms
        x = [self.transforms(img) for img in x]    
        x = torch.stack(x)

        #load pretext RCNN model (self supervised)
        # pretext = torch.load('best_model_2022-11-29.pt', map_location=self.device)
        # pretext = pretext.to(self.device)
        # pretext.eval()

        #run data through pretrained model
        # p_out = pretext(x)

        #run data through model
        out = self.conv_layers(x)
        out = self.flatten(out)
        out = self.linear_layers(out)

        labels = torch.reshape(self.labelout(out), (-1, num_boxes, 100))
        labels_t = self.labelsoftmax(labels)
        labels_i = torch.as_tensor(torch.argmax(labels_t, dim=2), dtype=torch.float32)

        #bounding box output is xywh -> normalized x1y1x2y2
        bbox_start = torch.sigmoid(torch.reshape(self.bboxoutstart(out), (-1,num_boxes,2)))
        bbox_size = torch.sigmoid(torch.reshape(self.bboxoutsize(out), (-1,num_boxes,2)))
        bbox_end = torch.add(bbox_start,bbox_size)
        bbox = torch.cat((bbox_start, bbox_end), dim=-1)

        # bbox[...,[2,3]] += bbox[..., [0,1]]
        score = torch.sigmoid(self.scoreout(out))

        #scale bbox
        bbox = bbox*scale_x*scale_y

        if self.training:
            return [ {"labels": l, 
            "boxes": b,  
            "scores": s} for l,b,s in zip(labels_t,bbox,score)]  #labels_t
            
        else:
            o = []
            for l,b,s in zip(labels_i,bbox,score): #labels_i
                keep_dim = []
                for i,score in enumerate(s):
                    if score > self.score_threshold:
                        keep_dim.append(i)
                o.append({"labels": l[keep_dim], "boxes": b[keep_dim,:], "scores": s[keep_dim]})
            return o   

#get model functon used for evalutatoin
def get_model():
    #load model
    network = VGG()
    network.load_state_dict(torch.load(load_path, map_location=device))
    return network