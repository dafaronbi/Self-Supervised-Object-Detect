import torch
import numpy as np
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
import torch.nn.functional as TF
import torch.nn as nn

load_path = "saved_model.pt"

#resize dimensions for model
image_x = 224
image_y = 224

#max number of boxes predictable
num_boxes = 5

#score threshold for inference
score_threshold = 0.5

class resNet(torch.nn.Module):
    def __init__(self, device):
        super(resNet, self).__init__()

        #model layers
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.ReLU1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5)
        self.ReLU2 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(1*218*218, 128)
        self.labelout = torch.nn.Linear(128, 100*num_boxes)
        self.bboxout = torch.nn.Linear(128, 4*num_boxes)
        self.scoreout = torch.nn.Linear(128, num_boxes)

        #transforms are resizing and converting to tensor
        self.transforms = T.Compose([T.Resize((image_x,image_y)),T.ConvertImageDtype(torch.float)])

        #device where tensors are loaded
        self.device = device

        
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

        #run data through model
        out = TF.relu(self.conv1(x))
        out = TF.relu(self.conv2(out))
        out = self.flatten(out)
        out = self.linear1(out)
        labels = torch.reshape(self.labelout(out), (-1, num_boxes, 100))
        labels = torch.as_tensor(torch.argmax(labels, dim=2), dtype=torch.float32)
        bbox = torch.sigmoid(torch.reshape(self.bboxout(out), (-1,num_boxes,4)))
        score = torch.sigmoid(self.scoreout(out))

        #scale bbox
        bbox = bbox*scale_x*scale_y

        if self.training:
            return [ {"labels": l, 
            "boxes": b,  
            "scores": s} for l,b,s in zip(labels,bbox,score)] 
            
        else:
            o = []
            for l,b,s in zip(labels,bbox,score):
                keep_dim = []
                for i,score in enumerate(s):
                    if score > score_threshold:
                        keep_dim.append(i)
                o.append({"labels": l[keep_dim], "boxes": b[keep_dim,:], "scores": s[keep_dim]})
            return o

#get model functon used for evalutatoin
def get_model():
    
    #load model
    network = resNet()
    network.load_state_dict(torch.load(load_path))

    return network

