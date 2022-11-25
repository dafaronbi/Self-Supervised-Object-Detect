import torch
import numpy as np
from torchvision.transforms import functional as F, InterpolationMode, transforms as T

load_path = "saved_model.pt"

class resNet(torch.nn.Module):
    def __init__(self):
        super(resNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.ReLU1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5)
        self.ReLU2 = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(47524, 128)
        self.labelout = torch.nn.Linear(128, 100)
        self.bboxout = torch.nn.Linear(128, 4)
        self.scoreout = torch.nn.Linear(128, 1)
        self.transforms = T.Compose([T.Resize((224,224)),T.ConvertImageDtype(torch.float)])


    def forward(self, x):
        x = torch.stack(x)
        x = self.transforms(x)
        out = self.ReLU1(self.conv1(x))
        out = self.ReLU2(self.conv2(out))
        out = self.flatten(out)
        out = self.linear1(out)
        labels = self.labelout(out)
        labels = torch.argmax(labels, dim=1)
        bbox = self.bboxout(out)
        score = self.scoreout(out)

        return [ {"labels": torch.stack([l]), 
        "boxes": torch.stack([b]),  
        "scores": torch.stack([s[0]])} for l,b,s in zip(labels,bbox,score)] 

#get model functon used for evalutatoin
def get_model():
    
    #load model
    network = resNet()
    network.load_state_dict(torch.load(load_path))

    return network

