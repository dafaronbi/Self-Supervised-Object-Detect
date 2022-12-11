# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torch.backends.cudnn as cudnn
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import os
# import copy
# import glob
# from pretext_dataset import *
# import torch.utils.data
# from tqdm import tqdm
# from datetime import datetime
# import os
# import torch
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
# import torchvision.transforms as T
# import glob
# import albumentations as A

# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
# from PIL import Image
# import yaml
# from common_functions import *

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()


# d1,d2 = dicts_common()
# class CropDataset(Dataset):
#     def __init__(self, root,transform1=None):
#         self.root = root
#         self.transform1 = transform1
#         self.images_folder = "images"
#         self.labels_folder = "labels"
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         self.imgs = list(sorted(os.listdir(os.path.join(root, self.images_folder))))[:100]
#         self.masks = list(sorted(os.listdir(os.path.join(root, self.labels_folder))))[:100]
        
#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, idx):
        
#         image = Image.open(os.path.join(self.root, self.images_folder,self.imgs[idx])).convert('RGB')
#         target_path = os.path.join(self.root, self.labels_folder, self.masks[idx])
#         target = None
#         with open(target_path, 'r') as file:
#             target = yaml.safe_load(file)
#         label = torch.tensor(d1[target['labels'][0]])
#         # predict_images = predict_images + [torchvision.utils.draw_bounding_boxes(torch.as_tensor(image,dtype=torch.uint8), torch.as_tensor([[0,0,0,0]],dtype=torch.float32), labels = [d2[(preds[i]).item()]] ) for i,image in enumerate(images)]

#         # if(self.transform1 is not None):
#         # image = self.transform1(image)
#         # image = torchvision.transforms.functional.crop(image,top = int(target['bboxes'][0][3]),left = int(target['bboxes'][0][0]) ,height=int(target['bboxes'][0][3])-int(target['bboxes'][0][1]),width=int(target['bboxes'][0][2])-int(target['bboxes'][0][0]))
#         # t2 = torchvision.transforms.CenterCrop(240)
#         # aug = A.Crop(x_min=int(target['bboxes'][0][0]), x_max=int(target['bboxes'][0][2]), y_min=int(target['bboxes'][0][1]), y_max=int(target['bboxes'][0][3]), p=1)
#         # augmented = aug(image=image)
#         # image_cropped = augmented['image']
#         t1 = transforms.PILToTensor()
    
#         return (t1(image))[:,target['bboxes'][0][1]:target['bboxes'][0][3],target['bboxes'][0][0]:target['bboxes'][0][2]], label
    


# # Data augmentation and normalization for training
# # Just normalization for validation
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.ToTensor(),
        
#         # transforms.RandomApply([transforms.RandomAffine(10),transforms.ColorJitter()],p=0.5),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# train_dataset = CropDataset('/labeled/labeled/training',data_transforms['train'])
# val_dataset = CropDataset('/labeled/labeled/validation',data_transforms['val'])


# image_datasets = {"train": train_dataset,"val":val_dataset}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,shuffle=True)
#                 for x in ['train','val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# print(dataset_sizes)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# truth_images = []

# for ind,(images,targets) in enumerate(dataloaders["train"]):
#     truth_images = truth_images + [torchvision.utils.draw_bounding_boxes(torch.as_tensor(image,dtype=torch.uint8), torch.as_tensor([[0,0,0,0]],dtype=torch.float32), labels=[d2[(targets[i]).item()]] ) for i,image in enumerate(images)]


# for i in range(len(truth_images)):
#     writer.add_image("truth image: "  +str(i), truth_images[i])
#     # writer.add_image("predict image: "  +str(i), predict_images[i])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import glob
from pretext_dataset import *
import torch.utils.data
from tqdm import tqdm
from datetime import datetime
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
import yaml
from common_functions import *

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


d1,d2 = dicts_common()
class CropDataset(Dataset):
    def __init__(self, root,transform1=None):
        self.root = root
        self.transform1 = transform1
        self.images_folder = "images"
        self.labels_folder = "labels"
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, self.images_folder))))
        self.masks = list(sorted(os.listdir(os.path.join(root, self.labels_folder))))
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.root, self.images_folder,self.imgs[idx])).convert('RGB')
        target_path = os.path.join(self.root, self.labels_folder, self.masks[idx])
        target = None
        with open(target_path, 'r') as file:
            target = yaml.safe_load(file)
        label = torch.tensor(d1[target['labels'][0]])

        # if(self.transform1 is not None):
        image = self.transform1(image)
        # image = torchvision.transforms.functional.crop(image,top = int(target['bboxes'][0][3]),left = int(target['bboxes'][0][0]),width=240 ,height=240)
        # image = torchvision.transforms.CenterCrop(240)
        # t1 = torchvision.transforms.Resize(240)
        t2 = transforms.Resize((240,240))
        return image[:,target['bboxes'][0][1]:target['bboxes'][0][3],target['bboxes'][0][0]:target['bboxes'][0][2]], label
    


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(240,240),
        transforms.RandomApply([transforms.RandomAffine(10),transforms.ColorJitter()],p=0.5),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

import torch.nn.functional as F
def custom_collate(data):
    image_batch, labels = zip(*data)
    max_height = max([img.size(1) for img in image_batch])
    max_width = max([img.size(2) for img in image_batch])

    return [
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
        for img in image_batch
    ],labels

data_dir = "/unlabeled/unlabeled"
images_list = glob.glob(f'{data_dir}/*.PNG')
num_train_images = 500000
num_val_images = 12000

train_dataset = CropDataset('/labeled/labeled/training',data_transforms['train'])
val_dataset = CropDataset('/labeled/labeled/validation',data_transforms['val'])


image_datasets = {"train": train_dataset,"val":val_dataset}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,shuffle=True,collate_fn = custom_collate)
                for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)
# class_names = image_datasets['train'].classes
# print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = models.resnet50()
#Classes being rotation of 0,90,180,270
#model.fc.out_features = 4
# model.load_state_dict(torch.load('best_model1.pt'))
model = models.resnet50()
# checkpoint = torch.load('models/pretext2.pt')
# model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
num_epochs=100

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
# log_id = date.today()
truth_images = []
predict_images = []

# log('------Training Starts---------')
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    # log(f'Epoch {epoch}/{num_epochs - 1}')
    time_elapsed = time.time() - since
    # log(f'Time Elapsed : {time.time() - since}')
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(dataloaders[phase]):
            inputs = [input.to(device) for input in inputs]
            labels = [label.to(device) for label in labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
 
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            # truth_images = truth_images + [torchvision.utils.draw_bounding_boxes(torch.as_tensor(image,dtype=torch.uint8), torch.as_tensor([0.0],dtype=torch.float32), labels=[d2[labels[i]]] ) for i,image in enumerate(inputs)]
            # predict_images = predict_images + [torchvision.utils.draw_bounding_boxes(torch.as_tensor(image,dtype=torch.uint8), torch.as_tensor([0.0],dtype=torch.float32), labels = [d2[preds[i]]] ) for i,image in enumerate(inputs)]

        # if phase == 'train':
        #     scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        # log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # torch.save(model,f'best_backbone_{log_id}.pt')
            torch.save({'epoch': epoch,'model_state_dict': model.state_dict()}, 'models/pretext2.pt')


time_elapsed = time.time() - since
# log(str(time_elapsed))
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best val Acc: {best_acc:4f}')
# log(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
# log(f'Best val Acc: {best_acc:4f}')

# # load best model weights
model.load_state_dict(best_model_wts)
# torch.save(model,f'best_backbone_{log_id}.pt')

for i in range(len(truth_images)):
    writer.add_image("truth image: "  +str(i), truth_images[i])
    writer.add_image("predict image: "  +str(i), predict_images[i])




