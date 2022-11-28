from __future__ import print_function, division

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

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "C:/Users/varsh/Documents/Courses/DL/Project/data/unlabeled_vv"
images_list = glob.glob(f'{data_dir}/*.PNG')

train_dataset0 = RotationDataset(images_list[:10],0)
train_dataset1 = RotationDataset(images_list[:10],90)
train_dataset2 = RotationDataset(images_list[:10],180)
train_dataset3 = RotationDataset(images_list[:10],270)
train_dataset = torch.utils.data.ConcatDataset([train_dataset0,train_dataset1,train_dataset2,train_dataset3])
val_dataset = RotationDataset(images_list[10:12],180)

image_datasets = {"train": train_dataset,"val":val_dataset}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=1)
                for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)
# class_names = image_datasets['train'].classes
# print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50()
#Classes being rotation of 0,90,180,270
model.fc.out_features = 4

num_epochs=2

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

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

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        # if phase == 'train':
        #     scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

# time_elapsed = time.time() - since
# print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
# print(f'Best val Acc: {best_acc:4f}')

# # load best model weights
# model.load_state_dict(best_model_wts)
# torch.save(model,'best_backbone_1.pt')




