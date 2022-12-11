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
from tqdm import tqdm
from datetime import date

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

data_dir = "/unlabeled/unlabeled"
images_list = glob.glob(f'{data_dir}/*.PNG')
num_train_images = 500000
num_val_images = 12000
train_dataset0 = RotationDataset(images_list[:num_train_images],0,transform=data_transforms['train'])
train_dataset1 = RotationDataset(images_list[:num_train_images],90,transform=data_transforms['train'])
train_dataset2 = RotationDataset(images_list[:num_train_images],180,transform=data_transforms['train'])
train_dataset3 = RotationDataset(images_list[:num_train_images],270,transform=data_transforms['train'])
train_dataset = torch.utils.data.ConcatDataset([train_dataset0,train_dataset1,train_dataset2,train_dataset3])

val_dataset0 = RotationDataset(images_list[num_train_images:num_train_images+num_val_images],0,transform=data_transforms['val'])
val_dataset1 = RotationDataset(images_list[num_train_images:num_train_images+num_val_images],90,transform=data_transforms['val'])
val_dataset2 = RotationDataset(images_list[num_train_images:num_train_images+num_val_images],180,transform=data_transforms['val'])
val_dataset3 = RotationDataset(images_list[num_train_images:num_train_images+num_val_images],270,transform=data_transforms['val'])
val_dataset =  torch.utils.data.ConcatDataset([val_dataset0,val_dataset1,val_dataset2,val_dataset3])

image_datasets = {"train": train_dataset,"val":val_dataset}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=256,shuffle=True, num_workers=2)
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
model = torch.load('best_model_2022-11-29.pt')
model = model.to(device)
num_epochs=10

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
log_id = date.today()

def log(s):
    with open(f'log_{log_id}.txt', 'a+') as f:
        f.write(str(s))
        f.write("\n")

log('------Training Starts---------')
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    log(f'Epoch {epoch}/{num_epochs - 1}')
    time_elapsed = time.time() - since
    log(f'Time Elapsed : {time.time() - since}')
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
        log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model,f'best_backbone_{log_id}.pt')


time_elapsed = time.time() - since
log(str(time_elapsed))
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best val Acc: {best_acc:4f}')
log(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
log(f'Best val Acc: {best_acc:4f}')

# # load best model weights
model.load_state_dict(best_model_wts)
torch.save(model,f'best_backbone_{log_id}.pt')




