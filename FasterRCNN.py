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


data_path = "C:/Users/varsh/Documents/Courses/DL/Project/data/labeled/"

class ObjectDetection(torch.utils.data.Dataset):
    def __init__(self, root,transforms=None):
        self.root = root
        self.transforms = transforms
        self.images_folder = "images"
        self.labels_folder = "labels"
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, self.images_folder))))
        self.masks = list(sorted(os.listdir(os.path.join(root, self.labels_folder))))
        

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.images_folder, self.imgs[idx])
        target_path = os.path.join(self.root, self.labels_folder, self.masks[idx])
        targets = None
        with open(target_path, 'r') as file:
            targets = yaml.safe_load(file)
        img = Image.open(img_path).convert("RGB")

        #---- TARGETS------
        target_return = {}
        target_return['boxes'] = torch.as_tensor(targets['bboxes'],dtype=torch.float32)
        target_return['labels'] = torch.as_tensor([label2id_encode(cat) for cat in targets['labels']])
        # target_return['area'] = torch.as_tensor([(box[2]-box[0])*(box[3]-box[1]) for box in targets['bboxes']],dtype=torch.float32)
        # target_return['iscrowd'] = False
        # print(self.masks[idx])
        # target_return['image_id'] = torch.tensor([int(self.masks[idx][:-4])])

        if self.transforms is not None:
            img = self.transforms(img)


        return img, target_return

    def __len__(self):
        return len(self.imgs)


transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
objDet_Dataset = ObjectDetection(data_path)
print(objDet_Dataset[0])
# split the dataset in train and test set
indices = torch.randperm(len(objDet_Dataset)).tolist()
dataset = torch.utils.data.Subset(objDet_Dataset, indices[:10])
dataset_test = torch.utils.data.Subset(objDet_Dataset, indices[10:12])


backbone_model = torch.load('best_model_2022-11-29.pt')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=1001,weights_backbone=backbone_model.state_dict())



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters()]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

# let's train it for 10 epochs
num_epochs = 2
classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()
for epoch in tqdm(range(num_epochs)):
    totalTrainLoss = 0.0
    trainCorrect = 0.0

    for (images, targets) in tqdm(data_loader):
        (images, boxes,labels) = (images.to(device),targets['boxes'].to(device), targets['labels'].to(device))
		# perform a forward pass and calculate the training loss
        predictions = model(images)
        bboxLoss = bboxLossFunc(predictions[0], boxes)
        classLoss = classLossFunc(predictions[1], labels)
        totalLoss = (bboxLoss) + (classLoss)
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()
        lr_scheduler.step()
		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
        totalTrainLoss += totalLoss
        trainCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()

    print(totalTrainLoss/len(dataset))
    print(trainCorrect/len(dataset))


print("That's it!")