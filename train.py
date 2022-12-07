import data
import model
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
import os


parser = argparse.ArgumentParser(description='Load training parameters yml')
parser.add_argument('-p', '--params', help="parameter yml file for training model")
args = parser.parse_args()

#load parameters for training model
with open(args.params) as f:
    training_params = yaml.safe_load(f)

print("training_parameters:")
print(training_params)

#set training parameters
epochs = int(training_params["epochs"])
lr = float(training_params["lr"])
save_path = training_params["save_path"]
data_path = training_params["data_path"]
batch_size = training_params["batch_size"]


#function for batching dataloader
def collate_fn(batch):
    return tuple(zip(*batch))

#get training and validation datasets
training_data = data.labeled_data(data_path, "training", data.get_transform(train=True))
validation_data = data.labeled_data(data_path, "validation", data.get_transform(train=True))

#get training and validation dataloaders
training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# label_criterion = torch.nn.CrossEntropyLoss()
label_criterion = torch.nn.CrossEntropyLoss()
bbox_criterion = torch.nn.MSELoss()
score_criterion = torch.nn.MSELoss()

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#load model
network = model.VGG(device=device)
network = network.to(device)

#initialize optimizer
optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9)

#log for tensorboard
writer = SummaryWriter()

#set start epoch
start_epoch = 0

#load checkpoint if fle exists
if os.path.exists(save_path):

    checkpoint = torch.load(save_path)

    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']


for epoch in range(start_epoch, epochs):
    running_loss = 0.0
    label_loss = 0 
    bbox_loss = 0
    score_loss = 0
    
    for ind, (inputs,labels) in enumerate(training_loader, 0):
        
        inputs = [img.to(device) for img in inputs]
        t_labels = [l['labels'].to(device) for l in labels]
        t_bboxes = [l['bboxes'].to(device) for l in labels]

        # zero the parameter gradients
        optimizer.zero_grad()
        loss =0
        # forward + backward + optimize 
        p_out = network(inputs)
        for i,p_dict in enumerate(p_out):
            for j in range(model.num_boxes):
                num_boxes = len(t_labels[i])
                #calculate loss when ground truth bboxes are available
                if j < num_boxes:
                    label_loss_j = 0.2*label_criterion(p_dict["labels"][j],t_labels[i][j])
                    bbox_loss_j = 0.6*torchvision.ops.distance_box_iou_loss(p_dict["boxes"][j],t_bboxes[i][j])
                    score_loss_j = 0.2*score_criterion(p_dict["scores"][j],torch.tensor(1.0).to(device))
                    label_loss += label_loss_j
                    bbox_loss += bbox_loss_j
                    score_loss += score_loss_j
                    loss += sum([label_loss_j, bbox_loss_j, score_loss_j])
                #only calculate loss on score when there is not ground truth bbox
                else:
                    score_loss_j = score_criterion(p_dict["scores"][j],torch.tensor(0.0).to(device))
                    score_loss += score_loss_j
                    loss +=  score_loss_j


        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if ind % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {ind + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    #save model every 10 epochs
    if epoch % 10 == 0:
        torch.save({'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, save_path)
    
    #document loss of epoch
    writer.add_scalar("Loss/label", label_loss, epoch)
    writer.add_scalar("Loss/bboxes", bbox_loss, epoch)
    writer.add_scalar("Loss/score", score_loss, epoch)      
    writer.add_scalar("Loss/all", label_loss+bbox_loss+score_loss, epoch)    

#write image results to tensor board
network.train(False)

images, labels = next(iter(validation_loader))
inputs = [img.to(device) for img in images]
p_boxes = [dic["boxes"] for dic in network(inputs)]
p_labels = [dic["labels"] for dic in network(inputs)]

#make sure x_min < x_max and y_min < y_max
for i in range(len(p_boxes)):
    for j in range(len(p_boxes[i])):
        if p_boxes[i][j][0] > p_boxes[i][j][2]:
            p_boxes[i][j][2] = p_boxes[i][j][0] +1
        if p_boxes[i][j][1] > p_boxes[i][j][3]:
            p_boxes[i][j][3] = p_boxes[i][j][1] +1



truth_images = [torchvision.utils.draw_bounding_boxes(image, labels[i]["bboxes"], labels=[list(data.class_dict.keys())[int(l)] for l in labels[i]["labels_i"]] ) for i,image in enumerate(images)]
predict_images = [torchvision.utils.draw_bounding_boxes(image, p_boxes[i], labels=[list(data.class_dict.keys())[int(l)] for l in p_labels[i]] ) for i,image in enumerate(images)]

#display predicted images on tesorboard
for i in range(len(images)):
    writer.add_image("truth image: "  +str(i), truth_images[i])
    writer.add_image("predict image: "  +str(i), predict_images[i])

#close tensorboard writer
writer.flush()
writer.close()

#save model
torch.save({'epoch': epochs,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0}, save_path)
   