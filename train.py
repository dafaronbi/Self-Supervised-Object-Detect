import data
import model
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



#set training parameters
epochs = 300
lr = 0.001
save_path = "saved_model.pt"
data_path = "labeled_data"

#function for batching dataloader
def collate_fn(batch):
    return tuple(zip(*batch))

#get training and validation datasets
training_data = data.labeled_data(data_path, "training", data.get_transform(train=True))
validation_data = data.labeled_data(data_path, "validation", data.get_transform(train=True))

#get training and validation dataloaders
training_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=8, shuffle=True, collate_fn=collate_fn)

# label_criterion = torch.nn.CrossEntropyLoss()
label_criterion = torch.nn.MSELoss()
bbox_criterion = torch.nn.MSELoss()
score_criterion = torch.nn.MSELoss()

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#load model
network = model.resNet(device)
network = network.to(device)

#initialize optimizer
optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9)

#log for tensorboard
writer = SummaryWriter()

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = [img.to(device) for img in inputs]
        t_labels = [l['labels'].to(device) for l in labels]
        t_bboxes = [l['bboxes'].to(device) for l in labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize 
        p_out = network(inputs)
        label_loss = 0 
        bbox_loss = 0
        score_loss = 0
        loss =0
        for i,p_dict in enumerate(p_out):
            for j in range(model.num_boxes):
                num_boxes = len(t_labels[i])
                #calculate loss when ground truth bboxes are available
                if j < num_boxes:
                    label_loss_j = label_criterion(p_dict["labels"][j],t_labels[i][j])
                    bbox_loss_j = bbox_criterion(p_dict["boxes"][j],t_bboxes[i][j])
                    score_loss_j = score_criterion(p_dict["scores"][j],torch.tensor(1.0))
                    label_loss += label_loss_j
                    bbox_loss += bbox_loss_j
                    score_loss += score_loss_j
                    loss += sum([label_loss_j, bbox_loss_j, score_loss_j])
                #only calculate loss on score when there is not ground truth bbox
                else:
                    score_loss_j = score_criterion(p_dict["scores"][j],torch.tensor(0.0))
                    score_loss += score_loss_j
                    loss +=  score_loss_j

        writer.add_scalar("Loss/label", label_loss, epoch)
        writer.add_scalar("Loss/bboxes", bbox_loss, epoch)
        writer.add_scalar("Loss/score", score_loss, epoch)      
        writer.add_scalar("Loss/all", loss, epoch)
        print(loss)       
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

writer.flush()
writer.close()
torch.save(network.state_dict(), save_path)
   

# print(training_data[int(sys.argv[1])])
# print(training_data[int(sys.argv[1])][0].shape)

