import data
import model
import sys
import torch
import torch.optim as optim

#function for batching dataloader
def collate_fn(batch):
    return tuple(zip(*batch))

#set training parameters
epochs = 100
lr = 0.001
save_path = "saved_model.pt"

#load model
network = model.resNet()

#get training and validation datasets
training_data = data.labelled_data("labeled_data", "training", data.get_transform(train=True))
validation_data = data.labelled_data("labeled_data", "validation", data.get_transform(train=True))

#get training and validation dataloaders
training_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=8, shuffle=True, collate_fn=collate_fn)

label_criterion = torch.nn.CrossEntropyLoss()
bbox_criterion = torch.nn.MSELoss()

optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = torch.stack(inputs)
        inputs.to(device)
        t_labels = torch.stack([l['labels'][0] for l in labels])
        t_bboxes = torch.stack([l['bboxes_norm'][0] for l in labels])
        t_labels.to(device)
        t_bboxes.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        p_labels, p_bboxes = network(inputs)

        loss = sum([label_criterion(p_labels, t_labels), bbox_criterion(p_bboxes, t_bboxes)])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

torch.save(network.state_dict(), save_path)
   

# print(training_data[int(sys.argv[1])])
# print(training_data[int(sys.argv[1])][0].shape)

