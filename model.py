import torch

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


    def forward(self, x):
        out = self.ReLU1(self.conv1(x))
        out = self.ReLU2(self.conv2(out))
        out = self.flatten(out)
        out = self.linear1(out)
        labels = self.labelout(out)
        bbox = self.bboxout(out)
        return labels, bbox