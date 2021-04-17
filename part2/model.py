import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def name(self):
        return "ConvNet"

class my_block(nn.Module):
    def __init__(self,ni):
        super(my_block, self).__init__()
        self.conv1 = nn.Conv2d(ni, ni, 1)
        self.bn1 = nn.BatchNorm2d(ni)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(ni, ni, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(ni)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(ni, ni, 1)
        self.bn3 = nn.BatchNorm2d(ni)
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out += residual
        out = self.relu4(out)

        return out

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.blk2 = my_block(10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=2, padding=1)

        self.blk3 = my_block(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=0)
        
        self.blk4 = my_block(24)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=2, padding=0)

        self.blk5 = my_block(32)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=40, kernel_size=3, stride=1, padding=0)

        self.blk6 = my_block(40)
        self.conv6 = nn.Conv2d(in_channels=40, out_channels=48, kernel_size=3, stride=1, padding=0)

        self.fc5 = nn.Linear(48, 24)
        self.relu5 = nn.ReLU(inplace=True)
        #self.fc6 = nn.Linear(80, 40)
        #self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Linear(24, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.blk2(x)
        x = self.conv2(x)
        x = self.blk3(x)
        x = self.conv3(x)
        x = self.blk4(x)
        x = self.conv4(x)
        x = self.blk5(x)
        x = self.conv5(x)
        x = self.blk6(x)
        x = self.conv6(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc5(x)
        x = self.relu5(x)
        #x = self.fc6(x)
        #x = self.relu6(x)
        x = self.fc7(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def name(self):
        return "MyNet"

