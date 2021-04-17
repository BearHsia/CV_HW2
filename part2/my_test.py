import os, sys
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model import ConvNet, MyNet
from data import TestDataset
import matplotlib.pyplot as plt

model = ConvNet()
print(model)

input_ = torch.randn(1, 1, 28, 28)
out_ = model(input_)
print(out_)


train_loss_list = [0.5591182804925368, 0.08189188422048464, 0.05800852186058182, 0.041594843079382555, 0.033893174811964855]
train_accu_list = [0.812675, 0.974025, 0.981175, 0.986725, 0.9889]
valid_loss_list = [0.11438984602478484, 0.08691138117302644, 0.05827350800308531, 0.05078429139296215, 0.05331853696916136]
valid_accu_list = [0.9648, 0.9745, 0.9824, 0.9859, 0.9848]
model_type = 'conv'
x_tick = list(range(5))
fig = plt.figure()
plt.plot(x_tick, train_loss_list, '-')
plt.title("Loss_"+model_type)
plt.xlabel("epoches")
plt.ylabel("Loss")
plt.legend()


fig = plt.figure()
plt.plot(x_tick, train_accu_list, '-')
plt.legend()


fig = plt.figure()
plt.plot(x_tick, valid_loss_list, '-')
plt.legend()


fig = plt.figure()
plt.plot(x_tick, valid_accu_list, '-')
plt.legend()
plt.show()