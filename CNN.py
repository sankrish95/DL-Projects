import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#FCC network
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,10)
        self.fc2 = nn.Linear(10,num_classes)

    def forward(self,x): #input size =64*784
        x = F.relu(self.fc1(x)) # 64*50
        x = self.fc2(x) #64*10
        return x

#CNN model

class CNN(nn.module):
    def __init__(self,in_channels=1,out_class=10):
        super(CNN,self).__init__()
        self.Con1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3))

#Set device
#device = 'cuda' if torch.cuda.is_available() else 'cpu'


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

#Hyperparameters

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10
