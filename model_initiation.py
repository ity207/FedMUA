# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.functional as F


def mnist_model_init(data_name):

    model = Net_mnist()

    return model


class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 10, 5, 1)
        self.fc1 = nn.Linear(2*2*10, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 4, 4)
        x = x.view(-1, 2*2*10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward_loss(self,x,y,w,criterion=nn.CrossEntropyLoss()):
        p = [w[:500].view(20,1,5,5),w[500:520],w[520:5520].view(10,20,5,5),w[5520:5530],
             w[5530:7530].view(50,40),w[7530:7580],w[7580:8080].view(10,50),w[8080:]]
        x = F.relu(F.conv2d(x,p[0],p[1]))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(F.conv2d(x,p[2],p[3]))
        x = F.max_pool2d(x, 4, 4)
        x = x.view(-1, 2*2*10)
        x = F.relu(F.linear(x,p[4],p[5]))
        x = F.linear(x,p[6],p[7])
        loss = criterion(x,y)
        return loss