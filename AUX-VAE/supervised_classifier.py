import sys
import collections
import ujson

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.distributed as dist

import modules as custom_nn

import numpy as np

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(128*202*201, 7)
        self.dropout = nn.Dropout(p=0.3)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = input.unsqueeze(1)
        pred = F.dropout(F.relu(self.batch_norm1(self.conv1(input))), p=0.3, training=True)
        pred = F.dropout(F.relu(self.batch_norm2(self.conv2(pred))), p=0.3, training=True)
        pred = F.dropout(F.relu(self.batch_norm3(self.conv3(pred))), p=0.3, training=True)
        #pred = F.dropout(F.relu(self.conv4(pred)), p=0.3, training=True)
        pred = pred.view(-1, 128*202*201)
        pred = F.relu((self.fc(pred)))
        pred = self.softmax(pred)

        return pred