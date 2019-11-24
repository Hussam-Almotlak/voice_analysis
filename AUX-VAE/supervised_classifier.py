import sys
import collections
import ujson
import os
import argparse
import OMG_dataset
from torchaudio import datasets, transforms, save
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pylab
import librosa
import librosa.display
import torch.distributed as dist
import modules as custom_nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler, distributed
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ['TORCH_MODEL_ZOO'] = '$WORK/.torch/models/'

parser = argparse.ArgumentParser(description='VAE Speech')

parser.add_argument('--cuda', type=int, default=1, metavar='N',
                    help='use cuda if possible (default: True)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--learning-rate', type=float, default=0.0008, metavar='N',
                    help='learning rate (default: 0.001)')
parser.add_argument('--num-epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 60)')
parser.add_argument('--model-type', type=str, default='vae_g_l', metavar='S',
                    help='model type; options: vae_g_l, vae_l (default: vae_g_l)')
parser.add_argument('--model-name', type=str, default=None, metavar='S',
                    help='model name (for saving) (default: model type)')
parser.add_argument('--checkpoint-interval', type=int, default=1, metavar='N',
                    help='Interval between epochs to print loss and save model (default: 1)')
parser.add_argument('--mode', type=str, default='train', metavar='S',
                    help='(operation mode default: train; to test: test, to analyse latent (for tsne): analyse-latent')
parser.add_argument('--resume', type=int, default=0, metavar='N',
                    help='continue training default: 0; to continue: 1)')
parser.add_argument('--debug-mode', type=str, default=0, metavar='N',
                    help='(debug mode (print dimensions) default: 0; to debug: 1)')
parser.add_argument('--beta', type=float, default=1., metavar='N',
                    help='(beta weight on KLD, default: 1. (no regularisation))')
parser.add_argument('--frame-dropout', type=float, default=0., metavar='N',
                    help='(audio frame dropout for decoder, default: 0. (no dropout))')
parser.add_argument('--decoder-dropout', type=float, default=0.0, metavar='N',
                    help='(general dropout for decoder, default: 0.5')
parser.add_argument('--anneal-function', type=str, default='logistic', metavar='S',
                    help='(anneal function (logistic or linear) default: logistic')
parser.add_argument('--k', type=float, default=0.0025, metavar='N',
                    help='(anneal function hyperparameter default: 0.0025')
parser.add_argument('--x0', type=int, default=2500, metavar='N',
                    help='(anneal function hyperparameter default: 2500')
parser.add_argument('--dataset', type=str, default='OMGEmotion', metavar='S',
                    help='(dataset used for training; LibriSpeech, VCTK; default: VCTK')
parser.add_argument('--beta-mi', type=float, default=1., metavar='N',
                    help='(beta weight for MI, default: 1.')
parser.add_argument('--beta-kl', type=float, default=1., metavar='N',
                    help='(beta weight for KL, default: 1.')
parser.add_argument('--z-size', type=int, default=256, metavar='N',
                    help='(latent feature depth, default: 256')
parser.add_argument('--predictive', type=int, default=1, metavar='N',
                    help='(predictive coding, if false reconstruct, default: 1')

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

class Trainer:
    def __init__(self, args, model, optimizer, train_loader, test_loader, test_dataset, writer):
        self.writer = writer
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.loss_function = args.loss_function




def load_dataset(dataset='VCTK', train_subset=1.0, person_filter=None):
    train_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='train',
                                               person_filter=person_filter, filter_mode='include')
    test_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='test',
                                              person_filter=person_filter, filter_mode='include')

    indices = list(range(len(train_dataset)))
    split = int(np.floor(len(train_dataset) * train_subset))

    train_sampler = sampler.RandomSampler(sampler.SubsetRandomSampler(indices[:split]))
    test_sampler = sampler.RandomSampler(test_dataset)
    # validate_sampler = sampler.RandomSampler(validate_dataset)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=test_sampler, drop_last=False, **kwargs)

    return train_loader, test_loader, train_dataset, test_dataset

def train(args):
    train_loader, test_loader, train_dataset, test_dataset = load_dataset(dataset=args.dataset, train_subset=1.0)
