import io
import os
import sys
import collections
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import socket
import argparse
import pickle
import numpy as np
import time
import math

import matplotlib
matplotlib.use('agg')
from sklearn.decomposition import PCA
from sklearn import cluster

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, sampler, distributed

import modules as custom_nn

from torchaudio import datasets, transforms, save

import torchvision
from torchvision import datasets, models
from torchvision import transforms as img_transforms
from torchvision.utils import save_image

import vctk_custom_dataset
import librispeech_custom_dataset
from torchvision.utils import save_image

import pickle
import ujson

import matplotlib.pyplot as plt
import pylab
import torch.distributions as distribution

import audio_utils as prepro
from audio_utils import griffinlim
from audio_utils import to_audio
import librosa
import librosa.display

import modules as custom_nn
import vae_g_l
import vae_l

from trainer import VAETrainer

from tensorboardX import SummaryWriter

import OMG_dataset

import PIL.Image
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser(description='Speaker Identification')

parser.add_argument('--cuda', type=int, default=1, metavar='N',
                    help='use cuda if possible (default: 1)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='N',
                    help='learning rate (default: 0.01)')
parser.add_argument('--num-epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--model-type', type=str, default='vae_g_l', metavar='S',
                    help='model type; options: vae_g_l, vae_l (default: vae_g_l)')
parser.add_argument('--model-name', type=str, default='speaker_id', metavar='S',
                    help='model name (for saving) (default: speaker_id_model)')
parser.add_argument('--pretrained-model', type=str, default='vae_g_l', metavar='S',
                    help='pretrained vae (default: recurrent_vae)')
parser.add_argument('--checkpoint-interval', type=int, default=1, metavar='N',
                    help='Interval between epochs to print loss and save model (default: 1)')
parser.add_argument('--mode', type=str, default='train', metavar='S',
                    help='(operation mode default: train; to test: testing)')
parser.add_argument('--resume', type=int, default=0, metavar='N',
                    help='continue training default: 0; to continue: 1)')
parser.add_argument('--beta', type=float, default=1., metavar='N',
                    help='(beta weight on KLD, default: 1. (no regularisation))')
parser.add_argument('--frame-dropout', type=float, default=0., metavar='N',
                    help='(audio frame dropout for decoder, default: 0. (no dropout))')
parser.add_argument('--decoder-dropout', type=float, default=0.0, metavar='N',
                    help='(general dropout for decoder, default: 0.5')
parser.add_argument('--anneal-function', type=str, default='logistic', metavar='S',
                    help='(anneal function (logistic or linear) default: logistic')
parser.add_argument('--z-size', type=int, default=512, metavar='N',
                    help='(latent feature depth, default: 256')
parser.add_argument('--type', type=str, default='emotion', metavar='N', help='Determine if you want to train the model on the person_id or on the emotion. values={emotion, person_id}')
parser.add_argument('--task', type=int, default=0, metavar='N',
                    help='0 for speaker identification, 1 for gender classification')

args = parser.parse_args()

if args.mode == "train":
    writer = SummaryWriter(comment=args.model_name)

args.hidden_size = args.z_size
args.local_z_size = args.z_size
if torch.cuda.is_available():
    args.use_cuda = True
else:
    args.use_cuda = False

if args.cuda and torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

print ("Using CUDA: {}".format(use_cuda))

train_dataset = librispeech_custom_dataset.LibriSpeech('../datasets/LibriSpeech/', preprocessed=True, split='train')
print("There are: {} Training samples.".format(len(train_dataset)))
test_dataset = librispeech_custom_dataset.LibriSpeech('../datasets/LibriSpeech/', preprocessed=True, split='test')
print("There are: {} testing samples.".format(len(test_dataset)))

if args.task == 0:
    if not os.path.isfile('librispeech_splits/speaker_labels.json'):
        print("preparing speaker labels")
        persons = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i][1][1]
            persons.append(int(sample))
        persons = sorted(set(persons))
        label_dict = {p : int(l) for p,l in zip(persons, range(len(persons)))}
        ujson.dump(label_dict,open("librispeech_splits/speaker_labels.json", 'w'))
    else:
        print("loading speaker labels")
        label_dict = ujson.load(open("librispeech_splits/speaker_labels.json", 'r'))
else:
    print("preparing speaker labels on M/F")
    label_dict = {'M':0, 'F':1}

test_sampler = sampler.RandomSampler(test_dataset)
train_sampler = sampler.RandomSampler(train_dataset)

kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    sampler = train_sampler, drop_last=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size,
    sampler = test_sampler, drop_last=False, **kwargs)

class SPEAKER_CLASSIFIER(nn.Module):
    def __init__(self):
        super(SPEAKER_CLASSIFIER, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.z_size, len(label_dict)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        pred = self.fc(input)

        return pred

def loss_function(pred, label):
    NLL = nn.CrossEntropyLoss()
    return NLL(pred, label)

def train(model, vae_model, optimizer):
    model.train()
    epoch_loss = 0
    tp = 0
    sample_amount = 0

    for batch_idx, (data, pers) in enumerate(train_loader):
        optimizer.zero_grad()
        
        if args.task==0:
            label = [label_dict[p] for p in pers[1]]
            label = torch.tensor(label)
        else:
            label = [label_dict[p] for p in pers[0]]
            label = torch.tensor(label)

        #data = torch.clamp(torch.div(data,(torch.min(data, dim=2, keepdim=True)[0]).repeat(1,1,data.size(2))), min=0, max=1)
        data = data / (torch.min(data))
        data = Variable(data)        
        
        data = data.transpose(1,2)
        
        if use_cuda:
            data = data.cuda()
            label = label.cuda()
        
        outs  = vae_model(data)

        if args.model_type == 'vae_g_l':
            global_sample = torch.mean(outs.encoder_out.local_sample, dim=1)

        if args.model_type == 'vae_l':
            global_sample = torch.mean(outs.encoder_out.local_sample, dim=1)

        #global_sample = pca.transform(global_sample.cpu())
        global_sample = torch.tensor(global_sample)
        global_sample = global_sample.cuda().type(torch.cuda.FloatTensor)

        pred = model(global_sample)
        max, indices = torch.max(pred, dim=1)

        tp += (indices == label).float().sum()
        sample_amount += len(label)

        loss = loss_function(pred, label)
        loss.backward()
        epoch_loss += loss.item()
        
        optimizer.step()
    acc = tp / sample_amount
    return epoch_loss/batch_idx, acc

def test(model, vae_model):
    model.eval()
    test_loss = 0
    tp = 0
    sample_amount = 0

    for batch_idx, (data, pers) in enumerate(test_loader):
        
        if args.task==0:
            label = [label_dict[p] for p in pers[1]]
            label = torch.tensor(label)
        else:
            label = [label_dict[p] for p in pers[0]]
            label = torch.tensor(label)
        
        #data = torch.clamp(torch.div(data,(torch.min(data, dim=2, keepdim=True)[0]).repeat(1,1,data.size(2))), min=0, max=1)
        data = data / (torch.min(data))
        data = Variable(data)        
        
        data = data.transpose(1,2)
        
        if use_cuda:
            data = data.cuda()
            label = label.cuda()        
        
        outs  = vae_model(data)        
        if args.model_type == 'vae_g_l':
            global_sample = torch.mean(outs.encoder_out.local_sample, dim=1)

        if args.model_type == 'vae_l':
            global_sample = torch.mean(outs.encoder_out.local_sample, dim=1)        

        #global_sample = pca.transform(global_sample.cpu())
        global_sample = torch.tensor(global_sample)
        global_sample = global_sample.cuda().type(torch.cuda.FloatTensor)

        pred = model(global_sample)

        max, indices = torch.max(pred, dim=1)

        tp += (indices == label).float().sum()
        sample_amount += len(label)
        
        loss = loss_function(pred, label)
        test_loss += loss.item()
    
    acc = tp/sample_amount
    
    return test_loss/batch_idx, acc

def train_epochs(model, vae_model, optimizer):
    
    last_loss = np.Inf
    for epoch in range(args.num_epochs):
        avg_train_loss, acc_train = train(model, vae_model, optimizer)
        if epoch % args.checkpoint_interval == 0:
            print('====> Epoch: {} Average train loss: {:.4f}'.format(
                epoch, avg_train_loss))
            print('====> Epoch: {} training ACC: {:.4f}'.format(
                epoch, acc_train))
            
            avg_test_loss, acc = test(model, vae_model)
            print('====> Epoch: {} Average test loss: {:.4f}'.format(
                epoch, avg_test_loss))
            print('====> Epoch: {} test ACC: {:.4f}'.format(
                epoch, acc))

            writer.add_scalar('data/train_loss', avg_train_loss, epoch)            
            writer.add_scalar('data/test_loss', avg_test_loss, epoch)
            
            print('--------------------')
            
            torch.save(model.state_dict(), 'experiments/'+args.model_name)

if __name__ == '__main__' and args.mode=='train':
    
    model = SPEAKER_CLASSIFIER()
    if args.model_type == 'vae_g_l':
        vae_model = vae_g_l.VAE(args)
        if args.use_cuda:
            vae_model.load_state_dict(torch.load('experiments/'+args.pretrained_model))
        else:
            vae_model.load_state_dict(torch.load('experiments/'+args.pretrained_model, map_location=lambda storage, loc: storage))
    elif args.model_type == 'vae_l':
        vae_model = vae_l.VAE(args)
        if args.use_cuda:
            vae_model.load_state_dict(torch.load('experiments/'+args.pretrained_model))
        else:
            vae_model.load_state_dict(torch.load('experiments/'+args.pretrained_model, map_location=lambda storage, loc: storage))
    
    vae_model.eval()
    
    for param in vae_model.parameters():
        param.requires_grad = False
    
    if args.resume == 1:
        model.load_state_dict(torch.load('experiments/'+args.model_name, map_location=lambda storage, loc: storage))
        print ("loaded model")
    
    if use_cuda:
        model.cuda()
        vae_model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_epochs(model, vae_model, optimizer)
#"""