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
import subprocess

parser = argparse.ArgumentParser(description='Speaker Identification')

parser.add_argument('--cuda', type=int, default=1, metavar='N',
                    help='use cuda if possible (default: 1)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
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
parser.add_argument('--dataset', type=str, default='OMGEmotion')
parser.add_argument('--task', type=int, default=0, metavar='N',
                    help='0 for speaker identification and 1 for M/F classification')

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

# Speakers with more than 50 records 664ea0ccc, b018c03de, e3b57cfd8
speaker_filter_50 = ['16f978d4a', '179b7f51a', '3ecbd1402', '8c1814283', 'f8da01d25',
                  'c2b9c66f3', 'dce7e5d5f', 'e0fbb351e', 'f1f277e5d']
label_dict = {'16f978d4a':0, '179b7f51a':1, '3ecbd1402':2, '8c1814283':3, 'f8da01d25':4,
            'c2b9c66f3':5, 'dce7e5d5f':6, 'e0fbb351e':7, 'f1f277e5d':8}
# good
#speaker_filter_good = ['179b7f51a', '3ecbd1402','f8da01d25', 'dce7e5d5f', 'e0fbb351e', 'f1f277e5d']
#label_dict = {'179b7f51a':0, '3ecbd1402':1,'f8da01d25':2, 'dce7e5d5f':3, 'e0fbb351e':4, 'f1f277e5d':5}
M_F_dict = {'179b7f51a':'F', '3ecbd1402':'M','f8da01d25':'F', 'dce7e5d5f':'M', 'e0fbb351e':'F', 'f1f277e5d':'M', '119334435':'F', '16f978d4a':'M',
             '71091df7b':'F','8c1814283':'F', 'c2b9c66f3':'M', 'd096f5992':'F', 'e8a276493':'M'}

if args.task==1:
    print('The task you have chosen is gender recognition')
    label_dict = {'F':0, 'M':1}

train_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', person_filter=speaker_filter_50, preprocessed=True, split='train')
print("There are: {} Training samples.".format(len(train_dataset)))
test_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', person_filter=speaker_filter_50, preprocessed=True, split='test')
print("There are: {} testing samples.".format(len(test_dataset)))

test_sampler = sampler.RandomSampler(test_dataset)
train_sampler = sampler.RandomSampler(train_dataset)

kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    sampler=train_sampler, drop_last=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size,
    sampler=test_sampler, drop_last=False, **kwargs)
"""
training_speakers = []
testing_speakers = []
for batch_idx, (data, pers) in enumerate(train_loader):
    training_speakers.append(pers[1][0])
for batch_idx, (data, pers) in enumerate(test_loader):
    testing_speakers.append(pers[1][0])

training_speakers = set(training_speakers)
testing_speakers = set(testing_speakers)

training_counter = {speaker:0 for speaker in training_speakers}
testing_counter = {speaker:0 for speaker in testing_speakers}

dir = "/home/hussam/two_aux/OMG_preprocessed"
for file in os.listdir(dir+"/train"):
    speaker = file.split('_')[0]
    #if speaker in speaker_filter_good:
    #    os.system('cp /home/hussam/voice-emotion-app/two_aux/OMG_preprocessed/train/'+file+' /home/hussam/voice-emotion-app/two_aux/OMG_preprocessed/sorted/'+str(label_dict[speaker]))
    if speaker in training_counter:
        training_counter[speaker] += 1
for key in training_counter:
    if training_counter[key]>0:
        print({key, training_counter[key]})
print("========================================")
for file in os.listdir(dir+"/test"):
    speaker = file.split('_')[0]
    #if speaker in speaker_filter_good:
    #    os.system('cp /home/hussam/voice-emotion-app/two_aux/OMG_preprocessed/test/'+file+' /home/hussam/voice-emotion-app/two_aux/OMG_preprocessed/sorted/'+str(label_dict[speaker]))
    if speaker in testing_counter:
        testing_counter[speaker] += 1
for key in testing_counter:
    if testing_counter[key]>0:
        print({key, testing_counter[key]})
"""
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
            label = [label_dict[M_F_dict[p]] for p in pers[1]]
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
            global_sample = torch.mean(outs.encoder_out.global_sample, dim=1)

        if args.model_type == 'vae_l':
            global_sample = torch.mean(outs.encoder_out.local_sample, dim=1)

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
            label = [label_dict[M_F_dict[p]] for p in pers[1]]
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
            global_sample = torch.mean(outs.encoder_out.global_sample, dim=1)

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