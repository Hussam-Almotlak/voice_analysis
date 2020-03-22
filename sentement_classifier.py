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
# import vae_g_l_exp

from trainer import VAETrainer

from tensorboardX import SummaryWriter

import OMG_dataset

import PIL.Image
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser(description='Speaker Identification')

parser.add_argument('--cuda', type=int, default=1, metavar='N',
                    help='use cuda if possible (default: 1)')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--learning-rate', type=float, default=0.01, metavar='N',
                    help='learning rate (default: 0.01)')
parser.add_argument('--num-epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--model-name', type=str, default='sentiment_classifier', metavar='S',
                    help='model name (for saving) (default: speaker_id_model)')
parser.add_argument('--pretrained-model', type=str, default='sentiment_classifier', metavar='S',
                    help='pretrained vae (default: recurrent_vae)')
parser.add_argument('--checkpoint-interval', type=int, default=1, metavar='N',
                    help='Interval between epochs to print loss and save model (default: 1)')
parser.add_argument('--mode', type=str, default='train', metavar='S',
                    help='(operation mode default: train; to test: testing)')
parser.add_argument('--resume', type=int, default=0, metavar='N',
                    help='continue training default: 0; to continue: 1)')
parser.add_argument('--rnn-size', type=int, default=2048, metavar='N',
                    help='(latent feature depth, default: 256')
parser.add_argument('--type', type=str, default='emotion', metavar='N',
                    help='Determine if you want to train the model on the person_id or on the emotion. values={emotion, person_id}')
parser.add_argument('--dataset', type=str, default='OMGEmotion')
parser.add_argument('--load_model', type=str, default='mlstm/mlstm_e6_0.00.pt')

args = parser.parse_args()

if args.mode == "train":
    writer = SummaryWriter(comment=args.model_name)

if torch.cuda.is_available():
    args.use_cuda = True
else:
    args.use_cuda = False

if args.cuda and torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

print("Using CUDA: {}".format(use_cuda))

label_dict = {}
if args.dataset == 'LibriSpeech':
    train_dataset = librispeech_custom_dataset.LibriSpeech('../datasets/LibriSpeech/', preprocessed=True, split='train')
    print("There are: {} Training samples.".format(len(train_dataset)))

    test_dataset = librispeech_custom_dataset.LibriSpeech('../datasets/LibriSpeech/', preprocessed=True, split='test')
    print("There are: {} Testing samples.".format(len(test_dataset)))

    if not os.path.isfile('librispeech_splits/speaker_labels.json'):
        print("preparing speaker labels")
        persons = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i][1][1]
            persons.append(int(sample))
        persons = sorted(set(persons))
        label_dict = {p: int(l) for p, l in zip(persons, range(len(persons)))}
        ujson.dump(label_dict, open("librispeech_splits/speaker_labels.json", 'w'))
    else:
        print("loading speaker labels")
        label_dict = ujson.load(open("librispeech_splits/speaker_labels.json", 'r'))

elif args.dataset == 'OMGEmotion':
    train_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', emotion_filter= ['happiness', 'sadness'], preprocessed=True, split='train')
    test_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', emotion_filter= ['happiness', 'sadness'], preprocessed=True, split='test')
    train_samples = len(train_dataset)
    print("There are: {} Training samples.".format(train_samples))
    test_samples = len(test_dataset)
    print("There are: {} Testing samples.".format(test_samples))

    #label_dict = {'0': 0, '1':1, '2': 2, '3':3, '4': 4, '5':5, '6':6}
    label_dict = {'3': 0, '5':1}
test_sampler = sampler.RandomSampler(test_dataset)
train_sampler = sampler.RandomSampler(train_dataset)

kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    sampler=train_sampler, drop_last=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size,
    sampler=test_sampler, drop_last=False, **kwargs)

input_size = 2048 # should be rnn_size
class SENTIMENT_CLASSIFIER(nn.Module):
    def __init__(self):
        super(SENTIMENT_CLASSIFIER, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, len(label_dict)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        pred = self.fc(input)
        return pred

def make_cuda(state):
    if isinstance(state, tuple):
    	return (state[0].cuda(), state[1].cuda())
    else:
    	return state.cuda()

def loss_function(pred, label):
    NLL = nn.CrossEntropyLoss()

    return NLL(pred, label)

def index_of_max(arr):
    max = arr[0]
    index = 0
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
            index = i
    return index

def train(model, embed, rnn, optimizer):
    model.train()
    epoch_loss = 0
    tp = 0
    sample_amount = 0
    counter = -1

    training_label = 0
    if args.type == 'person':
        training_label = 1

    for batch_idx, (data, pers) in enumerate(train_loader):
        counter += 1
        optimizer.zero_grad()

        label_orig = [label_dict[p] for p in pers[training_label]]
        label = torch.tensor(label_orig)

        data = torch.clamp(torch.div(data, (torch.min(data, dim=2, keepdim=True)[0]).repeat(1, 1, data.size(2))), min=0,
                           max=1)
        #print("The dimensions of the data are: {}".format(data.shape))
        data = data.transpose(1,2)
        data = Variable(data)

        if use_cuda:
            data = data.cuda()
            label = label.cuda()

        states = rnn.state0(args.batch_size)
        if isinstance(states, tuple):
            hidden, cell = states
        else:
            hidden = states
        if use_cuda:
            states = make_cuda(states)
        data = data.long()
        loss_avg = 0
        for s in range(args.batch_size):
            batch = Variable(data[s])
            emotions = [0 for i in range(len(label_dict))]
            loss = 0
            if use_cuda:
                batch = batch.cuda()
            for t in range(TIMESTEPS):
                emb = embed(batch[t])
                states, output = rnn(emb, states)
                if isinstance(states, tuple):
                    hidden, cell = states
                else:
                    hidden = states
                out = hidden.data
                out = torch.mean(out, dim=1)
                sample = torch.tensor(out)
                sample = sample.cuda().type(torch.cuda.FloatTensor)

                pred = model(sample)
                loss += loss_function(pred, label)
                max, indices = torch.max(pred, dim=1)
                # only for batch size = 1
                emotions[indices[0]] += 1

            loss_avg = loss_avg + loss / TIMESTEPS

        index = index_of_max(emotions)
        indices = torch.tensor([index]).cuda()
        tp += (indices == label).float().sum()
        sample_amount += len(label)

        loss_avg = loss_avg / args.batch_size
        loss_avg.backward()
        epoch_loss += loss_avg
        optimizer.step()

    print("The amount of taken samples is: {}".format(sample_amount))
    acc = tp / sample_amount

    return epoch_loss / batch_idx, acc


def test(model, embed, rnn):
    model.eval()
    test_loss = 0
    tp = 0
    sample_amount = 0
    counter = 0
    training_label = 0
    if args.type == 'person':
        training_label = 1

    for batch_idx, (data, pers) in enumerate(test_loader):
        label = [label_dict[p] for p in pers[training_label]]
        label = torch.tensor(label)

        data = torch.clamp(torch.div(data, (torch.min(data, dim=2, keepdim=True)[0]).repeat(1, 1, data.size(2))), min=0,
                           max=1)

        data = Variable(data)


        if use_cuda:
            data = data.cuda()
            label = label.cuda()
        states = rnn.state0(args.batch_size)
        if isinstance(states, tuple):
            hidden, cell = states
        else:
            hidden = states
        if use_cuda:
            states = make_cuda(states)
        data = data.long()
        for s in range(args.batch_size):
            batch = Variable(data[s])
            emotions = [0 for i in range(len(label_dict))]
            loss = 0
            if use_cuda:
                batch = batch.cuda()
            for t in range(TIMESTEPS):
                emb = embed(batch[t])
                states, output = rnn(emb, states)
                if isinstance(states, tuple):
                    hidden, cell = states
                else:
                    hidden = states
                out = hidden.data
                out = torch.mean(out, dim=1)
                sample = torch.tensor(out)
                sample = sample.cuda().type(torch.cuda.FloatTensor)

                pred = model(sample)
                max, indices = torch.max(pred, dim=1)
                # only for batch size = 1
                emotions[indices[0]] += 1
        index = index_of_max(emotions)
        indices = torch.tensor([index]).cuda()

        tp += (indices == label).float().sum()
        sample_amount += len(label)

        loss = loss_function(pred, label)
        test_loss += loss.item()
    print("The amount of taken samples is: {}".format(sample_amount))
    acc = tp / sample_amount

    return test_loss / batch_idx, acc


def train_epochs(model, embed, rnn, optimizer):
    for epoch in range(args.num_epochs):
        avg_train_loss, acc_train = train(model, embed, rnn, optimizer)
        if epoch % args.checkpoint_interval == 0:
            print('====> Epoch: {} Average train loss: {:.4f}'.format(
                epoch, avg_train_loss))
            print('====> Epoch: {} train ACC: {:.4f}'.format(
                epoch, acc_train))

            avg_test_loss, acc = test(model, embed, rnn)
            print('====> Epoch: {} Average test loss: {:.4f}'.format(
                epoch, avg_test_loss))
            print('====> Epoch: {} test ACC: {:.4f}'.format(
                epoch, acc))

            writer.add_scalar('data/train_loss', avg_train_loss, epoch)
            writer.add_scalar('data/test_loss', avg_test_loss, epoch)

            print('--------------------')

            torch.save(model.state_dict(), 'experiments/' + args.model_name)


if __name__ == '__main__' and args.mode == 'train':

    model = SENTIMENT_CLASSIFIER()
    checkpoint = torch.load(args.load_model)
    embed = checkpoint['embed']
    rnn = checkpoint['rnn']

    input_dim = args.batch_size
    TIMESTEPS = 140

    if args.resume == 1:
        model.load_state_dict(torch.load('experiments/' + args.model_name, map_location=lambda storage, loc: storage))
        print("loaded model")

    if use_cuda:
        model.cuda()
        embed.cuda()
        rnn.cuda()


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_epochs(model, embed, rnn, optimizer)
