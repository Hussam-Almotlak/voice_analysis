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

import matplotlib.pyplot as plt

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
from scipy.stats import pearsonr

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
parser.add_argument('--load_model', type=str, default='mlstm/mlstm_e15_0.00.pt')
parser.add_argument('--a-or-v', type=int, default=1, metavar='N',
                    help='0 for arousal, 1 for valence')

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

train_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='train')
test_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='test')

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
class Emotion_Regression(nn.Module):
    def __init__(self):
        super(Emotion_Regression, self).__init__()

        self.hidden_1 = nn.Linear(args.rnn_size, 1024)
        self.hidden_2 = nn.Linear(1024, 1)


    def forward(self, input):

        pred = F.relu(self.hidden_1(input))
        #pred = F.dropout(pred, p=0.2, training=True)
        pred = self.hidden_2(pred)
        return pred

def make_cuda(state):
    if isinstance(state, tuple):
    	return (state[0].cuda(), state[1].cuda())
    else:
    	return state.cuda()
def index_of_max(arr):
    max = arr[0]
    index = 0
    for i in range(len(arr)):
        if arr[i] > max:
            max = arr[i]
            index = i
    return index        

def loss_function(pred, label):

    pred = pred[:, 0].double()
    label = label[:, args.a_or_v].double()
    diff = label - pred
    l1 = torch.sum(diff*diff) / diff.numel()

    return l1

def ccc(y_pred, y_true):
    true_mean = np.mean(y_true)
    true_variance = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_variance = np.var(y_pred)

    rho,_ = pearsonr(y_pred,y_true)

    std_predictions = np.std(y_pred)

    std_gt = np.std(y_true)


    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)

    return ccc, rho

def calculate_ccc(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    
    ccc, rho_v = ccc(pred, label)
    return ccc

def train(model, embed, rnn, optimizer):
    model.train()
    epoch_loss = 0
    epoch_arousal = 0
    epoch_valence = 0
    mean_arousal_ccc = 0
    mean_valence_ccc = 0

    training_label = 0
    if args.type == 'person':
        training_label = 1
    labels = []
    preds = []
    for batch_idx, (data, pers) in enumerate(train_loader):
        optimizer.zero_grad()
        label = np.delete(pers, np.s_[0:2], 0)
        label = label.astype(np.float64)
        label = label.transpose()
        label = torch.tensor(label).double()

        data = torch.clamp(torch.div(data, (torch.min(data, dim=2, keepdim=True)[0]).repeat(1, 1, data.size(2))), min=0, max=1)
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
        #arousal_avg = 0

        for s in range(args.batch_size):
            batch = Variable(data[s])
            loss = 0
            #arousal_ccc = 0
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
                if t==0:
                    ccc_pred = pred
                else:
                    ccc_pred += pred
            ccc_pred /= TIMESTEPS
            labels.append(label[0][args.a_or_v].item())
            # needs to be changed
            preds.append(ccc_pred.item())
            loss_avg = loss_avg + loss / TIMESTEPS

        loss_avg = loss_avg / args.batch_size
        loss_avg.backward()
        epoch_loss += loss_avg
        optimizer.step()
    arousal_ccc = calculate_ccc(preds, labels)
    return epoch_loss / batch_idx, arousal_ccc

def test(model, embed, rnn, optimizer):
    model.eval()
    epoch_loss = 0
    epoch_arousal = 0
    epoch_valence = 0
    mean_arousal_ccc = 0
    mean_valence_ccc = 0

    training_label = 0
    if args.type == 'person':
        training_label = 1
    labels = []
    preds = []
    for batch_idx, (data, pers) in enumerate(train_loader):
        optimizer.zero_grad()
        label = np.delete(pers, np.s_[0:2], 0)
        label = label.astype(np.float64)
        label = label.transpose()
        label = torch.tensor(label).double()

        data = torch.clamp(torch.div(data, (torch.min(data, dim=2, keepdim=True)[0]).repeat(1, 1, data.size(2))), min=0, max=1)
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
        #arousal_avg = 0

        for s in range(args.batch_size):
            batch = Variable(data[s])
            loss = 0
            #arousal_ccc = 0
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
                if t==0:
                    ccc_pred = pred
                else:
                    ccc_pred += pred
            ccc_pred /= TIMESTEPS
                
            labels.append(label[0][args.a_or_v].item())
            # needs to be changed
            preds.append(ccc_pred.item())
            loss_avg = loss_avg + loss / TIMESTEPS

        loss_avg = loss_avg / args.batch_size
        #loss_avg.backward()
        epoch_loss += loss_avg
        optimizer.step()

    arousal_ccc = calculate_ccc(preds, labels)
    return epoch_loss / batch_idx, arousal_ccc

def train_epochs(model, embed, rnn, optimizer):
    last_loss = np.Inf

    train_losses = []
    test_losses = []
    train_arousals = []
    test_arousals = []
    train_valences = []
    test_valences =[]

    for epoch in range(args.num_epochs):
        avg_train_loss, mean_arousal_ccc = train(model, embed, rnn, optimizer)
        train_losses.append(avg_train_loss)
        if epoch % args.checkpoint_interval == 0:
            print('====> Epoch: {} Average train loss: {:.5f}'.format(epoch, avg_train_loss))
            print('====> Epoch: {} Average train arousal ccc: {:.5f}'.format(epoch, mean_arousal_ccc))
            print('-------------------------------------------------')
            avg_test_loss, mean_arousal_ccc = test(model, embed, rnn, optimizer)
            test_losses.append(avg_test_loss)
            print('====> Epoch: {} Average test loss: {:.5f}'.format(epoch, avg_test_loss))
            print('====> Epoch: {} Average test arousal ccc: {:.5f}'.format(epoch, mean_arousal_ccc))
            writer.add_scalar('experiments/regression_train_loss', avg_train_loss, epoch)
            writer.add_scalar('experiments/regression_test_loss', avg_test_loss, epoch)

            print('=================================================')

            plt.title("arousal, valence and overall losses")
            plt.plot(train_losses, label="overall training")
            plt.plot(test_losses, label="overall testing")
            plt.legend(loc='upper right')
            plt.savefig("experiments/regression_losses.png")
            plt.clf()

            torch.save(model.state_dict(), 'experiments/' + args.model_name)

if __name__ == '__main__' and args.mode == 'train':

    model = Emotion_Regression()
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

