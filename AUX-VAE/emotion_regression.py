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

#plt.use('agg')
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

#import matplotlib.pyplot as plt
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
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='N',
                    help='learning rate (default: 0.01)')
parser.add_argument('--num-epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--model-type', type=str, default='vae_g_l', metavar='S',
                    help='model type; options: vae_g_l, vae_l (default: vae_g_l)')
parser.add_argument('--model-name', type=str, default='emotion_regression', metavar='S',
                    help='model name (for saving) (default: speaker_id_model)')
parser.add_argument('--pretrained-model', type=str, default='vae_g_l', metavar='S',
                    help='pretrained vae (default: recurrent_vae)')
parser.add_argument('--checkpoint-interval', type=int, default=1, metavar='N',
                    help='Interval between epochs to print loss and save model (default: 1)')
parser.add_argument('--mode', type=str, default='train', metavar='S',
                    help='(operation mode default: train; to test: testing)')
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
parser.add_argument('--z-size', type=int, default=256, metavar='N',
                    help='(latent feature depth, default: 256')
parser.add_argument('--type', type=str, default='emotion', metavar='N',
                    help='Determine if you want to train the model on the person_id or on the emotion. values={emotion, person_id}')
parser.add_argument('--dataset', type=str, default='OMGEmotion')

args = parser.parse_args()

if args.mode == "train":
    writer = SummaryWriter(comment=args.model_name)

args.hidden_size = 256
# args.z_size = 256
args.local_z_size = 256
if torch.cuda.is_available():
    args.use_cuda = True
else:
    args.use_cuda = False

if args.cuda and torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

print("Using CUDA: {}".format(use_cuda))

#label_dict = {}

train_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='train')
test_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='test')

#label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}

test_sampler = sampler.RandomSampler(test_dataset)
train_sampler = sampler.RandomSampler(train_dataset)

kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    sampler=train_sampler, drop_last=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size,
    sampler=test_sampler, drop_last=False, **kwargs)


class Emotion_Regression(nn.Module):
    def __init__(self):
        super(Emotion_Regression, self).__init__()
        # """
        self.fc = nn.Sequential(
            nn.Linear(args.z_size, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 2)
            #nn.Linear(10, len(label_dict)),
        )

    def forward(self, input):

        pred = self.fc(input)

        if args.debug_mode: print(pred.size())

        return pred


def loss_function(pred, label):
    arousal_pred = pred[:, 0].double()
    arousal_label = label[:, 0].double()
    diff = arousal_label - arousal_pred
    l1 = torch.sum(diff*diff) / diff.numel()

    valence_pred = pred[:, 1].double()
    valence_label = label[:, 1].double()
    diff = valence_label - valence_pred
    l2 = torch.sum(diff*diff) / diff.numel()

    return (l1+l2)/2, l1, l2

"""
def fit_pca(model, vae_model):
    all_training = []
    for batch_idx, (data, pers) in enumerate(train_loader):
        data = torch.clamp(torch.div(data, (torch.min(data, dim=2, keepdim=True)[0]).repeat(1, 1, data.size(2))), min=0,
                           max=1)

        data = Variable(data)

        data = data.transpose(1, 2)

        if use_cuda:
            data = data.cuda()

        outs = vae_model(data)
        if args.model_type == 'vae_g_l':
            global_sample = torch.mean(outs.encoder_out.global_sample, dim=1)
            global_sample = np.array(global_sample.cpu())
            all_training.append(global_sample[0])

    print(all_training)
    pca = PCA(n_components=50, whiten=False)
    pca = pca.fit(all_training)
    #agglo = cluster.FeatureAgglomeration(n_clusters=10)
    #agglo = agglo.fit(all_training)
    torch.save(pca, "experiments/pca")

    return pca
"""

def train(model, vae_model, optimizer):
    model.train()
    epoch_loss = 0
    epoch_arousal = 0
    epoch_valence = 0

    training_label = 0
    if args.type == 'person':
        training_label = 1
    print('Training label = ' + str(training_label))

    # pca = fit_pca(model, vae_model)
    #pca = torch.load("experiments/pca")
    for batch_idx, (data, pers) in enumerate(train_loader):
        optimizer.zero_grad()

        label = np.delete(pers, np.s_[0:2], 0)
        label = label.astype(np.float64)
        label = label.transpose()
        label = torch.tensor(label).double()

        # label = label.unsqueeze(1)

        data = torch.clamp(torch.div(data, (torch.min(data, dim=2, keepdim=True)[0]).repeat(1, 1, data.size(2))), min=0, max=1)

        data = Variable(data)

        data = data.transpose(1, 2)

        if use_cuda:
            data = data.cuda()
            label = label.cuda()

        outs = vae_model(data)

        # print("The dimensions of the input data are: {}".format(data.shape))
        # print("The dimensions of the global sample are: {}".format(outs.encoder_out.global_sample.shape))
        # print("The dimensions of the local sample are: {}".format(outs.encoder_out.local_sample.shape))
        # print("The dimensions of the decoder output are: {}".format(outs.decoder_out.shape))
        # print("The dimensions of the predictor output are: {}".format(outs.predictor_out.rsample().shape))

        if args.model_type == 'vae_g_l':
            global_sample = torch.mean(outs.encoder_out.global_sample, dim=1)
            # global_sample = outs.encoder_out.global_sample
            # global_sample = global_sample.unsqueeze_(1)
            # global_sample = global_sample.expand(args.batch_size, 1, 256)  # 32 == batch size
            # print(global_sample.shape)
            # if using z is desired instead of h
            # global_sample = torch.mean(outs.encoder_out.local_sample, dim=1)
        if args.model_type == 'vae_l':
            global_sample = torch.mean(outs.encoder_out.local_sample, dim=1)
        """
        global_sample = pca.transform(global_sample.cpu())
        global_sample = torch.tensor(global_sample)
        global_sample = global_sample.cuda().type(torch.cuda.FloatTensor)
        """
        pred = model(global_sample)
        # print(pred[0][0][:])
        # pred = torch.squeeze(pred, dim=1)
        # print(pred.shape)

        loss, arousal_loss, valence_loss = loss_function(pred, label)
        loss.backward()
        epoch_loss += loss.item()
        epoch_arousal += arousal_loss.item()
        epoch_valence += valence_loss.item()

        optimizer.step()

    return epoch_loss / batch_idx, epoch_arousal/batch_idx, epoch_valence/batch_idx


def test(model, vae_model):
    model.eval()
    test_loss = 0
    test_arousal = 0
    test_valence = 0
    #tp = 0
    sample_amount = 0

    training_label = 0
    if args.type == 'person':
        training_label = 1

    #pca = torch.load("experiments/pca")
    for batch_idx, (data, pers) in enumerate(test_loader):

        label = np.delete(pers, np.s_[0:2], 0)
        label = label.astype(np.float64)
        label = label.transpose()
        label = torch.tensor(label).double()

        data = torch.clamp(torch.div(data, (torch.min(data, dim=2, keepdim=True)[0]).repeat(1, 1, data.size(2))), min=0, max=1)

        data = Variable(data)

        data = data.transpose(1, 2)

        if use_cuda:
            data = data.cuda()
            label = label.cuda()

        outs = vae_model(data)
        if args.model_type == 'vae_g_l':
            global_sample = torch.mean(outs.encoder_out.global_sample, dim=1)
            #global_sample = torch.mean(outs.encoder_out.local_sample, dim=1)
            # global_sample = outs.encoder_out.global_sample

        if args.model_type == 'vae_l':
            global_sample = torch.mean(outs.encoder_out.local_sample, dim=1)

            # print(global_sample.shape)
        """
        global_sample = pca.transform(global_sample.cpu())
        global_sample = torch.tensor(global_sample)
        global_sample = global_sample.cuda().type(torch.cuda.FloatTensor)
        """
        pred = model(global_sample)

        max, indices = torch.max(pred, dim=1)

        #tp += (indices == label).float().sum()
        sample_amount += len(label)

        loss, arousal_loss, valence_loss = loss_function(pred, label)
        test_loss += loss.item()
        test_arousal += arousal_loss.item()
        test_valence += valence_loss.item()

    #acc = tp / sample_amount

    return test_loss / batch_idx, test_arousal/batch_idx, test_valence/batch_idx


def train_epochs(model, vae_model, optimizer):
    last_loss = np.Inf

    train_losses = []
    test_losses = []
    train_arousals = []
    test_arousals = []
    train_valences = []
    test_valences =[]

    for epoch in range(args.num_epochs):
        avg_train_loss, avg_train_arousal, avg_train_valence = train(model, vae_model, optimizer)
        train_losses.append(avg_train_loss)
        train_arousals.append(avg_train_arousal)
        train_valences.append(avg_train_valence)
        if epoch % args.checkpoint_interval == 0:
            print('====> Epoch: {} Average train loss: {:.5f}'.format(epoch, avg_train_loss))
            print('====> Epoch: {} Average train arousal loss: {:.5f}'.format(epoch, avg_train_arousal))
            print('====> Epoch: {} Average train valence loss: {:.5f}'.format(epoch, avg_train_valence))
            print('-------------------------------------------------')
            avg_test_loss, avg_test_arousal, avg_test_valence = test(model, vae_model)
            test_losses.append(avg_test_loss)
            test_arousals.append(avg_test_arousal)
            test_valences.append(avg_test_valence)
            print('====> Epoch: {} Average test loss: {:.5f}'.format(epoch, avg_test_loss))
            print('====> Epoch: {} Average test arousal loss: {:.5f}'.format(epoch, avg_test_arousal))
            print('====> Epoch: {} Average test valence loss: {:.5f}'.format(epoch, avg_test_valence))
            #print('====> Epoch: {} test ACC: {:.4f}'.format(epoch, acc))

            writer.add_scalar('experiments/regression_train_loss', avg_train_loss, epoch)
            writer.add_scalar('experiments/regression_test_loss', avg_test_loss, epoch)

            print('=================================================')

            plt.title("arousal, valence and overall losses")
            plt.plot(train_losses, label="overall training")
            plt.plot(test_losses, label="overall testing")
            plt.plot(train_arousals, label="arousal loss training")
            plt.plot(test_arousals, label="arousal loss testing")
            plt.plot(train_valences, label="valence loss training")
            plt.plot(test_valences, label="valence loss testing")
            plt.legend(loc='upper right')
            plt.savefig("experiments/regression_losses.png")
            plt.clf()

            torch.save(model.state_dict(), 'experiments/' + args.model_name)


if __name__ == '__main__' and args.mode == 'train':

    model = Emotion_Regression()
    if args.model_type == 'vae_g_l':
        vae_model = vae_g_l.VAE(args)
        if args.use_cuda:
            vae_model.load_state_dict(torch.load('experiments/' + args.pretrained_model))
        else:
            vae_model.load_state_dict(
                torch.load('experiments/' + args.pretrained_model, map_location=lambda storage, loc: storage))
    elif args.model_type == 'vae_l':
        vae_model = vae_l.VAE(args)
        if args.use_cuda:
            vae_model.load_state_dict(torch.load('experiments/' + args.pretrained_model))
        else:
            vae_model.load_state_dict(
                torch.load('experiments/' + args.pretrained_model, map_location=lambda storage, loc: storage))

    vae_model.eval()

    for param in vae_model.parameters():
        param.requires_grad = False

    if args.resume == 1:
        model.load_state_dict(torch.load('experiments/' + args.model_name, map_location=lambda storage, loc: storage))
        print("loaded model")

    if use_cuda:
        model.cuda()
        vae_model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_epochs(model, vae_model, optimizer)
