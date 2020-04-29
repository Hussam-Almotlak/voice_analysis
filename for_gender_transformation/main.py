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
import OMG_dataset
import Ravdess_dataset
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
import supervised_classifier

from trainer import VAETrainer

from tensorboardX import SummaryWriter

import PIL.Image
from torchvision.transforms import ToTensor
torch.manual_seed(0)
import noisereduce as nr

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ['TORCH_MODEL_ZOO'] = '$WORK/.torch/models/'

parser = argparse.ArgumentParser(description='VAE Speech')

parser.add_argument('--cuda', type=int, default=1, metavar='N',
                    help='use cuda if possible (default: True)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='N',
                    help='learning rate (default: 0.001)')
parser.add_argument('--num-epochs', type=int, default=60, metavar='N',
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
parser.add_argument('--beta-mi', type=float, default=0.01, metavar='N',
                    help='(beta weight for MI, default: 1.')
parser.add_argument('--beta-kl', type=float, default=0.01, metavar='N',
                    help='(beta weight for KL, default: 1.')
parser.add_argument('--z-size', type=int, default=512, metavar='N',
                    help='(latent feature depth, default: 256')
parser.add_argument('--predictive', type=int, default=1, metavar='N',
                    help='(predictive coding, if false reconstruct, default: 1')

def load_dataset(dataset='VCTK', train_subset=1.0, person_filter=None):

    if dataset=='VCTK':
        person_filter = ['p249', 'p239', 'p276', 'p283', 'p243', 'p254', 'p258', 'p271']
        train_dataset = vctk_custom_dataset.VCTK('../datasets/VCTK-Corpus/', preprocessed=True, person_filter = person_filter, filter_mode = 'exclude')
        test_dataset = vctk_custom_dataset.VCTK('../datasets/VCTK-Corpus/', preprocessed=True, person_filter = person_filter, filter_mode = 'include')
    elif dataset=='LibriSpeech':
        train_dataset = librispeech_custom_dataset.LibriSpeech('../datasets/LibriSpeech/', preprocessed=True, split='train', person_filter = person_filter, filter_mode = 'include')
        test_dataset = librispeech_custom_dataset.LibriSpeech('../datasets/LibriSpeech/', preprocessed=True, split='test', person_filter = person_filter, filter_mode = 'include')
    elif dataset=='OMGEmotion':
        train_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='train', person_filter = person_filter, filter_mode = 'include')
        test_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='test', person_filter = person_filter, filter_mode = 'include')
    elif dataset == 'Ravdess':
        train_dataset = Ravdess_dataset.Ravdess_dataset('../datasets/Ravdess_preprocessed/', preprocessed=True, split='train',
                                               person_filter=person_filter, filter_mode='include')
        test_dataset = Ravdess_dataset.Ravdess_dataset('../datasets/Ravdess_preprocessed/', preprocessed=True, split='test',
                                              person_filter=person_filter, filter_mode='include')
    indices = list(range(len(train_dataset)))
    split = int(np.floor(len(train_dataset) * train_subset))
    
    train_sampler = sampler.RandomSampler(sampler.SubsetRandomSampler(indices[:split]))
    test_sampler = sampler.RandomSampler(test_dataset)
    
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler = train_sampler, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler = test_sampler, drop_last=False, **kwargs)

    return train_loader, test_loader, train_dataset, test_dataset


def train(args):
    
    train_loader, test_loader, train_dataset, test_dataset = load_dataset(dataset=args.dataset, train_subset=1.0)
    
    if args.model_type == 'vae_g_l':
        model = vae_g_l.VAE(args)
        if args.resume:
            model.load_state_dict(torch.load('experiments/'+args.model_name))
        args.loss_function = 'mi_loss'

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        writer = SummaryWriter(comment=args.model_name)

        if args.use_cuda:
            model = model.cuda()
        trainer = VAETrainer(args, model, optimizer, train_loader, test_loader, test_dataset, writer)
        
    elif args.model_type == 'vae_l':
        model = vae_l.VAE(args)
        if args.resume:
            model.load_state_dict(torch.load('experiments/'+args.model_name))
        args.loss_function = 'kl_loss'
        
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        writer = SummaryWriter(comment=args.model_name)
        
        if args.use_cuda:
            model = model.cuda()
        
        trainer = VAETrainer(args, model, optimizer, train_loader, test_loader, test_dataset, writer)


    else:
        raise Exception("No valid Model type provided")
    
    print ("Training Model Type {}".format(args.model_type))
    print ("Model Name: {}".format(args.model_name))
    
    trainer.train_epochs()
    
    print (args.model_name)

def test(args):
    
    if not os.path.exists('experiments'):
                    os.makedirs('experiments')
    
    transfs = transforms.Compose([
        prepro.DB_Spec(sr = 22050, n_fft=2048,hop_t=0.010,win_t=0.025)
        ])
    

    if args.model_type == 'vae_g_l':
        model = vae_g_l.VAE(args)        
        model.load_state_dict(torch.load('experiments/'+args.model_name, map_location=lambda storage, loc: storage))
    elif args.model_type == 'vae_l':
        model = vae_l.VAE(args)        
        model.load_state_dict(torch.load('experiments/'+args.model_name, map_location=lambda storage, loc: storage))

    model.eval()
    
    if args.dataset == "VCTK":
        # male example        
        # data, sr = prepro.read_audio('/work/invx030/datasets/VCTK-Corpus/wav48/p245/p245_002.wav')        
        # Female example
        data, sr = prepro.read_audio('/work/invx030/datasets/VCTK-Corpus/wav48/p233/p233_003.wav')
    elif args.dataset == "LibriSpeech":
        # male: /home/hussam/Bachelorarbeit/LibriSpeech/train-clean-100/26/495/26-495-0001.flac
        # female: /home/hussam/Bachelorarbeit/LibriSpeech/train-clean-100/2910/131096/2910-131096-0000.flac
        man, sr = librosa.load('/home/hussam/Bachelorarbeit/OMG_Emotion_audio/data/76eecf968/audio/utterance_2.mp4/audio.wav')
        man = man[0:4*sr]
        librosa.output.write_wav('experiments/data.wav', man, sr=22050, norm=True)

        woman, sr_w = librosa.load('/home/hussam/Bachelorarbeit/OMG_Emotion_audio/data/76eecf968/audio/utterance_9.mp4/audio.wav')
        woman = woman[0:4*sr]
        librosa.output.write_wav('experiments/woman.wav', woman, sr=22050, norm=True)
    else:
        raise Exception('No valid dataset provided (use --dataset)')
    
    
    data = man
    data = transfs(data)
    print(data.shape)

    woman = transfs(woman)
    woman = Variable(woman)
    woman = woman.unsqueeze(0)
    woman = woman.transpose(1,2)
    woman = woman / torch.min(woman)

    data = Variable(data)
    data = data.unsqueeze(0)
    
    data = data.transpose(1,2)
    original = data
    min = torch.min(data)
    data = data / min
    
    outs = model(data, woman, data)
    reconstruction = outs.decoder_out
    reconstruction = reconstruction * min #* 2 - 100
    reconstruction = reconstruction.transpose(1,2)
    reconstruction = reconstruction.squeeze(0)
    reconstruction = (reconstruction.data.cpu()).numpy()
    
    original = original.transpose(1,2)
    original = original.squeeze(0)
    original = (original.data.cpu()).numpy()
    
    librosa.display.specshow(original, sr=22050, x_axis='time', y_axis='linear', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original DB spectrogram')
    pylab.savefig('experiments/original_spec.png')
    plt.clf()
    original = librosa.core.db_to_power(original, ref=1)

    librosa.display.specshow(reconstruction,sr=22050, x_axis='time', y_axis='linear', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstruction DB spectrogram')
    pylab.savefig('experiments/reconstruction_spec.png')
    plt.clf()
    reconstruction = librosa.core.db_to_power(reconstruction, ref=1)


    inverse = librosa.feature.inverse.mel_to_stft(original, sr=22050, n_fft=2048)
    inverse = librosa.griffinlim(inverse)
    librosa.output.write_wav('experiments/original.wav', inverse, sr=22050, norm=True)

    inverse = librosa.feature.inverse.mel_to_stft(reconstruction, sr=22050, n_fft=2048)
    inverse = librosa.griffinlim(inverse)
    librosa.output.write_wav('experiments/reconstruction.wav', inverse, sr=22050, norm=True)

    #inverse = nr.reduce_noise(audio_clip=inverse, noise_clip=inverse, verbose=True)
    #librosa.output.write_wav('experiments/reconstruction_NR.wav', inverse, sr=22050, norm=True)

    

def analyse_latent(args):
    
    print("Creating latents for test set")
    
    if not os.path.exists('experiments'):
                    os.makedirs('experiments')
    
    if args.model_type == 'vae_g_l':
        print (args.model_name)
        model = vae_g_l.VAE(args)        
        model.load_state_dict(torch.load('experiments/'+args.model_name, map_location=lambda storage, loc: storage))
    elif args.model_type == 'vae_l':
        print (args.model_name)
        model = vae_l.VAE(args)        
        model.load_state_dict(torch.load('experiments/'+args.model_name, map_location=lambda storage, loc: storage))
    elif args.model_type == 'vae_g_l_exp':
        print (args.model_name)
        model = vae_g_l_exp.VAE(args) 
        model.load_state_dict(torch.load('experiments/'+args.model_name, map_location=lambda storage, loc: storage))
    
    model.eval()
    
    labels = []
    persons =  []
    globel_latent = []
    medium_latent = []
    
    if args.dataset == 'LibriSpeech':
        train_loader, test_loader, train_dataset, test_dataset = load_dataset(dataset=args.dataset, person_filter=['4640', '4788', '4853', '4830', '19', '26', '1221', '60', '83', '2300'])
    elif args.dataset == 'VCTK':
       train_loader, test_loader, train_dataset, test_dataset = load_dataset(dataset=args.dataset, person_filter=['p249', 'p239', 'p276', 'p283', 'p243', 'p254', 'p258', 'p271'])
       train_loader = test_loader
    elif args.dataset == 'OMGEmotion':
        train_loader, test_loader, train_dataset, test_dataset = load_dataset(dataset=args.dataset, person_filter=[ '4bf764f4f', '3ecbd1402', '5ba7890ab', '2debca276', '2af27e034', 'd1dae148e', 'e3b57cfd8'])
        # '7d0084def'3x, '9c3a9bce7'3x, '22e9f018d'3x, '78a906313'4x, 'd51aee260'3x, 'f1f277e5d'4x, 'e3b57cfd8'4x ... 664ea0ccc
    elif args.dataset == 'Ravdess':
        print(args.dataset)
        train_loader, test_loader, train_dataset, test_dataset = load_dataset(dataset=args.dataset)

    for batch_idx, (data, label) in enumerate(test_loader):
        label = np.transpose(np.array(label))
        
        if args.dataset == 'VCTK':
            labels.extend(label[:,0])
            persons.extend(label[:,3])
        elif args.dataset == 'LibriSpeech':
            labels.extend(label[:,0])
            persons.extend(label[:,1])
        elif args.dataset == 'OMGEmotion':
            labels.extend(label[:, 0])
            persons.extend(label[:, 1])
            persons = [person.split('_')[0] for person in persons]
        elif args.dataset == 'Ravdess':
            labels.extend(label[:, 0])
            persons.extend(label[:, 1])
            #persons = [person.split('_')[1] for person in persons]
        
        data = data.transpose(1,2)
        data = data / (torch.min(data))
        
        with torch.no_grad():
            data = Variable(data)
        
        original = data
        
        if args.predictive:
            data = F.pad(data, (0,0,1,0), "constant", 1)
            original = F.pad(original, (0,0,0,1), "constant", 1)
        
        outs  = model(data)
        
        if args.model_type == 'vae_g_l':
            globel_latent.extend(outs.encoder_out.global_sample.tolist())
            medium_latent.extend(outs.encoder_out.medium_sample.tolist())
            # globel_latent.extend(outs.encoder_out.local_sample.tolist())
        elif args.model_type == 'vae_l':
            globel_latent.extend(outs.encoder_out.local_sample.tolist())
        elif args.model_type == 'vae_g_l_exp':
            globel_latent.extend(outs.encoder_out.global_sample.unsqueeze(1).tolist())
    
    labels = np.array(labels)
    globel_latent = np.array(globel_latent)
    medium_latent = np.array(medium_latent)
    persons = np.array(persons)
    print(globel_latent.shape)

    ujson.dump(globel_latent, open("experiments/tsne_globel_latent.json", 'w'))
    ujson.dump(medium_latent,open("experiments/tsne_medium_latent.json", 'w'))
    ujson.dump(labels,open("experiments/tsne_labels.json", 'w'))
    ujson.dump(persons,open("experiments/tsne_persons.json", 'w'))


if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.model_name is None:
        args.model_name = args.model_type
    
    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False
    
    print ("Using CUDA: {}".format(args.use_cuda))
    
    args.hidden_size = args.z_size
    args.local_z_size = args.z_size
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'analyse-latent':
        analyse_latent(args)
    else:
        print("No --mode provided, options: train, test, analyse-latent")
