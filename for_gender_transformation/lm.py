import os
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import models
import argparse
import time
import math

import vctk_custom_dataset
import librispeech_custom_dataset
import OMG_dataset
import ujson
import audio_utils as prepro
from audio_utils import to_audio
import librosa
import librosa.display
from torch.utils.data import Dataset, DataLoader, sampler, distributed



parser = argparse.ArgumentParser(description='lm.py')

parser.add_argument('-save_model', default='mlstm',
                    help="""Model filename to save""")
parser.add_argument('-load_model', default='',
                    help="""Model filename to load""")
parser.add_argument('-rnn_type', default='mlstm',
                    help='mlstm, lstm or gru')
parser.add_argument('-layers', type=int, default=1,
                    help='Number of layers in the rnn')
parser.add_argument('-rnn_size', type=int, default=2048,
                    help='Size of hidden states')
parser.add_argument('-embed_size', type=int, default=128,
                    help='Size of embeddings')
parser.add_argument('-batch_size', type=int, default=32,
                    help='Maximum batch size')
parser.add_argument('-learning_rate', type=float, default=0.1,
                    help="""Starting learning rate.""")
parser.add_argument('-dropout', type=float, default=0.1,
                    help='Dropout probability.')
parser.add_argument('-param_init', type=float, default=0.05,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-clip', type=float, default=5,
                    help="""Clip gradients at this value.""")
parser.add_argument('-seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('-dataset', type=str, default='OMGEmotion', metavar='S',
                    help='(dataset used for training; LibriSpeech, VCTK; default: VCTK')
parser.add_argument('-first_epoch', type=int, default=0, 
                    help='The number of the first epoch (for saving the model)')

# GPU
parser.add_argument('-cuda', action='store_true',
                    help="Use CUDA")

         

args = parser.parse_args()
learning_rate = args.learning_rate

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data        

input_dim = 140
hidden_size =args.rnn_size
input_size = args.embed_size
data_size = 256
TIMESTEPS = 140

if len(args.load_model)>0:
    checkpoint = torch.load(args.load_model)
    embed = checkpoint['embed']
    rnn = checkpoint['rnn']
else:
    embed = nn.Embedding(256, input_size)
    if args.rnn_type == 'gru':
        rnn = models.StackedRNN(nn.GRUCell, args.layers, input_size, hidden_size, data_size, args.dropout)
    elif args.rnn_type == 'mlstm':
        rnn = models.StackedLSTM(models.mLSTM, args.layers, input_size, hidden_size, data_size, args.dropout)
    else:#default to lstm
        rnn = models.StackedLSTM(nn.LSTMCell, args.layers, input_size, hidden_size, data_size, args.dropout)



loss_fn = nn.CrossEntropyLoss() 

nParams = sum([p.nelement() for p in rnn.parameters()])
print('* number of parameters: %d' % nParams)

learning_rate =args.learning_rate

embed_optimizer = optim.SGD(embed.parameters(), lr=learning_rate)
rnn_optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
   
def update_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
    return

def clip_gradient_coeff(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

def calc_grad_norm(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    return math.sqrt(totalnorm)
    
def calc_grad_norms(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    norms = []
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        norms += [modulenorm]
    return norms
    
def clip_gradient(model, clip):
    """Clip the gradient."""
    totalnorm = 0
    for p in model.parameters():
        p.grad.data = p.grad.data.clamp(-clip,clip)

        
def make_cuda(state):
    if isinstance(state, tuple):
        return (state[0].cuda(), state[1].cuda())
    else:
        return state.cuda()

def copy_state(state):
    if isinstance(state, tuple):
        return (Variable(state[0].data), Variable(state[1].data))
    else:
        return Variable(state.data)


def load_dataset(dataset='VCTK', train_subset=1.0, person_filter=None):
    if dataset == 'VCTK':
        person_filter = ['p249', 'p239', 'p276', 'p283', 'p243', 'p254', 'p258', 'p271']
        train_dataset = vctk_custom_dataset.VCTK('../datasets/VCTK-Corpus/', preprocessed=True,
                                                 person_filter=person_filter, filter_mode='exclude')
        test_dataset = vctk_custom_dataset.VCTK('../datasets/VCTK-Corpus/', preprocessed=True,
                                                person_filter=person_filter, filter_mode='include')
    elif dataset == 'LibriSpeech':
        train_dataset = librispeech_custom_dataset.LibriSpeech('../datasets/LibriSpeech/', preprocessed=True,
                                                               split='train', person_filter=person_filter,
                                                               filter_mode='include')
        test_dataset = librispeech_custom_dataset.LibriSpeech('../datasets/LibriSpeech/', preprocessed=True,
                                                              split='test', person_filter=person_filter,
                                                              filter_mode='include')
    elif dataset == 'OMGEmotion':
        train_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, person_filter=person_filter, split='train')
        print("There are: {} Training samples.".format(len(train_dataset)))

        test_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, person_filter=person_filter, split='test')
        print("There are: {} Testing samples.".format(len(test_dataset)))

    indices = list(range(len(train_dataset)))
    split = int(np.floor(len(train_dataset) * train_subset))

    train_sampler = sampler.RandomSampler(sampler.SubsetRandomSampler(indices[:split]))
    test_sampler = sampler.RandomSampler(test_dataset)

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=test_sampler, drop_last=False, **kwargs)

    return train_loader, test_loader, train_dataset, test_dataset


def evaluate():
    hidden_init = rnn.state0(input_dim)
    if args.cuda:
        embed.cuda()
        rnn.cuda()
        hidden_init = make_cuda(hidden_init)


    nv_batch = 0
    train_loader, test_loader, train_dataset, test_dataset = load_dataset(dataset=args.dataset, train_subset=1.0)
    for batch_idx, (data, pers) in enumerate(test_loader):
        nv_batch = len(test_loader)
        data = torch.clamp(torch.div(data, (torch.min(data, dim=2, keepdim=True)[0]).repeat(1, 1, data.size(2))), min=0,
                           max=1)

        data = data.long()
        loss_avg = 0
        for s in range(args.batch_size):
            batch = Variable(data[s])
            batch = batch.t()
            start = time.time()
            hidden = hidden_init
            if args.cuda:
                batch = batch.cuda()

            loss = 0
            # TIMESTEPS is the number of charactors or units to process (201 time steps)
            for t in range(TIMESTEPS):
                emb = embed(batch[t])
                hidden, output = rnn(emb, hidden)
                loss += loss_fn(output, batch[t + 1])

            loss_avg = loss_avg + loss.data / TIMESTEPS
        hidden_init = copy_state(hidden)
        print('v %s / %s loss %.4f time %.4f' % (
              batch_idx, nv_batch, loss_avg / args.batch_size, time.time() - start))

    return loss_avg/nv_batch

def train_epoch(epoch):
    hidden_init = rnn.state0(input_dim)
    if (args.cuda):
        embed.cuda()
        rnn.cuda()
        hidden_init = make_cuda(hidden_init)


    n_batch = 0
    train_loader, test_loader, train_dataset, test_dataset = load_dataset(dataset=args.dataset, train_subset=1.0)
    for batch_idx, (data, pers) in enumerate(train_loader):
        n_batch = len(train_loader)
        data = torch.clamp(torch.div(data, (torch.min(data, dim=2, keepdim=True)[0]).repeat(1, 1, data.size(2))), min=0, max=1)
        data = data.long()
        loss_avg = 0
        for s in range(args.batch_size):
            embed_optimizer.zero_grad()
            rnn_optimizer.zero_grad()
            batch = Variable(data[s])
            batch = batch.t()
            start = time.time()
            hidden = hidden_init
            # print("The size of the hidden layer is: {}".format(len(hidden)))
            if args.cuda:
                batch = batch.cuda()
            loss = 0
            # TIMESTEPS is the number of charactors or units to process (201 time steps)
            for t in range(TIMESTEPS):
                emb = embed(batch[t])
                hidden, output = rnn(emb, hidden)
                loss += loss_fn(output, batch[t + 1])

            loss_avg = loss_avg + loss / TIMESTEPS

        loss_avg = loss_avg / args.batch_size
        loss_avg.backward() 
        hidden_init = copy_state(hidden)
        gn = calc_grad_norm(rnn)
        clip_gradient(rnn, args.clip)
        clip_gradient(embed, args.clip)
        embed_optimizer.step()
        rnn_optimizer.step()
        print('e%s %s / %s loss %.4f time %.4f grad_norm %.4f' % (
              epoch, batch_idx, n_batch, loss_avg, time.time() - start, gn))

for e in range(args.first_epoch, args.first_epoch+30):
    try:
        train_epoch(e)
    except KeyboardInterrupt:
        print('Exiting from training early')
    loss_avg = evaluate()
    checkpoint = {
            'rnn': rnn,
            'embed': embed,
            'opt': args,
            'epoch': e
        }
    save_file = ('%s_e%s_%.2f.pt' % (args.save_model, e, loss_avg))
    print('Saving to '+ save_file)
    torch.save(checkpoint, args.save_model+"/"+save_file)
    learning_rate *= 0.7
    update_lr(rnn_optimizer, learning_rate)
    update_lr(embed_optimizer, learning_rate)
