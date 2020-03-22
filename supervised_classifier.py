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
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ['TORCH_MODEL_ZOO'] = '$WORK/.torch/models/'

parser = argparse.ArgumentParser(description='VAE Speech')

parser.add_argument('--cuda', type=int, default=1, metavar='N',
                    help='use cuda if possible (default: True)')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='N',
                    help='learning rate (default: 0.001)')
parser.add_argument('--num-epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 60)')
parser.add_argument('--model-type', type=str, default='Supervised_Classifier', metavar='S',
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
parser.add_argument('--frame-dropout', type=float, default=0., metavar='N',
                    help='(audio frame dropout for decoder, default: 0. (no dropout))')
parser.add_argument('--dataset', type=str, default='OMGEmotion', metavar='S',
                    help='(dataset used for training; LibriSpeech, VCTK; default: VCTK')

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(256*25*25, 100)
        self.fc2 = nn.Linear(100, 5)
        self.dropout = nn.Dropout(p=0.3)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = input.unsqueeze(1)

        pred = F.dropout(F.relu(self.batch_norm1(self.conv1(input))), p=0.5)
        pred = F.max_pool2d(pred, (2,2))

        pred = F.dropout(F.relu(self.batch_norm2(self.conv2(pred))), p=0.5)
        pred = F.max_pool2d(pred, (2, 2))

        pred = F.dropout(F.relu(self.batch_norm3(self.conv3(pred))), p=0.3)
        pred = F.max_pool2d(pred, (2, 2))

        pred = F.dropout(F.relu(self.conv4(pred)), p=0.3)
        pred = pred.view(-1, 256 * 25*25)

        pred = F.relu(self.fc(pred))
        pred = self.fc2(pred)
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

        if self.loss_function == 'nllloss':
            self.Losses = collections.namedtuple("Losses", ["loss"])
        self.Losses.__new__.__defaults__ = (0,) * len(self.Losses._fields)

    def nllloss(self, pred, label):
        loss = nn.CrossEntropyLoss()
        classifier_loss = loss(pred, label)
        Losses = self.Losses(loss=classifier_loss)
        return Losses

    def train_epochs(self):
        step = 0
        if self.args.resume:
            all_train_losses = ujson.load(open("experiments/{}.json".format('train_losses_'+self.args.model_name), 'r'))
            all_test_losses = ujson.load(open("experiments/{}.json".format('test_losses_'+self.args.model_name), 'r'))
        for epoch in range(self.args.num_epochs):
            avg_epoch_losses, step, acc = self.train(step)

            if epoch % self.args.checkpoint_interval == 0:
                # print step
                print('====> Epoch: {} Average train losses'.format(epoch))
                print('-------------------------')
                print (avg_epoch_losses)
                print("The training accuracy is: {:.4f}".format(acc))
                print('-------------------------')

                avg_test_epoch_losses, acc, all_emotions_accuracy = self.test()

                print('====> Epoch: {} Average test losses'.format(epoch))
                print('-------------------------')
                print (avg_test_epoch_losses)
                print("The testing accuracy is: {:.4f}".format(acc))
                for i in range(5):
                    print("The accuracy on th emotion {} is: {:.4f}".format(i, all_emotions_accuracy[i]))
                print('-------------------------')

                if not os.path.exists('experiments'):
                    os.makedirs('experiments')
                
                torch.save(self.model.state_dict(), 'experiments/'+self.args.model_name)
                
                if epoch == 0 and not self.args.resume:                    
                    all_train_losses = []
                    all_test_losses = []
                
                all_train_losses.append(avg_epoch_losses)
                all_test_losses.append(avg_test_epoch_losses)
                ujson.dump(all_train_losses,open("experiments/{}.json".format('train_losses_'+self.args.model_name), 'w'))
                ujson.dump(all_test_losses,open("experiments/{}.json".format('test_losses_'+self.args.model_name), 'w'))

    def train(self, step):
        self.model.train()
        
        epoch_losses = {}
        tp = 0
        sample_amount = 0


        losses = self.Losses()
        
        for name, value in losses._asdict().items():
            epoch_losses[name] = value

        batch_amount = 0
        label_dict = {'0': 0, '1': 1, '3': 2, '4': 3, '5': 4}
        for batch_idx, (data, pers) in enumerate(self.train_loader):
            batch_amount += 1
            self.optimizer.zero_grad()

            #must be reviewed
            label = [label_dict[p] for p in pers[0]]
            label = torch.tensor(label).cuda()
            
            data = torch.clamp(torch.div(data,(torch.min(data, dim=2, keepdim=True)[0]).repeat(1,1,data.size(2))), min=0, max=1)
            
            data = Variable(data)
            
            if self.args.use_cuda:
                data = data.cuda()
            #"""
            data = data.transpose(1,2) #this is important
            #"""

            output = self.model(data)
            if self.loss_function == 'nllloss':
                losses = self.nllloss(output, label)

            loss = losses.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            max, indices = torch.max(output, dim=1)
            tp += (indices == label).float().sum()
            sample_amount += len(label)

            for name, value in losses._asdict().items():
                epoch_losses[name] += value.item()
            
            self.optimizer.step()
            step += 1
        
        avg_epoch_losses = {k : round(v/batch_amount,5) for k, v in epoch_losses.items()}

        acc = tp/sample_amount

        return avg_epoch_losses, step, acc
    
    
    def test(self):
        self.model.eval()
        
        epoch_losses = {}
        losses = self.Losses()
        
        for name, value in losses._asdict().items():
            epoch_losses[name] = value
        
        batch_amount = 0
        tp = 0
        sample_amount = 0

        label_dict = {'0': 0, '1': 1, '3': 2, '4': 3, '5': 4}
        correct = list(0. for i in range(5))
        all = list(0. for i in range(5))
        for batch_idx, (data, pers) in enumerate(self.test_loader):
            
            batch_amount += 1

            # must be reviewed
            label = [label_dict[p] for p in pers[0]]
            label = torch.tensor(label).cuda()

            data = torch.clamp(torch.div(data,(torch.min(data, dim=2, keepdim=True)[0]).repeat(1,1,data.size(2))), min=0, max=1)
            
            with torch.no_grad():
                data = Variable(data)
            if self.args.use_cuda:
                data = data.cuda()
            #"""
            data = data.transpose(1,2)
            #"""

            output = self.model(data)
            if self.loss_function == 'nllloss':
                losses = self.nllloss(output, label)

            max, indices = torch.max(output, dim=1)
            #calculating the accuracy on each label
            batch = (indices == label).float().squeeze()
            for i in range(len(label)):
                all[label[i]] += 1.
                correct[label[i]] += batch[i].item()
            tp += (indices == label).float().sum()
            sample_amount += len(label)
            
            for name, value in losses._asdict().items():
                epoch_losses[name] += value.item()
        
        avg_epoch_losses = {k : round(v/batch_amount,5) for k, v in epoch_losses.items()}

        acc = tp/sample_amount
        all_emotions_accuracy = list(0. for i in range(5))
        for i in range(5):
            all_emotions_accuracy[i] = correct[i] / all[i]

        return avg_epoch_losses, acc, all_emotions_accuracy




def load_dataset(dataset='VCTK', train_subset=1.0, person_filter=None):
    train_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='train',
                                               person_filter=person_filter, filter_mode='include')
    test_dataset = OMG_dataset.OMGEmotion('../datasets/OMG_preprocessed/', preprocessed=True, split='test',
                                              person_filter=person_filter, filter_mode='include')

    indices = list(range(len(train_dataset)))
    split = int(np.floor(len(train_dataset) * train_subset))

    train_sampler = sampler.RandomSampler(sampler.SubsetRandomSampler(indices[:split]))
    test_sampler = sampler.RandomSampler(test_dataset)

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

    label_dict = {'0': 0, '1': 1, '3': 2, '4': 3, '5': 4}
    train_emotions_distribution = list(0. for i in range(5))
    for batch_idx, (data, pers) in enumerate(train_loader):
        labels = [label for label in pers[0]]
        for i in range(len(labels)):
            train_emotions_distribution[label_dict[labels[i]]] += 1.
    print(train_emotions_distribution)

    test_emotions_distribution = list(0. for i in range(5))
    for batch_idx, (data, pers) in enumerate(test_loader):
        labels = [label for label in pers[0]]
        for i in range(len(labels)):
            test_emotions_distribution[label_dict[labels[i]]] += 1.
    print(test_emotions_distribution)

    model = Classifier(args)
    if args.resume:
        model.load_state_dict("experiments/"+args.model_name)
    args.loss_function = 'nllloss'
    optimizer = optim.SGD(model.parameters(), lr= args.learning_rate, momentum=0.9)
    writer = SummaryWriter(comment=args.model_name)
    if args.use_cuda:
        model = model.cuda()
    trainer = Trainer(args, model, optimizer, train_loader, test_loader, test_dataset, writer)

    print ("Training Model Type {}".format(args.model_type))
    print ("Model Name: {}".format(args.model_name))
    trainer.train_epochs()
    print (args.model_name)


if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.model_name is None:
        args.model_name = args.model_type
    
    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False
    
    print ("Using CUDA: {}".format(args.use_cuda))
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        print("No --mode provided, options: train or test")
