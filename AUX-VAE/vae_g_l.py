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

def output_to_dist(output, dim=-1):
    z_size = output.size(dim)//2    # 500/2=256
    mean, log_var = torch.split(output, z_size, dim=dim)
    return torch.distributions.Normal(mean, torch.exp(0.5*log_var))

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.predictor = Predictor(args)
        #self.classifier = Classifier()

        self.VAEOutput = collections.namedtuple("VAEOutput", ["encoder_out", "decoder_out", "predictor_out"])
        #self.VAEOutput = collections.namedtuple("VAEOutput", ["encoder_out", "decoder_out", "predictor_out", "classifier_out"])
    
    def forward(self, input, annealing=0):
        
        encoder_out = self.encoder(input)
        predictor_out = self.predictor(input, encoder_out.local_sample)
        
        decoder_out = self.decoder(encoder_out.local_sample)
        #classifier_out = self.classifier(encoder_out.local_sample)

        #print(torch.min(decoder_out))
        #if not(torch.min(decoder_out) >= 0.):
        #    print(pers)
        #print(torch.max(decoder_out))
        assert torch.min(decoder_out) >= 0.
        assert torch.max(decoder_out) <= 1.
        
        #return self.VAEOutput(encoder_out, decoder_out, predictor_out, classifier_out)
        return self.VAEOutput(encoder_out, decoder_out, predictor_out)
    
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        self.EncoderOutput = collections.namedtuple("EncoderOutput", ["local_dist", "local_sample", "global_dist", "global_sample"])
        
        self.global_net = nn.Sequential(
            custom_nn.Transpose((1, 2)),
            #201 because the input is on 201 channels after transpose
            #The Transpse is in my opinion because the gpu work only with channels last on tf
            nn.Conv1d(201, int(args.z_size/4), kernel_size = 3, stride = 1),
            nn.Tanh(),
            
            nn.BatchNorm1d(args.z_size//4),
            
            nn.Conv1d(args.z_size//4, args.z_size//2, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size//2),
            
            nn.Conv1d(args.z_size//2, args.z_size, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size),
            
            nn.Conv1d(args.z_size, 2*args.z_size, kernel_size = 1, stride = 1),
            custom_nn.Transpose((1, 2)),
            
        )
        
        self.local_net = nn.Sequential(
            custom_nn.Transpose((1, 2)),
            #201+args.z_size  this because the input to the local_net is a concatenation of the input and the global_net's output
            nn.Conv1d(201+args.z_size, args.local_z_size, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(args.local_z_size),
            
            nn.Conv1d(args.local_z_size, args.local_z_size, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(args.local_z_size),
            
            nn.Conv1d(args.local_z_size, 2*args.local_z_size, kernel_size = 1, stride = 1),
            custom_nn.Transpose((1, 2)),
        )
        self.light_dropout = nn.Dropout(0.3)
        
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input):
        
        # input is a tensor of batch x time x features
        assert len(input.size()) == 3
        
        global_out = self.global_net(input)
        # global average pooling
        global_out = torch.mean(global_out, dim=1)  # dim=1 is the channels
        
        # for testng purposes of speaker transformation
        # ujson.dump(global_out.tolist(), open("experiments/global_out.json", 'w'))
        # global_out = torch.FloatTensor(ujson.load(open('experiments/global_out.json', 'r')))
        # global_out = global_out + torch.rand(global_out.size())
        
        global_out = global_out.unsqueeze(1)
        global_out = global_out.repeat(1,input.size(1),1)
        
        global_dist = output_to_dist(global_out)
        
        global_z_sample = global_dist.rsample()  #rsample means random sample
        
        resized_sample = global_z_sample
        
        if self.training:
            local_out = self.local_net(torch.cat((F.dropout(input, p=0.2, training=True), resized_sample), 2))
        else:
            local_out = self.local_net(torch.cat((F.dropout(input, p=0.2, training=False), resized_sample), 2))
        
        local_dist = output_to_dist(local_out)
        
        local_z_sample = local_dist.rsample()
        return self.EncoderOutput(local_dist=local_dist, local_sample=local_z_sample, global_dist=global_dist, global_sample=global_z_sample)

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        
        self.fc = nn.Sequential(
            custom_nn.Transpose((1,2)),
            nn.Conv1d(args.local_z_size, 201, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(201),
            
            nn.Conv1d(201, 201, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(201),
            
            nn.Conv1d(201, 201, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(201),
            
            nn.Conv1d(201, 201, kernel_size=1, stride=1),
            nn.Sigmoid(),
            custom_nn.Transpose((1,2)),
        )
    
    def forward(self, input):        
        out = self.fc(input)
        
        return out


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        """
        self.fc = nn.Sequential(
            nn.Linear(args.z_size, len(label_dict)),
            nn.LogSoftmax(dim=1)
        )
        """
        label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)

        # This 64*64 came from 64 channels because of the last convolution, and 64 input size after two times reduction with pooling
        #self.fc1 = nn.Linear(64 * 64, 32 * 32)
        #self.fc2 = nn.Linear(32 * 32, 16 * 16)
        #self.fc3 = nn.Linear(16 * 16, 100)
        #self.fc4 = nn.Linear(100, len(label_dict))

        self.fc = nn.Linear(256*64, 7)
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.softmax = nn.LogSoftmax(dim=1)

        # """

    def forward(self, input):
        # if args.debug_mode: print ("input: {}".format(input.size()))
        global_sample = torch.mean(input, dim=1)

        global_sample = global_sample.unsqueeze(1)
        #"""
        pred = F.dropout(F.relu(self.batch_norm1(self.conv1(global_sample))), p=0.3, training=True)
        pred = F.dropout(F.relu(self.batch_norm2(self.conv2(pred))), p=0.3, training=True)
        pred = F.dropout(F.relu(self.batch_norm3(F.max_pool1d(self.conv3(pred), 2))), p=0.3, training=True)
        pred = F.dropout(F.relu(F.max_pool1d(self.conv4(pred), 2)), p=0.3, training=True)
        pred = pred.view(-1, 256*64)
        pred = F.relu((self.fc(pred)))

        """
        pred = self.pool(F.relu(self.conv1(global_sample)))
        pred = self.pool(F.relu(self.conv2(pred)))
        pred = pred.view(-1, 64 * 64)
        pred = F.relu(self.fc1(pred))
        pred = F.relu(self.fc2(pred))
        pred = F.relu(self.fc3(pred))
        pred = F.relu(self.fc4(pred))
        """
        pred = self.softmax(pred)
        # pred = self.fc(input)
        # pred = pred.squeeze(0)#

        return pred

class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()
        
        self.z_recon_fc = nn.Sequential(
            custom_nn.Transpose((1,2)),
            nn.Conv1d(args.local_z_size, args.z_size, kernel_size=1, stride=1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size),
            
            nn.Conv1d(args.z_size, args.z_size*2, kernel_size=1, stride=1),
            custom_nn.Transpose((1,2)),
        )

    def forward(self, input, z):
        
        input_with_z = z
        
        recon_out = self.z_recon_fc(input_with_z)
        
        recon_dist = output_to_dist(recon_out)
        return recon_dist
