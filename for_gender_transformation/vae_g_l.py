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
    z_size = output.size(dim)//2    # 512/2=256
    mean, log_var = torch.split(output, z_size, dim=dim) # The whole input will be divided into 2 tensors of size 256 in the dim=-1
    return torch.distributions.Normal(mean, torch.exp(0.5*log_var))

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.predictor = Predictor(args)
        self.medium = Medium(args)

        self.VAEOutput = collections.namedtuple("VAEOutput", ["encoder_out", "decoder_out", "predictor_out", "medium_out"])

    def forward(self, input_global, input_medium, input_reconstruction, annealing=0):
        
        encoder_out = self.encoder(input_global, input_medium, input_reconstruction)
        predictor_out = self.predictor(input_reconstruction, encoder_out.local_sample)
        medium_out = self.medium(input_reconstruction, encoder_out.local_sample, encoder_out.global_sample)
        decoder_out = self.decoder(encoder_out.local_sample)

        assert torch.min(decoder_out) >= 0.
        assert torch.max(decoder_out) <= 1.
        
        return self.VAEOutput(encoder_out, decoder_out, predictor_out, medium_out)
    
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        self.EncoderOutput = collections.namedtuple("EncoderOutput", ["local_dist", "local_sample", "global_dist", "global_sample", "medium_dist", "medium_sample"])

        self.global_net = nn.Sequential(
            custom_nn.Transpose((1, 2)),

            nn.Conv1d(140, int(args.z_size/4), kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size//4),
            
            nn.Conv1d(args.z_size//4, args.z_size//2, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size//2),
            
            nn.Conv1d(args.z_size//2, args.z_size, kernel_size = 3, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size),
            
            nn.Conv1d(args.z_size, 2*args.z_size, kernel_size = 1, stride = 1),
            custom_nn.Transpose((1, 2))
            
        )

        self.medium_net = nn.Sequential(
            custom_nn.Transpose((1, 2)),

            nn.Conv1d(140 + args.z_size, args.z_size // 4, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size // 4),

            nn.Conv1d(args.z_size // 4, args.z_size // 2, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size // 2),

            nn.Conv1d(args.z_size // 2, args.z_size, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size),

            nn.Conv1d(args.z_size, 2 * args.z_size, kernel_size=1, stride=1),
            custom_nn.Transpose((1, 2))
        )
        
        self.local_net = nn.Sequential(
            custom_nn.Transpose((1, 2)),
            #140+args.z_size  this because the input to the local_net is a concatenation of the input and the global_net's output
            nn.Conv1d(140+args.z_size, args.local_z_size, kernel_size = 1, stride = 1),
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

    def forward(self, input_global, input_medium, input_reconstruction):
        
        # input is a tensor of batch * time * features
        assert len(input_reconstruction.size()) == 3

        global_out = self.global_net(input_global)
        # global average pooling
        global_out = torch.mean(global_out, dim=1)  # dim=1 is the channels
        global_out = global_out.unsqueeze(1)
        global_out = global_out.repeat(1, input_global.size(1), 1)
        global_dist = output_to_dist(global_out)
        global_sample = global_dist.rsample()  # rsample means random sample


        medium_input = torch.cat((F.dropout(input_medium, p=0.2, training=True), global_sample), 2)
        medium_out = self.medium_net(medium_input)

        net_1, net_2, net_3, net_4 = torch.split(medium_out, 44, dim=1)
        net_1 = torch.mean(net_1, dim=1)
        net_2 = torch.mean(net_2, dim=1)
        net_3 = torch.mean(net_3, dim=1)
        net_4 = torch.mean(net_4, dim=1)

        net_1 = net_1.unsqueeze(1)
        net_2 = net_2.unsqueeze(1)
        net_3 = net_3.unsqueeze(1)
        net_4 = net_4.unsqueeze(1)

        net_1 = net_1.repeat(1, 44, 1)
        net_2 = net_2.repeat(1, 44, 1)
        net_3 = net_3.repeat(1, 44, 1)
        net_4 = net_4.repeat(1, input_medium.size(1)-44*3, 1)

        medium_out = torch.cat((net_1, net_2, net_3, net_4), dim=1)

        medium_dist = output_to_dist(medium_out)
        medium_sample = medium_dist.rsample()  #rsample means random sample
        
        resized_sample = medium_sample
        
        if self.training:
            local_out = self.local_net(torch.cat((F.dropout(input_reconstruction, p=0.2, training=True), resized_sample), 2))
        else:
            local_out = self.local_net(torch.cat((F.dropout(input_reconstruction, p=0.2, training=False), resized_sample), 2))
        
        local_dist = output_to_dist(local_out)
        
        local_z_sample = local_dist.rsample()
        return self.EncoderOutput(local_dist=local_dist, local_sample=local_z_sample, global_dist=global_dist, global_sample=global_sample, medium_dist=medium_dist, medium_sample=medium_sample)

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        
        self.fc = nn.Sequential(
            custom_nn.Transpose((1,2)),
            nn.Conv1d(args.local_z_size, 140, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(140),
            
            nn.Conv1d(140, 140, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(140),
            
            nn.Conv1d(140, 140, kernel_size = 1, stride = 1),
            nn.Tanh(),
            nn.BatchNorm1d(140),
            
            nn.Conv1d(140, 140, kernel_size=1, stride=1),
            nn.Sigmoid(),
            custom_nn.Transpose((1,2)),
        )
    
    def forward(self, input):        
        out = self.fc(input)
        
        return out


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


class Medium(nn.Module):
    def __init__(self, args):
        super(Medium, self).__init__()

        self.z_recon_fc = nn.Sequential(
            custom_nn.Transpose((1, 2)),
            nn.Conv1d(args.local_z_size*2, args.z_size*2, kernel_size=1, stride=1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size*2),

            nn.Conv1d(args.local_z_size*2, args.z_size*2, kernel_size=1, stride=1),
            nn.Tanh(),
            nn.BatchNorm1d(args.z_size*2),

            nn.Conv1d(args.z_size*2, args.z_size*2, kernel_size=1, stride=1),
            custom_nn.Transpose((1, 2)),
        )

    def forward(self, input, z, h):
        h_with_z = torch.cat((h,z), dim=2)

        recon_out = self.z_recon_fc(h_with_z)

        recon_dist = output_to_dist(recon_out)
        return recon_dist
