import argparse
import torch
import torchaudio
from torchaudio import transforms, save
import numpy as np
import ujson
from OMG_dataset import OMGEmotion
import librosa
from audio_utils import griffinlim

import audio_utils as prepro

parser = argparse.ArgumentParser(description='Prepare OMGEmotion Dataset')

parser.add_argument('--OMG-path', default="/home/hussam/Bachelorarbeit/OMG_Emotion_audio", type=str, metavar='S', help='(path to OMG-Corpus)')

args = parser.parse_args()
# //2 means downsampling to 11025
transfs = transforms.Compose([prepro.DB_Spec(sr = 22050, n_fft=2048,hop_t=0.010,win_t=0.025, max_len=22050*4)])

dataset = OMGEmotion(root = args.OMG_path, transform = transfs)
