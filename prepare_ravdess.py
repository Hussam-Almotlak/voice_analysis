import argparse
from torchaudio import transforms

import audio_utils as prepro
from Ravdess_dataset import Ravdess_dataset

parser = argparse.ArgumentParser(description="prepare the Ravdess emotional dataset")

parser.add_argument('--Ravdess-path', type=str, metavar='S', required=True, help="define the path of the dataset to process")

args = parser.parse_args()

transfs = transforms.Compose([prepro.DB_Spec(sr = 11025, n_fft=400,hop_t=0.010,win_t=0.025, max_len=40000)])

dataset = Ravdess_dataset(root= args.Ravdess_path, transform=transfs)