import torch.utils.data as data
import csv
import os
import os.path
import torch
import ujson
from audio_utils import read_audio
import numpy as np
import torch.nn.functional as F

def make_manifest(dir):
    audios = []
    for pers in sorted(os.listdir(dir)):
        person_id = pers
        person_dir = os.path.join(dir, person_id)
        person_dir = os.path.expanduser(person_dir)
        for record in sorted(os.listdir(person_dir)):
            if not record.endswith(".wav"):
                continue
            emotion = record.split('-')[2]
            actor = record.split('-')[6]
            record_dir = os.path.join(person_dir, record)
            record_dir = os.path.expanduser(record_dir)
            audios.append([record_dir, record, person_id, emotion])
    return audios


class Ravdess_dataset(data.Dataset):

    def __init__(self, root, downsample=True, transform=None, target_transform=None, dev_mode=False, preprocessed=False,
                 person_filter=None, filter_mode='exclude', max_len=201, split='train'):
        self.person_filter = person_filter
        self.filter_mode = filter_mode
        self.root = os.path.expanduser(root)
        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.dev_mode = dev_mode
        self.num_samples = 0
        self.max_len = max_len
        self.split = split
        if preprocessed:
            self.root_dir = os.path.expanduser('Ravdess_preprocessed/')
            if self.split == 'train':
                self.data_paths = os.listdir(os.path.join(self.root_dir, 'train'))
                self.root_dir = os.path.join(self.root_dir, 'train/')
            elif self.split == 'test':
                self.data_paths = os.listdir(os.path.join(self.root_dir, 'test'))
                self.root_dir = os.path.join(self.root_dir, 'test/')

            if person_filter:
                if self.filter_mode == 'include':
                    self.data_paths = [sample for sample in self.data_paths if
                                       any(pers in sample for pers in self.person_filter)]
                elif self.filter_mode == 'exclude':
                    self.data_paths = [sample for sample in self.data_paths if
                                       not any(sample.startswith(pers ) for pers in self.person_filter)]


            self.num_samples = len(self.data_paths)

        else:
            # making an array of all sound files with make_manifest
            audios = make_manifest(self.root)

            os.mkdir('Ravdess_preprocessed')
            os.mkdir('Ravdess_preprocessed/train')
            os.mkdir('Ravdess_preprocessed/test')

            counter = 0
            for audio in audios:
                counter += 1
                sig = read_audio(audio[0])
                if self.transform is not None:
                    sig = self.transform(sig[0])
                else:
                    sig = sig[0]

                try:
                    data = (sig.tolist(), [audio[3], audio[2]])
                    name = audio[1]
                    name = name.split('.wav')[0]
                    ujson.dump(data, open("Ravdess_preprocessed/train/{}.json".format(name), 'w'))
                    if counter % 100 == 0:
                        print("{} iterations".format(counter))

                    self.train_data_paths = os.listdir(os.path.expanduser('Ravdess_preprocessed/train/'))
                except:
                    continue

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # print self.data_paths[0]

        audio, label = ujson.load(open(self.root_dir + self.data_paths[index], 'r'))

        audio = torch.from_numpy(np.array(audio)).float()

        if audio.size(1) < self.max_len:
            audio = F.pad(audio, (0, self.max_len - audio.size(1)), "constant", -80.)
        elif audio.size(1) > self.max_len:
            audio = audio[:, :self.max_len]

        return audio, label

    def __len__(self):
        return self.num_samples