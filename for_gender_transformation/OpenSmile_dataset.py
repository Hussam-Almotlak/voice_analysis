import torch.utils.data as data
import csv
import string
import unicodedata
import re
import os
import os.path
import sys
import shutil
import errno
import torch
import torchaudio
import ujson
from audio_utils import griffinlim
from audio_utils import read_audio
import librosa
import numpy as np
from torchaudio import datasets, transforms, save
import torch.nn.functional as F
import time
import arff

class OpenSmile(data.Dataset):
    def __init__(self, root, downsample=True, transform=None, target_transform=None, dev_mode=False, preprocessed=False,
                 person_filter=None, filter_mode='include', max_len=44*4, emotion_filter=None, split='train'):
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
        self.emotion_filter = emotion_filter

        if preprocessed:
            self.root_dir = os.path.expanduser('OpenSmile_preprocessed/')
            if self.split == 'train':
                self.data_paths = os.listdir(os.path.join(self.root_dir, 'train'))
                self.root_dir = os.path.join(self.root_dir, 'train/')
            elif self.split == 'test':
                self.data_paths = os.listdir(os.path.join(self.root_dir, 'test'))
                self.root_dir = os.path.join(self.root_dir, 'test/')
            elif self.split == 'validate':
                self.data_paths = os.listdir(os.path.join(self.root_dir, 'validate'))
                self.root_dir = os.path.join(self.root_dir, 'validate/')

            if person_filter:
                if self.filter_mode == 'include':
                    self.data_paths = [sample for sample in self.data_paths if
                                       any(pers in sample for pers in self.person_filter)]
                elif self.filter_mode == 'exclude':
                    self.data_paths = [sample for sample in self.data_paths if
                                       not any(sample.startswith(pers ) for pers in self.person_filter)]

            if emotion_filter:
                if self.filter_mode == 'include':
                    self.data_paths = [sample for sample in self.data_paths if
                                       any(emotion in sample for emotion in self.emotion_filter)]
                elif self.filter_mode == 'exclude':
                    self.data_paths = [sample for sample in self.data_paths if
                                       not any(sample.startswith(emotion) for emotion in self.emotion_filter)]


            self.num_samples = len(self.data_paths)

        else:
            os.mkdir('OpenSmile_preprocessed')
            os.mkdir('OpenSmile_preprocessed/train')
            os.mkdir('OpenSmile_preprocessed/test')
            os.mkdir('OpenSmile_preprocessed/validate')

            with open(os.path.join(self.root, 'feat_comp16_train_data.arff'), 'rt') as arfffile:
                file = arff.load(arfffile)
                attributes = np.array(file['attributes'])
                data = np.array(file['data'])
                nAttributes = len(attributes)
                nRecords = len(data)

                for record in range(nRecords):
                    features = data[record, 1:nAttributes-2]
                    features = features.astype(np.float64)
                    arousal = data[record, nAttributes-2]
                    valence = data[record, nAttributes-1]
                    speaker = data[record, 0].split('/')[0]
                    utterance = data[record, 0].split('/')[1]
                    if record % 200 == 0:
                        print("{} samples".format(record))

                    try:
                        features = features.reshape(78, 15)
                        features = torch.from_numpy(features).float()
                        tosave = (features.tolist(), ['0', speaker, str(arousal), str(valence)])
                        name = speaker + "_" + utterance + "_" + str(record)
                        ujson.dump(tosave, open("OpenSmile_preprocessed/train/{}.json".format(name), 'w'))

                        self.train_data_paths = os.listdir(os.path.expanduser('OpenSmile_preprocessed/train/'))
                    except:
                        print("There was an exeption!!!!!!!!!!!")
                        continue

            with open(os.path.join(self.root, 'feat_comp16_test_data.arff'), 'rt') as arfffile:
                file = arff.load(arfffile)
                attributes = np.array(file['attributes'])
                data = np.array(file['data'])
                nAttributes = len(attributes)
                nRecords = len(data)

                for record in range(nRecords):
                    features = data[record, 1:nAttributes-2]
                    features = features.astype(np.float64)
                    arousal = data[record, nAttributes-2]
                    valence = data[record, nAttributes-1]
                    speaker = data[record, 0].split('/')[0]
                    utterance = data[record, 0].split('/')[1]
                    if record % 200 == 0:
                        print("{} samples".format(record))

                    try:
                        features = features.reshape(78, 15)
                        features = torch.from_numpy(features).float()
                        tosave = (features.tolist(), ['0', speaker, str(arousal), str(valence)])
                        name = speaker + "_" + utterance + "_" + str(record)
                        ujson.dump(tosave, open("OpenSmile_preprocessed/test/{}.json".format(name), 'w'))

                        self.test_data_paths = os.listdir(os.path.expanduser('OpenSmile_preprocessed/test/'))
                    except:
                        print("There was an exeption!!!!!!!!!!!")
                        continue

            with open(os.path.join(self.root, 'feat_comp16_dev_data.arff'), 'rt') as arfffile:
                file = arff.load(arfffile)
                attributes = np.array(file['attributes'])
                data = np.array(file['data'])
                nAttributes = len(attributes)
                nRecords = len(data)

                for record in range(nRecords):
                    features = data[record, 1:nAttributes-2]
                    features = features.astype(np.float64)
                    arousal = data[record, nAttributes-2]
                    valence = data[record, nAttributes-1]
                    speaker = data[record, 0].split('/')[0]
                    utterance = data[record, 0].split('/')[1]
                    if record % 200 == 0:
                        print("{} samples".format(record))

                    try:
                        features = features.reshape(78, 15)
                        features = torch.from_numpy(features).float()
                        tosave = (features.tolist(), ['0', speaker, str(arousal), str(valence)])
                        name = speaker + "_" + utterance + "_" + str(record)
                        ujson.dump(tosave, open("OpenSmile_preprocessed/validate/{}.json".format(name), 'w'))

                        self.validate_data_paths = os.listdir(os.path.expanduser('OpenSmile_preprocessed/validate/'))
                    except:
                        print("There was an exeption!!!!!!!!!!!")
                        continue

            self.train_data_paths = os.listdir(os.path.expanduser('OpenSmile_preprocessed/train/'))
            self.test_data_paths = os.listdir(os.path.expanduser('OpenSmile_preprocessed/test/'))
            self.validate_data_paths = os.listdir(os.path.expanduser('OpenSmile_preprocessed/validate/'))
            self.num_samples = len(self.train_data_paths) + len(self.test_data_paths) + len(self.validate_data_paths)
            print("{} samples processed".format(self.num_samples))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        audio, label = ujson.load(open(self.root_dir + self.data_paths[index], 'r'))

        audio = torch.from_numpy(np.array(audio)).float()
        return audio, label

    def __len__(self):
        return self.num_samples