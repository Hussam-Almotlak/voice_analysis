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


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_manifest(dir):
    audios = []
    dir = os.path.join(dir, "data")
    dir = os.path.expanduser(dir)
    for video in sorted(os.listdir(dir)):
        pers_dir = os.path.join(dir, video)
        pers_dir = os.path.expanduser(pers_dir)
        pers_dir = os.path.join(pers_dir, "audio")
        pers_dir = os.path.expanduser(pers_dir)
        for utterance in sorted(os.listdir(pers_dir)):
            flac_dir = os.path.join(pers_dir, utterance)
            flac_dir = os.path.expanduser(flac_dir)
            for audiofile in sorted(os.listdir(flac_dir)):
                if not audiofile.endswith(".wav"):
                    continue
                audios.append([os.path.expanduser(os.path.join(flac_dir, audiofile)), video, utterance])
    return audios

def find_path(audios, video, utterance):
    #print("Searching for: Video= {} and Utterance= {}".format(video, utterance))
    for index, content in enumerate(audios):
        if ((video==content[1])and(utterance==content[2])):
            return content[0]
    return ""

class OMGEmotion(data.Dataset):


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
            self.root_dir = os.path.expanduser('OMG_preprocessed/')
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


            self.num_samples = len(self.data_paths)

        else:
            # making an array of all sound files with make_manifest
            paths = make_manifest(self.root)

            os.mkdir('OMG_preprocessed')
            os.mkdir('OMG_preprocessed/train')
            os.mkdir('OMG_preprocessed/test')
            os.mkdir('OMG_preprocessed/validate')

            #important: Video name is the speaker id
            #instead of the the gender we use here the emotion

            with open(os.path.join(self.root, "omg_TrainVideos.csv")) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                counter = 0
                next(csvreader)
                for row in csvreader:
                    counter += 1

                    video = row[3]
                    utterance = row[4]
                    emotion = row[7]
                    arousal = row[5]
                    valence = row[6]

                    path = find_path(paths, video, utterance)
                    if(path == ''):
                        continue

                    sig = read_audio(path)
                    if self.transform is not None:
                        sig = self.transform(sig[0])
                    else:
                        sig = sig[0]

                    try:
                        speaker_id = video
                        speaker_id = speaker_id.split('_')[0]
                        data = (sig.tolist(), [emotion,speaker_id, arousal, valence])
                        utterance, keywork, after= utterance.partition('.mp4')
                        name = speaker_id+"_"+utterance
                        ujson.dump(data, open("OMG_preprocessed/train/{}.json".format(name), 'w'))
                        if  counter% 100 == 0:
                            print("{} iterations".format(counter))

                        self.train_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/train/'))
                    except:
                        continue

            with open(os.path.join(self.root, "omg_ValidationVideos.csv")) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                counter = 0
                next(csvreader)
                for row in csvreader:
                    counter += 1

                    video = row[3]
                    utterance = row[4]
                    emotion = row[7]
                    arousal = row[5]
                    valence = row[6]

                    path = find_path(paths, video, utterance)
                    if (path == ''):
                        continue
                    sig = read_audio(path)
                    if self.transform is not None:
                        sig = self.transform(sig[0])
                    else:
                        sig = sig[0]

                    try:
                        speaker_id = video
                        speaker_id = speaker_id.split('_')[0]
                        data = (sig.tolist(), [emotion, speaker_id, arousal, valence])
                        utterance, keywork, after = utterance.partition('.mp4')
                        name = speaker_id + "_" + utterance
                        ujson.dump(data, open("OMG_preprocessed/validate/{}.json".format(name), 'w'))
                        if  counter% 100 == 0:
                            print("{} iterations".format(counter))
                        self.validate_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/validate/'))

                    except:
                        continue

            with open(os.path.join(self.root, "omg_TestVideos_WithLabels.csv")) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                counter = 0
                next(csvreader)
                for row in csvreader:
                    counter += 1

                    video = row[3]
                    utterance = row[4]
                    emotion = row[7]
                    arousal = row[5]
                    valence = row[6]

                    path = find_path(paths, video, utterance)
                    if (path == ''):
                        continue
                    sig = read_audio(path)
                    if self.transform is not None:
                        sig = self.transform(sig[0])
                    else:
                        sig = sig[0]

                    try:
                        speaker_id = video
                        speaker_id = speaker_id.split('_')[0]
                        data = (sig.tolist(), [emotion,speaker_id, arousal, valence])
                        utterance, keywork, after = utterance.partition('.mp4')
                        name = speaker_id + "_" + utterance
                        ujson.dump(data, open("OMG_preprocessed/test/{}.json".format(name), 'w'))
                        if  counter% 100 == 0:
                            print("{} iterations".format(counter))
                        self.test_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/test/'))

                    except:
                        continue


            self.train_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/train/'))
            self.test_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/test/'))
            self.validate_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/validate/'))
            self.num_samples = len(self.train_data_paths)
            print("{} samples processed".format(self.num_samples))

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
