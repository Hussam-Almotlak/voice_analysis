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
            wav_dir = os.path.join(pers_dir, utterance)
            wav_dir = os.path.expanduser(wav_dir)
            for audiofile in sorted(os.listdir(wav_dir)):
                if not audiofile.endswith(".wav"):
                    continue
                audios.append([os.path.expanduser(os.path.join(wav_dir, audiofile)), video, utterance])
    return audios

def find_path(audios, video, utterance):
    for index, content in enumerate(audios):
        if ((video==content[1])and(utterance==content[2])):
            return content[0]
    return ""

class OMGEmotion(data.Dataset):


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

            if emotion_filter:
                if self.filter_mode == 'include':
                    self.data_paths = [sample for sample in self.data_paths if
                                       any(emotion in sample for emotion in self.emotion_filter)]
                elif self.filter_mode == 'exclude':
                    self.data_paths = [sample for sample in self.data_paths if
                                       not any(sample.startswith(emotion) for emotion in self.emotion_filter)]


            self.num_samples = len(self.data_paths)

        else:
            # making an array of all sound files with make_manifest
            paths = make_manifest(self.root)

            os.mkdir('OMG_preprocessed')
            os.mkdir('OMG_preprocessed/train')
            os.mkdir('OMG_preprocessed/test')
            os.mkdir('OMG_preprocessed/validate')
            for i in range(7):
                os.mkdir('OMG_preprocessed/train/' + str(i))
                os.mkdir('OMG_preprocessed/test/' + str(i))
                os.mkdir('OMG_preprocessed/validate/' + str(i))

            emotion_dict = {'0': 'anger', '1': 'disgust', '2': 'fear', '3': 'happiness', '4': 'neutral', '5': 'sadness', '6': 'surprise'}
            
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

                    signal, sr = librosa.load(path)
                    length = len(signal) #The length of the signal
                    rate = 4*sr # The sampling rate is one second
                    signals_nr = length // rate
                    speaker_id = video
                    speaker_id = speaker_id.split('_')[0]
                    utterance, keywork, after = utterance.partition('.mp4')
                    if counter % 100 == 0:
                        print("{} iterations".format(counter))

                    for nr in range(signals_nr):
                        sub_sig = signal[nr*rate:nr*rate+rate]
                        if self.transform is not None:
                            sub_sig = self.transform(sub_sig)
                        else:
                            sub_sig = sub_sig
                        try:
                            data = (sub_sig.tolist(), [emotion, speaker_id, arousal, valence])
                            name = speaker_id + "_" + utterance + "_" + emotion_dict[emotion] + "_" + str(nr)
                            ujson.dump(data, open("OMG_preprocessed/train/{}/{}.json".format(emotion,name), 'w'))

                            self.train_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/train/'))
                        except:
                            print("There was an exeption!!!!!!!!!!!")
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
                    signal, sr = librosa.load(path)
                    length = len(signal)  # The length of the signal
                    rate = 4 * sr  # The ampling rate is one second
                    signals_nr = length // rate
                    speaker_id = video
                    speaker_id = speaker_id.split('_')[0]
                    utterance, keywork, after = utterance.partition('.mp4')
                    if counter % 100 == 0:
                        print("{} iterations".format(counter))

                    for nr in range(signals_nr):
                        sub_sig = signal[nr*rate:nr*rate+rate]
                        if self.transform is not None:
                            sub_sig = self.transform(sub_sig)
                        else:
                            sub_sig = sub_sig

                        try:

                            data = (sub_sig.tolist(), [emotion, speaker_id, arousal, valence])
                            name = speaker_id + "_" + utterance + "_" + emotion_dict[emotion] + "_" + str(nr)
                            ujson.dump(data, open("OMG_preprocessed/validate/{}/{}.json".format(emotion,name), 'w'))

                            self.train_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/validate/'))
                        except:
                            print("There was an exeption!!!!!!!!!!!")
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
                    signal, sr = librosa.load(path)
                    length = len(signal)  # The length of the signal
                    rate = 4 * sr  # The ampling rate is one second
                    signals_nr = length // rate
                    speaker_id = video
                    speaker_id = speaker_id.split('_')[0]
                    utterance, keywork, after = utterance.partition('.mp4')
                    if counter % 100 == 0:
                        print("{} iterations".format(counter))

                    for nr in range(signals_nr):
                        sub_sig = signal[nr*rate:nr*rate+rate]
                        if self.transform is not None:
                            sub_sig = self.transform(sub_sig)
                        else:
                            sub_sig = sub_sig

                        try:

                            data = (sub_sig.tolist(), [emotion, speaker_id, arousal, valence])
                            name = speaker_id + "_" + utterance + "_" + emotion_dict[emotion] + "_" + str(nr)
                            ujson.dump(data, open("OMG_preprocessed/test/{}/{}.json".format(emotion,name), 'w'))

                            self.train_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/test/'))
                        except:
                            print("There was an exeption!!!!!!!!!!!")
                            continue


            self.train_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/train/'))
            self.test_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/test/'))
            self.validate_data_paths = os.listdir(os.path.expanduser('OMG_preprocessed/validate/'))
            self.num_samples = len(self.train_data_paths) + len(self.test_data_paths) + len(self.validate_data_paths)
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
