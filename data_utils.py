import collections
import os
import numpy as np
from torch import Tensor
import librosa
from torch.utils.data import Dataset

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1

    # padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]

    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


ASVFile = collections.namedtuple('ASVFile',
                                 ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, transform=None, is_train=True,
                 protocols_dir=None):
        '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.transform = transform
        self.protocols_fname = 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = protocols_dir
        self.sysid_dict = {
            '-': 0,  # bonafide speech
            'A01': 1,  # Wavenet vocoder
            'A02': 2,  # Conventional vocoder WORLD
            'A03': 3,  # Conventional vocoder MERLIN
            'A04': 4,  # Unit selection system MaryTTS
            'A05': 5,  # Voice conversion using neural networks
            'A06': 6,  # transform function-based voice conversion
            'A07': 7,
            'A08': 8,
            'A09': 9,
            'A10': 10,
            'A11': 11,
            'A12': 12,
            'A13': 13,
            'A14': 14,
            'A15': 15,
            'A16': 16,
            'A17': 17,
            'A18': 18,
            'A19': 19,
            # For PA:
            'AA': 27,
            'AB': 28,
            'AC': 29,
            'BA': 30,
            'BB': 31,
            'BC': 32,
            'CA': 33,
            'CB': 34,
            'CC': 35
        }
        self.protocols_fname = os.path.join(self.protocols_dir,
                                            'ASVspoof2019.{}.cm.{}.txt'.format('LA', self.protocols_fname))
        self.files_meta = self.parse_protocols_file(self.protocols_fname)
        data = list(map(self.read_file, self.files_meta))
        self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
        if self.transform:
            self.data_x = [self.transform(x) for x in self.data_x]

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        data_x, fs = librosa.load(meta.path, sr=16000)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
                       file_name=tokens[1],
                       path=os.path.join(os.path.join(self.base_dir, 'flac'), tokens[1] + '.flac'),
                       sys_id=self.sysid_dict[tokens[3]],
                       key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
