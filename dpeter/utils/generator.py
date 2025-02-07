"""
Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time.
"""

from itertools import groupby

import h5py
import numpy as np

import dpeter.utils.preprocessing as pp
from dpeter.utils.augmentators.augmentator import Augmentator, NullAugmentator


class DataGenerator:
    """Generator class with data streaming"""

    def __init__(self, source, batch_size, charset, max_text_length, augmentator: Augmentator = None, predict=False, train_flips=False):
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.batch_size = batch_size
        self.partitions = ['test'] if predict else ['train', 'valid']
        self.augmentator = augmentator or NullAugmentator()

        self.size = dict()
        self.steps = dict()
        self.index = dict()
        self.dataset = dict()

        self.train_flips = train_flips

        with h5py.File(source, "r") as f:
            for pt in self.partitions:
                self.dataset[pt] = dict()
                self.dataset[pt]['dt'] = np.array(f[pt]['dt'])
                self.dataset[pt]['gt'] = np.array([x.decode() for x in f[pt]['gt']])

                self.size[pt] = len(self.dataset[pt]['gt'])
                self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))

            randomize = np.arange(len(self.dataset['train']['gt']))
            np.random.seed(42)
            np.random.shuffle(randomize)

            self.dataset['train']['dt'] = self.dataset['train']['dt'][randomize]
            self.dataset['train']['gt'] = self.dataset['train']['gt'][randomize]

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        self.index['train'] = 0

        while True:
            if self.index['train'] >= self.size['train']:
                self.index['train'] = 0

            index = self.index['train']
            until = index + self.batch_size
            self.index['train'] = until

            x_train = self.dataset['train']['dt'][index:until]
            x_train = self.augmentator.augment(x_train)
            x_train = pp.normalization(x_train)

            y_train = [self.tokenizer.encode(y) for y in self.dataset['train']['gt'][index:until]]
            y_train = [np.pad(y, (0, self.tokenizer.maxlen - len(y))) for y in y_train]
            y_train = np.asarray(y_train, dtype=np.int16)

            if self.train_flips:
                if np.random.uniform() > 0.5:
                    y_train = np.array([1], dtype=np.int32)
                else:
                    x_train = x_train[:, :, ::-1]
                    y_train = np.array([0], dtype=np.int32)

            yield (x_train, y_train)

    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        self.index['valid'] = 0

        while True:
            if self.index['valid'] >= self.size['valid']:
                self.index['valid'] = 0

            index = self.index['valid']
            until = index + self.batch_size
            self.index['valid'] = until

            x_valid = self.dataset['valid']['dt'][index:until]
            x_valid = pp.normalization(x_valid)

            y_valid = [self.tokenizer.encode(y) for y in self.dataset['valid']['gt'][index:until]]
            y_valid = [np.pad(y, (0, self.tokenizer.maxlen - len(y))) for y in y_valid]
            y_valid = np.asarray(y_valid, dtype=np.int16)

            if self.train_flips:
                if np.random.uniform() > 0.5:
                    y_valid = np.array([1], dtype=np.int32)
                else:
                    x_valid = x_valid[:, :, ::-1]
                    y_valid = np.array([0], dtype=np.int32)

            yield (x_valid, y_valid)

    def next_test_batch(self):
        """Return model predict parameters"""

        self.index['test'] = 0

        while True:
            if self.index['test'] >= self.size['test']:
                self.index['test'] = 0
                break

            index = self.index['test']
            until = index + self.batch_size
            self.index['test'] = until

            x_test = self.dataset['test']['dt'][index:until]
            x_test = pp.normalization(x_test)

            yield x_test


class Tokenizer:
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        text = " ".join(text.split())

        groups = ["".join(group) for _, group in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []

        for item in text:
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
