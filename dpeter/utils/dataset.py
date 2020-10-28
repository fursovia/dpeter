"""Dataset reader and process"""

import os
import random

from tqdm import tqdm

import dpeter.utils.preprocessing as pp
from dpeter.utils.data import load_jsonlines, load_text


class Dataset:
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.dataset = None
        self.partitions = ['train', 'valid']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = self._dpeter()

        if not self.dataset:
            self.dataset = dict()

            for y in self.partitions:
                self.dataset[y] = {'dt': [], 'gt': []}

        for y in self.partitions:
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def preprocess_partitions(self, input_size):
        """Preprocess images and sentences from partitions"""

        for y in self.partitions:
            arange = range(len(self.dataset[y]['gt']))

            for i in reversed(arange):
                text = pp.text_standardize(self.dataset[y]['gt'][i])

                self.dataset[y]['gt'][i] = str(text).encode()

            results = []
            print(f"Partition: {y}")
            for path in tqdm(self.dataset[y]['dt']):
                results.append(pp.preprocess(str(path), input_size=input_size))

            self.dataset[y]['dt'] = results

    def _init_dataset(self):
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

        return dataset

    def _shuffle(self, *ls):
        random.seed(42)

        if len(ls) == 1:
            li = list(*ls)
            random.shuffle(li)
            return li

        li = list(zip(*ls))
        random.shuffle(li)
        return zip(*li)

    def _dpeter(self):
        dataset = self._init_dataset()

        for partition in ['train', 'valid']:
            data_path = os.path.join(self.data_dir, f'{partition}.json')
            data = load_jsonlines(data_path)

            for path_to_image_and_words in data:
                image_path = path_to_image_and_words["image_path"]
                words_path = path_to_image_and_words["text_path"]

                # text = load_text(words_path)
                with open(words_path) as f:
                    dataset[partition]['gt'].append(f.read().strip())

                dataset[partition]['dt'].append(image_path)

        return dataset
