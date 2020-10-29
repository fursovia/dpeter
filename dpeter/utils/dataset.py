"""Dataset reader and process"""

import os

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

        for partition in self.partitions:
            arange = range(len(self.dataset[partition]['gt']))

            for i in reversed(arange):
                text = pp.text_standardize(self.dataset[partition]['gt'][i])

                self.dataset[partition]['gt'][i] = str(text).encode()

            results = []
            print(f"Partition: {partition}")
            for path in tqdm(self.dataset[partition]['dt']):
                results.append(pp.preprocess(str(path), input_size=input_size))

            self.dataset[partition]['dt'] = results

    def _init_dataset(self):
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

        return dataset

    def _dpeter(self):
        dataset = self._init_dataset()

        for partition in ['train', 'valid']:
            data_path = os.path.join(self.data_dir, f'{partition}.json')
            data = load_jsonlines(data_path)

            for path_to_image_and_words in data:
                image_path = path_to_image_and_words["image_path"]
                words_path = path_to_image_and_words["text_path"]

                text = load_text(words_path)
                dataset[partition]['gt'].append(text)

                dataset[partition]['dt'].append(image_path)

        return dataset
