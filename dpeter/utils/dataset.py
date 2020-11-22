"""Dataset reader and process"""

import os

from tqdm import tqdm
import cv2

from dpeter.utils.preprocessors.preprocessor import Preprocessor
from dpeter.utils.data import load_jsonlines, load_text, load_image


class Dataset:
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, data_dir: str, preprocessor: Preprocessor, bertam=False):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.dataset = None
        self.bertam = bertam
        self.partitions = ['train', 'valid']

    def read(self):
        """Read images and sentences from dataset"""
        if not self.bertam:
            dataset = self._dpeter()
        else:
            dataset = self._bertam()

        if not self.dataset:
            self.dataset = dict()

            for partition in self.partitions:
                self.dataset[partition] = {'dt': [], 'gt': []}

        for partition in self.partitions:
            self.dataset[partition]['dt'] += dataset[partition]['dt']
            self.dataset[partition]['gt'] += dataset[partition]['gt']

        self.preprocess_partitions()

    def preprocess_partitions(self):
        """Preprocess images and sentences from partitions"""

        for partition in self.partitions:
            arange = range(len(self.dataset[partition]['gt']))

            for i in reversed(arange):
                text = self.dataset[partition]['gt'][i]
                self.dataset[partition]['gt'][i] = str(text).encode()

            results = []
            gt = []
            print(f"Partition: {partition}")
            for i, path in enumerate(tqdm(self.dataset[partition]['dt'])):
                img = load_image(str(path))
                img = self.preprocessor.preprocess(img)
                if img is None:
                    continue
                results.append(img)
                gt.append(self.dataset[partition]['gt'][i])
            self.dataset[partition]['gt'] = gt
            self.dataset[partition]['dt'] = results

    def _dpeter(self):
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

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

    def _bertam(self):
        dataset = dict()
        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

        for partition in ['Train', 'Validation']:
            data_path = os.path.join(self.data_dir, 'Partitions', f'{partition}Lines.lst')
            data = load_lst_lines(data_path)

            for filename in data:
                image_path = os.path.join(self.data_dir, 'Images/Lines', f'{filename}.png')
                words_path = os.path.join(self.data_dir, 'Transcriptions', f'{filename}.txt')

                text = load_text(words_path).strip()
                if partition == 'Train':
                    dataset['train']['gt'].append(text)
                    dataset['train']['dt'].append(image_path)
                else:
                    dataset['valid']['gt'].append(text)
                    dataset['valid']['dt'].append(image_path)

        return dataset

def load_lst_lines(file):
    with open(file) as f:
        res = [line.strip() for line in f.readlines()]
    return res
