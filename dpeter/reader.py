from typing import List, Optional, Tuple
import logging
import random

import numpy as np
import cv2
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer, Token

from dpeter.constants import END_TOKEN, START_TOKEN, HEIGHT, WIDTH
from dpeter.modules.augmentator import ImageAugmentator, NullAugmentator
from dpeter.modules.binarizator import ImageBinarizator, NullBinarizator
from dpeter.utils.data import load_jsonlines, load_image, load_text


logger = logging.getLogger(__name__)


@DatasetReader.register("peter_reader")
class PeterReader(DatasetReader):
    def __init__(
        self,
        binarizator: Optional[ImageBinarizator] = None,
        augmentator: Optional[ImageAugmentator] = None,
        shuffle: bool = False,
        add_start_end_tokens: bool = True,
        lazy: bool = False,
        manual_multi_process_sharding: bool = False,
    ) -> None:
        super().__init__(lazy=lazy, manual_multi_process_sharding=manual_multi_process_sharding)

        self._width, self._height = WIDTH, HEIGHT
        self._binarizator = binarizator or NullBinarizator()
        self._augmentator = augmentator or NullAugmentator()
        self._shuffle = shuffle
        self._add_start_end_tokens = add_start_end_tokens
        self._tokenizer = CharacterTokenizer()
        self._start_token = Token(START_TOKEN)
        self._end_token = Token(END_TOKEN)

    def _surround_with_start_end_tokens(self, tokens: List[Token]) -> List[Token]:
        return [self._start_token] + tokens + [self._end_token]

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        w, h, _ = img.shape

        if w > h * 2:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            w, h, _ = img.shape

        new_w = 128
        new_h = int(h * (new_w / w))
        img = cv2.resize(img, (new_h, new_w))
        w, h, _ = img.shape

        img = img.astype('float32')

        if w < 128:
            add_zeros = np.full((128 - w, h, 3), 255)
            img = np.concatenate((img, add_zeros))
            w, h, _ = img.shape

        if h < 1024:
            add_zeros = np.full((w, 1024 - h, 3), 255)
            img = np.concatenate((img, add_zeros), axis=1)
            w, h, _ = img.shape

        if h > 1024 or w > 128:
            dim = (1024, 128)
            img = cv2.resize(img, dim)

        return img.astype('uint8')

    def text_to_instance(
        self,
        image: np.ndarray,
        text: Optional[str] = None,
    ) -> Instance:

        image = self._resize_image(image)
        image = self._binarizator(image)
        image = self._augmentator(image)
        image = cv2.subtract(255, image)
        image = image / 255.0

        fields = {
            "image": ArrayField(array=image)
        }

        if text is not None:
            text = self._tokenizer.tokenize(text)
            fields["length"] = LabelField(len(text), skip_indexing=True)

            if self._add_start_end_tokens:
                text = self._surround_with_start_end_tokens(text)

            fields["text"] = TextField(text, {"tokens": SingleIdTokenIndexer()})

        return Instance(fields)

    def _read(self, file_path: str):

        data = load_jsonlines(file_path)
        if self._shuffle:
            random.shuffle(data)

        for items in data:
            image_path = items["image_path"]
            image = load_image(image_path)

            text_path = items.get("text_path")
            if text_path is not None:
                text = load_text(text_path)
            else:
                text = None

            instance = self.text_to_instance(image=image, text=text)
            yield instance
