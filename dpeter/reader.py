from typing import List, Optional

import numpy as np
import cv2
import jsonlines
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer, Token

from dpeter.constants import END_TOKEN, START_TOKEN, HEIGHT, WIDTH, NUM_CHANNELS, WHITE_CONSTANT
from dpeter.modules.augmentator import ImageAugmentator, NullAugmentator
from dpeter.modules.binarizator import ImageBinarizator, NullBinarizator
from dpeter.utils.data import load_image, load_text


@DatasetReader.register("peter_reader")
class PeterReader(DatasetReader):
    def __init__(
        self,
        binarizator: Optional[ImageBinarizator] = None,
        augmentator: Optional[ImageAugmentator] = None,
        add_start_end_tokens: bool = True,
        lazy: bool = False,
        manual_multi_process_sharding: bool = False,
    ) -> None:
        super().__init__(lazy=lazy, manual_multi_process_sharding=manual_multi_process_sharding)

        self._binarizator = binarizator or NullBinarizator()
        self._augmentator = augmentator or NullAugmentator()
        self._add_start_end_tokens = add_start_end_tokens
        self._tokenizer = CharacterTokenizer()
        self._start_token = Token(START_TOKEN)
        self._end_token = Token(END_TOKEN)

    def _surround_with_start_end_tokens(self, tokens: List[Token]) -> List[Token]:
        return [self._start_token] + tokens + [self._end_token]

    @staticmethod
    def resize_image(img: np.ndarray) -> np.ndarray:
        w, h, _ = img.shape

        if w > h * 2:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            w, h, _ = img.shape

        new_w = HEIGHT
        new_h = int(h * (new_w / w))
        img = cv2.resize(img, (new_h, new_w))
        w, h, _ = img.shape

        img = img.astype('float32')

        if w < HEIGHT:
            add_zeros = np.full((HEIGHT - w, h, NUM_CHANNELS), WHITE_CONSTANT)
            img = np.concatenate((img, add_zeros))
            w, h, _ = img.shape

        if h < WIDTH:
            add_zeros = np.full((w, WIDTH - h, NUM_CHANNELS), WHITE_CONSTANT)
            img = np.concatenate((img, add_zeros), axis=1)
            w, h, _ = img.shape

        if h > WIDTH or w > HEIGHT:
            dim = (WIDTH, HEIGHT)
            img = cv2.resize(img, dim)

        return img.astype('uint8')

    @staticmethod
    def to_float(image: np.ndarray) -> np.ndarray:
        image = cv2.subtract(WHITE_CONSTANT, image)
        image = image / WHITE_CONSTANT
        return image

    def text_to_instance(
        self,
        image: np.ndarray,
        text: Optional[str] = None,
    ) -> Instance:

        image = self.resize_image(image)
        image = self._binarizator(image)
        image = self._augmentator(image)
        image = self.to_float(image)

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

        with jsonlines.open(cached_path(file_path), "r") as reader:
            for items in reader:
                image_path = items["image_path"]
                image = load_image(image_path)

                text_path = items.get("text_path")
                if text_path is not None:
                    text = load_text(text_path)
                else:
                    text = None

                instance = self.text_to_instance(image=image, text=text)
                yield instance
