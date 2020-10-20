from typing import List, Optional, Tuple
import logging

import numpy as np
import cv2
import jsonlines
from allennlp.common import Registrable
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import CharacterTokenizer, Token

logger = logging.getLogger(__name__)

START_TOKEN = "<START>"
END_TOKEN = "<END>"


class ImageAugmentator(Registrable):

    def __call__(self, image: np.ndarray) -> np.ndarray:
        pass


@DatasetReader.register("peter_reader")
class PeterReader(DatasetReader):
    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 128),
        augmentator: Optional[ImageAugmentator] = None,
        add_start_end_tokens: bool = True,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)

        self._width, self._height = image_size
        self._augmentator = augmentator
        self._add_start_end_tokens = add_start_end_tokens
        self._tokenizer = CharacterTokenizer()
        self._start_token = Token(START_TOKEN)
        self._end_token = Token(END_TOKEN)

    def _surround_with_start_end_tokens(self, tokens: List[Token]) -> List[Token]:
        return [self._start_token] + tokens + [self._end_token]

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        img = np.stack([img, img, img], axis=-1)
        w, h, _ = img.shape

        new_w = self._height
        new_h = int(h * (new_w / w))
        img = cv2.resize(img, (new_h, new_w))
        w, h, _ = img.shape

        img = img.astype('float32')

        new_h = self._width
        if h < new_h:
            add_zeros = np.full((w, new_h - h, 3), 255)
            img = np.concatenate((img, add_zeros), axis=1)

        if h > new_h:
            img = cv2.resize(img, (new_h, new_w))

        return img.astype('uint8')

    def text_to_instance(
        self,
        image: np.ndarray,
        text: Optional[str] = None,
    ) -> Instance:

        if self._augmentator is not None:
            image = self._augmentator(image)

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
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = self._preprocess_image(image)

                text_path = items.get("text_path")
                if text_path is not None:
                    with open(text_path) as f:
                        text = f.read()
                else:
                    text = None

                instance = self.text_to_instance(image=image, text=text)
                yield instance
